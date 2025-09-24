import numpy as np
import os
import json
import pandas as pd
import mne
from mne.preprocessing import ICA, corrmap, create_ecg_epochs , create_eog_epochs
from scipy.stats import kurtosis
import lmdb
import pickle

selected_channels = [
    'FP1','FP2','F3','F4','F7','F8',
    'T7','T8','C3','C4','P7','P8',
    'P3','P4','O1','O2','FZ','CZ','PZ',
]

available_channels = [
    "FZ", "FC1", "FC2", "C3", "CZ", "C4", "CP1", "CP2", "PZ",
    "AF3", "AF4", "F3", "F4", "FC5", "FC6", "CP5", "CP6",
    "P3", "P4", "PO7", "PO8", "OZ"
]

root_dir = '/mnt/disk1/aiotlab/namth/EEGFoundationModel/datasets/Data_UET175'
dest_dir = '/mnt/disk1/aiotlab/namth/EEGFoundationModel/datasets/UET175_eegpt_official'

seed = 3407 # a`c wys`
np.random.seed(seed)

def load_eeg_csv_to_ct(path: str) -> np.ndarray:
    """
    Đọc CSV EEG với 22 cột (mỗi cột = 1 kênh), hàng = thời gian.
    Trả về mảng numpy shape [C, T] = (22, T).
    """
    df = pd.read_csv(path)
    if df.shape[1] != 22:
        raise ValueError(f"Expected 22 columns, got {df.shape[1]} in {path}")

    # Ép về float, lỗi chuyển đổi sẽ tạo NaN
    data = df.astype(float).to_numpy()       # shape (T, 22)
    return data.T    

def preprocess_eeg_mne_prefix(x: np.ndarray) -> np.ndarray:
    """
    Giả định: x shape (22, T), fs = 128 Hz.
    Pipeline: demean -> band-pass(1–50) -> notch 50 -> average reference (CAR) -> resample 256 Hz -> z-score
    Trả về: ndarray (22, round(T * 256 / 128))
    """
    assert isinstance(x, np.ndarray) and x.ndim == 2 and x.shape[0] == 22, "x must be (22, T)"
    fs = 128.0
    info = mne.create_info(ch_names=available_channels, sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(x.astype(float, copy=False), info, verbose=False)

    # Khử lệch DC nhẹ & high-pass >= 1 Hz để ICA ổn định (khuyến nghị của MNE)
    raw._data -= raw._data.mean(axis=1, keepdims=True)
    raw.filter(l_freq=1.0, h_freq=None, method="iir",
               iir_params=dict(order=5, ftype="butter"), verbose=False)
    raw.notch_filter(freqs=[50.0], method="iir", verbose=False)

    # Tham chiếu trung bình (CAR) giúp ICA tách tốt hơn
    raw.set_eeg_reference("average", verbose=False)

    # Ưu tiên Picard-ICA; nếu chưa có, fallback sang FastICA
    ica = mne.preprocessing.ICA(
        n_components=0.999, method="picard",
        random_state=seed, max_iter="auto", verbose=False
    )
    ica.fit(raw, verbose=True)

    raw_clean = ica.apply(raw.copy(), verbose=False)
    return raw_clean.get_data()

def preprocess_eeg_mne_suffix(x: np.ndarray) -> np.ndarray:
    assert isinstance(x, np.ndarray) and x.ndim == 2 and x.shape[0] == 22, "x must be (22, T)"
    fs = 128.0
    info = mne.create_info(ch_names=available_channels, sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(x.astype(float, copy=False), info, verbose=False)

    # 6) resample → 256 Hz
    raw.resample(256, verbose=False)

    # 7) z-score per-channel
    X = raw.get_data()
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
    return X

file = {
    'Male': [],
    'Female': []
}

subjects = sorted(os.listdir(root_dir))

for subject in subjects:
    path = os.path.join(root_dir, subject)
    with open(f"{path}/Info.json", "r", encoding="utf-8") as f:
        info = json.load(f)
    gender = info['Gender']
    file[gender].append(subject)

print(f'Number of Male: {len(file["Male"])}')
print(f'Number of Female: {len(file["Female"])}')

def stratified_split(fold_idx):
    splits = {'train': [], 'test': [], 'val': []}
    for lbl, lst in file.items():
        n = len(lst)
        n_test = n // 5
        test_start = fold_idx * n_test
        test_end = test_start + n_test if fold_idx < 4 else n  # last fold takes the remainder
        splits['test'] += lst[test_start:test_end]
        train = lst[:test_start] + lst[test_end:]
        num_val = int(len(train) / 4.5)
        splits['train'] += train[num_val:]
        splits['val'] += train[:num_val]
    print(f'fold {fold_idx}:')
    print(splits["train"])
    print(splits["val"])
    print(splits["test"])
    return splits

db = lmdb.open(dest_dir, map_size=1000000000)

dic = {}

def process(subject, day):
    sub_path = os.path.join(root_dir, subject, day)
    files = sorted(os.listdir(sub_path))
    runs = [f.split('_')[0] for f in os.listdir(sub_path) if f.endswith(".csv")]
    ds = []
    for run in runs:
        with open(f"{sub_path}/{run}_Session setup.json", "r", encoding="utf-8") as f:
            setup = json.load(f)

        raw = load_eeg_csv_to_ct(f"{sub_path}/{run}_Raw Signals.csv")  # shape (22, T)
        raw = preprocess_eeg_mne_prefix(raw)  # shape (22, T)

        with open(f"{sub_path}/{run}_Action label.txt", "r", encoding="utf-8") as f:
            actions = [int(line.strip()) for line in f if line.strip()]
        id2name = {3:"RH", 4:"LH", 5:"RF", 6:"LF"}
        mi_labels = [a - 3 for a in actions if a in id2name]

        with open(f"{sub_path}/{run}_Event timestamp.txt", "r", encoding="utf-8") as f:
            ts = [int(line.strip()) for line in f if line.strip()]  # len = 12

        # assert len(mi_labels) == len(ts)
        if len(mi_labels) != len(ts):
            continue

        for i in range(len(ts)):
            s = ts[i]
            e = ts[i] + 4 * 128
            y = mi_labels[i]
            x = preprocess_eeg_mne_suffix(raw[:, s:e]) 
            if x.shape[1] != 256 * 4:
                continue
            x = x.reshape(22, 4, 256)
            sample_key = f'{subject}_{day}_{run}_{i}'
            data_dict = {
                'sample': x, 'label': y
            }
            
            global db
            txn = db.begin(write=True)
            txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict, protocol=pickle.HIGHEST_PROTOCOL))
            txn.commit()
            ds.append(sample_key)
    return ds

for subject in subjects:
    sub_path = os.path.join(root_dir, subject)
    dirs = sorted(os.listdir(sub_path))
    ds = []
    for day in dirs:
        if os.path.isdir(os.path.join(sub_path, day)):
            ds += process(subject=subject, day=day)
    dic[subject] = ds
                
dataset = []

for i in range(5):
    ls_subjects = stratified_split(fold_idx=i)
    keys = {
        'train': [],
        'val': [],
        'test': []
    }
    for key in ls_subjects.keys():
        for subject in ls_subjects[key]:
            for sample_key in dic[subject]:
                keys[key].append(sample_key)
    dataset.append(keys)                

txn = db.begin(write=True)
txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset, protocol=pickle.HIGHEST_PROTOCOL))
txn.commit()
db.close()