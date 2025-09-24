import scipy
from scipy import signal
import os
import re
import lmdb
import pickle
import mne
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

root_dir = '/mnt/disk1/aiotlab/namth/ds004796-download'
output_dir = '/mnt/disk1/aiotlab/namth/EEGFoundationModel/datasets/pearl_30s_rest/eegpt_official'
csv_path = "/mnt/disk1/aiotlab/namth/ds004796-download/meta.csv"
fs = 256
seed = 3407 # a`c wys`
task = 'rest' # 'msit' or 'rest' or 'sternberg'
np.random.seed(seed)

# Only consider these subjects
NEUTRAL_SUBJECTS = [1,2,6,7,8,10,13,14,15,16,17,18,19,22,23,24,25,26,28,29,31]
RISKY_SUBJECTS   = [47,53,57,58,59,60,62,63,65,67,70,73,74,75,77,78,79,80]
LABELS = { "neutral": 0, "risky": 1 }

# collect only BrainVision task-msit_eeg .vhdr files under sub-*/eeg/
files = []
for sub in sorted([d for d in os.listdir(root_dir) if d.startswith('sub-') and os.path.isdir(os.path.join(root_dir, d))]):
    # filter to only subjects in neutral + risky sets
    m = re.match(r'^sub-(\d+)$', sub)
    if not m:
        continue
    sid = int(m.group(1))
    if (sid not in NEUTRAL_SUBJECTS) and (sid not in RISKY_SUBJECTS):
        continue
    eeg_dir = os.path.join(root_dir, sub, 'eeg')
    if not os.path.isdir(eeg_dir):
        continue
    for f in sorted(os.listdir(eeg_dir)):
        if f.endswith(f'_task-{task}_eeg.vhdr'):
            files.append(os.path.join(sub, 'eeg', f))
files = sorted(files)

np.random.shuffle(files)  # Randomly shuffle files
print(files)

selected_channels = [
    'Fp1','Fp2','F3','F4','F7','F8',
    'T7','T8','C3','C4','P7','P8',
    'P3','P4','O1','O2','Fz','Cz','Pz',
]

def subject_label_from_path(rel_path: str) -> int:
    # extract subject id (sub-XX) and map via NEUTRAL/RISKY lists
    m = re.search(r'sub-(\d+)', rel_path)
    if not m:
        return -1
    sid = int(m.group(1))
    if sid in NEUTRAL_SUBJECTS:
        return LABELS['neutral']
    if sid in RISKY_SUBJECTS:
        return LABELS['risky']
    return -1

files_by_label = {0: [], 1: []}
for f in files:
    lbl = subject_label_from_path(f)
    if lbl in (0, 1):
        files_by_label[lbl].append(f)

def subject_id_from_path(s: str) -> str:
        return s[0:6]

def id_from_path(s: str) -> str:
        return s[4:6]

def extract_df(df, subject_id):
    ids = [subject_id_from_path(f) for f in subject_id]
    return df[df["participant_id"].isin(ids)].copy().drop(columns=["participant_id"])

def apply_standardize_df(df, mean, std):
    # Lấy các cột số
    num_cols = df.select_dtypes(include="number").columns
    
    # Thay null bằng mean của từng cột (theo mean đã cho)
    df[num_cols] = df[num_cols].fillna(mean)
    
    # Standardize bằng mean, std đã cho
    df[num_cols] = (df[num_cols] - mean) / std
    return df

def standardize_df(df):
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    scaler = StandardScaler()
    mean, std = df[num_cols].mean(), df[num_cols].std(ddof=0)
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df, mean, std

def stratified_split(fold_idx):
    splits = {'train': [], 'test': [], 'val': []}
    for lbl, lst in files_by_label.items():
        n = len(lst)
        n_test = n // 5
        test_start = fold_idx * n_test
        test_end = test_start + n_test if fold_idx < 4 else n  # last fold takes the remainder
        splits['test'] += lst[test_start:test_end]
        train = lst[:test_start] + lst[test_end:]
        num_val = len(train) // 5
        splits['train'] += train[num_val:]
        splits['val'] += train[:num_val]
    return splits

eeg_dict = {}

def pre_reading_eeg():
    for files_key in files_by_label.keys():
        for (idx, file) in enumerate(files_by_label[files_key]):
            path = os.path.join(root_dir, file)
            try:
                raw = mne.io.read_raw_brainvision(path, preload=True, verbose='ERROR')
            except Exception as e:
                print(f'[WARN] failed to read {path}: {e}')
                continue

            # ensure required channels exist; skip if missing
            have = set(ch.upper() for ch in raw.ch_names)
            need = [ch.upper() for ch in selected_channels]
            if not all(ch in have for ch in need):
                print(f'[WARN] missing channels in {path}, skip.')
                continue

            raw.pick(selected_channels)
            raw.reorder_channels(selected_channels)
            raw.filter(0.3, 75)
            raw.notch_filter(50)
            try:
                raw.resample(fs)
            except Exception as e:
                print(f'[WARN] resample failed {path}: {e}')
                continue

            # data in microvolts as float32
            try:
                eeg = raw.get_data(units='uV', reject_by_annotation='omit').astype(np.float32)
            except TypeError:
                eeg = (raw.get_data(reject_by_annotation='omit') * 1e6).astype(np.float32)

            chs, points = eeg.shape
            a = points % (30 * fs)
            if a != 0:
                eeg = eeg[:, :-a]
            if eeg.size == 0:
                continue

            # reshape to (segments, chs, 30, fs)
            eeg = eeg.reshape(chs, -1, 30, fs).transpose(1, 0, 2, 3)

            eeg_dict[file] = eeg

def fold_construct(fold_idx):
    print(f'Constructing fold {fold_idx} ...')

    dataset = {
        'train': list(),
        'val': list(),
        'test': list(),
    }
    df = pd.read_csv(csv_path, encoding="utf-8", sep=",")  # đổi sep nếu ; hoặc \t
    df = df.drop(columns=['age', 'sex'])

    files_dict = stratified_split(fold_idx=fold_idx)

    df_dict = {
        'train': extract_df(df, files_dict['train']),
        'val': extract_df(df, files_dict['val']),
        'test': extract_df(df, files_dict['test']),
    }

    train, mean, std = standardize_df(df_dict['train'])
    val = apply_standardize_df(df_dict['val'], mean, std)
    test = apply_standardize_df(df_dict['test'], mean, std)

    mean = mean.to_numpy()
    std = std.to_numpy()

    # print(files_dict)
    subject = {
        'train': set(id_from_path(f) for f in files_dict['train']),
        'val': set(id_from_path(f) for f in files_dict['val']),
        'test': set(id_from_path(f) for f in files_dict['test']),
    }
    print(subject['train'])
    print(subject['val'])
    print(subject['test'])

    db = lmdb.open(f'{output_dir}/fold_{fold_idx}', map_size=1000000000)
    for files_key in files_dict.keys():
        for (idx, file) in enumerate(files_dict[files_key]):
            eeg = eeg_dict.get(file, None)
            if eeg is None:
                # print(f'[WARN] no pre-read eeg for {file}, skip.')
                continue
            # label per subject
            label = subject_label_from_path(file)
            if label == -1:
                print(f'[WARN] unknown label for {file}, skip.')
                continue
            meta = df_dict[files_key].iloc[idx].to_numpy()

            for i, sample in enumerate(eeg):
                sample_key = f'{file[:-5]}-{i}'  # Fixed typo from file[:-4]
                data_dict = {
                    'sample': sample, 'label': label, 'meta': meta
                }
                txn = db.begin(write=True)
                txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict, protocol=pickle.HIGHEST_PROTOCOL))
                txn.commit()
                dataset[files_key].append(sample_key)

    txn = db.begin(write=True)
    txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset, protocol=pickle.HIGHEST_PROTOCOL))
    txn.put(key='__mean__'.encode(), value=pickle.dumps(mean, protocol=pickle.HIGHEST_PROTOCOL))
    txn.put(key='__std__'.encode(), value=pickle.dumps(std, protocol=pickle.HIGHEST_PROTOCOL))
    txn.commit()
    db.close()

pre_reading_eeg()
for fold_idx in range(5):
    fold_construct(fold_idx)