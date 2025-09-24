import os
import mne
import numpy as np
import pickle
import lmdb
from scipy.signal import butter, resample, filtfilt
import json
from tqdm import tqdm
# Chỉ tắt cảnh báo RuntimeWarning có chứa "annotation(s) that were expanding outside the data range"
import warnings
warnings.filterwarnings(
    "ignore", 
    category=RuntimeWarning, 
    message=".*annotation.*outside the data range.*"
)

# ------------------ Preprocessing ------------------
root_dir = "/home/aiotlab/.cache/kagglehub/datasets/patrickiitmz/eeget-als-dataset/versions/1/EEGET-ALS Dataset"
output_db = "/mnt/disk1/aiotlab/namth/EEGFoundationModel/datasets/als_eegpt_official"
task = "LRF0"
win_sec = 4
des_fs = 256  # resample to 256 Hz
target_len = win_sec * des_fs
seed = 3407 # a`c wys`
np.random.seed(seed)

def process_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """
    Fit ICA trên bản high-pass 1 Hz, rồi apply lên bản band-pass 0.3–50 Hz.
    Trả về: raw đã clean (0.3–50 Hz + ICA). Raw gốc không đổi.
    """
    # --- 1) Bản để FIT ICA: high-pass 1 Hz bằng MNE (để MNE ghi nhận highpass, hết warning)
    raw_hp = raw.copy().filter(l_freq=1.0, h_freq=None, picks="eeg")
    raw_hp.notch_filter(freqs=[50.0], picks="eeg", method="iir", verbose=False)

    # --- 2) Bản để APPLY ICA: tự lọc 0.3–50 Hz (demean + band-pass, zero-phase)
    fs = float(raw.info["sfreq"])
    nyq = 0.5 * fs
    b, a = butter(5, [0.3/nyq, 50.0/nyq], btype="band")

    data = raw.get_data(picks="eeg")  # (n_eeg, n_samples)
    data = data - np.mean(data, axis=1, keepdims=True)
    data_bp = filtfilt(b, a, data, axis=-1)

    # Lắp lại thành Raw để apply ICA (giữ nguyên info/kênh EEG)
    info_eeg = raw.copy().pick("eeg").info  # info chỉ của EEG, đúng thứ tự kênh
    raw_bp = mne.io.RawArray(data_bp, info_eeg, verbose=False)
    raw_bp.notch_filter(freqs=[50.0], picks="eeg", method="iir", verbose=False)

    # --- 3) Fit ICA trên raw_hp (EEG)
    n_comp = max(5, min(20, raw_hp.info["nchan"] - 1))
    ica = mne.preprocessing.ICA(n_components=n_comp, method="picard",
                                random_state=97, max_iter="auto")
    ica.fit(raw_hp)

    # --- 4) Apply ICA học từ raw_hp lên raw_bp (cùng không gian kênh EEG)
    raw_clean_eeg = raw_bp.copy()
    ica.apply(raw_clean_eeg)

    # Có kênh khác: thay phần EEG trong bản gốc bằng EEG đã clean
    raw_out = raw.copy()
    eeg_picks = mne.pick_types(raw_out.info, eeg=True, exclude=())
    raw_out._data[eeg_picks, :] = raw_clean_eeg.get_data()
    return raw_out

import mne
import numpy as np
from scipy.signal import butter, filtfilt

def process_raw_no_ica(raw: mne.io.BaseRaw,
                       bp_band=(0.3, 50.0),
                       notch_freq=50.0,
                       order=5) -> mne.io.BaseRaw:
    """
    Làm sạch EEG KHÔNG ICA, xử lý trực tiếp trên raw:
      - Band-pass (Butterworth zero-phase)
      - Demean
      - Notch filter
    """
    raw_out = raw.copy()

    fs = float(raw.info["sfreq"])
    nyq = 0.5 * fs
    low, high = bp_band
    b, a = butter(order, [low/nyq, high/nyq], btype="band")

    eeg_picks = mne.pick_types(raw_out.info, eeg=True, exclude=())
    data = raw_out.get_data(picks=eeg_picks)

    # demean + band-pass
    data = data - np.mean(data, axis=1, keepdims=True)
    data_bp = filtfilt(b, a, data, axis=-1)

    # gán dữ liệu lọc trở lại raw_out
    raw_out._data[eeg_picks, :] = data_bp

    # notch filter ngay trên raw_out
    if notch_freq is not None and notch_freq > 0:
        raw_out.notch_filter(freqs=[float(notch_freq)],
                             picks=eeg_picks,
                             method="iir",
                             verbose=False)

    return raw_out


# ------------------ Label Mapping ------------------
def map_label(annotation: str, scenario_id: int, task="LRF0"):
    if annotation == "Resting":
        return 0
    if task == "LR0":     # 3-class
        if scenario_id == 1: return 1
        if scenario_id == 2: return 2
        return None
    elif task == "LRF0":  # 4-class
        if scenario_id == 1: return 1
        if scenario_id == 2: return 2
        if scenario_id == 3: return 3
        return None
    return None

standard_19 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6",
    "O1", "O2"
]

map_10_10 = {
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8"
}
mapped_channels = [map_10_10.get(ch, ch) for ch in standard_19]

train_ratio, val_ratio = 0.85, 0.15  # train, val
subjects = [x for x in os.listdir(root_dir) if x.startswith("id")]
np.random.shuffle(subjects)
n_subj = len(subjects)
n_train = int(n_subj * train_ratio)
n_val = n_subj - n_train

ls_dict = {
    "train": subjects[:n_train],
    "val": subjects[n_train:n_train + n_val],
    "test": [x for x in os.listdir(root_dir) if x.startswith("ALS")],
}

print("Train subjects:", ls_dict["train"])
print("Val subjects:", ls_dict["val"])
print("Test subjects:", ls_dict["test"])

rev_ls_dict = {}

for x in ls_dict.keys():
    for subj in ls_dict[x]:
        rev_ls_dict[subj] = x

db = lmdb.open(output_db, map_size=16106127360)
dataset = {"train": [], "val": [], "test": []}

for subject in sorted(os.listdir(root_dir)):
    subj_path = os.path.join(root_dir, subject)
    if not os.path.isdir(subj_path):
        continue

    for time_folder in sorted(os.listdir(subj_path)):
        time_path = os.path.join(subj_path, time_folder)
        if not os.path.isdir(time_path):
            continue

        for scenario_folder in sorted(os.listdir(time_path)):
            scenario_path = os.path.join(time_path, scenario_folder)
            if not os.path.isdir(scenario_path):
                continue

            scenario_id = int(scenario_folder.replace("scenario", ""))
            eeg_file = os.path.join(scenario_path, "EEG.edf")
            anno_file = os.path.join(scenario_path, "eeg.json")

            if not (os.path.exists(eeg_file) and os.path.exists(anno_file)):
                continue

            # Load scenario-level annotation (Resting/Thinking/Typing)
            with open(anno_file, "r") as f:
                meta = json.load(f)
            scenario_anno = meta.get("annotation", "Resting")

            # Load EDF
            raw = mne.io.read_raw_edf(eeg_file, preload=True, verbose=True)
            fs = int(raw.info["sfreq"]) 
            raw = process_raw_no_ica(raw)
            raw.pick([ch for ch in mapped_channels if ch in raw.ch_names])

            data = raw.get_data()

            # EDF annotations (per-frame events)
            for ann in raw.annotations:
                onset = int(    ann["onset"] * fs)
                duration = int(ann["duration"] * fs)
                desc = str(ann["description"])

                label = map_label(desc if desc else scenario_anno,
                                  scenario_id,
                                  task=task)
                if label is None:
                    continue

                seg_data = data[:, onset:onset+duration]
                win_len = fs * win_sec
                n_win = seg_data.shape[1] // win_len

                for w in range(n_win):
                    seg = seg_data[:, w*win_len:(w+1)*win_len]

                    # Resample each window to target_len (e.g. 800)
                    seg = resample(seg, target_len, axis=-1)

                    # Reshape (C, 4, 200) for 800
                    seg = seg.reshape(seg.shape[0], win_sec, des_fs)

                    sample_key = f"{subject}-{time_folder}-{scenario_folder}-{ann['onset']:.2f}-win{w}"
                    data_dict = {"sample": seg.astype(np.float32), "label": label}

                    txn = db.begin(write=True)
                    txn.put(sample_key.encode(), pickle.dumps(data_dict))
                    txn.commit()
                    dataset[rev_ls_dict[subject]].append(sample_key)

# save keys
txn = db.begin(write=True)
txn.put("__keys__".encode(), pickle.dumps(dataset))
txn.commit()
db.close()

print("✅ Done: 4s windows aligned with EDF annotations saved to LMDB")