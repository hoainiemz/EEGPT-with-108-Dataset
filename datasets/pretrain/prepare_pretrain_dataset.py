"""
use this script to prepare the mixed pretraining dataset
"""

import os
import torch
import shutil
import random
import mne
import numpy as np

import pandas as pd
from torcheeg.datasets import CSVFolderDataset
from torcheeg import transforms
import copy

import torcheeg
import torch
from torcheeg.datasets import M3CVDataset, TSUBenckmarkDataset, DEAPDataset, SEEDDataset, moabb
from torcheeg import transforms
from torcheeg.datasets.constants import SEED_CHANNEL_LIST, M3CV_CHANNEL_LIST, TSUBENCHMARK_CHANNEL_LIST
from torcheeg.datasets import CSVFolderDataset
from torchaudio.transforms import Resample
import lmdb, pickle, io
from typing import List, Optional

CUSTOM_CHANNEL_LIST = None  # will fill after reading the first file

use_channels_names = [      'FP1', 'FPZ', 'FP2', 
                               'AF3', 'AF4', 
            'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 
        'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 
            'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 
        'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
             'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 
                      'PO7', 'PO3', 'POZ',  'PO4', 'PO8', 
                               'O1', 'OZ', 'O2', ]

DATA_PATHS = [
    '/mnt/disk1/aiotlab/namth/EEGFoundationModel/datasets/108/mdb',
    '/mnt/disk1/aiotlab/namth/EEGFoundationModel/datasets/108/mdb'
]

def _normalize_eeg_ch_names(ch_names):
    # Chuẩn hoá: bỏ dấu chấm, viết hoa
    normed = [x.strip('.').upper() for x in ch_names]

    # Mapping alias để khớp với danh sách 58 kênh EEGPT
    mapping = {
        # Temporal (hệ cũ ↔ hệ mở rộng)
        'T3': 'T7',
        'T4': 'T8',
        'T5': 'P7',
        'T6': 'P8',
        # Earlobes / mastoid ↔ chuẩn mở rộng
        'A1': 'TP9',
        'A2': 'TP10',
    }

    normed = [mapping.get(ch, ch) for ch in normed]
    return normed
# ------------------- Custom LMDB Dataset
lmdb_selected_channels = _normalize_eeg_ch_names(['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'])

class LmdbTensorDataset(torch.utils.data.Dataset):
    """
    Đọc LMDB (.mdb) với cặp {key: mã_bệnh_nhân, value: tensor(c, t, d)}.
    Trả về x có shape (1, c, t*d) để tương thích với pipeline hiện tại (sau đó code main sẽ squeeze(0)).
    """

    def __init__(self, db_path: str, key_filter: Optional[List[bytes]] = None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, readonly=True, lock=False, readahead=True, meminit=False)
        with self.env.begin() as txn:
            raw = txn.get(b"__keys__")
            cursor = txn.cursor()
            if raw is not None:
                keys = pickle.loads(raw)
                self.keys = [k.encode() for k in keys]
            else:
                self.keys = [k for k, _ in cursor]

    def __len__(self):
        return len(self.keys)

    @staticmethod
    def _bytes_to_tensor(b: bytes) -> torch.Tensor:
        """
        Giải mã value bytes từ LMDB thành torch.Tensor.
        Hỗ trợ cả numpy.ndarray và torch.Tensor.
        """
        # thử pickle trước (LMDB thường lưu numpy bằng pickle)
        obj = pickle.loads(b)
        return torch.from_numpy(obj)

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.env.begin() as txn:
            val_bytes = txn.get(key)

        x = self._bytes_to_tensor(val_bytes).float()  # kỳ vọng (c, t, d) hoặc (c, t*d)
        if x.ndim == 3:
            c, t, d = x.shape
            x = x.reshape(c, t * d)
        elif x.ndim == 2:
            # đã là (c, t*d) thì giữ nguyên
            pass
        else:
            raise ValueError(f"Giá trị tại key {key} có shape không hợp lệ: {tuple(x.shape)} (kỳ vọng 2D hoặc 3D)")

        reorder_idx = [
            lmdb_selected_channels.index(ch) 
            for ch in use_channels_names 
            if ch in lmdb_selected_channels
        ]
        x = x[reorder_idx, :]   # (len(use_channels_names), t*d)

        y = 0
        return x, y

def get_lmdb_dataset(db_path: str) -> torch.utils.data.Dataset:
    """
    Tạo dataset đọc từ 1 file .mdb (LMDB). Mỗi sample trả về (C, T).
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Không tìm thấy LMDB: {db_path}")
    return LmdbTensorDataset(db_path=db_path)
# ------------------- Custom Dataset

def temporal_interpolation(x, desired_sequence_length, mode='nearest'):
    # squeeze and unsqueeze because these are done before batching
    x = x - x.mean(-2)
    if len(x.shape) == 2:
        return torch.nn.functional.interpolate(x.unsqueeze(0), desired_sequence_length, mode=mode).squeeze(0)
    # Supports batch dimension
    elif len(x.shape) == 3:
        return torch.nn.functional.interpolate(x, desired_sequence_length, mode=mode)
    else:
        raise ValueError("TemporalInterpolation only support sequence of single dim channels with optional batch")

def _read_edf_as_fixed_epochs(file_path, epoch_len_s=6.0, sfreq_target:int=256, reref_avg=True):
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

    # Chuẩn hoá tên kênh (giúp PickElectrode mapping chuẩn)
    raw.rename_channels({ch: ch.strip('.').upper() for ch in raw.ch_names})

    # (Tuỳ chọn) average re-reference để gần với thiết lập trong bài
    if reref_avg:
        raw.set_eeg_reference('average', projection=False, verbose=False)

    # (Tuỳ chọn) resample về 256 Hz cho đồng nhất kích thước
    if sfreq_target:
        raw.resample(sfreq_target, npad="auto")

    # thời lượng file (giây)
    dur = (raw.n_times / raw.info["sfreq"])
    if dur < epoch_len_s:
        return None, None

    # Cắt thành cửa sổ 6s (non-overlap) cho pretrain
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=epoch_len_s,
        overlap=0.0,
        preload=True,
        verbose=False
    )
    
    return epochs, _normalize_eeg_ch_names(raw.ch_names)

def _build_custom_meta(root_dir, epoch_len_s=6.0, sfreq_target=256, dataset_name=None):
    import glob
    rows = []
    global CUSTOM_CHANNEL_LIST
    CUSTOM_CHANNEL_LIST = None

    edf_files = sorted(glob.glob(os.path.join(root_dir, "**/*.edf"), recursive=True))
    if len(edf_files) == 0:
        raise FileNotFoundError(f"No .edf found under {root_dir}")

    # Đếm nhanh số epoch mỗi file để lưu xuống meta (đỡ phải load lại khi iterate)
    for fp in edf_files:
        epochs, chs = _read_edf_as_fixed_epochs(fp, epoch_len_s=epoch_len_s, sfreq_target=sfreq_target)
        if epochs is None:
            print(f"[SKIP] {fp} too short")
            continue
        else:    
            print(f"[SUCCESS] {fp} processed")
        if CUSTOM_CHANNEL_LIST is None:
            CUSTOM_CHANNEL_LIST = chs  # lấy layout kênh chuẩn theo file đầu tiên
        # lưu đường dẫn & số epoch -> để read_fn tái tạo đúng
        rows.append({"file_path": fp, "n_epochs": len(epochs)})
    df = pd.DataFrame(rows)
    os.makedirs(f"./{dataset_name}", exist_ok=True)
    csv_path = f"./{dataset_name}/meta.csv"
    df.to_csv(csv_path, index=False)
    print(f"{dataset_name} channel list: {CUSTOM_CHANNEL_LIST}")
    return csv_path

def _custom_read_fn(file_path, n_epochs=None, epoch_len_s=6.0, sfreq_target=256, **kwargs):
    epochs, _ = _read_edf_as_fixed_epochs(file_path, epoch_len_s=epoch_len_s, sfreq_target=sfreq_target)
    # (tuỳ) n_epochs trong meta chỉ để info; epochs đã đúng độ dài cửa sổ
    return epochs

# đặt ở cùng file, trước khi Compose được dùng
import numpy as np

class PadMissingChannels:
    def __init__(self, target_chs, current_chs):
        """
        target_chs: list[str]  -> danh sách 58 kênh chuẩn (use_channels_names)
        current_chs: list[str] -> danh sách kênh thực tế trong EDF
        """
        self.target_chs = target_chs
        self.current_chs = current_chs
        self.index_map = [
            (current_chs.index(ch) if ch in current_chs else None)
            for ch in target_chs
        ]

    def __call__(self, eeg=None, **kwargs):
        """
        eeg: np.ndarray shape (n_channels_actual, n_times)
        return: dict với 'eeg' shape (len(target_chs), n_times)
        """
        assert eeg is not None, "PadMissingChannels expects keyword arg 'eeg'"
        n_times = eeg.shape[-1]
        out = np.zeros((len(self.target_chs), n_times), dtype=eeg.dtype)
        for i, idx in enumerate(self.index_map):
            if idx is not None:
                out[i, :] = eeg[idx, :]
        # Trả về dict theo chuẩn torcheeg
        kwargs['eeg'] = out
        return kwargs


def get_custom_edf_dataset(root_dir:str, epoch_len_s=6.0, sfreq_target=256, dataset_name:str = ""):
    """
    Plug-and-play for any EDF folder structure. It builds a CSV meta and returns a CSVFolderDataset.
    """
    csv_path = _build_custom_meta(root_dir, epoch_len_s=epoch_len_s, sfreq_target=sfreq_target, dataset_name=dataset_name)

    # Tạo dataset dạng CSVFolderDataset để dùng chung transforms của bạn
    dataset = CSVFolderDataset(
        csv_path=csv_path,
        read_fn=lambda file_path, n_epochs=None, **kw: _custom_read_fn(
            file_path=file_path,
            n_epochs=n_epochs,
            epoch_len_s=epoch_len_s,
            sfreq_target=sfreq_target,
        ),
        io_path=data_root_path + 'io/' + dataset_name,
        online_transform=transforms.Compose([
            # map kênh của dataset về danh sách 58 kênh chuẩn "use_channels_names"
            transforms.PickElectrode(
                transforms.PickElectrode.to_index_list(use_channels_names, CUSTOM_CHANNEL_LIST)
            ),
            # PadMissingChannels(use_channels_names, CUSTOM_CHANNEL_LIST),
            transforms.ToTensor(),
            # Nội suy thời gian về đúng 256 * epoch_len_s mẫu / cửa sổ, scale sang µV cho thống nhất
            transforms.Lambda(lambda x: temporal_interpolation(x, int(256 * epoch_len_s)) * 1e3),
            transforms.To2d(),
        ]),
        label_transform=transforms.Compose([
            transforms.Lambda(lambda _: 0)  # pretrain không cần nhãn
        ]),
        num_worker=4
    )
    return dataset

# ------------------- PhysioMI
data_root_path = "./io_root/"

def temporal_interpolation(x, desired_sequence_length, mode='nearest'):
    # squeeze and unsqueeze because these are done before batching
    x = x - x.mean(-2)
    if len(x.shape) == 2:
        return torch.nn.functional.interpolate(x.unsqueeze(0), desired_sequence_length, mode=mode).squeeze(0)
    # Supports batch dimension
    elif len(x.shape) == 3:
        return torch.nn.functional.interpolate(x, desired_sequence_length, mode=mode)
    else:
        raise ValueError("TemporalInterpolation only support sequence of single dim channels with optional batch")
    


def get_physionet_dataset():
    channels_name = ['Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..', 'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 'Fpz.', 'Fp2.', 'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..', 'F4..', 'F6..', 'F8..', 'Ft7.', 'Ft8.', 'T7..', 'T8..', 'T9..', 'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..', 'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..', 'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.', 'O1..', 'Oz..', 'O2..', 'Iz..']

    """
    In summary, the experimental runs were:

    1   Baseline, eyes open
    2   Baseline, eyes closed
    3   Task 1 (open and close left or right fist)                  -> 4 5
    4   Task 2 (imagine opening and closing left or right fist)     -> 0 1
    5   Task 3 (open and close both fists or both feet)             -> 6 7
    6   Task 4 (imagine opening and closing both fists or both feet)-> 2 3
    7   Task 1
    8   Task 2
    9   Task 3
    10  Task 4
    11  Task 1
    12  Task 2
    13  Task 3
    14  Task 4

    """
    session_id2task_id = {
        3:1, 4:2, 5:3, 6:4,
        7:1, 8:2, 9:3, 10:4,
        11:1, 12:2, 13:3, 14:4,
        
    }
    task2event_id = {
        0:dict([('T1', 4), ('T2', 5)]),
        1:dict([('T1', 0), ('T2', 1)]),
        2:dict([('T1', 6), ('T2', 7)]),
        3:dict([('T1', 2), ('T2', 3)])
    }

    
    if not os.path.exists(data_root_path+'io/PhysioNetMI'):
        src_path = "./PhysioNetMI/files/eegmmidb/1.0.0/"
        ls = []
        channels_name = None
        for subject in range(1,110):
            for task in [0,1,2,3]:
                for session in [3,7,11]:
                    session += task
                    file_path = src_path + "S{:03d}".format(subject) + '/' + "S{:03d}R{:02d}.edf".format(subject,session)
                    raw = mne.io.read_raw_edf(file_path,preload=True)
                    
                    if channels_name is None:
                        channels_name = copy.deepcopy(raw.ch_names)
                    else:
                        assert channels_name == raw.ch_names
                        
                    event_id = task2event_id[session_id2task_id[session]-1]
                    # -- split epochs
                    epochs = mne.Epochs(raw, 
                            events = mne.events_from_annotations(raw, event_id=event_id, chunk_duration=None)[0], 
                            tmin=0, tmax=0 + 6 - 1 / raw.info['sfreq'], 
                            preload=True, 
                            decim=1,
                            baseline=None, 
                            reject_by_annotation=False)
                    
                    d = {
                        # "subject_id":[subject],
                        # "sess_id": [session],
                        # "task_id":[task],
                        "file_path":[file_path],
                        "labels":"".join([str(ev[-1]) for ev in epochs.events])
                    }
                    
                    ls.append(pd.DataFrame(d))
        table = pd.concat(ls, ignore_index=True)
        # print(table)
        print(channels_name)
        table.to_csv("./PhysioNetMI/physionetmi_meta.csv", index=False)

        def default_read_fn(file_path, task_id=None, session_id=None, subject_id=None, **kwargs):
            session_id = int(file_path.split('R')[-1].split('.')[0])
            # -- read raw file
            raw = mne.io.read_raw_edf(file_path,preload=True)
            
            event_id = task2event_id[session_id2task_id[session_id]-1]
            # -- split epochs
            epochs = mne.Epochs(raw, 
                    events = mne.events_from_annotations(raw, event_id=event_id, chunk_duration=None)[0], 
                    tmin=0, tmax=0 + 6 - 1 / raw.info['sfreq'], 
                    preload=True, 
                    decim=1,
                    baseline=None, 
                    reject_by_annotation=False)
            
            return epochs

        dataset = CSVFolderDataset(csv_path="./PhysioNetMI/physionetmi_meta.csv",
                                read_fn=default_read_fn,
                                io_path=data_root_path+'io/PhysioNetMI',
                            online_transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.To2d()
                            ]),
                                #    label_transform=transforms.Select('label'),
                                num_worker=4)
    dataset = CSVFolderDataset(
                        io_path=data_root_path+'io/PhysioNetMI',
                        online_transform=transforms.Compose([
                            transforms.PickElectrode(transforms.PickElectrode.to_index_list(use_channels_names, PHYSIONETMI_CHANNEL_LIST)),
                            transforms.ToTensor(),
                            #   transforms.RandomWindowSlice(window_size=160*4, p=1.0),
                            transforms.Lambda(lambda x: temporal_interpolation(x, 256*6) * 1e3), #V-> 1000uV
                            transforms.To2d()
                        ]),
                        label_transform=transforms.Compose([
                            #   transforms.Select('labels'),
                            #   transforms.StringToInt()
                            transforms.Lambda(lambda x : 0)
                        ]))
    return dataset


# --------------- merge to Fold Dataset

def get_TSU_dataset():
    dataset = TSUBenckmarkDataset(
            root_path="./TSUBenchmark/",
            io_path=data_root_path+'io/tsu_benchmark',
            online_transform=transforms.Compose([
                transforms.PickElectrode(transforms.PickElectrode.to_index_list(use_channels_names, TSUBENCHMARK_CHANNEL_LIST)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: temporal_interpolation(x, 256*4) / 1000),# 1000uV
                transforms.To2d(),
            ]),
            label_transform=transforms.Select('trial_id'))
    return dataset


def get_M3CV_dataset():
    dataset = M3CVDataset(
                        root_path="./aistudio/",
                        io_path=data_root_path+'io/m3cv',
                        online_transform=transforms.Compose([
                            transforms.PickElectrode(transforms.PickElectrode.to_index_list(use_channels_names, M3CV_CHANNEL_LIST)),
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: temporal_interpolation(x, 256*4) / 1000),# 1000uV
                            transforms.To2d(),
                        ]),
                        label_transform=transforms.Compose([
                            transforms.Select('subject_id'),
                            transforms.StringToInt()
                        ]))
    return dataset

def get_SEED_dataset():
    dataset = SEEDDataset(
                            root_path="./SEED/",
                            io_path=data_root_path+'io/seed',
                          online_transform=transforms.Compose([
                              transforms.PickElectrode(transforms.PickElectrode.to_index_list(use_channels_names, SEED_CHANNEL_LIST)),
                              transforms.ToTensor(),
                            #   transforms.RandomWindowSlice(window_size=250*4, p=1.0),
                              transforms.Lambda(lambda x: temporal_interpolation(x, 256*10)/1000),# 1000uV
                              transforms.To2d(),                          
                          ]),
                          label_transform=transforms.Compose([
                              transforms.Select('emotion'),
                              transforms.Lambda(lambda x: x + 1)
                          ]))
    return dataset


if __name__=="__main__":
    import random
    import os
    import tqdm
    
    for path in DATA_PATHS:
        dataset = get_lmdb_dataset(db_path=path)
        tag = os.path.basename(path)
        print(len(dataset))
        print(dataset[0][0].shape)
        print(dataset[0][0].min(),dataset[0][0].max(), dataset[0][0].mean(),dataset[0][0].std())
        for i, (x,y) in tqdm.tqdm(enumerate(dataset)):
            dst="./merged/"
            if random.random()<0.1:
                dst+="ValidFolder/0/"
            else:
                dst+="TrainFolder/0/"
            os.makedirs(dst, exist_ok=True)
            data = x.squeeze_(0)
            # data = data.clone().detach().cpu()
            print(i, data.shape, len(data.shape)==2 and data.shape[0]==58 and data.shape[1]>=1024)
            # assert len(data.shape)==2 and data.shape[0]==58 and data.shape[1]>=1024
            torch.save(data, dst + tag+f"_{i}.edf")
            del data, x