from scipy import signal
import re
import lmdb
import pickle
import numpy as np
import pandas as pd
from scipy.signal import resample
from tqdm import tqdm

dataset_dirs = [
    '/mnt/disk1/aiotlab/namth/EEGFoundationModel/datasets/nfeeg/94-Vietnamese-Characters EEG Dataset - Female.csv',
    '/mnt/disk1/aiotlab/namth/EEGFoundationModel/datasets/nfeeg/94-Vietnamese-Characters EEG Dataset - Male.csv'
]

preprocessed_data_dir = '/mnt/disk1/aiotlab/namth/EEGFoundationModel/datasets/nfm_eegpt_official'

seed = 3407 # a`c wys`
np.random.seed(seed)

selected_channels = [
    'Fp1'
]

train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15

def str_to_array(s):
    # Bỏ các ký tự [ và ]
    s = s.replace('[', ' ').replace(']', ' ')
    # Dùng regex tách theo khoảng trắng hoặc dấu phẩy
    tokens = re.split(r'[,\s]+', s.strip())
    # Bỏ chuỗi rỗng
    tokens = [t for t in tokens if t != '']
    # Chuyển sang numpy array kiểu int
    return np.array([int(t) for t in tokens], dtype=np.int64)
    
def str_to_label(s: str) -> np.ndarray:
    """
    Tách chuỗi theo khoảng trắng và xuống dòng,
    bỏ qua các khoảng trắng liên tiếp.
    Trả về numpy array kiểu int.
    """
    # Bỏ các ký tự [ và ]
    s = s.replace('[', ' ').replace(']', ' ')
    # Dùng regex tách theo khoảng trắng hoặc dấu phẩy
    tokens = re.split(r'[,\s]+', s.strip())
    # Bỏ chuỗi rỗng
    tokens = [t for t in tokens if t != '']
    # Chuyển sang numpy array kiểu int
    label = np.array([int(t) for t in tokens], dtype=np.int64)[0]
    return label.item()


def resample_512_to_256(x: np.ndarray) -> np.ndarray:
    """
    x: np.array có shape (512,) - 1 giây tín hiệu ở 512 Hz
    Trả về: np.array có shape (1, 256) - resample xuống 256 Hz
    """
    if x.shape[0] != 512:
        raise ValueError(f"Expected input shape (512,), got {x.shape}")
    
    y = resample(x, 256)           # resample xuống 256 điểm
    return y.reshape(1, -1)        # shape (1, 256)

dataset = {
    'train': list(),
    'val': list(),
    'test': list(),
}

db = lmdb.open(preprocessed_data_dir, map_size=1000000000)
for idx, dataset_dir in enumerate(dataset_dirs):
    print(f'Processing {dataset_dir} ...')
    df_dict = [[] for x in range(94)]
    signal = list()
    df = pd.read_csv(dataset_dir)
    n = len(df)
    lst = range(n)
    for i in lst:
        row = df.iloc[i]
        x = resample_512_to_256(str_to_array(row['col 1']))
        y = str_to_label(row['col 3'])
        signal.append(x)
        df_dict[y].append(i)

    for i in tqdm(range(94)):
        np.random.shuffle(df_dict[i])
        n_i = len(df_dict[i])
        n_train = int(n_i * train_ratio)
        n_val = int(n_i * val_ratio)
        indices = {
            'train': list(),
            'val': list(),
            'test': list(),
        }
        indices['train'] = df_dict[i][:n_train]
        indices['val'] = df_dict[i][n_train:n_train + n_val]
        indices['test'] = df_dict[i][n_train + n_val:]
        for split in ['train', 'val', 'test']:
            for j in indices[split]:
                sample_key = f'{idx}-{j}'
                sample = signal[j]
                data_dict = {
                    'sample': sample, 'label': i
                }
                    
                txn = db.begin(write=True)
                txn.put(sample_key.encode(), pickle.dumps(data_dict, protocol=pickle.HIGHEST_PROTOCOL))
                txn.commit()
                dataset[split].append(sample_key)

txn = db.begin(write=True)
txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset, protocol=pickle.HIGHEST_PROTOCOL))
txn.commit()
db.close()

print('Done!')