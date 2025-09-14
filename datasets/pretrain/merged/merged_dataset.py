import pickle
import numpy as np
import lmdb
from torch.utils.data import Dataset, DataLoader
from utils_cbramod.util import to_tensor

from typing import List

class MergedPretrainingDataset(Dataset):
    def __init__(
            self,
            dataset_dirs: List[str],
            mode: str = 'train'  # 'train', 'val', 'test'
    ):
        super(MergedPretrainingDataset, self).__init__()
        self.dbs = [lmdb.open(dataset_dir, readonly=True, lock=False, readahead=True, meminit=False) for dataset_dir in dataset_dirs]
        self.keys = []
        for db_idx, db in enumerate(self.dbs):
            with db.begin(write=False) as txn:
                raw = txn.get('__keys__'.encode())
                if raw is None:
                    continue
                keys = pickle.loads(raw)
                # Store (key, db_idx) pairs
                self.keys.extend((k, db_idx) for k in keys)
        # self.keys = self.keys[:100000]
        np.random.shuffle(self.keys)
        num_val = int(0.1 * len(self.keys))
        if mode == 'train':
            self.keys = self.keys[num_val:]
        elif mode == 'val':
            self.keys = self.keys[:num_val]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key, db_idx = self.keys[idx]

        with self.dbs[db_idx].begin(write=False) as txn:
            key_bytes = key if isinstance(key, bytes) else key.encode()
            raw = txn.get(key_bytes)
            if raw is None:
                raise KeyError(f"Key '{key}' not found in LMDB index {db_idx}")
            patch = pickle.loads(raw)

        patch = to_tensor(patch)
        # print(patch.shape)
        return patch.flatten(start_dim=1)
    
    def collate(self, batch):
        x_data = np.array([x for x in batch])
        y_label = np.array([0 for x in batch])
        return to_tensor(x_data), to_tensor(y_label)

class LoadDataset(object):
    def __init__(self, dataset_dirs: List[str], batch_size=64):
        self.dataset_dirs = dataset_dirs
        self.batch_size = batch_size

    def get_data_loader(self):
        train_set = MergedPretrainingDataset(self.dataset_dirs, mode='train')
        val_set = MergedPretrainingDataset(self.dataset_dirs, mode='val')
        print(len(train_set), len(val_set))
        print(len(train_set) + len(val_set))
        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.batch_size,
                collate_fn=train_set.collate,
                shuffle=True,
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.batch_size,
                collate_fn=val_set.collate,
                shuffle=True,
            )
        }
        return data_loader