import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from cbramod_utils.util import to_tensor
import os
import random
import lmdb
import pickle

class CustomDataset(Dataset):
    def __init__(
            self,
            data_dir,
            mode='train',
            num_fold = 0
    ):
        super(CustomDataset, self).__init__()
        self.db = lmdb.open(data_dir, readonly=True, lock=False, readahead=True, meminit=False)
        with self.db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get('__keys__'.encode()))[num_fold][mode]

    def __len__(self):
        return len((self.keys))

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
        data = pair['sample']
        label = pair['label']
        # print(label)
        return (data/100, label)

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch], dtype=np.int64)
        return (to_tensor(x_data).flatten(-2), to_tensor(y_label).long())


class LoadDataset(object):
    def __init__(self, num_fold, params):
        self.params = params
        self.datasets_dir = params.datasets_dir
        self.num_fold = num_fold

    def get_data_loader(self):
        train_set = CustomDataset(self.datasets_dir, mode='train', num_fold=self.num_fold)
        val_set = CustomDataset(self.datasets_dir, mode='val', num_fold=self.num_fold)
        test_set = CustomDataset(self.datasets_dir, mode='test', num_fold=self.num_fold)
        print(len(train_set), len(val_set), len(test_set))
        print(len(train_set)+len(val_set)+len(test_set))
        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                shuffle=True,
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_set.collate,
                shuffle=True,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_set.collate,
                shuffle=True,
            ),
        }
        return data_loader
