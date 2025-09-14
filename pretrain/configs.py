
import torchvision
import math
import random
import sys
import os
sys.path.append(os.path.abspath("../"))
from datasets.pretrain.merged import merged_dataset

max_epochs = 200
max_lr = 5e-4
batch_size=32
devices=[0]
dataset_dirs = [
    '/mnt/disk1/aiotlab/namth/EEGFoundationModel/datasets/dummy_data_ica_a7'
]


# train_dataset = torchvision.datasets.DatasetFolder(root="../datasets/pretrain/merged/TrainFolder/", loader=load_fn,  extensions=['.edf'])
# valid_dataset = torchvision.datasets.DatasetFolder(root="../datasets/pretrain/merged/ValidFolder/", loader=load_fn, extensions=['.edf'])

loaders = merged_dataset.LoadDataset(dataset_dirs, batch_size).get_data_loader()
train_loader = loaders['train']
valid_loader = loaders['val']


steps_per_epoch = math.ceil(len(train_loader)/len(devices))

tag = "tiny1"
variant = "D"

MODELS_CONFIGS = {
    "tiny1": {
        "embed_dim":64, "embed_num":1, "depth":[2,2,4], "num_heads":4},
    "tiny2": {
        "embed_dim":64, "embed_num":4, "depth":[2,2,4], "num_heads":4},
    "tiny3": {
        "embed_dim":64, "embed_num":4, "depth":[8,8,8], "num_heads":4},
    "little": {
        "embed_dim":128, "embed_num":4, "depth":[8,8,8], "num_heads":4},
    "base1": {
        "embed_dim":256, "embed_num":1, "depth":[6,6,6], "num_heads":4},
    "base2": {
        "embed_dim":256, "embed_num":4, "depth":[8,8,8], "num_heads":4},
    "base3": {
        "embed_dim":512, "embed_num":1, "depth":[6,6,6], "num_heads":8},
    "large": {
        "embed_dim":512, "embed_num":4, "depth":[8,8,8], "num_heads":8},
}

def get_config(embed_dim=512, embed_num=4, depth=[8,8,8], num_heads=4):
    
    models_configs = {
            'encoder': {
                    'embed_dim': embed_dim,
                    'embed_num': embed_num,
                    'depth': depth[0],
                    'num_heads': num_heads,
                },
            'predictor': {
                    'embed_dim': embed_dim,
                    'embed_num': embed_num,
                    'predictor_embed_dim': embed_dim,
                    'depth': depth[1],
                    'num_heads': num_heads,
                },
            'reconstructor': {
                    'embed_dim': embed_dim,
                    'embed_num': embed_num,
                    'reconstructor_embed_dim': embed_dim,
                    'depth': depth[2],
                    'num_heads': num_heads,
                },
    }
    return models_configs



        