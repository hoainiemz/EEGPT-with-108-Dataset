import random 
import os
import torch
from torch import nn
import pytorch_lightning as pl
import math
from torch.jit import fork, wait  

from functools import partial
import numpy as np
import random
import os 
import tqdm
from pytorch_lightning import loggers as pl_loggers
import torch.nn.functional as F
def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
seed_torch(7)

from downstream.Modules.models.EEGPT_mcae import EEGTransformer

from einops.layers.torch import Rearrange
from downstream.Modules.Network.utils import Conv1dWithConstraint, LinearWithConstraint
from downstream.utils_eval import get_metrics

use_channels_names = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "O2"
]

class Model(pl.LightningModule):

    def __init__(self, param):
        super().__init__()    

        if param.foundation_dir[-8:-5] == '108':
            param.patch_size = 200
            param.img_size = [19, 800]
        else:
            param.patch_size = 64
            param.img_size = [19, 1024]

        self.chans_num = 19
        # init model
        target_encoder = EEGTransformer(
            img_size=param.img_size,
            patch_size=param.patch_size,
            embed_num=4,
            embed_dim=128,
            depth=8,
            num_heads=8,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
            
        self.target_encoder = target_encoder
        self.chans_id       = target_encoder.prepare_chan_ids(use_channels_names)
        
        # -- load checkpoint
        pretrain_ckpt = torch.load(param.foundation_dir)
        
        target_encoder_stat = {}
        for k,v in pretrain_ckpt['state_dict'].items():
            if k.startswith("target_encoder."):
                target_encoder_stat[k[15:]]=v
                
        self.target_encoder.load_state_dict(target_encoder_stat)
        self.chan_conv       = Conv1dWithConstraint(19, self.chans_num, 1, max_norm=1)

        if param.foundation_dir[-8:-5] == '108':
            in1 = 512
            out1 = 256
            in2 = 1024
        else:
            in1 = 512
            out1 = 32
            in2 = 512

        self.linear_probe1 = LinearWithConstraint(in1, out1, max_norm=1)
        self.linear_probe2 = LinearWithConstraint(in2, param.num_of_classes, max_norm=0.25)
        
        self.drop           = torch.nn.Dropout(p=0.50)
        
    def forward(self, x):

        x = self.chan_conv(x) # [B, C, T]

        # self.target_encoder.eval()
        
        z = self.target_encoder(x, self.chans_id.to(x)) # [B, N_t, S, D]
        
        h = z.flatten(2) # [B, N_t, S D]
        
        h = self.linear_probe1(self.drop(h)) # [B, ]
        
        h = h.flatten(1)
        
        return self.linear_probe2(h)