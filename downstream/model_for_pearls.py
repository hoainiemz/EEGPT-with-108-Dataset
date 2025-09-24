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
    'FP1','FP2','F3','F4','F7','F8',
    'T7','T8','C3','C4','P7','P8',
    'P3','P4','O1','O2','FZ','CZ','PZ',
]

class ParamAttention(nn.Module):
    def __init__(self, d, dropout=0.0):
        super().__init__()
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_k: torch.Tensor, x_q: torch.Tensor) -> torch.Tensor:
        """
        x_k: (b, c, d), x_q: (b, d) -> out: (b, d)
        """
        q = self.Wq(x_q)              # (b, d)
        k = self.Wk(x_k)              # (b, c, d)
        v = self.Wv(x_k)              # (b, c, d)

        scores = torch.einsum('bcd,bd->bc', k, q) / math.sqrt(k.size(-1))  # (b, c)
        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)
        out = torch.einsum('bc,bcd->bd', attn, v)  # (b, d)
        return out

class Model(pl.LightningModule):

    def __init__(self, param):
        super().__init__()    

        if param.foundation_dir[-8:-5] == '108':
            param.patch_size = 200
            param.img_size = [19, 6000]
        else:
            param.patch_size = 64
            param.img_size = [19, 7680]

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

        if param.modality_mode == 'mono':
            if param.foundation_dir[-8:-5] == '108':
                in1 = 512
                out1 = 32
                in2 = 960
            else:
                in1 = 512
                out1 = 8
                in2 = 960

            self.linear_probe1 = LinearWithConstraint(in1, out1, max_norm=1)
            self.linear_probe2 = LinearWithConstraint(in2, 1, max_norm=0.25)
            
            self.drop           = torch.nn.Dropout(p=0.50)
            
            self.arranger = Rearrange('b 1 -> (b 1)')

            self.forward = self.mono_forward
        elif param.modality_mode == 'multi':
            self.meta_backbone = nn.Sequential(
                nn.Linear(28, 128),
                nn.ReLU(),
                nn.Dropout(param.dropout),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Dropout(param.dropout),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(param.dropout),
            )
            self.arranger = Rearrange('b c s d -> b (c s) d')
            self.attention = ParamAttention(d=256, dropout=param.dropout)
            self.classifier = nn.Sequential(
                LinearWithConstraint(256, 256, max_norm=1),
                LinearWithConstraint(256, 1, max_norm=0.25),
                Rearrange('b 1 -> (b 1)')
            )
            self.backbone = nn.Sequential(
                self.chan_conv,
                self.target_encoder,
                LinearWithConstraint(128, 256, max_norm=1),
                LinearWithConstraint(256, 256, max_norm=1),
                LinearWithConstraint(256, 256, max_norm=1),
            )
            self.forward = self.multi_forward
        else:
            raise NotImplementedError
        
    def mono_forward(self, x):

        x = self.chan_conv(x[0]) # [B, C, T]

        # self.target_encoder.eval()
        
        z = self.target_encoder(x, self.chans_id.to(x)) # [B, N_t, S, D]
        
        h = z.flatten(2) # [B, N_t, S D]
        
        h = self.linear_probe1(self.drop(h)) # [B, ]
        
        h = h.flatten(1)
        
        h = self.linear_probe2(h)
        
        return self.arranger(h)
    
    def multi_forward(self, x):
        
        f1 = fork(self.eeg_backbone_attend, x[0])
        f2 = fork(self.meta_backbone, x[1])

        feats, meta_feats = wait(f1), wait(f2)  # feats: b, c, s, d; meta_feats: b, d
        feats = self.attention(feats, meta_feats)  # b, d
        out = self.classifier(feats)  # b, 1
        return out
    
    def eeg_backbone_attend(self, x):
        return self.arranger(self.backbone(x))