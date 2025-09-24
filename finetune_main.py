import argparse
import random

import numpy as np
import torch

# import wandb

from datasets.downstream import nfm_dataset, eegtals_dataset
from finetune_trainer import Trainer
from downstream import model_for_nfm, model_for_eegtals


def main():
    parser = argparse.ArgumentParser(description='Big model downstream')
    parser.add_argument('--project_name', type=str, default='', help='project_name')
    parser.add_argument('--seed', type=int, default=3407, help='random seed (default: 0)')
    parser.add_argument('--cuda', type=int, default=0, help='cuda number (default: 1)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='weight decay (default: 1e-2)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer (AdamW, SGD)')
    parser.add_argument('--clip_value', type=float, default=1, help='clip_value')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--classifier', type=str, default='all_patch_reps',
                        help='[all_patch_reps, all_patch_reps_twolayer, '
                             'all_patch_reps_onelayer, avgpooling_patch_reps]')
    # all_patch_reps: use all patch features with a three-layer classifier;
    # all_patch_reps_twolayer: use all patch features with a two-layer classifier;
    # all_patch_reps_onelayer: use all patch features with a one-layer classifier;
    # avgpooling_patch_reps: use average pooling for patch features;

    """############ Downstream dataset settings ############"""
    parser.add_argument('--downstream_dataset', type=str, default='FACED',
                        help='[NFM, UET175, EEGTALS]')
    parser.add_argument('--datasets_dir', type=str,
                        default='/data/datasets/BigDownstream/Faced/processed',
                        help='datasets_dir')
    parser.add_argument('--num_of_classes', type=int, default=9, help='number of classes')
    parser.add_argument('--model_dir', type=str, default='./data/wjq/models_weights/Big/BigFaced', help='model_dir')
    """############ Downstream dataset settings ############"""

    parser.add_argument('--num_workers', type=int, default=16, help='num_workers')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label_smoothing')
    parser.add_argument('--multi_lr', type=bool, default=False,
                        help='multi_lr')  # set different learning rates for different modules
    parser.add_argument('--frozen', type=bool,
                        default=False, help='frozen')
    parser.add_argument('--use_pretrained_weights', type=bool,
                        default=True, help='use_pretrained_weights')
    parser.add_argument('--foundation_dir', type=str,
                        default='pretrained_weights/pretrained_weights.pth',
                        help='foundation_dir')

    params = parser.parse_args()
    
    group = 'Reproduce'

    # wandb.init(project=params.project_name, 
    #            group=group,
    #            name=f'data_{params.downstream_dataset}_seed_{params.seed}',
    #            config=vars(params))
    
    print(params)

    setup_seed(params.seed)
    # torch.cuda.set_device(params.cuda)
    print('The downstream dataset is {}'.format(params.downstream_dataset))
    if params.downstream_dataset == 'NFM':
        load_dataset = nfm_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_nfm.LitEEGPTCausal(params).cuda()
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'EEGTALS':
        load_dataset = eegtals_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_eegtals.Model(params).cuda()
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    print('Done!!!!!')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    # wandb.login(key='ca0cbcfb28dcbf7cd1b54e0a6d40d8a482fc730f')
    main()
    # wandb.finish()
