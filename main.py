# pylint: disable=E0611
from train import train
# import argparse
# import torch as th
# import numpy as np
import utils.plot as up
# from utils.args import str2bool, __range
from loader import RotDataset, LandslideDataset
from torch.utils.data import DataLoader
# from adjust_rotation import adjust_rot
from time import ctime
from sacred import Experiment

ex = Experiment('train_rotation')

@ex.config
def ex_cfg():
    train_param = {
        'lr': 0.0001,
        'n_epochs': 1,
        'bs': 9,
        'decay': 1e-5,
        'patience': 2,
        'pos_weight': 1,
        'model': 'FCNwPool'
    }
    data_param = {
        'n_workers': 4,
        'region': 'Veneto',
        'pix_res': 10,
        'stride': 200,
        'ws': 200,
        'pad': 64,
        'feature_num': 94,
        'oversample': False
    }
    loc_param = {
        'load_model': '',
        'data_path': '/dev/shm/rotated_dataset.h5',
        # 'save_model_to': '../models/CNN/',
        'save': 5
    }
    # test_param = {
    #     'validate': False,
    #     # 'save_res_to': '../output/CNN/',
    #     'random_sample': False # not sure to include it or not.
    # }

@ex.automain
def main(train_param, data_param, loc_param):
    data = []
    for flag in ['train', 'test']:
        data.append(RotDataset(loc_param['data_path'], data_param['region'], flag))
    loader = [DataLoader(d, batch_size=train_param['bs'], shuffle=True, num_workers=data_param['n_workers']) for d in data]
    
    _log.info('[{}]: created train and test datasets.'.format(ctime()))
    _log.info('[{}]: starting to train ...'.format(ctime()))
    train(loader[0], loader[1])
    _log.info('[{}]: training is finished!'.format(ctime()))