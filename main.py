from train import train
from loader import LandslideDataset, LandslideTrainDataset, create_oversample_data, PixDataset, SampledPixDataset
from torch.utils.data import DataLoader
# from dimension_reduction import reduce_dim
from time import ctime
from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment('CNN_pixelwise')

@ex.config
def ex_cfg():
    train_param = {
        'lr': 0.005,
        'n_epochs': 100,
        'bs': 5,
        'decay': 1e-5,
        'patience': 2,
        'pos_weight': 2,
        'model': 'FCNwBottleneck'
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
        'data_path': '/tmp/landslide_normalized.h5',
        'sample_path': '../image_data/',
        'save': 4
    }

# def my_collate(batch):
#     data = [d['data'] for d in batch]
#     gt = [d['gt'] for d in batch]
#     index = [d['index'] for d in batch]
#     return data, gt, index

@ex.automain
def main(train_param, data_param, loc_param, _log, _run):
    '''
    TODO: 
        SampledPixDataset should be changed if I do the dimensionality reduction first
        and writer the new dataset to be used.
    '''
    data = []
    data.append(
        SampledPixDataset(
            loc_param['data_path'],
            loc_param['sample_path']+'train_ones.npy',
            data_param['region'],
            data_param['pad'],
            data_param['ws'],
            'train'
        )
    )
    data.append(
        LandslideDataset(
            loc_param['data_path'],
            data_param['region'],
            data_param['ws'],
            'test',
            data_param['pad']
        )
    )
    
    loader = [DataLoader(d, batch_size=train_param['bs'], shuffle=True, num_workers=data_param['n_workers']) for d in data]
    
    _log.info('[{}]: created train and test datasets.'.format(ctime()))
    _log.info('[{}]: starting to train ...'.format(ctime()))
    train(loader[0], loader[1], train_param, data_param, loc_param, _log, _run)
    _log.info('[{}]: training is finished!'.format(ctime()))