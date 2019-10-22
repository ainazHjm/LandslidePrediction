import torch as th
import torch.nn as nn
import numpy as np
import model
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from loader import LandslideDataset, DistLandslideDataset
from time import ctime
from sacred import Experiment
from unet import UNet

ex = Experiment('validate_CNNpatch')

def validate(params, data_loader, _log, flag):
    with th.no_grad():
        sig = nn.Sigmoid()
        criterion = nn.BCEWithLogitsLoss()
        running_loss = 0
        shape = params['shape']
        res = th.zeros(shape)
        prune = params['prune']

        if params['model'] == 'FCNwBottleneck':
           trained_model = model.FCNwBottleneck(params['feature_num'], params['pix_res'])
        elif params['model'] == 'UNet':
            trained_model = UNet(params['feature_num'], 1)
        elif params['model'] == 'SimplerFCNwBottleneck':
            trained_model = model.SimplerFCNwBottleneck(params['feature_num'])
        elif params['model'] == 'Logistic':
            trained_model = model.Logistic(params['feature_num'])
        elif params['model'] == 'PolyLogistic':
            trained_model = model.PolyLogistic(params['feature_num'])
        
        trained_model = trained_model.cuda()
        if th.cuda.device_count() > 1:
            trained_model = nn.DataParallel(trained_model)
        trained_model.load_state_dict(th.load(params['load_model']))
        _log.info('[{}] model is successfully loaded.'.format(ctime()))

        data_iter = iter(data_loader)
        for iter_ in range(len(data_iter)):
            sample = data_iter.next()
            data, gt = sample['data'].cuda(), sample['gt'].cuda()
            ignore = gt < 0
            prds = trained_model.forward(data)[:, :, prune:-prune, prune:-prune]
            loss = criterion(prds[1-ignore], gt[1-ignore])
            running_loss += loss.item()
            
            prds = sig(prds)
            prds[ignore] = 0

            for idx in range(prds.shape[0]):
                row, col = sample['index'][0][idx], sample['index'][1][idx]
                res[
                    row*params['ws']:(row+1)*params['ws'],
                    col*params['ws']:(col+1)*params['ws']
                ] = prds[idx, 0, :, :]
            _log.info('[{}]: writing [{}/{}]'.format(ctime(), iter_, len(data_iter)))
        _log.info('all image patches are written!')
        save_image(res, '{}{}_{}_predictions.tif'.format(params['save_to'], params['region'], flag))
        np.save('{}{}_{}_predictions.npy'.format(params['save_to'],params['region'], flag), res.data.numpy())
        return running_loss/len(data_iter)

@ex.config
def ex_cfg():
    params = {
        'data_path': '/tmp/Veneto_data.h5',
        'index_path': '/home/ainaz/Projects/Landslides/image_data/new_partitioning/',
        'load_model': '',
        'save_to': '',
        'region': 'Veneto',
        'ws': 500,
        'pad': 64,
        'prune': 64,
        'shape': (21005, 19500), # final shape of the image
        'bs': 4,
        'n_workers': 2,
        'model': 'FCNwBottleneck',
        'feature_num': 94,
        'pix_res': 10,
        'write_image': True,
        'dist_feature': False,
        'dist_num': 3,
    }

@ex.automain
def main(params, _log):
    data = []
    for flag in ['test', 'train']:
        if params['dist_feature']:
            data.append(
                DistLandslideDataset(
                    params['data_path'],
                    np.load(params['index_path']+'{}_{}_indices.npy'.format(params['region'], flag)),
                    params['region'],
                    params['ws'],
                    params['pad'],
                    params['prune'],
                    params['dist_num']
                )
            )
        else:
            data.append(
                LandslideDataset(
                    params['data_path'],
                    np.load(params['index_path']+'{}_{}_indices.npy'.format(params['region'], flag)),
                    params['region'],
                    params['ws'],
                    params['pad'],
                    params['prune']
                )
            )
    test_loader = DataLoader(data[0], batch_size=params['bs'], shuffle=False, num_workers=params['n_workers'])
    _log.info('[{}] prepared the dataset and the data loader for test.'.format(ctime()))
    test_loss = validate(params, test_loader, _log, 'test')
    _log.info('[{}] average loss on test set is {}'.format(ctime(), test_loss))
    train_loader = DataLoader(data[1], batch_size=params['bs'], shuffle=False, num_workers=params['n_workers'])
    _log.info('[{}] prepared the dataset and the data loader for train.'.format(ctime()))
    train_loss = validate(params, train_loader, _log, 'train')
    _log.info('[{}] average loss on train set is {}'.format(ctime(), train_loss))

    if params['write_image']:
        if params['dist_feature']:
            dataset = DistLandslideDataset(
                params['data_path'],
                np.load(params['index_path']+'{}_data_indices.npy'.format(params['region'])),
                params['region'],
                params['ws'],
                params['pad'],
                params['prune'],
                params['dist_num']
            )
        else:
            dataset = LandslideDataset(
                params['data_path'],
                np.load(params['index_path']+'{}_data_indices.npy'.format(params['region'])),
                params['region'],
                params['ws'],
                params['pad'],
                params['prune']
            )
        data_loader = DataLoader(dataset, batch_size=params['bs'], num_workers=params['n_workers'])
        validate(params, data_loader, _log, 'data')
