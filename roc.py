import numpy as np
import torch as th
import matplotlib.pyplot as plt
import sklearn.metrics as sm
from sacred import Experiment
from loader import LandslideDataset
from torch.utils.data import DataLoader
from time import ctime

ex = Experiment('ROC Curve')

@ex.config
def ex_cfg():
    params = {
        'prediction_path': '',
        'data_path': '/tmp/Veneto_data.h5',
        'threshold': [0.01, 0.1, 0.3, 0.5],
        'index_path': '/home/ainaz/Projects/Landslides/image_data/new_partitioning/',
        'pad': 64,
        'prune': 64,
        'ws': 500,
        'region': 'Veneto',
        'n_workers': 0,
        'bs': 1,
        'model': 'Logistic',
    }

def find_stat(params, loader, prd, threshold, _log):
    with th.no_grad():
        loader_iter = iter(loader)
        # tpr, fpr = 0, 0
        # ones, zeros = 0, 0
        y, yp = [], []
        for iter_ in range(len(loader_iter)):
            sample = loader_iter.next()
            gt, index = sample['gt'][0, 0,:,:].numpy(), sample['index']
            row, col = int(index[0].numpy()), int(index[1].numpy())
            # import ipdb; ipdb.set_trace()
            yhat = prd[
                row*params['ws']:(row+1)*params['ws'],
                col*params['ws']:(col+1)*params['ws']
            ]
            data_indices = gt>=0
            y.extend(gt[data_indices])
            yp.extend(yhat[data_indices])
            # one_indices = (yhat>=threshold).nonzero()
            # tpr += np.sum(gt[one_indices])
            # ones += np.sum(gt==1)
            # zero_indices = (yhat<threshold).nonzero()
            # fpr += np.sum(gt[zero_indices]==0)
            # zeros += np.sum(gt==0)
            _log.info('[{}] getting stats for [{}]/[{}]'.format(ctime(), iter_+1, len(loader_iter)))
        # y = np.stack(y, axis=0).reshape(-1)
        # yp = np.stack(yp, axis=0).reshape(-1)
        # import ipdb; ipdb.set_trace()
        fpr, tpr, _ = sm.roc_curve(y, yp)
        r2_score = sm.r2_score(y, yp)
        # tpr, fpr = tpr/ones, fpr/zeros
        # import ipdb; ipdb.set_trace()
        return tpr, fpr, r2_score

def plot_curve(params, y, x, r2_score):
    '''
    y is tpr and x is fpr.
    '''
    auc = sm.auc(x, y)
    plt.plot(x, y, lw=1.5, label='ROC curve with AUC = %0.5f' % auc)
    plt.plot([0,1], [0,1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC cure and AUC for {} model'.format(params['model']))
    plt.legend(loc='lower right')
    plt.show()

@ex.automain
def main(params, _log):
    prd = np.load(params['prediction_path'])
    dset = LandslideDataset(
        params['data_path'],
        params['index_path']+'{}_test_indices.npy'.format(params['region']),
        params['region'],
        params['ws'],
        params['pad'],
        params['prune']
    )
    loader = DataLoader(dset, num_workers=params['n_workers'], shuffle=False, batch_size=params['bs'])
    _log.info('[{}] created the dataset and the loader.'.format(ctime()))
    # y, x = [], []
    # for th in params['threshold']:
    _log.info('[{}] finding tpr and fpr'.format(ctime()))
    y, x, r2_score = find_stat(params, loader, prd, th, _log)
    # y.append(y_)
    # x.append(x_)
    plot_curve(params, y, x, r2_score)
