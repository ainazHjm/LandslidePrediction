import numpy as np
import torch as th
import matplotlib.pyplot as plt
import sklearn.metrics as sm
from sacred import Experiment
from loader import LandslideDataset
from torch.utils.data import DataLoader
from time import ctime

ex = Experiment('ROC Curve')
colors = {
    'LACNN': 'tab:blue',
    'CNN': 'tab:orange',
    'NN': 'tab:green',
    'LLR': 'tab:red',
    'LANN': 'tab:purple',
    'Naive': 'tab:cyan'
}

@ex.config
def ex_cfg():
    params = {
        'prediction_path': [],
        'data_path': '/tmp/Veneto_data.h5',
        'threshold': [0.01, 0.1, 0.3, 0.5],
        'index_path': '/home/ainaz/Projects/Landslides/image_data/new_partitioning/',
        'pad': 64,
        'prune': 64,
        'ws': 500,
        'region': 'Veneto',
        'n_workers': 0,
        'bs': 1,
        'model': [],
        'save_to': ''
    }

def find_stat(params, loader, prd, _log):
    with th.no_grad():
        loader_iter = iter(loader)
        y, yp = [], []
        for iter_ in range(len(loader_iter)):
            sample = loader_iter.next()
            gt, index = sample['gt'][0, 0,:,:].numpy(), sample['index']
            row, col = int(index[0].numpy()), int(index[1].numpy())
            yhat = prd[
                row*params['ws']:(row+1)*params['ws'],
                col*params['ws']:(col+1)*params['ws']
            ]
            data_indices = gt>=0
            y.extend(gt[data_indices])
            yp.extend(yhat[data_indices])
            _log.info('[{}] getting stats for [{}]/[{}]'.format(ctime(), iter_+1, len(loader_iter)))
        fpr, tpr, _ = sm.roc_curve(y, yp)
        r2_score = sm.r2_score(y, yp)
        return tpr, fpr, r2_score

def plot_curve(params, y, x):
    '''
    y is tpr and x is fpr.
    '''
    for i in range(len(y)): 
        auc = sm.auc(x[i], y[i])
        plt.plot(
            x[i],
            y[i],
            lw=1.5,
            label='%s with AUC = %0.2f' % (params['model'][i], auc),
            color=colors[params['model'][i]])
    
    plt.plot([0,1], [0,1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('AUC and ROC curves')
    plt.legend(loc='lower right')
    plt.savefig(params['save_to'], bbox_inches='tight')
    # plt.show()

@ex.automain
def main(params, _log):
    tpr, fpr = [], []
    for prd_path in params['prediction_path']:
        prd = np.load(prd_path)
        dset = LandslideDataset(
            params['data_path'],
            np.load(params['index_path']+'{}_test_indices.npy'.format(params['region'])),
            params['region'],
            params['ws'],
            params['pad'],
            params['prune']
        )
        loader = DataLoader(dset, num_workers=params['n_workers'], shuffle=False, batch_size=params['bs'])
        _log.info('[{}] created the dataset and the loader.'.format(ctime()))
        _log.info('[{}] finding tpr and fpr'.format(ctime()))
        y, x, _ = find_stat(params, loader, prd, _log)
        tpr.append(y)
        fpr.append(x)
    plot_curve(params, tpr, fpr)
    _log.info('[{}] plot has been saved.'.format(ctime()))
