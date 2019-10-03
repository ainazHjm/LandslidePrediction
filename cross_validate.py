import numpy as np
from train import train
from loader import LandslideDataset, DistLandslideDataset
from torch.utils.data import DataLoader
from time import ctime
from sacred import Experiment

ex = Experiment('Cross_Validation')

train_param = {
    'optim': 'Adam',
    'lr': 0.0001,
    'n_epochs': 1,
    'bs': 15,
    'decay': 1e-3,
    'patience': 2,
    'pos_weight': 1,
    'model': 'Logistic'
}

@ex.config
def ex_cfg():
    data_param = {
    'cross_validate': True,
    'n_workers': 2,
    'region': 'Veneto',
    'pix_res': 10,
    'stride': 500,
    'ws': 500,
    'pad': 64,
    'feature_num': 94,
    'prune': 64,
    'dist_num': 3, #corresponding to 30,100,300
    'dist_feature': False
    }
    loc_param = {
        'load_model': '',
        'data_path': '/tmp/Veneto_data.h5',
        'index_path': '/home/ainaz/Projects/Landslides/image_data/new_partitioning/',
        'save': 20
    }

def plot_grid(x, y):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.add_subplot(1,1,1)
    plt.scatter(x, y['Adam'], c='b')
    plt.scatter(x, y['SGD'], c='r')
    plt.show()

def get_loader(data_indices, train_param, data_param, loc_param, k_index):
    n = data_indices.shape[0]
    val = data_indices[k_index*(n//5):(k_index+1)*(n//5), :]
    if n % 5 == 0 and k_index == 4:
        train = data_indices[0:k_index*(n//5), :]
    else:
        train = np.concatenate(
            (
                data_indices[0:k_index*(n//5), :],
                data_indices[(k_index+1)*(n//5):, :]
            ),
            axis=0
        )
    # import ipdb; ipdb.set_trace()
    data = []
    for flag in ['train', 'validation']:
        data.append(
            LandslideDataset(
                loc_param['data_path'],
                train if flag=='train' else val,
                data_param['region'],
                data_param['ws'],
                data_param['pad'],
                data_param['prune']
            )
        )
    loader = [DataLoader(d, batch_size=train_param['bs'], shuffle=True, num_workers=data_param['n_workers']) for d in data]
    return loader

def helper(train_param, data_param, loc_param, _log, _run):
    print(train_param['lr'], train_param['optim'])
    data_indices = np.load(loc_param['index_path']+data_param['region']+'_data_indices.npy')
    test_indices = np.load(loc_param['index_path']+data_param['region']+'_test_indices.npy')
    for i in range(test_indices.shape[0]):
        for j in range(data_indices.shape[0]):
            index = test_indices[i, :]
            if data_indices[j, 0]==index[0] and data_indices[j, 1]==index[1]:
                data_indices = np.delete(data_indices, j, 0)
                break
    k_fold_loss = 0
    for k in range(5):
        loader = get_loader(data_indices, train_param, data_param, loc_param, k)
        k_fold_loss += train(loader[0], loader[1], train_param, data_param, loc_param, _log, _run)
        print(k)
    _log.info('[%s] average k-fold loss: %.4f' %(ctime(), k_fold_loss/5))
    _log.info('--- lr: %.5f and optimizer: %s ---' %(train_param['lr'], train_param['optim']))
    return k_fold_loss/5        

@ex.automain
def cross_validate(data_param, loc_param, _log, _run):
    best_lr, best_optim = -1, 'Adam'
    min_error = 1e5
    for optim in ['Adam', 'SGD']:
        for lr in range(-15, 0):
            train_param['lr'] = 2**lr
            train_param['optim'] = optim
            error = helper(
                train_param,
                data_param,
                loc_param,
                _log,
                _run
            )
            if error < min_error:
                min_error = error
                best_lr = 2**lr
                best_optim = optim
    print('** best lr: %.4f, best optim: %s' %(best_lr, best_optim))