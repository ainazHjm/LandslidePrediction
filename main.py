from train import train
from loader import LandslideDataset, SampledPixDataset, LargeSample
from torch.utils.data import DataLoader
# from dimension_reduction import reduce_dim
from time import ctime
from sacred import Experiment
# from sacred.observers import MongoObserver

ex = Experiment('CNNPatch')

@ex.config
def ex_cfg():
    train_param = {
        'optim': 'SGD',
        'lr': 0.0001,
        'n_epochs': 100,
        'bs': 4,
        'decay': 1e-5,
        'patience': 2,
        'pos_weight': 1,
        'model': 'UNet'
    }
    data_param = {
        'grid_search': False,
        # 'div': {'train': (20,20), 'test': (4,20)},
        'n_workers': 4,
        'region': 'Veneto',
        'pix_res': 10,
        'stride': 500,
        'ws': 500,
        'pad': 64,
        'feature_num': 94,
        'oversample': False,
        'prune': 64
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

@ex.capture
def grid_search(loader, train_param, data_param, loc_param, _log, _run):
    n_train_param = train_param
    n_train_param['n_epochs'] = 1
    min_loss = +100
    best_lr, best_optim = -1, None
    x = list(range(-10, 0))
    y = {'Adam': [], 'SGD': []}
    for optim in ['Adam', 'SGD']:
        for lr in range(-10, 0):
            n_train_param['lr'] = 2**lr
            n_train_param['optim'] = optim
            loss_ = train(loader[0], loader[1], n_train_param, data_param, loc_param, _log, _run)
            y[optim].extend(loss_)
            if loss_ < min_loss:
                min_loss = loss_
                best_lr = 2**lr
                best_optim = optim
    plot_grid(x, y)
    return best_lr, best_optim

@ex.automain
def main(train_param, data_param, loc_param, _log, _run):
    data = []
    for flag in ['train', 'validation']:
        data.append(
            LandslideDataset(
                loc_param['data_path'],
                loc_param['index_path']+'{}_{}_indices.npy'.format(data_param['region'], flag),
                data_param['region'],
                data_param['ws'],
                data_param['pad'],
                data_param['prune']
            )
        )
    loader = [DataLoader(d, batch_size=train_param['bs'], shuffle=True, num_workers=data_param['n_workers']) for d in data]
    # if data_param['grid_search']:
    #     lr, optim = grid_search(loader, train_param, data_param, loc_param)
    #     train_param['lr'] = lr
    #     train_param['optim'] = optim
    
    _log.info('[{}]: created train and validation datasets.'.format(ctime()))
    _log.info('[{}]: starting to train ...'.format(ctime()))
    train(loader[0], loader[1], train_param, data_param, loc_param, _log, _run)
    _log.info('[{}]: training is finished!'.format(ctime()))
