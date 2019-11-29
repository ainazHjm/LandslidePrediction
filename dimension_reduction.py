import metric_learn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import ctime

def join_data(args, data_loader):
    length = len(data_loader.dataset)
    path = '/'.join(args.data_path.split('/')[:-1])+'/data_matrix.npy'
    if args.join_data:
        X = np.zeros((length, args.feature_num))
        y = np.zeros((length, 1))
        data_iter = iter(data_loader)
        for iter_ in range(len(data_iter)):
            sample = data_iter.next()
            (b, c, h, w) = sample['data'].shape
            X[iter_*b:(iter_+1)*b, :] = sample['data'][:, :, h//2, w//2].view(-1, c).numpy()
            y[iter_*b:(iter_+1)*b, :] = sample['gt'].view(-1, 1).numpy()
            print(ctime())
        data_mat = np.concatenate((X, y), 1)
        np.save(path, data_mat)
    else:
        data_mat = np.load(path)
        X = data_mat[:, :-1]
        y = data_mat[:, -1]
    
    return X, y

def reduce_dim(args, data_loader):
    save_to = '/'.join(args.data_path.split('/')[:-1])+'/landslide_reduced.npy'

    if args.reduce_dim == 'NCA':
        rdim = metric_learn.NCA(max_iter=10000000, num_dims=2, verbose=True, tol=0.0001)
    else:
        raise ValueError
    
    print('(%s) ---- preparing to join the data ----' %ctime())
    X, y = join_data(args, data_loader)
    print('(%s) ---- data is joined ----' %ctime())
    rdim.fit(X, y)
    print('(%s) ---- model is fit and dimension is successfully reduced ----' %ctime())

    X_new = rdim.transform(X)
    n_datamat = np.concatenate((X_new, y), 1)
    np.save(save_to, n_datamat)
    print('(%s) ---- new features are transformed and saved ----' %ctime())
    np.save(args.save_model_to+'metric.npy', rdim.transformer())
    print('(%s) ---- learned transformer matrix is saved ----' %ctime())
    
    if args.visualize:
        visualize(n_datamat)
    
    return rdim

def visualize(data):
    if data.shape[1] != 3:
        print('!!! something is wrong with the dimension !!!')
        raise ValueError
    fig, ax = plt.subplots()
    scatter = ax.scatter(data[:, 0], data[:, 1], c=data[:, -1])
    legend = ax.legend(*scatter.legend_elements(), loc="upper right", title="Label")
    ax.add_artist(legend)
    plt.show()
    import seaborn as sns
    sns.heatmap(data[:, :-1], cmap="YlGnBu")
