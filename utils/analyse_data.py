import seaborn as sbn
import pandas as pd
import numpy as np
import h5py
import argparse
import csv
import json

def get_args():
    parser = argparse.ArgumentParser(description='visualization')
    parser.add_argument('--data_path', type=str, default='../image_data/landslide_normalized.h5')
    parser.add_argument('--pad', type=int, default=64)
    parser.add_argument('--region', type=str, default='Veneto')
    parser.add_argument('--num_features', type=int, default=94)
    return parser.parse_args()

def create_csv(args, data, gt):
    data_dict = json.load(open('/home/ainaz/Projects/Landslides/CNN/data_dict.json', 'r'))
    path = '/'.join(args.data_path.split('/')[:-1])+'/'
    row_ = []
    for i in range(args.num_features):
        for e in data_dict:
            if int(data_dict[e]) == i:
                row_.append(e)
    row_.append('gt')
    with csv.writer(open(path+'data_csv.csv', 'w')) as f:
        f.writerow(row_)
        for r in data.shape[0]:
            n_row_ = []
            n_row_.extend(data[r, :])
            n_row_.extend(gt[r, :])
            f.writerow(n_row_)
    return path+'data_csv.csv'

def create_dataset(args):
    train_indices = np.load('../image_data/train_data.npy')
    f = h5py.File(args.data_path, 'r')
    d = f[args.region]['train']
    data = np.zeros((train_indices.shape[0], args.num_features))
    gt = np.zeros((train_indices.shape[0], 1))
    for i in range(train_indices.shape[0]):
        idx = train_indices[i, :]
        data[i, :] = d['data'][:, idx[0]+args.pad, idx[1]+args.pad]
        gt[i, :] = d['gt'][:, idx[0], idx[1]]
    
    # train_gt = f[args.region]['train']['gt'][:, :, :]
    # test_gt = f[args.region]['test']['gt'][:, :, :]
    # (h, w) = train_gt.shape[1]+test_gt.shape[1], train_gt.shape

    # test_data = f[args.region]['test']['data'][:, args.pad:-args.pad, args.pad:-args.pad]
    # train_data = f[args.region]['train']['data'][:, args.pad:-args.pad, args.pad:-args.pad]
    
    # data = np.concatenate(
    #     (
    #         np.concatenate(
    #             (train_data[:, 0:h//5, :], test_data),
    #             0
    #         ),
    #         train_data[:, h//5:, :]
    #     ),
    #     0
    # )
    # gt = np.concatenate(
    #     (
    #         np.concatenate(
    #             (train_gt[:, 0:h//5, :], test_gt),
    #             0
    #         ),
    #         train_gt[:, h//5:, :]
    #     ),
    #     0
    # )
    # indices = gt >= 0
    # f_gt = gt[indices].view(-1, 1)
    # f_data = data[indices.expand(train_data.shape[0], h, w)].view(-1, train_data.shape[0])
    path = create_csv(args, data, gt)
    data_table = pd.read_csv(path)
    import ipdb; ipdb.set_trace()

def main():
    args = get_args()
    create_dataset(args)

if __name__ == '__main__':
    main()