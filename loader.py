from torch.utils.data import Dataset
import h5py
import numpy as np
import torch as th

class LargeSample(Dataset):
    def __init__(self, data_path, region, pad, data_flag, div=(5,5)):
        super(LargeSample, self).__init__()
        self.data_path = data_path
        self.region = region
        self.pad = pad
        self.data_flag = data_flag
        (self.row, self.col) = div
    
    def __len__(self):
        return self.row*self.col
    
    def __getitem__(self, index):
        row, col = index//self.col, index-(index//self.col)*self.col
        with h5py.File(self.data_path, 'r') as f:
            (_, h, w) = f[self.region][self.data_flag]['gt'].shape
            h_len, w_len = h//self.row, w//self.col
            sample = {
                'data': f[self.region][self.data_flag]['data'][
                    :,
                    row*h_len:(row+1)*h_len+2*self.pad,
                    col*w_len:(col+1)*w_len+2*self.pad
                    ],
                'gt': f[self.region][self.data_flag]['gt'][
                    0,
                    row*h_len:(row+1)*h_len,
                    col*w_len:(col+1)*w_len
                    ],
                'index': (row, col),
                'div': (self.row, self.col)
            }
            return sample

class LandslideTrainDataset(Dataset):
    def __init__(self, path, region, stride, ws, pts, oversample_path, pad=64, feature_num=94):
        super(LandslideTrainDataset, self).__init__()
        self.path = path
        self.ws = ws
        self.stride = stride
        self.region = region
        self.pad = pad
        self.feature_num = feature_num
        self.pts = pts #nx4 matrix consisting of r1, c1, r2, c2 points, r1<r2 & c1<c2
        self.data_len = self.len()
        self.pts_len = self.len_oversample() #nx1 matrix containing length of the data corresponding to pts
        self.oversample_path = oversample_path

    def identify_idx(self, index):
        cum_sum = 0
        for i in range(self.pts_len.shape[0]):
            cum_sum += self.pts_len[i]
            if index < self.data_len+cum_sum:
                return i
        raise ValueError

    def len_oversample(self):
        stride = self.stride//4
        pts_len = np.zeros((self.pts.shape[0], 1))
        for i in range(self.pts.shape[0]):
            h = self.pts[i,2] - self.pts[i,0]
            w = self.pts[i,3] - self.pts[i,1]
            hnum = (h-self.ws)//stride + 1
            wnum = (w-self.ws)//stride + 1
            pts_len[i] = hnum*wnum
        return pts_len

    def len(self):
        with h5py.File(self.path, 'r') as f:
            (_, h, w) = f[self.region]['train']['gt'].shape
            hnum = (h-self.ws)//self.stride + 1
            wnum = (w-self.ws)//self.stride + 1
            return hnum*wnum

    def __len__(self):
        return self.data_len + int(np.sum(self.pts_len))

    def get_item(self, index, dataset, gt, stride):
        (_, _, wlen) = gt.shape
        wnum = (wlen-self.ws)//stride + 1
        row = index//wnum
        col = index - row*wnum
        sample = {
            'data': th.tensor(
                dataset[
                    :,
                    row*stride:row*stride+self.ws+2*self.pad,
                    col*stride:col*stride+self.ws+2*self.pad
                ]
            ),
            'gt': th.tensor(
                gt[
                    :,
                    row*stride:row*stride+self.ws,
                    col*stride:col*stride+self.ws
                ]
            ),
            'index': (row, col)
        }
        return sample

    def __getitem__(self, index):
        if index < self.data_len:
            with h5py.File(self.path, 'r') as f:
                dataset = f[self.region]['train']['data']
                gt = f[self.region]['train']['gt']
                return self.get_item(index, dataset, gt, self.stride)
        else:
            with h5py.File(self.oversample_path, 'r') as g:
                pts_idx = self.identify_idx(index)
                # r1, c1, r2, c2 = self.pts[pts_idx, :]
                n_index = index-self.data_len if pts_idx==0 else index-(self.data_len+int(np.sum(self.pts_len[:pts_idx])))
                dataset = g[str(pts_idx)]['data']
                gt = g[str(pts_idx)]['gt']
                return self.get_item(n_index, dataset, gt, self.stride//4)

class SampledPixDataset(Dataset):
    def __init__(self, data_path, data_indices_path, region, pad, data_flag):
        super(SampledPixDataset, self).__init__()
        self.data_path = data_path
        self.indices = np.load(data_indices_path) # a nx2 array
        self.region = region
        self.pad = pad
        self.data_flag = data_flag

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row, col = self.indices[index][0], self.indices[index][1]
        with h5py.File(self.data_path, 'r') as f:
            sample = {
                'data': f[self.region][self.data_flag]['data'][:, row-self.pad:row+self.pad+1, col-self.pad:col+self.pad+1],
                'gt': f[self.region][self.data_flag]['gt'][:, row, col],
                'index': (row, col)
            }
            return sample

class PixDataset(Dataset):
    def __init__(self, path, region, data_flag, pad=32):
        super(PixDataset, self).__init__()
        self.path = path
        self.region = region
        self.pad = pad
        self.data_flag = data_flag
        self.shape = self.get_shape()
    
    def get_shape(self):
        with h5py.File(self.path, 'r') as f:
            (_, h, w) = f[self.region][self.data_flag]['gt'].shape
            return (h,w)

    def __len__(self):
        with h5py.File(self.path, 'r') as f:
            (_, h, w) = f[self.region][self.data_flag]['gt'].shape
            return h*w

    def __getitem__(self, index):
        (_, w) = self.shape
        row = index//w
        col = index - (index//w)*w
        with h5py.File(self.path, 'r') as f:
            sample = {
                'data': f[self.region][self.data_flag]['data'][
                    :,
                    row:row+2*self.pad+1,
                    col:col+2*self.pad+1
                ],
                'gt': f[self.region][self.data_flag]['gt'][
                    :,
                    row,
                    col
                ],
                'index': (row, col)
            }
            return sample

class LandslideDataset(Dataset):
    '''
    This class doesn't support different stride sizes and oversampling.
    When testing, we don't need to have stride smaller than ws.
    Also, we don't need to oversample.
    '''
    def __init__(self, data_path, index_path, region, ws, pad, prune):
        super(LandslideDataset, self).__init__()
        self.indices = np.load(index_path) # validation and train indices should be handled in the indices path
        self.data_path = data_path
        self.ws = ws
        self.region = region
        self.pad = pad
        self.prune = prune

    def __len__(self):
        return self.indices.shape[0]
    
    def __getitem__(self, index):
        with h5py.File(self.data_path, 'r') as f:
            entry = self.indices[index, :]
            row, col = int(entry[0]), int(entry[1])
            sample = {
                'data': th.tensor(
                    f[self.region]['data'][
                        :,
                        row*self.ws+self.pad-self.prune:(row+1)*self.ws+self.pad+self.prune,
                        col*self.ws+self.pad-self.prune:(col+1)*self.ws+self.pad+self.prune
                    ]
                ),
                'gt': th.tensor(
                    f[self.region]['gt'][
                        :,
                        row*self.ws:(row+1)*self.ws,
                        col*self.ws:(col+1)*self.ws
                    ]
                ),
                'index': (row, col)
            }
            return sample
        
def initilize_data_oversample(args, fw):
    with h5py.File(args.data_path, 'r') as f:
        dataset = f[args.region]['train']['data']
        gt = f[args.region]['train']['gt']
        for idx in range(args.oversample_pts.shape[0]):
            r1, c1, r2, c2 = args.oversample_pts[idx, :]
            h, w = r2-r1, c2-c1
            fw[str(idx)]['data'][:, 0:args.pad, :] = 0
            fw[str(idx)]['data'][:, h+args.pad:, :] = 0
            fw[str(idx)]['data'][:, :, 0:args.pad] = 0
            fw[str(idx)]['data'][:, :, w+args.pad:] = 0
            fw[str(idx)]['data'][:, args.pad:args.pad+h, args.pad:args.pad+w] = dataset[:, r1:r2, c1:c2]
            fw[str(idx)]['gt'][:][:, :, :] = gt[:, r1:r2, c1:c2]
    return fw

def create_oversample_data(args):
    path = '/'.join(args.data_path.split('/')[:-1])+'/'+args.region+'_oversample.h5'
    # path = '/home/ainaz/Projects/Landslides/image_data/'+args.region+'_oversample.h5'
    if not args.oversample:
        return path
    f = h5py.File(path, 'a')
    for i in range(args.oversample_pts.shape[0]):
        r1, c1, r2, c2 = args.oversample_pts[i, :]
        f.create_dataset(
            str(i)+'/data',
            (args.feature_num, r2-r1+2*args.pad, c2-c1+2*args.pad),
            dtype='f',
            compression='lzf'
        )
        f.create_dataset(
            str(i)+'/gt',
            (1, r2-r1, c2-c1),
            dtype='f',
            compression='lzf'
        )
    f = initilize_data_oversample(args, f)
    f.close()
    return path
    