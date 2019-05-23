from torch.utils.data import Dataset
import h5py
import numpy as np
import torch as th

class LandslideTrainDataset(Dataset):
    def __init__(self, path, region, stride, ws, pts, pad=64):
        super(LandslideTrainDataset, self).__init__()
        self.path = path
        self.ws = ws
        self.stride = stride
        self.region = region
        self.pad = pad
        self.pts = pts #nx4 matrix consisting of r1, c1, r2, c2 points, r1<r2 & c1<c2
        self.data_len = self.len()
        self.pts_len = self.len_oversample() #nx1 matrix containing length of the data corresponding to pts

    def identify_idx(self, index):
        cum_sum = 0
        for i in range(self.pts_len.shape[0]):
            cum_sum += self.pts_len[i]
            if index < self.data_len+cum_sum:
                return i
        raise ValueError

    def len_oversample(self):
        stride = self.stride//10
        leng = 0
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
            )
        }
        return sample

    def __getitem__(self, index):
        with h5py.File(self.path, 'r') as f:
            dataset = f[self.region]['train']['data']
            gt = f[self.region]['train']['gt']
            if index < self.data_len:
                sample = self.get_item(index, dataset, gt, self.stride)
            else:
                pts_idx = self.identify_idx(index)
                n_index = index-self.data_len if pts_idx==0 else index-(self.data_len+int(np.sum(self.pts_len[:pts_idx])))
                sample = self.get_item(
                    n_index,
                    dataset[
                        :,
                        self.pts[pts_idx, 0]:self.pts[pts_idx, 2],
                        self.pts[pts_idx, 1]:self.pts[pts_idx, 3]
                    ],
                    gt[
                        :,
                        self.pts[pts_idx, 0]:self.pts[pts_idx, 2],
                        self.pts[pts_idx, 1]:self.pts[pts_idx, 3]
                    ],
                    self.stride//10
                )  
            return sample

class LandslideDataset(Dataset):
    '''
    This class doesn't support different stride sizes and oversampling.
    When testing, we don't need to have stride smaller than ws.
    Also, we don't need to oversample.
    '''
    def __init__(self, path, region, ws, pad=64):
        super(LandslideDataset, self).__init__()
        self.path = path
        self.ws = ws
        self.region = region
        self.pad = pad

    def __len__(self):
        with h5py.File(self.path, 'r') as f:
            (_, h, w) = f[self.region]['test']['gt'].shape
            hnum = h//self.ws
            wnum = w//self.ws
            return hnum*wnum
    
    def __getitem__(self, index):
        with h5py.File(self.path, 'r') as f:
            dataset = f[self.region]['test']['data']
            gt = f[self.region]['test']['gt']
            (_, _, wlen) = gt.shape
            wnum = wlen//self.ws
            row = index//wnum
            col = index - row*wnum
            sample = {
                'data': th.tensor(
                    dataset[
                        :,
                        row*self.ws:(row+1)*self.ws+2*self.pad,
                        col*self.ws:(col+1)*self.ws+2*self.pad
                    ]
                ),
                'gt': th.tensor(
                    gt[
                        :,
                        row*self.ws:(row+1)*self.ws,
                        col*self.ws:(col+1)*self.ws
                    ]
                )
            }
            return sample
        
