from torch.utils.data import Dataset
import h5py
import torch as th

class LandslideDataset(Dataset):
    def __init__(self, path, region, stride, ws, data='train', pad=64):
        super(LandslideDataset, self).__init__()
        self.path = path
        self.ws = ws
        self.stride = stride
        self.region = region
        self.data_flag = data
        self.pad = pad

    def __len__(self):
        with h5py.File(self.path, 'r') as f:
            (_, h, w) = f[self.region][self.data_flag]['gt'].shape
            hnum = (h-self.ws)//self.stride + 1
            wnum = (w-self.ws)//self.stride + 1
            return hnum*wnum

    def __getitem__(self, index):
        with h5py.File(self.path, 'r') as f:
            dataset = f[self.region][self.data_flag]['data']
            gt = f[self.region][self.data_flag]['gt']
            (_, _, wlen) = gt.shape
            wnum = (wlen-self.ws)//self.stride + 1
            row = index//wnum
            col = index - row*wnum
            sample = {
                'data': th.tensor(
                    dataset[
                        :,
                        row*self.stride:row*self.stride+self.ws+2*self.pad,
                        col*self.stride:col*self.stride+self.ws+2*self.pad
                    ]
                ),
                'gt': th.tensor(
                    gt[
                        :,
                        row*self.stride:row*self.stride+self.ws,
                        col*self.stride:col*self.stride+self.ws
                        ]
                )
            }
            return sample
