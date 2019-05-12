from torch.utils.data import Dataset, DataLoader
import h5py

class LandslideDataset(Dataset):
    def __init__(self, path, key, stride, ws):
        super(LandslideDataset, self).__init__()
        self.f = h5py.File(path, 'r')
        self.ws = ws
        self.stride = stride
        self.key = key

    def __len__(self):
        (_, h, w) = self.f[self.key]['data'].shape
        hnum = (h-self.ws)//self.stride + 1
        wnum = (w-self.ws)//self.stride + 1
        return hnum*wnum

    def __getitem__(self, index):
        dataset = self.f[self.key]['data']
        gt = self.f[self.key]['gt']
        sample = {
            'data': dataset[
                :,
                index*self.stride:index*self.stride+self.ws,
                index*self.stride+index*self.stride+self.ws
            ],
            'gt': gt[
                :,
                index*self.stride:index*self.stride+self.ws,
                index*self.stride+index*self.stride+self.ws
            ]
        }
        return sample
        
