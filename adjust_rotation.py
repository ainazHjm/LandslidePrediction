import h5py
import numpy as np
import scipy.ndimage as snd
from time import ctime
from loader import LandslideDataset
from sacred import Experiment

def rad2deg(theta):
    return (theta*180)/np.pi

def find_angle(pc, pr):
    theta = np.arctan2(abs(pr[0]-pc[0]), abs(pr[1]-pc[1]))
    return rad2deg(theta)

def init_dataset(args, num_samples):
    f = h5py.File(args['save_to'], 'w')
    for flag in ['train', 'test']:
        f.create_dataset(
            args['region']+'/'+flag+'/data',
            (num_samples, args['num_feature'], args['ws']+args['pad']*2, args['ws']+args['pad']*2),
            compression='lzf'
        )
        f.create_dataset(
            args['region']+'/'+flag+'/gt',
            (num_samples, 1, args['ws'], args['ws']),
            compression='lzf'
        )
    _log.info('initialized the dataset.')
    return f

def write_dataset_iter(args, f, sample, angle, idx, data_flag='train'):
    data = snd.rotate(sample['data'], angle, reshape=False) # TODO: check shape & type
    gt = snd.rotate(sample['gt'], angle, reshape=False)
    # import ipdb; ipdb.set_trace()
    f[args['region']][data_flag]['data'][idx, :, :, :] = data
    f[args['region']][data_flag]['gt'][idx, :, :, :] = gt
    return f

@ex.capture
def adjust_rot(args, dataset, data_flag, _log):
    f = init_dataset(args, len(dataset))
    _log.info('initialized the dataset.')
    for idx in range(len(dataset)):
        sample = dataset[idx]
        (h, w) = sample['data'][0, :, :].shape
        hill = sample['data'][0, :, :].max() # the 0th channel is the slope
        tailpt = (sample['data'][0, :, :] == hill).nonzero()[0].data.numpy()
        headpt = np.array([h//2, w//2])
        angle = find_angle(headpt, tailpt)
        _log.info('[{}][{}/{}] ---- angle: {} ---- '.format(ctime(), idx, len(dataset), angle))
        f = write_dataset_iter(args, f, sample, angle, idx, data_flag)
    f.close()
    _log.info('created the rotated dataset for {}',format(data_flag))

def rotate(args):
    train_dataset = LandslideDataset(
        args['data_path'],
        args['region'],
        args['ws'],
        'train',
        args['pad']
    )
    test_dataset = LandslideDataset(
        args['data_path'],
        args['region'],
        args['ws'],
        'test',
        args['pad']
    )
    adjust_rot(args, train_dataset, 'train')
    adjust_rot(args, test_dataset, 'test')

ex = Experiment('rotation_dataset')

@ex.config
def ex_cfg():
    args = {
        'data_path': '/dev/shm/landslide_normalized.h5',
        'region': 'Veneto',
        'ws': 200,
        'pad': 64,
        'num_feature': 94,
        'save_to': '/home/ainaz/Projects/Landslides/image_data/rotated_landslide.h5'
    }

@ex.automain
def main(args):
    rotate(args)
