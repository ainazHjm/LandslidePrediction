import h5py
import numpy as np
import scipy.ndimage as snd
from time import ctime
from loader import LandslideDataset
from sacred import Experiment

ex = Experiment('rotation_dataset')

@ex.config
def ex_cfg():
    args = {
        'data_path': '/dev/shm/landslide_normalized.h5',
        'region': 'Veneto',
        'ws': 400,
        'pad': 64,
        'num_feature': 94,
        'save_to': '/home/ainaz/Projects/Landslides/image_data/rotated_landslide_new.h5'
    }

def rad2deg(theta):
    return (theta*180)/np.pi

def find_angle(pc, pr):
    theta = np.arctan2(abs(pr[0]-pc[0]), abs(pr[1]-pc[1]))
    return 90-int(rad2deg(theta)) # only return the integer values

def init_dataset(args, num_samples):
    f = h5py.File(args['save_to'], 'w')
    for idx, flag in enumerate(['train', 'test']):
        f.create_dataset(
            args['region']+'/'+flag+'/data',
            (num_samples[idx], args['num_feature'], args['ws']+args['pad']*2, args['ws']+args['pad']*2),
            compression='lzf'
        )
        f.create_dataset(
            args['region']+'/'+flag+'/gt',
            (num_samples[idx], 1, args['ws'], args['ws']),
            compression='lzf'
        )
        f.create_dataset(
            args['region']+'/'+flag+'/index',
            (num_samples[idx], 2),
            compression='lzf'
        )
        f.create_dataset(
            args['region']+'/'+flag+'/angle',
            (num_samples[idx], 1),
            compression='lzf'
        )
    return f

def write_dataset_iter(args, f, sample, angle, idx, data_flag):
    gt = sample['gt'][0, :, :]
    rot_gt = snd.rotate(gt, angle, reshape=False)
    f[args['region']][data_flag]['gt'][idx, 0, :, :] = rot_gt
    f[args['region']][data_flag]['index'][idx, :] = sample['index']
    f[args['region']][data_flag]['angle'][idx, :] = angle
    for channel in range(sample['data'].shape[0]): # rotate for each channel
        data = sample['data'][channel, :, :]
        rot_data = snd.rotate(data, angle, reshape=False)
        f[args['region']][data_flag]['data'][idx, channel, :, :] = rot_data
    return f

@ex.capture
def adjust_rot(args, f, dataset, data_flag, _log, _run):
    for idx in range(len(dataset)):
        sample = dataset[idx]
        (h, w) = sample['data'][0, :, :].shape
        hill = sample['data'][0, :, :].max() # the 0th channel is the slope
        tailpt = (sample['data'][0, :, :] == hill).nonzero()[0].data.numpy()
        headpt = np.array([h//2, w//2])
        angle = find_angle(headpt, tailpt)
        _log.info('[{}][{}/{}] ---- angle: {} ---- '.format(ctime(), idx, len(dataset), angle))
        _run.log_scalar('rotation_{}.angle'.format(data_flag), angle, idx)
        f = write_dataset_iter(args, f, sample, angle, idx, data_flag)
    _log.info('created the rotated dataset for {}'.format(data_flag))
    return f

def rotate(args, _log):
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
    f = init_dataset(args, [len(train_dataset), len(test_dataset)])
    # import ipdb; ipdb.set_trace()
    _log.info('[{}] initialized the dataset.'.format(ctime()))
    f = adjust_rot(args, f, train_dataset, 'train')
    f = adjust_rot(args, f, test_dataset, 'test')
    f.close()

@ex.automain
def main(args, _log):
    rotate(args, _log)
