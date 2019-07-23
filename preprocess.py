import numpy as np
import argparse
# import torch as th
# from os import listdir
import os
import h5py
import json
from PIL import Image
from utils import args
from time import ctime
# from torchvision import transforms

Image.MAX_IMAGE_PIXELS = 1000000000

def get_args():
    parser = argparse.ArgumentParser(description="Data Preparation")
    parser.add_argument("--data_dir", action="append", type=str)
    parser.add_argument("--name", type=str, default="landslide.h5")
    parser.add_argument("--save_to", type=str, default="../image_data/")
    parser.add_argument("--feature_num", type=int, default=94)
    parser.add_argument('--shape', action='append', type=args.shape)
    parser.add_argument("--data_format", type=str, default=".tif")
    parser.add_argument("--pad", type=int, default=64)
    return parser.parse_args()

def convert_nodata(np_img):
    '''
    convert a binary image with no data pts (value=127) to a binary image with no data values being the mean.
    '''
    data = np_img.all()==1 + np_img.all()==0
    nodata = 1-data
    mean = np.mean(np_img[data])
    np_img[nodata] = mean
    # import ipdb; ipdb.set_trace()
    return np_img

def normalize(np_img, f = 'slope'):
    if f == 'slope':
        np_img[np_img < 0] = 0
        np_img[np_img > 180] = 0
    mean = np.mean(np_img)
    std = np.std(np_img)
    print('mean, std before normalizing: %f, %f' %(mean, std))
    np_img = (np_img - mean)/std
    print('after: %f, %f' %(np.mean(np_img), np.std(np_img)))
    return np_img

def zero_one(np_img):
    # ones = np_img==1
    ones = np_img!=0
    np_img[ones]=1
    return np_img

def initialize(f, key):
    (n, h, w) = f[key]['train/data'].shape
    (_, hg, wg) = f[key]['test/data'].shape
    zero_train = np.zeros((h, w))
    zero_test = np.zeros((hg, wg))
    for i in range(n):
        f[key]['train/data'][i] = zero_train
        f[key]['test/data'][i] = zero_test
        print('%s -> %d/%d' %(ctime(), i+1, n), end='\r')
    return f

def process_data():
    args = get_args()
    g = open('data_dict.json', 'r')
    data_dict = json.load(g)
    g.close()
    f = h5py.File(args.save_to+args.name, 'a')
    
    for data_path in args.data_dir:
        name = data_path.split('/')[-2]
        for n, h, w in args.shape:
            # hv = h//5
            nw = w//3
            if n == name and not name in f.keys():
                f.create_dataset(
                    name+'/test/data',
                    (args.feature_num, h+args.pad*2, w-2*nw+args.pad*2),
                    dtype='f',
                    compression='lzf'
                )
                f.create_dataset(name+'/test/gt', (1, h+args.pad*2, w-2*nw+args.pad*2), dtype='f', compression='lzf')
                f.create_dataset(
                    name+'/train/data',
                    (args.feature_num, h+args.pad*2, nw*2+args.pad*2),
                    dtype='f',
                    compression='lzf'
                )
                f.create_dataset(name+'/train/gt', (1, h+args.pad*2, nw*2+args.pad*2), dtype='f', compression='lzf')
                print('created data and gt in %s' %name)
                break
        f = initialize(f, name)

    print(list(f.keys()))
    for data_path in args.data_dir:
        name = data_path.split('/')[-2]
        images = os.listdir(data_path)
        for img in images:
            if args.data_format in img and not '.xml' in img and not 'gt' in img:
                t = np.array(Image.open(data_path+img))
                n_ = img.split('.')[0]
                if int(data_dict[n_]) == 0:
                    t = normalize(t, 'slope')
                elif int(data_dict[n_]) == args.feature_num-1:
                    t = normalize(t, 'DEM')
                print(data_dict[n_])
                wlen = t.shape[1]//3
                f[name+'/train/data'][int(data_dict[n_])] = np.pad(t[:, 0:2*wlen], args.pad, 'edge')
                f[name+'/test/data'][int(data_dict[n_])] = np.pad(t[:, 2*wlen:], args.pad, 'edge')
        gt = np.array(Image.open(data_path+'gt'+args.data_format))
        wlen = gt.shape[1]//3
        f[name+'/train/gt'][0] = np.pad(gt[:, 0:2*wlen], args.pad, 'edge')
        f[name+'/test/gt'][0] = np.pad(gt[:, 2*wlen:], args.pad, 'edge')

    f.close()

process_data()