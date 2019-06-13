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
    # parser.add_argument("--feature_names", type=str, default="litho, landcover, slope")
    # parser.add_argument("--ground_truth_name", type=str, default="polygon_shallow_soil_slide.tif")
    parser.add_argument("--data_format", type=str, default=".tif")
    parser.add_argument("--pad", type=int, default=64)
    # parser.add_argument("--label_pos", nargs='+', type=args.pos)
    # parser.add_argument("--img_size", type=(int,int), default=(20340, 26591))
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
    np_img = (np_img - mean)/std
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
            hv = h//5
            if n == name and not name in f.keys():
                f.create_dataset(
                    name+'/test/data',
                    (args.feature_num, hv+args.pad*2, w+args.pad*2),
                    dtype='f',
                    compression='lzf'
                )
                f.create_dataset(name+'/test/gt', (1, hv, w), dtype='f', compression='lzf')
                f.create_dataset(
                    name+'/train/data',
                    (args.feature_num, h-hv+args.pad*2, w+args.pad*2),
                    dtype='f',
                    compression='lzf'
                )
                f.create_dataset(name+'/train/gt', (1, h-hv, w), dtype='f', compression='lzf')
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
                if data_dict[n_] == 0:
                    t = normalize(t, 'slope')
                elif data_dict[n_] == args.feature_num-1:
                    t = normalize(t, 'DEM')
                # else:
                #     # t = convert_nodata(zero_one(t))
                #     t = zero_one(t)
                print(data_dict[n_])
                hlen = t.shape[0]//5
                f[name+'/train/data'][int(data_dict[n_])] = np.pad(np.concatenate((t[0:hlen, :], t[2*hlen:, :]), 0), args.pad, 'constant')
                f[name+'/test/data'][int(data_dict[n_])] = np.pad(t[hlen:2*hlen, :], args.pad, 'constant')
        gt = np.array(Image.open(data_path+'gt'+args.data_format))
        hlen = gt.shape[0]//5
        f[name+'/train/gt'][0] = np.concatenate((gt[0:hlen, :], gt[2*hlen:, :]), 0)
        f[name+'/test/gt'][0] = gt[hlen:2*hlen, :]

    f.close()

process_data()

# def oversample(args, directory_path, data, pad):
#     print('%s --- oversampling pos images ...' % ctime())
#     (h, w) = data.shape
#     h, w = h-pad*2, w-pad*2
#     lpos = args.label_pos
#     cnt = 0
#     for e in lpos:
#         (l0, l1, u0, u1) = e
#         l0, l1, u0, u1 = l0//10, l1//10, (u0//10)+1, (u1//10)+1
#         for row in range(l0, u0):
#             for col in range(l1, u1):
#                 if (row+20)*10+pad*2 > data.shape[0] or (col+20)*10+pad*2 > data.shape[1]:
#                     print('ignoring this batch')
#                     continue
#                 np.save(
#                     directory_path+str(row)+'_'+str(col)+'_10'+'.npy',
#                     data[row*10:(row+20)*10+pad*2, col*10:(col+20)*10+pad*2]
#                 )
#                 cnt += 1
#                 #if (row+10)*20+pad*2 > data.shape[0] or (col+10)*20+pad*2 > data.shape[2]:
#                 #    raise ValueError
#     print('%s --- oversampling done: %d.' %(ctime(), cnt))

# def write(directory_path, data, feature_num, pad):
#     print('%s --- writing images for feature %s.' %(ctime(), str(feature_num)))
#     if not os.path.exists(directory_path+str(feature_num)):
#         os.mkdir(directory_path+str(feature_num))
#     dir_name = directory_path+str(feature_num)+'/'
#     (h, w) = data.shape
#     h, w = h-pad*2, w-pad*2
#     for i in range(h//100-1):
#         for j in range(w//100-1):
#             np.save(
#                 dir_name+str(i)+'_'+str(j)+'_100'+'.npy',
#                 data[i*100:(i+2)*100+pad*2, j*100:(j+2)*100+pad*2]
#             )
#     print('%s --- wrote images with stride 100: %d.' %(ctime(), (h//100-1)*(w//100-1)))

# def preprocess():
#     args = get_args()
#     fn = args.feature_names.split(", ")
#     gtn = args.ground_truth_name
#     path = args.img_dir_path
#     files = os.listdir(path)

#     data_directory = args.save_to + 'data_' + str(args.pad*2)+'/'
#     if not os.path.exists(data_directory):
#         os.mkdir(data_directory)
#     cnt = 0
#     for feature_name in fn:
#         for img_path in files:
#             if feature_name in img_path:
#                 im = Image.open(path+img_path)
#                 if feature_name=="litho" or feature_name=="landcover":
#                     im = np.asarray(im, dtype=np.float16)
#                     im = convert_nodata(im)
#                 else: # it's either slope or DEM, they don't have a no-data pt and should be normalized
#                     im = np.array(im)
#                     im = normalize(im)
#                 if args.pad != 0:
#                     im = np.pad(im, args.pad, 'constant') # pads with zeros
#                 write(data_directory, im, cnt, args.pad)
#                 if args.label_pos:
#                     oversample(args, data_directory+str(cnt)+'/', im, args.pad)
#                 cnt += 1
#                 print("%d feature(s) are loaded ..." % cnt, end="\r")
#         print(">> all %s features have been loaded." % feature_name)
#     gt = np.array(Image.open(path+gtn))
#     gt = zero_one(gt)
#     write(data_directory, gt, 'gt', 0)
#     oversample(args, data_directory+'gt/', gt, 0)
#     print("all images are saved in %s." % args.save_to)

# preprocess()