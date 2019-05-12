import numpy as np
import argparse
# import torch as th
# from os import listdir
import os
import h5py
from PIL import Image
from utils import args
from time import ctime
# from torchvision import transforms

Image.MAX_IMAGE_PIXELS = 1000000000

def get_args():
    parser = argparse.ArgumentParser(description="Data Preparation")
    parser.add_argument("--data_dir", action="append", type=str)
    parser.add_argument("--name", type=str, default="landslide.h5")
    parser.add_argument("--save_to", type=str, default="../image_data/data/Piemonte/")
    parser.add_argument("--feature_num", type=int, default=21)
    parser.add_argument('--shape', action='append', type=args.shape)
    # parser.add_argument("--feature_names", type=str, default="litho, landcover, slope")
    # parser.add_argument("--ground_truth_name", type=str, default="polygon_shallow_soil_slide.tif")
    parser.add_argument("--data_format", type=str, default=".tif")
    parser.add_argument("--pad", type=int, default=64)
    parser.add_argument("--label_pos", nargs='+', type=args.pos)
    # parser.add_argument("--img_size", type=(int,int), default=(20340, 26591))
    return parser.parse_args()

def convert_nodata(np_img):
    '''
    convert a binary image with no data pts (value=127) to a binary image with no data values being the mean.
    '''
    nodata = np_img==127
    data = np_img!=127
    mean = np.mean(np_img[data])
    np_img[nodata] = mean
    return np_img

def normalize(np_img, f = 'slope'):
    # nodata = np_img < 0
    # overlimit = np_img > 180
    if f == 'slope':
        np_img[np_img < 0] = 0
        np_img[np_img > 180] = 0
    mean = np.mean(np_img)
    std = np.std(np_img)
    np_img = (np_img - mean)/std
    return np_img

def zero_one(np_img):
    # ones = np_img==1
    zeros = np_img==255
    np_img[zeros]=0
    return np_img

def oversample(args, directory_path, data, pad):
    print('%s --- oversampling pos images ...' % ctime())
    (h, w) = data.shape
    h, w = h-pad*2, w-pad*2
    lpos = args.label_pos
    cnt = 0
    for e in lpos:
        (l0, l1, u0, u1) = e
        l0, l1, u0, u1 = l0//10, l1//10, (u0//10)+1, (u1//10)+1
        for row in range(l0, u0):
            for col in range(l1, u1):
                if (row+20)*10+pad*2 > data.shape[0] or (col+20)*10+pad*2 > data.shape[1]:
                    print('ignoring this batch')
                    continue
                np.save(
                    directory_path+str(row)+'_'+str(col)+'_10'+'.npy',
                    data[row*10:(row+20)*10+pad*2, col*10:(col+20)*10+pad*2]
                )
                cnt += 1
                #if (row+10)*20+pad*2 > data.shape[0] or (col+10)*20+pad*2 > data.shape[2]:
                #    raise ValueError
    print('%s --- oversampling done: %d.' %(ctime(), cnt))

def write(directory_path, data, feature_num, pad):
    print('%s --- writing images for feature %s.' %(ctime(), str(feature_num)))
    if not os.path.exists(directory_path+str(feature_num)):
        os.mkdir(directory_path+str(feature_num))
    dir_name = directory_path+str(feature_num)+'/'
    (h, w) = data.shape
    h, w = h-pad*2, w-pad*2
    for i in range(h//100-1):
        for j in range(w//100-1):
            np.save(
                dir_name+str(i)+'_'+str(j)+'_100'+'.npy',
                data[i*100:(i+2)*100+pad*2, j*100:(j+2)*100+pad*2]
            )
    print('%s --- wrote images with stride 100: %d.' %(ctime(), (h//100-1)*(w//100-1)))

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

def process_data():
    args = get_args()
    f = h5py.File(args.save_to+args.name, 'a')

    for data_path in args.data_dir:
        name = data_path.split('/')[-2]
        # files = os.listdir(data_path)
        for n, h, w in args.shape:
            if n == name and not name in f.keys():
                f.create_dataset(name+'/data', (args.feature_num, h, w), dtype='f')
                f.create_dataset(name+'/gt', (1, h, w), dtype='f')
                break
    
    for data_path in args.data_dir:
        name = data_path.split('/')[-2]
        # files = os.listdir(data_path)
        for num in range(args.feature_num):
            img = Image.open(data_path+str(num)+args.data_format)
            img = np.asarray(img, dtype=np.float32)
            if num == 0 or num == args.feature_num-1:
                img = normalize(img)
            else:
                print('find_no_data') #TODO
    
    f.close()

process_data()



















