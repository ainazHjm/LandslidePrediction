import numpy as np
import argparse
# import torch as th
# from os import listdir
import os
from PIL import Image
# from torchvision import transforms

Image.MAX_IMAGE_PIXELS = 1000000000

def get_args():
    parser = argparse.ArgumentParser(description="Data Preparation")
    parser.add_argument("--img_dir_path", type=str, default="../image_data/Piemonte/")
    parser.add_argument("--save_to", type=str, default="../image_data/data/Piemonte/")
    # parser.add_argument("--patch_wsize", type=int, default=200)
    parser.add_argument("--feature_names", type=str, default="litho, landcover, slope")
    parser.add_argument("--ground_truth_name", type=str, default="polygon_shallow_soil_slide.tif")
    parser.add_argument("--img_format", type=str, default="tif")
    parser.add_argument("--pad", type=int, default=64)
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
    nodata = np_img < 0
    overlimit = np_img > 180
    mean = np.mean(np_img[1-(nodata+overlimit)])
    std = np.std(np_img[1-(nodata+overlimit)])
    np_img = (np_img - mean)/std
    np_img[nodata] = 0
    np_img[overlimit] = 0
    return np_img

def zero_one(np_img):
    # ones = np_img==1
    zeros = np_img==255
    np_img[zeros]=0
    return np_img

def write(directory_path, data, feature_num, pad):
    (h, w) = data.shape
    h, w = h-pad*2, w-pad*2
    for i in range(h//200):
        for j in range(w//200):
            np.save(directory_path+'/'+str(feature_num)+'_'+str(i)+'_'+str(j)+'.npy', data[i*200:(i+1)*200+pad*2, j*200:(j+1)*200+pad*2])

def preprocess():
    args = get_args()
    # ws = args.patch_wsize
    # ws = 600 # input_image size should be 600
    # gtws = 200 # output image size should be 200
    fn = args.feature_names.split(", ")
    # gtn = args.ground_truth_name
    path = args.img_dir_path
    files = os.listdir(path)

    # gt_directory = args.save_to + 'gt_' + str(200)
    # if not os.path.exists(gt_directory):
    #     os.mkdir(gt_directory)
    data_directory = args.save_to + 'data_' + str(args.pad*2)
    if not os.path.exists(data_directory):
        os.mkdir(data_directory)
    
    # data = []
    cnt = 0
    for feature_name in fn:
        for img_path in files:
            if feature_name in img_path:
                # print(feature_name)
                im = Image.open(path+img_path)
                if feature_name=="litho" or feature_name=="landcover":
                    im = np.asarray(im, dtype=np.float16)
                    im = convert_nodata(im)
                else: # it's either slope or DEM, they don't have a no-data pt and should be normalized
                    im = np.array(im)
                    im = normalize(im)
                if args.pad != 0:
                    im = np.pad(im, args.pad, 'constant') # pads with zeros
                write(data_directory, im, cnt, args.pad)
                cnt += 1
                print("%d feature(s) are loaded ..." % cnt, end="\r")
        print(">> all %s features have been loaded." % feature_name)
    # gt = np.array(Image.open(path+gtn))
    # gt = zero_one(gt)
    # (h, w) = gt.shape
    # for i in range(h//200):
    #     for j in range(w//200):
    #         np.save(gt_directory+'/'+str(i)+'_'+str(j)+'.npy', gt[i*200:(i+1)*200, j*200:(j+1)*200])
    print("all images are saved in %s." % args.save_to)
    

preprocess()
