import numpy as np
import argparse
import torch as th
from os import listdir
from PIL import Image
from torchvision import transforms

def get_args():
    parser = argparse.ArgumentParser(description="Data Preparation")
    parser.add_argument("--img_dir_path", type=str, default="../image_data/Piemonte/")
    parser.add_argument("--save_to", type=str, default="../image_data/Piemonte_patches")
    parser.add_argument("--patch_wsize", type=int, default=200)
    parser.add_argument("--feature_names", type=str, default="litho, landcover, slope, DEM")
    parser.add_argument("--ground_truth_name", type=str, default="polygon_shallow_soil_slide")
    parser.add_argument("--img_format", type=str, default="tif")
    parser.add_argument("--img_size", type=(int,int), default=(20340, 26591))
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

def normalize(np_img):
    mean = np.mean(np_img)
    std = np.std(np_img)
    np_img = (np_img - mean)/std
    return np_img

def zero_one(np_img):
    

def preprocess():
    args = get_args()
    ws = args.patch_wsize
    fn = args.feature_names.split(", ")
    gtn = args.ground_truth_name
    path = args.img_dir_path
    files = listdir(path)
    data = []
    
    for feature_name in fn:
        for img_path in files:
            if feature_name in img_path:
                im = Image.open(path+img_path)
                im = np.array(im)
                if feature_name=="litho" or feature_name=="landcover":
                    im = convert_nodata(im)
                else: # it's either slope or DEM, they don't have a no-data pt and should be normalized
                    im = normalize(im)
                data.append(th.from_numpy(im))
    gt = np.array(Image.open(path+gtn))

    
    
