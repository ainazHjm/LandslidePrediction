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

def preprocess():
    args = get_args()
    ws = args.patch_wsize
    fn = args.feature_names.split(", ")
    gtn = args.ground_truth_name
    path = args.img_dir_path

    data = th.zeros(len(fn), args.img_size)
    files = listdir(path)
    for feature_name in fn:
        for img_path in files:
            if feature_name in img_path:
                im = Image.open(img_path)
                transforms.To_Tensor()
    
    
