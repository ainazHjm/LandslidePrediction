import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Data Preparation")
    parser.add_argument("--img_dir_path", type=str, default="../image_data/Piemonte/")
    parser.add_argument("--save_to", type=str, default="../image_data/Piemonte_patches")
    parser.add_argument("--patch_wsize", type=int, default=200)
    parser.add_argument("--feature_names", type=str, default="litho, landcover, slope, DEM")
    parser.add_argument("--ground_truth_name", type=str, default="polygon_shallow_soil_slide")
    parser.add_argument("--img_format", type=str, default="tif")
    return parser.parse_args()

def preprocess():
    args = get_args()
    
