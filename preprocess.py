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
    # parser.add_argument("--img_size", type=(int,int), default=(20340, 26591))
    return parser.parse_args()

def convert_nodata(np_img):
    '''
    convert a binary image with no data pts (value=127) to a binary image with no data values being the mean.
    '''
    #print('nodata function >> ')
    nodata = np_img==127
    #print('nodata values: %d' % np.sum(nodata))
    data = np_img!=127
    #print('data values: %d' % np.sum(data))
    #print(np_img.shape)
    mean = np.mean(np_img[data])
    #print('mean >> %f' % mean)
    np_img[nodata] = mean
    #print(np.sum(np_img==mean))
    return np_img

def normalize(np_img):
    #print('normalize function >>')
    mean = np.mean(np_img)
    std = np.std(np_img)
    np_img = (np_img - mean)/std
    #print('mean: %f and std: %f' % (np.mean(np_img), np.std(np_img)))
    return np_img

def zero_one(np_img):
    #print('zero_one function >>')
    ones = np_img==1
    #print('ones: %d' % np.sum(ones))
    zeros = np_img==255
    #print('zeros: %d' % np.sum(zeros))
    np_img[zeros]=0
    #print('check >> %d ==? %d' %(np.sum(ones)+np.sum(zeros), np_img.shape[0]*np_img.shape[1]))
    return np_img

def write(directory_path, data, feature_num, ws=200):
    (h, w) = data.shape
    for i in range(h//ws - 2):
        for j in range(w//ws - 2):
            np.save(directory_path+'/'+str(feature_num)+'_'+str(i)+'_'+str(j)+'.npy', data[i*ws:(i+3)*ws, j*ws:(j+3)*ws])            

def preprocess():
    args = get_args()
    # ws = args.patch_wsize
    # ws = 600 # input_image size should be 600
    # gtws = 200 # output image size should be 200
    fn = args.feature_names.split(", ")
    gtn = args.ground_truth_name
    path = args.img_dir_path
    files = os.listdir(path)

    gt_directory = args.save_to + 'gt_' + str(200)
    if not os.path.exists(gt_directory):
        os.mkdir(gt_directory)
    data_directory = args.save_to + 'data_' + str(600)
    if not os.path.exists(data_directory):
        os.mkdir(data_directory)
    
    # data = []
    cnt = 0
    for feature_name in fn:
        for img_path in files:
            if feature_name in img_path:
                print(feature_name)
                im = Image.open(path+img_path)
                im = np.asarray(im, dtype=np.float16)
                if feature_name=="litho" or feature_name=="landcover":
                    im = convert_nodata(im)
                else: # it's either slope or DEM, they don't have a no-data pt and should be normalized
                    im = normalize(im)
                im = np.pad(im, 200, 'constant') # pads with zeros
                write(data_directory, im, cnt)
                cnt += 1
                # import ipdb; ipdb.set_trace()
                # print("%d feature(s) are loaded ..." % cnt, end="\r")
        print(">> all %s features have been loaded." % feature_name)
    gt = np.asarray(Image.open(path+gtn))
    gt = zero_one(gt)
    (h, w) = gt.shape
    for i in range(h//200):
        for j in range(w//200):
            np.save(gt_directory+'/'+str(i)+'_'+str(j)+'.npy', gt[i*200:(i+1)*200, j*200:(j+1)*200])
            # np.save(data_directory+'/'+str(i)+'_'+str(j)+'.npy', data[:, i*gtws:(i+3)*gtws, j*gtws:(j+3)*gtws])
    print("all images are saved in %s." % args.save_to)
    
    
# if __name__=="__preprocess__":
preprocess()
