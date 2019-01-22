import torch as th
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import random

# pylint: disable=E1101
bg_size = (221, 149)

def padding(image):
    (row, col) = image.shape
    padded_img = th.zeros(row+2, col+2)
    padded_img[1:-1, 1:-1] = image
    return padded_img

def divide(data):
    """
    This function divides the dataset into train and validation sets.
    For now, each feature is divided into 9 parts and the validation
    is one of these parts which is selected at random. The rest is the
    training set.
    :data: the dataset in shape of a 3D array: 5x6630x9387.
    """
    div = data.shape[2]/5
    val_idx = np.array(int(random.random()*5))
    np.save(val_idx, "../image_data/data/val_idx.npy")
    
    val_data = data[:, :, val_idx[0]*div:(val_idx[0]+1)*div]
    train_data = th.cat((data[:, :, 0:val_idx[0]*div], data[:, :, (val_idx[0]+1)*div:]), 1)
    th.save(val_data, "../image_data/data/val_data.pt")
    th.save(train_data, "../image_data/data/train_data.pt")
    return val_idx[0], val_data, train_data

def zero_one(image):
    # image = th.from_numpy(image)
    n_img = image
    n_img[image == 255] = 1
    n_img[image != 255] = 0
    return n_img

def process(model, divide, dir_path='../image_data/n_image_data/'):
    """
    This function loads the whole data consisting of 5 different maps:
        the last map is the label map which is converted to a zero-one
        map.
    :model: the name of the CNN model to be trained.
    :dir_path: the path to image data.
    """
    img_names = ['veneto_landcover.tif', 'veneto_lithology.tif', 'veneto_slope.tif',\
                 'veneto_DEM.tif', 'veneto_label.tif']
    img_shape = (7000, 10000)
    data = th.zeros(5, img_shape[0], img_shape[1])

    for i, e in enumerate(img_names):
        img = Image.open(dir_path+e)
        img = transforms.ToTensor(img)
        p_img = padding(img)
        data[i, :, :] = zero_one(p_img) if i == 4 else p_img
    
    if divide:
        val_idx, val_data, train_data = divide(data)
    else:
        val_idx = np.load("../image_data/data/val_idx.npy")
        val_data = th.load("../image_data/data/val_data.pt")
        train_data = th.load("../image_data/data/train_data.pt")
    return val_idx, val_data, train_data