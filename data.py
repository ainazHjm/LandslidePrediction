import torch as th
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import random

# pylint: disable=E1101
def padding(image):
    (row, col) = image.shape
    padded_img = th.zeros(row+2, col+2)
    padded_img[1:-1, 1:-1] = image
    return padded_img

def divide(data):
    """
    This function divides the dataset into train and validation sets.
    For now, each feature is divided into 5 parts and the validation
    is one of these parts(index=1). The rest is the
    training set.
    :data: the dataset in shape of a 3D array: 5xhxw.
    """
    div = data.shape[2]//5
    val_idx = np.array(1)
    np.save("../image_data/data/CNN/val_idx.npy", val_idx)
    
    val_data = data[:, :, val_idx*div:(val_idx+1)*div]
    train_data = th.cat((data[:, :, 0:val_idx*div], data[:, :, (val_idx+1)*div:]), 2)
    th.save(val_data, "../image_data/data/CNN/val_data.pt")
    th.save(train_data, "../image_data/data/CNN/train_data.pt")
    return val_idx, val_data, train_data

def zero_one(image):
    image = image.long()
    image[image == 255] = 1
    image[image == 100] = 0
    return image

def process(dir_path='../image_data/n_image_data/veneto/'):
    """
    This function loads the whole data consisting of 5 different maps:
        the last map is the label map which is converted to a zero-one
        map.
    :model: the name of the CNN model to be trained.
    :dir_path: the path to image data.
    """
    img_names = ['veneto_landcover.tif', 'veneto_lithology.tif', 'veneto_slope.tif',\
                 'veneto_DEM.tif', 'veneto_label.tif']
    # img_shape = (7000, 10000)
    img_shape = (6998, 9998)
    data = th.zeros(5, img_shape[0], img_shape[1])

    for i, e in enumerate(img_names):
        img = Image.open(dir_path+e)
        img = transforms.ToTensor()(img)
        data[i, :, :] = zero_one(img) if i == 4 else img
    label = data[-1, :, :]
    label[data[0, :, :] == -100] = -1
    data[-1, :, :] = label
    return data

def normalize(train_data, val_data):
    (c, _, _) = train_data[:-1, :, :].shape # >>>> we shouldn't normalize the labels
    indices = train_data[0, :, :] != -100
    v_indices = val_data[0, :, :] != -100
    print(indices.shape)
    for i in range(c):
        td = train_data[i, :, :]
        mean = th.mean(td[indices].view(-1))
        std = th.std(td[indices].view(-1))
        train_data[i, :, :][indices] = (td[indices]-mean)/std
        vd = val_data[i, :, :]
        val_data[i, :, :][v_indices] = (vd[v_indices]-mean)/std
        print("train>> before-> mean: %f, std: %f --- now-> mean: %f, std: %f" % (mean.item(), std.item(), th.mean(train_data[i, :, :][indices].view(-1)).item(), th.std(train_data[i, :, :][indices].view(-1)).item()))
        print("validation>> mean: %f, std: %f" % (th.mean(val_data[i, :, :][v_indices].view(-1)).item(), th.std(val_data[i, :, :][v_indices].view(-1)).item()))
    return train_data, val_data
