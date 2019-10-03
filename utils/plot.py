# pylint: disable=E1101
import torch as th
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import h5py
from torch.nn import Sigmoid
from time import ctime
from PIL import Image
from torchvision.utils import save_image

def unite_imgs(data_path, orig_shape, ws):
    (h, w) = orig_shape
    img_names = os.listdir(data_path)
    names = [e for e in img_names if '.npy' in e]
    big_img = np.zeros((h, w))

    for name in names:
        r, c = name.split('.')[0].split('_')
        r, c = int(r), int(c)
        big_img[r*ws:(r+1)*ws, c*ws:(c+1)*ws] = np.load(data_path+name)
    
    dir_name = data_path+'whole'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    np.save(dir_name+'/prediction.npy', big_img)

def magnify(img_path = "../image_data/veneto_new_version/n_label.tif"):
    im = Image.open(img_path)
    im = np.array(im)
    im[im == 100] = 0
    im[im == 255] = 1
    indices = np.where(im == 1)
    for i in range(len(indices[0])):
        r = indices[0][i]
        c = indices[1][i]
        im[r-2:r+3, c-2:c+3] = 1
    im = th.from_numpy(im)
    save_image(im, "../vis_res/n_label_magnified5x5.tif")

def vis_res(prd_path, bg_img_path):
    paste_loc = (1999, 0)
    fg = Image.open(prd_path)
    bg = Image.open(bg_img_path).convert("L")
    name = bg_img_path.split("/")[-1].split(".")[0]
    bg.save(name+".jpg")
    # bg.show()
    bg.paste(fg, paste_loc)
    bg.save("new_"+name+".jpg")
    # bg.show()

def save_config(path, train_param, data_param):
    with open(path, 'w') as f:
         for params in [train_param, data_param]:
            for e in params:
                f.write('{}: {}\n'.format(e, params[e]))

def plot(out_path, dset_path, colormap='coolwarm', region='Veneto'):
    # two other colormaps: bwr, seismic
    
    # bottom = mpl.cm.get_cmap('Oranges_r', 128)
    # top = mpl.cm.get_cmap('Blues', 128)
    # newcolors = np.vstack((top(np.linspace(0, 0.1, 128)), bottom(np.linspace(1.0, 1, 128))))
    # newcmp = mpl.colors.ListedColormap(newcolors, name='OrangeBlue')
    f = h5py.File(dset_path, 'r')
    gt = f[region]['gt'][0,:,:]
    gt[gt<0] = 0
    output = np.load(out_path)
    # __norm__ = mpl.colors.Normalize(vmin=0, vmax=output.max())
    __norm__ = mpl.colors.LogNorm(vmin=0.001, vmax=1)
    # plt.subplot(1,2,1)
    plt.imshow(output, cmap=colormap, norm=__norm__)
    # plt.subplot(1,2,2)
    # plt.imshow(gt, cmap='Reds')
    # plt.show()
    save_to = '/'.join(out_path.split('/')[:-1])
    name = out_path.split('/')[-3]
    plt.savefig(save_to+'/'+name+'.png', bbox_inches='tight')
    plt.savefig(save_to+'/'+name+'.eps', bbox_inches='tight')
    plt.show()