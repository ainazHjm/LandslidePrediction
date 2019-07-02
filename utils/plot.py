# pylint: disable=E1101
import torch as th
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.nn import Sigmoid
from time import ctime
from PIL import Image
from torchvision.utils import save_image
from sklearn.manifold import TSNE

def embed(data):
    (c, h, w) = data.shape
    X = data.reshape((h*w, c))
    X_transformed = TSNE(n_components=2).fit_transform(X)
    fig = plt.figure()
    fig.add_subplot(1,1,1)
    plt.scatter(X_transformed[:,0], X_transformed[:,1])
    plt.show()

def visualize(data_path, sample_path, region='Veneto', pad=32):
    from loader import SampledPixDataset
    dataset = SampledPixDataset(data_path, sample_path+'train_data.npy', region, pad, 'train')
    idx = np.random.choice(np.arange(len(dataset)), 1)
    data = dataset[idx[0]]['data']
    gt = np.zeros((200, 200))
    gt[100, 100] = dataset[idx[0]]['gt']
    fig = plt.figure(figsize=(65, 65))
    for i in range(1, 96):
        fig.add_subplot(5, 19, i)
        if i == 95:
            plt.imshow(gt)    
        else:
            plt.imshow(data[i-1, :, :])
    plt.show()
    # embed(data)

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