# pylint: disable=E1101
import torch as th
import numpy as np
import os
# import scipy.misc
from torch.nn import Sigmoid
from time import ctime
from PIL import Image
from torchvision.utils import save_image

def data_loader(args, fname, feature_num=21):
    # max_shape = (464, 464)
    dp = args.data_path
    data_dir = 'data_'+str(args.pad*2)+'/'
    # label_dir = 'gt_200/'
    data = []
    names = []
    for name in fname:
        features = []
        for i in range(feature_num):
           features.append(np.load(dp+data_dir+str(i)+'_'+name))
        features = np.asarray(features)
        data.append(features)
        names.append(name)
    return np.asarray(data), np.asarray(names) #4d shape

def save_results(args, model, idx):
    dir_name = args.save_res_to + args.region + '/' + args.load_model.split('/')[-1].split('.')[0]
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    
    data_idx = np.load(args.data_path+'tdIdx.npy') if idx == 'train' else np.load(args.data_path+'vdIdx.npy')
    num_iters = (data_idx.shape[0])//args.batch_size
    sig = Sigmoid()

    for i in range(num_iters):
        in_d, names = data_loader(args, data_idx[i*args.batch_size:(i+1)*args.batch_size])
        in_d = th.tensor(in_d)
        ignore = 1 - ((in_d[:, 0, :, :]==1) + (in_d[:, 0, :, :]==0))
        prds = sig(model.forward(in_d.cuda()))
        del in_d
        prds[ignore.unsqueeze(1)] = 0
        for j in range(prds.shape[0]):
            np.save(dir_name+'/'+names[j], prds[j, 0, args.pad:-args.pad, args.pad:-args.pad].cpu().data.numpy())
    del prds
    in_d, names = data_loader(args, data_idx[-data_idx.shape[0]+num_iters*args.batch_size:])
    in_d = th.tensor(in_d)
    ignore = 1 - ((in_d[:, 0, :, :]==1) + (in_d[:, 0, :, :]==0))
    prds = sig(model.forward(in_d.cuda()))
    del in_d
    prds[ignore.unsqueeze(1)] = 0
    for i in range(prds.shape[0]):
        np.save(dir_name+'/'+names[i], prds[i, 0, args.pad:-args.pad, args.pad:-args.pad].cpu().data.numpy())

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

def save_config(path, args):
    with open(path, 'w') as f:
        for key in args.__dict__.keys():
            f.write(str(key)+': '+str(args.__dict__[key]))
