# pylint: disable=E1101
import torch as th
import numpy as np
# from matplotlib import pyplot
from torch.nn import Sigmoid
# from torch import save
from time import ctime
from PIL import Image
from torchvision.utils import save_image
import os
# from train import load_data

def data_loader(args, fname):
    dp = args.data_path
    data = []
    names = []
    for name in fname:
        im = np.load(dp+name) # 3d shape
        im = im.astype(np.uint8)
        data.append(im[:-1, :, :])
        names.append(name)
    return np.asarray(data), np.asarray(names) #4d shape

def save_results(args, model, data_idx):
    th.cuda.empty_cache()
    dir_name = args.save_res_to + args.load_model.split('/')[-1].split('.')[0]
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    sig = Sigmoid()
    bs = args.batch_size
    num_iters = (data_idx.shape[0])//bs
    for i in range(num_iters+1):
        in_d, names = data_loader(args, data_idx[i*bs:(i+1)*bs]) if i < num_iters else data_loader(args, data_idx[i*bs:])
        in_d = th.tensor(in_d).float().cuda()
        ignore = 1 - ((in_d[:, 0, :, :]==1) + (in_d[:, 0, :, :]==0))
        prds = sig(model.forward(in_d))
        prds[ignore.unsqueeze(0)] = 0
        for j in range(prds.shape[0]):
            save_image(prds[j, 0, :, :], dir_name+'/'+names[j].split('.')[0]+'.tif')
            np.save(dir_name+'/'+names[j], prds[j, 0, :, :].cpu().data.numpy())

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
