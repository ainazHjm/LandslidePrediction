import torch as th
import numpy as np
from matplotlib import pyplot
from torch.nn import Sigmoid
from torch import save
from time import ctime
from PIL import Image
from torchvision.utils import save_image

# pylint: disable=E1101

def save_results(model, val_data):
    th.cuda.empty_cache()
    sig = Sigmoid()
    (_, h, w) = val_data.shape
    predictions = sig(model(val_data[:-1, :, :].unsqueeze(0).cuda().detach()))
    im = predictions.view(h, w).detach()
    name = ctime()
    save(im, "../output/"+name+".pt")
    save_image(im, "output/"+name+".jpg")
    # Image.fromarray(im.numpy()).save("../output/val_res"+name+".tif")
    # im = predictions.view(h, w).detach()
    # pyplot.imshow(im.numpy())

# def create_geotif(
#     name,
#     val_idx = '../image_data/data/val_idx.npy',
#     predictions = '../output/'+name+'.pt',
#     train_data = '../image_data/data/train_data.pt',
#     ):
#     vi = np.load(val_idx)
#     prds = th.load("../output/CNN/"+prd_name+".pt")
#     td = th.load(train_data)
#     (h, w1) = prds.shape
#     (_, _, w2) = td.shape
#     # print(w2)
#     res = th.zeros(h, w1+w2)
#     div = (w1+w2)//5
#     res = th.cat((td[:, 0:vi*div], prds, td[:, vi*div:]), 0)
#     save_image(res, "../output/"+name+".pt")

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
