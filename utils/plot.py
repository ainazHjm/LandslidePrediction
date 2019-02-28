# pylint: disable=E1101
import torch as th
import numpy as np
from matplotlib import pyplot
from torch.nn import Sigmoid
from torch import save
from time import ctime
from PIL import Image
from torchvision.utils import save_image

def save_results(model, val_data):
    th.cuda.empty_cache()
    sig = Sigmoid()
    (_, h, w) = val_data.shape
    (hs, ws) = (999, 999)
    predictions = th.zeros(h, w)
    for i in range(h//hs):
        for j in range(w//ws):
            input_data = val_data[:-1, i*hs:(i+1)*hs, j*ws:(j+1)*ws].unsqueeze(0).cuda()
            predictions[i*hs:(i+1)*hs, j*ws:(j+1)*ws] = sig(model.forward(input_data).squeeze(0).squeeze(0)).detach()
    input_data = val_data[:-1, (h//hs)*hs:, (w//ws)*ws:].unsqueeze(0).cuda()
    predictions[hs*(h//hs):, ws*(w//ws):] = sig(model.forward(input_data).squeeze(0).squeeze(0)).detach()
    name = ctime()
    save(predictions, "../output/"+name+".pt")
    save_image(predictions, "output/"+name+".jpg")

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
