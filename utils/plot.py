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
    im[val_data[-1, :, :]==-1] = 0
    cum_logprob = th.sum(th.log(im[val_data[-1, :, :]!=-1].cpu()))

    name = ctime().replace("  ", " ").replace(" ", "_").replace(":", "_")
    save(im, "../output/CNN/"+name+".pt")
    save_image(im, "../output/CNN/"+name+".jpg")
    print("validating model ...")
    print(">> cumulative log probability: %f" % cum_logprob)

def create_geotif(
        prd_name,
        val_idx = '../image_data/data/val_idx.npy',
        train_data = '../image_data/data/train_data.pt'
        ):
    # predictions = '../output/'+name+'.pt',
    vi = np.load(val_idx)
    prds = th.load("../output/CNN/"+prd_name+".pt")
    td = th.load(train_data)
    (h, w1) = prds.shape
    (_, _, w2) = td.shape
    # print(w2)
    res = th.zeros(h, w1+w2)
    div = (w1+w2)//5
    res = th.cat((td[-1, :, 0:vi*div].cuda(), prds, td[-1, :, vi*div:].cuda()), 1)
    save_image(res, "../output/CNN/"+prd_name+".jpg")
