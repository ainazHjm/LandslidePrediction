import torch as th
import numpy as np
from matplotlib import pyplot
from torch.nn import Sigmoid
from torch import save
from time import ctime
from PIL import Image
from torchvision.utils import save_image

def save_results(args, model, val_data):
    sig = Sigmoid()
    (_, h, w) = val_data.shape
    # model = model.cuda()
    predictions = sig(model(val_data[:-1, :, :].unsqueeze(0).cuda()))
    im = predictions.view(h, w).detach()
    if args.save_result:
        name = ctime()
        save(im, "../output/"+name+".pt")
        save_image(im, "output/"+name+".jpg")
        # Image.fromarray(im.numpy()).save("../output/val_res"+name+".tif")
    # im = predictions.view(h, w).detach()
    # pyplot.imshow(im.numpy())

def create_geotif(
    name,
    val_idx = '../image_data/data/val_idx.npy',
    predictions = '../output/'+name+'.pt',
    train_data = '../image_data/data/train_data.pt',
    )
    vi = np.load(val_idx)
    prds = th.load(predictions)
    td = th.load(train_data)
    (h, w1) = prdss.shape
    (_, w2) = td.shape
    res = th.zeros(h, w1+w2)
    div = (w1+w2)//5
    res = th.cat((td[:, 0:vi*div], prds, td[:, vi*div:]), 0)
    save_image(res, "../output/"+name+".pt")