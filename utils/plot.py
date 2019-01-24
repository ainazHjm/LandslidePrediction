from matplotlib import pyplot
from torch.nn import Sigmoid
from torch import save
from time import ctime

def visualise(args, model, val_data):
    sig = Sigmoid()
    (_, h, w) = val_data.shape
    predictions = sig(model(val_data[:-1, :, :].unsqueeze(0)))
    if args.save_result:
        save(predictions.view(h, w), "../output/val_res"+ctime()+".pt")
        save(predictions.view(h, w), "../output/val_res"+ctime()+".tif")
    im = predictions.view(h, w)
    pyplot.imshow(im)