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
    return name
