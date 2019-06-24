# pylint: disable=E1101
import torch as th
import numpy as np
import os
# import scipy.misc
from torch.nn import Sigmoid
from time import ctime
from PIL import Image
from torchvision.utils import save_image

def validate_all(args, model, test_loader, shape=(4201, 19250)):
    sig = Sigmoid()
    if not os.path.exists(args.save_res_to+args.region+'/'+args.load_model.split('/')[-1].split('.')[0]):
        os.mkdir(args.save_res_to+args.region+'/'+args.load_model.split('/')[-1].split('.')[0])
    save_to = args.save_res_to+args.region+'/'+args.load_model.split('/')[-1].split('.')[0]+'/'
    res_img = np.zeros(shape)
    test_loader_iter = iter(test_loader)
    for _ in range(len(test_loader_iter)):
        batch_sample = test_loader_iter.next()
        (_, c, h, w) = batch_sample['data'].shape
        indices = batch_sample['gt'] >= 0
        data = batch_sample['data'][indices.expand(-1, c, h, w)].cuda()
        prds = sig(model.forward(data))[:, :, h//2, w//2].view(-1, 1)
        for num in range(prds.shape[0]):
            row, col = batch_sample['index'][num]
            res_img[row, col] = prds[num]
    np.save(save_to+'predictions.npy', res_img)
            
def find_positives(testData):
    indices = []
    for i in range(len(testData)):
        if th.sum(testData[i]['gt'].cuda()) > 0:
            indices.append(i)
    return indices

def validate_on_ones(args, model, testData):
    import matplotlib.pyplot as plt
    sig = Sigmoid()
    if args.pos_indices:
        indices = np.load(args.pos_indices)
        print('loaded positive indices.')
    else:
        indices = find_positives(testData)
        print('found positive indices.')
        np.save(('/').join(args.data_path.split('/')[:-1])+'/pos_indices.npy', np.array(indices))
        print('wrote pos_indices')
    num_samples = 4
    samples = np.random.choice(indices, num_samples)
    for i in range(num_samples):
        d = testData[samples[i]]['data']
        prds = sig(model.forward(d.view(1, args.feature_num, args.ws+2*args.pad, args.ws+2*args.pad).cuda()))[0, 0, args.pad:-args.pad, args.pad:-args.pad]
        print(prds.shape)
        plt.subplot(num_samples, 2, i*2+1)
        plt.imshow(prds.cpu().data.numpy())
        plt.subplot(num_samples, 2, (i+1)*2)
        plt.imshow(testData[samples[i]]['gt'][0, :, :].data.numpy())
    plt.show()

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
            f.write('\n')