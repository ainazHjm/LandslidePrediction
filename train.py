import model
import numpy as np
import torch as th
import torch.optim as to
import torch.nn as nn
from time import ctime
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
from torch.optim.lr_scheduler import ReduceLROnPlateau
# pylint: disable=E1101,E0401,E1123

writer = SummaryWriter()

def make_patches(train_data_path, window=999):
    train_data = th.load(train_data_path)
    (c, h, w) = train_data.shape
    hnum = h//window
    wnum = w//window
    input_data = th.zeros(hnum*wnum, c-1, window, window)
    label = th.zeros(hnum*wnum, 1, window, window)
    for i in range(hnum):
        for j in range(wnum):
            input_data[i*wnum+j, :, :, :] = train_data[:-1, i*window:(i+1)*window, j*window:(j+1)*window]
            label[i*wnum+j, :, :, :] = train_data[-1, i*window:(i+1)*window, j*window:(j+1)*window]
    return input_data, label


def validate(model, valset):
    (hs, ws) = (999, 999)
    (_, h, w) = valset.shape
    criterion = nn.BCEWithLogitsLoss(pos_weight=th.Tensor([50]).cuda())
    running_loss = 0
    for i in range(h//hs):
        for j in range(w//ws):
            label = valset[-1, i*hs:(i+1)*hs, j*ws:(j+1)*ws].cuda()
            input_data = valset[:-1, i*hs:(i+1)*hs, j*ws:(j+1)*ws].unsqueeze(0).cuda()
            predictions = model.forward(input_data).squeeze(0).squeeze(0)
            indices = input_data[0, 0, :, :] != -100
            if len(predictions[indices]) == 0:
                continue
            loss = criterion(
                predictions[indices],
                label[indices]
            )
            running_loss += loss.item()
    return running_loss/((h//hs) * (w//ws))

def train(args, val_data, train_data_path="../image_data/data/CNN/train_data.pt"):
    th.cuda.empty_cache()
    train_data, train_label = make_patches(train_data_path)
    # num_iters = (7, 8) # dividing the whole image into 56 patches of size (999x999)
    # (hs, ws) = (999, 999)
    train_model = model.FCN().cuda() if args.model == "FCN" else model.FCNwPool((4, 999, 999)).cuda()
    if args.load_model_path:
        train_model.load_state_dict(th.load(args.load_model_path).state_dict())
    optimizer = to.Adam(train_model.parameters(), lr = args.lr, weight_decay = args.decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5)
    # criterion = nn.NLLLoss()
    criterion = nn.BCEWithLogitsLoss(pos_weight=th.Tensor([50]).cuda())
    # criterion = nn.BCEWithLogitsLoss()
    print("model is initialized ...")

    bs = args.batch_size
    num_iters = train_data.shape[0]//bs
    for i in range(args.n_epochs):
        running_loss = 0
        v_running_loss = 0        
        for j in range(num_iters):
            optimizer.zero_grad()
            input_data = train_data[j*bs:(j+1)*bs, :, :, :].cuda()
            label = train_label[j*bs:(j+1)*bs, :, :, :].cuda().squeeze(1)
            predictions = train_model.forward(input_data).squeeze(1)
            indices = input_data[:, 0, :, :] != -100
            # print(predictions.shape, indices.shape)
            if len(predictions[indices]) == 0:
                continue
            loss = criterion(
                predictions[indices],
                label[indices]
                )
            print(">> loss: %f" % loss.item())
            running_loss += loss.item()
            v_loss = validate(train_model, val_data)
            v_running_loss += v_loss
            # import ipdb
            # ipdb.set_trace()
            print(">> val loss: %f" % v_loss)

            loss.backward()
            optimizer.step()
        
                
            # writer.add_scalar("train loss", loss.item(), i*num_iters+j)
            # writer.add_scalar("validation loss", v_loss, i*num_iters+j)
        scheduler.step(v_running_loss/(num_iters*bs))
        writer.add_scalar("train loss", running_loss/(num_iters*bs), i)
        writer.add_scalar("validation loss", v_running_loss/(num_iters*bs), i)
        
    th.save(train_model, "../models/CNN/"+ctime().replace("  "," ").replace(" ", "_").replace(":","_")+".pt")
    print("model has been trained and saved.")
    return running_loss/(num_iters*args.n_epochs*bs)

# def find_accuracy(model, data):
#     predictions = model(data[:-1, :, :].unsqueeze(0).cuda()).view(-1)
#     print(th.sum(predictions > 0.5), th.sum(predictions <= 0.5))
#     predictions[predictions > 0.5] = 1
#     predictions[predictions <= 0.5] = 0
#     print(predictions)
#     acc = th.sum(data[-1, :, :].long().view(-1).cuda() == predictions.long())
#     print(acc, predictions.shape)
#     print(data[-1, :, :].long())
#     print(th.sum(data[-1, :, :].long().view(-1).cuda() == 1))
#     return acc/predictions.shape[0]

def cross_validate(args, data):
    '''
    TODO: complete this function
    '''
    (_, _, w) = data.shape
    div = w//5
    min_loss = th.inf
    idx = -1
    for val_idx in range(0, 5):
        val_data = data[:, :, val_idx*div:(val_idx+1)*div]
        train_data = th.cat((data[:, :, 0:val_idx*div], data[:, :, (val_idx+1)*div:]), 2)
        loss = train(args, val_data).item() 
        if loss[-1] < min_loss:
            min_loss = loss[-1]
            idx = val_idx
    return th.cat((data[:, :, 0:idx*div], data[:, :, (idx+1)*div:]), 2), data[:, :, idx*div:(idx+1)*div], idx
