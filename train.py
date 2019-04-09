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
    criterion = nn.BCEWithLogitsLoss(pos_weight=th.Tensor([500]).cuda())
    running_loss = 0
    cnt = 0
    # print(h//hs, w//ws)
    for i in range(h//hs):
        for j in range(w//ws):
            label = valset[-1, i*hs:(i+1)*hs, j*ws:(j+1)*ws].cuda()
            input_data = valset[:-1, i*hs:(i+1)*hs, j*ws:(j+1)*ws].unsqueeze(0).cuda()
            # print(input_data)
            # indices = 1 - th.isnan(input_data[0, 0, :, :])
            if th.sum(1 - th.isnan(input_data[0, 0, :, :])) > 0:
                # print(i, j)
                # print("number of not nan, number of all points >> %d, %d" % (th.sum(1 - th.isnan(input_data[0, 0, :, :])), 999*999))
                predictions = model.forward(input_data).squeeze(0).squeeze(0)
                indices = (predictions == th.tensor(float('-inf')).cuda()) + th.isnan(predictions)
                indices = 1 - indices
                # print(predictions)
                # print(indices)
                if th.sum(indices) == 0:
                    # print(">>>> ignore batch in validate")
                    # print(input_data, predictions)
                    # print("all predictions nan ...")
                    continue
                loss = criterion(
                    predictions[indices],
                    label[indices]
                )
                running_loss += loss.item()
                cnt = cnt + 1
    return running_loss/cnt

def sanity_check(model, train_data, train_label, valset, criterion):
    v_loss = validate(model, valset)
    t_loss = 0
    cnt = 0
    for i in range(train_data.shape[0]):
        in_data = train_data[i, :, :, :].cuda()
        label = train_label[i, :, :, :].squeeze(0).cuda()
        # indices = 1 - th.isnan(in_data[0, :, :])
        if th.sum(1 - th.isnan(in_data[0, :, :])) > 0:
            # print(in_data.shape)
            predictions = model.forward(in_data.unsqueeze(0)).squeeze(0).squeeze(0)
            indices = (predictions == th.tensor(float('-inf')).cuda()) + th.isnan(predictions)
            indices = 1 - indices
            if th.sum(indices) == 0:
                continue
            t_loss += criterion(predictions[indices], label[indices]).item()
            cnt = cnt + 1
    t_loss = t_loss / cnt
    print("running loss before training >> val: %f train: %f" %(v_loss, t_loss))

def train(args, val_data, train_data_path="../image_data/data/Veneto/train_data.pt"):
    '''
    Trains on a batch of patches of size (4, 999, 999).
    '''
    th.cuda.empty_cache()
    train_data, train_label = make_patches(train_data_path)
    train_model = model.FCN().cuda() if args.model == "FCN" else model.FCNwPool((4, 999, 999)).cuda()
    if args.load_model_path:
        train_model.load_state_dict(th.load(args.load_model_path).state_dict())
    print("model is initialized ...")

    optimizer = to.Adam(train_model.parameters(), lr = args.lr, weight_decay = args.decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=th.Tensor([500]).cuda())

    bs = args.batch_size
    num_iters = train_data.shape[0]//bs
    sanity_check(train_model, train_data, train_label, val_data, criterion)
    th.cuda.empty_cache()
    
    for i in range(args.n_epochs):
        running_loss = 0
        cnt = 0
        for j in range(num_iters):
            optimizer.zero_grad()
            input_data = train_data[j*bs:(j+1)*bs, :, :, :].cuda()
            label = train_label[j*bs:(j+1)*bs, :, :, :].cuda()
            # indices = 1 - th.isnan(input_data[:, 0, :, :])
            if th.sum(1 - th.isnan(input_data[:, 0, :, :])) > 0:
                # indices = indices.unsqueeze(1)
                predictions = train_model.forward(input_data)
                indices = (predictions == th.tensor(float('-inf')).cuda()) + th.isnan(predictions)
                indices = 1 - indices
                if th.sum(indices) == 0:
                    continue
                loss = criterion(
                    predictions[indices],
                    label[indices]
                    )
                print("%d >> loss: %f" % (cnt, loss.item()))
                running_loss += loss.item()
                cnt = cnt + 1
                loss.backward()
                optimizer.step()
        
        v_loss = validate(train_model, val_data)
        scheduler.step(v_loss)
        print("--- validation loss: %f" % v_loss)
        writer.add_scalar("train loss", running_loss/cnt*bs, i)
        writer.add_scalar("validation loss", v_loss, i)
        
    th.save(train_model, "../models/CNN/"+ctime().replace("  "," ").replace(" ", "_").replace(":","_")+".pt")
    print("model has been trained and saved.")
    return running_loss

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
