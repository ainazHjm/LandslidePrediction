import model
import numpy as np
import torch as th
import torch.optim as to
import torch.nn as nn
from time import ctime
from tensorboardX import SummaryWriter
# pylint: disable=E1101,E0401

writer = SummaryWriter()

def validate(model, valset):
    # criterion = nn.BCELoss()
    # (_, h, w)= valset.shape
    criterion = nn.NLLLoss()
    predictions = model(valset[:-1, :, :].unsqueeze(0).cuda())
    indices = (valset[0, :, :].view(-1) == -100).nonzero()
    loss = criterion(
        predictions.view(-1),
        valset[-1, :, :].cuda().view(-1),
        ignore_index = indices
        )
    return loss.item()

def train(args, train_data, val_data):
    (c, h, w) = train_data.shape
    train_model = model.FCN().cuda() if args.model == "FCN" else model.FCNwPool().cuda()
    optimizer = to.Adam(train_model.parameters(), lr = args.lr, weight_decay = args.decay)
    # criterion = nn.BCELoss()
    criterion = nn.NLLLoss()
    print("model is initialized ...")

    running_loss = []
    for i in range(args.n_epochs):
        optimizer.zero_grad()

        input_data = train_data[:-1, :, :].unsqueeze(0).cuda()
        if input_data.shape != (1, c-1, h, w):
            raise ValueError("the shape of the input data does not match.")
        
        predictions = train_model(input_data)
        # print(train_data[-1, :, :])
        if predictions.shape != (1, 1, h, w):
            raise ValueError("the shape of the output data does not match.")
        # print(predictions.shape)
        # print(train_data[-1, :, :].unsqueeze(0).shape)
        indices = (train_data[0, :, :].view(-1) == -100).nonzero()
        loss = criterion(
            predictions.view(-1),
            train_data[-1, :, :].cuda().view(-1),
            ignore_index = indices
            )

        print(">> loss: %f" % loss.item())
        writer.add_scalar("train loss", loss.item(), i)
        running_loss.append(loss.item())

        v_loss = validate(train_model, val_data)
        # import ipdb
        # ipdb.set_trace()
        print(">> val loss: %f" % loss.item())
        writer.add_scalar("validation loss", v_loss, i)

        loss.backward()
        optimizer.step()
    th.save(train_model, "../models/CNN/"+ctime()+".pt")
    return running_loss

def find_accuracy(model, data):
    predictions = model(data[:-1, :, :].unsqueeze(0).cuda()).view(-1)
    print(th.sum(predictions > 0.5), th.sum(predictions <= 0.5))
    predictions[predictions > 0.5] = 1
    predictions[predictions <= 0.5] = 0
    print(predictions)
    acc = th.sum(data[-1, :, :].long().cuda().view(-1) == predictions.long())
    print(acc, predictions.shape)
    print(data[-1, :, :].long())
    print(th.sum(data[-1, :, :].long().cuda().view(-1) == 1))
    return acc/predictions.shape[0]

def cross_validate(args, data):
    (_, _, w) = data.shape
    div = w//5
    min_loss = th.inf
    idx = -1
    for val_idx in range(0, 5):
        val_data = data[:, :, val_idx*div:(val_idx+1)*div]
        train_data = th.cat((data[:, :, 0:val_idx*div], data[:, :, (val_idx+1)*div:]), 2)
        loss = train(args, train_data, val_data).item()
        if loss[-1] < min_loss:
            min_loss = loss[-1]
            idx = val_idx
    return th.cat((data[:, :, 0:idx*div], data[:, :, (idx+1)*div:]), 2), data[:, :, idx*div:(idx+1)*div], idx
