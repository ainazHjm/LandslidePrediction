from data import process
import model
import numpy as np
import torch as th
import torch.optim as to
from tensorboardX import SummaryWriter

writer = SummaryWriter()

def validate(model, valset):
    criterion = nn.NLLLoss()
    input_data = valset[:-1, :, :].unsqueeze(0)
    labels = valset[-1, :, :]
    predictions = model(input_data)
    loss = criterion(predictions, labels)
    return loss.item()

def train(args, train_data, val_data):
    # val_idx, val_data, train_data = process(args.model, args.divide) # data: 5x6630x9387
    (c, h, w) = train_data.shape
    trainset = train_data[:-1, :, :]
    labels = train_data[-1, :, :]

    if args.model == "SimpleCNN":
        train_model = model.SimpleCNN
    elif args.model == "SimpleNgbCNN":
        train_model = model.SimpleNgbCNN
    elif args.model == "FCN":
        train_model = model.FCN
    else:
        train_model = model.ComplexCNN
    
    optimizer = to.Adam(train_model.parameters(), lr = args.lr, weight_decay = args.decay)
    criterion = nn.NLLLoss()
    running_loss = []

    for i in range(args.n_epochs):
        optimizer.zero_grad()

        input_data = train_set.unsqueeze(0)
        ground_truth = labels
        if input_data.shape != (1, c, h, w):
            raise ValueError("the shape of the input data does not match.")
        
        predictions = train_model(input_data)
        if predictions.shape != (1, c, h, w):
            raise ValueError("the shape of the output data does not match.")

        loss = criterion(predictions, groundtruth)
        print(">> loss: %f" % loss.item())
        writer.add_scalar("train loss", loss.item(), i)
        running_loss.append(loss.item())

        v_loss = validate(train_model, val_data)
        print(">> val loss: %f" % loss.item())
        writer.add_scalar("validation loss", v_loss, i)

        loss.backward()
        optimizer.step()
    th.save(train_model, "../models/CNN/"+args.model+str(args.lr))
    return running_loss

def find_accuracy(model, data):
    input_data = data[:-1, :, :].unsqueeze(0)
    labels = data[-1, :, :].view(-1)
    predictions = model(input_data).view(-1)
    acc = th.sum(labels == predictions)
    return acc/labels.shape[0]

def cross_validate(args, data):
    (c, h, w) = data.shape
    div = w//5
    min_loss = th.inf
    idx = -1
    for val_idx in range(0, 5):
        val_data = data[:, :, val_idx*div:(val_idx+1)*div]
        train_data = th.cat((data[:, :, 0:val_idx*div], data[:, :, (val_idx+1)*div:]), 1)
        loss = train(args, train_data, val_data).item()
        if loss[-1] < min_loss:
            min_loss = loss[-1]
            idx = val_idx
    return th.cat((data[:, :, 0:idx*div], data[:, :, (idx+1)*div:]), 1), data[:, :, idx*div:(idx+1)*div], idx
