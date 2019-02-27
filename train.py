import model
import numpy as np
import torch as th
import torch.optim as to
import torch.nn as nn
from time import ctime
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
# pylint: disable=E1101,E0401,E1123

writer = SummaryWriter()

def validate(model, valset):
    label = valset[-1, :, :]
    criterion = nn.BCEWithLogitsLoss(pos_weight=th.Tensor([50]).cuda())
    # criterion = nn.BCEWithLogitsLoss()
    (predictions, _) = model.forward(valset[:-1, :, :].unsqueeze(0).cuda())
    prds = predictions[0, 0, :, :]
    loss = criterion(
        prds[valset[0, :, :] != -100],
        label[valset[0, :, :] != -100].cuda()
        )
    return loss.detach().item()

def train(args, train_data, val_data):
    th.cuda.empty_cache()
    (c, h, w) = train_data.shape
    num_iters = (7, 8) # dividing the whole image into 56 patches of size (999x999)
    hs = h // num_iters[0]
    ws = w // num_iters[1]
    train_model = model.FCN().cuda() if args.model == "FCN" else model.FCNwPool((c-1, hs, ws)).cuda()
    if args.load_model_path:
        train_model.load_state_dict(th.load(args.load_model_path).state_dict())
    optimizer = to.Adam(train_model.parameters(), lr = args.lr, weight_decay = args.decay)
    # criterion = nn.NLLLoss()
    criterion = nn.BCEWithLogitsLoss(pos_weight=th.Tensor([50]).cuda())
    # criterion = nn.BCEWithLogitsLoss()
    print("model is initialized ...")

    running_loss = 0
    for i in range(args.n_epochs):
        for j in range(num_iters[0]):
            for k in range(num_iters[1]):
                optimizer.zero_grad()

                input_data = train_data[:-1, j*hs:(j+1)*hs, k*ws:(k+1)*ws].unsqueeze(0).cuda()
                # input_data = input_data.unsqueeze(0).cuda()
                label = train_data[-1, j*hs:(j+1)*hs, k*ws:(k+1)*ws]
                print(input_data.shape)
                (predictions, layer_outputs) = train_model.forward(input_data)
                prds = predictions[0, 0, :, :]
                print("------ %d" %i)
                loss = criterion(
                    prds[train_data[0, :, :] != -100],
                    label[train_data[0, :, :] != -100].cuda()
                    )

                print(">> loss: %f" % loss.item())
                writer.add_scalar("train loss", loss.item(), i)
                running_loss = running_loss + loss.item()

                v_loss = validate(train_model, val_data)
                # import ipdb
                # ipdb.set_trace()
                print(">> val loss: %f" % v_loss)
                writer.add_scalar("validation loss", v_loss, i)

                loss.backward()
                optimizer.step()
                if args.model == "FCNwPool":
                    if (i+1) % 20 == 0:
                        t = ctime()
                        for idx, out_img in enumerate(layer_outputs):
                            save_image(out_img, "../output/visualise/CNN/layer"+str(idx)+"_"+t+".jpg")
        
    th.save(train_model, "../models/CNN/"+ctime().replace("  "," ").replace(" ", "_").replace(":","_")+".pt")
    print("model has been trained and saved.")
    return running_loss/args.n_epochs

def find_accuracy(model, data):
    predictions = model(data[:-1, :, :].unsqueeze(0).cuda()).view(-1)
    print(th.sum(predictions > 0.5), th.sum(predictions <= 0.5))
    predictions[predictions > 0.5] = 1
    predictions[predictions <= 0.5] = 0
    print(predictions)
    acc = th.sum(data[-1, :, :].long().view(-1).cuda() == predictions.long())
    print(acc, predictions.shape)
    print(data[-1, :, :].long())
    print(th.sum(data[-1, :, :].long().view(-1).cuda() == 1))
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
