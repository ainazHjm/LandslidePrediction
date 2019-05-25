import model
import numpy as np
import torch as th
import torch.optim as to
import torch.nn as nn
import os
from time import ctime
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils import shuffle
from utils.plot import save_config
# pylint: disable=E1101,E0401,E1123

# def load_data(args, fname, feature_num=21):
#     dp = args.data_path
#     data = []
#     label = []
#     for name in fname:
#         features = []
#         for i in range(feature_num):
#             features.append(np.load(dp+str(i)+'/'+name)) # 2d shape
#         features = np.asarray(features)
#         data.append(features)
#         gt = np.load(dp+'gt/'+name)
#         label.append(gt.reshape((1, 200, 200)))
#     return np.asarray(data), np.asarray(label) #4d

def validate(args, model, test_loader):
    with th.no_grad():
        # criterion = nn.BCEWithLogitsLoss(pos_weight=th.Tensor([20]).cuda())
        criterion = nn.BCEWithLogitsLoss()
        running_loss = 0
        test_loader_iter = iter(test_loader)
        for _ in range(len(test_loader_iter)):
            batch_sample = test_loader_iter.next()
            prds = model.forward(batch_sample['data'].cuda())
            loss = criterion(
                prds[:, :, args.pad:-args.pad, args.pad:-args.pad].view(-1, 1, args.ws, args.ws),
                batch_sample['gt'].cuda())
            running_loss += loss.item()
        return running_loss/len(test_loader_iter)

def train(args, train_loader, test_loader):
    '''
    Trains on a batch of patches of size (22, 200, 200).
    TODO: 
        1. try with patience = 1
        2. find the correponding features names
        3. train the model with higher penalty weight for postive samples (ratio1:1)
        4. (*) train without batch normalization > not good
        5. train the model without the penalty weight (ratio 0.02:100)
    '''
    writer = SummaryWriter()
    sig = nn.Sigmoid()

    exe_time = ctime().replace("  "," ").replace(" ", "_").replace(":","_")
    dir_name = args.save_model_to + args.region +'/'+ exe_time if args.c else args.save_model_to
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    
    train_model = model.FCN().cuda() if args.model == "FCN" else model.FCNwPool((args.feature_num, args.ws+2*args.pad, args.ws+2*args.pad), args.pix_res).cuda()
    if args.load_model:
        train_model.load_state_dict(th.load(args.load_model).state_dict())
    print("model is initialized ...")

    optimizer = to.Adam(train_model.parameters(), lr = args.lr, weight_decay = args.decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=args.patience)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=th.Tensor([20]).cuda())
    criterion = nn.BCEWithLogitsLoss()

    loss_100 = 0
    for epoch in range(args.n_epochs):
        running_loss = 0
        train_loader_iter = iter(train_loader)
        for batch_idx in range(len(train_loader_iter)):
            optimizer.zero_grad()
            batch_sample = train_loader_iter.next()
            # import ipdb; ipdb.set_trace()
            prds = train_model.forward(batch_sample['data'].cuda())
            loss = criterion(
                prds[:, :, args.pad:-args.pad, args.pad:-args.pad].view(-1, 1, args.ws, args.ws),
                batch_sample['gt'].cuda()
                )
            running_loss += loss.item()
            loss_100 += loss.item()
            loss.backward()
            optimizer.step()

            writer.add_scalar("loss/train_iter", loss.item(), epoch*len(train_loader.dataset)+batch_idx)
            writer.add_scalars(
                "probRange",
                {'min': th.min(sig(prds)), 'max': th.max(sig(prds))},
                epoch*len(train_loader.dataset)+batch_idx
            )
            if (epoch*len(train_loader.dataset)+batch_idx+1) % 100 == 0:
                writer.add_scalar("loss/train_100", loss_100/100, epoch*len(train_loader.dataset)+batch_idx)
                # print("%d,%d >> loss: %f" % (epoch, batch_idx, loss_100/100))
                loss_100 = 0
    
        if (epoch+1) % args.s == 0:
            th.save(train_model, dir_name+'/'+str(epoch)+'_'+exe_time+'.pt')
        v_loss = validate(args, train_model, test_loader)
        scheduler.step(v_loss)
        writer.add_scalars(
            "loss/grouped",
            {'test': v_loss, 'train': running_loss/len(train_loader_iter)},
            epoch
        )
        for name, param in train_model.named_parameters():
            writer.add_histogram(
                name,
                param.clone().cpu().data.numpy(),
                epoch
            )

    writer.export_scalars_to_json(dir_name+'/loss.json')
    th.save(train_model, dir_name+'/final@'+str(epoch)+'_'+exe_time+'.pt')
    save_config(dir_name+'/config.txt', args)
    print("model has been trained and config file has been saved.")