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

def filter_batch(batch_sample):
    bs = batch_sample['data'].shape[0]
    data, gt, ignore_idx = [], [], 0
    for i in range(bs):    
        ignore = batch_sample['data'][i, 45, :, :]<0 # feature 45 is the landcover_1
        if th.sum(ignore) > 0:
            ignore_idx += 1
        else:
            data.append(batch_sample['data'][i, :, :, :])
            gt.append(batch_sample['gt'][i, :, :, :])
    if not data: # the whole batch is ignored
        return -1, data, gt, bs
    else:
        return 1, th.stack(data), th.stack(gt), ignore_idx

def validate(args, model, test_loader):
    with th.no_grad():
        criterion = nn.BCEWithLogitsLoss(pos_weight=th.Tensor([20]).cuda())
        # criterion = nn.BCEWithLogitsLoss()
        test_loader_iter = iter(test_loader)
        ignore_idx, running_loss, cnt = 0, 0, 0
        for _ in range(len(test_loader_iter)):
            batch_sample = test_loader_iter.next()
            ret, data, gt, ignore = filter_batch(batch_sample)
            if ret < 0:
                continue
            ignore_idx += ignore    
            prds = model.forward(data.cuda())
            loss = criterion(
                prds[:, :, args.pad:-args.pad, args.pad:-args.pad].view(-1, 1, args.ws, args.ws),
                gt.cuda())
            running_loss += loss.item()
            cnt += 1
        print('~~~ validating ~~~: %f percent of the data is ignored.' %(ignore_idx/(len(test_loader_iter)*args.batch_size)))
        return running_loss/cnt

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
    
    train_model = model.FCN((args.feature_num, args.ws+args.pad*2, args.ws+args.pad*2)).cuda() if args.model == "FCN" else model.FCNwPool((args.feature_num, args.ws+2*args.pad, args.ws+2*args.pad), args.pix_res).cuda()
    if args.load_model:
        train_model.load_state_dict(th.load(args.load_model).state_dict())
    print("model is initialized ...")

    optimizer = to.Adam(train_model.parameters(), lr = args.lr, weight_decay = args.decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=args.patience)
    criterion = nn.BCEWithLogitsLoss(pos_weight=th.Tensor([20]).cuda())
    # criterion = nn.BCEWithLogitsLoss()

    loss_100, iter_cnt = 0, 0
    for epoch in range(args.n_epochs):
        running_loss = 0
        train_loader_iter = iter(train_loader)
        ignore_idx, epoch_iters = 0, 0
        for _ in range(len(train_loader_iter)):
            optimizer.zero_grad()
            batch_sample = train_loader_iter.next()
            ret, data, gt, ignore = filter_batch(batch_sample)
            del batch_sample
            
            if ret < 0:
                continue
            ignore_idx += ignore
            prds = train_model.forward(data.cuda())
            loss = criterion(
                prds[:, :, args.pad:-args.pad, args.pad:-args.pad].view(-1, 1, args.ws, args.ws),
                gt.cuda()
                )
            
            running_loss += loss.item()
            loss_100 += loss.item()
            loss.backward()
            optimizer.step()

            writer.add_scalar("loss/train_iter", loss.item(), iter_cnt+1)
            writer.add_scalars(
                "probRange",
                {'min': th.min(sig(prds)), 'max': th.max(sig(prds))},
                iter_cnt+1
            )
            if (iter_cnt+1) % 100 == 0:
                writer.add_scalar("loss/train_100", loss_100/100, iter_cnt+1)
                loss_100 = 0
            iter_cnt += 1
            epoch_iters += 1
            del prds, gt
        print('--- epoch %d: %f percent of the data is ignored.' %(epoch, ignore_idx/(len(train_loader_iter)*args.batch_size)))

        if (epoch+1) % args.s == 0:
            th.save(train_model, dir_name+'/'+str(epoch)+'_'+exe_time+'.pt')
        v_loss = validate(args, train_model, test_loader)
        scheduler.step(v_loss)
        writer.add_scalars(
            "loss/grouped",
            {'test': v_loss, 'train': running_loss/epoch_iters},
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