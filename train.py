import model
import numpy as np
import torch as th
import torch.optim as to
import torch.nn as nn
import os
from time import ctime
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.plot import save_config
# pylint: disable=E1101,E0401,E1123

def filter_batch(batch_sample):
    (b, _, h, w) = batch_sample['data'].shape
    data, gt, ignore = [], [], 0
    for i in range(b):
        if batch_sample['data'][i, 45, h//2, w//2] < 0:
            ignore += 1
        else:
            data.append(batch_sample['data'][i, :, :, :])
            gt.append(batch_sample['gt'][i, :])
    return data, gt, ignore

def validate(args, model, test_loader):
    with th.no_grad():
        criterion = nn.BCEWithLogitsLoss(pos_weight=th.Tensor([20]).cuda())
        # criterion = nn.BCEWithLogitsLoss()
        running_loss, ignore_cnt, iter_cnt = 0, 0, 0
        test_loader_iter = iter(test_loader)
        for _ in range(len(test_loader_iter)):
            batch_sample = test_loader_iter.next()
            data, gt, ignore = filter_batch(batch_sample)
            ignore_cnt += ignore
            if not data:
                continue
            prds = model.forward(th.stack(data).cuda())
            loss = criterion(
                prds[:, :, args.pad:-args.pad, args.pad:-args.pad].view(-1, 1),
                th.stack(gt).cuda())
            running_loss += loss.item()
            iter_cnt += 1
        print('~~~ validating ~~~: %f of test data is ignored.' %(ignore_cnt/(len(test_loader_iter)*args.batch_size)))
        return running_loss/iter_cnt

def train(args, train_loader, test_loader):
    writer = SummaryWriter()
    sig = nn.Sigmoid()

    exe_time = ctime().replace("  "," ").replace(" ", "_").replace(":","_")
    dir_name = args.save_model_to + args.region +'/'+ exe_time if args.c else args.save_model_to
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    
    train_model = model.FCN(args.feature_num).cuda() if args.model == "FCN" else model.FCNwPool(args.feature_num, args.pix_res).cuda()
    if args.load_model:
        train_model.load_state_dict(th.load(args.load_model).state_dict())
    print("model is initialized ...")
    optimizer = to.Adam(train_model.parameters(), lr = args.lr, weight_decay = args.decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=args.patience)
    criterion = nn.BCEWithLogitsLoss(pos_weight=th.Tensor([20]).cuda())
    # criterion = nn.BCEWithLogitsLoss()

    loss_100, iter_cnt = 0, 0
    for epoch in range(args.n_epochs):
        running_loss, ignore_cnt, epoch_iter_cnt = 0, 0, 0
        train_loader_iter = iter(train_loader)
        for _ in range(len(train_loader_iter)):
            optimizer.zero_grad()
            batch_sample = train_loader_iter.next()
            data, gt, ignore = filter_batch(batch_sample)
            ignore_cnt += ignore
            if not data:
                continue
            # import ipdb; ipdb.set_trace()
            prds = train_model.forward(th.stack(data).cuda())
            loss = criterion(
                prds[:, :, args.pad:-args.pad, args.pad:-args.pad].view(-1, 1),
                th.stack(gt).cuda()
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
        print('--- epoch %d ---: %f of training data is ignored.' %(epoch, ignore_cnt/(len(train_loader_iter)*args.batch_size)))
        if (epoch+1) % args.s == 0:
            th.save(train_model, dir_name+'/'+str(epoch)+'_'+exe_time+'.pt')
        v_loss = validate(args, train_model, test_loader)
        scheduler.step(v_loss)
        writer.add_scalars(
            "loss/grouped",
            {'test': v_loss, 'train': running_loss/epoch_iter_cnt},
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