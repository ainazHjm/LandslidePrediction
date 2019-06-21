import model
import numpy as np
import torch as th
import torch.optim as to
import torch.nn as nn
import os
import torch.nn.functional as F
from time import ctime
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.plot import save_config
# pylint: disable=E1101,E0401,E1123

def validate(args, model, test_loader):
    with th.no_grad():
        # criterion = nn.BCEWithLogitsLoss(pos_weight=th.Tensor([20]).cuda())
        criterion = nn.BCEWithLogitsLoss()

        test_loader_iter = iter(test_loader)
        ignore_idx, running_loss, cnt = 0, 0, 0
        for _ in range(len(test_loader_iter)):
            batch_sample = test_loader_iter.next()
            (_, _, h, w) = batch_sample['data'].shape
            data = batch_sample['data'].cuda()
            gt = batch_sample['gt'].cuda()
            prds = model.forward(data)
            loss = criterion(
                prds[:, :, h//2, w//2].view(-1, 1),
                gt)
            running_loss += loss.item()
            cnt += 1
       return running_loss/cnt

def train(args, train_loader, test_loader):
    '''
    For the pixel wise prediction, the window size (ws) should be 1 and the padding size (pad) should be 32
    so that the input to the model is 65x65.
    '''
    writer = SummaryWriter()
    sig = nn.Sigmoid()

    exe_time = ctime().replace("  "," ").replace(" ", "_").replace(":","_")
    dir_name = args.save_model_to + args.region +'/'+ exe_time if args.c else args.save_model_to
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    
    if args.model == "FCN":
        train_model = model.FCN((args.feature_num, args.ws+args.pad*2, args.ws+args.pad*2)).cuda()
    elif args.model == 'FCNwPool':
        train_model = model.FCNwPool((args.feature_num, 1+2*args.pad, 1+2*args.pad), args.pix_res).cuda()
    else:
        train_model = model.UNET((args.feature_num, 1+2*args.pad, 1+2*args.pad)).cuda()
    
    if args.load_model:
        train_model.load_state_dict(th.load(args.load_model).state_dict())
    print("model is initialized ...")

    if th.cuda.device_count() > 1:
        train_model = nn.DataParallel(train_model)
    
    optimizer = to.Adam(train_model.parameters(), lr = args.lr, weight_decay = args.decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=args.patience, verbose=True)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=th.Tensor([20]).cuda())
    criterion = nn.BCEWithLogitsLoss()

    loss_100, iter_cnt = 0, 0
    for epoch in range(args.n_epochs):
        running_loss = 0
        train_loader_iter = iter(train_loader)
        ignore_idx, epoch_iters = 0, 0
        for _ in range(len(train_loader_iter)):
            optimizer.zero_grad()
            batch_sample = train_loader_iter.next()
            (_, _, h, w) = batch_sample['data'].shape
            data = batch_sample['data'].cuda()
            gt = batch_sample['gt'].cuda()
            prds = train_model.forward(data)
            loss = criterion(
                prds[:, :, h//2, w//2].view(-1, 1),
                gt
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
            del prds, gt, data

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
