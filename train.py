import model
import numpy as np
import torch as th
import torch.optim as to
import torch.nn as nn
import os
import scipy.ndimage
from time import ctime
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils import shuffle
from utils.plot import save_config
# pylint: disable=E1101,E0401,E1123

def find_greater_slope(args, data, distance):
    (_, h, w) = data.shape
    center = (h//2, w//2)
    pix_distance = distance//args.pix_res
    if data[0, center[0], center[1]] < data[0, center[0]+pix_distance, center[1]] or data[0, center[0], center[1]] < data[0, center[0]-pix_distance, center[1]]:
        return True
    return False

def find_direction(args, batch):
    import scipy.ndimage as snd
    angle = np.arange(45, 360, 45)
    best_angles = []
    for i in range(batch['data'].shape[0]):
        best_theta = -1
        for theta in angle:
            rot_img = snd.rotate(batch['data'][i].numpy(), theta, reshape=True)
            if find_greater_slope(args, rot_img, 320):
                best_theta = theta
                break
        if best_theta!=-1:
            best_angles.append(best_theta)
        else:
            best_angles.append(0)
    return best_angles

def process_batch(args, batch):
    import scipy.ndimage as snd
    angles = find_direction(args, batch)
    for idx, e in enumerate(angles):
        if e != 0:
            batch['data'][idx] = snd.rotate(batch['data'][idx].numpy(), e, reshape=False)
            batch['gt'][idx] = snd.rotate(batch['gt'][idx].numpy(), e, reshape=True)
    return batch

def validate(args, model, test_loader):
    with th.no_grad():
        criterion = nn.BCEWithLogitsLoss(pos_weight=th.Tensor([20]).cuda())
        # criterion = nn.BCEWithLogitsLoss()
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
    
    train_model = model.FCN((args.feature_num, args.ws+args.pad*2, args.ws+args.pad*2)).cuda() if args.model == "FCN" else model.FCNwPool((args.feature_num, args.ws+2*args.pad, args.ws+2*args.pad), args.pix_res).cuda()
    if args.load_model:
        train_model.load_state_dict(th.load(args.load_model).state_dict())
    print("model is initialized ...")
    optimizer = to.Adam(train_model.parameters(), lr = args.lr, weight_decay = args.decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=args.patience)
    criterion = nn.BCEWithLogitsLoss(pos_weight=th.Tensor([20]).cuda())
    # criterion = nn.BCEWithLogitsLoss()

    loss_100 = 0
    for epoch in range(args.n_epochs):
        running_loss = 0
        train_loader_iter = iter(train_loader)
        for batch_idx in range(len(train_loader_iter)):
            optimizer.zero_grad()
            batch_sample = train_loader_iter.next()
            batch_sample = process_batch(args, batch_sample)
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

            writer.add_scalar("loss/train_iter", loss.item(), epoch*len(train_loader_iter)+batch_idx+1)
            writer.add_scalars(
                "probRange",
                {'min': th.min(sig(prds)), 'max': th.max(sig(prds))},
                epoch*len(train_loader_iter)+batch_idx+1
            )
            if (epoch*len(train_loader_iter)+batch_idx+1) % 100 == 0:
                writer.add_scalar("loss/train_100", loss_100/100, epoch*len(train_loader_iter)+batch_idx+1)
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