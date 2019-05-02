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
# pylint: disable=E1101,E0401,E1123

# def find_max_shape(path):
#     img = np.load(path)
#     img = np.asarray(img, dtype='float32')
#     max_h, max_w = 0, 0
#     for angle in np.arange(0, 360, 36):
#         rotated = scipy.ndimage.rotate(img, angle)
#         (h, w) = rotated.shape
#         if h > max_h:
#             max_h = h
#         if w > max_w:
#             max_w = w
#     return (max_h, max_w)

def load_data(args, fname, angle, feature_num=21):
    dp = args.data_path
    data_dir = 'data_'+str(args.pad*2)+'/'
    label_dir = 'gt_200/'
    data = []
    label = []
    for name in fname:
        features = []
        for i in range(feature_num):
            img = np.load(dp+data_dir+str(i)+'_'+name)
            img = np.asarray(img, dtype='float32')
            rotated = scipy.ndimage.rotate(img, angle)
            # hdif = h - rotated.shape[0]
            # wdif = w - rotated.shape[1]
            # if hdif< 0 or wdif < 0:
            #     raise ValueError
            # features.append(np.pad(rotated, ((hdif//2, hdif-hdif//2), (wdif//2, wdif-wdif//2)), 'constant')) # 2d shape
            hdif = rotated.shape[0] - img.shape[0] - args.pad
            wdif = rotated.shape[1] - img.shape[1] - args.pad
            features.append(rotated[hdif//2:-(hdif-hdif//2), wdif//2:-(wdif-wdif//2)]) # 2d shape
        features = np.asarray(features)
        data.append(features)
        gt = np.load(dp+label_dir+name)
        rotated = scipy.ndimage.rotate(gt, angle)
        # gt_hdif = gt_h - rotated.shape[0]
        # gt_wdif = gt_w - rotated.shape[1]
        # if gt_hdif < 0 or gt_wdif < 0:
        #     raise ValueError
        # gt = np.pad(rotated, ((gt_hdif//2, gt_hdif-gt_hdif//2), (gt_wdif//2, gt_wdif-gt_wdif//2)), 'constant')
        gt_hdif = rotated.shape[0] - gt.shape[0]
        gt_wdif = rotated.shape[1] - gt.shape[1]
        gt = rotated[gt_hdif//2:-(gt_hdif-gt_hdif//2), gt_wdif//2:-(gt_wdif-gt_wdif//2)]
        label.append(gt.reshape(1, gt.shape[0], gt.shape[1]))
    return np.asarray(data), np.asarray(label) #4d shape

def validate(args, model):
    with th.no_grad():
        valIdx = np.load(args.data_path+'vdIdx.npy')
        np.random.shuffle(valIdx)
        criterion = nn.BCEWithLogitsLoss(pos_weight=th.Tensor([1000]).cuda())
        # criterion = nn.BCEWithLogitsLoss()
        # hdif = max_shape[0] - 200
        # wdif = max_shape[1] - 200
        running_loss = 0
        bs = args.batch_size
        #bs = 1
        num_iters = valIdx.shape[0]//bs
        for i in range(num_iters):
            # in_d, gt = load_data(args, valIdx[i*bs:(i+1)*bs]) if i < num_iters else load_data(args, valIdx[i*bs:])
            in_d, gt = load_data(args, valIdx[i*bs:(i+1)*bs], 0)
            in_d, gt = th.tensor(in_d).cuda(), th.tensor(gt).float().cuda()
            # print(in_d.shape, gt.shape, valIdx[i])
            prds = model.forward(in_d)
            # loss = criterion(prds[:, :, hdif//2:hdif//2+200, wdif//2:wdif//2+200].view(-1, 1, 200, 200), gt)
            loss = criterion(prds[:, :, args.pad:-args.pad, args.pad:-args.pad].view(-1, 1, gt.shape[2], gt.shape[3]), gt)
            running_loss += loss.item()
        return running_loss/num_iters

def train(args):
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
    exe_time = ctime().replace("  "," ").replace(" ", "_").replace(":","_")
    dir_name = args.save_model_to + exe_time if args.c else args.save_model_to
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    train_data = np.load(args.data_path+'tdIdx.npy')
    # data_max_shape = find_max_shape(args.data_path+'data_'+str(2*args.pad)+'/0_2_3.npy')
    # gt_max_shape = (data_max_shape[0]-2*args.pad, data_max_shape[1]-2*args.pad)

    train_model = model.FCN().cuda() if args.model == "FCN" else model.FCNwPool((22-1, 200+args.pad*2, 200+args.pad*2), args.pix_res).cuda()
    if args.load_model:
        train_model.load_state_dict(th.load(args.load_model).state_dict())
    print("model is initialized ...")
    optimizer = to.Adam(train_model.parameters(), lr = args.lr, weight_decay = args.decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=th.Tensor([1000]).cuda())
    # criterion = nn.BCEWithLogitsLoss()

    bs = args.batch_size
    num_iters = train_data.shape[0]//bs
    print(num_iters)
    running_loss, loss_100, loss_20 = 0, 0, 0

    for i in range(args.n_epochs):
        np.random.shuffle(train_data)

        for j in range(num_iters):
            for k, angle in enumerate(np.arange(0, 360, 36)):
                optimizer.zero_grad()
                in_d, gt = load_data(args, train_data[j*bs:(j+1)*bs], angle)
                in_d, gt = th.tensor(in_d).cuda(), th.tensor(gt).float().cuda()
                prds = train_model.forward(in_d)
                loss = criterion(prds[:, :, args.pad:-args.pad, args.pad:-args.pad].view(-1, 1, gt.shape[2], gt.shape[3]), gt)
                loss.backward()
                optimizer.step()

                writer.add_scalar("loss/train_iter", loss.item(), i*num_iters+j)
                running_loss += loss.item()
                loss_100 += loss.item()
                loss_20 += loss.item

                if (i*num_iters*10+j*10+k+1) % 20 == 0:
                    print("average training loss after 20 iters: %f" % loss_20/20)
                    writer.add_scalar("loss/train_20", loss_20/20, i*num_iters*10+j*10+k+1)
                    loss_20 = 0

                if (i*num_iters*10+j*10+k+1) % 100 == 0:
                    writer.add_scalar("loss/train_100", loss_100/100, i*num_iters*10+j*10+k+1)
                    loss_100 = 0

        del in_d, gt, prds
        v_loss = validate(args, train_model)
        scheduler.step(v_loss)
        print("--- validation loss: %f" % v_loss)
        writer.add_scalars("loss/grouped", {'validation': v_loss, 'train': running_loss/(num_iters*10)}, i)
        writer.add_scalar("loss/validation", v_loss, i)
        writer.add_scalar("loss/train", running_loss/(num_iters*10), i)
        running_loss = 0
        for name, param in train_model.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), i*num_iters+j)

        if (i+1) % args.s == 0:
            th.save(train_model, dir_name+'/'+str(i)+'_'+exe_time+'.pt')
        
    writer.export_scalars_to_json(dir_name+'/loss.json')    
    th.save(train_model, dir_name+'/final@'+str(i)+'_'+exe_time+'.pt')
    print("model has been trained and saved.")

# def cross_validate(args, data):
#     '''
#     TODO: complete this function
#     '''
#     (_, _, w) = data.shape
#     div = w//5
#     min_loss = th.inf
#     idx = -1
#     for val_idx in range(0, 5):
#         val_data = data[:, :, val_idx*div:(val_idx+1)*div]
#         train_data = th.cat((data[:, :, 0:val_idx*div], data[:, :, (val_idx+1)*div:]), 2)
#         loss = train(args, val_data).item() 
#         if loss[-1] < min_loss:
#             min_loss = loss[-1]
#             idx = val_idx
#     return th.cat((data[:, :, 0:idx*div], data[:, :, (idx+1)*div:]), 2), data[:, :, idx*div:(idx+1)*div], idx
