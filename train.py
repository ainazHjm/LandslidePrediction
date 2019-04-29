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
# pylint: disable=E1101,E0401,E1123

def load_data(args, fname, feature_num=21):
    dp = args.data_path
    data_dir = 'data_600/'
    label_dir = 'gt_200/'
    data = []
    label = []
    for name in fname:
        features = []
        for i in range(feature_num):
            features.append(np.load(dp+data_dir+str(i)+'_'+name)) # 2d shape
        features = np.asarray(features)
        data.append(features)
        gt = np.load(dp+label_dir+name)
        label.append(gt.reshape(1, gt.shape[0], gt.shape[1]))
    return np.asarray(data), np.asarray(label) #4d shape

def validate(args, model, valIdx):
    with th.no_grad():
        criterion = nn.BCEWithLogitsLoss(pos_weight=th.Tensor([1000]).cuda())
        # criterion = nn.BCEWithLogitsLoss()
        running_loss = 0
        bs = args.batch_size
        #bs = 1
        num_iters = valIdx.shape[0]//bs
        for i in range(num_iters):
            # in_d, gt = load_data(args, valIdx[i*bs:(i+1)*bs]) if i < num_iters else load_data(args, valIdx[i*bs:])
            in_d, gt = load_data(args, valIdx[i*bs:(i+1)*bs])
            in_d, gt = th.tensor(in_d).cuda(), th.tensor(gt).float().cuda()
            # print(in_d.shape, gt.shape, valIdx[i])
            prds = model.forward(in_d)
            loss = criterion(prds[:, :, 200:400, 200:400].view(-1, 1, 200, 200), gt)
            running_loss += loss.item()
        return running_loss/num_iters

def train(args, train_data, val_data):
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
    # th.cuda.empty_cache()
    exe_time = ctime().replace("  "," ").replace(" ", "_").replace(":","_")
    dir_name = args.save_model_to + exe_time if args.c else args.save_model_to
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    # train_data, train_label = make_patches(train_data_path)
    train_model = model.FCN().cuda() if args.model == "FCN" else model.FCNwPool((22-1, 600, 600), args.pix_res).cuda()
    if args.load_model:
        train_model.load_state_dict(th.load(args.load_model).state_dict())
    print("model is initialized ...")

    optimizer = to.Adam(train_model.parameters(), lr = args.lr, weight_decay = args.decay)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2)
    criterion = nn.BCEWithLogitsLoss(pos_weight=th.Tensor([1000]).cuda())
    # criterion = nn.BCEWithLogitsLoss()

    bs = args.batch_size
    num_iters = train_data.shape[0]//bs
    print(num_iters)
    running_loss = 0
    
    for i in range(args.n_epochs):
        # running_loss = 0
        # shuffling the train and validation indices
        np.random.shuffle(train_data)
        np.random.shuffle(val_data)

        # if (i+1) % args.s == 0:
        #    th.save(train_model, dir_name+'/'+str(i)+'_'+exe_time+'.pt')

        for j in range(num_iters):
            optimizer.zero_grad()
            in_d, gt = load_data(args, train_data[j*bs:(j+1)*bs])
            # print(in_d.shape, gt.shape)
            in_d, gt = th.tensor(in_d).cuda(), th.tensor(gt).float().cuda()
            # print(in_d.shape, gt.shape)
            prds = train_model.forward(in_d)
            # print(prds.shape)
            loss = criterion(prds[:, :, 200:400, 200:400].view(-1, 1, 200, 200), gt)
            if (i*num_iters+j+1) % 20 == 0:
                writer.add_scalar("loss/train@20", loss.item(), i*num_iters+j)
                print("%d,%d >> loss: %f" % (i, j, loss.item()))
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            if (i*num_iters+j+1) % 1000 == 0:
                del in_d, gt, prds
                v_loss = validate(args, train_model, val_data)
                # scheduler.step(v_loss)
                print("--- validation loss: %f" % v_loss)
                writer.add_scalars("loss/grouped", {'validation': v_loss, 'train': running_loss/1000}, i*num_iters+j)
                writer.add_scalar("loss/validation", v_loss, i*num_iters+j)
                writer.add_scalar("loss/train", running_loss/1000, i*num_iters+j)
                running_loss = 0

                for name, param in train_model.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), i*num_iters+j)
                
                #tidx = np.random.choice(train_data.shape[0], 1, replace=False)
                #vidx = np.random.choice(val_data.shape[0], 1, replace=False)
                #print(tidx, vidx)
                #t_img, t_gt = load_data(args, train_data[tidx])
                #v_img, v_gt = load_data(args, val_data[vidx])
                #t_prd = train_model.forward(th.tensor(t_img).cuda())
                #v_prd = train_model.forward(th.tensor(v_img).cuda())
                #writer.add_image('GroundTruth/train', gt[0,0,:,:], i*num_iters+j)
                #writer.add_image('GroundTruth/validation', v_gt[0,0,:,:], i*num_iters+j)
                #writer.add_image('Prediction/train', t_prd[0,0,200:400,200:400], i*num_iters+j)
                #writer.add_image('Prediction/validation', v_prd[0,0,200:400,200:400], i*num_iters+j)
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
