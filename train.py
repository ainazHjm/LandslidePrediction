import model
import torch as th
import torch.optim as to
import torch.nn as nn
import os
from time import ctime
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.plot import save_config
from unet import UNet
# pylint: disable=E1101,E0401,E1123

def create_dir(dir_name):
    model_dir = dir_name+'/model/'
    res_dir = dir_name+'/result/'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    return model_dir, res_dir

def validate(model, val_loader, data_param, train_param, _log):
    with th.no_grad():
        criterion = nn.BCEWithLogitsLoss(pos_weight=th.Tensor([train_param['pos_weight']]).cuda())
        val_iter = iter(val_loader)
        running_loss = 0
        prune = data_param['prune']
        for _ in range(len(val_iter)):
            batch_sample = val_iter.next()
            data = batch_sample['data'].cuda()
            gt = batch_sample['gt'].cuda()
            prds = model.forward(data)[:, :, prune:-prune, prune:-prune]
            indices = gt>=0
            loss = criterion(prds[indices], gt[indices])
            running_loss += loss.item()
        del data, gt, prds, indices
        return running_loss/len(val_iter)

def train(train_loader, val_loader, train_param, data_param, loc_param, _log, _run):
    writer = SummaryWriter()
    model_dir, _ = create_dir(writer.file_writer.get_logdir())
    sig = nn.Sigmoid()

    if train_param['model'] == "FCN":
        train_model = model.FCN(data_param['feature_num']).cuda()
    elif train_param['model'] == 'FCNwPool':
        train_model = model.FCNwPool(data_param['feature_num'], data_param['pix_res']).cuda()
    elif train_param['model'] == 'UNet':
        train_model = UNet(data_param['feature_num'], 1).cuda()
    elif train_param['model'] == 'FCNwBottleneck':
        train_model = model.FCNwBottleneck(data_param['feature_num'], data_param['pix_res']).cuda()
    elif train_param['model'] == 'SimplerFCNwBottleneck':
        train_model = model.SimplerFCNwBottleneck(data_param['feature_num']).cuda()
    elif train_param['model'] == 'Logistic':
        train_model = model.Logistic(data_param['feature_num']).cuda()
    elif train_param['model'] == 'PolyLogistic':
        train_model = model.PolyLogistic(data_param['feature_num']).cuda()
    
    if th.cuda.device_count() > 1:
        train_model = nn.DataParallel(train_model)
    
    if loc_param['load_model']:
        train_model.load_state_dict(th.load(loc_param['load_model']))
    _log.info('[{}] model is initialized ...'.format(ctime()))
    
    if train_param['optim'] == 'Adam':
        optimizer = to.Adam(train_model.parameters(), lr=train_param['lr'], weight_decay=train_param['decay'])
    else:
        optimizer = to.SGD(train_model.parameters(), lr=train_param['lr'], weight_decay=train_param['decay'])
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=train_param['patience'], verbose=True, factor=0.5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=th.Tensor([train_param['pos_weight']]).cuda())

    valatZero = validate(train_model, val_loader, data_param, train_param, _log)
    _log.info('[{}] validation loss before training: {}'.format(ctime(), valatZero))
    _run.log_scalar('training.val_loss', valatZero, 0)
    trainatZero = validate(train_model, train_loader, data_param, train_param, _log)
    _log.info('[{}] train loss before training: {}'.format(ctime(), trainatZero))
    _run.log_scalar('training.loss_epoch', trainatZero, 0)
    
    loss_ = 0
    prune = data_param['prune']
    for epoch in range(train_param['n_epochs']):
        running_loss = 0
        train_iter = iter(train_loader)
        for iter_ in range(len(train_iter)):
            optimizer.zero_grad()

            batch_sample = train_iter.next()
            data, gt = batch_sample['data'].cuda(), batch_sample['gt'].cuda()
            if train_param['model'] == 'UNET':
                prds = train_model(data)[:, :, prune:-prune, prune:-prune]
            else:
                prds = train_model.forward(data)[:, :, prune:-prune, prune:-prune]
            indices = gt>=0
            loss = criterion(prds[indices], gt[indices])
            running_loss += loss.item()
            loss_ += loss.item()
            loss.backward()
            optimizer.step()

            _run.log_scalar("training.loss_iter", loss.item(), epoch*len(train_iter)+iter_+1)
            _run.log_scalar("training.max_prob", th.max(sig(prds)).item(), epoch*len(train_iter)+iter_+1)
            _run.log_scalar("training.min_prob", th.min(sig(prds)).item(), epoch*len(train_iter)+iter_+1)

            writer.add_scalar("loss/train_iter", loss.item(), epoch*len(train_iter)+iter_+1)
            writer.add_scalars(
                "probRange",
                {'min': th.min(sig(prds)), 'max': th.max(sig(prds))},
                epoch*len(train_iter)+iter_+1
            )
            if (epoch*len(train_iter)+iter_+1) % 20 == 0:
                _run.log_scalar("training.loss_20", loss_/20, epoch*len(train_iter)+iter_+1)
                writer.add_scalar("loss/train_20", loss_/20, epoch*len(train_iter)+iter_+1)
                _log.info(
                    '[{}] loss at [{}/{}]: {}'.format(
                        ctime(),
                        epoch*len(train_iter)+iter_+1,
                        train_param['n_epochs']*len(train_iter),
                        loss_/20
                    )
                )
                loss_ = 0
        
        v_loss = validate(train_model, val_loader, data_param, train_param, _log)
        scheduler.step(v_loss)
        _log.info('[{}] validation loss at [{}/{}]: {}'.format(ctime(), epoch+1, train_param['n_epochs'], v_loss))
        _run.log_scalar('training.val_loss', v_loss, epoch+1)
        _run.log_scalar('training.loss_epoch', running_loss/len(train_iter), epoch+1)
        writer.add_scalars(
            "loss/grouped",
            {'test': v_loss, 'train': running_loss/len(train_iter)},
            epoch+1
        )
        del data, gt, prds, indices
        if (epoch+1) % loc_param['save'] == 0:
            th.save(train_model.cpu().state_dict(), model_dir+'model_{}.pt'.format(str(epoch+1)))
            train_model = train_model.cuda()
    
    writer.export_scalars_to_json(model_dir+'loss.json')
    th.save(train_model.cpu().state_dict(), model_dir+'trained_model.pt')
    save_config(writer.file_writer.get_logdir()+'/config.txt', train_param, data_param)
    _log.info('[{}] model has been trained and config file has been saved.'.format(ctime()))
    
    return v_loss
