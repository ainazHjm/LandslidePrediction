import model
import torch as th
import torch.optim as to
import torch.nn as nn
import os
from time import ctime
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.plot import save_config
# pylint: disable=E1101,E0401,E1123

def create_dir(dir_name):
    model_dir = dir_name+'/model/'
    res_dir = dir_name+'/result/'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    return model_dir, res_dir

def validate(model, test_loader, data_param, train_param):
    with th.no_grad():
        criterion = nn.BCEWithLogitsLoss(pos_weight=th.Tensor([train_param['pos_weight']]).cuda())
        test_iter = iter(test_loader)
        running_loss = 0
        for _ in range(len(test_iter)):
            batch_sample = test_iter.next()
            (_, _, h, w) = batch_sample['data'].shape
            data = batch_sample['data'].cuda()
            gt = batch_sample['gt'].cuda()
            prds = model.forward(data)
            loss = criterion(
                prds[:, :, h//2, w//2].view(-1, 1),
                gt)
            # import ipdb; ipdb.set_trace()
            running_loss += loss.item()
        return running_loss/len(test_iter)

def train(train_loader, test_loader, train_param, data_param, loc_param, _log):
    writer = SummaryWriter()
    model_dir, _ = create_dir(writer.file_writer.get_logdir())
    sig = nn.Sigmoid()

    if train_param['model'] == "FCN":
        train_model = model.FCN(data_param['feature_num']).cuda()
    elif train_param['model'] == 'FCNwPool':
        train_model = model.FCNwPool(data_param['feature_num'], data_param['pix_res']).cuda()
    else:
        train_model = model.UNET(data_param['feature_num']).cuda()
    
    if loc_param['load_model']:
        train_model.load_state_dict(th.load(loc_param['load_model']).state_dict())
    _log.info('[{}] model is initialized ...'.format(ctime()))

    if th.cuda.device_count() > 1:
        train_model = nn.DataParallel(train_model)
    
    optimizer = to.Adam(train_model.parameters(), lr=train_param['lr'], weight_decay=train_param['decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=train_param['patience'], verbose=True)
    criterion = nn.BCEWithLogitsLoss(pos_weight=th.Tensor([train_param['pos_weight']]).cuda())

    loss_100 = 0
    for epoch in range(train_param['n_epochs']):
        running_loss = 0
        train_iter = iter(train_loader)
        for iter_ in range(len(train_iter)):
            optimizer.zero_grad()

            batch_sample = train_iter.next()
            (_, _, h, w) = batch_sample['data'].shape
            data, gt = batch_sample['data'].cuda(), batch_sample['gt'].cuda()
            prds = train_model.forward(data)
            loss = criterion(prds[:, :, h//2, w//2].view(-1, 1), gt)
            running_loss += loss.item()
            loss_100 += loss.item()
            
            loss.backward()
            optimizer.step()

            writer.add_scalar("loss/train_iter", loss.item(), epoch*len(train_iter)+iter_+1)
            writer.add_scalars(
                "probRange",
                {'min': th.min(sig(prds)), 'max': th.max(sig(prds))},
                epoch*len(train_iter)+iter_+1
            )
            if (epoch*len(train_iter)+iter_+1) % 100 == 0:
                writer.add_scalar("loss/train_100", loss_100/100, epoch*len(train_iter)+iter_+1)
                _log.info(
                    '[{}] loss at [{}/{}]: {}'.format(
                        ctime(),
                        epoch*len(train_iter)+iter_+1,
                        train_param['n_epochs']*len(train_iter),
                        loss_100/100
                    )
                )
                loss_100 = 0
            del prds, gt, data

        if (epoch+1) % loc_param['save'] == 0:
            th.save(train_model, model_dir+'model_{}.pt'.format(str(epoch+1)))
        
        v_loss = validate(train_model, test_loader, data_param, train_param)
        scheduler.step(v_loss)
        _log.info('[{}] validation loss at [{}/{}]: {}'.format(ctime(), epoch+1, train_param['n_epochs'], v_loss))
        writer.add_scalars(
            "loss/grouped",
            {'test': v_loss, 'train': running_loss/len(train_iter)},
            epoch
        )

    writer.export_scalars_to_json(model_dir+'loss.json')
    th.save(train_model, model_dir+'trained_model.pt')
    save_config(writer.file_writer.get_logdir()+'/config.txt', train_param, data_param)
    _log.info('[{}] model has been trained and config file has been saved.'.format(ctime()))
