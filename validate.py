import torch as th
import torch.nn as nn
from torchvision.utils import save_image
from loader import RotDataset
from time import ctime
from sacred import Experiment

def validate(params, test_dataset, _log):
    sig = nn.Sigmoid()
    shape = params['shape']
    r, c = shape[0]//params['ws'], shape[1]//params['ws']
    res = th.zeros(shape)
    pad = params['pad']
    model = th.load(params['load_model'])
    _log.info('[{}] model is successfully loaded.'.format(ctiem()))
    for idx_ in range(len(test_dataset)):
        data, gt = test_dataset[idx_]['data'].cuda().unqueeze(0), test_dataset[idx_]['gt'].cuda()
        ignore = gt < 0
        prds = model.forward(data)[:, :, pad:-pad, pad:-pad]
        prds[ignore] = 0
        row, col = idx_//c, idx_-(idx_//c)*c
        res[row:row+params['ws'], col:col+params['ws']] = prds.squeeze(0)
        _log.info('[{}]: writing [{}/{}]'.format(ctime(), idx_, len(test_dataset)))
    save_image(res, params['save_to']+'predictions.tif')

ex = Experiment('validate_rotation')

@ex.config
def ex_cfg():
    params = {
        'data_path': '/dev/shm/rotated_dataset.h5',
        'load_model': '',
        'save_to': 'runs/result/'
        'region': 'Veneto',
        'ws': 200,
        'pad': 64,
        'shape': (4201, 19250)
    }

@ex.automain
def main(params):
    vd = RotDataset(params['data_path'], params['region'], 'test')
    _log.info('[{}] prepared the dataset for validation.'.format(ctime()))
    validate(params, vd, _log)