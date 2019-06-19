# pylint: disable=E0611
import train
import argparse
import torch as th
import numpy as np
import utils.plot as up
from utils.args import str2bool, __range
from loader import LandslideDataset, LandslideTrainDataset, create_oversample_data, RotationDataset
from torch.utils.data import DataLoader
from adjust_rotation import adjust_rot
from time import ctime

def custom_collate_fn(batch):
    in_data = th.stack([e['data'] for e in batch], 0)
    label = th.stack([e['gt'] for e in batch], 0)
    return {'data': in_data, 'gt': label}

def get_args():
    parser = argparse.ArgumentParser(description="Training a CNN-Classifier for landslide prediction")
    # parser.add_argument("--cross_validation", type=str2bool, default=False)
    parser.add_argument("--model", type=str, default="FCNwPool")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--decay", type=float, default=1e-5)
    parser.add_argument("--load_model", type=str, default='')
    parser.add_argument("--validate", type=str2bool, default=False)
    parser.add_argument("--data_path", type=str, default="../image_data/landslide.h5")
    parser.add_argument("--save_model_to", type=str, default="../models/CNN/")
    parser.add_argument("--region", type=str, default='Veneto')
    parser.add_argument("--pix_res", type=int, default=10)
    parser.add_argument("--stride", type=int, default=200)
    parser.add_argument("--ws", type=int, default=200)
    parser.add_argument("--s", type=int, default=5) #save the model at how many epochs
    parser.add_argument("--c", type=str2bool, default=True)
    parser.add_argument("--pad", type=int, default=32)
    parser.add_argument("--feature_num", type=int, default=94)
    parser.add_argument("--oversample_pts", action='append', type=__range)
    parser.add_argument("--save_res_to", type=str, default='../output/CNN/')
    parser.add_argument("--oversample", type=str2bool, default=False)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--random_sample", type=str2bool, default=True)
    parser.add_argument("--pos_indices", type=str, default='')
    parser.add_argument('--write_rotation', type=str2bool, default=False)
    return parser.parse_args()

def main():
    args = get_args()

    if args.write_rotation:
        train_rot_path = adjust_rot(args, 'train')
        print('%s: found the rotation for training data.' %ctime())
        test_rot_path = adjust_rot(args, 'test')
        print('%s: found the rotation for test data.' %ctime())
    else:
        train_rot_path = '/'.join(args.data_path.split('/')[:-1])+'/train_rot.npy'
        test_rot_path = '/'.join(args.data_path.split('/')[:-1])+'/test_rot.npy'
    
    # trainData = RotationDataset(
        # args.data_path,
        # train_rot_path,
        # args.region,
        # args.feature_num,
        # 'train',
        # args.pad #should be 32
    # )
    # testData = RotationDataset(
        # args.data_path,
        # test_rot_path,
        # args.region,
        # args.feature_num,
        # 'test',
        # args.pad
    # )
    # test_loader = DataLoader(
        # testData,
        # batch_size=args.batch_size,
        # shuffle=False,
        # num_workers=args.num_workers
    # )
    # train_loader = DataLoader(
        # trainData,
        # batch_size=args.batch_size,
        # shuffle=True,
        # num_workers=args.num_workers
    # )
    # print('%s: created the train and test datasets.' %ctime())
# 
    # if args.validate:
        # print("%s: loading a trained model..." %ctime())
        # model = th.load(args.load_model)
        # if args.random_sample:
            # up.validate_on_ones(args, model, testData)
            # print('%s: the model is validated on random samples.' %ctime())
        # else:
            # up.validate_all(args, model, test_loader)
            # print('%s: the model is validated on test dataset.' %ctime())
    # else:
        # print("%s: starting to train ..." %ctime())
        # train.train(args, train_loader, test_loader)
        # print('%s: training is finished.' %ctime())

if __name__ == "__main__":
    main()
