# pylint: disable=E0611
import train
import argparse
import torch as th
import numpy as np
from utils.args import str2bool, __range
from utils.plot import save_results
from loader import LandslideDataset, LandslideTrainDataset
from torch.utils.data import DataLoader

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
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--ws", type=int, default=200)
    parser.add_argument("--s", type=int, default=5) #save the model at how many epochs
    parser.add_argument("--c", type=str2bool, default=True)
    parser.add_argument("--pad", type=int, default=64)
    parser.add_argument("--feature_num", type=int, default=94)
    parser.add_argument("--oversample_pts", action='append', type=__range)
    parser.add_argument("--save_res_to", type=str, default='../output/CNN/')
    return parser.parse_args()

# def oversample(args, dataset):
#     data_batch_len = len(dataset)//10
#     loader = DataLoader(dataset, batch_size=data_batch_len, num_workers=4)
#     loader_iter = iter(loader)
#     oversample_idx = []
#     for i in range(len(loader_iter)):
#         samples = loader_iter.next()
#         (b, _, h, w) = samples['gt'].shape # _ should be 1
#         indices = th.sum(samples['gt'].view(b, -1), 1) > 0
#         th.nonzero(indices)

def main():
    args = get_args()
    args.oversample_pts = np.asarray(args.oversample_pts).reshape(-1, 4)
    print(args.oversample_pts)
    trainData = LandslideTrainDataset(
        args.data_path,
        args.region,
        args.stride,
        args.ws,
        args.oversample_pts,
        args.pad
    )
    testData = LandslideDataset(
        args.data_path,
        args.region,
        args.ws,
        args.pad
    )
    import ipdb; ipdb.set_trace()
    train_loader = DataLoader(
        trainData,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        testData,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    if args.validate:
        print("loading a trained model...", end='\r')
        model = th.load(args.load_model)
        print("validating the model on validation data ...", end='\r')
        save_results(args, model, 'validation')
        print("validating the model on training data ...", end='\r')
        save_results(args, model, 'train')
        print("model is validated and the results are saved.")      
    else:
        print("starting to train ...")
        train.train(args, train_loader, test_loader)

if __name__ == "__main__":
    main()
