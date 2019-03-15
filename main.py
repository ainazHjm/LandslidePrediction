# pylint: disable=E0611
from data import process, normalize
from train import train, cross_validate
import argparse
import torch as th
import numpy as np
from utils.args import str2bool
from utils.plot import save_results

def get_args():
    parser = argparse.ArgumentParser(description="Training a CNN-Classifier for landslide prediction")
    parser.add_argument("--cross_validation", type=str2bool, default=False)
    parser.add_argument("--model", type=str, default="FCN")
    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--decay", type=float, default=1e-5)
    parser.add_argument("--load_model_path", type=str, default='')
    parser.add_argument("--validate", type=str2bool, default=False)
    return parser.parse_args()

def concat_data(val_idx, val_data, train_data=th.load("../image_data/data/Veneto/train_data.pt")):
    s = train_data.shape[2]//4
    # data = th.zeros(train_data.shape[0], train_data.shape[1], train_data.shape[2]+val_data.shape[2])
    # data[]
    d = th.cat(
            (
                th.cat(
                    (train_data[:, :, 0:val_idx*s], val_data),
                    dim=2
                ),
                train_data[:, :, val_idx*s:]
            ),
            dim=2
        )
    return d

def main():
    args = get_args()
    if args.cross_validation:
        data = process()
        train_data, val_data, val_idx = cross_validate(args, data)
    else:
        # the data that is loaded is standardized with mean 0 and std 1
        val_data = th.load("../image_data/data/Veneto/val_data.pt")
        val_idx = np.load("../image_data/data/Veneto/val_idx.npy")
    
    if args.validate:
        print("loading a trained model...")
        model = th.load(args.load_model_path)
        save_results(model, val_data)
        print("validating on the whole dataset ...")
        data = concat_data(val_idx, val_data)
        save_results(model, data)
        print("model is validated and the results are saved.")
    else:
        print("starting to train ...")
        train(args, val_data)

if __name__ == "__main__":
    main()
