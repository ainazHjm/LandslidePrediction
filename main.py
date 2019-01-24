from data import process, normalize
from train import train, find_accuracy, cross_validate
import argparse
import torch as th
import numpy as np
# pylint: disable=E0611
from utils.args import str2bool

def get_args():
    parser = argparse.ArgumentParser(description="Training a CNN-Classifier for landslide prediction")
    parser.add_argument("--cross_validation", type=str2bool, default=False)
    parser.add_argument("--model", type=str, default="FCN")
    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--decay", type=float, default=1e-5)
    parser.add_argument("--load_model_path", type=str, default='')
    parser.add_argument("--save_features", type=str2bool, default=False)
    return parser.parse_args()

def main():
    args = get_args()
    data = process()

    if args.cross_validation:
        train_data, val_data, val_idx = cross_validate(args, data)
    else:
        val_data = th.load("../image_data/data/val_data.pt")
        val_idx = np.load("../image_data/data/val_idx.npy")
        train_data = th.load("../image_data/data/train_data.pt")

    td, vd = normalize(train_data, val_data)
    
    if args.load_model_path:
        print("loading a trained model...")
        model = th.load(args.load_model_path)
        acc = find_accuracy(model, vd)
        print(">> accuracy on the validation set: %f" % acc)
    else:
        print("model is training ...")
        loss = train(args, td, vd)
        print("model has been trained.")

if __name__ == "__main__":
    main()
