# pylint: disable=E0611
import train
import argparse
import torch as th
import numpy as np
from utils.args import str2bool
from utils.plot import save_results

def get_args():
    parser = argparse.ArgumentParser(description="Training a CNN-Classifier for landslide prediction")
    # parser.add_argument("--cross_validation", type=str2bool, default=False)
    parser.add_argument("--model", type=str, default="FCN")
    # parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--decay", type=float, default=0)
    parser.add_argument("--load_model", type=str, default='')
    parser.add_argument("--validate", type=str2bool, default=False)
    parser.add_argument("--data_path", type=str, default="../image_data/data/Piemonte/")
    parser.add_argument("--save_model_to", type=str, default="../models/CNN/Piemonte/")
    parser.add_argument("--pix_res", type=int, default=10)
    parser.add_argument("--pad", type=int, default=64)
    parser.add_argument("--s", type=int, default=5) #save the model at how many epochs
    parser.add_argument("--c", type=str2bool, default=False)
    parser.add_argument("--save_res_to", type=str, default='../output/CNN/Piemonte/')
    # parser.add_argument("--region", type=str, default='Piemonte')
    return parser.parse_args()

def main():
    args = get_args()
    # if args.cross_validation:
    #     data = process()
    #     # train_data, val_data, val_idx = cross_validate(args, data)
    # else:
    #     # the data that is loaded is standardized with mean 0 and std 1
    #     val_data = th.load("../image_data/data/Veneto/nan/val_data.pt")
    #     val_idx = np.load("../image_data/data/Veneto/val_idx.npy")
    td = np.load(args.data_path+'tdIdx.npy')
    vd = np.load(args.data_path+'vdIdx.npy')
    print("data index is loaded ...")

    if args.validate:
        print("loading a trained model...")
        print("validating the model on validation data ...")
        model = th.load(args.load_model)
        save_results(args, model, vd)
        print("validating the model on training data ...")
        save_results(args, model, td)
        print("model is validated and the results are saved.")      
    else:
        print("starting to train ...")
        train.train(args, td, vd)

if __name__ == "__main__":
    main()
