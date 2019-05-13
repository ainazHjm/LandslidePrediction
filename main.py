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
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--decay", type=float, default=0)
    parser.add_argument("--load_model", type=str, default='')
    parser.add_argument("--validate", type=str2bool, default=False)
    parser.add_argument("--data_path", type=str, default="../image_data/data/")
    parser.add_argument("--save_model_to", type=str, default="../models/CNN/")
    parser.add_argument("--region", type=str, default='Veneto')
    parser.add_argument("--pix_res", type=int, default=10)
    parser.add_argument("--s", type=int, default=5) #save the model at how many epochs
    parser.add_argument("--c", type=str2bool, default=False)
    parser.add_argument("--pad", type=int, default=64)
    parser.add_argument("--save_res_to", type=str, default='../output/CNN/Piemonte/')
    return parser.parse_args()

def main():
    args = get_args()
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
        train.train(args)

if __name__ == "__main__":
    main()
