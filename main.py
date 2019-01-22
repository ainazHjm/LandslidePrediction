# import data
# import model
from train import train
import argparse
# pylint: disable=E0611
from utils.args import str2bool

def get_args():
    parser = argparse.ArgumentParser(description="Training a CNN-Classifier for landslide prediction")
    parser.add_argument("--divide_data", type=str2bool, default=False)
    parser.add_argument("--model", type=str, default="SimpleCNN")
    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=100)
    # parser.add_argument("--regularizer")
    parser.add_argument("--load_model_path", type=str, default='')
    parser.add_argument("--save_features", type=str2bool, default=False)
    return parser.parse_args()

def main():
    args = get_args()
    if args.load_model_path:
        #TODO: add load
        print("loading a trained model")
    else:
        train(args)

if __name__ == "__main__":
    main()
