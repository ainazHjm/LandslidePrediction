from PIL import Image
import numpy as np
import argparse
Image.MAX_IMAGE_PIXELS = 10000000000

def get_args():
    parser = argparse.ArgumentParser(description='sampling from data')
    parser.add_argument('--gt_path', type=str, default='/home/ainaz/Projects/Landslides/image_data/Veneto_NEW/data/Veneto/gt.tif')
    parser.add_argument('--save_to', type=str, default='/home/ainaz/Projects/Landslides/image_data/')
    return parser.parse_args()

def sample(args):
    gt = np.array(Image.open(args.gt_path))
    ones = (gt==1).nonzero() # gives two arrays: array 1 consists of rows and array 2 consists of cols
    zeros = (gt==0).nonzero()
    num_samples = ones[0].shape[0]
    zero_samples_idx = np.random.choice(np.arange(0, zeros[0].shape[0]), size=num_samples, replace=False)
    one_samples = np.array(
        list(
            map(
                lambda idx: (ones[0][idx], ones[1][idx]),
                np.arange(0, len(ones[0]))
            )
        )
    )
    zero_samples = np.array(
        list(
            map(
                lambda idx: (zeros[0][zero_samples_idx[idx]], zeros[1][zero_samples_idx[idx]]),
                np.arange(0, num_samples)
            )
        )
    )
    # import ipdb; ipdb.set_trace()
    np.save(args.save_to+'ones.npy', one_samples)
    np.save(args.save_to+'zeros.npy', zero_samples)
    data = np.concatenate((one_samples, zero_samples))
    np.random.shuffle(data)
    idx = (8*len(data))//10
    train_data = data[0:idx]
    test_data = data[idx:]
    np.save(args.save_to+'train_data.npy', train_data)
    np.save(args.save_to+'test_data.npy', test_data)

def main():
    args = get_args()
    sample(args)

if __name__=='__main__':
    main()