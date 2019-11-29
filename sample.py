from PIL import Image
import numpy as np
import argparse
Image.MAX_IMAGE_PIXELS = 10000000000

def get_args():
    parser = argparse.ArgumentParser(description='sampling from data')
    parser.add_argument('--gt_path', type=str, default='/home/ainaz/Projects/Landslides/image_data/Veneto_NEW/data/Veneto/gt.tif')
    parser.add_argument('--save_to', type=str, default='/home/ainaz/Projects/Landslides/image_data/')
    return parser.parse_args()

def sample_fn(args, data, flag='train'):
    ones = (data==1).nonzero() # gives two arrays: array 1 consists of rows and array 2 consists of cols
    zeros = (data==0).nonzero()
    num_samples = ones[0].shape[0]
    zero_samples_idx = np.random.choice(np.arange(0, zeros[0].shape[0]), size=num_samples, replace=False)
    one_samples = np.array(
        list(
            map(
                lambda idx: (ones[0][idx], ones[1][idx]),
                np.arange(0, num_samples)
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
    res = np.concatenate((one_samples, zero_samples))
    np.random.shuffle(res)
    np.save(args.save_to+flag+'_data.npy', res)
    
def sample(args):
    gt = np.array(Image.open(args.gt_path))
    (h, _) = gt.shape

    train_gt = np.concatenate((gt[0:h//5, :], gt[2*(h//5):, :]))
    sample_fn(args, train_gt, 'train')

    test_gt = gt[h//5:2*(h//5), :]
    sample_fn(args, test_gt, 'test')
    

def main():
    args = get_args()
    sample(args)

if __name__=='__main__':
    main()