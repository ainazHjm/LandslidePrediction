import h5py
import numpy as np
import scipy.ndimage as snd
from time import ctime

# def find_hill(args, data, distance):
#     look = distance//args.pix_res
#     (h, w) = data.shape
#     slope = data[h//2, w//2] # look at slope at the center pixel
#     if data[h//2-look, w//2] > slope or data[h//2+look, w//2] > slope:
#         return 1
#     else:
#         return 0

def rad2deg(theta):
    return (theta*180)/np.pi

def find_angle(pc, pr):
    theta = np.arctan2(abs(pr[0]-pc[0]), abs(pr[1]-pc[1]))
    return rad2deg(theta)

def adjust_rot(args, dataset, flag='train'):
    rotations = np.zeros((len(dataset), 3)) # the first two are row and col and the last one is angle
    for idx in range(len(dataset)):
        sample = dataset[idx]
        (h, w) = sample['data'][0, :, :].shape
        hill = sample['data'][0, :, :].max() # the 0th channel is the slope
        tailpt = (sample['data'][0, :, :] == hill).nonzero()[0].data.numpy()
        headpt = np.array([h//2, w//2])
        angle = find_angle(headpt, tailpt)

# def adjust_rot(args, data_flag='train'):
#     f = h5py.File(args.data_path, 'r') # the path to the dataset
#     data = f[args.region][data_flag]['data']
#     (_, h, w) = data.shape
#     h_orig, w_orig = h-2*args.pad, w-2*args.pad
#     rots = np.zeros((h_orig, w_orig))
#     print('shape: (%d, %d) >> (%d, %d)' %(h, w, h_orig, w_orig))
#     print('%s initialized the rotation matrix ...' %ctime())
#     for row in range(h_orig):
#         for col in range(w_orig):
#             if data[45, row+args.pad, col+args.pad] < 0:
#                 rots[row, col] = 0
#                 print('~~~ %s ~~~ ignoring data at (%d, %d)' %(ctime(), row, col), end='\r')
#             else:
#                 inp = data[0, row:row+2*args.pad+1, col:col+2*args.pad+1] # args.pad should be 32 to get 1x1 pixel
#                 best_angle = 0
#                 for angle in np.arange(10, 360, 10):
#                     rot_data = snd.rotate(inp, angle, reshape=True)
#                     if find_hill(args, rot_data, 320): # looking at 320m distance
#                         best_angle = angle
#                         break
#                 print('--- %s --- the best angle found for (%d, %d): %d' %(ctime(), row, col, best_angle), end='\r')
#                 rots[row, col] = best_angle
#     save_to = '/'.join(args.data_path.split('/')[:-1])+'/'+data_flag+'_rot.npy'
#     np.save(save_to, rots)
#     print('%s: the rotation file is saved.' %ctime())
#     return save_to