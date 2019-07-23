import numpy as np
import h5py
from sacred import Experiment
from PIL import Image
from scipy.ndimage import rotate

Image.MAX_IMAGE_PIXELS=1e7
ex = Experiment()

@ex.config
def config():
    params = {
        'ws': 100,
        'slope_dist': 640,
        'angle_step': 15,
        'image_path': '../image_data/...',
        'data_path': '../image_data/n_dataset_oldgt.h5',
        'feature_num': 94,
        'save_to': '../image_data/rot_landslide.h5',
        'region': 'Veneto',
        'pad': 64
    }

def find_maxws(params):
    return int(params['ws']*np.sqrt(2))+1

def normalize(data):
    data[data<0]=0
    data[data>180]=0
    mean = np.mean(data)
    std = np.std(data)
    data = (data-mean)/std
    return data

def load_slopeFiles(params, data_flag='train'):
    slope = np.array(Image.open(params['image_path']+'slope.tif'))
    slope = normalize(slope)
    (h, w) = slope.shape
    if data_flag == 'train':
        data = slope[:, 0:2*(w//3)]
    else:
        data = slope[:, 2*(w//3):]
    slope_array = {}
    slope_array['0'] = data
    for angle in np.arange(params['angle_step'], 360, params['angle_step']):
        # rot = rotate(slope, angle, reshape=True, mode='nearest')
        rot = rotate(data, angle, reshape=True, mode='reflect')
        slope_array[str(angle)] = rot
    return slope_array

def initialize_dataset(f, shape, data_flag, params):
    (h, w) = shape
    f.create_dataset(
        '{}/{}/data/'.format(params['region'], data_flag),
        (params['feature_num'], h, w),
        compression='lzf'
    )
    f.create_dataset('{}/{}/gt'.format(params['region'], data_flag), (1, h, w), compression='lzf')
    
    zero_train = np.zeros((h, w))
    for idx in range(params['feature_num']):
        f['{}/{}/data'.format(params['region'], data_flag)][idx, :, :] = zero_train
    return f

def deg2rad(theta):
    return (np.pi*theta)/180

def find_nshape(theta, prev_shape):
    (h, w) = prev_shape
    nh = np.cos(theta)*h + np.sin(theta)*w
    nw = np.sin(theta)*h + np.cos(theta)*w
    return (nh, nw)

def transform_mat(theta, scale, c_coord, prev_shape):
    theta_rad = deg2rad(theta)
    (nh, nw) = find_nshape(theta_rad, prev_shape)
    alpha = scale * np.cos(theta_rad)
    beta = scale * np.sin(theta_rad)
    mat = np.array(
        [[alpha, beta, (1-alpha)*c_coord[1]-beta*c_coord[0]+(nw/2-c_coord[1])],
        [-beta, alpha, beta*c_coord[1]+(1-alpha)*c_coord[0]+(nh/2-c_coord[0])]]
    )
    return mat

def find_rotated_coord(point, theta, scale, c_coord, prev_shape):
    M = transform_mat(theta, scale, c_coord, prev_shape)
    v = [point[1], point[0], 1]
    v_transformed = np.dot(M, v)
    return (v_transformed[1], v_transformed[0]) # swap x, y ro correspond to row, col

def my_rotate(params, angle, target_shape, index, flag):
    f = h5py.File(params['data_path'], 'r')
    (row, col) = index
    (nh, nw) = find_nshape(angle, (params['ws'], params['ws']))
    rot_data = np.zeros((params['feature_num'], target_shape[0], target_shape[1]))
    for channel in range(params['feature_num']):
        data = f[params['region']][flag]['data'][:, row*params['ws']][channel, :, :]
        dif_h = target_shape[0]-nh
        dif_w = target_shape[1]-nw
        rot_data[channel, :, :] = np.pad(
            rotate(data, angle, reshape=True, mode='reflect'),
            ((dif_h//2, dif_h-dif_h//2), (dif_w//2, dif_w-dif_w//2)),
            mode='edge'
        )
    return 


def find_angles(params, _log):
    f = h5py.File(params['save_to'], 'w')
    for flag in ['train', 'test']:
        slope_array = load_slopeFiles(params)
        (h, w) = slope_array['0'].shape
        hnum, wnum = h//params['ws'], w//params['ws']
        rot_ws = find_maxws(params)
        n_h, n_w = rot_ws*hnum, rot_ws*wnum
        f = initialize_dataset(f, (n_h, n_w), flag, params)
        for row in range(hnum):   
            for col in range(wnum):
                pt_value = slope_array[0][
                    row*params['ws']:(row+1)*params['ws'],
                    col*params['ws']:(col+1)*params['ws']
                    ][
                        params['ws']//2, params['ws']//2
                    ]
                pt_coord = (row*params['ws']+(params['ws']//2), col*params['ws']+(params['ws']//2))
                center_coord = (h//2, w//2)
                best_angle = 0
                angle = 0
                for idx in range(1, len(slope_array)):
                    angle += params['angle_step']
                    rot_coord = find_rotated_coord(pt_coord, angle, 1, center_coord, (h, w))
                    rot_value = slope_array[idx][rot_coord[0], rot_coord[1]]
                    if rot_value != pt_value:
                        print('the values for the rotated image and the original image are not matching.')
                        raise ValueError
                    d = params['slope_dist']
                    dist_value = slope_array[idx][rot_coord[0]-d, rot_coord[1]]
                    if dist_value > pt_value:
                        best_angle = angle
                        break
                

                    
            
        


