import argparse
import numpy as np

def str2bool(input):
    if input.lower() in ['yes', 'y', '1', 'true', 't']:
        return True
    elif input.lower() in ['no', 'n', '0', 'false', 'f']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean input is expected.')

def process_in(s):
    '''
    the data format is (r1,c1,r2,c2)
    '''
    inp = s.split(',')
    r1, c1, r2, c2 = int(inp[0].split('(')[1]), int(inp[1]), int(inp[2]), int(inp[3].split(')')[0])
    #r1 = s.split('),(')[0].split(',') # '(r1' and 'c1'
    #r2 = s.split('),(')[1].split(',') # '(r2' and 'c2'
    #row1, col1 = int(r1[0].split('(')[1]), int(r1[1])
    #row2, col2 = int(r2[0].split('(')[1]), int(r2[1])
    # row1, col1, row2, col2 = int(row1), int(col1), int(row2), int(col2)
    return np.array([r1, c1, r2, c2]).reshape(1, 4)

def __range(s):
    try:
        return process_in(s)
    except:
        raise argparse.ArgumentTypeError("Input type must be (r1, c1, r2, c2).")

def shape(s):
    try:
        name, h, w = s.split(',')
        return name, int(h), int(w)
    except:
        raise argparse.ArgumentTypeError('Input must be name(of the dataset), height, width.')

# def append_dict(*dicts):
#     '''
#     append multiple one element dictionaries.
#     '''
#     return {list(d.keys()): d[list(d.keys())[0]] for d in dicts}
