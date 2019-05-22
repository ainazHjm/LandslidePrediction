import argparse

def str2bool(input):
    if input.lower() in ['yes', 'y', '1', 'true', 't']:
        return True
    elif input.lower() in ['no', 'n', '0', 'false', 'f']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean input is expected.')

def pos(s):
    '''
    input is in this format >> name:y0,y1,x0,x1
    '''
    try:
        input = s.split(':')
        return {input[0]: tuple(map(int, input[1:].split(',')))}
    except:
        raise argparse.ArgumentTypeError("Input type must be name,y[0],y[1],x[0],x[1].")

def shape(s):
    try:
        name, h, w = s.split(',')
        return name, int(h), int(w)
    except:
        raise argparse.ArgumentTypeError('Input must be name(of the dataset), height, width.')

def append_dict(*dicts):
    '''
    append multiple one element dictionaries.
    '''
    return {list(d.keys()): d[list(d.keys())[0]] for d in dicts}