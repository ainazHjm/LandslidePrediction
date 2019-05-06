import argparse

def str2bool(input):
    if input.lower() in ['yes', 'y', '1', 'true', 't']:
        return True
    elif input.lower() in ['no', 'n', '0', 'false', 'f']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean input is expected.')

def pos(s):
    try:
        x, y, z, w = map(int, s.split(','))
        return x, y, z, w
    except:
        raise argparse.ArgumentTypeError("Input type must be x, y, z, w.")