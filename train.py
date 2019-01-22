from data import process
import model
import numpy as np
import torch as th

def find_neighbors(train_data):
    neighbors = th.zeros(4, )

def train(args):
    val_idx, val_data, train_data = process(args.model, args.divide) # data: 5x6630x9387
    trainset = train_data.reshape(5, -1)
    valset = val_data.reshape(5, -1)

    if args.model == "SimpleCNN":
        train_model = model.SimpleCNN
    elif args.model == "SimpleNgbCNN":
        train_model = model.SimpleNgbCNN
    else:
        train_model = model.ComplexCNN

    bs = args.batch_size
    for i in range(args.n_epochs):
        num_iters = trainset.shape[1] // bs
        for j in range(num_iters):
            in_data = trainset[:-1, j*bs:(j+1)*bs]
            
            true_labels = trainset[-1, j*bs:(j+1)*bs]
            #TODO: implement the complex CNN


            




    
    

    