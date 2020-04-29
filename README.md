# Landslide
Classification task for predicting landslides based on GIS maps.

## Requirements
* `numpy`
* `torch`
* `sacred`
* `tensorboard`
* `matplotlib`
* `sklearn`
* `h5py`

## Arguments
* `optim`: choice of the optimizer
* `lr`: learning rate
* `n_epochs`: number of epochs to train the model
* `bs`: batch size
* `decay`: L2 regularization parameter for the optimizer
* `patience`: number of epochs to wait before changing the learning rate in the scheduler
* `pos_weight`: positive sample weight in the loss function
* `model`: model name to use for training
* `n_workers`: number of workers to use for loading the data in data loader
* `region`: the region of the dataset
* `pix_res`: resolution of the pixels in the dataset
* `stride`: stride of the CNN
* `ws`: window/kernel size of the CNN
* `pad`: padding size in CNN
* `feature_num`: total number of features
* `oversample`: boolean value indicating whether we want to oversample the data or not
* `prune`: prunning size for the input images
* `dist_num`: number of distance features (how far do we want to look)
* `dist_feature`: boolean value indicating if we want to use distance features or not
* `load_model`: path to the model
* `data_path`: path to the data
* `index_path`: path to the indices showing the partitioning of the data
* `save`: how often (how many epochs) we want to save the training models

Train the model using sacred specifying the arguments if you don't want to use the default values:
`python main.py with 'train_params.optim="SGD"' ... 'loc_param.data_path="/tmp/Veneto_data.h5"' -m CNN`

Validate the model:
`python validate.py with 'params.load_model="/tmp/m1.pt"' ...`

### Extra links:

[Sacred Documentation](https://sacred.readthedocs.io/en/stable/quickstart.html)

[Tensorbaord Documentation for pytorch](https://github.com/lanpa/tensorboardX)

## Citation
This repository implements the paper "Predicting Landslides Using Locally Aligned Convolutional Neural Networks, A. Hajimoradlou, G. Roberti, and D. Poole". Please cite our project, if you use the data or the code provided here in your work.

**Links**:
 - https://arxiv.org/abs/1911.04651

## LICENSE
<a rel="license" href="http://creativecommons.org/licenses/by-sa/3.0/"><img alt="Creative Commons License" style="border-width:0" src="https://licensebuttons.net/l/by-nc-sa/3.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="https://creativecommons.org/licenses/by-nc-sa/3.0/">Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License</a>.

