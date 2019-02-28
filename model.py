# pylint: disable=E1101
import torch.nn as nn
import torch as th
# from torchvision import models

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(4, 4, kernel_size = (5,5), stride=(1,1)),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=(3,3), stride=(1,1)),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=(3,3), stride=(1,1))
        )
    def forward(self, features):
        return self.net(features)

'''
TODO: implement resnet
'''

class FCNBasicBlock(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel):
        super(FCNBasicBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=(3,3), stride=(1,1)),
            nn.ReLU(),
            nn.Conv2d(mid_channel, out_channel, kernel_size=(3,3), stride=(1,1)),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)
   
class FCNwPool(nn.Module):
    '''
    TODO: try the weighted network as well. instead of using the adaptive avg pooling,
    learn the weights of each output feature map. size = 3499x1999
    '''
    def __init__(self, shape):
        super(FCNwPool, self).__init__()
        self.shape = shape # CxHxW
        self.net = nn.Sequential(
            FCNBasicBlock(shape[0], 8, 16),
            nn.MaxPool2d(kernel_size=(4,4), stride=(4,4)),
            FCNBasicBlock(16, 32, 64),
            nn.MaxPool2d(kernel_size=(4,4), stride=(4,4)),
            FCNBasicBlock(64, 128, 256),
            nn.MaxPool2d(kernel_size=(4,4), stride=(4,4)),
        )
        self.res0 = nn.Sequential(
            nn.ConvTranspose2d(16, 4, kernel_size=(3,3), stride=(1,1)),
            nn.ConvTranspose2d(4, 1, kernel_size=(3,3), stride=(1,1)),
        )
        self.res1 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=(3,3), stride=(16,16)),
            nn.ConvTranspose2d(16, 4, kernel_size=(3,3), stride=(1,1)),
            *[nn.ConvTranspose2d(4, 4, kernel_size=(3,3), stride=(1,1)) for i in range(16)],
            nn.ConvTranspose2d(4, 1, kernel_size=(3,3), stride=(1,1)),
        )
        self.res2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=(3,3), stride=(76, 76)),
            nn.ConvTranspose2d(64, 16, kernel_size=(3,3), stride=(1,1)),
            *[nn.ConvTranspose2d(16, 16, kernel_size=(3,3), stride=(1,1)) for i in range(2)],
            nn.ConvTranspose2d(16, 1, kernel_size=(3,3), stride=(1,1)),
        )
        # self.avgpool = nn.AdaptiveAvgPool2d((self.shape[1], self.shape[2]))

    def forward(self, x):
        out0 = self.net[0](x)
        out1 = self.net[1:4](out0)
        out2 = self.net[4:](out1)
        out = th.stack(
            (
                self.res0(out0),
                self.res1(out1),
                self.res2(out2)
            )
        )
        return th.sum(out,0)

# class SimpleCNN(nn.Module):
#     """
#     A simple conv-net:
#         - there are only two filters corresponding to the 3x3 neighbourhood
#         - the input to the classifier is only the pixel itself along with 
#             the features from 3x3 convolution.
#     """
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(4, 8, kernel_size=(3, 3), stride=(1,1)),
#             nn.ReLU(),
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(in_features=4+8, out_features=32),
#             nn.ReLU(),
#             nn.Linear(in_features=32, out_features=32),
#             nn.ReLU(),
#             nn.Linear(in_features=32, out_features=1),
#             nn.Sigmoid(),
#         )
#     def forward(self, features):
#         return self.classifier(features)

# class SimpleNgbCNN(nn.Module):
#     """
#     This conv-net considers the neighbours themselves as the input features 
#     to the classifier. The rest is the same as the CimpleCNN.
#     """
#     def __init__(self):
#         super(SimpleNgbCNN, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(4, 8, kernel_size=(3, 3), stride=(1,1)),
#             nn.ReLU(),
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(in_features=4+32+8, out_features=64),
#             nn.ReLU(),
#             nn.Linear(in_features=64, out_features=64),
#             nn.ReLU(),
#             nn.Linear(in_features=64, out_features=1),
#             nn.Sigmoid(),
#         )
#     def forward(self, features):
#         return self.classifier(features)

# class ComplexCNN(nn.Module):
#     """
#     A conv-net for making predictions on pixels. The input to the classifier is the pixel itself along
#     with the features obtained from the 3x3 neighborhood and the background which is a 4x220x148 image.
#     TODO: add changes for the padded image with resolution 10000x7000
#     """
#     def __init__(self):
#         super(ComplexCNN, self).__init__()
#         self.bgfeatures = nn.Sequential(
#             nn.Conv2d(4, 8, kernel_size=(3,3), stride=(1,1)),
#             nn.ReLU(),
#             nn.Conv2d(8, 32, kernel_size=(3,3), stride=(1,1)),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
#             nn.Conv2d(32, 32, kernel_size=(3,3), stride=(1,1)),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, kernel_size=(3,3), stride=(1,1)),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
#             nn.Conv2d(32, 32, kernel_size=(3,3), stride=(1,1)),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, kernel_size=(3,3), stride=(1,1)),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
#         )
#         self.ngbfeatures = nn.Sequential(
#             nn.Conv2d(4, 8, kernel_size=(3,3), stride=(1,1)),
#             nn.ReLU(),
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(in_features=1+8+5376, out_features=64),
#             nn.ReLU(),
#             nn.Linear(in_features=64, out_features=16),
#             nn.ReLU(),
#             nn.Linear(in_features=16, out_features=1),
#             nn.Sigmoid()
#         )
#     def forward(self, feature):
#         return self.classifier(feature)
