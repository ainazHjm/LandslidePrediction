# pylint: disable=E1101
import torch.nn as nn
import torch as th
import torch.nn.functional as F

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
        self.in_channel = in_channel
        self.mid_channel = mid_channel
        self.out_channel = out_channel

    def batchNormNet(self, x, eps=1e-5):
        (n, c, h, w) = x.shape
        indices = x[:, 0, :, :] != -100 # only care about the pixels that we have data points for
        input_data = x[indices].view(-1, c)
        curr_batch_size = input_data.shape[0]
        
        mean = th.mean(input_data, 0) # this is a vector of size c
        var = th.var(input_data, 0) # this is also a vector of size c
        normalized_data = -100 * th.ones(n, c, h, w)
        normalized_data[indices] = (x[indices]-mean)/th.sqrt(var+eps)
        projection = nn.Linear(c, c)
        output = projection(normalized_data) # this has the same shape as the input (batch_size x c)
        return output # output should have the same size as input (n, c, h, w)

    def forward(self, x):
        conv1 = nn.Conv2d(self.in_channel, self.mid_channel, kernel_size=(3,3), stride=(1,1))
        conv2 = nn.Conv2d(self.mid_channel, self.out_channel, kernel_size=(3,3), stride=(1,1))
        data = F.relu(self.batchNormNet(conv1(x)))
        return F.relu(self.batchNormNet(conv2(data)))



class FCNwPool(nn.Module):
    '''
    The model uses four different resolutions:
        - 20m ~ 25m
        - 80m ~ 100m
        - 320m ~ 400m
        - 1280m ~ 1600m
    The output is the sum of all these resolutions.

    TODO: try the weighted network as well. instead of using the adaptive avg pooling,
    learn the weights of each output feature map. size = 3499x1999
    '''
    def __init__(self, shape, pixel_res=20):
        super(FCNwPool, self).__init__()
        self.shape = shape # CxHxW
        self.pixel_res = pixel_res
        self.net = nn.Sequential(
            FCNBasicBlock(shape[0], 8, 16),
            nn.MaxPool2d(kernel_size=(4,4), stride=(4,4)),
            FCNBasicBlock(16, 32, 64),
            nn.MaxPool2d(kernel_size=(4,4), stride=(4,4)),
            FCNBasicBlock(64, 128, 256),
            nn.MaxPool2d(kernel_size=(4,4), stride=(4,4)),
        )
        self.res0 = nn.Sequential(
            nn.ConvTranspose2d(16, 4, kernel_size=(4,4), stride=(1,1)),
            nn.ConvTranspose2d(4, 1, kernel_size=(2,2), stride=(1,1)),
        )
        self.res1 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=(2,2), stride=(2,2)),
            nn.ConvTranspose2d(8, 4, kernel_size=(2,2), stride=(2,2)),
            nn.ConvTranspose2d(4, 2, kernel_size=(4,4), stride=(1,1)),
            nn.ConvTranspose2d(2, 1, kernel_size=(4,4), stride=(1,1)),
        )
        self.res2 = nn.Sequential(
            *[
                nn.ConvTranspose2d(2**(6-i), 2**(5-i), kernel_size=(2,2), stride=(2,2)) 
                for i in range(4)
            ],
            *[
                nn.ConvTranspose2d(2**(2-i), 2**(1-i), kernel_size=(4,4), stride=(1,1)) if 2-i != 0
                else nn.ConvTranspose2d(1, 1, kernel_size=(4,4), stride=(1,1))
                    for i in range(5)
            ],
        )
        self.res3 = nn.Sequential(
            *[
                nn.ConvTranspose2d(2**(8-i), 2**(7-i), kernel_size=(2,2), stride=(2,2))
                for i in range(6)
            ],
            *[
                nn.ConvTranspose2d(2**(3-i), 2**(2-i), kernel_size=(4,4), stride=(1,1)) if 3-i != 0
                else nn.ConvTranspose2d(1, 1, kernel_size=(4,4), stride=(1,1))
                    for i in range(33)
            ],
            nn.ConvTranspose2d(1, 1, kernel_size=(2,2), stride=(1,1)),
        )
        # self.avgpool = nn.AdaptiveAvgPool2d((self.shape[1], self.shape[2]))
        # self.last = nn.Conv2d(3, 1, kernel_size=(1,1), stride=(1,1))
        self.last = nn.Conv2d(20, 1, kernel_size=(1,1), stride=(1,1))

    def create_mask(self, padding):
        kernel = th.zeros(5, 1, 2*padding+1, 2*padding+1)
        kernel[0, 0, 0, padding] = 1
        kernel[1, 0, padding, 0] = 1
        kernel[2, 0, padding, -1] = 1
        kernel[3, 0, -1, padding] = 1
        kernel[-1, 0, padding, padding] = 1
        #print("kernel shape:")
        #print(kernel.shape)
        return kernel.cuda()

    def get_neighbors(self, features, pixel_res):
        # print(features.shape)
        (b, c, h, w) = features.shape # c should be 4 because we have three different resolutions
        n_features = th.zeros(b, c*5, h, w).cuda()

        n_features[:, 0:5, :, :] = F.conv2d(
            features[:, 0, :, :].view(-1, 1, h, w),
            self.create_mask(10//pixel_res) + 1),
            padding=10//pixel_res + 1,
            )
        n_features[:, 5:10, :, :] = F.conv2d(
            features[:, 1, :, :].view(-1, 1, h, w),
            self.create_mask(40//pixel_res + 1),
            padding=40//pixel_res + 1,
            )
        n_features[:, 10:15, :, :] = F.conv2d(
            features[:, 2, :, :].view(-1, 1, h, w),
            self.create_mask(160//pixel_res + 1),
            padding=160//pixel_res + 1,
            )
        n_features[:, 15:20, :, :] = F.conv2d(
            features[:, 3, :, :].view(-1, 1, h, w),
            self.create_mask(640//pixel_res + 1),
            padding=640//pixel_res + 1,
            )
        return n_features

    def forward(self, x):
        out0 = self.net[0](x)
        out1 = self.net[1](out0)
        out2 = self.net[2:4](out1)
        out3 = self.net[4:6](out2)
        out = th.stack((
            self.res0(out0),
            self.res1(out1),
            self.res2(out2),
            self.res3(out3),
        )).view(-1, 4, self.shape[1], self.shape[2])
        #print("out shape:")
        #print(out.shape)
        fx = self.last(self.get_neighbors(out, self.pixel_res))
        # print(fx.shape)
        return fx