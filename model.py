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
    '''
    This class consists of convolutions but doesn't change the size.
    '''
    def __init__(self, in_channel, out_channel):
        super(FCNBasicBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

class FCNDownSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FCNDownSample, self).__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(kernel_size=(4,4), stride=(4,4)),
            FCNBasicBlock(in_channel, out_channel),
            FCNBasicBlock(out_channel, out_channel),
        )
    def forward(self, x):
        return self.net(x)

class FCNUpSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FCNUpSample, self).__init__()
        # self.k = (hout - hin*4) + 1
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=(5,5), stride=(1,1), padding=(2,2)),
            nn.ConvTranspose2d(out_channel, out_channel, kernel_size=(4,4), stride=(4,4)),
        ) 
        # self.net = nn.Sequential(
        #     *[self.block for _ in range(n-1)],
        #     nn.ConvTranspose2d(out_channel, out_channel, kernel_size=(5,5), stride=(1,1)),
        # )
    def forward(self, x):
        return self.net(x)

class FCNwPool(nn.Module):
    '''
    The model uses four different resolutions:
        - 20m ~ 25m
        - 80m ~ 100m
        - 320m ~ 400m
        - 1280m ~ 1600m
    The output is the sum of all these resolutions.
    '''
    def __init__(self, shape, pixel_res):
        super(FCNwPool, self).__init__()
        self.shape = shape # CxHxW
        self.pixel_res = pixel_res

        self.d1 = nn.Sequential(
            FCNBasicBlock(shape[0], 64),
            FCNBasicBlock(64, 128),
        )
        self.d2 = FCNDownSample(128, 256)
        self.d3 = FCNDownSample(256, 512)
        self.d4 = FCNDownSample(512, 1024)

        self.u1 = nn.Sequential(
            FCNUpSample(1024, 512),
            FCNUpSample(512, 256),
            FCNUpSample(256, 128),
            nn.ConvTranspose2d(128, 1, kernel_size=(5,5), stride=(1,1), padding=(2,2)),
        )
        self.u2 = nn.Sequential(
            FCNUpSample(512, 256),
            FCNUpSample(256, 128),
            nn.ConvTranspose2d(128, 1, kernel_size=(5,5), stride=(1,1), padding=(2,2)),
        )
        self.u3 = nn.Sequential(
            FCNUpSample(256, 128),
            nn.ConvTranspose2d(128, 1, kernel_size=(5,5), stride=(1,1), padding=(2,2)),
        )
        self.u4 = nn.ConvTranspose2d(128, 1, kernel_size=(5,5), stride=(1,1), padding=(2,2))

        self.last = nn.Conv2d(20, 1, kernel_size=(1,1), stride=(1,1))

    def create_mask(self, padding):
        kernel = th.zeros(5, 1, 2*padding+1, 2*padding+1)
        kernel[0, 0, 0, padding] = 1
        kernel[1, 0, padding//2, padding] = 1
        kernel[2, 0, padding//2 + padding, padding] = 1
        kernel[3, 0, -1, padding] = 1
        kernel[-1, 0, padding, padding] = 1
        #print("kernel shape:")
        #print(kernel.shape)
        return kernel.cuda()

    def get_neighbors(self, features, pixel_res):
        # print(features.shape)
        (b, c, h, w) = features.shape # c should be 4 because we have four different resolutions
        n_features = th.zeros(b, c*5, h, w).cuda()

        n_features[:, 0:5, :, :] = F.conv2d(
            features[:, 0, :, :].view(-1, 1, h, w),
            self.create_mask(640//pixel_res + 1),
            padding=640//pixel_res + 1,
            )
        n_features[:, 5:10, :, :] = F.conv2d(
            features[:, 1, :, :].view(-1, 1, h, w),
            self.create_mask(160//pixel_res + 1),
            padding=160//pixel_res + 1,
            )
        n_features[:, 10:15, :, :] = F.conv2d(
            features[:, 2, :, :].view(-1, 1, h, w),
            self.create_mask(40//pixel_res + 1),
            padding=40//pixel_res + 1,
            )
        n_features[:, 15:20, :, :] = F.conv2d(
            features[:, 3, :, :].view(-1, 1, h, w),
            self.create_mask(10//pixel_res + 1),
            padding=10//pixel_res + 1,
            )
        return n_features

    def pad(self, x, xt):
        (_, _, h, w) = x.shape
        (_, _, ht, wt) = xt.shape
        hdif = ht - h
        wdif = wt - w
        x = F.pad(x, (wdif//2, wdif-wdif//2, hdif//2, hdif-hdif//2))
        # print(x.shape, xt.shape)
        return x

    def forward(self, x):
        # print(x.shape)
        o1 = self.d1(x)
        o2 = self.d2(o1)
        o3 = self.d3(o2)
        o4 = self.d4(o3)
        res1 = self.pad(self.u1(o4), x)
        res2 = self.pad(self.u2(o3), x)
        res3 = self.pad(self.u3(o2), x)
        res4 = self.pad(self.u4(o1), x)
        
        # print(o1.shape, o2.shape, o3.shape, o4.shape)
        # print(self.u1(o4).shape)
        # print(self.u2(o3).shape)
        # print(self.u3(o2).shape)
        # print(self.u4(o1).shape)
        out = th.stack((res1, res2, res3, res4)).view(-1, 4, self.shape[1], self.shape[2])
        #print("out shape:")
        # print(out.shape)
        fx = self.last(self.get_neighbors(out, self.pixel_res))
        # print(fx.shape)
        return fx
