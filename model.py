# pylint: disable=E1101
import torch.nn as nn
import torch as th
import torch.nn.functional as F

class Logistic(nn.Module):
    def __init__(self, in_channel):
        super(Logistic, self).__init__()
        self.net = nn.Conv2d(in_channel, 1, kernel_size=(1,1), stride=(1,1))
    def forward(self, x):
        return self.net(x)

class PolyLogistic(nn.Module):
    def __init__(self, in_channel):
        super(PolyLogistic, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=(1,1), stride=(1,1)),
            # nn.Conv2d(in_channel, 64, kernel_size=(7,7), stride=(1,1), padding=(3,3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 4, kernel_size=(1,1), stride=(1,1)),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 1, kernel_size=(1,1), stride=(1,1)),
        )
    def forward(self, x):
        return self.net(x)

class FCN(nn.Module):
    def __init__(self, in_channel):
        super(FCN, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channel, 4, kernel_size = (5,5), stride=(1,1)),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=(3,3), stride=(1,1)),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=(3,3), stride=(1,1))
        )
    def forward(self, features):
        return self.net(features)

class FCNBasicBlock(nn.Module):
    '''
    This class consists of convolutions but doesn't change the size.
    '''
    def __init__(self, in_channel, out_channel):
        super(FCNBasicBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, out_channel, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

class FCNDownSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FCNDownSample, self).__init__()
        self.net = nn.Sequential(
            FCNBasicBlock(in_channel, in_channel),
            FCNBasicBlock(in_channel, out_channel),
            nn.MaxPool2d(kernel_size=(4,4), stride=(4,4)),
        )
    def forward(self, x):
        return self.net(x)

class FCNUpSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FCNUpSample, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channel, in_channel, kernel_size=(5,5), stride=(1,1), padding=(2,2)),
            nn.ConvTranspose2d(in_channel, in_channel, kernel_size=(4,4), stride=(4,4)),
            FCNBasicBlock(in_channel, in_channel),
            FCNBasicBlock(in_channel, out_channel),
        )
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
    def __init__(self, in_channel, pixel_res):
        super(FCNwPool, self).__init__()
        self.pixel_res = pixel_res

        self.d1 = nn.Sequential(
            FCNBasicBlock(in_channel, 64),
            FCNBasicBlock(64, 128),
        )
        self.d2 = FCNDownSample(128, 256)
        self.d3 = FCNDownSample(256, 512)
        self.d4 = FCNDownSample(512, 512)
        
        self.upsample = nn.Sequential(
            FCNUpSample(512, 512),
            FCNUpSample(512, 256),
            FCNUpSample(256, 128),
            nn.ConvTranspose2d(128, 1, kernel_size=(5,5), stride=(1,1), padding=(2,2)),
        )
        # self.last = nn.Conv2d(20, 1, kernel_size=(1,1), stride=(1,1))
        self.last = nn.Conv2d(4, 1, kernel_size=(1,1), stride=(1,1))

    def create_mask(self, padding):
        kernel = th.zeros(5, 1, 2*padding+1, 2*padding+1)
        kernel[0, 0, 0, padding] = 1
        kernel[1, 0, padding//2, padding] = 1
        kernel[2, 0, padding//2 + padding, padding] = 1
        kernel[3, 0, -1, padding] = 1
        kernel[-1, 0, padding, padding] = 1
        return kernel.cuda()

    def get_neighbors(self, features, pixel_res):
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
        return x

    def forward(self, x):
        (_, _, h, w) = x.shape
        o1 = self.d1(x)
        o2 = self.d2(o1)
        o3 = self.d3(o2)
        o4 = self.d4(o3)
        res4 = self.pad(self.upsample(o4), x)
        res3 = self.pad(self.upsample[1:](o3), x)
        res2 = self.pad(self.upsample[2:](o2), x)
        res1 = self.pad(self.upsample[3](o1), x)
        
        out = th.stack((res1, res2, res3, res4)).view(-1, 4, h, w)
        fx = self.last(out)
        return fx

class InConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(InConv, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(7,7), stride=(1,1), padding=(3,3)),
            # nn.Conv2d(in_channel, out_channel, kernel_size=(7,7), stride=(1,1), dilation=(4,4)),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1,1), stride=(4,4), bias=True), # downsample by 4
        )
    def forward(self, x):
        return self.net(x)

class OutConv(nn.Module):
    def __init__(self, in_channel):
        super(OutConv, self).__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            # nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, 1, kernel_size=(1,1), stride=(1,1), bias=True),
            # nn.ConvTranspose2d(4, 1, kernel_size=(2,2), stride=(2,2), bias=False),
        )
    def forward(self, x):
        return self.net(x)

class BottleNeck(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(BottleNeck, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(1,1), stride=(1,1)),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1,1), stride=(1,1)),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1,1), stride=(1,1)),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class BNwDownSample(nn.Module):
    def __init__(self, in_channel, out_channel, ds):
        super(BNwDownSample, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(1,1), stride=(ds, ds)),
            BottleNeck(out_channel, out_channel),
            # nn.BatchNorm2d(out_channel),
        )
    def forward(self, x):
        return self.net(x)

class BNwUpSample(nn.Module):
    def __init__(self, in_channel, out_channel, us):
        super(BNwUpSample, self).__init__()
        self.net = nn.Sequential(
            # nn.ConvTranspose2d(in_channel, out_channel, kernel_size=(2,2), stride=(2,2), bias=False), #Upsampling here
            nn.Upsample(scale_factor=us, mode='bilinear', align_corners=False),
            # nn.BatchNorm2d(in_channel),
            BottleNeck(in_channel, out_channel),
        )
    def forward(self, x):
        return self.net(x)

class DSLayer(nn.Module):
    def __init__(self, in_channel, out_channel, scale):
        super(DSLayer, self).__init__()
        self.net = nn.Sequential(
            # BottleNeck(in_channel, mid_channel),
            # BottleNeck(in_channel, in_channel, mid_channel),
            # BottleNeck(mid_channel, mid_channel, mid_channel),
            BNwDownSample(in_channel, out_channel, scale),
        )
    def forward(self, x):
        return self.net(x)

class USLayer(nn.Module):
    def __init__(self, in_channel, out_channel, scale):
        super(USLayer, self).__init__()
        self.net = nn.Sequential(
            BNwUpSample(in_channel, out_channel, scale),
            # BottleNeck(mid_channel, mid_channel, out_channel),
            # BottleNeck(out_channel, out_channel, out_channel),
            # BottleNeck(out_channel, out_channel, out_channel),
        )
    def forward(self, x):
        return self.net(x)

class FCNwBottleneck(nn.Module):
    def __init__(self, in_channel, pix_res):
        super(FCNwBottleneck, self).__init__()
        self.pix_res = pix_res
        self.downsample = nn.Sequential(
            InConv(in_channel, 16), # downsampled by 4
            DSLayer(16, 64, 4), # downsampled by 4
            DSLayer(64, 256, 2), # downsampled by 2
            DSLayer(256, 512, 2), # downsampled by 2
        )
        self.upsample = nn.Sequential(
            USLayer(512, 256, 2),
            USLayer(256, 64,  2),
            USLayer(64, 16, 4),
            OutConv(16), #upsampled by 4
        )
        self.last = nn.Conv2d(4, 1, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        # self.last = nn.Conv2d(20, 1, kernel_size=(1,1), stride=(1,1))
    
    def pad(self, x, xt):
        (_, _, h, w) = x.shape
        (_, _, ht, wt) = xt.shape
        hdif = ht - h
        wdif = wt - w
        x = F.pad(x, (wdif//2, wdif-wdif//2, hdif//2, hdif-hdif//2), mode='reflect')
        return x

    def create_mask(self, padding):
        kernel = th.zeros(5, 1, 2*padding+1, 2*padding+1)
        kernel[0, 0, 0, padding] = 1
        kernel[1, 0, padding//2, padding] = 1
        kernel[2, 0, padding//2 + padding, padding] = 1
        kernel[3, 0, -1, padding] = 1
        kernel[-1, 0, padding, padding] = 1
        return kernel.cuda()

    def get_neighbors(self, features, pixel_res):
        (b, c, h, w) = features.shape # c should be 4 because we have four different resolutions
        n_features = th.zeros(b, c*5, h, w).cuda()

        n_features[:, 0:5, :, :] = F.conv2d(
            features[:, 0, :, :].view(-1, 1, h, w),
            self.create_mask(40//pixel_res + 1),
            padding=40//pixel_res + 1,
            padding_mode='reflect',
            )
        n_features[:, 5:10, :, :] = F.conv2d(
            features[:, 1, :, :].view(-1, 1, h, w),
            self.create_mask(160//pixel_res + 1),
            padding=160//pixel_res + 1,
            padding_mode='reflect',
            )
        n_features[:, 10:15, :, :] = F.conv2d(
            features[:, 2, :, :].view(-1, 1, h, w),
            self.create_mask(320//pixel_res + 1),
            padding=320//pixel_res + 1,
            padding_mode='reflect',
            )
        n_features[:, 15:20, :, :] = F.conv2d(
            features[:, 3, :, :].view(-1, 1, h, w),
            self.create_mask(640//pixel_res + 1),
            padding=640//pixel_res + 1,
            padding_mode='reflect',
            )
        return n_features

    def forward(self, x):
        o1 = self.downsample[0](x)
        o2 = self.downsample[1](o1)
        o3 = self.downsample[2](o2)
        o4 = self.downsample[3](o3)
        u1 = self.upsample[-1](o1)
        u2 = self.upsample[2:](o2)
        u3 = self.upsample[1:](o3)
        u4 = self.upsample[0:](o4)
        res1 = self.pad(u1, x)
        res2 = self.pad(u2, x)
        res3 = self.pad(u3, x)
        res4 = self.pad(u4, x)
        out = th.stack((res1, res2, res3, res4)).view(-1, 4, x.shape[2], x.shape[3])
        return self.last(out)

class SimplerFCNwBottleneck(nn.Module):
    def __init__(self, in_channel):
        super(SimplerFCNwBottleneck, self).__init__()
        self.downsample = InConv(in_channel, 16)
        self.upsample = OutConv(16)
    
    def pad(self, x, xt):
        (_, _, h, w) = x.shape
        (_, _, ht, wt) = xt.shape
        hdif = ht - h
        wdif = wt - w
        x = F.pad(x, (wdif//2, wdif-wdif//2, hdif//2, hdif-hdif//2), mode='reflect')
        return x
    
    def forward(self, x):
        out = self.upsample(self.downsample(x))
        return self.pad(out, x)