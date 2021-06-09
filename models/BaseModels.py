import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
import torch.nn.functional as F
from torch.autograd import Function
from .NN2D_parts import * # for training and testing
from .NN1D_parts import *
# from NN2D_parts import * # for code debuging 
# from NN1D_parts import *


class EncoderDecoder(nn.Module):
    def __init__(self,n_channels, n_classes):
        super(EncoderDecoder,self).__init__()
        kernel_size = 2
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.Encoder = nn.Sequential(
            nn.Conv1d(n_channels,3,kernel_size=kernel_size,padding=1),
            nn.BatchNorm1d(3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(3,5,kernel_size=kernel_size,padding=1),
            nn.BatchNorm1d(5),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
            )
        self.MidCNN = nn.Sequential(
            nn.Conv1d(5,8,kernel_size=kernel_size),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.Conv1d(8,12,kernel_size=kernel_size),
            nn.BatchNorm1d(12),
            nn.ReLU(inplace=True)
            )
        self.Decoder = nn.Sequential(
            nn.Conv1d(12,5,kernel_size=kernel_size,padding=1),
            nn.BatchNorm1d(5),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True),
            nn.Conv1d(5,3,kernel_size=kernel_size,padding=1),
            nn.BatchNorm1d(3),
            nn.Upsample(scale_factor=2,mode='linear', align_corners=True)
            )
        self.OutConv = nn.Conv1d(3,n_classes,kernel_size=1,padding=1)

    def forward(self,x):
        x = self.Encoder(x)
        x = self.MidCNN(x)
        x = self.Decoder(x)
        x = self.OutConv(x)
        return x

class EncoderDecoder_ss(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(EncoderDecoder_ss, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv1D(n_channels, 2)
        self.down1 = Down1D(2, 4)
        self.down2 = Down1D(4, 8)
        self.down3 = Down1D(8, 16)
        factor = 2 if bilinear else 1
        self.down4 = Down1D(16, 32 // factor)
        self.up1 = Up1D(32, 16 // factor, bilinear)
        self.up2 = Up1D(16, 8 // factor, bilinear)
        self.up3 = Up1D(8, 4 // factor, bilinear)
        self.up4 = Up1D(4, 2, bilinear)
        self.outc = OutConv1D(2, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class EncoderDecoder_s(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(EncoderDecoder_s, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv1D(n_channels, 4)
        self.down1 = Down1D(4, 8)
        self.down2 = Down1D(8, 16)
        self.down3 = Down1D(16, 32)
        factor = 2 if bilinear else 1
        self.down4 = Down1D(32, 64 // factor)
        self.up1 = Up1D(64, 32 // factor, bilinear)
        self.up2 = Up1D(32, 16 // factor, bilinear)
        self.up3 = Up1D(16, 8 // factor, bilinear)
        self.up4 = Up1D(8, 4, bilinear)
        self.outc = OutConv1D(4, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class EncoderDecoder_m(nn.Module):
    def __init__(self,n_channels, n_classes):
        super(EncoderDecoder_m,self).__init__()
        kernel_size = 2
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.Encoder = nn.Sequential(
            nn.Conv1d(n_channels,8,kernel_size=kernel_size,padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(8,32,kernel_size=kernel_size,padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
            )
        self.MidCNN = nn.Sequential(
            nn.Conv1d(32,64,kernel_size=kernel_size),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64,128,kernel_size=kernel_size),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128,256,kernel_size=kernel_size),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
            )
        self.Decoder = nn.Sequential(
            nn.Conv1d(256,64,kernel_size=kernel_size,padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64,16,kernel_size=kernel_size,padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True),
            nn.Conv1d(16,3,kernel_size=kernel_size,padding=1),
            nn.BatchNorm1d(3),
            nn.Upsample(scale_factor=2,mode='linear', align_corners=True)
            )
        self.OutConv = nn.Conv1d(3,n_classes,kernel_size=1,padding=1)

    def forward(self,x):
        x = self.Encoder(x)
        x = self.MidCNN(x)
        x = self.Decoder(x)
        x = self.OutConv(x)
        return x

if __name__=='__main__':
    net = EncoderDecoder1(1,1)
    x = torch.randn(2,1,100)
    y = net(x)
    print(y.size())