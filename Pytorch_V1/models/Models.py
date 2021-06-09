import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable,Function
import torch.nn.functional as F
from torch.fft import fft,ifft,rfft,irfft
import pdb
#from .NN2D_parts import * # for training and testing
from .NN1D_parts import *
# from NN2D_parts import * # for code debuging 
# from NN1D_parts import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class WienerNet(nn.Module):
    def __init__(self,n_channels,pre_train_para):
        super(WienerNet,self).__init__()
        self.n_channels = n_channels
        if pre_train_para is None:
            self.beta = Variable(torch.tensor(0.5),requires_grad=True)
        else:
            self.beta = pre_train_para
        self.inc = DoubleConv1D(n_channels, 2)
        self.down1 = Down1D(2, 4)
        self.down2 = Down1D(4, 8)
        self.down3 = Down1D(8, 16)
    
    def forward(self, x):
        _, h0 = Wiener_Filter_Tensor(x)
        h = self.inc(h0)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        h = F.max_pool1d(h,kernel_size=4).permute(1,0,2)
        h = torch.mean(h,dim=1,keepdim=True)
        return h, self.beta


class ICNN_s(nn.Module):
    def __init__(self,n_channels, n_classes, bilinear=True,pre_train_para=None):
        super(ICNN_s,self).__init__()
        kernel_size = 2
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.wienernet = WienerNet(1,pre_train_para=pre_train_para)
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

    def weight_teakle(self,k,h,beta):
        h = h.repeat(1,k.size()[1],1)
        return k + beta * h * k

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        k = self.down4.maxpool_conv[1].double_conv[0].weight.data
        h,beta = self.wienernet(x)
        nk = self.weight_teakle(k,h,beta)
        self.down4.maxpool_conv[1].double_conv[0].weight = nn.Parameter(nk)
        
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class ICNN_s_v0(nn.Module):
    def __init__(self,n_channels, n_classes, bilinear=True):
        super(ICNN_s_v0,self).__init__()
        kernel_size = 2
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

class ICNN_m(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(ICNN_m, self).__init__()
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


class ICNN_l(nn.Module):
    def __init__(self,n_channels, n_classes):
        super(ICNN_l,self).__init__()
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

class logistic_F(Function):
    @staticmethod
    def forward(self, x, c):
        #print('loss_forward')
        a = -c.mul(x)
        b = torch.max(a,torch.zeros(a.size()).cuda())
        #b = torch.max(a, torch.zeros(a.size()))
        t = b + torch.log(torch.exp(-b) + torch.exp(a-b))
        t = torch.sum(t)
        #t1 = torch.sum((b>0))
        self.save_for_backward(x, c)
        return t

    @staticmethod
    def backward(self, grad_output):
        #print('loss_backward')
        x,c = self.saved_tensors
        x_grad = c_grad = None
        x_grad = -grad_output*c.div(1+torch.exp(c.mul(x)))
        return x_grad , c_grad


def Wiener_Filter_Tensor(sig,mysize=None,noise=None):
    sig = sig.detach().cpu()
    batch, chanels, length = sig.size()[0], sig.size()[1],sig.size()[-1]
    
    # Intialize kernel
    if mysize is None:
        mysize = [1,sig.size()[1]]
        mysize = np.hstack([mysize,[3] * (sig.dim()-2)]) # kernel size = 3
    kernel = torch.ones(list(mysize),dtype=sig.dtype)

    # Padding the input signal to realize 'same' mode
    pad = [1] * (sig.dim()-2) * 2 
    _sig = F.pad(sig,pad)

    # Estimate the local mean
    lMean = torch.div(F.conv1d(_sig,kernel), torch.sum(kernel,dtype=sig.dtype))
    #print('Mean',lMean)

    # Estimate the local variance
    lVar = torch.div(F.conv1d(torch.pow(_sig,2),kernel), torch.sum(kernel,dtype=sig.dtype)) \
    - torch.pow(lMean,2)
    # print('Variance',lVar)

    if noise is None:
        noise = torch.mean(torch.clone(lVar).view(batch,chanels,-1),axis=-1)
        _noise = torch.clone(noise)
    # print('noise',noise)

    res = sig - lMean
    # print('1',res)
    res *= (1.0 - torch.div(torch.unsqueeze(_noise,dim=-1),lVar))
    # print('2',res)
    res += lMean
    # print('3',res)
    out = torch.where(lVar < noise.repeat(1,1,length).view(batch,chanels,-1), lMean, res)
    # print('4',out.size(),'\n',out)
    
    h_sys_inverse = torch.abs(ifft(fft(out,length,dim=-1)/fft(sig,length,dim=-1),length,dim=-1))
    # print(h_sys_inverse.size(),'\n',h_sys_inverse)
    return out.to(device), h_sys_inverse.to(device)

if __name__=='__main__':
    net = ICNN_s(1,1)
    x = torch.randn(2,1,100)
    y = net(x)
    print(y.size())

    # x = torch.randn(2,1,100)
    # wnet = WienerNet(1)
    # y = wnet(x)
    # print(y.size())
    # sig = torch.tensor([1,2,3,4,5,6,7,8,9,10],dtype=torch.float32).view(2,1,5)
    # # sig = torch.tensor([1,2,3,4,5],dtype=torch.float32).view(-1,1,5)
    # #sig = torch.randn(2,1,5,5)
    # Wiener_Filter_Tensor(sig)