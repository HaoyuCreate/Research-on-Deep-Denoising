#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
import torch.nn.functional as F
import pdb


# In[31]:


class DoubleConv1D(nn.Module):
    '''(convolution => [BN] => ReLU) * 2'''
    
    def __init__(self, in_channels,out_channels,mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels,mid_channels,kernel_size=3,padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self,x):
        return self.double_conv(x)


# In[32]:


class Down1D(nn.Module):
    '''Downscaling with maxpool then double conv'''
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv1D(in_channels,out_channels)
        )
    def forward(self,x):
        return self.maxpool_conv(x)


# In[45]:


class Up1D(nn.Module):
    '''Upscaling then double conv'''
    def __init__(self,in_channels,out_channels,linear=True):
        super().__init__()
        
        if linear:
            self.up = nn.Upsample(scale_factor=2,mode='linear',align_corners=True)
            self.conv=DoubleConv1D(in_channels,out_channels,in_channels//2)
        else:
            self.up=nn.ConvTranspose1d(in_channels,in_channels//2,kernels_size=2,stride=2)
            self.conv = DoubleConv1D(in_channels,out_channels)
    
    def forward(self,x1,x2):
        x1 = self.up(x1)
        
        if x1.size()[2] <= x2.size()[2]:
            diff = x2.size()[2] - x1.size()[2]
            x1 = F.pad(x1,[diff //2, diff - diff//2])
        else:
            diff = x1.size()[2] - x2.size()[2]
            x2 = F.pad(x2,[diff //2, diff - diff//2])
        
        x = torch.cat([x2,x1],dim=1)
        return self.conv(x)


# In[ ]:


class OutConv1D(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(OutConv1D,self).__init__()
        self.conv = nn.Conv1d(in_channels,out_channels,kernel_size=1)
        
    def forward(self,x):
        return self.conv(x)

