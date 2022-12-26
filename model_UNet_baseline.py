# -*- coding: utf-8 -*- 
# @Time : 12/19/2022 4:07 PM 
# @Author : Liujie Gu
# @E-mail : georgeglj21@gmail.com
# @Site :  
# @File : model.py 
# @Software: VScode

import torch.nn as nn
from collections import OrderedDict

def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv1", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn1", nn.BatchNorm2d(filter_out)),
        ("relu1", nn.LeakyReLU(0.2)),
        ("conv2", nn.Conv2d(filter_out, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn2", nn.BatchNorm2d(filter_out)),
        ("relu2", nn.LeakyReLU(0.2)),
    ]))


class unet(nn.Module):
    def __init__(self,in_channel):
        super(unet,self).__init__()
        # Encoder
        self.in_channel = in_channel
        self.conv1   = conv2d(self.in_channel,64,3)
        self.pool1   = nn.MaxPool2d(2,stride=2)
        self.conv2   = conv2d(64,128,3)
        self.pool2   = nn.MaxPool2d(2,stride=2)
        self.conv3   = conv2d(128,256,3)
        self.pool3   = nn.MaxPool2d(2,stride=2)
        self.conv4   = conv2d(256,512,3)
        self.pool4   = nn.MaxPool2d(2,stride=2)
        self.conv5   = conv2d(512,1024,3)
        # Decoder
        self.conv6   = conv2d(1024,512,3)
        self.unpool1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv7   = conv2d(512,256,3)
        self.unpool2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv8   = conv2d(256,128,3)
        self.unpool3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv9   = conv2d(128,64,3)
        self.unpool4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv10  = conv2d(64,1,3)
    def forward(self,x):
        # encoder
        x1  = self.conv1(x)      #b,64,512,512
        x2  = self.pool1(x1)  
        x2  = self.conv2(x2)     #b,128,256,256
        x3  = self.pool2(x2)  
        x3  = self.conv3(x3)     #b,256,128,128
        x4  = self.pool3(x3)  
        x4  = self.conv4(x4)     #b,512,64,64
        x5  = self.pool4(x4)  
        x5  = self.conv5(x5)     #b,1024,32,32
        # skip connection & decoder
        x6  = self.unpool1(x5)   
        x7  = self.conv6(x6)+x4  #b,512,64,64 
        x8  = self.unpool2(x7)
        x8  = self.conv7(x8)+x3  #b,256,128,128
        x9  = self.unpool3(x8)
        x9  = self.conv8(x9)+x2  #b,128,256,256
        x10 = self.unpool4(x9)
        x10 = self.conv9(x10)+x1 #b,64,512,512
        out = self.conv10(x10)
        return nn.Tanh()(out)