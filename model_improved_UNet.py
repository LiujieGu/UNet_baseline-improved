# -*- coding: utf-8 -*- 
# @Time : 12/6/2022 4:07 PM 
# @Author : Liujie Gu
# @E-mail : georgeglj21@gmail.com
# @Site :  
# @File : model.py 
# @Software: VScode

import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange
from einops.layers.torch import Rearrange


class ffNet(nn.Module):
    def __init__(self,input_nc):
        super(ffNet,self).__init__()
        layer_1 = [   nn.Conv2d(input_nc ,16 , 3 , stride=2 , padding=1 , bias=True ),
                      nn.InstanceNorm2d(16),
                      nn.LeakyReLU(0.2, inplace=True),
                      nn.Conv2d(16 ,16 , 3 , stride=1 , padding=1 , bias=True ),
                      nn.InstanceNorm2d(16),
                      nn.LeakyReLU(0.2, inplace=True)
                      ] #16.256,256

        layer_2 = [  nn.Conv2d(16 ,32 , 3 , stride=2 , padding=1 , bias=True ),
                      nn.InstanceNorm2d(32),
                      nn.LeakyReLU(0.2, inplace=True),
                      nn.Conv2d(32 ,32 , 3 , stride=1 , padding=1 , bias=True ),
                      nn.InstanceNorm2d(32),
                      nn.LeakyReLU(0.2, inplace=True)] #32,128,128

        layer_3 = [   nn.Conv2d(32 ,64 , 3 , stride=2 , padding=1 , bias=True ),
                        nn.InstanceNorm2d(64),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(64 ,64 , 3 , stride=1 , padding=1 , bias=True ),
                        nn.InstanceNorm2d(64),
                        nn.LeakyReLU(0.2, inplace=True)] #64,64,64

        layer_4 = [   nn.Conv2d(64 ,128 , 3 , stride=2 , padding=1 , bias=True ),
                        nn.InstanceNorm2d(128),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(128 ,128 , 3 , stride=1 , padding=1 , bias=True ),
                        nn.InstanceNorm2d(128),
                        nn.LeakyReLU(0.2, inplace=True)] #128,32,32
        layer_5 = [   nn.Conv2d(128 ,256 , 3 , stride=2 , padding=1 , bias=True ),
                        nn.InstanceNorm2d(256),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Conv2d(256 ,256 , 3 , stride=1 , padding=1 , bias=True ),
                        nn.InstanceNorm2d(256),
                        nn.LeakyReLU(0.2, inplace=True)] #128,32,32
        self.layer_1 = nn.Sequential(*layer_1)
        self.layer_2 = nn.Sequential(*layer_2)
        self.layer_3 = nn.Sequential(*layer_3)
        self.layer_4 = nn.Sequential(*layer_4)
        self.layer_5 = nn.Sequential(*layer_5)
    def forward(self,x):
        x1 = self.layer_1(x)
        x2 = self.layer_2(x1) #32,128,128
        x3 = self.layer_3(x2)
        x4 = self.layer_4(x3)
        x5 = self.layer_5(x4)
        return x1,x2,x3,x4,x5
class upsNet(nn.Module):
    def __init__(self):
        super(upsNet,self).__init__()
        model1 = []
        in_features = 256
        out_features = 128
        for _ in range(1):
            model1 += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2
        
        model2 = []
        for _ in range(1):
            model2 += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        model3 = []
        for _ in range(1):
            model3 += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        model4 = []
        for _ in range(1):
            model4 += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2
        model5 = []
        for _ in range(1):
            model5 += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2
        model6 = [  nn.ReflectionPad2d(1),
                    nn.Conv2d(8, 1, 3),
                    nn.Tanh() ]
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
    def forward(self,x1,x2,x3,x4,x5):
        y1 = self.model1(x5)
        y2 = self.model2(x4+y1)
        y3 = self.model3(x3+y2)
        y4 = self.model4(x2+y3)
        y5 = self.model5(x1+y4)
        y6 = self.model6(y5)
        return y6


class UNet(nn.Module):
    def __init__(self,input_nc):
        super(UNet,self).__init__()
        self.Lfnet = ffNet(input_nc)
        self.upsample= upsNet()
    def forward(self,FF):
        x1,x2,x3,x4,x5 = self.Lfnet(FF)
        x = self.upsample(x1,x2,x3,x4,x5)
        return x

# net = SSFormer(1)
# x1 = torch.ones(8,1,512,512)
# x2 = torch.ones(8,1,512,512)
# y = net(x1,x2)
# print('done')