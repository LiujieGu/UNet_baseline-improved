# -*- coding: utf-8 -*- 
# @Time : 12/6/2022 4:07 PM 
# @Author : Liujie Gu
# @E-mail : georgeglj21@gmail.com
# @Site :  
# @File : test.py 
# @Software: VScode

#!/usr/bin/python3

import argparse
import sys
import os
import pandas as pd
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from metrics.evaluate import compute_mse,compute_ssim,compute_psnr
from model import unet
from datasets import ImageDataset
import numpy as np
from PIL import Image

def float2int(fig):
    fig = np.array(fig)
    fig = np.uint8(np.interp(fig, (fig.min(), fig.max()), (0, 255)))
    fig = Image.fromarray(fig)
    return fig
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='datasets/', help='root directory of the dataset')
    parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
    parser.add_argument('--size', type=int, default=512, help='size of the data (squared assumed)')
    parser.add_argument('--cuda', action='store_true',default='True',help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
    parser.add_argument('--generator_A2B', type=str, default='output/net_epoch150.pth', help='checkpoint file')
    opt = parser.parse_args()
    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    ###### Definition of variables ######
    # Networks
    # net = SSFormer(opt.input_nc)
    net = unet(opt.input_nc)

    if opt.cuda:
        net.cuda()

    # Load state dicts
    net.load_state_dict(torch.load(opt.generator_A2B))

    # Set model's test mode
    net.eval()

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
    input_C = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

    # Dataset loader
    # 
    transforms_ = [transforms.ToTensor(),transforms.Normalize((0.5), (0.5))]
    dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='test'), 
                            batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
    ###################################

    ###### Testing######

    # Create output dirs if they don't exist
    if not os.path.exists('output/A'):
        os.makedirs('output/A')
    if not os.path.exists('output/B'):
        os.makedirs('output/B')
    MSE_test = []
    SSIM_test = []
    PSNR_test = []
    for i, batch in enumerate(dataloader):
        # Set model input
        A = Variable(input_A.copy_(batch['SS']))
        C = Variable(input_C.copy_(batch['GT']))
        
        # Generate output
        # output = 0.5*(net(A,A_l).data + 1.0)
        # target = 0.5*(C+1.0)
        output = 0.5*(net(A).data + 1.0)
        target = 0.5*(C+1.0)

        # evaluation metrics
        X = output.cpu().numpy().squeeze()
        Y = target.cpu().numpy().squeeze()
        MSE_test.append(compute_mse(X,Y))
        SSIM_test.append(compute_ssim(X,Y)[0])
        x = float2int(X)
        y = float2int(Y)
        PSNR_test.append(compute_psnr(X,Y,255))
        # Save image files
        save_image(output, 'test/output%04d.png' % (i+1))
        save_image(target, 'test/target%04d.png' % (i+1))

        sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))
    data = {'MSE':MSE_test,'SSIM':SSIM_test,'PSNR':PSNR_test}
    df = pd.DataFrame(data)
    df.to_excel("metrics_ssformer.xls")
    sys.stdout.write('\n')
###################################
