# -*- coding: utf-8 -*- 
# @Time : 12/19/2022 4:07 PM 
# @Author : Liujie Gu
# @E-mail : georgeglj21@gmail.com
# @Site :  
# @File : train.py 
# @Software: VScode

#!/usr/bin/python3

import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from model import unet
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from datasets import ImageDataset

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs of training')
    parser.add_argument('--batchSize', type=int, default=2, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='datasets', help='root directory of the dataset')
    parser.add_argument('--lr', type=float, default=0.00005, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=200, help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--size', type=int, default=512, help='size of the data crop (squared assumed)')
    parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
    parser.add_argument('--cuda', action='store_true',default='True', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    opt = parser.parse_args()
    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    ###### Definition of variables ######
    # Networks
    net = unet(opt.input_nc)

    if opt.cuda:
        net.cuda()

    net.apply(weights_init_normal)
   

    # Lossess
    criterion = torch.nn.MSELoss()

    # Optimizers & LR schedulers
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
    input_C = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

    

    # Dataset loader
    # transforms_ = [ transforms.Resize(int(opt.size*1.12), Image.BICUBIC), 
    #                 transforms.RandomCrop(opt.size), 
    #                 transforms.RandomHorizontalFlip(),
    #                 transforms.ToTensor(),
    #                 transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    transforms_ = [transforms.ToTensor(),transforms.Normalize((0.5), (0.5)) ]
    dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_), 
                            batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

    # Loss plot
    logger = Logger(opt.n_epochs, len(dataloader))
    ###################################

    ###### Training ######
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):
            # Set model input
            A = Variable(input_A.copy_(batch['SS']))
            C = Variable(input_C.copy_(batch['GT']))
            optimizer.zero_grad()
            B = net(A)
            loss = criterion(B,C)
            
            
            loss.backward()
            
            optimizer.step()
            
            ###################################

            # Progress report (http://localhost:8097)
            logger.log({'loss': loss }, 
                        images={'SS': A, 'output': B,'target':C})

        # Update learning rates
        lr_scheduler.step()


        # Save models checkpoints
    torch.save(net.state_dict(), 'output/net_epoch%d.pth'% (epoch+1))
    ###################################

