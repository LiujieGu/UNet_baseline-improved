# -*- coding: utf-8 -*- 
# @Time : 12/19/2022 4:07 PM 
# @Author : Liujie Gu
# @E-mail : georgeglj21@gmail.com
# @Site :  
# @File : datasets.py 
# @Software: VScode

import glob
import random
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)

        self.files_sparse_sampling = sorted(glob.glob(os.path.join(root, '%s/sparse' % mode) + '/*.*'))
        self.files_ground_truth    = sorted(glob.glob(os.path.join(root, '%s/gt' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_SS              = self.transform(float2int(Image.open(self.files_sparse_sampling[index % len(self.files_sparse_sampling)])))
        item_ground_truth    = self.transform(float2int(Image.open(self.files_ground_truth[index % len(self.files_ground_truth)])))
        
        
        return {'SS':item_SS , 'GT': item_ground_truth}

    def __len__(self):
        return max(len(self.files_sparse_sampling), len(self.files_ground_truth))

def float2int(fig):
    fig = np.array(fig)
    fig = np.uint8(np.interp(fig, (fig.min(), fig.max()), (0, 255)))
    fig = Image.fromarray(fig)
    return fig