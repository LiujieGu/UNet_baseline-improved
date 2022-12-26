# -*- coding: utf-8 -*- 
# @Time : 12/6/2022 4:07 PM 
# @Author : Liujie Gu
# @E-mail : georgeglj21@gmail.com
# @Site :  
# @File : datasets.py 
# @Software: VScode

import os
from shutil import copy, rmtree
import random


def mk_file(file_path: str):
    if os.path.exists(file_path):
        rmtree(file_path)
    os.makedirs(file_path)


def main():
    random.seed(0)

    split_rate = 0.05
    cwd = os.getcwd()
    data_root = 'D:\Tsinghua\SparseSampling'
    origin_figure_path = os.path.join(data_root, "DATASET")
    assert os.path.exists(origin_figure_path)
    class_fig = [cla for cla in os.listdir(origin_figure_path)
                    if os.path.isdir(os.path.join(origin_figure_path, cla))]

    train_root = os.path.join(data_root, "train")
    mk_file(train_root)
    for cla in class_fig:
        mk_file(os.path.join(train_root, cla))

    val_root = os.path.join(data_root, "test")
    mk_file(val_root)
    for cla in class_fig:
        mk_file(os.path.join(val_root, cla))

    cla = 'gt'
    cla_path = os.path.join(origin_figure_path, cla)
    clb = 'sparse'
    clb_path = os.path.join(origin_figure_path, clb)
    images = os.listdir(cla_path)
    num = len(images)
    eval_index = random.sample(images, k=int(num*split_rate))
    for index, image in enumerate(images):
        if image in eval_index:
            image_path = os.path.join(cla_path, image)
            image_path2 = os.path.join(clb_path, image)
            new_path = os.path.join(val_root, cla)
            new_path2 = os.path.join(val_root,clb)
            copy(image_path, new_path)
            copy(image_path2,new_path2)
        else:
            image_path = os.path.join(cla_path, image)
            image_path2 = os.path.join(clb_path, image)
            new_path = os.path.join(train_root, cla)
            new_path2 = os.path.join(train_root,clb)
            copy(image_path, new_path)
            copy(image_path2,new_path2)
        print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar
    print()

    print("processing done!")


if __name__ == '__main__':
    main()
