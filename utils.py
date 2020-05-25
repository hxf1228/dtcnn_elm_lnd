"""Some useful tools

Created on: 10/10/2019 17:22

@File: utils.py
@Author：Xufeng Huang (xufenghuang1228@gmail.com & xfhuang@umich.edu)
@Copy Right: Copyright © 2019-2020 HUST. All Rights Reserved.
@Requirement: [ Python-3.6.9~3.7.4, scikit-image-0.15.0 ]

"""
import itertools  
from matplotlib import pyplot as plt  
from skimage import transform 
from skimage import io
import os  
import glob  
import numpy as np 
np.random.seed(66)


# padding to same size
def img_zeroPadding(padSize, dir, save_dir):
    files = os.listdir(dir)
    files.sort(key=lambda x: int(x[0:1])) 
    cate = [dir + x for x in files if os.path.isdir(dir + x)]
    
    for _ , folder in enumerate(cate):
        print('Preprossing images from: %s' % folder)
        (_ ,class_dir) = os.path.split(folder)
        save_dir_1 = os.path.join(save_dir, class_dir)
        make_dir(save_dir_1)
        imgs_path = os.path.join(folder, '*.png')
        imageList = io.ImageCollection(imgs_path)
        for iImage in range(0,len(imageList)):
            img_pre = imageList[iImage]
            pad_w = int((padSize - img_pre.shape[0])/2)
            img_pad = np.pad(img_pre, pad_width=pad_w, mode='constant', constant_values=0)              
            io.imsave(save_dir_1+os.sep+str(iImage+1)+'.png', img_pad)
        print('Total: %s' % len(imageList))
        print('Save to: %s' % save_dir_1)
        print('--------------------------------------------')

    

# load data from .npz
def img_to_npz(path, npz_path, img_rows, img_channels, img_suffix):
    files = os.listdir(path)
    files.sort(key=lambda x: int(x[0:1]))
    cate = [path + x for x in files if os.path.isdir(path + x)]
    print(cate)
    imgs = []
    labels = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(
                folder + '\*' + img_suffix): 
            img = io.imread(im)
            print('image: %s -----label: %s' % (im,idx))  
            img_cols = img_rows
            img = transform.resize(img, (img_rows, img_cols, img_channels))
            imgs.append(img)  
            labels.append(idx)  
    return np.savez(npz_path, imgs=imgs, labels=labels)


# load data from imgs
def read_img(path, img_rows, img_channels, img_suffix):
    files = os.listdir(path)
    files.sort(key=lambda x: int(x[0:1]))  

    cate = [path + x for x in files if os.path.isdir(path + x)]
    print(cate)
    imgs = []
    labels = []

    for idx, folder in enumerate(cate):
        for im in glob.glob(
                folder + '\*' + img_suffix):  
            img = io.imread(im)
            print('reading the images:%s' % (im))  
            img_cols = img_rows
            img = transform.resize(img, (img_rows, img_cols, img_channels))
            imgs.append(img)  
            labels.append(idx)  
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


# make directory
def make_dir(create_dir):
    if not os.path.isdir(create_dir):
        os.makedirs(create_dir)
