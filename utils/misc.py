#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 16:35:06 2020

@author: baekduchoi
"""

"""
    Script for miscellaneous functions used
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'.'))

import json
import torch
from torch.utils.data import DataLoader, ConcatDataset
from data import HalftoneDataset, screenImage, readScreen
from torch.nn import functional as F

import cv2
import scipy.signal
import numpy as np

"""
    Function that reads the json file and generates the dataloader to be used
    Only generates training and validation dataloader
"""
def create_dataloaders(params,random_screen=False) :
    train_img_root = params["datasets"]["train"]["root_img"]
    train_halftone_root = params["datasets"]["train"]["root_halftone"]
    batch_size = int(params["datasets"]["train"]["batch_size"])
    train_img_type = params['datasets']['train']['img_type']
    n_workers = int(params['datasets']['train']['n_workers'])
    train_use_aug = params['datasets']['train']['use_aug']
    
    val_img_root = params["datasets"]["val"]["root_img"]
    val_halftone_root = params["datasets"]["val"]["root_halftone"]
    val_img_type = params['datasets']['val']['img_type']
    
    train_dataset = HalftoneDataset(train_img_root,
                                        train_halftone_root,
                                        train_img_type,
                                        train_use_aug,
                                        random_screen)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  num_workers=n_workers,
                                  shuffle=True)
    
    # no need to use augmentation for validation data
    val_dataset = HalftoneDataset(val_img_root,
                                        val_halftone_root,
                                        val_img_type,
                                        random_screen)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=1,
                                num_workers=n_workers,
                                shuffle=False)
    
    return train_dataloader, val_dataloader

"""
    Added extra smooth patch image / halftone pairs
"""
def create_dataloaders_extra(params) :
    train_img_root = params["datasets"]["train"]["root_img"]
    train_halftone_root = params["datasets"]["train"]["root_halftone"]
    batch_size = int(params["datasets"]["train"]["batch_size"])
    train_img_type = params['datasets']['train']['img_type']
    n_workers = int(params['datasets']['train']['n_workers'])
    train_use_aug = params['datasets']['train']['use_aug']
    
    val_img_root = params["datasets"]["val"]["root_img"]
    val_halftone_root = params["datasets"]["val"]["root_halftone"]
    val_img_type = params['datasets']['val']['img_type']
    
    train_dataset1 = HalftoneDataset(train_img_root,
                                        train_halftone_root,
                                        train_img_type,
                                        train_use_aug)
    train_dataset2 = HalftoneDataset('./img_patch/',
                                        './halftone_patch/',
                                        '.png',
                                        train_use_aug)
    train_dataset = ConcatDataset([train_dataset1,train_dataset2])
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  num_workers=n_workers,
                                  shuffle=True)
    
    # no need to use augmentation for validation data
    val_dataset = HalftoneDataset(val_img_root,
                                        val_halftone_root,
                                        val_img_type)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=1,
                                num_workers=n_workers,
                                shuffle=False)
    
    return train_dataloader, val_dataloader

"""
    Function that reads the components of the json file and returns a dataloader for test dataset
    Refer to test_naive.json for the structure of json file
    For test dataset we do not use data augmentation

    params : output of read_json(json_file_location)
"""
def create_test_dataloaders(params) :
    test_img_root = params["datasets"]["test"]["root_img"]
    test_halftone_root = params["datasets"]["test"]["root_halftone"]
    test_img_type = params['datasets']['test']['img_type']
    n_workers = int(params['datasets']['test']['n_workers'])
    
    test_dataset = HalftoneDataset(test_img_root,
                                    test_halftone_root,
                                    test_img_type,
                                    False)
    test_dataloader = DataLoader(test_dataset,
                                  batch_size=1,
                                  num_workers=n_workers,
                                  shuffle=False)
    
    return test_dataloader

"""
    Function that reads the json file
"""
def read_json(json_dir) : 
    with open(json_dir,'r') as f :
        params = json.load(f)
    return params

"""
    Nasanen's HVS model
"""
class HVS(object) :
    
    def __init__(self) :
        N = 23
        c = 0.525
        d = 3.91
        G = 11.0
        pi = np.pi
        fs = pi*3500.0/180.0
        k = fs/(c*np.log(G)+d)
        
        self.hvs = np.zeros((2*N+1,2*N+1))
        
        for i in range(2*N+1) :
            for j in range(2*N+1) :
                m = i-N
                n = j-N
                
                denom = ((k**2)+4.0*(pi**2)*((m**2)+(n**2)))**1.5                
                val = 2.0*pi*k/denom
                
                dist = (float(m)**2.0+float(n)**2.0)**0.5
                if dist > float(N) :
                    self.hvs[i][j] = 0.0
                else :
                    self.hvs[i][j] = val*(float(N)+1-dist)
                
        # print(np.sum(self.hvs)**2)
        self.hvs = self.hvs/np.sum(self.hvs)
        self.N = N
    
    def __getitem__(self, keys) :
        m = keys[0]+self.N
        n = keys[1]+self.N
        return self.hvs[m][n]
    
    def getHVS(self) :
        return self.hvs.astype(np.float32)
    
    def size(self) :
        return self.hvs.shape

"""
    HVS error loss function
"""
def HVSloss(img1,img2,hvs) :
    k = hvs.size(2)
    M = img1.size(2)
    N = img1.size(3)

    pd = (k-1)//2

    img1p = F.pad(img1,(pd,pd,pd,pd),mode='circular')
    img2p = F.pad(img2,(pd,pd,pd,pd),mode='circular')
    img1_filtered = F.conv2d(img1p,hvs)
    img2_filtered = F.conv2d(img2p,hvs)

    return F.mse_loss(img1_filtered,img2_filtered)

"""
    HVS error loss function
"""
def HVSlossL1(img1,img2,hvs) :
    k = hvs.size(2)
    M = img1.size(2)
    N = img1.size(3)

    pd = (k-1)//2

    img1p = F.pad(img1,(pd,pd,pd,pd),mode='circular')
    img2p = F.pad(img2,(pd,pd,pd,pd),mode='circular')
    img1_filtered = F.conv2d(img1p,hvs)
    img2_filtered = F.conv2d(img2p,hvs)

    return F.l1_loss(img1_filtered,img2_filtered)

if __name__ == '__main__' :
    img_id = str(int(np.floor(np.random.random()*10000)))
    print(img_id)

    hvs = HVS()
    img_name = './images_div2k_all/'+img_id+'.png'
    halftone_name = './halftones_div2k_all/'+img_id+'h.png'

    img = cv2.imread(img_name,0).astype(np.float32)/255.0
    imgH = cv2.imread(halftone_name,0).astype(np.float32)/255.0
    imgS = screenImage(img,readScreen())

    img_hvs = scipy.signal.correlate2d(img,hvs.getHVS()\
                                           ,mode='same',boundary='wrap')
    imgH_hvs = scipy.signal.correlate2d(imgH,hvs.getHVS()\
                                           ,mode='same',boundary='wrap')
    imgS_hvs = scipy.signal.correlate2d(imgS,hvs.getHVS()\
                                           ,mode='same',boundary='wrap')
    
    E = np.sum(np.power(img_hvs-imgH_hvs,2))
    Es = np.sum(np.power(img_hvs-imgS_hvs,2))
    print(E.dtype)
    print(E/256/256)
    print(Es/256/256)

    img = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(img),0),0)
    imgH = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(imgH),0),0)

    hvs = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(hvs.getHVS().astype(np.float32)),0),0)
    E2 = HVSloss(img,imgH,hvs).item()

    print((E-E2*256*256)/E)

