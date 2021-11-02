# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 14:37:06 2021

@author: baekd
"""

import torch
import numpy as np
from torch.nn import functional as F


class HVSKim(object) :
    def __init__(self,device) :
        N = 23
        S = 3500.0
        D = S/600.0
        
        cpp1 = np.zeros((2*N+1,2*N+1),dtype=float)
        cpp2 = np.zeros((2*N+1,2*N+1),dtype=float)
        self.N = N
        
        for i in range(2*N+1) :
            for j in range(2*N+1) :
                m = i-N
                n = j-N
                x = 180.0*m/(np.pi*S)
                y = 180.0*n/(np.pi*S)
                A = (180.0**2)/((np.pi*D)**2)
                r = x**2+y**2
                
                k1 = 43.2
                k2 = 38.7
                s1 = 0.0219**2
                s2 = 0.0598**2
                
                cpp1[i,j] = A*(k1*np.exp(-r/2/s1)+k2*np.exp(-r/2/s2))
        
        for i in range(2*N+1) :
            for j in range(2*N+1) :
                m = i-N
                n = j-N
                x = 180.0*m/(np.pi*S)
                y = 180.0*n/(np.pi*S)
                A = (180.0**2)/((np.pi*D)**2)
                r = x**2+y**2
                
                k1 = 19.1
                k2 = 42.7
                s1 = 0.0330**2
                s2 = 0.0569**2
                
                cpp2[i,j] = A*(k1*np.exp(-r/2/s1)+k2*np.exp(-r/2/s2))
        
        cpp1 = cpp1/np.sum(cpp1)
        cpp2 = cpp2/np.sum(cpp2)
        
        cpp1 = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(cpp1),dim=0),dim=0)
        cpp2 = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(cpp2),dim=0),dim=0)
        self.cpp1 = cpp1.to(torch.float32).to(device)
        self.cpp2 = cpp2.to(torch.float32).to(device)
    
    def error(self,img,halftone) :
        
        pd = self.N

        b,ch,H,W = img.shape
        
        mask1 = (img<0.25).to(torch.float32)
        mask2 = ((img>=0.25).to(torch.float32))*((img<0.75).to(torch.float32))
        mask3 = (img>0.75).to(torch.float32)
        img1 = img*mask1
        img2 = img*mask2
        img3 = img*mask3
        
        w1 = torch.sqrt((1.0-torch.pow((4.0*img1-1.0),2))*mask1)+\
            torch.abs(4.0*img2-2.0)*mask2+\
            torch.sqrt((1.0-torch.pow((4.0*img3-3.0),2))*mask3)
        w2 = 1-w1
        
        error_img1 = w1*(img-halftone)
        error_img2 = w2*(img-halftone)
        
        error_img1p = F.pad(error_img1,(pd,pd,pd,pd),mode='circular')
        error_img2p = F.pad(error_img2,(pd,pd,pd,pd),mode='circular')
        
        img1f = F.conv2d(error_img1p,self.cpp1)
        img2f = F.conv2d(error_img2p,self.cpp2)
        
        return torch.sum(img1f*error_img1+img2f*error_img2)/H/W
    
    
    
    
    
    
    
    