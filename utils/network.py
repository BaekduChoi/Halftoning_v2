import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'.'))

from torch import nn
import torch
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.utils import spectral_norm as SN

from blocks import *
from pixelcnn_layers import *

"""
    Discriminator for GAN-based implementation
    Added Dropout for slower discriminator fitting
"""
class Discriminator2(nn.Module) :
    def __init__(self,in_ch=1,ndf=64) :
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,ndf,kernel_size=1,stride=1),
            nn.LeakyReLU(0.2,True),
            nn.Dropout(),
            nn.Conv2d(ndf,ndf,kernel_size=2,stride=2,bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2,True),
            nn.Dropout(),
            nn.Conv2d(ndf,ndf*2,kernel_size=2,stride=2,bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2,True),
            nn.Dropout(),
            nn.Conv2d(ndf*2,1,kernel_size=1,stride=1)
        )
    
    def forward(self,x,noise=0.0) :
        return self.block(x+noise*torch.randn_like(x))

"""
    Discriminator for GAN-based implementation
    Based on the PatchGAN structure in https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    Added Dropout for slower discriminator fitting
"""
class Discriminator(nn.Module) :
    def __init__(self,in_ch=1,ndf=64) :
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,ndf,kernel_size=4,stride=2,padding=1,padding_mode='circular'),
            nn.LeakyReLU(0.2,True),
            nn.Dropout(),
            nn.Conv2d(ndf,ndf*2,kernel_size=4,stride=2,padding=1,bias=False,padding_mode='circular'),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2,True),
            nn.Dropout(),
            nn.Conv2d(ndf*2,ndf*4,kernel_size=4,stride=2,padding=1,bias=False,padding_mode='circular'),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2,True),
            nn.Dropout(),
            nn.Conv2d(ndf*4,ndf*8,kernel_size=4,stride=1,padding=1,bias=False,padding_mode='circular'),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2,True),
            nn.Dropout(),
            nn.Conv2d(ndf*8,1,kernel_size=4,stride=1,padding=1,padding_mode='circular')
        )
    
    def forward(self,x,noise=0.0) :
        return self.block(x+noise*torch.randn_like(x))

class SimpleNetDenseINDepth(nn.Module) :
    def __init__(self,in_ch=1,out_nch=1,ndf=32,ksize=7,depth=4) :
        super().__init__()

        self.in_ch = in_ch
        self.out_nch = out_nch
        self.ndf = ndf

        padding = (ksize-1)//2

        self.inblock = nn.Sequential(
            nn.Conv2d(in_ch,self.ndf,1),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
        )

        L = [ConvBlockINEDense(self.ndf,act='relu',ksize=ksize) for i in range(depth)]
        self.block = nn.Sequential(*L)

        self.blockout = nn.Sequential(
            nn.Conv2d(self.ndf,self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.out_nch,1),
            nn.Sigmoid()
        )

    def forward(self,x) :

        x1 = self.inblock(x)
        x2 = self.block(x1)

        return self.blockout(x2)

class SimpleNetDenseINDepth3(nn.Module) :
    def __init__(self,in_ch=1,out_nch=1,ndf=32,ksize=7,depth=6) :
        super().__init__()

        self.in_ch = in_ch
        self.out_nch = out_nch
        self.ndf = ndf

        padding = (ksize-1)//2

        self.inblock = nn.Sequential(
            nn.Conv2d(in_ch,self.ndf,1),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
        )

        L1 = [ConvBlockINEDense(self.ndf,act='relu',ksize=ksize) for i in range(depth-2)]
        self.block1 = nn.Sequential(*L1)

        L2 = [ConvBlockINEDense(self.ndf,act='relu',ksize=ksize) for i in range(2)]
        self.block2 = nn.Sequential(*L2)

        self.blockmid = nn.Sequential(
            nn.Conv2d(self.ndf,self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.out_nch,1),
            nn.Sigmoid()
        )

        self.blockout = nn.Sequential(
            nn.Conv2d(self.ndf,self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.out_nch,1),
            nn.Sigmoid()
        )

    def forward(self,x) :

        x1 = self.inblock(x)
        x2 = self.block1(x1)
        x3 = self.block2(x2)

        return self.blockout(x3), self.blockmid(x2)

class SimpleNetDenseLN(nn.Module) :
    def __init__(self,in_ch=1,out_nch=1,ndf=32,ksize=7,depth=6) :
        super().__init__()

        self.in_ch = in_ch
        self.out_nch = out_nch
        self.ndf = ndf

        padding = (ksize-1)//2

        self.inblock = nn.Sequential(
            nn.Conv2d(in_ch,self.ndf//2,1),
            nn.ReLU(True),
            nn.Conv2d(self.ndf//2,self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
        )

        L1 = [ConvBlockLNEDense(self.ndf,act='relu',ksize=ksize) for i in range(depth-2)]
        self.block1 = nn.Sequential(*L1)

        L2 = [ConvBlockLNEDense(self.ndf,act='relu',ksize=ksize) for i in range(2)]
        self.block2 = nn.Sequential(*L2)

        self.blockmid = nn.Sequential(
            nn.Conv2d(self.ndf,self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.ndf//2,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(self.ndf//2,self.out_nch,1),
            nn.Sigmoid()
        )

        self.blockout = nn.Sequential(
            nn.Conv2d(self.ndf,self.ndf,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.ndf//2,ksize,1,padding,padding_mode='circular'),
            nn.ReLU(True),
            nn.Conv2d(self.ndf//2,self.out_nch,1),
            nn.Sigmoid()
        )

    def forward(self,x) :

        x1 = self.inblock(x)
        x2 = self.block1(x1)
        x3 = self.block2(x2)

        return self.blockout(x3), self.blockmid(x2)

def init_weights_small(m) :
    L = list(m.modules())

    for layer in L :
        if isinstance(layer,nn.Sequential) :
            for idx in range(len(layer)) :
                init_weights_small(layer[idx])
        elif isinstance(layer,nn.Conv2d) :
            layer.weight.data.normal_(0.0,0.001)
            layer.bias.data.fill_(0)
        elif isinstance(layer,DenseWSAN) :
            init_weights_small(layer.block)

def init_weights_kaiming(m) :
    L = list(m.modules())

    for layer in L :
        if isinstance(layer,nn.Sequential) :
            for idx in range(len(layer)) :
                init_weights_kaiming(layer[idx])
        elif isinstance(layer,nn.Conv2d) :
            init.kaiming_normal_(layer.weight,mode='fan_out',nonlinearity='relu')
            layer.bias.data.fill_(0)
        elif isinstance(layer,DenseWSAN) :
            init_weights_kaiming(layer.block)

class FeatureExtractor(nn.Module) :
    def __init__(self,in_ch=1,ndf=32,ksize=7,depth=6,padding_mode='zeros') :
        super().__init__()

        self.in_ch = in_ch
        self.ndf = ndf

        padding = (ksize-1)//2

        self.inblock = nn.Sequential(
            nn.Conv2d(in_ch,self.ndf,1),
            nn.ReLU(True),
            nn.Conv2d(self.ndf,self.ndf,ksize,1,padding,padding_mode=padding_mode),
            nn.ReLU(True),
        )

        L1 = [ConvBlockINEDense(self.ndf,act='relu',ksize=ksize,padding_mode=padding_mode) for i in range(depth)]
        self.block1 = nn.Sequential(*L1)    

    def forward(self,x) :

        x1 = self.inblock(x)
        x2 = self.block1(x1)

        return x2

class PixCNNPrior(nn.Module) :
    def __init__(self,feature_input_dim=32,out_ch=1,ksize=3,depth=4,padding_mode='zeros') :
        super().__init__()

        in_ch = feature_input_dim+out_ch
        self.ndf = 256

        padding = (ksize-1)//2

        L2 = [GatedMaskedConv2d('A',in_ch,self.ndf,ksize,1,padding,padding_mode=padding_mode)]
        for _ in range(depth-1) :
            L2 += [GatedMaskedConv2d('B',self.ndf,self.ndf,ksize,1,padding,padding_mode=padding_mode)]
        L2 += [nn.Conv2d(self.ndf,out_ch,1),nn.Sigmoid()]

        self.block = nn.Sequential(*L2)
    
    def forward(self,feature,halftone) :
        x = torch.cat([feature,halftone],dim=1)
        return self.block(x)