from torch import nn
import torch
from torch.nn import functional as F
from torch.nn.utils import spectral_norm as SN

"""
    Conditional instance normalization layer
"""
class ConditionalInstanceNorm(nn.Module) :
    def __init__(self,num_ch) :
        super().__init__()
        self.num_ch = num_ch

    def forward(self,x,gamma,beta) :
        x = F.instance_norm(x)

        g = gamma.view(x.size(0),self.num_ch,1,1)
        b = beta.view(x.size(0),self.num_ch,1,1)

        return x*g+b

"""
    Conv block
"""
class ConvBlock(nn.Module) :
    def __init__(self,in_ch,out_ch,cat=False,norm='bn',act='relu',ksize=3) :
        super().__init__()

        self.cat = cat
        if self.cat and in_ch != out_ch :
            raise ValueError('channel numbers for input and output do not match')

        if norm == 'bn' :
            self.norm = nn.BatchNorm2d(out_ch)
        elif norm == 'in' :
            self.norm = nn.InstanceNorm2d(out_ch,affine=True)
        elif norm == 'cin' :
            self.norm = ConditionalInstanceNorm(out_ch)
        else :
            self.norm = None
        self.normtype = norm

        padding = (ksize-1)//2
        
        if act == 'lrelu' :
            self.act = nn.LeakyReLU(0.2,True)
        else : # default = ReLU
            self.act = nn.ReLU(True)
            
        self.conv1 = nn.Conv2d(in_ch,out_ch,kernel_size=ksize,padding=padding,padding_mode='circular')
        self.conv2 = nn.Conv2d(out_ch,out_ch,kernel_size=ksize,padding=padding,padding_mode='circular')
    
    def forward(self,x,g=None,b=None) :
        x1 = self.conv1(x)
        x1 = self.act(x1)
        x1 = self.conv2(x1)
        x1 = self.act(x1)

        if self.norm is not None :
            x1 = self.norm(x1,g,b) if self.normtype == 'cin' else self.norm(x1)

        out = torch.cat([x,x1],dim=1) if self.cat else x1

        return out

"""
    Conv block w Downscaling x2
"""
class DSConvBlock(nn.Module) :
    def __init__(self,in_ch,out_ch,norm='bn',act='relu',ds='mp',ksize=3) :
        super().__init__()

        if norm == 'bn' :
            self.norm = nn.BatchNorm2d(out_ch)
        elif norm == 'in' :
            self.norm = nn.InstanceNorm2d(out_ch,affine=True)
        else :
            self.norm = None
        
        if act == 'lrelu' :
            self.act = nn.LeakyReLU(0.2,True)
        else : # default = ReLU
            self.act = nn.ReLU(True)
        
        padding = (ksize-1)//2

        self.ds = nn.MaxPool2d(2) if ds == 'mp' \
            else nn.Conv2d(in_ch,in_ch,kernel_size=3,stride=2,padding=1,padding_mode='circular')
        self.conv1 = nn.Conv2d(in_ch,out_ch,kernel_size=ksize,padding=padding,padding_mode='circular')
        self.conv2 = nn.Conv2d(out_ch,out_ch,kernel_size=ksize,padding=padding,padding_mode='circular')
    
    def forward(self,x,g=None,b=None) :
        x1 = self.ds(x)
        x1 = self.conv1(x1)
        x1 = self.act(x1)
        x1 = self.conv2(x1)
        x1 = self.act(x1)

        if self.norm is not None :
            out = self.norm(x1)
        else :
            out = x1

        return out

"""
    Conv block w Upscaling x2
"""
class USConvBlock(nn.Module) :
    def __init__(self,in_ch,out_ch,norm='bn',act='relu',ksize=3) :
        super().__init__()

        if norm == 'bn' :
            self.norm = nn.BatchNorm2d(out_ch)
        elif norm == 'in' :
            self.norm = nn.InstanceNorm2d(out_ch,affine=True)
        else :
            self.norm = None
        
        if act == 'lrelu' :
            self.act = nn.LeakyReLU(0.2,True)
        else : # default = ReLU
            self.act = nn.ReLU(True)
        
        padding = (ksize-1)//2

        self.us = nn.ConvTranspose2d(out_ch,out_ch,kernel_size=2,stride=2)
        self.conv1 = nn.Conv2d(in_ch,out_ch,kernel_size=ksize,padding=padding,padding_mode='circular')
        self.conv2 = nn.Conv2d(out_ch,out_ch,kernel_size=ksize,padding=padding,padding_mode='circular')
    
    def forward(self,x,g=None,b=None) :
        x1 = self.act(self.conv1(x))
        x1 = self.act(self.conv2(x1))
        x1 = self.act(self.us(x1))

        if self.norm is not None :
            out = self.norm(x1)
        else :
            out = x1

        return out

class ConvCatBlockCIN(nn.Module) :
    def __init__(self,ch_num) :
        super().__init__()
        self.conv1 = nn.Conv2d(ch_num,ch_num,3,1,1,padding_mode='circular')
        self.conv2 = nn.Conv2d(ch_num,ch_num,3,1,1,padding_mode='circular')
        self.relu = nn.ReLU(True)
    
    def forward(self,x,gamma1,beta1,gamma2,beta2) :
        x1 = self.conv1(x)
        x2 = F.instance_norm(x1)

        g1 = gamma1.view(x.size(0),-1,1,1)
        b1 = beta1.view(x.size(0),-1,1,1)

        x3 = F.instance_norm(self.conv2(self.relu(x2*g1+b1)))

        g2 = gamma2.view(x.size(0),-1,1,1)
        b2 = beta2.view(x.size(0),-1,1,1)

        return torch.cat([x,x3*g2+b2],dim=1)

"""
    Conv block w spectral norm
"""
class SNConvBlock(nn.Module) :
    def __init__(self,in_ch,out_ch,cat=False,norm=None,act='relu',ksize=7) :
        super().__init__()

        self.cat = cat
        if self.cat and in_ch != out_ch :
            raise ValueError('channel numbers for input and output do not match')

        if norm == 'bn' :
            self.norm = nn.BatchNorm2d(out_ch)
        elif norm == 'in' :
            self.norm = nn.InstanceNorm2d(out_ch,affine=True)
        elif norm == 'cin' :
            self.norm = ConditionalInstanceNorm(out_ch)
        else :
            self.norm = None
        self.normtype = norm
        
        if act == 'lrelu' :
            self.act = nn.LeakyReLU(0.2,True)
        else : # default = ReLU
            self.act = nn.ReLU(True)
        
        padding = (ksize-1)//2
            
        self.conv1 = SN(nn.Conv2d(in_ch,out_ch,kernel_size=ksize,padding=padding,padding_mode='circular'))
        self.conv2 = SN(nn.Conv2d(out_ch,out_ch,kernel_size=ksize,padding=padding,padding_mode='circular'))
    
    def forward(self,x,g=None,b=None) :
        x1 = self.conv1(x)
        x1 = self.act(x1)
        x1 = self.conv2(x1)
        x1 = self.act(x1)

        if self.norm is not None :
            x1 = self.norm(x1,g,b) if self.normtype == 'cin' else self.norm(x1)

        out = torch.cat([x,x1],dim=1) if self.cat else x1

        return out

# attention module code based on https://github.com/voletiv/self-attention-GAN-pytorch
class Attention(nn.Module) :
    def __init__(self,in_ch,sn=True) :
        super().__init__()

        self.in_ch = in_ch
        self.mid_ch1 = in_ch//8
        self.mid_ch2 = in_ch//2

        self.wf = nn.Conv2d(self.in_ch,self.mid_ch1,1)
        self.wg = nn.Conv2d(self.in_ch,self.mid_ch1,1)
        self.wh = nn.Conv2d(self.in_ch,self.mid_ch2,1)
        self.wv = nn.Conv2d(self.mid_ch2,self.in_ch,1)

        if sn :
            self.wf = SN(self.wf)
            self.wg = SN(self.wg)
            self.wh = SN(self.wh)
            self.wv = SN(self.wv)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.mp = nn.MaxPool2d(2,stride=2,padding=0)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x) :

        b,ch,he,wi = x.size()
        f = self.wf(x).view(b,ch//8,he*wi)
        g = self.mp(self.wg(x)).view(b,ch//8,he*wi//4)
        attn = self.softmax(torch.bmm(f.permute(0,2,1),g))
        h = self.mp(self.wh(x)).view(b,ch//2,he*wi//4)
        attn_g = torch.bmm(h,attn.permute(0,2,1)).view(b,ch//2,he,wi)
        v = self.wv(attn_g)

        return self.gamma*v+x

"""
    Conv block
"""
class ConvBlockINE(nn.Module) :
    def __init__(self,in_ch,out_ch,act='relu',ksize=3) :
        super().__init__()

        padding = (ksize-1)//2
        
        if act == 'lrelu' :
            self.act = nn.LeakyReLU(0.2,True)
        else : # default = ReLU
            self.act = nn.ReLU(True)
            
        self.conv1 = nn.Conv2d(in_ch,out_ch,kernel_size=ksize,padding=padding,padding_mode='circular')
        self.conv2 = nn.Conv2d(out_ch,out_ch,kernel_size=ksize,padding=padding,padding_mode='circular')

        self.norm1 = nn.InstanceNorm2d(out_ch,affine=True)
        self.norm2 = nn.InstanceNorm2d(out_ch,affine=True)
    
    def forward(self,x,g=None,b=None) :
        x1 = self.conv1(x)
        x1 = self.act(x1)
        x1 = self.norm1(x1)
        x1 = self.conv2(x1)
        x1 = self.act(x1)
        out = self.norm2(x1)

        return out

class ConvBlockINEDense(nn.Module) :
    def __init__(self,n_ch,act='relu',ksize=3,norm='in',padding_mode='circular') :
        super().__init__()

        padding = (ksize-1)//2
        
        if act == 'lrelu' :
            self.act = nn.LeakyReLU(0.2,True)
        else : # default = ReLU
            self.act = nn.ReLU(True)
            
        self.conv1 = nn.Conv2d(n_ch,n_ch,kernel_size=ksize,padding=padding,padding_mode=padding_mode)
        self.conv2 = nn.Conv2d(2*n_ch,n_ch,kernel_size=ksize,padding=padding,padding_mode=padding_mode)
        self.conv3 = nn.Conv2d(3*n_ch,n_ch,kernel_size=ksize,padding=padding,padding_mode=padding_mode)
        self.conv4 = nn.Conv2d(4*n_ch,n_ch,kernel_size=ksize,padding=padding,padding_mode=padding_mode)
        
        self.norm = norm
        if norm == 'in' :
            self.norm1 = nn.InstanceNorm2d(n_ch,affine=True)
            self.norm2 = nn.InstanceNorm2d(n_ch,affine=True)
            self.norm3 = nn.InstanceNorm2d(n_ch,affine=True)
    
    def forward(self,x,g=None,b=None) :
        x1 = self.conv1(x)
        x1 = self.act(x1)
        if self.norm == 'in' :
            x1 = self.norm1(x1)

        x2 = torch.cat([x1,x],dim=1)
        x2 = self.conv2(x2)
        x2 = self.act(x2)
        if self.norm == 'in' :
            x2 = self.norm2(x2)

        x3 = torch.cat([x2,x1,x],dim=1)
        x3 = self.conv3(x3)
        x3 = self.act(x3)
        if self.norm == 'in' :
            x3 = self.norm3(x3)

        x4 = torch.cat([x3,x2,x1,x],dim=1)
        out = self.conv4(x4)

        return out

class ConvBlockLNEDense(nn.Module) :
    def __init__(self,n_ch,act='relu',ksize=3) :
        super().__init__()

        padding = (ksize-1)//2
        
        if act == 'lrelu' :
            self.act = nn.LeakyReLU(0.2,True)
        else : # default = ReLU
            self.act = nn.ReLU(True)
            
        self.conv1 = nn.Conv2d(n_ch,n_ch,kernel_size=ksize,padding=padding,padding_mode='circular')
        self.conv2 = nn.Conv2d(2*n_ch,n_ch,kernel_size=ksize,padding=padding,padding_mode='circular')
        self.conv3 = nn.Conv2d(3*n_ch,n_ch,kernel_size=ksize,padding=padding,padding_mode='circular')
        self.conv4 = nn.Conv2d(4*n_ch,n_ch,kernel_size=ksize,padding=padding,padding_mode='circular')
        
        self.norm1 = nn.GroupNorm(1,n_ch,affine=True)
        self.norm2 = nn.GroupNorm(1,n_ch,affine=True)
        self.norm3 = nn.GroupNorm(1,n_ch,affine=True)
    
    def forward(self,x,g=None,b=None) :
        x1 = self.conv1(x)
        x1 = self.act(x1)
        x1 = self.norm1(x1)

        x2 = torch.cat([x1,x],dim=1)
        x2 = self.conv2(x2)
        x2 = self.act(x2)
        x2 = self.norm2(x2)

        x3 = torch.cat([x2,x1,x],dim=1)
        x3 = self.conv3(x3)
        x3 = self.act(x3)
        x3 = self.norm3(x3)

        x4 = torch.cat([x3,x2,x1,x],dim=1)
        out = self.conv4(x4)

        return out

class ConvBlockINEDenseSN(nn.Module) :
    def __init__(self,n_ch,act='relu',ksize=3,norm='in') :
        super().__init__()

        padding = (ksize-1)//2
        
        if act == 'lrelu' :
            self.act = nn.LeakyReLU(0.2,True)
        else : # default = ReLU
            self.act = nn.ReLU(True)
            
        self.conv1 = SN(nn.Conv2d(n_ch,n_ch,kernel_size=ksize,padding=padding,padding_mode='circular'))
        self.conv2 = SN(nn.Conv2d(2*n_ch,n_ch,kernel_size=ksize,padding=padding,padding_mode='circular'))
        self.conv3 = SN(nn.Conv2d(3*n_ch,n_ch,kernel_size=ksize,padding=padding,padding_mode='circular'))
        self.conv4 = SN(nn.Conv2d(4*n_ch,n_ch,kernel_size=ksize,padding=padding,padding_mode='circular'))
        
        self.norm = norm
        if norm == 'in' :
            self.norm1 = nn.InstanceNorm2d(n_ch,affine=True)
            self.norm2 = nn.InstanceNorm2d(n_ch,affine=True)
            self.norm3 = nn.InstanceNorm2d(n_ch,affine=True)
    
    def forward(self,x,g=None,b=None) :
        x1 = self.conv1(x)
        x1 = self.act(x1)
        if self.norm == 'in' :
            x1 = self.norm1(x1)

        x2 = torch.cat([x1,x],dim=1)
        x2 = self.conv2(x2)
        x2 = self.act(x2)
        if self.norm == 'in' :
            x2 = self.norm2(x2)

        x3 = torch.cat([x2,x1,x],dim=1)
        x3 = self.conv3(x3)
        x3 = self.act(x3)
        if self.norm == 'in' :
            x3 = self.norm3(x3)

        x4 = torch.cat([x3,x2,x1,x],dim=1)
        out = self.conv4(x4)

        return out

"""
    Dense block w. spatially-adaptive conditional normalization
    similar to that in SPADE (https://github.com/NVlabs/SPADE/blob/master/)
"""
class DenseWSAN(nn.Module) :
    def __init__(self,block,instance_norm=True,n_ch=None) :
        super().__init__()
        self.block = block
        self.instance_norm = instance_norm
        if self.instance_norm :
            self.norm = nn.InstanceNorm2d(n_ch)
    
    def forward(self,x,g,b) :
        x = self.block(x)
        if self.instance_norm :
            x = self.norm(x)
        return x*(1+g)+b

"""
    Pixelnorm borrowed from https://github.com/huangzh13/StyleGAN.pytorch/blob/master/models/CustomLayers.py
    Added affine transformation after normalization
"""
class PixelNormAffine(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(1.0)
        self.beta = nn.Parameter(0.0)

    def forward(self, x):
        return self.gamma * (x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)) + self.beta