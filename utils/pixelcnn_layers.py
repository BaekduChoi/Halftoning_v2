"""
    PixelCNN layers from https://github.com/tuelwer/conditional-pixelcnn-pytorch/tree/0c37e0df6a77f1bf8b2f61e702d021e6e1190225
"""

from torch import nn


class MaskedConv2d(nn.Conv2d):
    """
    Implementation by jzbontar/pixelcnn-pytorch
    
    mask_type: must be 'A' or 'B' (see [1])
    """
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in ['A', 'B']
        self.register_buffer('mask', self.weight.data.clone())
        h = self.weight.size()[2]
        w = self.weight.size()[3]
        self.mask.fill_(1)
        self.mask[:, :, h // 2, w // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, h // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)
    

class GatedMaskedConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super(GatedMaskedConv2d, self).__init__()
        self.masked_conv_1 = MaskedConv2d(*args, **kwargs)
        self.masked_conv_2 = MaskedConv2d(*args, **kwargs)
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        """
        x: input
        """
        inp = self.tanh(self.masked_conv_1(x))
        gate = self.sigm(self.masked_conv_2(x))
        return inp*gate

class _GatedMaskedConv2d(nn.Module):
    def __init__(self,mask_type,in_ch,out_ch,ksize,stride=1,padding=0,padding_mode='zeros'):
        super(GatedMaskedConv2d, self).__init__()
        self.masked_conv = MaskedConv2d(
            mask_type,
            in_channels=in_ch,
            out_channels=2*out_ch,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode)
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()
        self.out_ch = out_ch

    def forward(self, x):
        """
        x: input
        """
        x = self.masked_conv(x)
        x1 = x[:,:self.out_ch,:,:]
        x2 = x[:,self.out_ch:,:,:]

        inp = self.tanh(x1)
        gate = self.sigm(x2)
        return inp*gate

class CondGatedMaskedConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super(CondGatedMaskedConv2d, self).__init__()
        self.masked_conv_1 = MaskedConv2d(*args, **kwargs)
        self.masked_conv_2 = MaskedConv2d(*args, **kwargs)
        self.cond_conv_1 = nn.Conv2d(1, args[2], 1)
        self.cond_conv_2 = nn.Conv2d(1, args[2], 1)
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()

    def forward(self, x, h):
        """
        x: input
        h: conditional input (should have the same shape as input)
        """
        inp = self.tanh(self.masked_conv_1(x)
                        + self.cond_conv_1(h))
        gate = self.sigm(self.masked_conv_2(x)
                         + self.cond_conv_2(h))
        return inp * gate
    

