# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from abc import ABCMeta, abstractmethod

class BaseComplexConv(nn.Module, metaclass=ABCMeta):
    _dtype_mapping = {torch.complex64: torch.float, torch.complex128: torch.double, torch.complex32: torch.half}
    
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, device=None, dtype=None):
        super().__init__()
        conv_factory = self._convolution_factory()
        
        dtype = self._dtype_mapping.get(dtype, dtype)
        self.conv_re = conv_factory(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, device=device, dtype=dtype)
        self.conv_im = conv_factory(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, device=device, dtype=dtype)
        
    def forward(self, x): # shape of x : [batch,channel,axis, ...]
        real = self.conv_re(x.real) - self.conv_im(x.imag)
        imaginary = self.conv_re(x.imag) + self.conv_im(x.real)
        output = torch.cat((real.unsqueeze_(-1), imaginary.unsqueeze_(-1)), dim=-1)
        return torch.view_as_complex(output)
    
    @abstractmethod
    def _convolution_factory(self):
        return None
    
class ComplexConv1d(BaseComplexConv):
    
    def _convolution_factory(self):
        return nn.Conv1d
    
class ComplexConv2d(BaseComplexConv):
    
    def _convolution_factory(self):
        return nn.Conv2d
    
class ComplexConv3d(BaseComplexConv):
    
    def _convolution_factory(self):
        return nn.Conv3d
    
#%%
if __name__ == "__main__":
    ## Random Tensor for Input
    ## shape : [batchsize,channel,axis1_size,axis2_size]
    ## Below dimensions are totally random
    x = torch.randn((10,3,100,100), dtype=torch.cfloat)
    
    # 1. Make ComplexConv Object
    ## (in_channel, out_channel, kernel_size) parameter is required
    complexConv = ComplexConv2d(3,10,(5,5))
    
    # 2. compute
    y = complexConv(x)

