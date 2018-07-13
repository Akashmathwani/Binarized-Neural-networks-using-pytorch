# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 12:29:33 2018

@author: akash
"""

import math
import torch.nn.parameter as Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F

from functions import *


class BinaryTanh(nn.Module):
    def __init__(self):
        super(BinaryTanh, self).__init__()
        self.hardtanh = nn.Hardtanh()

    def forward(self, input):
        output = self.hardtanh(input)
        output = binarize(output)
        return output
        

class BinaryLinear(nn.Linear):

    def forward(self, input):
        binary_weight = binarize(self.weight)
        #if input.size(1) != 784:
        input.data=binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=binarize(self.weight.org)
        out = nn.functional.linear(input, self.weight)
        #print("((((((((((((( ",input.data)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out
        if self.bias is None:
            return F.linear(input, binary_weight)
        else:
            return F.linear(input, binary_weight, self.bias)

    def reset_parameters(self):
        # Glorot initialization
        in_features, out_features = self.weight.size()
        stdv = math.sqrt(1.5 / (in_features + out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

        self.weight.lr_scale = 1. / stdv
        
        """
        
class BinaryLSTM(nn.LSTM):
    def forward(self, input, hx):
        bwi = binarize(self.weight_ih)
        bwh = binarize(self.weight_hh)
        return nn.LSTM(input,hx,bwi,bwh,bias_ih=True,bias_hh=True  )

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


"""
class BinConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinConv2d, self).__init__(*kargs, **kwargs)


    def forward(self, input):
        if input.size(1) != 3:
            input.data = binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=binarize(self.weight.org)

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

"""
        
class BinGRU(nn.GRU):
    def __init__(self, *kargs, **kwargs):  
        super(BinGRU, self).__init__(*kargs, **kwargs)
    
    
    def forward(self, input,hx):
        input.data = binarize(input.data)
        hx.data = binarize(hx.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=binarize(self.weight.org)
        #self.w_ih.data=binarize(self.weight.org)
        #self.w_hh.data=binarize(self.weight.org)
        

        out = self._backend.GRUCell(input,hx, self.weight,self.weight,None, None,)
        
        return out
   
        if not self.bias is None:
            self.bias_ih.org=self.bias_ih.data.clone()
            self.bias_hh.org=self.bias_hh.data.clone()
            out += self.bias_ih.view(1, -1, 1, 1).expand_as(out)
            out += self.bias_hh.view(1, -1, 1, 1).expand_as(out)
            
            
"""

