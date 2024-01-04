import numpy as np
import torch
import torch.nn as nn
#from FOD.DCNv2 import DeformableConv2d as dcn_v2

class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x

class FeatureFusion(nn.Module):
    def __init__(self,ch_ins,ch_out,di=False):
        super(FeatureFusion,self).__init__()
        self.conv1x1 = nn.ModuleList([])
        self.ch_out = ch_out
        for ch in ch_ins:
            res = ch
            ch = ch+self.ch_out
            if di ==True:
                ch_out = res
            self.conv1x1.append(nn.Conv2d(ch, ch_out, kernel_size=1, stride=1, padding=0))
        
    def forward(self,x,y):
        results = []
        for conv, x_, y_ in zip(self.conv1x1,x,y):
            results.append(conv(torch.cat([x_,y_],dim=1)))
        return results
    
class Fusion(nn.Module):
    def __init__(self, resample_dim):
        super(Fusion, self).__init__()
        self.res_conv1 = ResidualConvUnit(resample_dim)
        self.res_conv2 = ResidualConvUnit(resample_dim)
        #self.resample = nn.ConvTranspose2d(resample_dim, resample_dim, kernel_size=2, stride=2, padding=0, bias=True, dilation=1, groups=1)

    def forward(self, x, previous_stage=None):
        if previous_stage == None:
            previous_stage = torch.zeros_like(x)
        output_stage1 = self.res_conv1(x)
        output_stage1 += previous_stage
        output_stage2 = self.res_conv2(output_stage1)
        output_stage2 = nn.functional.interpolate(output_stage2, scale_factor=2, mode="bilinear", align_corners=True)
        return output_stage2

class NDFusion(nn.Module):
    def __init__(self, resample_dim):
        super(NDFusion, self).__init__()
        self.res_conv1 = ResidualConvUnit(resample_dim)
        self.res_conv2 = ResidualConvUnit(resample_dim)
        #self.resample = nn.ConvTranspose2d(resample_dim, resample_dim, kernel_size=2, stride=2, padding=0, bias=True, dilation=1, groups=1)

    def forward(self, x, previous_stage=None, index=None):
        if previous_stage == None:
            previous_stage = torch.zeros_like(x)
        output_stage1 = self.res_conv1(x)
        output_stage1 += previous_stage
        output_stage2 = self.res_conv2(output_stage1)
        if index==0:
            output_stage2 = nn.functional.interpolate(output_stage2, scale_factor=2, mode="bilinear", align_corners=True)
        return output_stage2