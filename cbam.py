import os

import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CAM(nn.Module):
    """Channel Attention Module
    """

    def __init__(self, channels, h, w, reduction_ratio=16):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=(h,w), stride=(h,w))
        self.avgpool = nn.AvgPool2d(kernel_size=(h,w), stride=(h,w))

        reduced_channels_num = (channels // reduction_ratio) if (channels >= reduction_ratio) else 1
        pointwise_in = nn.Conv2d(kernel_size=1,
                                 in_channels=channels,
                                 out_channels=reduced_channels_num)
        pointwise_out = nn.Conv2d(kernel_size=1,
                                 in_channels=reduced_channels_num,
                                 out_channels=channels)
        self.MLP = nn.Sequential(
            pointwise_in,
            nn.ReLU(),
            pointwise_out,
        )
        self.sigmoid = nn.Sigmoid()


    def forward(self, input_tensor):
        max_feat = self.maxpool(input_tensor)
        avg_feat = self.avgpool(input_tensor)
        max_feat_mlp = self.MLP(max_feat)
        avg_feat_mlp = self.MLP(avg_feat)
        channel_attention_map = self.sigmoid(max_feat_mlp + avg_feat_mlp)
        return channel_attention_map


class SAM(nn.Module):
    """Spatial Attention Module"""

    def __init__(self, channels, h, w, ks=7):
        super().__init__()
        self.c = channels
        self.h = h
        self.w = w
        self.maxpool = nn.MaxPool1d(kernel_size=channels, stride=channels)
        self.avgpool = nn.AvgPool1d(kernel_size=channels, stride=channels)

        # Handle situation if feature tensor dims are less than default kernel size (which is 7, acc. to paper)
        self.ks = min(h,w) if ((h < 7) or (w < 7)) else ks
        self.sigmoid = nn.Sigmoid()

    def _get_padding(self,
                     dim_in,
                     kernel_size,
                     stride):
        padding = (stride * (dim_in - 1) - dim_in + kernel_size) // 2
        return padding

    def forward(self, input_tensor):
        # Permute input tensor for being able to apply MaxPool and AvgPool along the channel axis
        permuted = input_tensor.view(-1, self.c, self.h * self.w).permute(0,2,1)
        max_feat = self.maxpool(permuted)
        max_feat = max_feat.permute(0,2,1).view(-1, 1, self.h, self.w)

        avg_feat = self.avgpool(permuted)
        avg_feat = avg_feat.permute(0,2,1).view(-1, 1, self.h, self.w)

        concatenated = torch.cat([max_feat, avg_feat], dim=1)
        # Get pad values for SAME padding for conv2d
        h_pad = self._get_padding(concatenated.shape[2], self.ks, 1)
        w_pad = self._get_padding(concatenated.shape[3], self.ks, 1)
        spatial_attention_map = self.sigmoid(
            nn.Conv2d(kernel_size=self.ks, in_channels=2, out_channels=1, padding=(h_pad, w_pad))(concatenated)
        )
        return spatial_attention_map

class CBAM(nn.Module):

    def __init__(self):
        pass

    def forward(self, *input):
        pass