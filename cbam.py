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

    def __init__(self):
        pass

    def forward(self, *input):
        pass


class CBAM(nn.Module):

    def __init__(self):
        pass

    def forward(self, *input):
        pass