import torch
import torch.nn as nn
import torch.nn.functional as F


class CAM(nn.Module):
    """Channel Attention Module
    """

    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()

        reduced_channels_num = (in_channels // reduction_ratio) if (in_channels >= reduction_ratio) else 1
        pointwise_in = nn.Conv2d(kernel_size=1,
                                 in_channels=in_channels,
                                 out_channels=reduced_channels_num)
        pointwise_out = nn.Conv2d(kernel_size=1,
                                 in_channels=reduced_channels_num,
                                 out_channels=in_channels)
        self.MLP = nn.Sequential(
            pointwise_in,
            nn.ReLU(),
            pointwise_out,
        )
        self.sigmoid = nn.Sigmoid()


    def forward(self, input_tensor):
        h, w = input_tensor.size(2), input_tensor.size(3)
        max_feat = F.max_pool2d(input_tensor, kernel_size=(h, w), stride=(h, w))
        avg_feat = F.avg_pool2d(input_tensor, kernel_size=(h, w), stride=(h, w))
        max_feat_mlp = self.MLP(max_feat)
        avg_feat_mlp = self.MLP(avg_feat)
        channel_attention_map = self.sigmoid(max_feat_mlp + avg_feat_mlp)
        return channel_attention_map


class SAM(nn.Module):
    """Spatial Attention Module"""

    def __init__(self, ks=7):
        super().__init__()
        self.ks = ks
        self.sigmoid = nn.Sigmoid()

    def _get_padding(self,
                     dim_in,
                     kernel_size,
                     stride):
        padding = (stride * (dim_in - 1) - dim_in + kernel_size) // 2
        return padding

    def forward(self, input_tensor):
        c, h, w = input_tensor.size(1), input_tensor.size(2), input_tensor.size(3)

        # Permute input tensor for being able to apply MaxPool and AvgPool along the channel axis
        permuted = input_tensor.view(-1, c, h * w).permute(0,2,1)
        max_feat = F.max_pool1d(permuted, kernel_size=c, stride=c)
        max_feat = max_feat.permute(0,2,1).view(-1, 1, h, w)

        avg_feat = F.avg_pool1d(permuted, kernel_size=c, stride=c)
        avg_feat = avg_feat.permute(0,2,1).view(-1, 1, h, w)

        concatenated = torch.cat([max_feat, avg_feat], dim=1)
        # Get pad values for SAME padding for conv2d
        h_pad = self._get_padding(concatenated.shape[2], self.ks, 1)
        w_pad = self._get_padding(concatenated.shape[3], self.ks, 1)
        # Handle situation if feature tensor dims are less than default kernel size (which is 7, acc. to paper)
        self.ks = min(h, w) if ((h < 7) or (w < 7)) else self.ks
        spatial_attention_map = self.sigmoid(
            nn.Conv2d(kernel_size=self.ks, in_channels=2, out_channels=1, padding=(h_pad, w_pad))(concatenated)
        )
        return spatial_attention_map

class CBAM(nn.Module):
    """Convolutional Block Attention Module
    https://eccv2018.org/openaccess/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf
    """

    def __init__(self, in_channels):
        super().__init__()
        self.CAM = CAM(in_channels)
        self.SAM = SAM()

    def forward(self, input_tensor):
        channel_att_map = self.CAM(input_tensor)
        gated_tensor = torch.mul(input_tensor, channel_att_map)
        spatial_att_map = self.SAM(gated_tensor)
        refined_tensor = torch.mul(gated_tensor, spatial_att_map)
        return refined_tensor
