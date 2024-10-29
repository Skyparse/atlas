# src/models/components/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Calculate reduced channels with a minimum of 1
        self.channels = channels
        reduced_channels = max(1, channels // reduction_ratio)

        # Create separate layers instead of Sequential to better control dimensions
        self.fc1 = nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        # Average branch
        avg_pool = self.avg_pool(x)
        avg_out = self.fc1(avg_pool)
        avg_out = self.relu(avg_out)
        avg_out = self.fc2(avg_out)

        # Max branch
        max_pool = self.max_pool(x)
        max_out = self.fc1(max_pool)
        max_out = self.relu(max_out)
        max_out = self.fc2(max_out)

        # Combine branches
        out = torch.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(x_cat))
        return x * out


class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16, spatial_kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction_ratio)
        self.spatial_att = SpatialAttention(spatial_kernel_size)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x
