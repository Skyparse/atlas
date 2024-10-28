# src/models/enhanced_sunet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .components.attention import ChannelAttention, SpatialAttention
from .components.transformer import TransformerBlock
from .components.fpn import FPN


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class EnhancedSNUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Initial convolution
        self.init_conv = ConvBlock(config.in_channels, config.base_channel)

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.encoder_pools = nn.ModuleList()
        current_channels = config.base_channel

        for i in range(config.depth):
            self.encoder_blocks.append(
                ConvBlock(current_channels, current_channels * 2)
            )
            self.encoder_pools.append(nn.MaxPool2d(2, 2))
            current_channels *= 2

        # Transformer block for bottleneck
        if config.use_transformer:
            self.transformer = TransformerBlock(
                dim=current_channels,
                num_heads=config.transformer_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop=config.transformer_dropout,
                attn_drop=config.transformer_dropout,
            )

        # FPN
        if config.use_fpn:
            in_channels_list = [
                config.base_channel * (2**i) for i in range(config.depth)
            ]
            self.fpn = FPN(in_channels_list, config.fpn_channels)

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.decoder_ups = nn.ModuleList()

        for i in range(config.depth - 1, -1, -1):
            self.decoder_ups.append(
                nn.ConvTranspose2d(current_channels, current_channels // 2, 2, stride=2)
            )
            self.decoder_blocks.append(
                ConvBlock(current_channels, current_channels // 2)
            )
            current_channels //= 2

        # Attention modules
        self.channel_attention = ChannelAttention(config.base_channel)
        self.spatial_attention = SpatialAttention()

        # Final layers
        self.final_conv = nn.Conv2d(config.base_channel, config.num_classes, 1)

        if config.deep_supervision:
            self.deep_outputs = nn.ModuleList(
                [
                    nn.Conv2d(config.fpn_channels, config.num_classes, 1)
                    for _ in range(config.depth)
                ]
            )

    def forward(self, xA, xB):
        # Initial convolution
        xA = self.init_conv(xA)
        xB = self.init_conv(xB)

        # Encoder
        encoder_features_A = []
        encoder_features_B = []

        for block, pool in zip(self.encoder_blocks, self.encoder_pools):
            xA = block(xA)
            xB = block(xB)
            encoder_features_A.append(xA)
            encoder_features_B.append(xB)
            xA = pool(xA)
            xB = pool(xB)

        # Fusion at bottleneck
        x = xA + xB

        # Apply transformer if configured
        if self.config.use_transformer:
            b, c, h, w = x.shape
            x = x.flatten(2).transpose(1, 2)  # (B, HW, C)
            x = self.transformer(x)
            x = x.transpose(1, 2).reshape(b, c, h, w)  # (B, C, H, W)

        # FPN if configured
        if self.config.use_fpn:
            fpn_features = self.fpn(
                [f_a + f_b for f_a, f_b in zip(encoder_features_A, encoder_features_B)]
            )

            if self.config.deep_supervision and self.training:
                return [
                    output_conv(feat)
                    for feat, output_conv in zip(fpn_features, self.deep_outputs)
                ]

        # Decoder
        for up, block in zip(self.decoder_ups, self.decoder_blocks):
            x = up(x)
            f_a = encoder_features_A.pop()
            f_b = encoder_features_B.pop()
            x = torch.cat([x, f_a + f_b], dim=1)
            x = block(x)

        # Apply attention
        x = self.channel_attention(x)
        x = self.spatial_attention(x)

        # Final convolution
        x = self.final_conv(x)

        return x
