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
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class EnhancedSNUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Calculate channels for each level
        self.channels = [config.base_channel * (2**i) for i in range(config.depth)]

        # Initial convolution
        self.init_conv = ConvBlock(config.in_channels, self.channels[0])

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.encoder_pools = nn.ModuleList()

        for i in range(config.depth - 1):
            self.encoder_blocks.append(
                ConvBlock(self.channels[i], self.channels[i + 1])
            )
            self.encoder_pools.append(nn.MaxPool2d(2, 2))

        # Transformer for bottleneck if configured
        if config.use_transformer:
            self.transformer = TransformerBlock(
                dim=self.channels[-1],
                num_heads=config.transformer_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop=config.transformer_dropout,
            )

        # FPN if configured
        if config.use_fpn:
            self.fpn = FPN(self.channels, config.fpn_channels)
            out_channels = config.fpn_channels
        else:
            out_channels = self.channels[0]

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for i in range(config.depth - 1, 0, -1):
            self.upsamples.append(
                nn.ConvTranspose2d(
                    self.channels[i], self.channels[i - 1], kernel_size=2, stride=2
                )
            )
            self.decoder_blocks.append(
                ConvBlock(self.channels[i], self.channels[i - 1])
            )

        # Attention modules
        self.channel_attention = ChannelAttention(out_channels)
        self.spatial_attention = SpatialAttention()

        # Final layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels, config.num_classes, kernel_size=1),
            nn.BatchNorm2d(config.num_classes),
        )

        if config.deep_supervision and config.use_fpn:
            self.deep_outputs = nn.ModuleList(
                [
                    nn.Conv2d(config.fpn_channels, config.num_classes, kernel_size=1)
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
            encoder_features_A.append(xA)
            encoder_features_B.append(xB)
            xA = block(pool(xA))
            xB = block(pool(xB))

        # Add final features
        encoder_features_A.append(xA)
        encoder_features_B.append(xB)

        # Fusion at bottleneck
        x = xA + xB

        # Apply transformer if configured
        if self.config.use_transformer:
            b, c, h, w = x.shape
            x = x.view(b, c, h * w).transpose(1, 2)
            x = self.transformer(x)
            x = x.transpose(1, 2).view(b, c, h, w)

        # FPN if configured
        if self.config.use_fpn:
            combined_features = [
                f_a + f_b for f_a, f_b in zip(encoder_features_A, encoder_features_B)
            ]
            fpn_features = self.fpn(combined_features)

            if self.config.deep_supervision and self.training:
                return [
                    output_conv(feat)
                    for feat, output_conv in zip(fpn_features, self.deep_outputs)
                ]

            x = fpn_features[0]  # Use the finest resolution feature map

        # Apply attention
        x = self.channel_attention(x)
        x = self.spatial_attention(x)

        # Final prediction
        x = self.final_conv(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
