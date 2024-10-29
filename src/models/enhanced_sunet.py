# src/models/enhanced_sunet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .components.attention import CBAM
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
            self.out_channels = config.fpn_channels
        else:
            self.out_channels = self.channels[
                0
            ]  # Final output channels will be same as initial

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        current_channels = self.channels[-1]  # Start with bottleneck channels
        for i in range(len(self.channels) - 1):
            target_channels = self.channels[-(i + 2)]  # Channels for this level

            # Upsample block reduces channels by half
            self.upsamples.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        current_channels, target_channels, kernel_size=2, stride=2
                    ),
                    nn.BatchNorm2d(target_channels),
                    nn.ReLU(inplace=True),
                )
            )

            # Decoder block takes concatenated features
            # Input channels = upsampled channels + skip connection channels
            decoder_in_channels = (
                target_channels * 2
            )  # target_channels from upsample + target_channels from skip
            self.decoder_blocks.append(ConvBlock(decoder_in_channels, target_channels))

            current_channels = target_channels

        # Attention modules
        self.attention = CBAM(
            channels=self.out_channels, reduction_ratio=4, spatial_kernel_size=7
        )

        # Final layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(self.out_channels, config.num_classes, kernel_size=1),
            nn.BatchNorm2d(config.num_classes),
        )

        # Deep supervision if configured
        if config.deep_supervision and config.use_fpn:
            self.deep_outputs = nn.ModuleList(
                [
                    nn.Conv2d(config.fpn_channels, config.num_classes, kernel_size=1)
                    for _ in range(config.depth)
                ]
            )

    def forward(self, xA, xB):
        # Store input size for later upsampling
        input_size = xA.shape[2:]

        # Initial convolution
        xA = self.init_conv(xA)
        xB = self.init_conv(xB)

        # Encoder
        encoder_features_A = []
        encoder_features_B = []

        for i, (block, pool) in enumerate(zip(self.encoder_blocks, self.encoder_pools)):
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
                outputs = []
                for feat, output_conv in zip(fpn_features, self.deep_outputs):
                    out = output_conv(feat)
                    out = F.interpolate(
                        out, size=input_size, mode="bilinear", align_corners=True
                    )
                    outputs.append(out)
                return outputs

            x = fpn_features[0]
        else:
            # Decoder path
            for i, (up, block) in enumerate(zip(self.upsamples, self.decoder_blocks)):
                # Upsample current features
                x = up(x)

                # Get and combine skip connections
                skip_A = encoder_features_A[-(i + 2)]
                skip_B = encoder_features_B[-(i + 2)]
                skip_connection = skip_A + skip_B

                # Concatenate skip connection
                x = torch.cat([x, skip_connection], dim=1)

                # Apply decoder block
                x = block(x)

        # Apply attention
        x = self.attention(x)

        # Final prediction
        x = self.final_conv(x)

        # Ensure output size matches input size
        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=True)

        return x

    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
