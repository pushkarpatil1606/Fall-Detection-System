
import torch
import torch.nn as nn


# ============================================================
# Channel Attention
# ============================================================

class ChannelAttention(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()

        hidden_channels = channels // reduction

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(
                channels,
                hidden_channels
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                hidden_channels,
                channels
            )
        )

        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x):

        batch_size, channels, _, _ = x.shape

        # Global Average Pooling
        y = self.avg_pool(x)

        # [B, C, 1, 1] → [B, C]
        y = y.view(
            batch_size,
            channels
        )

        # MLP
        y = self.mlp(y)

        # Batch Normalization
        y = self.bn(y)

        # [B, C] → [B, C, 1, 1]
        y = y.view(
            batch_size,
            channels,
            1,
            1
        )

        return y


# ============================================================
# Spatial Attention
# ============================================================

class SpatialAttention(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()

        hidden_channels = channels // reduction

        # ----------------------------------------------------
        # 1×1 Convolution
        # C → C/r
        # ----------------------------------------------------

        self.conv1 = nn.Conv2d(
            channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(hidden_channels)

        self.relu = nn.ReLU(inplace=True)

        # ----------------------------------------------------
        # First dilated 3×3 convolution
        # dilation = 2
        # ----------------------------------------------------

        self.dilated_conv1 = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            stride=1,
            padding=2,
            dilation=2,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(hidden_channels)

        # ----------------------------------------------------
        # Second dilated 3×3 convolution
        # dilation = 1
        # ----------------------------------------------------

        self.dilated_conv2 = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=False
        )

        self.bn3 = nn.BatchNorm2d(hidden_channels)

        # ----------------------------------------------------
        # Final 1×1 convolution
        # C/r → 1
        # ----------------------------------------------------

        self.conv2 = nn.Conv2d(
            hidden_channels,
            1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        self.bn4 = nn.BatchNorm2d(1)

    def forward(self, x):

        # x:
        # [B, C, H, W]

        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.dilated_conv1(y)
        y = self.bn2(y)
        y = self.relu(y)

        y = self.dilated_conv2(y)
        y = self.bn3(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.bn4(y)

        return y

# ============================================================
# Bottleneck Attention Module (BAM)
# ============================================================

class BAM(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()

        self.channel_attention = ChannelAttention(
            channels=channels,
            reduction=reduction
        )

        self.spatial_attention = SpatialAttention(
            channels=channels,
            reduction=reduction
        )

    def forward(self, x):

        # ----------------------------------------------------
        # Channel attention
        # [B, C, H, W]
        #       ↓
        # [B, C, 1, 1]
        # ----------------------------------------------------

        channel_map = self.channel_attention(x)

        # ----------------------------------------------------
        # Spatial attention
        # [B, C, H, W]
        #       ↓
        # [B, 1, H, W]
        # ----------------------------------------------------

        spatial_map = self.spatial_attention(x)

        # ----------------------------------------------------
        # Combine channel + spatial attention
        # Broadcasting automatically expands:
        #
        # [B,C,1,1] + [B,1,H,W]
        #       ↓
        # [B,C,H,W]
        # ----------------------------------------------------

        attention = channel_map + spatial_map

        # ----------------------------------------------------
        # Sigmoid
        # Convert attention values into 0 → 1
        # ----------------------------------------------------

        attention = torch.sigmoid(attention)

        # ----------------------------------------------------
        # Apply attention to original feature map
        # ----------------------------------------------------

        out = x * attention

        # ----------------------------------------------------
        # Residual connection
        # ----------------------------------------------------

        out = out + x

        return out
