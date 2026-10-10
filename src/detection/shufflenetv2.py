"""
ShuffleNetV2 backbone blocks for Milestone 1.

Implements the ShuffleNetV2-style S_Unit(1) and S_Unit(2) described in:
Shi, H., Wang, X., & Shi, J. (2025), Sensors 25, 4128.

Paper-specified points used here:
- Backbone channel widths: 24, 116, 232, 464.
- S_Unit(1): split -> identity/feature branch -> concatenate -> channel shuffle.
- S_Unit(2): stride-2 downsampling -> concatenate -> channel shuffle.
- ReLU activation.
"""

from __future__ import annotations

import torch
from torch import nn


def channel_shuffle(x: torch.Tensor, groups: int = 2) -> torch.Tensor:
    """Shuffle channels between groups."""
    if x.ndim != 4:
        raise ValueError(f"channel_shuffle expects BCHW input, got shape {tuple(x.shape)}")
    batch, channels, height, width = x.shape
    if channels % groups != 0:
        raise ValueError(
            f"channels ({channels}) must be divisible by groups ({groups})"
        )

    channels_per_group = channels // groups
    x = x.reshape(batch, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    return x.reshape(batch, channels, height, width)


class ConvBNReLU(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU."""

    def __init__(
        self,
        c1: int,
        c2: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                c1,
                c2,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise 3x3 convolution followed by pointwise 1x1 convolution."""

    def __init__(self, c1: int, c2: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = ConvBNReLU(
            c1, c1, kernel_size=3, stride=stride, groups=c1
        )
        self.pointwise = ConvBNReLU(
            c1, c2, kernel_size=1, stride=1, groups=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class SUnit1(nn.Module):
    """
    ShuffleNetV2 basic unit, stride 1.

    Input/output channel count is unchanged and must be even.
    One half is kept as the identity branch; the other half is processed
    with 1x1 -> 3x3 depthwise -> 1x1, then the branches are concatenated
    and channel-shuffled.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels < 2 or channels % 2 != 0:
            raise ValueError(
                f"SUnit1 requires an even channel count >= 2, got {channels}"
            )

        branch_channels = channels // 2
        self.branch_channels = branch_channels

        self.branch2 = nn.Sequential(
            ConvBNReLU(branch_channels, branch_channels, 1),
            ConvBNReLU(
                branch_channels,
                branch_channels,
                3,
                stride=1,
                groups=branch_channels,
            ),
            ConvBNReLU(branch_channels, branch_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.chunk(x, 2, dim=1)
        x2 = self.branch2(x2)
        out = torch.cat((x1, x2), dim=1)
        return channel_shuffle(out, groups=2)


class SUnit2(nn.Module):
    """
    ShuffleNetV2 spatial downsampling unit, stride 2.

    The input spatial dimensions are reduced by 2 while the output channel
    count is set by the target stage width. The two downsampling branches are
    concatenated and channel-shuffled.
    """

    def __init__(self, c1: int, c2: int) -> None:
        super().__init__()
        if c2 < 2 or c2 % 2 != 0:
            raise ValueError(f"SUnit2 output channels must be even and >= 2, got {c2}")

        branch_channels = c2 // 2

        # Branch 1: depthwise downsample + pointwise projection.
        self.branch1 = nn.Sequential(
            nn.Conv2d(
                c1,
                c1,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=c1,
                bias=False,
            ),
            nn.BatchNorm2d(c1),
            nn.Conv2d(
                c1,
                branch_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )

        # Branch 2: pointwise -> depthwise downsample -> pointwise.
        self.branch2 = nn.Sequential(
            ConvBNReLU(c1, branch_channels, 1),
            ConvBNReLU(
                branch_channels,
                branch_channels,
                3,
                stride=2,
                groups=branch_channels,
            ),
            ConvBNReLU(branch_channels, branch_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        return channel_shuffle(out, groups=2)


class ShuffleV2Stage(nn.Module):
    """
    A stage whose first unit downsamples and remaining units preserve size.

    Args:
        c1: input channels
        c2: output channels
        repeats: total number of units in this stage; must be >= 1
    """

    def __init__(self, c1: int, c2: int, repeats: int = 1) -> None:
        super().__init__()
        if repeats < 1:
            raise ValueError(f"repeats must be >= 1, got {repeats}")

        units = [SUnit2(c1, c2)]
        units.extend(SUnit1(c2) for _ in range(repeats - 1))
        self.units = nn.Sequential(*units)
        self.out_channels = c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.units(x)


class ShuffleNetV2Backbone(nn.Module):
    """
    Standalone backbone returning P3/P4/P5 feature maps.

    The spatial sizes for a 640x640 input are:
        P3: 80x80 with 116 channels
        P4: 40x40 with 232 channels
        P5: 20x20 with 464 channels

    The 24-channel stem is retained before the first downsampling stage.
    """

    def __init__(
        self,
        stage_repeats: tuple[int, int, int] = (4, 8, 4),
    ) -> None:
        super().__init__()

        self.stem = ConvBNReLU(3, 24, kernel_size=3, stride=2)

        # ShuffleNetV2-style stem pooling produces 160x160 from 640x640.
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage2 = ShuffleV2Stage(24, 116, stage_repeats[0])
        self.stage3 = ShuffleV2Stage(116, 232, stage_repeats[1])
        self.stage4 = ShuffleV2Stage(232, 464, stage_repeats[2])

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)       # 640 -> 320, 24 ch
        x = self.pool(x)       # 320 -> 160, 24 ch

        p3 = self.stage2(x)    # 160 -> 80, 116 ch
        p4 = self.stage3(p3)   # 80 -> 40, 232 ch
        p5 = self.stage4(p4)   # 40 -> 20, 464 ch

        return p3, p4, p5


__all__ = [
    "channel_shuffle",
    "ConvBNReLU",
    "DepthwiseSeparableConv",
    "SUnit1",
    "SUnit2",
    "ShuffleV2Stage",
    "ShuffleNetV2Backbone",
]
