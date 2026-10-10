
from __future__ import annotations

import torch
from torch import nn


def channel_shuffle(x: torch.Tensor, groups: int = 2) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {tuple(x.shape)}")
    b, c, h, w = x.shape
    if groups < 1 or c % groups != 0:
        raise ValueError(f"Invalid groups={groups} for channels={c}")
    x = x.reshape(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.reshape(b, c, h, w)


class ShuffleAttention(nn.Module):
    """G=2 Shuffle Attention with a parser-compatible constructor."""

    def __init__(
        self,
        c1: int,
        c2: int | None = None,
        groups: int = 2,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()

        if c2 is None:
            c2 = c1
        if c1 != c2:
            raise ValueError(
                f"ShuffleAttention must preserve channels: {c1} != {c2}"
            )
        if groups < 1 or c1 % groups != 0:
            raise ValueError(f"Invalid groups={groups} for channels={c1}")

        group_channels = c1 // groups
        if group_channels < 2 or group_channels % 2:
            raise ValueError(
                "Each group must have an even channel count; "
                f"got {group_channels}"
            )

        branch_channels = group_channels // 2

        self.channels = c1
        self.groups = groups
        self.branch_channels = branch_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Keep the independently tested Step-2 initialization.
        self.channel_weight = nn.Parameter(
            torch.zeros(1, branch_channels, 1, 1)
        )
        self.channel_bias = nn.Parameter(
            torch.ones(1, branch_channels, 1, 1)
        )

        self.group_norm = nn.GroupNorm(
            num_groups=branch_channels,
            num_channels=branch_channels,
            eps=eps,
            affine=False,
        )

        self.spatial_weight = nn.Parameter(
            torch.zeros(1, branch_channels, 1, 1)
        )
        self.spatial_bias = nn.Parameter(
            torch.ones(1, branch_channels, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected BCHW tensor, got shape {tuple(x.shape)}")
        if x.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, got {x.shape[1]}"
            )

        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, c // self.groups, h, w)
        x_channel, x_spatial = torch.chunk(x, 2, dim=1)

        pooled = self.avg_pool(x_channel)
        channel_gate = torch.sigmoid(
            self.channel_weight * pooled + self.channel_bias
        )
        x_channel = x_channel * channel_gate

        normalized = self.group_norm(x_spatial)
        spatial_gate = torch.sigmoid(
            self.spatial_weight * normalized + self.spatial_bias
        )
        x_spatial = x_spatial * spatial_gate

        out = torch.cat((x_channel, x_spatial), dim=1)
        out = out.reshape(b, c, h, w)

        return channel_shuffle(out, groups=2)


__all__ = ["ShuffleAttention", "channel_shuffle"]
