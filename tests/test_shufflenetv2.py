from __future__ import annotations

import torch

from src.detection.shufflenetv2 import (
    SUnit1,
    SUnit2,
    ShuffleNetV2Backbone,
    channel_shuffle,
)


def test_channel_shuffle_keeps_shape() -> None:
    x = torch.randn(2, 8, 10, 10)
    y = channel_shuffle(x, groups=2)
    assert y.shape == x.shape


def test_sunit1_shape() -> None:
    x = torch.randn(2, 116, 80, 80)
    y = SUnit1(116)(x)
    assert y.shape == (2, 116, 80, 80)


def test_sunit2_shape_and_downsampling() -> None:
    x = torch.randn(2, 24, 160, 160)
    y = SUnit2(24, 116)(x)
    assert y.shape == (2, 116, 80, 80)


def test_backbone_feature_shapes() -> None:
    model = ShuffleNetV2Backbone(stage_repeats=(4, 8, 4))
    x = torch.randn(1, 3, 640, 640)

    with torch.no_grad():
        p3, p4, p5 = model(x)

    assert p3.shape == (1, 116, 80, 80)
    assert p4.shape == (1, 232, 40, 40)
    assert p5.shape == (1, 464, 20, 20)


if __name__ == "__main__":
    test_channel_shuffle_keeps_shape()
    test_sunit1_shape()
    test_sunit2_shape_and_downsampling()
    test_backbone_feature_shapes()
    print("ShuffleNetV2 Step 1 tests passed.")
