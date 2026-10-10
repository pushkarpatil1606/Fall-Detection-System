from __future__ import annotations

import torch

from src.detection.shuffle_attention import ShuffleAttention


def test_sa_shape_464() -> None:
    """Paper backbone output size can be processed by SA."""
    model = ShuffleAttention(channels=464, groups=2)
    x = torch.randn(2, 464, 20, 20)

    with torch.no_grad():
        y = model(x)

    assert y.shape == x.shape


def test_sa_shape_256() -> None:
    """Typical YOLO neck feature width can be processed by SA."""
    model = ShuffleAttention(channels=256, groups=2)
    x = torch.randn(1, 256, 40, 40)

    with torch.no_grad():
        y = model(x)

    assert y.shape == x.shape


def test_sa_shape_512() -> None:
    model = ShuffleAttention(channels=512, groups=2)
    x = torch.randn(1, 512, 20, 20)

    with torch.no_grad():
        y = model(x)

    assert y.shape == x.shape


def test_sa_shape_1024() -> None:
    model = ShuffleAttention(channels=1024, groups=2)
    x = torch.randn(1, 1024, 20, 20)

    with torch.no_grad():
        y = model(x)

    assert y.shape == x.shape


def test_sa_is_differentiable() -> None:
    model = ShuffleAttention(channels=116, groups=2)
    x = torch.randn(2, 116, 80, 80, requires_grad=True)

    y = model(x)
    loss = y.mean()
    loss.backward()

    assert x.grad is not None
    assert model.channel_weight.grad is not None
    assert model.spatial_weight.grad is not None


if __name__ == "__main__":
    test_sa_shape_464()
    test_sa_shape_256()
    test_sa_shape_512()
    test_sa_shape_1024()
    test_sa_is_differentiable()
    print("Shuffle Attention Step 2 tests passed.")
