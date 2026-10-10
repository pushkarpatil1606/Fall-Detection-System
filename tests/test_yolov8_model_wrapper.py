from __future__ import annotations

import torch

from src.detection.yolov8_model import create_model


def test_model_wrapper_builds_and_runs() -> None:
    model = create_model(verbose=False)
    x = torch.zeros(1, 3, 640, 640)

    with torch.no_grad():
        _ = model.model(x)

    assert model.model.model[-1].__class__.__name__ == "Detect"


if __name__ == "__main__":
    test_model_wrapper_builds_and_runs()
    print("YOLOv8 model wrapper test passed.")
