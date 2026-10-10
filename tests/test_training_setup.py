
from __future__ import annotations

from pathlib import Path


def test_training_files_exist() -> None:
    root = Path(__file__).resolve().parents[1]

    assert (root / "scripts" / "train.py").exists()
    assert (root / "configs" / "detection_person.yaml").exists()
    assert (root / "configs" / "yolov8_shufflenet.yaml").exists()


if __name__ == "__main__":
    test_training_files_exist()
    print("Training setup file test passed.")
