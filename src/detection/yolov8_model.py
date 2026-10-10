"""
Project-level wrapper for the Milestone 1 Improved YOLOv8 detector.

Architecture:
    ShuffleNetV2 backbone
    -> SPPF
    -> YOLOv8 neck
    -> Shuffle Attention
    -> YOLOv8 Detect
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG = PROJECT_ROOT / "configs" / "yolov8_shufflenet.yaml"


def _ensure_project_on_path() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def register_custom_modules() -> None:
    """Register project custom classes for Ultralytics YAML parsing."""
    _ensure_project_on_path()

    from src.detection.shufflenetv2 import ShuffleV2Stage
    from src.detection.shuffle_attention import ShuffleAttention
    import ultralytics.nn.tasks as tasks

    # parse_model() resolves custom module names with globals()[m].
    tasks.ShuffleV2Stage = ShuffleV2Stage
    tasks.ShuffleAttention = ShuffleAttention

    source = Path(tasks.__file__).read_text(encoding="utf-8")
    required = (
        "elif m is ShuffleV2Stage:",
        "elif m is ShuffleAttention:",
    )
    missing = [item for item in required if item not in source]

    if missing:
        raise RuntimeError(
            "Ultralytics parser integration is incomplete. Run:\n"
            "    python scripts/patch_ultralytics.py\n"
            f"Missing parser branches: {missing}"
        )


def create_model(
    cfg: str | Path | None = None,
    *,
    nc: int | None = None,
    verbose: bool = True,
):
    """Construct the Milestone 1 Improved YOLOv8 model."""
    _ensure_project_on_path()
    register_custom_modules()

    from ultralytics import YOLO

    config = Path(cfg).resolve() if cfg is not None else MODEL_CONFIG

    if not config.exists():
        raise FileNotFoundError(f"Model configuration not found: {config}")

    if nc is None:
        return YOLO(str(config), task="detect", verbose=verbose)

    import yaml

    model_data: dict[str, Any] = yaml.safe_load(
        config.read_text(encoding="utf-8")
    )
    model_data["nc"] = nc

    return YOLO(model_data, task="detect", verbose=verbose)


def load_model(weights: str | Path, *, verbose: bool = False):
    """Load a trained Milestone 1 checkpoint."""
    _ensure_project_on_path()
    register_custom_modules()

    from ultralytics import YOLO

    checkpoint = Path(weights).resolve()

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    return YOLO(str(checkpoint), task="detect", verbose=verbose)


__all__ = [
    "PROJECT_ROOT",
    "MODEL_CONFIG",
    "register_custom_modules",
    "create_model",
    "load_model",
]
