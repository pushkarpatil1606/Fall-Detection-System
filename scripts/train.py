"""
Milestone 1 resumable training script.

Architecture:
    ShuffleNetV2 backbone -> SPPF -> YOLOv8 neck
    -> Shuffle Attention -> YOLOv8 Detect

Paper-reported settings used as defaults:
    imgsz=640
    lr0=0.01
    batch=16
    momentum=0.937

The paper does not state an epoch count or optimizer name in the training
details available to us. This script uses SGD as the explicit project choice.

Typical workflow on the CPU machine:

1) Measure one full-data epoch:
    python -m scripts.train --epochs 1 --batch 16 --device cpu \
        --workers 0 --fraction 1.0 --name full-epoch-1

2) Continue from a saved checkpoint:
    python -m scripts.train --resume \
        results/milestone-1/full-epoch-1/weights/last.pt

3) Final run after choosing the total number of epochs:
    python -m scripts.train --epochs 20 --batch 16 --device cpu \
        --workers 0 --fraction 1.0 --name final-training

4) Promote the selected final checkpoint:
    Add --promote-best to the final non-resume run.

The script does NOT automatically overwrite models/detection/best.pt while
resuming. This avoids accidentally replacing the handoff checkpoint before
we have finished evaluating it.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.detection.yolov8_model import create_model, load_model


ROOT = Path(__file__).resolve().parents[1]
DATA_CFG = ROOT / "configs" / "detection_person.yaml"
RESULTS_DIR = ROOT / "results" / "milestone-1"
FINAL_MODEL_DIR = ROOT / "models" / "detection"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or resume the Milestone 1 Improved YOLOv8 detector."
    )

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--name", type=str, default="full-epoch-1")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to last.pt/best.pt from an existing training run.",
    )
    parser.add_argument(
        "--promote-best",
        action="store_true",
        help="Copy the current run's best.pt to models/detection/best.pt.",
    )

    return parser.parse_args()


def validate_resume_path(path: Path) -> Path:
    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")

    if path.suffix.lower() != ".pt":
        raise ValueError(f"Resume checkpoint must be a .pt file: {path}")

    return path


def promote_checkpoint(run_dir: Path) -> None:
    best = run_dir / "weights" / "best.pt"

    if not best.exists():
        raise FileNotFoundError(
            f"Cannot promote checkpoint because it does not exist: {best}"
        )

    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    destination = FINAL_MODEL_DIR / "best.pt"
    shutil.copy2(best, destination)

    print(f"Final handoff checkpoint copied to: {destination}")


def main() -> None:
    args = parse_args()

    if not DATA_CFG.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {DATA_CFG}\n"
            "Run scripts/create_detection_dataset.py first."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.resume is not None:
        checkpoint = validate_resume_path(args.resume)

        print("=== Resuming Milestone 1 training ===")
        print(f"Checkpoint : {checkpoint}")
        print("Ultralytics will restore the saved training state.")
        print()

        model = load_model(checkpoint, verbose=True)

        # Resume the original run. Ultralytics restores optimizer, scheduler,
        # epoch, and related training state from the checkpoint.
        model.train(resume=str(checkpoint))

        run_dir = checkpoint.parent.parent
    else:
        print("=== Starting Milestone 1 training ===")
        print(f"Dataset    : {DATA_CFG}")
        print(f"Epochs     : {args.epochs}")
        print(f"Batch      : {args.batch}")
        print(f"Image size : {args.imgsz}")
        print(f"LR0        : {args.lr0}")
        print(f"Momentum   : {args.momentum}")
        print(f"Optimizer  : {args.optimizer}")
        print(f"Device     : {args.device}")
        print(f"Workers    : {args.workers}")
        print(f"Fraction   : {args.fraction}")
        print(f"Run name   : {args.name}")
        print()

        model = create_model(verbose=True)

        model.train(
            data=str(DATA_CFG),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            lr0=args.lr0,
            momentum=args.momentum,
            optimizer=args.optimizer,
            device=args.device,
            workers=args.workers,
            fraction=args.fraction,
            cache=False,
            amp=False if args.device == "cpu" else True,
            project=str(RESULTS_DIR),
            name=args.name,
            exist_ok=True,
            pretrained=False,
            save=True,
            save_period=1,
        )

        run_dir = RESULTS_DIR / args.name

    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"

    print()
    print("=== Run finished ===")
    print(f"Run directory : {run_dir}")
    print(f"best.pt       : {best}")
    print(f"last.pt       : {last}")

    if args.promote_best:
        promote_checkpoint(run_dir)


if __name__ == "__main__":
    main()
