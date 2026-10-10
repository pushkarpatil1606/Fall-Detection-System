from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Create a targeted Milestone 1 fine-tuning dataset from the "
            "15 annotated hard cases plus a reproducible sample of normal "
            "person-detection training images."
        )
    )
    p.add_argument(
        "--train-root",
        type=Path,
        default=Path(r"D:\Fall-Detection-Data\processed\detection\yolo\train"),
    )
    p.add_argument(
        "--hardcase-root",
        type=Path,
        default=Path(r"D:\Fall-Detection-Data\milestone1_hard_cases"),
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"D:\Fall-Detection-Data\processed\detection\hardcase_ft"),
    )
    p.add_argument("--normal-count", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def image_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def require_label(label: Path) -> None:
    if not label.exists():
        raise FileNotFoundError(f"Missing YOLO label: {label}")


def copy_pair(image: Path, label: Path, out_images: Path, out_labels: Path, prefix: str) -> None:
    destination_stem = f"{prefix}{image.stem}"
    # Preserve the original image extension while using a unique prefixed stem.
    shutil.copy2(image, out_images / f"{destination_stem}{image.suffix.lower()}")
    shutil.copy2(label, out_labels / f"{destination_stem}.txt")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    train_images = args.train_root / "images"
    train_labels = args.train_root / "labels"
    hard_images = args.hardcase_root / "images"
    hard_labels = args.hardcase_root / "labels"

    for path in (train_images, train_labels, hard_images, hard_labels):
        if not path.exists():
            raise FileNotFoundError(f"Required directory not found: {path}")

    normal_candidates = []
    hardcase_names = {p.name.lower() for p in image_files(hard_images)}

    for image in image_files(train_images):
        # The training directory already contains the 15 hard cases after the merge.
        # Exclude them so the normal replay sample contains only original training data.
        if image.name.lower() in hardcase_names:
            continue
        label = train_labels / f"{image.stem}.txt"
        if label.exists():
            normal_candidates.append((image, label))

    if len(normal_candidates) < args.normal_count:
        raise RuntimeError(
            f"Only {len(normal_candidates)} normal labeled images are available; "
            f"cannot sample {args.normal_count}."
        )

    rng.shuffle(normal_candidates)
    selected_normal = normal_candidates[: args.normal_count]

    hard_pairs = []
    for image in image_files(hard_images):
        label = hard_labels / f"{image.stem}.txt"
        require_label(label)
        hard_pairs.append((image, label))

    if len(hard_pairs) != 15:
        raise RuntimeError(f"Expected 15 hard cases, found {len(hard_pairs)}.")

    out_train_images = args.output_root / "train" / "images"
    out_train_labels = args.output_root / "train" / "labels"
    out_train_images.mkdir(parents=True, exist_ok=True)
    out_train_labels.mkdir(parents=True, exist_ok=True)

    # Avoid stale data from a previous run.
    for directory in (out_train_images, out_train_labels):
        for old in directory.iterdir():
            if old.is_file():
                old.unlink()

    for image, label in hard_pairs:
        copy_pair(image, label, out_train_images, out_train_labels, "hard_")

    for image, label in selected_normal:
        copy_pair(image, label, out_train_images, out_train_labels, "normal_")

    yaml_path = args.output_root / "data.yaml"
    yaml_path.write_text(
        "path: D:/Fall-Detection-Data/processed/detection/hardcase_ft\n"
        "train: train/images\n"
        "val: D:/Fall-Detection-Data/processed/detection/yolo/val/images\n"
        "names:\n"
        "  0: person\n",
        encoding="utf-8",
    )

    manifest = args.output_root / "manifest.txt"
    lines = [
        "Milestone 1 targeted fine-tuning dataset",
        f"Hard cases: {len(hard_pairs)}",
        f"Normal replay images: {len(selected_normal)}",
        f"Total training images: {len(hard_pairs) + len(selected_normal)}",
        f"Random seed: {args.seed}",
        "Validation: original COCO val2017-derived validation split (2693 images)",
        "",
        "Hard cases:",
    ]
    lines.extend(f"hard_{image.stem}{image.suffix.lower()}" for image, _ in hard_pairs)
    lines.append("")
    lines.append("Normal replay images:")
    lines.extend(f"normal_{image.stem}{image.suffix.lower()}" for image, _ in selected_normal)
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== Hard-case fine-tuning dataset created ===")
    print(f"Output root       : {args.output_root}")
    print(f"Hard cases        : {len(hard_pairs)}")
    print(f"Normal replay     : {len(selected_normal)}")
    print(f"Total train       : {len(hard_pairs) + len(selected_normal)}")
    print(f"Validation images : 2693 (unchanged original val split)")
    print(f"Seed              : {args.seed}")
    print(f"YAML              : {yaml_path}")


if __name__ == "__main__":
    main()
