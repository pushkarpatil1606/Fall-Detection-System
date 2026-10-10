"""
Create a YOLO-ready person-detection dataset without copying the original images.

Source layout assumed:
    D:\Fall-Detection-Data\
        COCO\train2017
        COCO\val2017
        VOC2007\JPEGImages
        VOC2007\ImageSets\Main\train.txt
        processed\detection\labels\coco_train
        processed\detection\labels\coco_val
        processed\detection\labels\voc2007

The images are HARD-LINKED into the training folder, so no second copy of
the large COCO/VOC image files is created. Labels are copied because their
filenames are prefixed to avoid possible COCO/VOC name collisions.

Project choice for this milestone:
    Train: COCO train2017 persons + VOC2007 train split persons
    Val:   COCO val2017 persons

This is the project's explicit split; the supplied paper states that COCO
train2017 and Pascal VOC 2007 are used for object detection but does not
specify this exact merged split.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


DEFAULT_DATA_ROOT = Path(r"D:\Fall-Detection-Data")


def hardlink_or_skip(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        return

    try:
        os.link(source, destination)
    except FileExistsError:
        pass
    except OSError as exc:
        raise OSError(
            f"Could not create hard link:\n  source: {source}\n"
            f"  destination: {destination}\n"
            f"Make sure both paths are on the same drive."
        ) from exc


def copy_label(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)


def read_voc_train_ids(voc_root: Path) -> set[str]:
    split_file = voc_root / "ImageSets" / "Main" / "train.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"VOC train split not found: {split_file}")

    ids = {
        line.strip()
        for line in split_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if not ids:
        raise RuntimeError(f"VOC train split is empty: {split_file}")
    return ids


def process_coco(
    image_dir: Path,
    label_dir: Path,
    target_images: Path,
    target_labels: Path,
    prefix: str,
) -> tuple[int, int]:
    linked = 0
    missing = 0

    for label in sorted(label_dir.glob("*.txt")):
        stem = label.stem
        image = image_dir / f"{stem}.jpg"

        if not image.exists():
            missing += 1
            continue

        out_stem = f"{prefix}_{stem}"
        hardlink_or_skip(image, target_images / f"{out_stem}.jpg")
        copy_label(label, target_labels / f"{out_stem}.txt")
        linked += 1

    return linked, missing


def process_voc_train(
    image_dir: Path,
    label_dir: Path,
    train_ids: set[str],
    target_images: Path,
    target_labels: Path,
) -> tuple[int, int]:
    linked = 0
    missing = 0

    for stem in sorted(train_ids):
        label = label_dir / f"{stem}.txt"
        image = image_dir / f"{stem}.jpg"

        # Only images with person labels belong in the person detector set.
        if not label.exists():
            continue

        if not image.exists():
            missing += 1
            continue

        out_stem = f"voc2007_{stem}"
        hardlink_or_skip(image, target_images / f"{out_stem}.jpg")
        copy_label(label, target_labels / f"{out_stem}.txt")
        linked += 1

    return linked, missing


def write_yaml(yaml_path: Path, dataset_root: Path) -> None:
    # Use forward slashes so the YAML is portable within Windows.
    dataset_root_text = dataset_root.resolve().as_posix()

    yaml = f"""# Milestone 1: Improved YOLOv8 person detector
# Train = COCO train2017 + VOC2007 train split
# Val   = COCO val2017
#
# Generated for this machine. Teammates should regenerate/update the path
# for their own local dataset root.

path: {dataset_root_text}
train: train/images
val: val/images

names:
  0: person
"""
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Original dataset root, e.g. D:\\Fall-Detection-Data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="YOLO-ready output directory. Defaults to <data-root>\\processed\\detection\\yolo",
    )
    parser.add_argument(
        "--yaml",
        type=Path,
        default=Path("configs/detection_person.yaml"),
        help="Path for the Ultralytics dataset YAML.",
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    output = (args.output or data_root / "processed" / "detection" / "yolo").resolve()

    coco = data_root / "COCO"
    voc = data_root / "VOC2007"
    labels = data_root / "processed" / "detection" / "labels"

    coco_train_images = coco / "train2017"
    coco_val_images = coco / "val2017"
    voc_images = voc / "JPEGImages"

    coco_train_labels = labels / "coco_train"
    coco_val_labels = labels / "coco_val"
    voc_labels = labels / "voc2007"

    required = [
        coco_train_images,
        coco_val_images,
        voc_images,
        coco_train_labels,
        coco_val_labels,
        voc_labels,
        voc / "ImageSets" / "Main" / "train.txt",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("Missing required paths:")
        for p in missing:
            print(f"  {p}")
        raise SystemExit(1)

    train_images = output / "train" / "images"
    train_labels = output / "train" / "labels"
    val_images = output / "val" / "images"
    val_labels = output / "val" / "labels"

    train_ids = read_voc_train_ids(voc)

    print("=== Building YOLO-ready dataset ===")
    print(f"Source: {data_root}")
    print(f"Output: {output}")
    print("Images will be hard-linked; they are NOT copied.")

    coco_train_n, coco_train_missing = process_coco(
        coco_train_images,
        coco_train_labels,
        train_images,
        train_labels,
        "coco",
    )
    coco_val_n, coco_val_missing = process_coco(
        coco_val_images,
        coco_val_labels,
        val_images,
        val_labels,
        "coco",
    )
    voc_train_n, voc_train_missing = process_voc_train(
        voc_images,
        voc_labels,
        train_ids,
        train_images,
        train_labels,
    )

    write_yaml(args.yaml, output)

    train_image_count = len(list(train_images.glob("*.jpg")))
    train_label_count = len(list(train_labels.glob("*.txt")))
    val_image_count = len(list(val_images.glob("*.jpg")))
    val_label_count = len(list(val_labels.glob("*.txt")))

    print()
    print("=== Result ===")
    print(f"COCO train person images linked : {coco_train_n}")
    print(f"VOC train person images linked  : {voc_train_n}")
    print(f"COCO val person images linked   : {coco_val_n}")
    print(f"COCO train missing images       : {coco_train_missing}")
    print(f"VOC train missing images        : {voc_train_missing}")
    print(f"COCO val missing images         : {coco_val_missing}")
    print()
    print(f"Final train images: {train_image_count}")
    print(f"Final train labels: {train_label_count}")
    print(f"Final val images  : {val_image_count}")
    print(f"Final val labels  : {val_label_count}")
    print(f"Dataset YAML      : {args.yaml.resolve()}")
    print()
    print("No large image copies were created; train/val images are hard links.")


if __name__ == "__main__":
    main()
