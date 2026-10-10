"""
Verify the YOLO-ready dataset created by create_detection_dataset.py.

Checks:
- every image has a corresponding label
- every label has class 0 (person)
- YOLO boxes have 5 values
- normalized xywh values are in [0, 1]
- train/val counts match
"""

from __future__ import annotations

import argparse
from pathlib import Path


def check_split(split: Path) -> tuple[int, int]:
    images = sorted((split / "images").glob("*.jpg"))
    labels = sorted((split / "labels").glob("*.txt"))

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    missing_labels = image_stems - label_stems
    orphan_labels = label_stems - image_stems

    errors: list[str] = []

    if missing_labels:
        errors.append(f"Missing labels for {len(missing_labels)} images")
    if orphan_labels:
        errors.append(f"Labels without images: {len(orphan_labels)}")

    box_count = 0
    for label in labels:
        for lineno, line in enumerate(
            label.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue

            parts = line.split()
            if len(parts) != 5:
                errors.append(
                    f"{label.name}:{lineno} has {len(parts)} fields; expected 5"
                )
                continue

            try:
                cls = int(parts[0])
                vals = [float(x) for x in parts[1:]]
            except ValueError:
                errors.append(f"{label.name}:{lineno} contains non-numeric values")
                continue

            if cls != 0:
                errors.append(f"{label.name}:{lineno} has class {cls}; expected 0")
            if any(v < 0 or v > 1 for v in vals):
                errors.append(
                    f"{label.name}:{lineno} has non-normalized xywh values {vals}"
                )

            box_count += 1

    if errors:
        print(f"[FAIL] {split}")
        for e in errors[:30]:
            print(" ", e)
        if len(errors) > 30:
            print(f"  ... and {len(errors) - 30} more")
        return 0, box_count

    print(
        f"[PASS] {split.name}: "
        f"{len(images)} images, {len(labels)} labels, {box_count} person boxes"
    )
    return len(images), box_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(r"D:\Fall-Detection-Data\processed\detection\yolo"),
    )
    args = parser.parse_args()

    train_images, train_boxes = check_split(args.dataset / "train")
    val_images, val_boxes = check_split(args.dataset / "val")

    if train_images == 0 or val_images == 0:
        raise SystemExit(1)

    print()
    print("Dataset verification completed.")


if __name__ == "__main__":
    main()
