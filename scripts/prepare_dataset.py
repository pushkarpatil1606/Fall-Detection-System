from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Iterable


PERSON_CLASS_ID = 0


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def coco_box_to_yolo(
    bbox: list[float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    """
    Convert COCO bbox:
        [x, y, width, height]

    into YOLO normalized format:
        x_center y_center width height
    """

    if len(bbox) != 4:
        return None

    x, y, width, height = map(float, bbox)

    if width <= 0 or height <= 0:
        return None

    x1 = clamp(x, 0.0, float(image_width))
    y1 = clamp(y, 0.0, float(image_height))
    x2 = clamp(x + width, 0.0, float(image_width))
    y2 = clamp(y + height, 0.0, float(image_height))

    box_width = x2 - x1
    box_height = y2 - y1

    if box_width <= 0 or box_height <= 0:
        return None

    x_center = (x1 + x2) / 2.0 / image_width
    y_center = (y1 + y2) / 2.0 / image_height
    norm_width = box_width / image_width
    norm_height = box_height / image_height

    return (
        clamp(x_center, 0.0, 1.0),
        clamp(y_center, 0.0, 1.0),
        clamp(norm_width, 0.0, 1.0),
        clamp(norm_height, 0.0, 1.0),
    )


def write_yolo_label_file(
    output_path: Path,
    boxes: Iterable[tuple[float, float, float, float]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    for x_center, y_center, width, height in boxes:
        lines.append(
            f"{PERSON_CLASS_ID} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

    output_path.write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8",
    )


def convert_coco(
    coco_root: Path,
    output_root: Path,
    split: str = "train",
) -> None:
    """
    Convert COCO person bounding boxes to YOLO format.

    Expected:
        COCO/
        ├── train2017/
        └── annotations/
            └── instances_train2017.json
    """

    image_dir = coco_root / f"{split}2017"
    annotation_file = (
        coco_root
        / "annotations"
        / f"instances_{split}2017.json"
    )

    output_dir = output_root / (
        "coco_train" if split == "train" else f"coco_{split}"
    )

    if not image_dir.exists():
        print(f"[COCO] Image directory not found: {image_dir}")
        return

    if not annotation_file.exists():
        print(f"[COCO] Annotation file not found: {annotation_file}")
        return

    print(f"\n[COCO] Processing {split}2017...")

    with annotation_file.open("r", encoding="utf-8") as file:
        data = json.load(file)

    person_category_ids = {
        category["id"]
        for category in data["categories"]
        if category["name"].lower() == "person"
    }

    if not person_category_ids:
        raise RuntimeError("COCO 'person' category was not found.")

    annotations_by_image: dict[int, list[dict]] = defaultdict(list)

    for annotation in data["annotations"]:
        if annotation["category_id"] in person_category_ids:
            annotations_by_image[annotation["image_id"]].append(annotation)

    processed_images = 0
    images_with_person = 0
    total_boxes = 0
    invalid_boxes = 0

    manifest_lines: list[str] = []

    for image_info in data["images"]:
        image_id = image_info["id"]
        file_name = image_info["file_name"]

        image_path = image_dir / file_name

        if not image_path.exists():
            print(f"[WARNING] Missing COCO image: {image_path}")
            continue

        processed_images += 1

        person_boxes = []

        for annotation in annotations_by_image.get(image_id, []):
            converted = coco_box_to_yolo(
                annotation["bbox"],
                image_info["width"],
                image_info["height"],
            )

            if converted is None:
                invalid_boxes += 1
                continue

            person_boxes.append(converted)

        # We only generate labels for images containing at least one person.
        if not person_boxes:
            continue

        images_with_person += 1
        total_boxes += len(person_boxes)

        label_path = output_dir / f"{Path(file_name).stem}.txt"

        write_yolo_label_file(
            label_path,
            person_boxes,
        )

        relative_image = (
            Path("COCO") / f"{split}2017" / file_name
        ).as_posix()

        relative_label = (
            Path("processed")
            / "detection"
            / "labels"
            / output_dir.name
            / f"{Path(file_name).stem}.txt"
        ).as_posix()

        manifest_lines.append(
            f"{relative_image}\t{relative_label}"
        )

    manifest_path = (
        output_root.parent
        / "lists"
        / f"coco_{split}_person.tsv"
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_path.write_text(
        "\n".join(manifest_lines) + (
            "\n" if manifest_lines else ""
        ),
        encoding="utf-8",
    )

    print("[COCO] Done")
    print(f"  Images processed       : {processed_images:,}")
    print(f"  Images with persons    : {images_with_person:,}")
    print(f"  Person boxes           : {total_boxes:,}")
    print(f"  Invalid boxes          : {invalid_boxes:,}")
    print(f"  Labels                 : {output_dir}")
    print(f"  Manifest               : {manifest_path}")


def get_voc_split_ids(
    image_sets_dir: Path,
) -> dict[str, list[str]]:
    """
    Read VOC split files.

    Expected:
        ImageSets/Main/train.txt
        ImageSets/Main/val.txt
        ImageSets/Main/trainval.txt
        ImageSets/Main/test.txt
    """

    splits: dict[str, list[str]] = {}

    for split_name in (
        "train",
        "val",
        "trainval",
        "test",
    ):
        split_file = image_sets_dir / f"{split_name}.txt"

        if not split_file.exists():
            continue

        ids = []

        for line in split_file.read_text(
            encoding="utf-8"
        ).splitlines():
            line = line.strip()

            if not line:
                continue

            # VOC split files normally contain only an image ID.
            image_id = line.split()[0]
            ids.append(image_id)

        splits[split_name] = ids

    return splits


def parse_voc_annotation(
    xml_file: Path,
) -> tuple[int, int, list[tuple[str, float, float, float, float]]]:
    """
    Read one Pascal VOC XML file.

    Returns:
        image_width,
        image_height,
        list of:
            (class_name, xmin, ymin, xmax, ymax)
    """

    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find("size")

    if size is None:
        raise ValueError(f"Missing <size> in {xml_file}")

    width = int(size.findtext("width", "0"))
    height = int(size.findtext("height", "0"))

    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid image dimensions in {xml_file}"
        )

    objects = []

    for obj in root.findall("object"):
        class_name = obj.findtext("name", "").strip().lower()

        bbox = obj.find("bndbox")

        if bbox is None:
            continue

        xmin = float(bbox.findtext("xmin", "0"))
        ymin = float(bbox.findtext("ymin", "0"))
        xmax = float(bbox.findtext("xmax", "0"))
        ymax = float(bbox.findtext("ymax", "0"))

        objects.append(
            (
                class_name,
                xmin,
                ymin,
                xmax,
                ymax,
            )
        )

    return width, height, objects


def voc_box_to_yolo(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:

    xmin = clamp(xmin, 0.0, float(image_width))
    ymin = clamp(ymin, 0.0, float(image_height))
    xmax = clamp(xmax, 0.0, float(image_width))
    ymax = clamp(ymax, 0.0, float(image_height))

    if xmax <= xmin or ymax <= ymin:
        return None

    x_center = ((xmin + xmax) / 2.0) / image_width
    y_center = ((ymin + ymax) / 2.0) / image_height
    width = (xmax - xmin) / image_width
    height = (ymax - ymin) / image_height

    return (
        clamp(x_center, 0.0, 1.0),
        clamp(y_center, 0.0, 1.0),
        clamp(width, 0.0, 1.0),
        clamp(height, 0.0, 1.0),
    )


def convert_voc(
    voc_root: Path,
    output_root: Path,
) -> None:

    annotations_dir = voc_root / "Annotations"
    images_dir = voc_root / "JPEGImages"
    image_sets_dir = voc_root / "ImageSets" / "Main"

    output_dir = output_root / "voc2007"

    if not annotations_dir.exists():
        raise FileNotFoundError(
            f"VOC annotations not found: {annotations_dir}"
        )

    if not images_dir.exists():
        raise FileNotFoundError(
            f"VOC images not found: {images_dir}"
        )

    splits = get_voc_split_ids(image_sets_dir)

    if not splits:
        raise RuntimeError(
            "No VOC split files were found."
        )

    # We process each unique image only once.
    all_image_ids = set()

    for ids in splits.values():
        all_image_ids.update(ids)

    processed_images = 0
    images_with_person = 0
    total_boxes = 0
    invalid_boxes = 0

    for image_id in sorted(all_image_ids):

        xml_file = annotations_dir / f"{image_id}.xml"
        image_file = images_dir / f"{image_id}.jpg"

        if not xml_file.exists():
            print(f"[WARNING] Missing XML: {xml_file}")
            continue

        if not image_file.exists():
            print(f"[WARNING] Missing image: {image_file}")
            continue

        try:
            image_width, image_height, objects = (
                parse_voc_annotation(xml_file)
            )
        except Exception as exc:
            print(
                f"[WARNING] Failed to parse {xml_file}: {exc}"
            )
            continue

        processed_images += 1

        person_boxes = []

        for (
            class_name,
            xmin,
            ymin,
            xmax,
            ymax,
        ) in objects:

            if class_name != "person":
                continue

            converted = voc_box_to_yolo(
                xmin,
                ymin,
                xmax,
                ymax,
                image_width,
                image_height,
            )

            if converted is None:
                invalid_boxes += 1
                continue

            person_boxes.append(converted)

        if not person_boxes:
            continue

        images_with_person += 1
        total_boxes += len(person_boxes)

        label_path = output_dir / f"{image_id}.txt"

        write_yolo_label_file(
            label_path,
            person_boxes,
        )

    # Write split manifests.
    lists_dir = output_root.parent / "lists"
    lists_dir.mkdir(parents=True, exist_ok=True)

    for split_name, ids in splits.items():

        manifest_lines = []

        for image_id in ids:

            image_path = (
                Path("VOC2007")
                / "JPEGImages"
                / f"{image_id}.jpg"
            )

            label_path = (
                Path("processed")
                / "detection"
                / "labels"
                / "voc2007"
                / f"{image_id}.txt"
            )

            manifest_lines.append(
                f"{image_path.as_posix()}\t"
                f"{label_path.as_posix()}"
            )

        manifest_file = (
            lists_dir
            / f"voc2007_{split_name}_person.tsv"
        )

        manifest_file.write_text(
            "\n".join(manifest_lines)
            + ("\n" if manifest_lines else ""),
            encoding="utf-8",
        )

    print("\n[VOC2007] Done")
    print(f"  Images processed       : {processed_images:,}")
    print(f"  Images with persons    : {images_with_person:,}")
    print(f"  Person boxes           : {total_boxes:,}")
    print(f"  Invalid boxes          : {invalid_boxes:,}")
    print(f"  Labels                 : {output_dir}")
    print(f"  Splits                 : {', '.join(splits.keys())}")


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Prepare COCO and Pascal VOC person "
            "bounding-box annotations in YOLO format."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"D:\Fall-Detection-Data"),
        help="Root directory containing COCO and VOC2007.",
    )

    args = parser.parse_args()

    data_root = args.data_root.resolve()

    coco_root = data_root / "COCO"
    voc_root = data_root / "VOC2007"

    processed_root = (
        data_root
        / "processed"
        / "detection"
        / "labels"
    )

    print("=" * 65)
    print("Milestone 1 Dataset Preparation")
    print("=" * 65)
    print(f"Data root : {data_root}")
    print(f"COCO     : {coco_root}")
    print(f"VOC2007  : {voc_root}")
    print("=" * 65)

    if not coco_root.exists():
        raise FileNotFoundError(
            f"COCO directory not found: {coco_root}"
        )

    if not voc_root.exists():
        raise FileNotFoundError(
            f"VOC2007 directory not found: {voc_root}"
        )

    # COCO train2017 → YOLO person labels
    convert_coco(
        coco_root=coco_root,
        output_root=processed_root,
        split="train",
    )

    # COCO validation conversion is optional and only performed
    # when val2017 images actually exist.
    val_images = coco_root / "val2017"

    if val_images.exists():
        convert_coco(
            coco_root=coco_root,
            output_root=processed_root,
            split="val",
        )
    else:
        print(
            "\n[COCO] val2017 images were not found."
            "\n       Skipping COCO validation conversion."
        )

    # Pascal VOC 2007
    convert_voc(
        voc_root=voc_root,
        output_root=processed_root,
    )

    print("\n" + "=" * 65)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()