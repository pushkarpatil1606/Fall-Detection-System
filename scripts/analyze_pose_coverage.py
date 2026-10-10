from pathlib import Path

ROOT = Path(r"D:\Fall-Detection-Data\processed\detection\yolo")

for split in ["train", "val"]:
    label_dir = ROOT / split / "labels"
    image_dir = ROOT / split / "images"

    labels = list(label_dir.glob("*.txt"))
    person_boxes = 0
    horizontal_boxes = 0
    strong_horizontal_boxes = 0
    horizontal_images = set()
    strong_horizontal_images = set()

    for label_file in labels:
        has_horizontal = False
        has_strong_horizontal = False

        for line in label_file.read_text().splitlines():
            parts = line.split()

            if len(parts) != 5:
                continue

            class_id, _, _, width, height = parts

            if int(float(class_id)) != 0:
                continue

            width = float(width)
            height = float(height)

            person_boxes += 1

            if width <= 0:
                continue

            aspect_ratio = height / width

            if aspect_ratio < 1.0:
                horizontal_boxes += 1
                has_horizontal = True

            if aspect_ratio < 0.75:
                strong_horizontal_boxes += 1
                has_strong_horizontal = True

        if has_horizontal:
            horizontal_images.add(label_file.stem)

        if has_strong_horizontal:
            strong_horizontal_images.add(label_file.stem)

    person_images = len(list(image_dir.glob("*.jpg")))

    print(f"\n===== {split.upper()} =====")
    print(f"Person images            : {person_images:,}")
    print(f"Person boxes             : {person_boxes:,}")
    print(f"A < 1 boxes              : {horizontal_boxes:,}")
    print(f"A < 0.75 boxes           : {strong_horizontal_boxes:,}")
    print(f"Images with A < 1        : {len(horizontal_images):,}")
    print(f"Images with A < 0.75     : {len(strong_horizontal_images):,}")

    if person_boxes:
        print(
            f"Horizontal box coverage  : "
            f"{100 * horizontal_boxes / person_boxes:.2f}%"
        )

    if person_images:
        print(
            f"Horizontal image coverage: "
            f"{100 * len(horizontal_images) / person_images:.2f}%"
        )