from pathlib import Path
import shutil
import csv

from src.detection.yolov8_model import load_model


# --------------------------------------------------
# Paths
# --------------------------------------------------

MODEL_PATH = Path("models/detection/best.pt")

DATASET_ROOT = Path(
    r"C:\Users\DELL\Downloads\Fall Detection\fall-detection-main\fall_dataset"
)

OUTPUT_ROOT = Path("results/milestone-1/fall-dataset-analysis")

MISSED_FALL = OUTPUT_ROOT / "missed" / "fall"
MISSED_NOT_FALL = OUTPUT_ROOT / "missed" / "not-fall"

MISSED_FALL.mkdir(parents=True, exist_ok=True)
MISSED_NOT_FALL.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Settings
# --------------------------------------------------

CONFIDENCE = 0.10
IMAGE_SIZE = 640
DEVICE = "cpu"


# --------------------------------------------------
# Load model once
# --------------------------------------------------

print("Loading model...")
model = load_model(str(MODEL_PATH))
print("Model loaded successfully.\n")


# --------------------------------------------------
# Process dataset
# --------------------------------------------------

rows = []

total_images = 0
total_detected = 0
total_missed = 0

category_stats = {}


for category in ["fall", "not-fall"]:

    folder = DATASET_ROOT / category

    images = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    images.sort()

    detected_count = 0
    missed_count = 0

    print(f"\n===== {category.upper()} =====")
    print(f"Images: {len(images)}")

    for index, image_path in enumerate(images, start=1):

        results = model.predict(
            source=str(image_path),
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            device=DEVICE,
            verbose=False,
        )

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for i in range(len(result.boxes)):

                class_id = int(result.boxes.cls[i].item())

                if class_id != 0:
                    continue

                confidence = float(result.boxes.conf[i].item())

                x1, y1, x2, y2 = result.boxes.xyxy[i].tolist()

                detections.append({
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                })

        total_images += 1

        if detections:
            detected_count += 1
            total_detected += 1

            max_confidence = max(
                d["confidence"] for d in detections
            )

            detection_status = "detected"

        else:
            missed_count += 1
            total_missed += 1

            max_confidence = 0.0

            detection_status = "missed"

            # Copy missed images for manual inspection
            destination = (
                MISSED_FALL / image_path.name
                if category == "fall"
                else MISSED_NOT_FALL / image_path.name
            )

            shutil.copy2(image_path, destination)

        rows.append({
            "category": category,
            "filename": image_path.name,
            "status": detection_status,
            "num_detections": len(detections),
            "max_confidence": max_confidence,
        })

        if index % 25 == 0 or index == len(images):
            print(
                f"Processed {index}/{len(images)} | "
                f"Detected: {detected_count} | "
                f"Missed: {missed_count}"
            )

    category_stats[category] = {
        "total": len(images),
        "detected": detected_count,
        "missed": missed_count,
    }


# --------------------------------------------------
# Save CSV report
# --------------------------------------------------

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

csv_path = OUTPUT_ROOT / "detection_results.csv"

with csv_path.open("w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "category",
            "filename",
            "status",
            "num_detections",
            "max_confidence",
        ],
    )

    writer.writeheader()
    writer.writerows(rows)


# --------------------------------------------------
# Final report
# --------------------------------------------------

print("\n" + "=" * 50)
print("FINAL RESULT")
print("=" * 50)

for category in ["fall", "not-fall"]:

    stats = category_stats[category]

    detection_rate = (
        stats["detected"] / stats["total"] * 100
        if stats["total"] > 0
        else 0
    )

    print(f"\n{category.upper()}")
    print(f"Total images : {stats['total']}")
    print(f"Detected     : {stats['detected']}")
    print(f"Missed       : {stats['missed']}")
    print(f"Detection rate: {detection_rate:.2f}%")

overall_rate = (
    total_detected / total_images * 100
    if total_images > 0
    else 0
)

print("\nOVERALL")
print(f"Total images : {total_images}")
print(f"Detected     : {total_detected}")
print(f"Missed       : {total_missed}")
print(f"Detection rate: {overall_rate:.2f}%")

print("\nReport saved to:")
print(csv_path)

print("\nMissed images saved to:")
print(MISSED_FALL)
print(MISSED_NOT_FALL)
