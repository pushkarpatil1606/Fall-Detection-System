
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from pathlib import Path
import csv
import shutil
from ultralytics import YOLO

# --------------------------------------------------
# PATHS
# --------------------------------------------------
MODEL_PATH = Path(r"C:\Users\DELL\Desktop\Fall-Detection\Fall-Detection-System\models\detection\best.pt")

FALL_DIR = Path(
    r"C:\Users\DELL\Downloads\Fall Detection\fall-detection-main\fall_dataset\fall"
)

OUTPUT_DIR = Path(
    r"D:\Fall-Detection-Data\processed\detection\hardcase_030"
)

MISSED_DIR = OUTPUT_DIR / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MISSED_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "missed_at_030.csv"

CONF_THRESHOLD = 0.30
PERSON_CLASS = 0

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
print("Loading model...")
model = YOLO(str(MODEL_PATH))

# --------------------------------------------------
# GET IMAGES
# --------------------------------------------------
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

images = sorted(
    p for p in FALL_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in extensions
)

print(f"Total fall images: {len(images)}")
print(f"Confidence threshold: {CONF_THRESHOLD}")
print()

missed = []
detected = []

# --------------------------------------------------
# TEST EACH IMAGE
# --------------------------------------------------
for i, image_path in enumerate(images, start=1):

    results = model.predict(
        source=str(image_path),
        conf=CONF_THRESHOLD,
        classes=[PERSON_CLASS],
        verbose=False
    )

    result = results[0]

    person_detected = (
        result.boxes is not None
        and len(result.boxes) > 0
    )

    if person_detected:
        detected.append(image_path.name)
    else:
        missed.append(image_path.name)

    print(
        f"[{i:3}/{len(images)}] "
        f"{image_path.name:<40} "
        f"{'DETECTED' if person_detected else 'MISSED'}"
    )

# --------------------------------------------------
# COPY MISSED IMAGES
# --------------------------------------------------
for filename in missed:
    src = FALL_DIR / filename
    dst = MISSED_DIR / filename
    shutil.copy2(src, dst)

# --------------------------------------------------
# SAVE CSV
# --------------------------------------------------
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "status", "confidence_threshold"])

    for filename in detected:
        writer.writerow([filename, "detected", CONF_THRESHOLD])

    for filename in missed:
        writer.writerow([filename, "missed", CONF_THRESHOLD])

# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------
print("\n" + "=" * 60)
print("RESULT")
print("=" * 60)

print(f"Total images       : {len(images)}")
print(f"Detected           : {len(detected)}")
print(f"Missed             : {len(missed)}")
print(
    f"Detection rate     : "
    f"{len(detected) / len(images) * 100:.2f}%"
)

print("\nMissed images:")
for filename in missed:
    print(" -", filename)

print("\nHard-case images saved to:")
print(MISSED_DIR)

print("\nCSV saved to:")
print(CSV_PATH)