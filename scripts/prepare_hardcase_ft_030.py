from pathlib import Path
import random
import shutil


# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# 1. 52 hard-case fall images
#
# Actual structure:
# C:\Users\DELL\Downloads\Fall Dataset\
# ├── images\
# │   └── train\
# └── labels\
#     └── train\
# ------------------------------------------------------------
FALL_DATASET = Path(
    r"C:\Users\DELL\Downloads\Fall Dataset"
)

FALL_IMAGES = FALL_DATASET / "images" / "train"
FALL_LABELS = FALL_DATASET / "labels" / "train"


# ------------------------------------------------------------
# 2. Existing original YOLO training dataset
#
# Structure:
# D:\Fall-Detection-Data\processed\detection\yolo\train\
# ├── images\
# └── labels\
# ------------------------------------------------------------
ORIGINAL_TRAIN = Path(
    r"D:\Fall-Detection-Data\processed\detection\yolo\train"
)

NORMAL_IMAGES = ORIGINAL_TRAIN / "images"
NORMAL_LABELS = ORIGINAL_TRAIN / "labels"


# ------------------------------------------------------------
# 3. Final fine-tuning dataset
# ------------------------------------------------------------
OUTPUT = Path(
    r"D:\Fall-Detection-Data\processed\detection\hardcase_ft_030"
)


# ============================================================
# SETTINGS
# ============================================================

NORMAL_IMAGES_TO_ADD = 1000
RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_images(folder):
    """Return all supported image files from a folder."""
    if not folder.exists():
        raise FileNotFoundError(
            f"Folder does not exist:\n{folder}"
        )

    return sorted(
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def clear_output():
    """Delete previous output dataset so we start fresh."""
    if OUTPUT.exists():
        print(f"Removing existing output dataset:\n{OUTPUT}")
        shutil.rmtree(OUTPUT)

    (OUTPUT / "images" / "train").mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT / "labels" / "train").mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# START
# ============================================================

print("=" * 70)
print("Milestone 1 - Hard Case Fine-Tuning Dataset Builder")
print("=" * 70)
print()


# ============================================================
# CHECK SOURCE FOLDERS
# ============================================================

print("Checking source folders...")
print()

print(f"Fall images : {FALL_IMAGES}")
print(f"Fall labels : {FALL_LABELS}")
print()

print(f"Normal images : {NORMAL_IMAGES}")
print(f"Normal labels : {NORMAL_LABELS}")
print()

# Check existence
for folder in [
    FALL_IMAGES,
    FALL_LABELS,
    NORMAL_IMAGES,
    NORMAL_LABELS
]:
    if not folder.exists():
        raise FileNotFoundError(
            f"\nRequired folder was not found:\n{folder}"
        )

print("All source folders found.")
print()


# ============================================================
# CLEAN / CREATE OUTPUT
# ============================================================

clear_output()

OUTPUT_IMAGES = OUTPUT / "images" / "train"
OUTPUT_LABELS = OUTPUT / "labels" / "train"


# ============================================================
# STEP 1
# COPY 52 HARD-CASE IMAGES + LABELS
# ============================================================

print("=" * 70)
print("STEP 1 - COPYING HARD-CASE FALL IMAGES")
print("=" * 70)

fall_images = get_images(FALL_IMAGES)

print(f"Hard-case images found: {len(fall_images)}")

if len(fall_images) != 52:
    print(
        f"WARNING: Expected 52 hard-case images, "
        f"but found {len(fall_images)}."
    )

copied_hard = 0

for image_path in fall_images:

    label_path = FALL_LABELS / f"{image_path.stem}.txt"

    if not label_path.exists():
        print(
            f"WARNING: Missing label for:\n"
            f"  {image_path.name}"
        )
        continue

    # Copy original hard-case image
    shutil.copy2(
        image_path,
        OUTPUT_IMAGES / image_path.name
    )

    # Copy corresponding YOLO label
    shutil.copy2(
        label_path,
        OUTPUT_LABELS / label_path.name
    )

    copied_hard += 1


print()
print(f"Hard-case pairs copied: {copied_hard}")


# ============================================================
# STEP 2
# FIND NORMAL IMAGE + LABEL PAIRS
# ============================================================

print()
print("=" * 70)
print("STEP 2 - SEARCHING NORMAL TRAINING DATA")
print("=" * 70)

normal_images = get_images(NORMAL_IMAGES)

normal_pairs = []

for image_path in normal_images:

    label_path = NORMAL_LABELS / f"{image_path.stem}.txt"

    if label_path.exists():
        normal_pairs.append(
            (image_path, label_path)
        )


print(f"Normal image/label pairs available: {len(normal_pairs)}")

if len(normal_pairs) < NORMAL_IMAGES_TO_ADD:
    raise RuntimeError(
        f"\nNot enough normal image/label pairs.\n"
        f"Available : {len(normal_pairs)}\n"
        f"Required  : {NORMAL_IMAGES_TO_ADD}"
    )


# ============================================================
# STEP 3
# RANDOMLY SELECT 1000 NORMAL PAIRS
# ============================================================

print()
print("=" * 70)
print("STEP 3 - SELECTING 1000 NORMAL IMAGES")
print("=" * 70)

random.seed(RANDOM_SEED)

selected_normal = random.sample(
    normal_pairs,
    NORMAL_IMAGES_TO_ADD
)

print(
    f"Selected {len(selected_normal)} normal "
    f"image/label pairs."
)


# ============================================================
# STEP 4
# COPY NORMAL IMAGES + LABELS
# ============================================================

print()
print("=" * 70)
print("STEP 4 - COPYING NORMAL REPLAY DATA")
print("=" * 70)

copied_normal = 0

for image_path, label_path in selected_normal:

    # Prefix all normal images/labels to prevent filename
    # collisions with the 52 hard-case files.
    new_image_name = (
        f"normal_{image_path.name}"
    )

    new_label_name = (
        f"normal_{image_path.stem}.txt"
    )

    shutil.copy2(
        image_path,
        OUTPUT_IMAGES / new_image_name
    )

    shutil.copy2(
        label_path,
        OUTPUT_LABELS / new_label_name
    )

    copied_normal += 1


print(f"Normal pairs copied: {copied_normal}")


# ============================================================
# STEP 5
# CREATE data.yaml
# ============================================================

print()
print("=" * 70)
print("STEP 5 - CREATING data.yaml")
print("=" * 70)

# Use forward slashes because they work cleanly in YAML.
yaml_path = OUTPUT.as_posix()

yaml_content = f"""path: "{yaml_path}"
train: images/train

nc: 1
names:
  0: person
"""

data_yaml = OUTPUT / "data.yaml"

data_yaml.write_text(
    yaml_content,
    encoding="utf-8"
)

print(f"Created:\n{data_yaml}")


# ============================================================
# STEP 6
# FINAL VERIFICATION
# ============================================================

print()
print("=" * 70)
print("STEP 6 - FINAL DATASET VERIFICATION")
print("=" * 70)

final_images = [
    p for p in OUTPUT_IMAGES.iterdir()
    if p.is_file()
    and p.suffix.lower() in IMAGE_EXTENSIONS
]

final_labels = [
    p for p in OUTPUT_LABELS.iterdir()
    if p.is_file()
    and p.suffix.lower() == ".txt"
]

expected_images = 52 + 1000
expected_labels = 52 + 1000


print()
print(f"Hard-case pairs : {copied_hard}")
print(f"Normal pairs    : {copied_normal}")
print()
print(f"Final images    : {len(final_images)}")
print(f"Final labels    : {len(final_labels)}")
print(f"Expected images : {expected_images}")
print(f"Expected labels : {expected_labels}")
print()


# ============================================================
# CHECK IMAGE/LABEL MATCHING
# ============================================================

print("Checking image/label matching...")

image_stems = {
    p.stem
    for p in final_images
}

label_stems = {
    p.stem
    for p in final_labels
}

missing_labels = image_stems - label_stems
missing_images = label_stems - image_stems


if missing_labels:
    print()
    print("ERROR: Images without labels:")
    for name in sorted(missing_labels):
        print(f"  {name}")

if missing_images:
    print()
    print("ERROR: Labels without images:")
    for name in sorted(missing_images):
        print(f"  {name}")


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 70)

if (
    len(final_images) == expected_images
    and len(final_labels) == expected_labels
    and not missing_labels
    and not missing_images
):
    print("SUCCESS")
    print("=" * 70)
    print()
    print("The fine-tuning dataset is ready.")
    print()
    print(f"Images : {len(final_images)}")
    print(f"Labels : {len(final_labels)}")
    print()
    print(f"Dataset:")
    print(f"  {OUTPUT}")
    print()
    print("data.yaml:")
    print(f"  {data_yaml}")
    print()
else:
    print("DATASET VERIFICATION FAILED")
    print("=" * 70)
    print()
    print("Please check the warnings/errors above.")


print()
print("Done.")