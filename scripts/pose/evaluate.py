
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to Python path
PROJECT_ROOT = "/kaggle/working/Fall-Detection-System"
sys.path.insert(0, PROJECT_ROOT)

from src.pose.dataset import COCOKeypointDataset
from src.pose.bam_hrnet import BAMHRNet


# ============================================================
# Configuration
# ============================================================

VAL_JSON = "/kaggle/input/datasets/pushkarpatil16062005/fall-detection-dataset/COCO Keypoint Dataset/COCO Keypoint Dataset/person_keypoints_val2017.json"

VAL_IMAGES = "/kaggle/input/datasets/pushkarpatil16062005/fall-detection-dataset/Fall-Detection-YOLO/yolo/val/images"

MODEL_PATH = f"{PROJECT_ROOT}/models/pose/best.pth"

RESULTS_DIR = f"{PROJECT_ROOT}/results/milestone-2/evaluation"

BATCH_SIZE = 16
NUM_WORKERS = 4
NUM_KEYPOINTS = 17
NUM_BLOCKS = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("BAM-HRNet Evaluation")
    print("=" * 60)

    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Best model not found:\n{MODEL_PATH}"
        )

    print(f"\nLoading model:")
    print(MODEL_PATH)

    # --------------------------------------------------------
    # Validation dataset
    # --------------------------------------------------------

    print("\nLoading validation dataset...")

    val_dataset = COCOKeypointDataset(
        json_path=VAL_JSON,
        image_dir=VAL_IMAGES,
        input_size=256,
        heatmap_size=64,
        random_erasing=False
    )

    print(f"Validation samples: {len(val_dataset)}")

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    print("\nCreating BAM-HRNet...")

    model = BAMHRNet(
        num_keypoints=NUM_KEYPOINTS,
        num_blocks=NUM_BLOCKS
    )

    model = model.to(DEVICE)

    # --------------------------------------------------------
    # Load best checkpoint
    # --------------------------------------------------------

    print("\nLoading best checkpoint...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    print("✓ Best model loaded successfully")

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    criterion = nn.MSELoss()

    model.eval()

    total_loss = 0.0
    total_samples = 0

    print("\nEvaluating...")

    with torch.no_grad():

        for batch_idx, (images, targets) in enumerate(val_loader):

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            targets = targets.to(
                DEVICE,
                non_blocking=True
            )

            predictions = model(images)

            loss = criterion(
                predictions,
                targets
            )

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"Batch [{batch_idx + 1}/{len(val_loader)}] "
                    f"Loss: {loss.item():.6f}"
                )

    average_loss = total_loss / total_samples

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    results_file = os.path.join(
        RESULTS_DIR,
        "evaluation_results.txt"
    )

    print("\n" + "=" * 60)
    print("Evaluation Complete")
    print("=" * 60)

    print(f"Validation samples : {total_samples}")
    print(f"Validation Loss    : {average_loss:.6f}")

    with open(results_file, "w") as f:

        f.write("BAM-HRNet Evaluation Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Validation samples : {total_samples}\n")
        f.write(f"Validation loss    : {average_loss:.6f}\n")
        f.write(f"Model              : {MODEL_PATH}\n")

    print(f"\nResults saved to:")
    print(results_file)


if __name__ == "__main__":
    main()
