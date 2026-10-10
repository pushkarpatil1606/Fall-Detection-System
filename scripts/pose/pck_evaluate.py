
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = "/kaggle/working/Fall-Detection-System"
sys.path.insert(0, PROJECT_ROOT)

from src.pose.dataset import COCOKeypointDataset
from src.pose.bam_hrnet import BAMHRNet


VAL_JSON = "/kaggle/input/datasets/pushkarpatil16062005/fall-detection-dataset/COCO Keypoint Dataset/COCO Keypoint Dataset/person_keypoints_val2017.json"

VAL_IMAGES = "/kaggle/input/datasets/pushkarpatil16062005/fall-detection-dataset/Fall-Detection-YOLO/yolo/val/images"

MODEL_PATH = f"{PROJECT_ROOT}/models/pose/best.pth"

OUTPUT_DIR = f"{PROJECT_ROOT}/results/milestone-2/evaluation"

BATCH_SIZE = 16
NUM_WORKERS = 4
NUM_KEYPOINTS = 17
NUM_BLOCKS = 4

INPUT_SIZE = 256
HEATMAP_SIZE = 64

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

JOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle"
]


def heatmaps_to_keypoints(heatmaps):

    batch_size = heatmaps.shape[0]

    keypoints = []

    for b in range(batch_size):

        sample_points = []

        for j in range(NUM_KEYPOINTS):

            heatmap = heatmaps[b, j]

            y, x = torch.nonzero(
                heatmap == heatmap.max(),
                as_tuple=False
            )[0]

            x = x.item() * INPUT_SIZE / HEATMAP_SIZE
            y = y.item() * INPUT_SIZE / HEATMAP_SIZE

            sample_points.append([x, y])

        keypoints.append(sample_points)

    return np.asarray(keypoints)


def main():

    print("=" * 65)
    print("BAM-HRNet PCK Evaluation — Corrected")
    print("=" * 65)

    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("\nLoading validation dataset...")

    dataset = COCOKeypointDataset(
        json_path=VAL_JSON,
        image_dir=VAL_IMAGES,
        input_size=INPUT_SIZE,
        heatmap_size=HEATMAP_SIZE,
        random_erasing=False
    )

    print(f"Validation samples: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nCreating BAM-HRNet...")

    model = BAMHRNet(
        num_keypoints=NUM_KEYPOINTS,
        num_blocks=NUM_BLOCKS
    ).to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

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
    model.eval()

    print("✓ Best model loaded successfully")

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    thresholds = [0.05, 0.10, 0.20]

    correct = {
        threshold: np.zeros(NUM_KEYPOINTS)
        for threshold in thresholds
    }

    # IMPORTANT:
    # Count visibility only ONCE.
    total_visible = np.zeros(NUM_KEYPOINTS)

    total_samples = 0

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print("\nEvaluating PCK...")

    with torch.no_grad():

        for batch_idx, (images, targets) in enumerate(loader):

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            targets = targets.to(
                DEVICE,
                non_blocking=True
            )

            predictions = model(images)

            pred_points = heatmaps_to_keypoints(
                predictions
            )

            target_points = heatmaps_to_keypoints(
                targets
            )

            # Distance in 256x256 coordinates
            distances = np.linalg.norm(
                pred_points - target_points,
                axis=2
            )

            # Normalize by image diagonal
            normalized_distances = (
                distances /
                np.sqrt(
                    INPUT_SIZE ** 2 +
                    INPUT_SIZE ** 2
                )
            )

            # Ground-truth visibility
            target_max = targets.amax(
                dim=(2, 3)
            ).cpu().numpy()

            visible = target_max > 0.05

            # Count visible joints ONCE
            for j in range(NUM_KEYPOINTS):

                total_visible[j] += np.sum(
                    visible[:, j]
                )

            # Calculate correct predictions
            for threshold in thresholds:

                matches = (
                    normalized_distances <= threshold
                )

                for j in range(NUM_KEYPOINTS):

                    joint_visible = visible[:, j]

                    correct[threshold][j] += np.sum(
                        matches[:, j] &
                        joint_visible
                    )

            total_samples += images.shape[0]

            if (batch_idx + 1) % 50 == 0:

                print(
                    f"Batch [{batch_idx + 1}/{len(loader)}]"
                )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("PCK RESULTS")
    print("=" * 65)

    overall_results = {}

    for threshold in thresholds:

        overall = (
            np.sum(correct[threshold]) /
            np.maximum(
                np.sum(total_visible),
                1
            )
        )

        overall_results[threshold] = overall

        print(
            f"\nPCK@{threshold:.2f}: "
            f"{overall * 100:.2f}%"
        )

    # --------------------------------------------------------
    # Per-joint results
    # --------------------------------------------------------

    print("\n" + "-" * 65)
    print("PER-JOINT PCK")
    print("-" * 65)

    print(
        f"{'Joint':<16}"
        f"{'PCK@0.05':>12}"
        f"{'PCK@0.10':>12}"
        f"{'PCK@0.20':>12}"
    )

    print("-" * 52)

    for j, name in enumerate(JOINT_NAMES):

        scores = []

        for threshold in thresholds:

            score = (
                correct[threshold][j] /
                max(total_visible[j], 1)
            )

            scores.append(score * 100)

        print(
            f"{name:<16}"
            f"{scores[0]:>11.2f}%"
            f"{scores[1]:>11.2f}%"
            f"{scores[2]:>11.2f}%"
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        "pck_results.txt"
    )

    with open(output_file, "w") as f:

        f.write("BAM-HRNet PCK Evaluation — Corrected\n")
        f.write("=" * 60 + "\n\n")

        f.write(
            f"Validation samples: {total_samples}\n\n"
        )

        for threshold in thresholds:

            f.write(
                f"PCK@{threshold:.2f}: "
                f"{overall_results[threshold] * 100:.2f}%\n"
            )

        f.write("\nPer-joint PCK\n")
        f.write("-" * 60 + "\n")

        for j, name in enumerate(JOINT_NAMES):

            f.write(f"\n{name}\n")

            for threshold in thresholds:

                score = (
                    correct[threshold][j] /
                    max(total_visible[j], 1)
                )

                f.write(
                    f"  PCK@{threshold:.2f}: "
                    f"{score * 100:.2f}%\n"
                )

    print("\nResults saved to:")
    print(output_file)


if __name__ == "__main__":
    main()
