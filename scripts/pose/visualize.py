
import os
import sys
import random

import cv2
import numpy as np
import torch

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

OUTPUT_DIR = f"{PROJECT_ROOT}/results/milestone-2/visualizations"

NUM_KEYPOINTS = 17
NUM_BLOCKS = 4

INPUT_SIZE = 256
HEATMAP_SIZE = 64

NUM_SAMPLES = 10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# COCO 17-keypoint names
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


# COCO skeleton connections
SKELETON = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),

    (5, 6),

    (5, 7),
    (7, 9),

    (6, 8),
    (8, 10),

    (5, 11),
    (6, 12),

    (11, 12),

    (11, 13),
    (13, 15),

    (12, 14),
    (14, 16)
]


# ============================================================
# Heatmap -> keypoints
# ============================================================

def heatmaps_to_keypoints(heatmaps):

    keypoints = []
    scores = []

    for joint in range(NUM_KEYPOINTS):

        heatmap = heatmaps[joint]

        y, x = np.unravel_index(
            np.argmax(heatmap),
            heatmap.shape
        )

        score = float(heatmap[y, x])

        x = x * INPUT_SIZE / HEATMAP_SIZE
        y = y * INPUT_SIZE / HEATMAP_SIZE

        keypoints.append([x, y])
        scores.append(score)

    return (
        np.array(keypoints),
        np.array(scores)
    )


# ============================================================
# Draw skeleton
# ============================================================

def draw_skeleton(
    image,
    keypoints,
    joint_scores=None,
    point_color=(0, 255, 0),
    line_color=(0, 255, 0),
    score_threshold=None
):

    image = image.copy()

    # Draw bones
    for a, b in SKELETON:

        if score_threshold is not None:

            if (
                joint_scores[a] < score_threshold
                or
                joint_scores[b] < score_threshold
            ):
                continue

        xa, ya = keypoints[a]
        xb, yb = keypoints[b]

        xa, ya = int(round(xa)), int(round(ya))
        xb, yb = int(round(xb)), int(round(yb))

        cv2.line(
            image,
            (xa, ya),
            (xb, yb),
            line_color,
            2
        )

    # Draw joints
    for i, (x, y) in enumerate(keypoints):

        if score_threshold is not None:

            if joint_scores[i] < score_threshold:
                continue

        x = int(round(x))
        y = int(round(y))

        if (
            0 <= x < image.shape[1]
            and
            0 <= y < image.shape[0]
        ):

            cv2.circle(
                image,
                (x, y),
                4,
                point_color,
                -1
            )

    return image


# ============================================================
# Add labels
# ============================================================

def add_title(image, title):

    image = image.copy()

    cv2.rectangle(
        image,
        (0, 0),
        (image.shape[1], 30),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        image,
        title,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return image


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 65)
    print("BAM-HRNet Ground Truth vs Prediction Visualization")
    print("=" * 65)

    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

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

    print(
        f"Validation samples: {len(dataset)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nLoading BAM-HRNet...")

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
    # Random samples
    # --------------------------------------------------------

    random.seed(42)

    indices = random.sample(
        range(len(dataset)),
        min(NUM_SAMPLES, len(dataset))
    )

    print(
        f"\nGenerating {len(indices)} comparisons..."
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    for sample_number, index in enumerate(
        indices,
        start=1
    ):

        image_tensor, target_heatmaps = dataset[index]

        input_tensor = (
            image_tensor
            .unsqueeze(0)
            .to(DEVICE)
        )

        with torch.no_grad():

            prediction = model(
                input_tensor
            )

        prediction = (
            prediction[0]
            .cpu()
            .numpy()
        )

        target_heatmaps = (
            target_heatmaps
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # Convert heatmaps to coordinates
        # ----------------------------------------------------

        predicted_points, predicted_scores = (
            heatmaps_to_keypoints(prediction)
        )

        ground_truth_points, _ = (
            heatmaps_to_keypoints(target_heatmaps)
        )

        # ----------------------------------------------------
        # Recover RGB image
        # ----------------------------------------------------

        image = image_tensor.cpu().numpy()

        image = np.transpose(
            image,
            (1, 2, 0)
        )

        # Undo ImageNet normalization
        mean = np.array(
            [0.485, 0.456, 0.406]
        )

        std = np.array(
            [0.229, 0.224, 0.225]
        )

        image = (
            image * std + mean
        )

        image = np.clip(
            image,
            0,
            1
        )

        image = (
            image * 255
        ).astype(np.uint8)

        # ----------------------------------------------------
        # Ground truth
        # Green
        # ----------------------------------------------------

        gt_image = draw_skeleton(
            image,
            ground_truth_points,
            point_color=(0, 255, 0),
            line_color=(0, 255, 0)
        )

        gt_image = add_title(
            gt_image,
            "GROUND TRUTH"
        )

        # ----------------------------------------------------
        # Prediction
        # Red
        # ----------------------------------------------------

        pred_image = draw_skeleton(
            image,
            predicted_points,
            joint_scores=predicted_scores,
            point_color=(0, 0, 255),
            line_color=(0, 0, 255)
        )

        pred_image = add_title(
            pred_image,
            "BAM-HRNet PREDICTION"
        )

        # ----------------------------------------------------
        # Side by side
        # ----------------------------------------------------

        comparison = np.hstack(
            [gt_image, pred_image]
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            f"comparison_{sample_number:02d}_"
            f"index_{index}.jpg"
        )

        cv2.imwrite(
            output_path,
            cv2.cvtColor(
                comparison,
                cv2.COLOR_RGB2BGR
            )
        )

        # ----------------------------------------------------
        # Save confidence information
        # ----------------------------------------------------

        confidence_path = os.path.join(
            OUTPUT_DIR,
            f"comparison_{sample_number:02d}_"
            f"index_{index}_scores.txt"
        )

        with open(
            confidence_path,
            "w"
        ) as f:

            f.write(
                "BAM-HRNet Peak Heatmap Activations\n"
            )
            f.write("=" * 50 + "\n\n")

            for joint, score in zip(
                JOINT_NAMES,
                predicted_scores
            ):

                f.write(
                    f"{joint:<16}: "
                    f"{score:.6f}\n"
                )

        print(
            f"[{sample_number}/{len(indices)}] "
            f"{output_path}"
        )

    print("\n" + "=" * 65)
    print("Visualization complete!")
    print("=" * 65)

    print(
        f"Results saved to:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
