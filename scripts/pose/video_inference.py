"""
BAM-HRNet video inference for the final Fall-Detection-System project.

Pipeline:
    video frame
      -> Milestone 1 PersonDetector (YOLO)
      -> person crop
      -> BAM-HRNet
      -> 17 COCO keypoints
      -> skeleton + bounding box
      -> output video

This script does not retrain either model.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.detection import PersonDetector
from src.pose.bam_hrnet import BAMHRNet


# ---------------------------------------------------------------------
# Model / pose configuration matches the training notebook
# ---------------------------------------------------------------------
NUM_KEYPOINTS = 17
NUM_BLOCKS = 4
INPUT_SIZE = 256
HEATMAP_SIZE = 64

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
    "right_ankle",
]

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_pose_model(model_path: Path, device: torch.device):
    model = BAMHRNet(
        num_keypoints=NUM_KEYPOINTS,
        num_blocks=NUM_BLOCKS,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)

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

    return model


def heatmaps_to_keypoints(heatmaps):
    """
    Exactly follows the heatmap -> keypoint approach used in
    the supplied BAM-HRNet notebook.

    Returns:
        keypoints: (17, 2), coordinates in 256x256 crop space
        scores:    (17,), raw heatmap maximum values
    """
    keypoints = []
    scores = []

    for joint in range(NUM_KEYPOINTS):
        heatmap = heatmaps[joint]

        y, x = np.unravel_index(
            np.argmax(heatmap),
            heatmap.shape,
        )

        score = float(heatmap[y, x])

        x = x * INPUT_SIZE / HEATMAP_SIZE
        y = y * INPUT_SIZE / HEATMAP_SIZE

        keypoints.append([x, y])
        scores.append(score)

    return np.asarray(keypoints, dtype=np.float32), np.asarray(scores)


def preprocess_crop(crop):
    """Match the dataset normalization used during BAM-HRNet training."""
    crop = cv2.resize(
        crop,
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )

    # OpenCV BGR -> RGB
    image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = (image - MEAN) / STD

    # HWC -> CHW
    image = np.transpose(image, (2, 0, 1))
    tensor = torch.from_numpy(image.copy()).float().unsqueeze(0)

    return tensor


def draw_skeleton(
    image,
    keypoints,
    scores=None,
    point_threshold=None,
):
    image = image.copy()

    # Draw bones
    for a, b in SKELETON:
        if point_threshold is not None and scores is not None:
            if scores[a] < point_threshold or scores[b] < point_threshold:
                continue

        xa, ya = keypoints[a]
        xb, yb = keypoints[b]

        cv2.line(
            image,
            (int(round(xa)), int(round(ya))),
            (int(round(xb)), int(round(yb))),
            (0, 255, 0),
            2,
        )

    # Draw joints
    for i, (x, y) in enumerate(keypoints):
        if point_threshold is not None and scores is not None:
            if scores[i] < point_threshold:
                continue

        x = int(round(x))
        y = int(round(y))

        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(image, (x, y), 4, (0, 0, 255), -1)

    return image


def main():
    parser = argparse.ArgumentParser(
        description="Run Milestone 1 YOLO + BAM-HRNet on a video."
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Input video path.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output video path. Defaults to results/milestone-2/video_output.mp4",
    )

    parser.add_argument(
        "--detector-model",
        default="models/detection/best.pt",
        help="Milestone 1 YOLO model.",
    )

    parser.add_argument(
        "--pose-model",
        default="models/pose/best.pth",
        help="BAM-HRNet checkpoint.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.30,
        help="YOLO person confidence threshold.",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch/YOLO device, e.g. cpu or 0.",
    )

    parser.add_argument(
        "--keypoint-threshold",
        type=float,
        default=None,
        help="Optional raw heatmap threshold. Leave unset to draw all predicted joints.",
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    detector_path = PROJECT_ROOT / args.detector_model
    pose_path = PROJECT_ROOT / args.pose_model

    if args.output is None:
        output_path = (
            PROJECT_ROOT
            / "results"
            / "milestone-2"
            / "video_output.mp4"
        )
    else:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path

    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    if not detector_path.exists():
        raise FileNotFoundError(
            f"YOLO detector model not found: {detector_path}"
        )

    if not pose_path.exists():
        raise FileNotFoundError(
            f"BAM-HRNet model not found: {pose_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Device
    # -----------------------------------------------------------------
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device.isdigit() and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device(args.device)

    print("=" * 70)
    print("YOLO + BAM-HRNet VIDEO INFERENCE")
    print("=" * 70)
    print(f"Project : {PROJECT_ROOT}")
    print(f"Video   : {video_path}")
    print(f"YOLO    : {detector_path}")
    print(f"Pose    : {pose_path}")
    print(f"Device  : {device}")
    print(f"Output  : {output_path}")

    # -----------------------------------------------------------------
    # Load models
    # -----------------------------------------------------------------
    print("\nLoading person detector...")
    detector = PersonDetector(
        model_path=detector_path,
        confidence=args.conf,
        image_size=640,
        device=args.device,
    )
    print("✓ Person detector loaded")

    print("\nLoading BAM-HRNet...")
    pose_model = load_pose_model(pose_path, device)
    print("✓ BAM-HRNet loaded")

    # -----------------------------------------------------------------
    # Open video
    # -----------------------------------------------------------------
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or np.isnan(fps) or fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # mp4v is widely supported on Windows/OpenCV
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    print(
        f"\nVideo: {width}x{height}, {fps:.2f} FPS, "
        f"{frame_count if frame_count > 0 else '?'} frames"
    )
    print("\nProcessing... Press Q in the preview window to stop.")

    frame_index = 0

    try:
        with torch.no_grad():
            while True:
                ok, frame = cap.read()

                if not ok:
                    break

                frame_index += 1
                output_frame = frame.copy()

                detections = detector.detect(frame)

                for detection in detections:
                    x1, y1, x2, y2 = detection["bbox"]
                    confidence = detection["confidence"]

                    # Clamp bbox to frame
                    x1 = max(0, min(int(round(x1)), width - 1))
                    y1 = max(0, min(int(round(y1)), height - 1))
                    x2 = max(0, min(int(round(x2)), width))
                    y2 = max(0, min(int(round(y2)), height))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = frame[y1:y2, x1:x2]

                    if crop.size == 0:
                        continue

                    crop_h, crop_w = crop.shape[:2]

                    # BAM-HRNet input
                    pose_input = preprocess_crop(crop).to(
                        device,
                        non_blocking=True,
                    )

                    heatmaps = pose_model(pose_input)

                    # The supplied notebook uses raw heatmap argmax.
                    heatmaps_np = heatmaps[0].detach().cpu().numpy()

                    keypoints_256, scores = heatmaps_to_keypoints(
                        heatmaps_np
                    )

                    # Map 256x256 crop coordinates back to original frame.
                    keypoints_frame = keypoints_256.copy()
                    keypoints_frame[:, 0] = (
                        x1 + keypoints_256[:, 0] * crop_w / INPUT_SIZE
                    )
                    keypoints_frame[:, 1] = (
                        y1 + keypoints_256[:, 1] * crop_h / INPUT_SIZE
                    )

                    output_frame = draw_skeleton(
                        output_frame,
                        keypoints_frame,
                        scores,
                        args.keypoint_threshold,
                    )

                    # Person bounding box
                    cv2.rectangle(
                        output_frame,
                        (x1, y1),
                        (x2, y2),
                        (255, 0, 0),
                        2,
                    )

                    cv2.putText(
                        output_frame,
                        f"Person {confidence:.2f}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )

                # Frame counter
                cv2.putText(
                    output_frame,
                    f"Frame: {frame_index}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                writer.write(output_frame)

                # Preview
                cv2.imshow("YOLO + BAM-HRNet", output_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nStopped by user.")
                    break

                if frame_index % 50 == 0:
                    if frame_count > 0:
                        percent = 100.0 * frame_index / frame_count
                        print(
                            f"Processed {frame_index}/{frame_count} "
                            f"({percent:.1f}%)"
                        )
                    else:
                        print(f"Processed {frame_index} frames")

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Processed frames : {frame_index}")
    print(f"Output video     : {output_path}")


if __name__ == "__main__":
    main()
