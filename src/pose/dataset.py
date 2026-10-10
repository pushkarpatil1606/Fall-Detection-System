
import json
import os
import random

import cv2
import numpy as np
import torch

from torch.utils.data import Dataset


# ============================================================
# COCO Keypoint Dataset
# ============================================================

class COCOKeypointDataset(Dataset):

    def __init__(
        self,
        json_path,
        image_dir,
        input_size=256,
        heatmap_size=64,
        random_erasing=False
    ):

        self.json_path = json_path
        self.image_dir = image_dir

        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.random_erasing = random_erasing

        # ----------------------------------------------------
        # Load annotations
        # ----------------------------------------------------

        print("Loading annotations...")

        with open(json_path, "r") as f:
            self.coco = json.load(f)

        print(
            f"JSON loaded: "
            f"{len(self.coco['images'])} images, "
            f"{len(self.coco['annotations'])} annotations"
        )

        # ----------------------------------------------------
        # Build image lookup
        # ----------------------------------------------------
        #
        # IMPORTANT:
        # We scan the directory ONCE instead of calling
        # os.path.exists() for every annotation.
        #
        # ----------------------------------------------------

        print("Scanning image directory...")

        image_files = set(
            os.listdir(image_dir)
        )

        print(
            f"Images found on disk: "
            f"{len(image_files)}"
        )

        # ----------------------------------------------------
        # Create image metadata lookup
        # ----------------------------------------------------

        self.images = {
            image["id"]: image
            for image in self.coco["images"]
        }

        # ----------------------------------------------------
        # Build usable person samples
        # ----------------------------------------------------

        self.samples = []

        skipped_no_image = 0
        skipped_not_person = 0
        skipped_no_keypoints = 0

        print("Building person samples...")

        for annotation in self.coco["annotations"]:

            # ------------------------------------------------
            # Person only
            # ------------------------------------------------

            if annotation.get("category_id") != 1:

                skipped_not_person += 1

                continue

            # ------------------------------------------------
            # Must have keypoints
            # ------------------------------------------------

            if annotation.get(
                "num_keypoints",
                0
            ) <= 0:

                skipped_no_keypoints += 1

                continue

            image_id = annotation["image_id"]

            if image_id not in self.images:

                skipped_no_image += 1

                continue

            image_info = self.images[image_id]

            filename = image_info["file_name"]

            # ------------------------------------------------
            # Kaggle dataset uses coco_ prefix
            # ------------------------------------------------

            coco_filename = "coco_" + filename

            if coco_filename in image_files:

                image_path = os.path.join(
                    self.image_dir,
                    coco_filename
                )

            elif filename in image_files:

                image_path = os.path.join(
                    self.image_dir,
                    filename
                )

            else:

                skipped_no_image += 1

                continue

            # ------------------------------------------------
            # Store sample
            # ------------------------------------------------

            self.samples.append({

                "image_id": image_id,

                "image_path": image_path,

                "bbox": annotation["bbox"],

                "keypoints": annotation["keypoints"],

                "width": image_info["width"],

                "height": image_info["height"]

            })

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("Dataset preparation complete")
        print("=" * 60)

        print(
            "Usable person samples:",
            len(self.samples)
        )

        print(
            "Skipped - not person:",
            skipped_not_person
        )

        print(
            "Skipped - no keypoints:",
            skipped_no_keypoints
        )

        print(
            "Skipped - missing image:",
            skipped_no_image
        )

        print("=" * 60)

    # ========================================================
    # Length
    # ========================================================

    def __len__(self):

        return len(self.samples)

    # ========================================================
    # Random Erasing
    # ========================================================

    def apply_random_erasing(self, image):

        if random.random() > 0.5:

            return image

        height, width, _ = image.shape

        erase_h = random.randint(
            max(1, int(height * 0.1)),
            max(1, int(height * 0.4))
        )

        erase_w = random.randint(
            max(1, int(width * 0.1)),
            max(1, int(width * 0.4))
        )

        y = random.randint(
            0,
            max(0, height - erase_h)
        )

        x = random.randint(
            0,
            max(0, width - erase_w)
        )

        # ImageNet mean in RGB
        mean_rgb = np.array(
            [123.675, 116.28, 103.53],
            dtype=np.uint8
        )

        image[
            y:y + erase_h,
            x:x + erase_w
        ] = mean_rgb

        return image

    # ========================================================
    # Generate Gaussian Heatmaps
    # ========================================================

    def generate_heatmaps(self, keypoints):

        heatmaps = np.zeros(
            (
                17,
                self.heatmap_size,
                self.heatmap_size
            ),
            dtype=np.float32
        )

        scale = (
            self.heatmap_size /
            self.input_size
        )

        sigma = 2.0

        radius = int(3 * sigma)

        for joint_idx in range(17):

            x, y, visibility = keypoints[joint_idx]

            if visibility <= 0:
                continue

            center_x = x * scale
            center_y = y * scale

            if (
                center_x < 0
                or center_x >= self.heatmap_size
                or center_y < 0
                or center_y >= self.heatmap_size
            ):
                continue

            center_x_int = int(
                round(center_x)
            )

            center_y_int = int(
                round(center_y)
            )

            x_min = max(
                0,
                center_x_int - radius
            )

            x_max = min(
                self.heatmap_size,
                center_x_int + radius + 1
            )

            y_min = max(
                0,
                center_y_int - radius
            )

            y_max = min(
                self.heatmap_size,
                center_y_int + radius + 1
            )

            for yy in range(y_min, y_max):

                for xx in range(x_min, x_max):

                    distance_sq = (
                        (xx - center_x) ** 2
                        +
                        (yy - center_y) ** 2
                    )

                    value = np.exp(
                        -distance_sq /
                        (2 * sigma * sigma)
                    )

                    heatmaps[
                        joint_idx,
                        yy,
                        xx
                    ] = max(
                        heatmaps[
                            joint_idx,
                            yy,
                            xx
                        ],
                        value
                    )

        return heatmaps

    # ========================================================
    # Get Item
    # ========================================================

    def __getitem__(self, index):

        sample = self.samples[index]

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = cv2.imread(
            sample["image_path"]
        )

        if image is None:

            raise RuntimeError(
                f"Could not read image: "
                f"{sample['image_path']}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        x, y, w, h = sample["bbox"]

        image_height, image_width = image.shape[:2]

        x1 = int(round(x))
        y1 = int(round(y))

        x2 = int(round(x + w))
        y2 = int(round(y + h))

        # ----------------------------------------------------
        # Clip bounding box
        # ----------------------------------------------------

        x1 = max(
            0,
            min(x1, image_width - 1)
        )

        y1 = max(
            0,
            min(y1, image_height - 1)
        )

        x2 = max(
            x1 + 1,
            min(x2, image_width)
        )

        y2 = max(
            y1 + 1,
            min(y2, image_height)
        )

        # ----------------------------------------------------
        # Crop person
        # ----------------------------------------------------

        crop = image[
            y1:y2,
            x1:x2
        ].copy()

        crop_height, crop_width = crop.shape[:2]

        # ----------------------------------------------------
        # Keypoints
        # ----------------------------------------------------

        keypoints = np.array(
            sample["keypoints"],
            dtype=np.float32
        ).reshape(17, 3)

        # Convert full-image coordinates
        # to crop coordinates

        keypoints[:, 0] -= x1
        keypoints[:, 1] -= y1

        # ----------------------------------------------------
        # Resize crop
        # ----------------------------------------------------

        crop = cv2.resize(
            crop,
            (
                self.input_size,
                self.input_size
            ),
            interpolation=cv2.INTER_LINEAR
        )

        # ----------------------------------------------------
        # Transform keypoints
        # ----------------------------------------------------

        scale_x = (
            self.input_size /
            crop_width
        )

        scale_y = (
            self.input_size /
            crop_height
        )

        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y

        # ----------------------------------------------------
        # Random erasing
        # ----------------------------------------------------

        if self.random_erasing:

            crop = self.apply_random_erasing(
                crop
            )

        # ----------------------------------------------------
        # Heatmaps
        # ----------------------------------------------------

        heatmaps = self.generate_heatmaps(
            keypoints
        )

        # ----------------------------------------------------
        # Normalize image
        # ----------------------------------------------------

        image = crop.astype(
            np.float32
        ) / 255.0

        mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32
        )

        std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32
        )

        image = (
            image - mean
        ) / std

        # HWC → CHW

        image = np.transpose(
            image,
            (2, 0, 1)
        )

        # ----------------------------------------------------
        # Tensor conversion
        # ----------------------------------------------------

        image_tensor = torch.from_numpy(
            image.copy()
        ).float()

        heatmap_tensor = torch.from_numpy(
            heatmaps
        ).float()

        return (
            image_tensor,
            heatmap_tensor
        )
