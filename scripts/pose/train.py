
import os
import sys
import csv
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


# ============================================================
# Project Path
# ============================================================

PROJECT_ROOT = "/kaggle/working/Fall-Detection-System"

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# Imports
# ============================================================

from src.pose.dataset import COCOKeypointDataset
from src.pose.bam_hrnet import BAMHRNet


# ============================================================
# Configuration
# ============================================================

CONFIG_PATH = os.path.join(
    PROJECT_ROOT,
    "configs",
    "pose",
    "bam_hrnet.yaml"
)

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("BAM-HRNet Training")
print("=" * 60)

print(f"Device: {device}")

if device.type == "cuda":
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# Weighted MSE Loss
# ============================================================

class WeightedHeatmapMSELoss(nn.Module):

    def __init__(self, alpha=10.0):
        super().__init__()

        self.alpha = alpha

    def forward(self, prediction, target):

        # Give higher importance to pixels
        # around the actual keypoint locations.
        weight = 1.0 + self.alpha * target

        loss = weight * (
            prediction - target
        ) ** 2

        return loss.mean()


# ============================================================
# Training Dataset
# ============================================================

print("\nLoading training dataset...")

train_dataset = COCOKeypointDataset(
    json_path=config["train_json"],
    image_dir=config["train_images"],
    input_size=config["input_size"],
    heatmap_size=config["heatmap_size"],
    random_erasing=config["random_erasing"]
)

print(
    f"Training samples: {len(train_dataset)}"
)


# ============================================================
# Validation Dataset
# ============================================================

print("\nLoading validation dataset...")

val_dataset = COCOKeypointDataset(
    json_path=config["val_json"],
    image_dir=config["val_images"],
    input_size=config["input_size"],
    heatmap_size=config["heatmap_size"],
    random_erasing=False
)

print(
    f"Validation samples: {len(val_dataset)}"
)


# ============================================================
# Apply Subsets
# ============================================================

train_subset_size = config.get(
    "train_subset"
)

val_subset_size = config.get(
    "val_subset"
)


if train_subset_size is not None:

    train_dataset = Subset(
        train_dataset,
        range(
            min(
                train_subset_size,
                len(train_dataset)
            )
        )
    )

    print(
        f"Using training subset: {len(train_dataset)}"
    )


if val_subset_size is not None:

    val_dataset = Subset(
        val_dataset,
        range(
            min(
                val_subset_size,
                len(val_dataset)
            )
        )
    )

    print(
        f"Using validation subset: {len(val_dataset)}"
    )


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=config["num_workers"],
    pin_memory=config["pin_memory"]
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=config["num_workers"],
    pin_memory=config["pin_memory"]
)


# ============================================================
# Create Model
# ============================================================

print("\nCreating BAM-HRNet...")

model = BAMHRNet(
    num_keypoints=config["num_keypoints"],
    num_blocks=config["num_blocks"]
).to(device)


# ============================================================
# Loss
# ============================================================

criterion = WeightedHeatmapMSELoss(
    alpha=10.0
)


# ============================================================
# Optimizer
# ============================================================

optimizer = AdamW(
    model.parameters(),
    lr=config["learning_rate"],
    weight_decay=config["weight_decay"]
)


# ============================================================
# Scheduler
# ============================================================

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# ============================================================
# AMP
# ============================================================

use_amp = (
    config["use_amp"]
    and device.type == "cuda"
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp
)


# ============================================================
# Output Directories
# ============================================================

checkpoint_dir = config[
    "checkpoint_dir"
]

training_results_dir = config[
    "training_results_dir"
]

os.makedirs(
    checkpoint_dir,
    exist_ok=True
)

os.makedirs(
    training_results_dir,
    exist_ok=True
)


# ============================================================
# Training History
# ============================================================

history = []

best_val_loss = float("inf")


# ============================================================
# Training Loop
# ============================================================

for epoch in range(
    config["epochs"]
):

    model.train()

    running_train_loss = 0.0

    total_batches = len(
        train_loader
    )

    for batch_idx, (
        images,
        target_heatmaps
    ) in enumerate(train_loader):

        images = images.to(
            device,
            non_blocking=True
        )

        target_heatmaps = target_heatmaps.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Forward + Loss
        # ----------------------------------------------------

        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp
        ):

            predicted_heatmaps = model(
                images
            )

            loss = criterion(
                predicted_heatmaps,
                target_heatmaps
            )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        running_train_loss += (
            loss.item()
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            batch_idx == 0
            or (batch_idx + 1) % 100 == 0
            or batch_idx + 1 == total_batches
        ):

            print(
                f"Epoch [{epoch + 1}/{config['epochs']}] "
                f"Batch [{batch_idx + 1}/{total_batches}] "
                f"Loss: {loss.item():.6f}"
            )


    # ========================================================
    # Average Training Loss
    # ========================================================

    train_loss = (
        running_train_loss
        / total_batches
    )


    # ========================================================
    # Validation
    # ========================================================

    model.eval()

    running_val_loss = 0.0

    with torch.no_grad():

        for (
            images,
            target_heatmaps
        ) in val_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            target_heatmaps = target_heatmaps.to(
                device,
                non_blocking=True
            )

            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp
            ):

                predicted_heatmaps = model(
                    images
                )

                loss = criterion(
                    predicted_heatmaps,
                    target_heatmaps
                )

            running_val_loss += (
                loss.item()
            )


    val_loss = (
        running_val_loss
        / len(val_loader)
    )


    # ========================================================
    # Learning Rate Scheduler
    # ========================================================

    scheduler.step(
        val_loss
    )

    current_lr = optimizer.param_groups[0][
        "lr"
    ]


    # ========================================================
    # Save Latest Checkpoint
    # ========================================================

    latest_checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss
    }

    torch.save(
        latest_checkpoint,
        os.path.join(
            checkpoint_dir,
            "latest.pth"
        )
    )


    # ========================================================
    # Save Best Checkpoint
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            latest_checkpoint,
            os.path.join(
                checkpoint_dir,
                "best.pth"
            )
        )

        print(
            f"✓ Best model saved: "
            f"{os.path.join(checkpoint_dir, 'best.pth')}"
        )


    # ========================================================
    # Record History
    # ========================================================

    history.append(
        [
            epoch + 1,
            train_loss,
            val_loss,
            current_lr
        ]
    )


    # ========================================================
    # Epoch Summary
    # ========================================================

    print()
    print("-" * 60)
    print(
        f"Epoch {epoch + 1}/{config['epochs']}"
    )

    print(
        f"Train Loss : {train_loss:.6f}"
    )

    print(
        f"Val Loss   : {val_loss:.6f}"
    )

    print(
        f"Learning Rate: {current_lr:.8f}"
    )

    print("-" * 60)
    print()


# ============================================================
# Save Training History
# ============================================================

history_path = os.path.join(
    training_results_dir,
    "training_history.csv"
)

with open(
    history_path,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "epoch",
            "train_loss",
            "val_loss",
            "learning_rate"
        ]
    )

    writer.writerows(
        history
    )


# ============================================================
# Final Summary
# ============================================================

print("=" * 60)
print("Training completed!")
print("=" * 60)

print(
    f"Best validation loss: "
    f"{best_val_loss}"
)

print(
    f"Training history saved to:"
)

print(
    history_path
)
