
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.pose.hrnet import HRNetBackbone
from src.pose.bam import BAM


# ============================================================
# BAM-HRNet
# ============================================================

class BAMHRNet(nn.Module):

    def __init__(
        self,
        num_keypoints=17,
        num_blocks=4
    ):
        super().__init__()

        # ----------------------------------------------------
        # HRNet backbone
        # ----------------------------------------------------

        self.backbone = HRNetBackbone(
            num_blocks=num_blocks
        )

        # ----------------------------------------------------
        # BAM modules
        #
        # Branch 1 is intentionally left unchanged.
        #
        # Branch 2 → 64 channels
        # Branch 3 → 128 channels
        # Branch 4 → 256 channels
        # ----------------------------------------------------

        self.bam_branch2 = BAM(
            channels=64
        )

        self.bam_branch3 = BAM(
            channels=128
        )

        self.bam_branch4 = BAM(
            channels=256
        )

        # ----------------------------------------------------
        # Convert all branches to 32 channels
        # ----------------------------------------------------

        self.branch1_proj = nn.Sequential(
            nn.Conv2d(
                32,
                32,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.branch2_proj = nn.Sequential(
            nn.Conv2d(
                64,
                32,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.branch3_proj = nn.Sequential(
            nn.Conv2d(
                128,
                32,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.branch4_proj = nn.Sequential(
            nn.Conv2d(
                256,
                32,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # ----------------------------------------------------
        # Final feature processing
        # ----------------------------------------------------

        self.feature_conv = nn.Sequential(
            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # ----------------------------------------------------
        # 17-keypoint heatmap head
        # ----------------------------------------------------

        self.keypoint_head = nn.Conv2d(
            32,
            num_keypoints,
            kernel_size=1
        )

    def forward(self, x):

        # ====================================================
        # HRNet
        # ====================================================

        branches = self.backbone(x)

        branch1, branch2, branch3, branch4 = branches

        # ====================================================
        # BAM
        #
        # Highest-resolution branch remains unchanged.
        # ====================================================

        branch2 = self.bam_branch2(branch2)

        branch3 = self.bam_branch3(branch3)

        branch4 = self.bam_branch4(branch4)

        # ====================================================
        # Channel projection
        # ====================================================

        branch1 = self.branch1_proj(branch1)

        branch2 = self.branch2_proj(branch2)

        branch3 = self.branch3_proj(branch3)

        branch4 = self.branch4_proj(branch4)

        # ====================================================
        # Upsample all branches to highest resolution
        # ====================================================

        target_size = branch1.shape[-2:]

        branch2 = F.interpolate(
            branch2,
            size=target_size,
            mode="bilinear",
            align_corners=False
        )

        branch3 = F.interpolate(
            branch3,
            size=target_size,
            mode="bilinear",
            align_corners=False
        )

        branch4 = F.interpolate(
            branch4,
            size=target_size,
            mode="bilinear",
            align_corners=False
        )

        # ====================================================
        # Final feature fusion
        # ====================================================

        fused = (
            branch1 +
            branch2 +
            branch3 +
            branch4
        )

        fused = self.feature_conv(fused)

        # ====================================================
        # Keypoint heatmaps
        # ====================================================

        heatmaps = self.keypoint_head(fused)

        return heatmaps
