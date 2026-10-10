
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Bottleneck Block
# ============================================================

class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_channels, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            planes,
            kernel_size=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# ============================================================
# 2. HRNet Stem
# ============================================================

class HRNetStem(nn.Module):

    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(
            64,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


# ============================================================
# 3. Stage 1
# ============================================================

class HRNetStage1(nn.Module):

    def __init__(self, num_blocks=4):
        super().__init__()

        blocks = []

        for _ in range(num_blocks):

            blocks.append(
                Bottleneck(
                    in_channels=64,
                    planes=16
                )
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):

        return self.blocks(x)


# ============================================================
# 4. Transition 1
# ============================================================

class HRNetTransition1(nn.Module):

    def __init__(self):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(
                64,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        branch1 = self.branch1(x)
        branch2 = self.branch2(x)

        return [branch1, branch2]


# ============================================================
# 5. Fusion - 2 Branches
# ============================================================

class HRNetFusion2(nn.Module):

    def __init__(self):
        super().__init__()

        self.branch2_to_branch1 = nn.Sequential(
            nn.Conv2d(
                64,
                32,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(32)
        )

        self.branch1_to_branch2 = nn.Sequential(
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, branches):

        branch1, branch2 = branches

        b2 = self.branch2_to_branch1(branch2)

        b2 = F.interpolate(
            b2,
            size=branch1.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        out_branch1 = branch1 + b2

        b1 = self.branch1_to_branch2(branch1)

        out_branch2 = branch2 + b1

        out_branch1 = self.relu(out_branch1)
        out_branch2 = self.relu(out_branch2)

        return [
            out_branch1,
            out_branch2
        ]


# ============================================================
# 6. Transition 2
# ============================================================

class HRNetTransition2(nn.Module):

    def __init__(self):
        super().__init__()

        self.branch2_to_branch3 = nn.Sequential(
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

    def forward(self, branches):

        branch1, branch2 = branches

        branch3 = self.branch2_to_branch3(branch2)

        return [
            branch1,
            branch2,
            branch3
        ]


# ============================================================
# 7. Fusion - 3 Branches
# ============================================================

class HRNetFusion3(nn.Module):

    def __init__(self):
        super().__init__()

        self.branch2_to_branch1 = nn.Sequential(
            nn.Conv2d(
                64,
                32,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(32)
        )

        self.branch3_to_branch1 = nn.Sequential(
            nn.Conv2d(
                128,
                32,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(32)
        )

        self.branch1_to_branch2 = nn.Sequential(
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.branch3_to_branch2 = nn.Sequential(
            nn.Conv2d(
                128,
                64,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(64)
        )

        self.branch1_to_branch3 = nn.Sequential(
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.branch2_to_branch3 = nn.Sequential(
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, branches):

        branch1, branch2, branch3 = branches

        # Branch 1

        b2 = self.branch2_to_branch1(branch2)

        b2 = F.interpolate(
            b2,
            size=branch1.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        b3 = self.branch3_to_branch1(branch3)

        b3 = F.interpolate(
            b3,
            size=branch1.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        out_branch1 = branch1 + b2 + b3

        # Branch 2

        b1 = self.branch1_to_branch2(branch1)

        b3 = self.branch3_to_branch2(branch3)

        b3 = F.interpolate(
            b3,
            size=branch2.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        out_branch2 = branch2 + b1 + b3

        # Branch 3

        b1 = self.branch1_to_branch3(branch1)

        b2 = self.branch2_to_branch3(branch2)

        out_branch3 = branch3 + b1 + b2

        out_branch1 = self.relu(out_branch1)
        out_branch2 = self.relu(out_branch2)
        out_branch3 = self.relu(out_branch3)

        return [
            out_branch1,
            out_branch2,
            out_branch3
        ]


# ============================================================
# 8. Generic HRNet Branch
# ============================================================

class HRNetBranch(nn.Module):

    def __init__(self, channels, num_blocks=4):
        super().__init__()

        planes = channels // Bottleneck.expansion

        blocks = []

        for _ in range(num_blocks):

            blocks.append(
                Bottleneck(
                    in_channels=channels,
                    planes=planes
                )
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):

        return self.blocks(x)


# ============================================================
# 9. Stage 2
# ============================================================

class HRNetStage2(nn.Module):

    def __init__(self, num_blocks=4):
        super().__init__()

        self.branch1 = HRNetBranch(32, num_blocks)
        self.branch2 = HRNetBranch(64, num_blocks)
        self.branch3 = HRNetBranch(128, num_blocks)

    def forward(self, branches):

        branch1, branch2, branch3 = branches

        branch1 = self.branch1(branch1)
        branch2 = self.branch2(branch2)
        branch3 = self.branch3(branch3)

        return [
            branch1,
            branch2,
            branch3
        ]


# ============================================================
# 10. Transition 3
# ============================================================

class HRNetTransition3(nn.Module):

    def __init__(self):
        super().__init__()

        self.branch3_to_branch4 = nn.Sequential(
            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

    def forward(self, branches):

        branch1, branch2, branch3 = branches

        branch4 = self.branch3_to_branch4(branch3)

        return [
            branch1,
            branch2,
            branch3,
            branch4
        ]


# ============================================================
# 11. Fusion - 4 Branches
# ============================================================

class HRNetFusion4(nn.Module):

    def __init__(self):
        super().__init__()

        # ----------------------------------------------------
        # Branch 2 → Branch 1
        # ----------------------------------------------------

        self.b2_to_b1 = nn.Sequential(
            nn.Conv2d(
                64,
                32,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(32)
        )

        # Branch 3 → Branch 1

        self.b3_to_b1 = nn.Sequential(
            nn.Conv2d(
                128,
                32,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(32)
        )

        # Branch 4 → Branch 1

        self.b4_to_b1 = nn.Sequential(
            nn.Conv2d(
                256,
                32,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(32)
        )

        # ----------------------------------------------------
        # Branch 1 → Branch 2
        # ----------------------------------------------------

        self.b1_to_b2 = nn.Sequential(
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Branch 3 → Branch 2

        self.b3_to_b2 = nn.Sequential(
            nn.Conv2d(
                128,
                64,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(64)
        )

        # Branch 4 → Branch 2

        self.b4_to_b2 = nn.Sequential(
            nn.Conv2d(
                256,
                64,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(64)
        )

        # ----------------------------------------------------
        # Branch 1 → Branch 3
        # ----------------------------------------------------

        self.b1_to_b3 = nn.Sequential(
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Branch 2 → Branch 3

        self.b2_to_b3 = nn.Sequential(
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Branch 4 → Branch 3

        self.b4_to_b3 = nn.Sequential(
            nn.Conv2d(
                256,
                128,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(128)
        )

        # ----------------------------------------------------
        # Branch 1 → Branch 4
        # ----------------------------------------------------

        self.b1_to_b4 = nn.Sequential(
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Branch 2 → Branch 4

        self.b2_to_b4 = nn.Sequential(
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Branch 3 → Branch 4

        self.b3_to_b4 = nn.Sequential(
            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, branches):

        b1, b2, b3, b4 = branches

        # ====================================================
        # Fused Branch 1
        # ====================================================

        x21 = self.b2_to_b1(b2)

        x21 = F.interpolate(
            x21,
            size=b1.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        x31 = self.b3_to_b1(b3)

        x31 = F.interpolate(
            x31,
            size=b1.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        x41 = self.b4_to_b1(b4)

        x41 = F.interpolate(
            x41,
            size=b1.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        out1 = b1 + x21 + x31 + x41

        # ====================================================
        # Fused Branch 2
        # ====================================================

        x12 = self.b1_to_b2(b1)

        x32 = self.b3_to_b2(b3)

        x32 = F.interpolate(
            x32,
            size=b2.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        x42 = self.b4_to_b2(b4)

        x42 = F.interpolate(
            x42,
            size=b2.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        out2 = b2 + x12 + x32 + x42

        # ====================================================
        # Fused Branch 3
        # ====================================================

        x13 = self.b1_to_b3(b1)

        x23 = self.b2_to_b3(b2)

        x43 = self.b4_to_b3(b4)

        x43 = F.interpolate(
            x43,
            size=b3.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        out3 = b3 + x13 + x23 + x43

        # ====================================================
        # Fused Branch 4
        # ====================================================

        x14 = self.b1_to_b4(b1)

        x24 = self.b2_to_b4(b2)

        x34 = self.b3_to_b4(b3)

        out4 = b4 + x14 + x24 + x34

        # ====================================================
        # ReLU
        # ====================================================

        out1 = self.relu(out1)
        out2 = self.relu(out2)
        out3 = self.relu(out3)
        out4 = self.relu(out4)

        return [
            out1,
            out2,
            out3,
            out4
        ]


# ============================================================
# 12. Stage 3
# ============================================================

class HRNetStage3(nn.Module):

    def __init__(self, num_blocks=4):
        super().__init__()

        self.branch1 = HRNetBranch(32, num_blocks)
        self.branch2 = HRNetBranch(64, num_blocks)
        self.branch3 = HRNetBranch(128, num_blocks)
        self.branch4 = HRNetBranch(256, num_blocks)

    def forward(self, branches):

        branch1, branch2, branch3, branch4 = branches

        branch1 = self.branch1(branch1)
        branch2 = self.branch2(branch2)
        branch3 = self.branch3(branch3)
        branch4 = self.branch4(branch4)

        return [
            branch1,
            branch2,
            branch3,
            branch4
        ]


# ============================================================
# 13. Final Multi-Resolution Fusion
# ============================================================

class HRNetFinalFusion(nn.Module):

    def __init__(self):
        super().__init__()

        # Branch 2 → Branch 1

        self.b2_to_b1 = nn.Sequential(
            nn.Conv2d(
                64,
                32,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(32)
        )

        # Branch 3 → Branch 1

        self.b3_to_b1 = nn.Sequential(
            nn.Conv2d(
                128,
                32,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(32)
        )

        # Branch 4 → Branch 1

        self.b4_to_b1 = nn.Sequential(
            nn.Conv2d(
                256,
                32,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(32)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, branches):

        b1, b2, b3, b4 = branches

        b2 = self.b2_to_b1(b2)

        b2 = F.interpolate(
            b2,
            size=b1.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        b3 = self.b3_to_b1(b3)

        b3 = F.interpolate(
            b3,
            size=b1.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        b4 = self.b4_to_b1(b4)

        b4 = F.interpolate(
            b4,
            size=b1.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        out = b1 + b2 + b3 + b4

        return self.relu(out)


# ============================================================
# 14. Keypoint Heatmap Head
# ============================================================

class KeypointHead(nn.Module):

    def __init__(
        self,
        in_channels=32,
        num_keypoints=17
    ):
        super().__init__()

        self.head = nn.Sequential(

            nn.Conv2d(
                in_channels,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                num_keypoints,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )

    def forward(self, x):

        return self.head(x)


# ============================================================
# 15. Complete HRNet Backbone
# ============================================================

class HRNet(nn.Module):

    def __init__(
        self,
        num_keypoints=17,
        num_blocks=4
    ):
        super().__init__()

        self.stem = HRNetStem()

        self.stage1 = HRNetStage1(
            num_blocks=num_blocks
        )

        self.transition1 = HRNetTransition1()

        self.fusion2 = HRNetFusion2()

        self.transition2 = HRNetTransition2()

        self.fusion3 = HRNetFusion3()

        self.stage2 = HRNetStage2(
            num_blocks=num_blocks
        )

        self.transition3 = HRNetTransition3()

        self.fusion4 = HRNetFusion4()

        self.stage3 = HRNetStage3(
            num_blocks=num_blocks
        )

        self.final_fusion = HRNetFinalFusion()

        self.keypoint_head = KeypointHead(
            in_channels=32,
            num_keypoints=num_keypoints
        )

    def forward(self, x):

        # Stem

        x = self.stem(x)

        # Stage 1

        x = self.stage1(x)

        # Stage 2 preparation

        branches = self.transition1(x)

        branches = self.fusion2(branches)

        branches = self.transition2(branches)

        branches = self.fusion3(branches)

        branches = self.stage2(branches)

        # Stage 3 preparation

        branches = self.transition3(branches)

        branches = self.fusion4(branches)

        branches = self.stage3(branches)

        # Final fusion

        x = self.final_fusion(branches)

        # Keypoint heatmaps

        heatmaps = self.keypoint_head(x)

        return heatmaps

# ============================================================
# HRNet Backbone
# ============================================================

class HRNetBackbone(nn.Module):

    def __init__(self, num_blocks=4):
        super().__init__()

        self.stem = HRNetStem()

        self.stage1 = HRNetStage1(
            num_blocks=num_blocks
        )

        self.transition1 = HRNetTransition1()

        self.fusion2 = HRNetFusion2()

        self.transition2 = HRNetTransition2()

        self.fusion3 = HRNetFusion3()

        self.stage2 = HRNetStage2(
            num_blocks=num_blocks
        )

        self.transition3 = HRNetTransition3()

        self.fusion4 = HRNetFusion4()

        self.stage3 = HRNetStage3(
            num_blocks=num_blocks
        )

    def forward(self, x):

        # ----------------------------------------------------
        # Stem
        # ----------------------------------------------------

        x = self.stem(x)

        # ----------------------------------------------------
        # Stage 1
        # ----------------------------------------------------

        x = self.stage1(x)

        # ----------------------------------------------------
        # Create 2 branches
        # ----------------------------------------------------

        branches = self.transition1(x)

        branches = self.fusion2(branches)

        # ----------------------------------------------------
        # Create 3 branches
        # ----------------------------------------------------

        branches = self.transition2(branches)

        branches = self.fusion3(branches)

        branches = self.stage2(branches)

        # ----------------------------------------------------
        # Create 4 branches
        # ----------------------------------------------------

        branches = self.transition3(branches)

        branches = self.fusion4(branches)

        branches = self.stage3(branches)

        return branches
