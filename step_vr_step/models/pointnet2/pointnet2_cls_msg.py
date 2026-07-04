"""PointNet++ MSG classification model.

Architecture follows the original paper with Multi-Scale Grouping (MSG):
  SA1: 512 pts, radii [0.1, 0.2, 0.4], K [16, 32, 128]
  SA2: 128 pts, radii [0.2, 0.4, 0.8], K [32, 64, 128]
  SA3: global, MLP [256, 512, 1024]
  FC:  1024 → 512 → 256 → num_classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointnet2_utils import (
    PointNetSetAbstractionMsg,
    farthest_point_sample,
    index_points,
)


class PointNet2ClsMSG(nn.Module):
    """PointNet++ MSG classifier."""

    def __init__(self, num_classes: int = 40, use_normals: bool = True):
        super().__init__()
        self.use_normals = use_normals
        in_channel = 3 if use_normals else 0

        self.sa1 = PointNetSetAbstractionMsg(
            npoint=512,
            radius_list=[0.1, 0.2, 0.4],
            nsample_list=[16, 32, 128],
            in_channel=in_channel,
            mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]],
        )
        # SA1 output: 64 + 128 + 128 = 320

        self.sa2 = PointNetSetAbstractionMsg(
            npoint=128,
            radius_list=[0.2, 0.4, 0.8],
            nsample_list=[32, 64, 128],
            in_channel=320,
            mlp_list=[[64, 64, 128], [128, 128, 256], [128, 128, 256]],
        )
        # SA2 output: 128 + 256 + 256 = 640

        # SA3: global aggregation
        self.sa3_mlp = nn.Sequential(
            nn.Conv1d(640 + 3, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )

        # Classification head
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, N) where C=6 if use_normals else C=3.

        Returns:
            logits: (B, num_classes)
            global_feat: (B, 1024) global feature vector
        """
        B, C, N = x.shape
        xyz = x[:, :3, :].transpose(1, 2)  # (B, N, 3)

        if self.use_normals and C >= 6:
            normals = x[:, 3:6, :].transpose(1, 2)  # (B, N, 3)
        else:
            normals = None

        # SA1
        sa1_xyz, sa1_points = self.sa1(xyz, normals)  # (B, 512, 3), (B, 512, 320)
        # SA2
        sa2_xyz, sa2_points = self.sa2(sa1_xyz, sa1_points)  # (B, 128, 3), (B, 128, 640)

        # SA3: global aggregation
        # Concat xyz with features
        sa3_input = torch.cat([sa2_xyz, sa2_points], dim=-1)  # (B, 128, 643)
        sa3_input = sa3_input.transpose(1, 2)  # (B, 643, 128)
        sa3_out = self.sa3_mlp(sa3_input)  # (B, 1024, 128)
        global_feat = torch.max(sa3_out, dim=-1)[0]  # (B, 1024)

        # FC head
        x = self.drop1(F.relu(self.bn1(self.fc1(global_feat))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        logits = self.fc3(x)

        return logits, global_feat
