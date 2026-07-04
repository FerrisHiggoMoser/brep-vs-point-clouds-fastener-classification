"""2D CNN encoder for face UV-grids.

Input:  (batch, Nf, 10, 10, 7)
Output: (batch, Nf, 256) — per-face feature vectors.
"""

import torch
import torch.nn as nn


class FaceEncoder(nn.Module):
    """2D CNN that encodes face UV-grids into fixed-size feature vectors."""

    def __init__(self, in_channels: int = 7, out_dim: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),  # -> (batch*Nf, out_dim, 1, 1)
        )
        self.out_dim = out_dim

    def forward(self, face_grids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            face_grids: (B, Nf, 10, 10, 7) UV-grid features per face.

        Returns:
            face_features: (B, Nf, 256) per-face embeddings.
        """
        B, Nf, H, W, C = face_grids.shape
        # Reshape to process all faces as a batch
        x = face_grids.reshape(B * Nf, H, W, C)
        x = x.permute(0, 3, 1, 2)  # (B*Nf, C, H, W)

        x = self.cnn(x)  # (B*Nf, out_dim, 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # (B*Nf, out_dim)
        x = x.reshape(B, Nf, self.out_dim)

        return x
