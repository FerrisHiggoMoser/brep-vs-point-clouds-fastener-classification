"""1D CNN encoder for edge curves.

Input:  (batch, Ne, 10, 12)
Output: (batch, Ne, 256) — per-edge feature vectors.
"""

import torch
import torch.nn as nn


class EdgeEncoder(nn.Module):
    """1D CNN that encodes edge curve samples into fixed-size feature vectors."""

    def __init__(self, in_channels: int = 12, out_dim: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),  # -> (batch*Ne, out_dim, 1)
        )
        self.out_dim = out_dim

    def forward(self, edge_curves: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_curves: (B, Ne, 10, 12) edge curve samples.

        Returns:
            edge_features: (B, Ne, 256) per-edge embeddings.
        """
        B, Ne, L, C = edge_curves.shape
        x = edge_curves.reshape(B * Ne, L, C)
        x = x.permute(0, 2, 1)  # (B*Ne, C, L)

        x = self.cnn(x)  # (B*Ne, out_dim, 1)
        x = x.squeeze(-1)  # (B*Ne, out_dim)
        x = x.reshape(B, Ne, self.out_dim)

        return x
