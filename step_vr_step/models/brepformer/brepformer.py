"""BRepFormer: topology-aware transformer for B-Rep classification.

Architecture:
  1. Face encoder (2D CNN) → face tokens [Nf x 256]
  2. Prepend Virtual Face Token (learnable [1 x 256])
  3. Edge encoder (1D CNN) → edge features
  4. Build attention bias from edge features + topology distances
  5. 8× BRepFormerLayer (GQA + SwiGLU)
  6. Recognition head: VFT output → MLP → num_classes
"""

import torch
import torch.nn as nn

from .face_encoder import FaceEncoder
from .edge_encoder import EdgeEncoder
from .attention_bias import TopologyAttentionBias
from .transformer import BRepFormerLayer


class BRepFormer(nn.Module):
    """BRepFormer — supports both per-face segmentation (as in the paper) and per-part
    classification (for our binary fastener task).

    Head mode:
      - "segmentation": outputs (B, Nf, num_classes) — paper-faithful, one label per face.
        Concatenates local face tokens with broadcast VFT (global) → 512 → num_classes.
      - "classification": outputs (B, num_classes) — VFT only → num_classes. For part-level tasks.
    """

    def __init__(
        self,
        num_classes: int = 10,
        dim: int = 256,
        num_layers: int = 8,
        num_heads: int = 8,
        num_kv_groups: int = 2,
        ffn_dim: int = 1024,
        dropout: float = 0.0,
        head_mode: str = "classification",
    ):
        super().__init__()
        assert head_mode in ("segmentation", "classification")
        self.dim = dim
        self.head_mode = head_mode

        # Encoders
        self.face_encoder = FaceEncoder(in_channels=7, out_dim=dim)
        self.edge_encoder = EdgeEncoder(in_channels=12, out_dim=dim)

        # Virtual Face Token (learnable CLS-like token)
        self.vft = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        # Topology attention bias
        self.topo_bias = TopologyAttentionBias(dim=dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            BRepFormerLayer(dim, ffn_dim, num_heads, num_kv_groups, dropout)
            for _ in range(num_layers)
        ])

        # Output heads
        if head_mode == "segmentation":
            # Per-face head: concat (local_face_token, broadcast_global_VFT) → MLP
            self.head = nn.Sequential(
                nn.LayerNorm(dim * 2),
                nn.Linear(dim * 2, dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim, num_classes),
            )
        else:
            # Classification head (from VFT only)
            self.head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim, num_classes),
            )

    def forward(
        self,
        face_grids: torch.Tensor,
        edge_curves: torch.Tensor,
        topo_distances: dict[str, torch.Tensor],
        mask: torch.Tensor = None,
        edge_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            face_grids: (B, Nf, 10, 10, 7) face UV-grid features.
            edge_curves: (B, Ne, 10, 12) edge curve samples.
            topo_distances: dict of 4 distance matrices, each (B, Nf, Nf).
            mask: (B, Nf) boolean mask (True = real face, False = padding).
            edge_mask: (B, Ne) boolean mask for edges (True = real).

        Returns:
            logits: (B, num_classes)
        """
        B = face_grids.shape[0]
        Nf = face_grids.shape[1]

        # Encode faces and edges
        face_tokens = self.face_encoder(face_grids)    # (B, Nf, dim)
        edge_features = self.edge_encoder(edge_curves) # (B, Ne, dim)

        # Global edge context: masked mean-pool of edge features.
        # Injected into the Virtual Face Token so edges actually influence classification.
        if edge_mask is not None:
            em = edge_mask.float().unsqueeze(-1)                    # (B, Ne, 1)
            denom = em.sum(dim=1).clamp(min=1.0)                    # (B, 1)
            edge_context = (edge_features * em).sum(dim=1) / denom  # (B, dim)
        else:
            edge_context = edge_features.mean(dim=1)                # (B, dim)

        # Prepend Virtual Face Token with edge-context injection
        vft = self.vft.expand(B, -1, -1) + edge_context.unsqueeze(1)  # (B, 1, dim)
        tokens = torch.cat([vft, face_tokens], dim=1)                 # (B, 1+Nf, dim)

        # Build attention bias (expand to include VFT position)
        bias = self.topo_bias(topo_distances)  # (B, Nf, Nf)
        # Pad bias for VFT: VFT has zero bias to all faces
        vft_row = torch.zeros(B, 1, Nf, device=bias.device)
        vft_col = torch.zeros(B, Nf + 1, 1, device=bias.device)
        bias = torch.cat([vft_row, bias], dim=1)  # (B, 1+Nf, Nf)
        bias = torch.cat([vft_col, bias], dim=2)  # (B, 1+Nf, 1+Nf)

        # Expand mask for VFT (VFT is always valid)
        if mask is not None:
            vft_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([vft_mask, mask], dim=1)  # (B, 1+Nf)

        # Transformer
        for layer in self.layers:
            tokens = layer(tokens, attention_bias=bias, mask=mask)

        vft_output = tokens[:, 0, :]    # (B, dim)
        face_tokens = tokens[:, 1:, :]  # (B, Nf, dim)

        if self.head_mode == "segmentation":
            # Per-face: concat local + broadcast global, then classify each face
            global_feat = vft_output.unsqueeze(1).expand(-1, Nf, -1)  # (B, Nf, dim)
            combined = torch.cat([face_tokens, global_feat], dim=-1)  # (B, Nf, 2*dim)
            logits = self.head(combined)  # (B, Nf, num_classes)
        else:
            logits = self.head(vft_output)  # (B, num_classes)

        return logits
