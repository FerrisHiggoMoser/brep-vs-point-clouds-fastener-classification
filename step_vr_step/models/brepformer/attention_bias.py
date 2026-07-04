"""Topology-aware attention bias for BRepFormer.

Combines edge features and topology distance matrices into an additive
bias for the self-attention mechanism, so the transformer is aware of
the B-Rep topology.
"""

import torch
import torch.nn as nn
import math


class TopologyAttentionBias(nn.Module):
    """Compute attention bias from topology distances and edge features.

    The bias is added to the attention logits before softmax, encoding
    topological proximity between faces.
    """

    def __init__(self, dim: int = 256, num_distance_types: int = 4, num_buckets: int = 32):
        super().__init__()
        self.num_distance_types = num_distance_types
        self.num_buckets = num_buckets

        # Learnable embedding for each distance bucket per distance type
        self.distance_embeddings = nn.ModuleList([
            nn.Embedding(num_buckets, 1) for _ in range(num_distance_types)
        ])

        # Project edge features to a scalar bias via a face-face interaction
        self.edge_proj = nn.Linear(dim, 1)

    def forward(
        self,
        topo_distances: dict[str, torch.Tensor],
        edge_features: torch.Tensor = None,
        face_adj_edge_map: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            topo_distances: dict with 4 distance matrices, each (B, Nf, Nf).
                Keys: "face_shortest", "face_centroid", "face_angular", "edge_path"
            edge_features: (B, Ne, dim) per-edge feature vectors (optional).
            face_adj_edge_map: (B, Nf, Nf) -> edge index mapping (optional).

        Returns:
            bias: (B, Nf, Nf) attention bias to add to QK^T / sqrt(d).
        """
        # Get shape from any distance matrix
        dist_keys = ["face_shortest", "face_centroid", "face_angular", "edge_path"]
        sample_key = next(k for k in dist_keys if k in topo_distances)
        B, Nf, _ = topo_distances[sample_key].shape

        bias = torch.zeros(B, Nf, Nf, device=topo_distances[sample_key].device)

        # Add bucketized distance embeddings
        for i, key in enumerate(dist_keys):
            if key not in topo_distances:
                continue
            dist = topo_distances[key]  # (B, Nf, Nf)
            buckets = self._bucketize(dist)  # (B, Nf, Nf) long
            emb = self.distance_embeddings[i](buckets)  # (B, Nf, Nf, 1)
            bias = bias + emb.squeeze(-1)

        return bias

    def _bucketize(self, distances: torch.Tensor) -> torch.Tensor:
        """Convert continuous distances to bucket indices.

        Uses log-spaced buckets for better resolution at small distances.
        """
        # Clamp infinite distances
        max_dist = 100.0
        distances = distances.clamp(0, max_dist)

        # Log-space bucketization
        log_dist = torch.log1p(distances)
        max_log = math.log1p(max_dist)
        bucket_size = max_log / self.num_buckets
        buckets = (log_dist / bucket_size).long()
        buckets = buckets.clamp(0, self.num_buckets - 1)

        return buckets
