"""PointNet++ utility modules: FPS, ball query, set abstraction, feature propagation.

Adapted from the architecture described in:
  Qi et al., "PointNet++: Deep Hierarchical Feature Learning on Point Sets
  in a Metric Space", NeurIPS 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Farthest Point Sampling.

    Args:
        xyz: (B, N, 3) point positions.
        npoint: number of points to sample.

    Returns:
        centroids: (B, npoint) indices of sampled points.
    """
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)  # (B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)  # (B, N)
        distance = torch.min(distance, dist)
        farthest = torch.argmax(distance, dim=-1)

    return centroids


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points by index.

    Args:
        points: (B, N, C) input points.
        idx: (B, S) or (B, S, K) index tensor.

    Returns:
        new_points: (B, S, C) or (B, S, K, C) gathered points.
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).reshape(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def query_ball_point(
    radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor
) -> torch.Tensor:
    """Ball query.

    Args:
        radius: local region radius.
        nsample: max sample number in local region.
        xyz: (B, N, 3) all points.
        new_xyz: (B, S, 3) query points (centroids).

    Returns:
        group_idx: (B, S, nsample) indices of grouped points.
    """
    device = xyz.device
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape

    # (B, S, N)
    sqrdists = (
        torch.sum((new_xyz.unsqueeze(2) - xyz.unsqueeze(1)) ** 2, dim=-1)
    )

    group_idx = torch.arange(N, dtype=torch.long, device=device).reshape(1, 1, N).repeat(B, S, 1)
    group_idx[sqrdists > radius ** 2] = N  # mark out-of-radius as N

    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]  # take closest nsample

    # Fill remaining slots with the first point in the group
    group_first = group_idx[:, :, 0].unsqueeze(-1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx


def sample_and_group(
    npoint: int, radius: float, nsample: int,
    xyz: torch.Tensor, points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample centroids with FPS, group neighbours with ball query.

    Args:
        npoint: number of centroids.
        radius: ball query radius.
        nsample: neighbours per centroid.
        xyz: (B, N, 3) positions.
        points: (B, N, D) features (or None).

    Returns:
        new_xyz: (B, npoint, 3) centroid positions.
        new_points: (B, npoint, nsample, 3+D) grouped features.
    """
    fps_idx = farthest_point_sample(xyz, npoint)  # (B, npoint)
    new_xyz = index_points(xyz, fps_idx)  # (B, npoint, 3)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)  # (B, npoint, nsample)
    grouped_xyz = index_points(xyz, idx)  # (B, npoint, nsample, 3)
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)  # relative coords

    if points is not None:
        grouped_points = index_points(points, idx)  # (B, npoint, nsample, D)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    """Multi-Scale Grouping (MSG) Set Abstraction module."""

    def __init__(
        self,
        npoint: int,
        radius_list: list[float],
        nsample_list: list[int],
        in_channel: int,
        mlp_list: list[list[int]],
    ):
        super().__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list

        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()

        for i, mlp in enumerate(mlp_list):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3  # +3 for relative xyz
            for out_channel in mlp:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(
        self, xyz: torch.Tensor, points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: (B, N, 3)
            points: (B, N, D) or None

        Returns:
            new_xyz: (B, npoint, 3)
            new_points: (B, npoint, sum_of_mlp_outputs)
        """
        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)

        new_points_list = []
        for i, (radius, nsample) in enumerate(zip(self.radius_list, self.nsample_list)):
            group_idx = query_ball_point(radius, nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.unsqueeze(2)

            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            # (B, npoint, nsample, C) -> (B, C, npoint, nsample)
            grouped_points = grouped_points.permute(0, 3, 1, 2).contiguous()

            for j, (conv, bn) in enumerate(zip(self.conv_blocks[i], self.bn_blocks[i])):
                grouped_points = F.relu(bn(conv(grouped_points)))

            # Max pool over nsample -> (B, C, npoint)
            new_points = torch.max(grouped_points, dim=-1)[0]
            new_points_list.append(new_points)

        new_xyz = new_xyz
        # Concat features from all scales -> (B, sum_C, npoint) -> (B, npoint, sum_C)
        new_points_concat = torch.cat(new_points_list, dim=1).transpose(1, 2).contiguous()

        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    """Feature Propagation module for segmentation decoder."""

    def __init__(self, in_channel: int, mlp: list[int]):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(
        self,
        xyz1: torch.Tensor, xyz2: torch.Tensor,
        points1: torch.Tensor, points2: torch.Tensor,
    ) -> torch.Tensor:
        """Propagate features from xyz2 (fewer points) to xyz1 (more points).

        Args:
            xyz1: (B, N, 3) target positions.
            xyz2: (B, S, 3) source positions.
            points1: (B, N, D1) target features (or None).
            points2: (B, S, D2) source features.

        Returns:
            new_points: (B, N, mlp[-1])
        """
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated = points2.repeat(1, N, 1)
        else:
            dists = torch.sum((xyz1.unsqueeze(2) - xyz2.unsqueeze(1)) ** 2, dim=-1)  # (B, N, S)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # 3 nearest

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=-1, keepdim=True)
            weight = dist_recip / norm  # (B, N, 3)

            interpolated = torch.sum(
                index_points(points2, idx) * weight.unsqueeze(-1), dim=2
            )  # (B, N, D2)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated], dim=-1)
        else:
            new_points = interpolated

        # (B, N, C) -> (B, C, N)
        new_points = new_points.transpose(1, 2)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        # (B, C, N) -> (B, N, C)
        return new_points.transpose(1, 2)
