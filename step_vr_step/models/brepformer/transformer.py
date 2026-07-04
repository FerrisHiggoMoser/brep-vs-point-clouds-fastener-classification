"""Transformer components for BRepFormer.

Implements:
  - RMSNorm (Root Mean Square Layer Normalization)
  - SwiGLU (Swish-Gated Linear Unit FFN)
  - GQAttentionWithBias (Grouped Query Attention with additive topology bias)
  - BRepFormerLayer (Pre-norm transformer block)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    x_norm = x / RMS(x) * gain, where RMS(x) = sqrt(mean(x^2) + eps).
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.gain


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network.

    out = W3 * (Swish(W1 * x) ⊙ (W2 * x))
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class GQAttentionWithBias(nn.Module):
    """Grouped Query Attention with additive attention bias.

    Uses fewer KV heads than Q heads (grouped query attention) for
    efficiency, and adds a topology-aware bias to the attention logits
    before softmax.
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        num_kv_groups: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert num_heads % num_kv_groups == 0
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.num_kv_heads = num_heads // num_kv_groups
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, dim) input tokens.
            attention_bias: (B, N, N) additive bias for attention logits.
            mask: (B, N) boolean mask (True = keep, False = pad).

        Returns:
            out: (B, N, dim)
        """
        B, N, _ = x.shape

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        k = self.k_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)  # (B, Hkv, N, d)
        v = self.v_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Expand KV heads to match Q heads via repetition
        if self.num_kv_heads != self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, N, N)

        # Add topology bias (broadcast over heads)
        if attention_bias is not None:
            attn = attn + attention_bias.unsqueeze(1)  # (B, 1, N, N)

        # Apply padding mask
        if mask is not None:
            # mask: (B, N) -> (B, 1, 1, N)
            pad_mask = ~mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(pad_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, H, N, d)
        out = out.transpose(1, 2).reshape(B, N, -1)  # (B, N, dim)
        out = self.out_proj(out)

        return out


class BRepFormerLayer(nn.Module):
    """Single BRepFormer transformer layer.

    Pre-norm → GQA+Bias → Residual → Pre-norm → SwiGLU → Residual
    """

    def __init__(self, dim: int = 256, ffn_dim: int = 1024, num_heads: int = 8,
                 num_kv_groups: int = 2, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = GQAttentionWithBias(dim, num_heads, num_kv_groups, dropout)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attention_bias, mask)
        x = x + self.ffn(self.norm2(x))
        return x
