import math
from collections.abc import Callable

import torch
from einops import pack, rearrange, unpack
from torch import nn
from torch.nn import functional as F

from .nn import LayerNormNoBias2d, StarReLU
from .utils import to_pair


class GWAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        window_size: int = 8,
        drop_rate: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        dim_inner = dim_head * heads

        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.window_size = window_size
        self.drop_rate = drop_rate
        self.dim_inner = dim_inner

        # Credit to @lucidrains for this idea.
        self.to_qkv = nn.Linear(dim, dim_inner * 2, bias=bias)
        self.attn_gate = nn.Linear(dim_inner, heads)  # bias always set to True
        self.to_out = nn.Linear(dim_inner, dim, bias=bias)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.constant_(self.attn_gate.bias, 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b c (h p1) (w p2) -> b h w (p1 p2) c", p1=self.window_size, p2=self.window_size)
        x, ps = pack([x], "* n d")

        qkv = self.to_qkv(x).chunk(2, dim=-1)
        q, kv = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        x_attn = F.scaled_dot_product_attention(q, kv, kv, dropout_p=self.drop_rate if self.training else 0.0)
        x_gate = self.attn_gate(x)  # shape = (b, n, h)
        x_gate = rearrange(x_gate, "b n h -> b h n ()")
        x = x_attn * torch.sigmoid(x_gate)
        x = rearrange(x, "b h n d -> b n (h d)")

        x = self.to_out(x)

        (x,) = unpack(x, ps, "* n d")
        x = rearrange(x, "b h w (p1 p2) c -> b c (h p1) (w p2)", p1=self.window_size, p2=self.window_size)
        return x


class FusedGWAttentionFF(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_head: int = 32,
        window_size: int = 8,
        dim_ff: int | None = None,
        attn_drop_rate: float = 0.0,
        ff_drop_rate: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        if dim_ff is None:
            dim_ff = dim * 4
        heads = dim // dim_head

        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.window_size = window_size
        self.attn_drop_rate = attn_drop_rate
        self.ff_drop_rate = ff_drop_rate
        self.bias = bias

        self.fused_dims = (dim, dim, dim_ff)
        self.fused_attn_ff_proj = nn.Linear(dim, sum(self.fused_dims), bias=bias)
        self.attn_gate = nn.Linear(dim, heads)  # bias always set to True
        self.attn_out = nn.Linear(dim, dim, bias=bias)
        self.ff_out = nn.Sequential(
            StarReLU(),
            nn.Dropout(ff_drop_rate),
            nn.Linear(dim_ff, dim, bias=bias),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.constant_(self.attn_gate.bias, 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b c (h p1) (w p2) -> b h w (p1 p2) c", p1=self.window_size, p2=self.window_size)
        x, ps = pack([x], "* n d")

        q, kv, x_ff = self.fused_attn_ff_proj(x).split(self.fused_dims, dim=-1)

        q, kv = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, kv))
        x_attn = F.scaled_dot_product_attention(q, kv, kv, dropout_p=self.attn_drop_rate if self.training else 0.0)
        x_gate = self.attn_gate(x)  # shape = (b, n, h)
        x_gate = rearrange(x_gate, "b n h -> b h n ()")
        x_attn = x_attn * torch.sigmoid(x_gate)
        x_attn = rearrange(x_attn, "b h n d -> b n (h d)")

        x = self.attn_out(x_attn) + self.ff_out(x_ff)

        (x,) = unpack(x, ps, "* n d")
        x = rearrange(x, "b h w (p1 p2) c -> b c (h p1) (w p2)", p1=self.window_size, p2=self.window_size)
        return x


class ParallelGWAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 32,
        window_size: int = 8,
        drop_rate: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.window_size = window_size
        self.drop_rate = drop_rate
        self.dim_inner = dim_head * heads
        # Two parallel paths plus shared kv attention -> 2 * 2 = 4
        self.to_parallel_qkv = nn.Linear(dim, self.dim_inner * 4, bias=bias)
        # Gated attention - fused to support gating both paths in a single layer
        self.parallel_attn_gate = nn.Linear(dim, heads * 2)
        # Parallel output layer - thanks to grouped convolution, we can do this in a single layer
        self.to_parallel_out = nn.Conv1d(self.dim_inner * 2, dim * 2, kernel_size=1, groups=2, bias=bias)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.constant_(self.parallel_attn_gate.bias, 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b c (h p1) (w p2) -> b h w (p1 p2) c", p1=self.window_size, p2=self.window_size)
        x, ps = pack([x], "* n d")
        # Split across q and kv
        par_qkv = self.to_parallel_qkv(x).chunk(2, dim=-1)
        # We stack the parallel q and kv tensors to allow for a single attention call
        par_q, par_kv = map(lambda t: rearrange(t, "b n (p h d) -> b p h n d", p=2, h=self.heads), par_qkv)
        par_x_attn = F.scaled_dot_product_attention(
            par_q, par_kv, par_kv, dropout_p=self.drop_rate if self.training else 0.0
        )
        # Appply gated attention
        par_x_gate = self.parallel_attn_gate(x)  # shape = (b, n, h * 2)
        par_x_gate = rearrange(par_x_gate, "b n (p h) -> b p h n ()", p=2, h=self.heads)
        par_x_attn = par_x_attn * torch.sigmoid(par_x_gate)
        # Generate parallel output
        par_x_attn = rearrange(par_x_attn, "b p h n d -> b (p h d) n")
        par_x = self.to_parallel_out(par_x_attn)
        # Split back into two parallel paths and sum
        par_x1, par_x2 = par_x.chunk(2, dim=1)
        x = par_x1 + par_x2
        # Rearrange back to original shape
        x = rearrange(x, "b d n -> b n d")
        (x,) = unpack(x, ps, "* n d")
        x = rearrange(x, "b h w (p1 p2) c -> b c (h p1) (w p2)", p1=self.window_size, p2=self.window_size)
        return x


class SequentialFF(nn.Module):
    def __init__(self, dim: int, dim_inner: int | None = None, drop_rate: float = 0.0, bias: bool = False):
        super().__init__()

        if dim_inner is None:
            dim_inner = dim * 4

        self.layers = nn.Sequential(
            nn.Conv2d(dim, dim_inner, kernel_size=1, bias=bias),
            StarReLU(),
            nn.Dropout(drop_rate),
            nn.Conv2d(dim_inner, dim, kernel_size=1, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ParallelFF(nn.Module):
    def __init__(self, dim: int, dim_inner: int | None = None, drop_rate: float = 0.0, bias: bool = False):
        super().__init__()

        if dim_inner is None:
            dim_inner = dim * 4

        self.parallel_layers = nn.Sequential(
            nn.Conv2d(dim * 2, dim_inner * 2, kernel_size=1, groups=2, bias=bias),
            StarReLU(),
            nn.Dropout(drop_rate),
            nn.Conv2d(dim_inner * 2, dim * 2, kernel_size=1, groups=2, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat(1, 2, 1, 1)
        par_x = self.parallel_layers(x)
        par_x1, par_x2 = par_x.chunk(2, dim=1)
        x = par_x1 + par_x2
        return x


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        patch_size: int | tuple[int, int],
        norm_layer: Callable[[int], nn.Module] = LayerNormNoBias2d,
        bias: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.patch_size = to_pair(patch_size)

        dim_in = math.prod(self.patch_size) * dim
        self.proj = nn.Sequential(
            norm_layer(dim_in),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias),
            norm_layer(dim_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p1, p2 = self.patch_size
        x = rearrange(x, " b c (h p1) (w p2) -> b (c p1 p2) h w", p1=p1, p2=p2)
        return self.proj(x)


class PEG(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()

        self.proj = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=dim)

    def forward(self, x):
        return x + self.proj(x)


class Shift2d(nn.Module):
    def __init__(self, amount: int = 1):
        super().__init__()

        self.amount = amount

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        x_shifted = x.clone()
        x_shifted[:, : c // 4, self.amount :, :] = x[:, : c // 4, : -self.amount, :]
        x_shifted[:, c // 4 : c // 2, : -self.amount, :] = x[:, c // 4 : c // 2, self.amount :, :]
        x_shifted[:, c // 2 : (3 * c) // 4, :, self.amount :] = x[:, c // 2 : (3 * c) // 4, :, : -self.amount]
        x_shifted[:, (3 * c) // 4 :, :, : -self.amount] = x[:, (3 * c) // 4 :, :, self.amount :]
        return x_shifted


__all__ = [
    "FusedGWAttentionFF",
    "GWAttention",
    "PEG",
    "PatchEmbedding",
    "ParallelFF",
    "ParallelGWAttention",
    "SequentialFF",
    "Shift2d",
]
