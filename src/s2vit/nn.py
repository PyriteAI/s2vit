import math
from collections.abc import Callable, Sequence

import torch
from einops import pack, rearrange, unpack
from timm.layers import LayerNorm2d
from torch import nn
from torch.nn import functional as F

from .utils import to_pair


class StarReLU(nn.Module):
    def __init__(
        self,
        scale_value: float = 0.8944,
        bias_value: float = -0.4472,
        scale_learnable: bool = False,
        bias_learnable: bool = False,
        inplace: bool = False,
    ):
        super().__init__()

        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1), requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1), requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


class LayerNormNoBias(nn.LayerNorm):
    def __init__(self, normalized_shape: int | Sequence[int], eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__(normalized_shape, eps, elementwise_affine)  # type: ignore
        self.bias = None


class LayerNormNoBias2d(LayerNorm2d):
    def __init__(self, num_channels: int, eps: float = 1e-6, affine: bool = True):
        super().__init__(num_channels, eps=eps, affine=affine)

        self.bias = None


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


class BlockMHSA(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        block_size: int = 8,
        drop_rate: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.block_size = block_size
        self.drop_rate = drop_rate

        self.to_qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.to_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b c (h p1) (w p2) -> b h w (p1 p2) c", p1=self.block_size, p2=self.block_size)
        x, ps = pack([x], "* n d")

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop_rate if self.training else 0.0)
        x = rearrange(x, "b h n d -> b n (h d)")
        (x,) = unpack(x, ps, "* n d")
        x = rearrange(x, "b h w (p1 p2) c -> b c (h p1) (w p2)", p1=self.block_size, p2=self.block_size)
        return self.to_out(x)


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


__all__ = ["BlockMHSA", "LayerNormNoBias", "LayerNormNoBias2d", "PatchEmbedding", "PEG", "Shift2d", "StarReLU"]
