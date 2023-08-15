import warnings
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any

import torch
from einops import pack, rearrange, unpack
from einops.layers.torch import Reduce
from torch import nn
from torch.nn import functional as F
from torchvision.ops import DropBlock2d

from .nn import PEG, LayerNormNoBias, LayerNormNoBias2d, PatchEmbedding, Shift2d, StarReLU


class ParallelGatedWindowedAttention(nn.Module):
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


class S2ViTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_head: int = 32,
        window_size: int = 8,
        dim_ff: int | None = None,
        norm_layer: Callable[[int], nn.Module] = LayerNormNoBias2d,
        attn_drop_rate: float = 0.0,
        ff_drop_rate: float = 0.0,
        drop_block_rate: float = 0.0,
        drop_block_size: int = 7,
        bias: bool = False,
    ):
        super().__init__()

        self.layers = nn.Sequential(
            Shift2d(),
            ParallelGatedWindowedAttention(
                dim,
                dim_head=dim_head,
                window_size=window_size,
                dim_ff=dim_ff,
                attn_drop_rate=attn_drop_rate,
                ff_drop_rate=ff_drop_rate,
                bias=bias,
            ),
            norm_layer(dim),
            DropBlock2d(p=drop_block_rate, block_size=drop_block_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class S2ViTStage(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        depth: int,
        use_peg: bool = True,
        patch_size: int | tuple[int, int] | None = None,
        dim_head: int = 32,
        window_size: int = 8,
        dim_ff: int | None = None,
        norm_layer: Callable[[int], nn.Module] = LayerNormNoBias2d,
        input_norm: Callable[[int], nn.Module] = LayerNormNoBias2d,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_block_rate: float = 0.0,
        drop_block_size: int = 7,
        bias: bool = False,
    ):
        super().__init__()

        if patch_size is not None:
            patch_embedding = PatchEmbedding(dim_in, dim_out, patch_size=patch_size, norm_layer=input_norm, bias=bias)
        else:
            patch_embedding = nn.Identity()
        blocks: list[S2ViTBlock | PEG] = []
        for i in range(depth):
            if i == 1 and use_peg and patch_size is not None:
                blocks.append(PEG(dim_out))
            block = S2ViTBlock(
                dim_out,
                dim_head=dim_head,
                window_size=window_size,
                dim_ff=dim_ff,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop_rate,
                ff_drop_rate=drop_rate,
                drop_block_rate=drop_block_rate,
                drop_block_size=drop_block_size,
                bias=bias,
            )
            blocks.append(block)
        self.patch_embedding = patch_embedding
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(x)
        x = self.blocks(x)
        return x


class S2ViT(nn.Module):
    def __init__(
        self,
        depths: Sequence[int] = (2, 2, 6, 2),
        dims: Sequence[int] = (64, 128, 160, 320),
        patch_sizes: Sequence[int | tuple[int, int]] = (4, 2, 2, 2),
        in_channels: int = 3,
        global_pool: bool = False,
        num_classes: int | None = None,
        use_peg: bool = True,
        dim_head: int = 32,
        window_sizes: Sequence[int] = (8, 8, 8, 8),
        ff_expansions: Sequence[int] = (4, 4, 4, 4),
        norm_layer: Callable[[int], nn.Module] = LayerNormNoBias2d,
        input_norm: Callable[[int], nn.Module] = LayerNormNoBias2d,
        output_norm: Callable[[int], nn.Module] = LayerNormNoBias,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_block_rate: float = 0.0,
        drop_block_size: int = 7,
        bias: bool = False,
        **kwargs: Any,
    ):
        super().__init__()

        if "block_size" in kwargs:
            warnings.warn(
                "block_size is deprecated, use window_sizes instead", category=DeprecationWarning, stacklevel=1
            )
            window_sizes = [kwargs.pop("block_size")] * len(depths)
        if "block_sizes" in kwargs:
            warnings.warn(
                "block_sizes is deprecated, use window_sizes instead", category=DeprecationWarning, stacklevel=1
            )
            window_sizes = kwargs.pop("block_sizes")

        stages: list[S2ViTStage] = []
        for dim_in, dim_out, depth, patch_size, window_size, ff_expansion in zip(
            (in_channels, *dims[:-1]), dims, depths, patch_sizes, window_sizes, ff_expansions, strict=True
        ):
            stage = S2ViTStage(
                dim_in,
                dim_out,
                depth,
                use_peg=use_peg,
                patch_size=patch_size,
                dim_head=dim_head,
                window_size=window_size,
                dim_ff=dim_out * ff_expansion,
                norm_layer=norm_layer,
                input_norm=input_norm,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_block_rate=drop_block_rate,
                drop_block_size=drop_block_size,
                bias=bias,
            )
            stages.append(stage)
        self.encoder = nn.Sequential(*stages)
        self.head = nn.Sequential(
            OrderedDict(
                [
                    ("global_pool", Reduce("b c h w -> b c", "mean") if global_pool else nn.Identity()),
                    ("norm", output_norm(dims[-1])),
                    ("drop", nn.Dropout(drop_rate)),
                    ("fc", nn.Linear(dims[-1], num_classes, bias=bias) if num_classes else nn.Identity()),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.head(x)
        return x


__all__ = ["ParallelGatedWindowedAttention", "S2ViT", "S2ViTBlock", "S2ViTStage"]
