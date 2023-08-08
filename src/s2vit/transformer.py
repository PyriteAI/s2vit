from collections import OrderedDict
from collections.abc import Callable, Sequence

import torch
from einops.layers.torch import Reduce
from torch import nn
from torchvision.ops import DropBlock2d

from .nn import PEG, BlockMHSA, LayerNormNoBias, LayerNormNoBias2d, PatchEmbedding, Shift2d, StarReLU


class S2ViTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_head: int = 32,
        block_size: int = 8,
        dim_ff: int | None = None,
        norm_layer: Callable[[int], nn.Module] = LayerNormNoBias2d,
        attn_drop_rate: float = 0.0,
        ff_drop_rate: float = 0.0,
        drop_block_rate: float = 0.0,
        drop_block_size: int = 7,
        bias: bool = False,
    ):
        super().__init__()

        if dim_ff is None:
            dim_ff = dim * 4
        heads = dim // dim_head

        self.attn = nn.Sequential(
            norm_layer(dim),
            Shift2d(),
            BlockMHSA(dim, heads=heads, dim_head=dim_head, block_size=block_size, drop_rate=attn_drop_rate, bias=bias),
            DropBlock2d(drop_block_rate, drop_block_size),
        )
        self.ff = nn.Sequential(
            norm_layer(dim),
            nn.Conv2d(dim, dim_ff, kernel_size=1, bias=bias),
            StarReLU(),
            nn.Dropout(ff_drop_rate),
            nn.Conv2d(dim_ff, dim, kernel_size=1, bias=bias),
            DropBlock2d(drop_block_rate, drop_block_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.ff(x)
        return x


class S2ViTStage(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        depth: int,
        use_peg: bool = True,
        patch_size: int | tuple[int, int] | None = None,
        dim_head: int = 32,
        block_size: int = 8,
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
            if i == 1 and use_peg:
                blocks.append(PEG(dim_out))
            block = S2ViTBlock(
                dim_out,
                dim_head=dim_head,
                block_size=block_size,
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
        block_sizes: Sequence[int] = (8, 8, 8, 8),
        ff_expansions: Sequence[int] = (4, 4, 4, 4),
        norm_layer: Callable[[int], nn.Module] = LayerNormNoBias2d,
        input_norm: Callable[[int], nn.Module] = LayerNormNoBias2d,
        output_norm: Callable[[int], nn.Module] = LayerNormNoBias,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_block_rate: float = 0.0,
        drop_block_size: int = 7,
        bias: bool = False,
    ):
        super().__init__()

        stages: list[S2ViTStage] = []
        for dim_in, dim_out, depth, patch_size, block_size, ff_expansion in zip(
            (in_channels, *dims[:-1]), dims, depths, patch_sizes, block_sizes, ff_expansions, strict=True
        ):
            stage = S2ViTStage(
                dim_in,
                dim_out,
                depth,
                use_peg=use_peg,
                patch_size=patch_size,
                dim_head=dim_head,
                block_size=block_size,
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


__all__ = ["S2ViT", "S2ViTBlock", "S2ViTStage"]
