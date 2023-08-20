from collections.abc import Sequence

import torch
from timm.layers import LayerNorm2d
from torch import nn


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


__all__ = ["LayerNormNoBias", "LayerNormNoBias2d", "StarReLU"]
