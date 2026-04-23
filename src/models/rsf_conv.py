from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class RSFConv2d(nn.Module):
    """
    A compact RSF-style convolutional layer with Fourier-parameterized kernels.

    This implementation is designed for reproducible experimentation:
    - parameterizes kernels in the Fourier domain,
    - maps to spatial kernels by iRFFT,
    - averages transformed kernels across rotation-scale groups.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        bias: bool = False,
        rotations: Sequence[float] | None = None,
        scales: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2 if padding is None else padding
        self.rotations = list(rotations) if rotations is not None else [i * math.pi / 4.0 for i in range(8)]
        self.scales = list(scales) if scales is not None else [1.0, 1.25, 1.25**2, 1.25**3]

        # Real-valued tensor that stores both real/imag components.
        freq_w = torch.randn(out_channels, in_channels, kernel_size, kernel_size // 2 + 1, 2) * 0.02
        self.freq_weight = nn.Parameter(freq_w)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def _spatial_kernel(self) -> torch.Tensor:
        complex_w = torch.view_as_complex(self.freq_weight.contiguous())
        return torch.fft.irfftn(complex_w, s=(self.kernel_size, self.kernel_size), dim=(-2, -1)).real

    def _transform_kernel(self, kernel: torch.Tensor, theta: float, scale: float) -> torch.Tensor:
        # kernel: [out, in, k, k] -> reshape to [out*in, 1, k, k] for grid sampling
        k = kernel.shape[-1]
        flat = kernel.reshape(-1, 1, k, k)
        cos_t = math.cos(theta) * scale
        sin_t = math.sin(theta) * scale
        affine = torch.tensor(
            [[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0]],
            dtype=kernel.dtype,
            device=kernel.device,
        )
        affine = affine.unsqueeze(0).repeat(flat.shape[0], 1, 1)
        grid = F.affine_grid(affine, size=flat.shape, align_corners=False)
        transformed = F.grid_sample(flat, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        return transformed.reshape(self.out_channels, self.in_channels, k, k)

    def _equivariant_kernel(self) -> torch.Tensor:
        base = self._spatial_kernel()
        kernels = []
        for th in self.rotations:
            for sc in self.scales:
                kernels.append(self._transform_kernel(base, th, sc))
        stacked = torch.stack(kernels, dim=0)
        return stacked.mean(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self._equivariant_kernel()
        return F.conv2d(x, k, bias=self.bias, stride=self.stride, padding=self.padding)


class RSFConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            RSFConv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            RSFConv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
