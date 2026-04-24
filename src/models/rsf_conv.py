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
        kernel_size: int = 6,
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
        if padding is None:
            if kernel_size % 2 == 1:
                self.padding: int | tuple[int, int, int, int] = kernel_size // 2
            else:
                lo = kernel_size // 2 - 1
                hi = kernel_size // 2
                self.padding = (lo, hi, lo, hi)
        else:
            self.padding = padding
        self.rotations = list(rotations) if rotations is not None else [i * math.pi / 4.0 for i in range(8)]
        self.scales = list(scales) if scales is not None else [1.25**i for i in range(4)]

        freq_w = torch.randn(out_channels, in_channels, kernel_size, kernel_size // 2 + 1, 2)
        freq_w.mul_(1.0 / (kernel_size * math.sqrt(max(in_channels, 1))))
        self.freq_weight = nn.Parameter(freq_w)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        affine = self._build_affine_bank()
        self.register_buffer("_affine_bank", affine, persistent=False)

    def _build_affine_bank(self) -> torch.Tensor:
        rows = []
        for th in self.rotations:
            for sc in self.scales:
                cos_t = math.cos(th) * sc
                sin_t = math.sin(th) * sc
                rows.append(torch.tensor([[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0]]))
        return torch.stack(rows, dim=0)

    def _spatial_kernel(self) -> torch.Tensor:
        complex_w = torch.view_as_complex(self.freq_weight.contiguous())
        return torch.fft.irfftn(complex_w, s=(self.kernel_size, self.kernel_size), dim=(-2, -1)).real

    def _equivariant_kernel(self) -> torch.Tensor:
        base = self._spatial_kernel()
        k = self.kernel_size
        g = self._affine_bank.shape[0]
        outc, inc = self.out_channels, self.in_channels
        flat = base.reshape(1, outc * inc, k, k).expand(g, outc * inc, k, k).contiguous()
        affine = self._affine_bank.to(dtype=base.dtype, device=base.device)
        grid = F.affine_grid(affine, size=(g, outc * inc, k, k), align_corners=False)
        transformed = F.grid_sample(
            flat.reshape(g * outc * inc, 1, k, k),
            grid.repeat_interleave(outc * inc, dim=0),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        transformed = transformed.reshape(g, outc, inc, k, k)
        return transformed.mean(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self._equivariant_kernel()
        if isinstance(self.padding, tuple):
            x = F.pad(x, self.padding)
            return F.conv2d(x, k, bias=self.bias, stride=self.stride, padding=0)
        return F.conv2d(x, k, bias=self.bias, stride=self.stride, padding=self.padding)


class RSFConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 6):
        super().__init__()
        self.block = nn.Sequential(
            RSFConv2d(in_ch, out_ch, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            RSFConv2d(out_ch, out_ch, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
