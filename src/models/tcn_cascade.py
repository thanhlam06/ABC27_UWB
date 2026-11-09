"""
1D Temporal Convolutional Network (`FrameWise1DTCN`) & Fast Proposal (`MobileNetV3ProposalNet`)
9th ABC Challenge — UWB Fall Detection Track

Provides the lightweight 1D-TCN (dilations 1, 2, 4, 8, 16 -> 377-frame / 7.54s receptive field)
with Spatial Band Attention and the Stage-1 MobileNetV3-Small proposal network used in our
Bi-Level Hybrid Cascade experiments.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialBandAttention(nn.Module):
    """Squeeze-and-Excitation channel attention across 12 spatial range bands."""

    def __init__(self, channels: int = 12, reduction: int = 3) -> None:
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = x.mean(dim=-1)
        w = torch.sigmoid(self.fc2(F.gelu(self.fc1(w)))).unsqueeze(-1)
        return x * w


class DilatedResidualBlock1D(nn.Module):
    """Non-causal dilated 1D residual block preserving exact temporal length T."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        dilation: int = 1,
        dropout: float = 0.12,
    ) -> None:
        super().__init__()
        pad = ((kernel_size - 1) * dilation) // 2
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, dilation=dilation, padding=pad, bias=False
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, dilation=dilation, padding=pad, bias=False
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return F.gelu(out + x)


class FrameWise1DTCN(nn.Module):
    """
    12-Channel Spatial-Attention Dilated 1D-TCN for 50 Hz Frame-Wise Localization.
    Receptive field: 5 + 2 * (7 - 1) * (1 + 2 + 4 + 8 + 16) = 377 frames (7.54 seconds).
    """

    def __init__(
        self,
        in_channels: int = 12,
        hidden_channels: int = 32,
        kernel_size: int = 7,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
        dropout: float = 0.12,
        prior_prob: float = 0.012,
    ) -> None:
        super().__init__()
        self.spatial_attn = SpatialBandAttention(in_channels)
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                DilatedResidualBlock1D(
                    channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout,
                )
                for d in dilations
            ]
        )
        self.head = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        bias_init = float(torch.logit(torch.tensor(prior_prob)).item())
        nn.init.constant_(self.head.bias, bias_init)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(self.spatial_attn(x))
        for block in self.blocks:
            h = block(h)
        return self.head(h).squeeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x))


class MobileNetV3ProposalNet(nn.Module):
    """Lightweight 2D MobileNetV3-style fast proposal network scanning 2.0s RTM windows."""

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Hardswish(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Hardswish(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x)).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(x))
