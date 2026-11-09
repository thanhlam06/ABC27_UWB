"""
Bi-Path Dynamic Triple-Ensemble Model Architectures
9th ABC Challenge — UWB Fall Detection Track

Defines the three pure-PyTorch neural architectures forming the SOTA Triple-Ensemble:
1. `ResNetSequenceClassifier` (w_1 = 0.45):
    ResNet-18 backbone -> Range dimension pooling (.mean(dim=3)) -> 1D Temporal Head
    -> Linear interpolation to 100 frames (2.0s @ 50Hz).
2. `ResNetV1Standard` (w_2 = 0.40):
    ResNet-18 backbone -> Global Spatio-Temporal GAP (.mean(dim=(2, 3)))
    -> 2-class Softmax Classifier Head.
3. `EfficientNetSequenceClassifier` (w_3 = 0.15):
    EfficientNet-B0 backbone (7 MBConv stages with Squeeze-and-Excitation and SiLU)
    -> Range dimension pooling (.mean(dim=3)) -> 1D Temporal Head -> Linear interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"

ENSEMBLE_WEIGHTS: tuple[float, float, float] = (0.45, 0.40, 0.15)


class _BasicBlock(nn.Module):
    """Standard 2D ResNet BasicBlock."""

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


def _make_resnet_layer(
    in_planes: int, planes: int, blocks: int, stride: int
) -> nn.Sequential:
    downsample = None
    if stride != 1 or in_planes != planes:
        downsample = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes),
        )
    layers = [_BasicBlock(in_planes, planes, stride, downsample)]
    for _ in range(1, blocks):
        layers.append(_BasicBlock(planes, planes))
    return nn.Sequential(*layers)


class ResNetSequenceClassifier(nn.Module):
    """
    Model 1 (w_1 = 0.45): ResNet-18 2D Backbone with Range-Dimension Pooling
    and 1D Temporal Sequence Classification Head.
    Input: (B, 3, T_win=100, N_bins=100) -> Output logits: (B, T_win=100)
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = _make_resnet_layer(64, 64, 2, stride=1)
        self.layer2 = _make_resnet_layer(64, 128, 2, stride=2)
        self.layer3 = _make_resnet_layer(128, 256, 2, stride=2)
        self.layer4 = _make_resnet_layer(256, 512, 2, stride=2)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        feats = self.layer4(self.layer3(self.layer2(self.layer1(feats))))
        seq_logits = self.head(feats.mean(dim=3).transpose(1, 2)).transpose(1, 2)
        return F.interpolate(
            seq_logits, size=x.shape[2], mode="linear", align_corners=False
        ).squeeze(1)


class ResNetV1Standard(nn.Module):
    """
    Model 2 (w_2 = 0.40): Standard ResNet-18 with Global Spatio-Temporal Average Pooling.
    Input: (B, 3, T_win=100, N_bins=100) -> Output logits: (B, 2)
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = _make_resnet_layer(64, 64, 2, stride=1)
        self.layer2 = _make_resnet_layer(64, 128, 2, stride=2)
        self.layer3 = _make_resnet_layer(128, 256, 2, stride=2)
        self.layer4 = _make_resnet_layer(256, 512, 2, stride=2)
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        feats = self.layer4(self.layer3(self.layer2(self.layer1(feats))))
        return self.fc(feats.mean(dim=(2, 3)))


class _SqueezeExcitation(nn.Module):
    """Channel-wise Squeeze-and-Excitation attention block for MBConv."""

    def __init__(self, in_c: int, squeeze_c: int) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(in_c, squeeze_c, kernel_size=1)
        self.fc2 = nn.Conv2d(squeeze_c, in_c, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.sigmoid(self.fc2(F.silu(self.fc1(F.adaptive_avg_pool2d(x, 1)))))
        return x * scale


class _MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block with Depthwise Separable Conv & SE."""

    def __init__(
        self, in_c: int, exp_c: int, out_c: int, kernel_size: int, stride: int
    ) -> None:
        super().__init__()
        self.use_res_connect = stride == 1 and in_c == out_c
        layers: list[nn.Module] = []
        if exp_c != in_c:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_c, exp_c, kernel_size=1, bias=False),
                    nn.BatchNorm2d(exp_c),
                    nn.SiLU(inplace=True),
                )
            )
        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    exp_c,
                    exp_c,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    groups=exp_c,
                    bias=False,
                ),
                nn.BatchNorm2d(exp_c),
                nn.SiLU(inplace=True),
            )
        )
        layers.append(_SqueezeExcitation(exp_c, max(1, in_c // 4)))
        layers.append(
            nn.Sequential(
                nn.Conv2d(exp_c, out_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_c),
            )
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x) if self.use_res_connect else self.block(x)


class EfficientNetSequenceClassifier(nn.Module):
    """
    Model 3 (w_3 = 0.15): EfficientNet-B0 Backbone (7 MBConv stages) with Range Pooling
    and 1D Temporal Sequence Classification Head.
    Input: (B, 3, T_win=100, N_bins=100) -> Output logits: (B, T_win=100)
    """

    def __init__(self) -> None:
        super().__init__()
        b0_cfg = [
            (32, 32, 16, 3, 1, 1),
            (16, 96, 24, 3, 2, 2),
            (24, 144, 40, 5, 2, 2),
            (40, 240, 80, 3, 2, 3),
            (80, 480, 112, 5, 1, 3),
            (112, 672, 192, 5, 2, 4),
            (192, 1152, 320, 3, 1, 1),
        ]
        stages: list[nn.Module] = [
            nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.SiLU(inplace=True),
            )
        ]
        for in_c, exp_c, out_c, k, s, repeats in b0_cfg:
            blocks = [_MBConvBlock(in_c, exp_c, out_c, k, s)]
            expand_ratio = exp_c // in_c
            for _ in range(1, repeats):
                blocks.append(_MBConvBlock(out_c, out_c * expand_ratio, out_c, k, 1))
            stages.append(nn.Sequential(*blocks))
        stages.append(
            nn.Sequential(
                nn.Conv2d(320, 1280, kernel_size=1, bias=False),
                nn.BatchNorm2d(1280),
                nn.SiLU(inplace=True),
            )
        )
        self.features = nn.Sequential(*stages)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        seq_logits = self.classifier(feats.mean(dim=3).transpose(1, 2)).transpose(1, 2)
        return F.interpolate(
            seq_logits, size=x.shape[2], mode="linear", align_corners=False
        ).squeeze(1)


@dataclass
class TripleEnsembleModel:
    """Container holding the 3 initialized & weight-loaded ensemble models."""

    resnet_seq: ResNetSequenceClassifier
    resnet_std: ResNetV1Standard
    effnet_seq: EfficientNetSequenceClassifier
    device: torch.device
    weights: tuple[float, float, float] = ENSEMBLE_WEIGHTS


def load_triple_ensemble(
    models_dir: Path = DEFAULT_MODELS_DIR,
    device: torch.device | str | None = None,
) -> TripleEnsembleModel:
    """
    Instantiate and load trained state dicts for all three ensemble members:
      - `resnet18_physics_composite.pth` (w_1 = 0.45)
      - `best_resnet18.pth`              (w_2 = 0.40)
      - `efficientnet_b0_physics.pth`    (w_3 = 0.15)
    """
    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    m_resnet_seq = ResNetSequenceClassifier().to(dev)
    m_resnet_seq.load_state_dict(
        torch.load(
            models_dir / "resnet18_physics_composite.pth",
            map_location=dev,
            weights_only=False,
        )
    )
    m_resnet_seq.eval()

    m_resnet_std = ResNetV1Standard().to(dev)
    m_resnet_std.load_state_dict(
        torch.load(
            models_dir / "best_resnet18.pth",
            map_location=dev,
            weights_only=False,
        )
    )
    m_resnet_std.eval()

    m_effnet_seq = EfficientNetSequenceClassifier().to(dev)
    m_effnet_seq.load_state_dict(
        torch.load(
            models_dir / "efficientnet_b0_physics.pth",
            map_location=dev,
            weights_only=False,
        )
    )
    m_effnet_seq.eval()

    return TripleEnsembleModel(
        resnet_seq=m_resnet_seq,
        resnet_std=m_resnet_std,
        effnet_seq=m_effnet_seq,
        device=dev,
    )
