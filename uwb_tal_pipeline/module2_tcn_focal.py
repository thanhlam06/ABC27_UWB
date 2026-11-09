"""
Module 2: Frame-wise 1D Temporal Convolutional Network (1D-TCN) & Focal Loss
9th ABC Challenge - UWB Fall Track (Temporal Action Localization Pipeline)

1. Architecture: Lightweight Multi-Scale Dilated 1D-TCN accepting (Batch, Channels=3, Length)
   and outputting frame-wise Fall probability p(t) in [0, 1] of identical temporal length.
2. Loss Function: Numerically stable Binary Focal Loss addressing severe class imbalance
   (~1.23% positive Fall frames vs. ~98.77% background frames).
3. Leak-Free Training & Inference: 3-Fold Contiguous Temporal Block training (each fold trains
   on 20,000 frames and predicts out-of-fold frame-wise probabilities on the unseen 10,000-frame block).
"""

from __future__ import annotations

from dataclasses import dataclass
from scipy.ndimage import maximum_filter1d
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss for extreme foreground/background frame imbalance:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Implemented directly on raw logits for numerical stability.
    """

    def __init__(self, alpha: float = 0.85, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        modulating_factor = (1.0 - p_t).pow(self.gamma)
        focal_loss = alpha_t * modulating_factor * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class DilatedResidualBlock1D(nn.Module):
    """ Length-preserving 1D Dilated Residual Temporal Block. """

    def __init__(self, channels: int, kernel_size: int = 7, dilation: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return F.gelu(out + residual)


class FrameWise1DTCN(nn.Module):
    """
    Lightweight Frame-wise 1D Temporal Convolutional Network (1D-TCN).
    Input : (Batch, Channels=3, Length)
    Output: p(t) in [0, 1] of shape (Batch, Length) (and raw logits for Focal Loss training)
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 24,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [DilatedResidualBlock1D(hidden_channels, kernel_size=7, dilation=d) for d in dilations]
        )
        self.head = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        # Initialize final bias to prior log-odds log(0.012 / 0.988) ≈ -4.4 for fast Focal convergence
        nn.init.constant_(self.head.bias, -4.4)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw frame-wise logits of shape (Batch, Length)."""
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h).squeeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return frame-wise Fall probability sequence p(t) in [0, 1] of shape (Batch, Length)."""
        return torch.sigmoid(self.forward_logits(x))


@dataclass
class TCNTrainingOutput:
    """Container for Module 2 trained 1D-TCN predictions and diagnostics."""
    frame_probs: np.ndarray        # Out-of-fold frame-wise probability p(t) in [0, 1], shape (30000,)
    fold_losses: list[float]
    param_count: int


def run_module2_tcn_segmentation(
    norm_envelope: np.ndarray,
    frame_targets: np.ndarray,
    seed: int = 42,
    epochs: int = 65,
    coarse_margin_frames: int = 22,
) -> TCNTrainingOutput:
    """
    Train the Frame-wise 1D-TCN with Binary Focal Loss using leak-free 3-Fold Temporal Block
    Cross-Validation across the 30,000-frame stream and return out-of-fold p(t).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_radars, n_frames = norm_envelope.shape
    oof_probs = np.zeros(n_frames, dtype=np.float32)
    block_edges = [(0, 10_000), (10_000, 20_000), (20_000, 30_000)]

    # Dilate binary targets slightly (+/- 0.44s) for the coarse proposal stage so that
    # p(t) >= 0.5 safely brackets the full fall before Module 3 gradient shrinking.
    coarse_targets = maximum_filter1d(frame_targets, size=2 * coarse_margin_frames + 1).astype(np.float32)

    criterion = BinaryFocalLoss(alpha=0.75, gamma=2.0)
    fold_final_losses: list[float] = []
    param_count = 0

    for fold_idx, (b_start, b_end) in enumerate(block_edges):
        torch.manual_seed(seed + fold_idx)
        model = FrameWise1DTCN(in_channels=n_radars, hidden_channels=24, dilations=(1, 2, 4, 8, 16))
        if fold_idx == 0:
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

        optimizer = torch.optim.AdamW(model.parameters(), lr=6e-3, weight_decay=1e-4)

        # Extract 4 x 2,500-frame (50s) sub-sequences covering the entire 10,000-frame training blocks
        train_crops_x = []
        train_crops_y = []
        for tb_start, tb_end in block_edges:
            if (tb_start, tb_end) == (b_start, b_end):
                continue
            for c0 in (0, 2500, 5000, 7500):
                train_crops_x.append(norm_envelope[:, tb_start + c0 : tb_start + c0 + 2500])
                train_crops_y.append(coarse_targets[tb_start + c0 : tb_start + c0 + 2500])

        x_train = torch.from_numpy(np.stack(train_crops_x, axis=0))  # (4, 3, 2500)
        y_train = torch.from_numpy(np.stack(train_crops_y, axis=0))  # (4, 2500)

        model.train()
        last_loss = 0.0
        for _ in range(epochs):
            optimizer.zero_grad()
            logits = model.forward_logits(x_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())

        fold_final_losses.append(last_loss)

        # Out-of-fold full-stream inference on the held-out 10,000-frame block: shape (1, 3, 10000)
        model.eval()
        with torch.no_grad():
            x_val = torch.from_numpy(norm_envelope[:, b_start:b_end][None, :, :])
            probs_val = model(x_val).squeeze(0).cpu().numpy()
            oof_probs[b_start:b_end] = probs_val

    pos_probs = oof_probs[frame_targets == 1.0]
    neg_probs = oof_probs[frame_targets == 0.0]

    print("=" * 80)
    print("MODULE 2: FRAME-WISE 1D-TCN & FOCAL LOSS TRAINING")
    print("=" * 80)
    print(f"Model Architecture             : FrameWise1DTCN (Dilations=(1, 2, 4, 8, 16), Params={param_count:,})")
    print(f"Input -> Output Tensor Shape   : (Batch, 3, 30000) -> p(t) of shape (Batch, 30000) in [0, 1]")
    print(f"Loss Function                  : BinaryFocalLoss(alpha=0.90, gamma=2.0)")
    print(f"3-Fold Final Focal Losses      : {[round(l, 5) for l in fold_final_losses]}")
    print(f"OOF Mean p(t) on Fall Frames   : {pos_probs.mean():.4f} (Max: {pos_probs.max():.4f})")
    print(f"OOF Mean p(t) on Non-Fall      : {neg_probs.mean():.4f} (99th Pct: {np.percentile(neg_probs, 99):.4f})")
    print()

    return TCNTrainingOutput(
        frame_probs=oof_probs,
        fold_losses=fold_final_losses,
        param_count=param_count,
    )
