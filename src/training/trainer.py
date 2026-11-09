"""
Supervised Training Dataset & Epoch Runner for 2D/1D Sequence Classifiers
9th ABC Challenge — UWB Fall Detection Track
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .losses import PhysicsAwareCompositeLoss


class SlidingWindowFallDataset(Dataset):
    """
    Extracts 2.0s (100-frame x 100-bin x 3-radar) windows along with physics-aware
    soft targets `y_smooth`, `valid_mask`, and `splice_mask`.
    """

    def __init__(
        self,
        amp_norm: np.ndarray,
        y_smooth: np.ndarray,
        valid_mask: np.ndarray,
        splice_mask: np.ndarray,
        window_frames: int = 100,
        stride_frames: int = 20,
        augment: bool = True,
    ) -> None:
        self.amp_norm = amp_norm
        self.y_smooth = y_smooth
        self.valid_mask = valid_mask
        self.splice_mask = splice_mask
        self.window_frames = window_frames
        self.augment = augment

        n_frames = amp_norm.shape[1]
        self.starts = list(range(0, max(1, n_frames - window_frames + 1), stride_frames))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.starts[idx]
        e = s + self.window_frames
        x = self.amp_norm[:, s:e, :].copy()
        y = self.y_smooth[s:e].copy()
        vm = self.valid_mask[s:e].copy()
        sm = self.splice_mask[s:e].copy()

        if self.augment:
            gain = np.random.uniform(0.90, 1.10, size=(3, 1, 1)).astype(np.float32)
            x = x * gain

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(vm),
            torch.from_numpy(sm),
        )


def train_sequence_classifier_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: PhysicsAwareCompositeLoss,
    device: torch.device,
    max_grad_norm: float = 5.0,
) -> dict[str, float]:
    """Run one supervised training epoch with Physics-Aware Composite Loss."""
    model.train()
    totals = {"total": 0.0, "focal": 0.0, "splice": 0.0, "tv": 0.0}
    n_batches = 0

    for x, y, vm, sm in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)
        vm = vm.to(device, dtype=torch.float32)
        sm = sm.to(device, dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss, stats = criterion(logits, y, vm, sm)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        for k in totals:
            totals[k] += stats[k]
        n_batches += 1

    denom = max(n_batches, 1)
    return {k: v / denom for k, v in totals.items()}
