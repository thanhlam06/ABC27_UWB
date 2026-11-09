"""
Fine-Stride Sliding-Window Test-Time Augmentation (TTA) Inference Engine
9th ABC Challenge — UWB Fall Detection Track

Executes 2.0s (100-frame) sliding windows with a fine stride of 5 frames (0.10s),
yielding 20x voting overlap across all three ensemble models:
    P_ens(t) = 0.45 * p_1(t) + 0.40 * p_2(t) + 0.15 * p_3(t)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.dataset import N_ACTIVE_BINS
from src.models.triple_ensemble import TripleEnsembleModel
from src.preprocessing.clutter_removal import preprocess_deep_branch

WINDOW_FRAMES: int = 100
HOP_FRAMES: int = 5


def infer_stream_tta(
    model: nn.Module,
    cir_clean: np.ndarray,
    batch_size: int = 64,
    is_sequence: bool = True,
    window_frames: int = WINDOW_FRAMES,
    hop_frames: int = HOP_FRAMES,
    n_bins: int = N_ACTIVE_BINS,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """
    Process EMA-cleaned UWB stream using sliding windows with fine stride (HOP_FRAMES = 5).
    Accumulates 20x overlapping window predictions per slow-time frame t.
    """
    if device is None:
        dev = next(model.parameters()).device
    else:
        dev = torch.device(device)

    amp_norm = preprocess_deep_branch(cir_clean, n_bins=n_bins)
    n_frames = amp_norm.shape[1]

    if n_frames <= window_frames:
        starts = [0]
    else:
        starts = list(range(0, n_frames - window_frames + 1, hop_frames))
        if starts[-1] != n_frames - window_frames:
            starts.append(n_frames - window_frames)

    prob_frames = np.zeros(n_frames, dtype=np.float64)
    prob_counts = np.zeros(n_frames, dtype=np.float64)

    model.eval()
    with torch.no_grad():
        for i in range(0, len(starts), batch_size):
            batch_starts = starts[i : i + batch_size]
            windows = np.stack(
                [amp_norm[:, s : s + window_frames, :] for s in batch_starts],
                axis=0,
            )
            batch_x = torch.from_numpy(windows).to(dev, dtype=torch.float32)
            outputs = model(batch_x)

            if is_sequence:
                if outputs.ndim == 3:
                    outputs = (
                        outputs.squeeze(1)
                        if outputs.shape[1] == 1
                        else outputs.squeeze(-1)
                    )
                if outputs.shape[-1] != window_frames:
                    outputs = F.interpolate(
                        outputs.unsqueeze(1),
                        size=window_frames,
                        mode="linear",
                        align_corners=False,
                    ).squeeze(1)
                if bool((outputs.min() < 0.0) or (outputs.max() > 1.0)):
                    probs = torch.sigmoid(outputs)
                else:
                    probs = outputs
                probs_np = probs.detach().cpu().numpy()
                for j, s in enumerate(batch_starts):
                    end = min(n_frames, s + window_frames)
                    prob_frames[s:end] += probs_np[j, : end - s]
                    prob_counts[s:end] += 1.0
            else:
                if outputs.ndim == 2 and outputs.shape[-1] == 2:
                    probs = torch.softmax(outputs, dim=-1)[:, 1]
                else:
                    outputs_1d = outputs.view(-1)
                    probs = (
                        torch.sigmoid(outputs_1d)
                        if bool((outputs_1d.min() < 0.0) or (outputs_1d.max() > 1.0))
                        else outputs_1d
                    )
                probs_np = probs.detach().cpu().numpy()
                for j, s in enumerate(batch_starts):
                    end = min(n_frames, s + window_frames)
                    prob_frames[s:end] += float(probs_np[j])
                    prob_counts[s:end] += 1.0

    return (prob_frames / np.maximum(prob_counts, 1.0)).astype(np.float32)


def predict_triple_ensemble_probabilities(
    ensemble: TripleEnsembleModel,
    cir_clean: np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Compute weighted consensus probability curve across all 3 models:
        P_ens(t) = 0.45 * p_1(t) + 0.40 * p_2(t) + 0.15 * p_3(t)
    """
    w1, w2, w3 = ensemble.weights
    p1 = infer_stream_tta(
        ensemble.resnet_seq,
        cir_clean,
        batch_size=batch_size,
        is_sequence=True,
        device=ensemble.device,
    )
    p2 = infer_stream_tta(
        ensemble.resnet_std,
        cir_clean,
        batch_size=batch_size,
        is_sequence=False,
        device=ensemble.device,
    )
    p3 = infer_stream_tta(
        ensemble.effnet_seq,
        cir_clean,
        batch_size=batch_size,
        is_sequence=True,
        device=ensemble.device,
    )
    return (w1 * p1 + w2 * p2 + w3 * p3).astype(np.float32)
