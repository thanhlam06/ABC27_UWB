"""
Physics-Aware Composite Loss & Temporal DIoU Loss Functions
9th ABC Challenge — UWB Fall Detection Track

Implements:
1. `build_physics_targets_and_masks`:
    - 1D Gaussian Label Smoothing (sigma = 0.147s, +/- 0.44s = 22 frames)
    - Ignore-Flagged Valid Mask M_valid(t) = 0.0 in `quality=low` or `needs_review` zones
    - Splicing Crossfade Mask M_splice(t) = 1.0 around 0.2s segment boundaries
2. `PhysicsAwareCompositeLoss`:
    L_total = L_Focal_soft(alpha=0.85, gamma=2.0) + 5.0 * L_splice(tau=0.30) + 0.08 * L_TV
3. `TemporalDIoULoss`:
    1D Distance-IoU boundary regression loss for ActionFormer/FPN temporal decoders.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.dataset import FS_HZ


def build_physics_targets_and_masks(
    stream_id: str,
    n_frames: int,
    events_df: pd.DataFrame,
    composition_df: pd.DataFrame,
    fs_hz: float = FS_HZ,
    pad_s: float = 0.44,
    sigma_s: float = 0.147,
    ignore_margin_s: float = 2.0,
    splice_pre_s: float = 0.10,
    splice_post_s: float = 0.60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct physics-aware supervision arrays for a continuous UWB stream:
      - `y_smooth`    : (T,) float32 in [0, 1], Gaussian-smoothed target for clean falls.
      - `valid_mask`  : (T,) float32 in {0, 1}, 0 inside ignored (`low`/`needs_review`) zones.
      - `splice_mask` : (T,) float32 in {0, 1}, 1 around 0.2s crossfade segment joins.
    """
    y_raw = np.zeros(n_frames, dtype=np.float32)
    valid_mask = np.ones(n_frames, dtype=np.float32)
    splice_mask = np.zeros(n_frames, dtype=np.float32)
    clean_fall_guard = np.zeros(n_frames, dtype=bool)

    stream_evs = events_df[events_df["stream_id"] == stream_id]
    pad_frames = int(round(pad_s * fs_hz))
    ign_frames = int(round(ignore_margin_s * fs_hz))

    for _, row in stream_evs.iterrows():
        s_idx = max(0, int(round(float(row["t_start"]) * fs_hz)))
        e_idx = min(n_frames, int(round(float(row["t_end"]) * fs_hz)) + 1)
        needs_rev = str(row.get("needs_review", "")).strip().lower() in ("true", "1", "yes")
        low_qual = str(row.get("quality", "")).strip().lower() == "low"

        if needs_rev or low_qual:
            ig_s = max(0, s_idx - ign_frames)
            ig_e = min(n_frames, e_idx + ign_frames)
            valid_mask[ig_s:ig_e] = 0.0
        else:
            p_s = max(0, s_idx - pad_frames)
            p_e = min(n_frames, e_idx + pad_frames)
            y_raw[p_s:p_e] = 1.0
            clean_fall_guard[max(0, s_idx - ign_frames) : min(n_frames, e_idx + ign_frames)] = True

    if y_raw.max() > 0.0:
        y_smooth = scipy.ndimage.gaussian_filter1d(
            y_raw, sigma=sigma_s * fs_hz, mode="constant", cval=0.0
        )
        core_mask = y_raw > 0.5
        if np.any(core_mask):
            peak = float(np.percentile(y_smooth[core_mask], 95))
            if peak > 1e-6:
                y_smooth = np.clip(y_smooth / peak, 0.0, 1.0)
        y_smooth = np.maximum(y_smooth, (y_raw > 0.5).astype(np.float32) * 0.85)
    else:
        y_smooth = y_raw.copy()

    y_smooth = np.clip(y_smooth, 0.0, 1.0).astype(np.float32)

    comp_stream = composition_df[composition_df["stream_id"] == stream_id].sort_values("t_start")
    if len(comp_stream) > 1:
        t_starts = comp_stream["t_start"].to_numpy(dtype=np.float64)
        t_ends = comp_stream["t_end"].to_numpy(dtype=np.float64)
        pre_f = int(round(splice_pre_s * fs_hz))
        post_f = int(round(splice_post_s * fs_hz))
        for seg_idx in range(1, len(comp_stream)):
            fade_start = float(t_starts[seg_idx])
            fade_end = max(float(t_ends[seg_idx - 1]), fade_start + 0.20)
            s_sp = max(0, int(round(fade_start * fs_hz)) - pre_f)
            e_sp = min(n_frames, int(round(fade_end * fs_hz)) + post_f)
            splice_mask[s_sp:e_sp] = 1.0

    splice_mask[(y_smooth > 1e-4) | clean_fall_guard | (valid_mask < 0.5)] = 0.0
    return y_smooth, valid_mask, splice_mask


class PhysicsAwareCompositeLoss(nn.Module):
    """
    Multi-Component Physics-Aware Objective Function:
      L_total = L_Focal_soft(alpha=0.85, gamma=2.0)
              + 5.0 * L_splice(tau_thresh=0.30)
              + 0.08 * L_TV
    """

    def __init__(
        self,
        alpha: float = 0.85,
        gamma: float = 2.0,
        splice_prob_thresh: float = 0.30,
        splice_multiplier: float = 5.0,
        tv_weight: float = 0.08,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.splice_prob_thresh = splice_prob_thresh
        self.splice_multiplier = splice_multiplier
        self.tv_weight = tv_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        splice_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        probs = torch.sigmoid(logits)
        log_p = F.logsigmoid(logits)
        log_1_minus_p = F.logsigmoid(-logits)

        if valid_mask is None:
            valid_mask = torch.ones_like(targets)
        if splice_mask is None:
            splice_mask = torch.zeros_like(targets)

        # 1. Soft-Target Focal Loss
        pos_term = -self.alpha * targets * ((1.0 - probs) ** self.gamma) * log_p
        neg_term = -(1.0 - self.alpha) * (1.0 - targets) * (probs ** self.gamma) * log_1_minus_p
        focal_per_frame = (pos_term + neg_term) * valid_mask
        denom = torch.clamp(valid_mask.sum(), min=1.0)
        focal_loss = focal_per_frame.sum() / denom

        # 2. 5x Splicing Anti-Clutter Hard-Negative Penalty
        hard_splice_gate = splice_mask * (probs > self.splice_prob_thresh).to(probs.dtype)
        splice_neg_focal = -(1.0 - self.alpha) * (probs ** self.gamma) * log_1_minus_p
        splice_excess = F.relu(probs - self.splice_prob_thresh).pow(2)
        splice_penalty = (
            self.splice_multiplier * hard_splice_gate * (splice_neg_focal + 0.15 * splice_excess)
        )
        splice_loss = splice_penalty.sum() / denom

        # 3. Total Variation (TV) Temporal Regularization
        pair_valid = valid_mask[:, 1:] * valid_mask[:, :-1]
        tv_denom = torch.clamp(pair_valid.sum(), min=1.0)
        tv_raw = (torch.abs(probs[:, 1:] - probs[:, :-1]) * pair_valid).sum() / tv_denom
        tv_loss = self.tv_weight * tv_raw

        total_loss = focal_loss + splice_loss + tv_loss
        stats = {
            "total": float(total_loss.detach().item()),
            "focal": float(focal_loss.detach().item()),
            "splice": float(splice_loss.detach().item()),
            "tv": float(tv_loss.detach().item()),
        }
        return total_loss, stats


def temporal_diou(
    pred_bounds: torch.Tensor,
    tgt_bounds: torch.Tensor,
    t: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """1D Distance-IoU for interval regression [t - ds, t + de] vs [t - gs, t + ge]."""
    ps, pe = t - pred_bounds[:, 0], t + pred_bounds[:, 1]
    gs, ge = t - tgt_bounds[:, 0], t + tgt_bounds[:, 1]
    inter = (torch.minimum(pe, ge) - torch.maximum(ps, gs)).clamp(min=0.0)
    union = (pe - ps) + (ge - gs) - inter
    iou = inter / (union + eps)
    center_dist2 = (0.5 * (ps + pe) - 0.5 * (gs + ge)).pow(2)
    enclose_w = torch.maximum(pe, ge) - torch.minimum(ps, gs)
    diou = iou - center_dist2 / (enclose_w.pow(2) + eps)
    return 1.0 - diou, iou


class TemporalDIoULoss(nn.Module):
    """Combined Focal + Splice Penalty + 1D Temporal DIoU Loss."""

    def __init__(
        self,
        alpha: float = 0.85,
        gamma: float = 2.0,
        splice_weight: float = 5.0,
        diou_weight: float = 2.0,
        fs: float = FS_HZ,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.splice_weight = splice_weight
        self.diou_weight = diou_weight
        self.fs = fs

    def forward(
        self,
        p_fall: torch.Tensor,
        reg_bounds: torch.Tensor,
        cls_target: torch.Tensor,
        reg_target: torch.Tensor,
        splice_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        p_fall = p_fall.to(torch.float32).clamp(1e-6, 1.0 - 1e-6)
        reg_bounds = reg_bounds.to(torch.float32)
        pos = cls_target > 0.5
        n_pos = pos.sum().clamp(min=1).to(torch.float32)

        p_t = torch.where(pos, p_fall, 1.0 - p_fall)
        alpha_t = torch.where(
            pos, torch.full_like(p_fall, self.alpha), torch.full_like(p_fall, 1.0 - self.alpha)
        )
        fl = -alpha_t * (1.0 - p_t).pow(self.gamma) * torch.log(p_t)
        focal = fl.sum() / n_pos

        if splice_mask is not None:
            junction_neg = splice_mask.to(torch.bool) & (~pos)
            splice = self.splice_weight * (fl * junction_neg.to(torch.float32)).sum() / n_pos
        else:
            splice = fl.new_zeros(())

        if pos.any():
            b_sz, length = p_fall.shape
            t = (
                torch.arange(length, device=p_fall.device, dtype=torch.float32) / self.fs
            ).expand(b_sz, length)
            pred = reg_bounds.permute(0, 2, 1)[pos]
            tgt = reg_target.permute(0, 2, 1)[pos]
            diou_l, iou = temporal_diou(pred, tgt, t[pos])
            diou_loss, mean_iou = diou_l.mean(), iou.mean().detach()
        else:
            diou_loss, mean_iou = reg_bounds.sum() * 0.0, reg_bounds.new_zeros(())

        total = focal + splice + self.diou_weight * diou_loss
        return total, {
            "focal": float(focal.detach()),
            "splice": float(splice.detach()),
            "diou": float(diou_loss.detach()),
            "iou": float(mean_iou),
        }
