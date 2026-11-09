"""
Final Physics-Aware Training Pipeline for ABC2027 UWB Fall Detection Challenge.

Implements:
- Module 1: 12-Channel Spatial Range Banding (4 range bands x 3 radars), per-channel
  IQR robust normalization, and Spatial-Attention 1D-TCN (receptive field = 7.54s).
- Module 2: 1D Gaussian Label Smoothing (+/- 0.44s = 22 frames) and Physics-Aware
  Composite Loss (Modified Focal Loss + 5x Splicing Anti-Clutter Hard-Negative Penalty
  + Total Variation Temporal Smoothness Loss) trained over all 13 real Train streams.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, lfilter, lfilter_zi
from torch.utils.data import DataLoader, Dataset

# Workspace paths
PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_PKG_DIR = PROJECT_ROOT / "abc2027_uwb_train_v1"
TRAIN_ZIP_PATH = PROJECT_ROOT / "abc2027_uwb_train_v1.zip"
DEFAULT_CKPT_PATH = PROJECT_ROOT / "checkpoints" / "best_framewise_1d_tcn.pth"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "checkpoints" / "training_metrics.json"
DEFAULT_CACHE_DIR = Path(tempfile.gettempdir()) / "abc2027_12band_cache_v1"

if str(TRAIN_PKG_DIR / "src") not in sys.path:
    sys.path.insert(0, str(TRAIN_PKG_DIR / "src"))

from abc2027.evaluate import evaluate as official_evaluate
from abc2027.evaluate import load_ground_truth

# Physical radar constants
FS_HZ: float = 50.0
BACKGROUND_TAU_S: float = 3.0
NUM_RADARS: int = 3

# Module 1: 4 Spatial Range Bands per radar (inclusive bin ranges converted to Python slices)
#   Band 1 (Near)         : Bins 0 - 65    -> [0:66]
#   Band 2 (Static)       : Bins 75 - 180  -> [75:181]
#   Band 3 (Mat/Fall Zone): Bins 183 - 225 -> [183:226]
#   Band 4 (Far)          : Bins 260 - 367 -> [260:368]
SPATIAL_BANDS: tuple[tuple[int, int], ...] = (
    (0, 66),
    (75, 181),
    (183, 226),
    (260, 368),
)
SPATIAL_BAND_NAMES: tuple[str, ...] = (
    "Near_0_65",
    "Static_75_180",
    "Mat_183_225",
    "Far_260_367",
)
IN_CHANNELS: int = NUM_RADARS * len(SPATIAL_BANDS)  # 12 channels


# ============================================================================
# MODULE 1: DATA REPRESENTATION, SPATIAL BANDING & MODEL ARCHITECTURE
# ============================================================================


def causal_ema_background_subtraction(
    cir: np.ndarray,
    tau_s: float = BACKGROUND_TAU_S,
    rate_hz: float = FS_HZ,
) -> np.ndarray:
    """Subtract causal EMA background independently per radar and range bin (exact abc2027 formula)."""
    try:
        from abc2027.background import subtract_background

        return subtract_background(cir, tau_s=tau_s, rate_hz=rate_hz)
    except ImportError:
        alpha = float(np.exp(-1.0 / (rate_hz * tau_s)))
        radars, count, bins = cir.shape
        frames = np.ascontiguousarray(np.moveaxis(cir, 1, -1), dtype=np.float64).reshape(radars * bins, count)
        numerator = [1.0 - alpha]
        denominator = [1.0, -alpha]
        steady = lfilter_zi(numerator, denominator)
        residual = np.empty_like(frames)
        for idx in range(frames.shape[0]):
            avg, _ = lfilter(numerator, denominator, frames[idx], zi=frames[idx, 0] * steady)
            residual[idx] = frames[idx] - avg
        restored = np.moveaxis(residual.reshape(radars, bins, count), -1, 1)
        return restored.astype(np.float32, copy=False)


def robust_iqr_normalize(raw_bands: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Apply per-channel robust IQR normalization:
        norm_c(t) = (e_c(t) - Median_c) / IQR_c
    Avoids P99.2 compression on streams with high real fall event density.
    """
    median = np.median(raw_bands, axis=1, keepdims=True)
    q75 = np.percentile(raw_bands, 75.0, axis=1, keepdims=True)
    q25 = np.percentile(raw_bands, 25.0, axis=1, keepdims=True)
    iqr = np.maximum(q75 - q25, eps)
    norm_bands = (raw_bands - median) / iqr
    return norm_bands.astype(np.float32, copy=False)


def extract_spatial_bands(
    cir: np.ndarray,
    radar_serials: np.ndarray | None = None,
    tau_s: float = BACKGROUND_TAU_S,
    rate_hz: float = FS_HZ,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform raw (3, T, 368) CIR stream into 12-channel spatial range band envelopes.

    Returns:
        norm_bands: IQR-normalized 12-channel array of shape (12, T), float32.
        raw_bands : Unnormalized L2 spatial band energies of shape (12, T), float32.
    """
    if cir.ndim != 3 or cir.shape[0] != NUM_RADARS:
        raise ValueError(f"Expected CIR shape (3, T, 368), got {cir.shape}")

    # Deterministic radar ordering by hardware serial number when present
    if radar_serials is not None and len(radar_serials) == NUM_RADARS:
        order = np.argsort(np.asarray(radar_serials))
        if not np.array_equal(order, np.arange(NUM_RADARS)):
            cir = cir[order]

    residual = causal_ema_background_subtraction(cir, tau_s=tau_s, rate_hz=rate_hz)

    band_channels: list[np.ndarray] = []
    for r_idx in range(NUM_RADARS):
        for b_start, b_end in SPATIAL_BANDS:
            sub_band = residual[r_idx, :, b_start:b_end].astype(np.float64, copy=False)
            l2_energy = np.sqrt(np.sum(sub_band * sub_band, axis=1)).astype(np.float32)
            band_channels.append(l2_energy)

    raw_bands = np.stack(band_channels, axis=0)  # (12, T)
    norm_bands = robust_iqr_normalize(raw_bands)  # (12, T)
    return norm_bands, raw_bands


class SpatialBandAttention1D(nn.Module):
    """
    Pointwise (kernel_size=1) frame-wise spatial-band attention module.
    Learns cross-band & cross-radar contrast (e.g., Mat Zone 183-225 spike vs.
    uniform Near/Static/Far splice clutter) without altering the temporal receptive field.
    """

    def __init__(self, channels: int = IN_CHANNELS, hidden: int = 24) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv1d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.mix = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.gate(x)
        return x * (0.5 + attn) + 0.25 * self.mix(x)


class DilatedResidualBlock1D(nn.Module):
    """Length-preserving 1D Dilated Residual Temporal Block."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        dilation: int = 1,
        dropout: float = 0.15,
    ) -> None:
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
    Spatial-Banded Frame-wise 1D Temporal Convolutional Network (FrameWise1DTCN).

    Input : (Batch, Channels=12, Seq_Len)
    Output: Frame-wise fall probability p(t) in [0, 1] of shape (Batch, Seq_Len)

    Receptive Field Verification (at 50 Hz):
      - SpatialBandAttention1D (k=1)                 : +0 frames
      - Stem Conv1d (k=5)                            : +4 frames
      - 5x DilatedResidualBlock1D (2 convs, k=7, d)  : +2 * 6 * (1+2+4+8+16) = +372 frames
      - Head Conv1d (k=1)                            : +0 frames
      Total Receptive Field = 1 + 4 + 372 = 377 frames = 7.54 seconds.
    """

    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        hidden_channels: int = 32,
        stem_kernel_size: int = 5,
        block_kernel_size: int = 7,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
        dropout: float = 0.15,
        head_bias_init: float = -4.4,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.stem_kernel_size = stem_kernel_size
        self.block_kernel_size = block_kernel_size
        self.dilations = tuple(dilations)

        self.spatial_attn = SpatialBandAttention1D(channels=in_channels, hidden=max(24, in_channels * 2))
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels,
                hidden_channels,
                kernel_size=stem_kernel_size,
                padding=(stem_kernel_size - 1) // 2,
            ),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                DilatedResidualBlock1D(
                    channels=hidden_channels,
                    kernel_size=block_kernel_size,
                    dilation=d,
                    dropout=dropout,
                )
                for d in self.dilations
            ]
        )
        self.head = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        nn.init.constant_(self.head.bias, head_bias_init)

    @property
    def receptive_field_frames(self) -> int:
        stem_rf = self.stem_kernel_size - 1
        blocks_rf = sum(2 * (self.block_kernel_size - 1) * d for d in self.dilations)
        return 1 + stem_rf + blocks_rf

    @property
    def receptive_field_seconds(self) -> float:
        return self.receptive_field_frames / FS_HZ

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Compute raw frame-wise logits of shape (Batch, Seq_Len)."""
        h = self.spatial_attn(x)
        h = self.stem(h)
        for block in self.blocks:
            h = block(h)
        return self.head(h).squeeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute frame-wise probability p(t) in [0, 1] of shape (Batch, Seq_Len)."""
        return torch.sigmoid(self.forward_logits(x))


# ============================================================================
# MODULE 2: PHYSICS-AWARE LABELS, SPLICE MASKS & COMPOSITE LOSS FUNCTION
# ============================================================================


def _as_bool_flag(val: object) -> bool:
    return str(val).strip().lower() in {"1", "true", "yes"}


def build_physics_targets_and_masks(
    stream_id: str,
    n_frames: int,
    events_df: pd.DataFrame,
    composition_df: pd.DataFrame,
    fs_hz: float = FS_HZ,
    smooth_half_width_frames: int = 22,
    ignore_margin_s: float = 1.50,
    splice_pre_s: float = 0.10,
    splice_post_s: float = 0.60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct frame-wise Gaussian-smoothed targets, valid loss masks, and splice masks:
      1. y_smooth (T,): Binary clean fall labels convolved with a 1D Gaussian filter
         spreading "1" out by +/- 0.44s (22 frames at 50 Hz; sigma = 22/3, truncate = 3.0).
      2. valid_mask (T,): 1.0 on evaluable frames; 0.0 inside flagged ignore zones
         (quality == 'low' or needs_review == True), matching official --ignore-flagged scoring.
      3. splice_mask (T,): 1.0 around known 0.2s crossfade segment joins in streams_composition.csv
         (excluding any frame near a real fall).
    """
    y_binary = np.zeros(n_frames, dtype=np.float32)
    valid_mask = np.ones(n_frames, dtype=np.float32)
    splice_mask = np.zeros(n_frames, dtype=np.float32)

    ev_stream = events_df[events_df["stream_id"] == stream_id]

    # First mark flagged/ambiguous ignore intervals so noisy/unreviewed falls don't corrupt gradients
    margin_frames = int(round(ignore_margin_s * fs_hz))
    for _, row in ev_stream.iterrows():
        is_low = str(row.get("quality", "")).strip().lower() == "low"
        needs_rev = _as_bool_flag(row.get("needs_review", False))
        is_ign = _as_bool_flag(row.get("ignore", False))
        if is_low or needs_rev or is_ign:
            s_idx = max(0, int(round(float(row["t_start"]) * fs_hz)) - margin_frames)
            e_idx = min(n_frames, int(round(float(row["t_end"]) * fs_hz)) + margin_frames)
            valid_mask[s_idx:e_idx] = 0.0

    # Mark clean evaluable falls and ensure they are always active in valid_mask
    clean_fall_guard = np.zeros(n_frames, dtype=bool)
    for _, row in ev_stream.iterrows():
        is_low = str(row.get("quality", "")).strip().lower() == "low"
        needs_rev = _as_bool_flag(row.get("needs_review", False))
        is_ign = _as_bool_flag(row.get("ignore", False))
        if not (is_low or needs_rev or is_ign):
            s_idx = max(0, int(round(float(row["t_start"]) * fs_hz)))
            e_idx = min(n_frames, int(round(float(row["t_end"]) * fs_hz)))
            if e_idx > s_idx:
                y_binary[s_idx:e_idx] = 1.0
                pad_s = max(0, s_idx - smooth_half_width_frames)
                pad_e = min(n_frames, e_idx + smooth_half_width_frames)
                valid_mask[pad_s:pad_e] = 1.0
                clean_fall_guard[max(0, s_idx - int(fs_hz)) : min(n_frames, e_idx + int(fs_hz))] = True

    # 1D Gaussian smoothing spreading "1" out by +/- 22 frames (+/- 0.44s)
    sigma = float(smooth_half_width_frames) / 3.0
    y_smooth = gaussian_filter1d(y_binary.astype(np.float64), sigma=sigma, truncate=3.0, mode="nearest")
    y_smooth = np.clip(y_smooth, 0.0, 1.0).astype(np.float32)

    # Mark known 0.2s crossfade splice regions between concatenated source segments
    comp_stream = composition_df[composition_df["stream_id"] == stream_id].sort_values("t_start")
    if len(comp_stream) > 1:
        t_starts = comp_stream["t_start"].to_numpy(dtype=np.float64)
        t_ends = comp_stream["t_end"].to_numpy(dtype=np.float64)
        pre_frames = int(round(splice_pre_s * fs_hz))
        post_frames = int(round(splice_post_s * fs_hz))
        for seg_idx in range(1, len(comp_stream)):
            fade_start = float(t_starts[seg_idx])
            fade_end = max(float(t_ends[seg_idx - 1]), fade_start + 0.20)
            s_sp = max(0, int(round(fade_start * fs_hz)) - pre_frames)
            e_sp = min(n_frames, int(round(fade_end * fs_hz)) + post_frames)
            splice_mask[s_sp:e_sp] = 1.0

    # Never apply splice negative penalty on or adjacent to real falls or ignored falls
    splice_mask[(y_smooth > 1e-4) | clean_fall_guard | (valid_mask < 0.5)] = 0.0
    return y_smooth, valid_mask, splice_mask


class PhysicsAwareCompositeLoss(nn.Module):
    """
    Multi-component Physics-Aware Loss Function:
      1. Modified Focal Loss (gamma=2.0, alpha=0.85) supporting continuous Gaussian-smoothed labels.
      2. Splicing Anti-Clutter Hard-Negative Penalty (multiplier=5.0 when p(t) > 0.30 at splice points).
      3. Total Variation (TV) Loss on |p(t) - p(t-1)| to enforce temporal smoothness.
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
        valid_mask: torch.Tensor,
        splice_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        probs = torch.sigmoid(logits)
        log_p = F.logsigmoid(logits)
        log_1_minus_p = F.logsigmoid(-logits)

        # 1. Modified Focal Loss for soft Gaussian-smoothed targets y in [0, 1]
        pos_term = -self.alpha * targets * ((1.0 - probs) ** self.gamma) * log_p
        neg_term = -(1.0 - self.alpha) * (1.0 - targets) * (probs ** self.gamma) * log_1_minus_p
        focal_per_frame = (pos_term + neg_term) * valid_mask

        denom = torch.clamp(valid_mask.sum(), min=1.0)
        focal_loss = focal_per_frame.sum() / denom

        # 2. Splicing Anti-Clutter Hard-Negative Mining:
        #    Penalize heavily (multiplier = 5.0) whenever p(t) > 0.3 at known 0.2s splice regions
        hard_splice_gate = splice_mask * (probs > self.splice_prob_thresh).to(probs.dtype)
        splice_neg_focal = -(1.0 - self.alpha) * (probs ** self.gamma) * log_1_minus_p
        splice_excess = F.relu(probs - self.splice_prob_thresh).pow(2)
        splice_penalty_frame = self.splice_multiplier * hard_splice_gate * (splice_neg_focal + 0.15 * splice_excess)
        splice_loss = splice_penalty_frame.sum() / denom

        # 3. Total Variation (TV) Loss: |p(t) - p(t-1)| for temporal smoothness
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


# ============================================================================
# STREAM LOADING (DISK OR IN-MEMORY ZIP), CACHING & CHUNKED DATALOADER
# ============================================================================


@dataclass
class PreparedStream:
    stream_id: str
    norm_bands: np.ndarray  # (12, T) float32
    raw_bands: np.ndarray  # (12, T) float32
    y_smooth: np.ndarray  # (T,) float32
    valid_mask: np.ndarray  # (T,) float32
    splice_mask: np.ndarray  # (T,) float32

    @property
    def n_frames(self) -> int:
        return int(self.norm_bands.shape[1])

    @property
    def duration_s(self) -> float:
        return float(self.n_frames) / FS_HZ


def _load_cir_from_disk_or_zip(
    split: str,
    index: int,
    train_pkg_dir: Path = TRAIN_PKG_DIR,
    zip_path: Path = TRAIN_ZIP_PATH,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load (cir, radar_serials) from extracted HDF5 or seamlessly in-memory from zip archive."""
    disk_path = train_pkg_dir / "streams" / split / f"stream_{index:02d}.h5"
    if disk_path.is_file():
        with h5py.File(disk_path, "r") as handle:
            cir = handle["cir"][:]
            serials = handle["radar_serials"][:] if "radar_serials" in handle else None
        return cir, serials

    if zip_path.is_file():
        member = f"abc2027_uwb_train_v1/streams/{split}/stream_{index:02d}.h5"
        with zipfile.ZipFile(zip_path, "r") as zf:
            if member in zf.namelist():
                raw_bytes = zf.read(member)
                with h5py.File(io.BytesIO(raw_bytes), "r") as handle:
                    cir = handle["cir"][:]
                    serials = handle["radar_serials"][:] if "radar_serials" in handle else None
                return cir, serials

    raise FileNotFoundError(f"Could not find {split}/stream_{index:02d}.h5 on disk or in {zip_path}")


def load_prepared_streams(
    split: str,
    stream_indices: list[int],
    events_df: pd.DataFrame,
    composition_df: pd.DataFrame,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
) -> list[PreparedStream]:
    """Load and preprocess all streams for a split, using lightweight compressed caching."""
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    streams: list[PreparedStream] = []
    for idx in stream_indices:
        stream_id = f"{split}_{idx:02d}"
        cache_file = cache_dir / f"{stream_id}_12band.npz" if cache_dir is not None else None

        t0 = time.perf_counter()
        if cache_file is not None and cache_file.is_file():
            with np.load(cache_file) as data:
                norm_bands = data["norm_bands"].astype(np.float32)
                raw_bands = data["raw_bands"].astype(np.float32)
            source_tag = "cache"
        else:
            cir, serials = _load_cir_from_disk_or_zip(split, idx)
            norm_bands, raw_bands = extract_spatial_bands(cir, radar_serials=serials)
            if cache_file is not None:
                np.savez_compressed(cache_file, norm_bands=norm_bands, raw_bands=raw_bands)
            source_tag = "computed"

        y_smooth, valid_mask, splice_mask = build_physics_targets_and_masks(
            stream_id=stream_id,
            n_frames=norm_bands.shape[1],
            events_df=events_df,
            composition_df=composition_df,
        )
        dt = time.perf_counter() - t0
        n_pos = int((y_smooth > 0.5).sum())
        n_sp = int((splice_mask > 0.5).sum())
        print(
            f"  [{stream_id}] frames={norm_bands.shape[1]:6d} ({norm_bands.shape[1]/FS_HZ:6.1f}s) | "
            f"pos_frames={n_pos:4d} | splice_frames={n_sp:4d} | {source_tag} in {dt:.2f}s"
        )
        streams.append(
            PreparedStream(
                stream_id=stream_id,
                norm_bands=norm_bands,
                raw_bands=raw_bands,
                y_smooth=y_smooth,
                valid_mask=valid_mask,
                splice_mask=splice_mask,
            )
        )
    return streams


class ChunkedStreamDataset(Dataset):
    """
    Slices continuous multi-radar streams into 10,000-frame chunks with 500-frame overlap
    (stride = 9,500). Supports random temporal jitter & mild channel gain scaling during training.
    """

    def __init__(
        self,
        streams: list[PreparedStream],
        chunk_size: int = 10_000,
        overlap: int = 500,
        augment: bool = True,
    ) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be strictly smaller than chunk_size")
        self.streams = streams
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.stride = chunk_size - overlap
        self.augment = augment

        self.index_Spec: list[tuple[int, int]] = []
        for s_idx, st in enumerate(streams):
            n_f = st.n_frames
            if n_f <= chunk_size:
                self.index_Spec.append((s_idx, 0))
                continue
            starts = list(range(0, n_f - chunk_size + 1, self.stride))
            if starts[-1] + chunk_size < n_f:
                starts.append(n_f - chunk_size)
            for st_frame in starts:
                self.index_Spec.append((s_idx, st_frame))

    def __len__(self) -> int:
        return len(self.index_Spec)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s_idx, base_start = self.index_Spec[idx]
        st = self.streams[s_idx]
        n_f = st.n_frames

        start = base_start
        if self.augment and n_f > self.chunk_size:
            jitter = int(np.random.randint(-self.overlap // 2, self.overlap // 2 + 1))
            start = min(max(0, base_start + jitter), n_f - self.chunk_size)

        end = min(start + self.chunk_size, n_f)
        x = st.norm_bands[:, start:end].copy()
        y = st.y_smooth[start:end].copy()
        vm = st.valid_mask[start:end].copy()
        sm = st.splice_mask[start:end].copy()

        # Pad short streams if any (< 10,000 frames)
        if x.shape[1] < self.chunk_size:
            pad_len = self.chunk_size - x.shape[1]
            x = np.pad(x, ((0, 0), (0, pad_len)), mode="edge")
            y = np.pad(y, (0, pad_len), mode="constant", constant_values=0.0)
            vm = np.pad(vm, (0, pad_len), mode="constant", constant_values=0.0)
            sm = np.pad(sm, (0, pad_len), mode="constant", constant_values=0.0)

        if self.augment:
            # Mild per-channel amplitude scaling (+/- 8%) to improve cross-session generalization
            gain = np.random.uniform(0.92, 1.08, size=(x.shape[0], 1)).astype(np.float32)
            x = x * gain

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(vm),
            torch.from_numpy(sm),
        )


# ============================================================================
# SHARED CHUNKED STREAM INFERENCE & COARSE-TO-FINE POST-PROCESSING FOR VAL
# ============================================================================


@torch.no_grad()
def infer_stream_probabilities(
    model: FrameWise1DTCN,
    norm_bands: np.ndarray,
    chunk_size: int = 10_000,
    overlap: int = 500,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """
    Predict full-stream frame-wise probabilities p(t) using 10,000-frame chunks with 500-frame
    overlap, trimming 250 boundary frames at interior joins (> 188-frame half receptive field).
    """
    model.eval()
    total_len = int(norm_bands.shape[1])
    if total_len <= chunk_size:
        x_t = torch.from_numpy(norm_bands[None, :, :]).to(device)
        return model(x_t).squeeze(0).cpu().numpy().astype(np.float32)

    stride = chunk_size - overlap
    half_trim = overlap // 2  # 250 frames > (377 - 1)//2 = 188 frames -> exact interior context
    probs_out = np.zeros(total_len, dtype=np.float32)

    starts = list(range(0, total_len - chunk_size + 1, stride))
    if starts[-1] + chunk_size < total_len:
        starts.append(total_len - chunk_size)

    for c_i, s_idx in enumerate(starts):
        e_idx = s_idx + chunk_size
        chunk = norm_bands[:, s_idx:e_idx]
        x_t = torch.from_numpy(chunk[None, :, :]).to(device)
        p_chunk = model(x_t).squeeze(0).cpu().numpy().astype(np.float32)

        write_s = 0 if c_i == 0 else half_trim
        write_e = chunk_size if c_i == len(starts) - 1 else (chunk_size - half_trim)
        # If the final tail chunk jumped less than stride, avoid overwriting already-valid center frames
        if c_i == len(starts) - 1 and c_i > 0:
            prev_valid_end = starts[c_i - 1] + (chunk_size - half_trim)
            write_s = max(half_trim, prev_valid_end - s_idx)

        probs_out[s_idx + write_s : s_idx + write_e] = p_chunk[write_s:write_e]

    return probs_out


def coarse_to_fine_decode(
    stream_id: str,
    frame_probs: np.ndarray,
    raw_bands: np.ndarray,
    prob_threshold: float = 0.50,
    merge_gap_s: float = 0.60,
    min_duration_s: float = 0.60,
    fs_hz: float = FS_HZ,
) -> list[dict]:
    """
    Coarse-to-Fine Post-Processing:
      1. Coarse Proposal: Binarize p(t) >= 0.50, merge gaps <= 0.60s, drop events < 0.60s.
      2. Fine Gradient Refinement on total energy envelope E(t) = sum_c raw_bands[c, t]:
         - t_start: first major positive peak of dE/dt (onset of fall).
         - t_end  : frame where E(t) stabilizes back to baseline + 10% dynamic span.
    """
    n_frames = len(frame_probs)
    stream_dur_s = n_frames / fs_hz

    mask = frame_probs >= prob_threshold
    pos_idx = np.flatnonzero(mask)
    if len(pos_idx) == 0:
        return []

    cuts = np.flatnonzero(np.diff(pos_idx) > 1)
    starts = np.concatenate([[pos_idx[0]], pos_idx[cuts + 1]])
    ends = np.concatenate([pos_idx[cuts] + 1, [pos_idx[-1] + 1]])

    max_gap_frames = int(round(merge_gap_s * fs_hz))
    merged: list[list[int]] = [[int(starts[0]), int(ends[0])]]
    for s_f, e_f in zip(starts[1:], ends[1:]):
        if int(s_f) - merged[-1][1] <= max_gap_frames:
            merged[-1][1] = int(e_f)
        else:
            merged.append([int(s_f), int(e_f)])

    min_frames = int(round(min_duration_s * fs_hz))
    coarse_intervals = [(s_f, e_f) for s_f, e_f in merged if (e_f - s_f) >= min_frames]
    if not coarse_intervals:
        return []

    # Total energy envelope across all 12 spatial bands + light Gaussian smoothing
    total_energy = np.sum(raw_bands.astype(np.float64), axis=0)
    smooth_energy = gaussian_filter1d(total_energy, sigma=2.5, mode="nearest")
    grad_energy = np.gradient(smooth_energy, 1.0 / fs_hz)

    events_out: list[dict] = []
    for s_coarse, e_coarse in coarse_intervals:
        # Local quiet baseline from flanking regions outside coarse event
        flank_l0 = max(0, s_coarse - int(3.0 * fs_hz))
        flank_l1 = max(0, s_coarse - int(0.3 * fs_hz))
        flank_r0 = min(n_frames, e_coarse + int(0.3 * fs_hz))
        flank_r1 = min(n_frames, e_coarse + int(3.0 * fs_hz))
        flanks = []
        if flank_l1 > flank_l0:
            flanks.append(smooth_energy[flank_l0:flank_l1])
        if flank_r1 > flank_r0:
            flanks.append(smooth_energy[flank_r0:flank_r1])
        if flanks:
            baseline = float(np.percentile(np.concatenate(flanks), 25.0))
        else:
            baseline = float(np.percentile(smooth_energy, 25.0))

        # Anchor main impact peak inside high-probability core so pre-fall walking at boundary is ignored
        ev_probs = frame_probs[s_coarse:e_coarse]
        core_thresh = max(prob_threshold, 0.75 * float(np.max(ev_probs)))
        core_rel = np.flatnonzero(ev_probs >= core_thresh)
        if len(core_rel) > 0:
            c_s = s_coarse + int(core_rel[0])
            c_e = s_coarse + int(core_rel[-1]) + 1
            impact_idx = c_s + int(np.argmax(smooth_energy[c_s:c_e]))
        else:
            impact_idx = s_coarse + int(np.argmax(smooth_energy[s_coarse:e_coarse]))

        peak_e = float(smooth_energy[impact_idx])
        dynamic_span = max(peak_e - baseline, 1e-8)

        # 1. t_start: first major positive peak of dE/dt in onset zone
        search_s = max(0, s_coarse - int(0.20 * fs_hz))
        search_e = max(search_s + 3, impact_idx + 1)
        onset_grad = grad_energy[search_s:search_e]
        max_pos_grad = float(np.max(onset_grad)) if len(onset_grad) > 0 else 0.0

        s_fine = s_coarse
        if max_pos_grad > 1e-8 and len(onset_grad) >= 3:
            peaks, _ = find_peaks(onset_grad, height=0.25 * max_pos_grad, distance=4)
            # Filter peaks where energy reaches at least baseline + 5% dynamic span shortly after
            valid_peaks = [
                int(p)
                for p in peaks
                if float(np.max(smooth_energy[search_s + int(p) : min(n_frames, search_s + int(p) + 8)]))
                >= baseline + 0.05 * dynamic_span
            ]
            if valid_peaks:
                cand_s = search_s + valid_peaks[0]
                # Guard against over-trimming slow-onset falls
                s_fine = int(np.clip(cand_s, max(0, s_coarse - int(0.20 * fs_hz)), min(impact_idx, s_coarse + int(0.55 * fs_hz))))

        # 2. t_end: frame where E(t) stabilizes back to baseline + 10% dynamic span
        stab_thresh = baseline + 0.10 * dynamic_span
        post_end_limit = min(n_frames, e_coarse + int(0.30 * fs_hz))
        post_energy = smooth_energy[impact_idx:post_end_limit]

        e_fine = e_coarse
        if len(post_energy) > 0:
            above = np.flatnonzero(post_energy > stab_thresh)
            if len(above) > 0:
                cand_e = impact_idx + int(above[-1]) + 1
            else:
                cand_e = impact_idx + 1
            e_fine = int(
                np.clip(
                    cand_e,
                    max(s_fine + min_frames, e_coarse - int(0.65 * fs_hz)),
                    min(n_frames, e_coarse + int(0.26 * fs_hz)),
                )
            )

        if e_fine - s_fine < min_frames:
            e_fine = min(n_frames, s_fine + min_frames)
            if e_fine - s_fine < min_frames:
                s_fine = max(0, e_fine - min_frames)

        t_start = round(max(0.0, s_fine / fs_hz), 3)
        t_end = round(min(stream_dur_s, e_fine / fs_hz), 3)
        if t_end <= t_start:
            t_end = min(stream_dur_s, round(t_start + min_duration_s, 3))
        if t_end <= t_start:
            continue

        seg_probs = frame_probs[s_coarse:e_coarse]
        score = float(0.75 * np.max(seg_probs) + 0.25 * np.mean(seg_probs))

        events_out.append(
            {
                "stream_id": stream_id,
                "t_start": t_start,
                "t_end": t_end,
                "score": round(score, 6),
                "t_start_coarse": round(s_coarse / fs_hz, 3),
                "t_end_coarse": round(e_coarse / fs_hz, 3),
            }
        )

    return events_out


def evaluate_on_validation(
    model: FrameWise1DTCN,
    val_streams: list[PreparedStream],
    gt_events_val: list[dict],
    device: torch.device | str = "cpu",
    prob_threshold: float = 0.50,
) -> tuple[dict, dict]:
    """Run full-stream inference + Coarse & Fine evaluation on the 5 real Val streams."""
    durations = {st.stream_id: st.duration_s for st in val_streams}
    preds_fine: list[dict] = []
    preds_coarse: list[dict] = []

    for st in val_streams:
        probs = infer_stream_probabilities(model, st.norm_bands, device=device)
        decoded = coarse_to_fine_decode(
            stream_id=st.stream_id,
            frame_probs=probs,
            raw_bands=st.raw_bands,
            prob_threshold=prob_threshold,
        )
        for item in decoded:
            preds_fine.append(
                {
                    "stream_id": item["stream_id"],
                    "t_start": item["t_start"],
                    "t_end": item["t_end"],
                    "score": item["score"],
                    "t_alarm": item["t_start"],
                }
            )
            preds_coarse.append(
                {
                    "stream_id": item["stream_id"],
                    "t_start": item["t_start_coarse"],
                    "t_end": item["t_end_coarse"],
                    "score": item["score"],
                    "t_alarm": item["t_start_coarse"],
                }
            )

    report_fine = official_evaluate(gt_events_val, preds_fine, durations)
    report_coarse = official_evaluate(gt_events_val, preds_coarse, durations)
    return report_fine, report_coarse


# ============================================================================
# TRAINING LOOP ENTRYPOINT
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Physics-Aware 12-Channel FrameWise1DTCN on 13 Real Streams")
    parser.add_argument("--epochs", type=int, default=45, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size of 10,000-frame chunks")
    parser.add_argument("--chunk-size", type=int, default=10_000, help="Chunk length in frames (200s at 50Hz)")
    parser.add_argument("--overlap", type=int, default=500, help="Chunk overlap in frames (10s at 50Hz)")
    parser.add_argument("--hidden-channels", type=int, default=32, help="Hidden channels in FrameWise1DTCN")
    parser.add_argument("--lr", type=float, default=2.5e-3, help="Peak AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--alpha", type=float, default=0.85, help="Focal loss alpha for minority Fall class")
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal loss focusing parameter gamma")
    parser.add_argument("--splice-mult", type=float, default=5.0, help="Hard-negative multiplier at 0.2s splices")
    parser.add_argument("--tv-weight", type=float, default=0.08, help="Total Variation smoothness loss weight")
    parser.add_argument("--seed", type=int, default=2027, help="Random seed")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT_PATH, help="Output model checkpoint path")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Cache dir for 12-band features")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 88)
    print("ABC2027 FINAL PHYSICS-AWARE 12-CHANNEL FrameWise1DTCN TRAINING PIPELINE")
    print("=" * 88)
    print(f"Device               : {device}")
    print(f"Spatial Bands (x3 R) : {list(zip(SPATIAL_BAND_NAMES, SPATIAL_BANDS))} -> {IN_CHANNELS} channels")

    events_csv = TRAIN_PKG_DIR / "streams_events.csv"
    comp_csv = TRAIN_PKG_DIR / "streams_composition.csv"
    events_df = pd.read_csv(events_csv)
    composition_df = pd.read_csv(comp_csv)
    gt_events_val = load_ground_truth(events_csv, ignore_flagged=True)

    print("\n[1/4] Preparing 13 Real Training Streams (train_00..train_12)...")
    train_streams = load_prepared_streams(
        split="train",
        stream_indices=list(range(13)),
        events_df=events_df,
        composition_df=composition_df,
        cache_dir=args.cache_dir,
    )

    print("\n[2/4] Preparing 5 Real Validation Streams (val_00..val_04)...")
    val_streams = load_prepared_streams(
        split="val",
        stream_indices=list(range(5)),
        events_df=events_df,
        composition_df=composition_df,
        cache_dir=args.cache_dir,
    )

    train_dataset = ChunkedStreamDataset(
        streams=train_streams,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        augment=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    model = FrameWise1DTCN(
        in_channels=IN_CHANNELS,
        hidden_channels=args.hidden_channels,
        dilations=(1, 2, 4, 8, 16),
        head_bias_init=-4.4,
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n[3/4] Model & Physics-Aware Loss Configuration")
    print(
        f"  Architecture       : FrameWise1DTCN(in_channels={model.in_channels}, "
        f"hidden={model.hidden_channels}, dilations={model.dilations})"
    )
    print(
        f"  Receptive Field    : {model.receptive_field_frames} frames = "
        f"{model.receptive_field_seconds:.2f}s at {FS_HZ:.0f} Hz"
    )
    print(f"  Trainable Params   : {param_count:,}")
    print(f"  Chunk Loader       : {len(train_dataset)} chunks ({args.chunk_size} frames, {args.overlap} overlap)")
    print(
        f"  Composite Loss     : Focal(alpha={args.alpha}, gamma={args.gamma}) + "
        f"SplicePenalty(mult={args.splice_mult}, p>0.3) + TV(w={args.tv_weight})"
    )

    criterion = PhysicsAwareCompositeLoss(
        alpha=args.alpha,
        gamma=args.gamma,
        splice_prob_thresh=0.30,
        splice_multiplier=args.splice_mult,
        tv_weight=args.tv_weight,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-4)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_rank_f1 = -1.0
    best_epoch = 0
    best_report_summary: dict = {}
    history: list[dict] = []

    print("\n[4/4] Training over Real Streams...")
    print("-" * 108)
    print(
        f"{'Ep':>3s} | {'Loss':>7s} ({'Focal':>7s} {'Splice':>7s} {'TV':>6s}) | "
        f"{'ValRankF1':>9s} {'CoarseF1':>8s} | {'F1@0.1':>6s} {'F1@0.3':>6s} {'F1@0.5':>6s} | "
        f"{'P@0.3':>6s} {'R@0.3':>6s} {'FA/h':>5s} {'Time':>5s}"
    )
    print("-" * 108)

    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.perf_counter()
        model.train()
        sum_tot = 0.0
        sum_foc = 0.0
        sum_spl = 0.0
        sum_tv = 0.0
        n_batches = 0

        for x_b, y_b, vm_b, sm_b in train_loader:
            x_b = x_b.to(device)
            y_b = y_b.to(device)
            vm_b = vm_b.to(device)
            sm_b = sm_b.to(device)

            optimizer.zero_grad()
            logits = model.forward_logits(x_b)
            loss, stats = criterion(logits, y_b, vm_b, sm_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            sum_tot += stats["total"]
            sum_foc += stats["focal"]
            sum_spl += stats["splice"]
            sum_tv += stats["tv"]
            n_batches += 1

        scheduler.step()

        # Evaluate on the 5 real validation streams every epoch (after initial warmup epoch 1)
        report_fine, report_coarse = evaluate_on_validation(
            model=model,
            val_streams=val_streams,
            gt_events_val=gt_events_val,
            device=device,
            prob_threshold=0.50,
        )
        rank_f1_fine = float(report_fine["ranking_score"]["f1_mean"])
        rank_f1_coarse = float(report_coarse["ranking_score"]["f1_mean"])
        f1_01 = float(report_fine["by_theta"]["0.1"]["f1"])
        f1_03 = float(report_fine["by_theta"]["0.3"]["f1"])
        f1_05 = float(report_fine["by_theta"]["0.5"]["f1"])
        f1_07 = float(report_fine["by_theta"]["0.7"]["f1"])
        prec_03 = float(report_fine["primary"]["precision"])
        rec_03 = float(report_fine["primary"]["recall"])
        fah_03 = float(report_fine["primary"]["false_alarms_per_hour"] or 0.0)
        ep_dt = time.perf_counter() - ep_t0

        is_best = rank_f1_fine > best_rank_f1
        marker = " *" if is_best else ""
        print(
            f"{epoch:3d} | {sum_tot/n_batches:7.4f} ({sum_foc/n_batches:7.4f} "
            f"{sum_spl/n_batches:7.4f} {sum_tv/n_batches:6.4f}) | "
            f"{rank_f1_fine:9.4f} {rank_f1_coarse:8.4f} | "
            f"{f1_01:6.4f} {f1_03:6.4f} {f1_05:6.4f} | "
            f"{prec_03:6.3f} {rec_03:6.3f} {fah_03:5.2f} {ep_dt:4.1f}s{marker}"
        )

        row_record = {
            "epoch": epoch,
            "loss_total": round(sum_tot / n_batches, 6),
            "loss_focal": round(sum_foc / n_batches, 6),
            "loss_splice": round(sum_spl / n_batches, 6),
            "loss_tv": round(sum_tv / n_batches, 6),
            "val_ranking_f1_fine": round(rank_f1_fine, 6),
            "val_ranking_f1_coarse": round(rank_f1_coarse, 6),
            "val_f1_0.1": round(f1_01, 6),
            "val_f1_0.3": round(f1_03, 6),
            "val_f1_0.5": round(f1_05, 6),
            "val_f1_0.7": round(f1_07, 6),
            "val_precision_0.3": round(prec_03, 6),
            "val_recall_0.3": round(rec_03, 6),
            "val_fa_per_hour_0.3": round(fah_03, 4),
            "val_start_err_median_0.3": report_fine["primary"]["start_error_median_abs"],
        }
        history.append(row_record)

        if is_best:
            best_rank_f1 = rank_f1_fine
            best_epoch = epoch
            best_report_summary = {
                "best_epoch": epoch,
                "fine_report": report_fine,
                "coarse_report": report_coarse,
            }
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "in_channels": IN_CHANNELS,
                        "hidden_channels": args.hidden_channels,
                        "dilations": [1, 2, 4, 8, 16],
                        "receptive_field_frames": model.receptive_field_frames,
                        "receptive_field_seconds": model.receptive_field_seconds,
                        "spatial_bands": [list(b) for b in SPATIAL_BANDS],
                    },
                    "val_ranking_f1": rank_f1_fine,
                    "val_report_fine": report_fine,
                },
                args.checkpoint,
            )

    metrics_out_path = args.checkpoint.with_name("training_metrics.json")
    metrics_out_path.write_text(
        json.dumps({"best": best_report_summary, "history": history}, indent=2),
        encoding="utf-8",
    )

    print("-" * 108)
    print(f"Training Complete! Best Validation Ranking F1 = {best_rank_f1:.4f} (Epoch {best_epoch})")
    print(f"Saved best model weights to : {args.checkpoint}")
    print(f"Saved training diagnostics  : {metrics_out_path}")


if __name__ == "__main__":
    main()
