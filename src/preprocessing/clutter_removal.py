"""
Dual-Path Signal Conditioning (Causal EMA & Overlapped Block Gram-Matrix SVD)
9th ABC Challenge — UWB Fall Detection Track

Resolves the fundamental physical trade-off between causal real-time filtering and
zero-phase-lag boundary localization:
- Path 1 (Deep Learning Feature Conditioning):
    Causal Exponential Moving Average (EMA, tau = 3.0s, alpha = 0.99336) with
    `scipy.signal.lfilter_zi` steady-state initialization, followed by active range
    bin slicing [0 : 100] (0 - 5.3m) and stream-level per-radar Z-score normalization.
- Path 2 (Zero-Lag Physical Kinematics):
    Overlapped 8.0s (400-frame) Hann-windowed local block Gram-Matrix SVD (K = 2)
    computing G_r = X_r^T @ X_r via `np.linalg.eigh` to strip both room static
    multipath reflections (v_1) and stationary post-fall body clutter (v_2) without
    any causal EMA tail lag.
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage
from scipy.signal import lfilter, lfilter_zi

from src.data.dataset import FS_HZ, N_ACTIVE_BINS, ensure_radar_first_layout

# 4 Physical Range Bands per Radar -> 12 Spatial Channels (for 1D-TCN / Cascade models)
SPATIAL_BANDS: tuple[tuple[int, int], ...] = (
    (0, 66),     # Near field (0.0m - 3.5m)
    (75, 181),   # Mid-static corridor
    (183, 226),  # Primary Mattress / Fall Zone
    (260, 368),  # Far wall boundary
)
SPATIAL_BAND_NAMES: tuple[str, ...] = (
    "Near_0_65",
    "Static_75_180",
    "Mat_183_225",
    "Far_260_367",
)


def causal_ema_background_subtraction(
    cir_raw: np.ndarray,
    tau_s: float = 3.0,
    fs_hz: float = FS_HZ,
) -> np.ndarray:
    """
    Apply causal Exponential Moving Average (EMA) clutter suppression per radar and bin.

    Formula:
        c(t) = alpha * c(t - 1) + (1 - alpha) * r(t),  alpha = exp(-1 / (fs_hz * tau_s))
        y_clean(t) = r(t) - c(t)
    Initialized at steady state c(0) = r(0) via `scipy.signal.lfilter_zi`.
    """
    cir = ensure_radar_first_layout(cir_raw)
    alpha = float(np.exp(-1.0 / (fs_hz * tau_s)))
    n_radars, n_frames, n_bins = cir.shape

    flat = np.ascontiguousarray(np.moveaxis(cir, 1, -1), dtype=np.float64).reshape(
        n_radars * n_bins, n_frames
    )
    numerator = [1.0 - alpha]
    denominator = [1.0, -alpha]
    steady = lfilter_zi(numerator, denominator)

    residual = np.empty_like(flat)
    for idx in range(flat.shape[0]):
        bg_est, _ = lfilter(numerator, denominator, flat[idx], zi=flat[idx, 0] * steady)
        residual[idx] = flat[idx] - bg_est

    restored = np.moveaxis(residual.reshape(n_radars, n_bins, n_frames), -1, 1)
    return np.ascontiguousarray(restored, dtype=np.float32)


def preprocess_deep_branch(
    cir_clean: np.ndarray,
    n_bins: int = N_ACTIVE_BINS,
) -> np.ndarray:
    """
    Slice active range bins [0 : n_bins] from EMA-cleaned CIR and apply stream-level
    per-radar Z-score normalization:
        A_norm = (|y_clean[:, :, :n_bins]| - mu_r) / (sigma_r + 1e-6)
    Returns tensor of shape (3, T, n_bins) in float32.
    """
    amp = np.abs(cir_clean[:, :, :n_bins])
    mean = amp.mean(axis=(1, 2), keepdims=True)
    std = amp.std(axis=(1, 2), keepdims=True) + 1e-6
    return ((amp - mean) / std).astype(np.float32)


def compute_svd_energy_envelope(
    cir_raw: np.ndarray,
    k_components: int = 2,
    block_frames: int = 400,
    block_hop: int = 200,
) -> np.ndarray:
    """
    Calculate zero-phase-lag total kinetic energy envelope E(t) via Overlapped Local Block
    Gram-Matrix SVD (K = 2) clutter suppression.

    For continuous 30-minute streams (~88,500 frames), overlapped 8.0s (400-frame)
    Hann-weighted blocks subtract static room reflections (v_1) and stationary post-fall
    body clutter (v_2) via top-K eigenvectors of G_r = X_r^T @ X_r without causal EMA lag.
    """
    cir_rf = ensure_radar_first_layout(cir_raw)
    n_radars, n_frames, _ = cir_rf.shape

    if n_frames <= block_frames:
        starts = [0]
        win_template = np.ones(n_frames, dtype=np.float64)
    else:
        starts = list(range(0, n_frames - block_frames + 1, block_hop))
        if starts[-1] + block_frames < n_frames:
            starts.append(n_frames - block_frames)
        win_template = np.hanning(block_frames) + 0.1

    energy_acc = np.zeros(n_frames, dtype=np.float64)
    weight_acc = np.zeros(n_frames, dtype=np.float64)

    for s in starts:
        e = min(n_frames, s + block_frames)
        seg_len = e - s
        block_energy = np.zeros(seg_len, dtype=np.float64)

        for r in range(n_radars):
            rtm = cir_rf[r, s:e].astype(np.float64, copy=False)
            k_eff = min(k_components, min(rtm.shape) - 1)
            if seg_len >= 4 and k_eff > 0:
                gram = rtm.T @ rtm
                _, eigvecs = np.linalg.eigh(gram)
                top_k_vecs = eigvecs[:, -k_eff:]
                rtm_clean = rtm - (rtm @ top_k_vecs) @ top_k_vecs.T
            else:
                rtm_clean = rtm - rtm.mean(axis=0, keepdims=True)
            block_energy += np.sqrt(np.sum(rtm_clean * rtm_clean, axis=1))

        w = win_template[:seg_len]
        energy_acc[s:e] += block_energy * w
        weight_acc[s:e] += w

    return (energy_acc / np.maximum(weight_acc, 1e-8)).astype(np.float32)


def compute_smoothed_kinematics(
    energy_envelope: np.ndarray,
    fs_hz: float = FS_HZ,
    sigma_s: float = 0.14,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Smooth the zero-lag SVD kinetic energy envelope E(t) using a 1D Gaussian kernel
    (sigma = 0.14s) and compute its first temporal derivative dE/dt.
    """
    smooth_e = scipy.ndimage.gaussian_filter1d(
        energy_envelope.astype(np.float64),
        sigma=sigma_s * fs_hz,
        mode="nearest",
    )
    dE_dt = np.gradient(smooth_e, 1.0 / fs_hz)
    return smooth_e, dE_dt


def extract_12channel_spatial_bands(
    cir_raw: np.ndarray,
    tau_s: float = 3.0,
    fs_hz: float = FS_HZ,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract 12-channel spatial range-band L2 envelopes (4 physical bands x 3 radars)
    and apply per-channel robust IQR normalization `(e(t) - median) / (P75 - P25 + 1e-6)`.
    """
    residual = causal_ema_background_subtraction(cir_raw, tau_s=tau_s, fs_hz=fs_hz)
    n_radars, n_frames, _ = residual.shape
    n_bands = len(SPATIAL_BANDS)
    raw_bands = np.empty((n_radars * n_bands, n_frames), dtype=np.float32)

    ch = 0
    for r in range(n_radars):
        for b_start, b_end in SPATIAL_BANDS:
            sub = residual[r, :, b_start:b_end].astype(np.float64, copy=False)
            raw_bands[ch] = np.sqrt(np.mean(sub * sub, axis=1)).astype(np.float32)
            ch += 1

    med = np.median(raw_bands, axis=1, keepdims=True)
    q75 = np.percentile(raw_bands, 75.0, axis=1, keepdims=True)
    q25 = np.percentile(raw_bands, 25.0, axis=1, keepdims=True)
    iqr = np.maximum(q75 - q25, 1e-6)
    norm_bands = np.clip((raw_bands - med) / iqr, -5.0, 25.0).astype(np.float32)
    return norm_bands, raw_bands
