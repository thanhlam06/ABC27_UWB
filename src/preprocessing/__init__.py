"""
Dual-Path Signal Conditioning for UWB Fall Detection:
1. Causal EMA Background Subtraction (`tau_s = 3.0s`) + Spatial Range Slicing `[0:100]`
   + Stream-level Per-Radar Z-Score Normalization (Deep Learning Path).
2. Local Block Overlapped Gram-Matrix SVD (`K = 2`) Zero-Lag Kinetic Energy Envelope
   `E(t)` & 1st Derivative `dE/dt` (Physical Kinematics Path).
"""

from .clutter_removal import (
    SPATIAL_BAND_NAMES,
    SPATIAL_BANDS,
    causal_ema_background_subtraction,
    compute_smoothed_kinematics,
    compute_svd_energy_envelope,
    extract_12channel_spatial_bands,
    preprocess_deep_branch,
)

__all__ = [
    "SPATIAL_BANDS",
    "SPATIAL_BAND_NAMES",
    "causal_ema_background_subtraction",
    "preprocess_deep_branch",
    "compute_svd_energy_envelope",
    "compute_smoothed_kinematics",
    "extract_12channel_spatial_bands",
]
