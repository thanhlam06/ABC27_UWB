"""
Module 1: Data Representation, SVD Clutter Removal & 1D Temporal Energy Envelope
9th ABC Challenge - UWB Fall Track (Temporal Action Localization Pipeline)

1. Input: Continuous UWB stream of shape (30000, 3, 368) representing 10 minutes @ 50 Hz.
2. SVD Clutter Removal: Removes the top K=2 singular components (static clutter + slow drift)
   from the Range-Time Matrix (RTM) of each radar.
3. 1D Temporal Energy Envelope Extraction: Computes the L2 Norm across the 368 spatial bins
   for every frame and radar, producing an envelope tensor of shape (3, 30000).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.linalg import svd as scipy_svd
from scipy.ndimage import uniform_filter1d

# Reuse the realistic 30,000-frame UWB stream generator from the baseline pipeline
BASELINE_DIR = Path(__file__).resolve().parent.parent / "uwb_fall_pipeline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from module1_eda import generate_mock_uwb_stream, UWBStreamDataset, N_FRAMES, N_RADARS, N_BINS, FS_HZ


@dataclass
class SVDEnvelopeData:
    """Container for Module 1 SVD-filtered stream and (3, 30000) 1D Energy Envelope."""
    raw_stream: np.ndarray          # Shape: (30000, 3, 368)
    svd_stream: np.ndarray          # Shape: (30000, 3, 368)
    energy_envelope: np.ndarray     # Shape: (3, 30000) — L2 norm per radar across 368 bins
    norm_envelope: np.ndarray       # Shape: (3, 30000) — Z-score / robust normalized for 1D-TCN
    fused_envelope: np.ndarray      # Shape: (30000,) — Smooth multi-radar mean L2 envelope E(t)
    frame_targets: np.ndarray       # Shape: (30000,) — Binary ground truth y(t) in {0.0, 1.0}
    events_df: object               # Ground-truth DataFrame of Falls and ADLs
    singular_values: list[np.ndarray]


def svd_remove_top_k(raw_stream: np.ndarray, k_components: int = 2) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Remove top K=2 singular components (static clutter subspace) from each radar's RTM.

    Parameters
    ----------
    raw_stream : np.ndarray
        Input stream of shape (T=30000, R=3, B=368).
    k_components : int, default=2
        Number of leading singular values to zero out.

    Returns
    -------
    svd_stream : np.ndarray
        Clutter-suppressed stream of shape (30000, 3, 368), float32.
    sv_list : list[np.ndarray]
        Singular values per radar.
    """
    n_frames, n_radars, n_bins = raw_stream.shape
    svd_stream = np.empty((n_frames, n_radars, n_bins), dtype=np.float32)
    sv_list: list[np.ndarray] = []

    for r in range(n_radars):
        rtm = raw_stream[:, r, :].astype(np.float64)
        u_mat, s_vals, vt_mat = scipy_svd(rtm, full_matrices=False, lapack_driver="gesvd")
        sv_list.append(s_vals)

        s_clean = s_vals.copy()
        s_clean[:k_components] = 0.0
        svd_stream[:, r, :] = ((u_mat * s_clean[None, :]) @ vt_mat).astype(np.float32)

    return svd_stream, sv_list


def extract_1d_energy_envelope(svd_stream: np.ndarray, mode: str = "l2") -> np.ndarray:
    """
    Extract 1D Temporal Energy Envelope across the 368 spatial bins for every frame and radar.

    Parameters
    ----------
    svd_stream : np.ndarray
        Clutter-removed stream of shape (30000, 3, 368).
    mode : {'l2', 'max'}, default='l2'
        Spatial aggregation across the 368 fast-time bins.

    Returns
    -------
    envelope : np.ndarray
        1D Temporal Energy Envelope of shape (3, 30000), float32.
    """
    x = svd_stream.astype(np.float64, copy=False)
    if mode == "l2":
        # L2 Norm across axis=2 (368 bins) -> shape (30000, 3), then transpose to (3, 30000)
        env_tr = np.linalg.norm(x, ord=2, axis=2)
    elif mode == "max":
        env_tr = np.max(np.abs(x), axis=2)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return np.ascontiguousarray(env_tr.T, dtype=np.float32)


def run_module1_svd_representation(seed: int = 42, k_components: int = 2) -> SVDEnvelopeData:
    """Execute Module 1: synthesize (30000, 3, 368) stream, apply SVD(K=2), extract (3, 30000) envelope."""
    ds: UWBStreamDataset = generate_mock_uwb_stream(seed=seed)
    svd_stream, sv_list = svd_remove_top_k(ds.raw_stream, k_components=k_components)
    energy_envelope = extract_1d_energy_envelope(svd_stream, mode="l2")  # Shape: (3, 30000)
    assert energy_envelope.shape == (N_RADARS, N_FRAMES), f"Expected (3, 30000), got {energy_envelope.shape}"

    # Linear robust standardization per radar channel (preserves 3x peak contrast of Falls vs ADLs)
    med = np.median(energy_envelope, axis=1, keepdims=True)
    scale = np.percentile(energy_envelope, 99.2, axis=1, keepdims=True) - med + 1e-6
    norm_envelope = np.maximum((energy_envelope - med) / scale, 0.0).astype(np.float32)

    # Smooth multi-radar fused L2 energy envelope E(t) for Module 3 gradient refinement
    raw_fused = np.mean(energy_envelope.astype(np.float64), axis=0)
    fused_envelope = uniform_filter1d(raw_fused, size=7).astype(np.float32)

    # Frame-wise binary ground truth (1.0 during Fall, 0.0 during ADL or Empty)
    frame_targets = (ds.frame_labels == 2).astype(np.float32)

    print("=" * 80)
    print("MODULE 1: DATA REPRESENTATION & SVD PREPROCESSING")
    print("=" * 80)
    print(f"Input Mock Stream Shape        : {ds.raw_stream.shape} (Frames=30000, Radars=3, Bins=368)")
    print(f"SVD Clutter Removal            : Removed Top K = {k_components} Singular Values per Radar RTM")
    print(f"1D Energy Envelope Shape (L2)  : {energy_envelope.shape} (Radars=3, Frames=30000)")
    print(f"Envelope Dynamic Range (L2)    : [{energy_envelope.min():.2f}, {energy_envelope.max():.2f}]")
    print(f"Positive 'Fall' Frames         : {int(frame_targets.sum())} / {N_FRAMES} ({100.0 * frame_targets.mean():.2f}%)")
    print()

    return SVDEnvelopeData(
        raw_stream=ds.raw_stream,
        svd_stream=svd_stream,
        energy_envelope=energy_envelope,
        norm_envelope=norm_envelope,
        fused_envelope=fused_envelope,
        frame_targets=frame_targets,
        events_df=ds.events_df,
        singular_values=sv_list,
    )
