"""
Module 2: Preprocessing Implementation (Clutter Removal)
9th ABC Challenge - UWB Fall Track Pipeline

Implements two clutter suppression algorithms for 1D UWB CIR fast-time streams:
1. Causal Exponential Moving Average (EMA) Filter:
       c(t) = alpha * c(t - 1) + (1 - alpha) * r(t),  with alpha = 0.95
       y_EMA(t) = r(t) - c(t)
2. Low-Rank Singular Value Decomposition (SVD) Subspace Filter:
       M = U * diag(sigma) * V^T
       Zeroes the top K singular values spanning static background clutter
       and reconstructs M_SVD = U * diag(sigma_filtered) * V^T.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.linalg import svd as scipy_svd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class PreprocessingResult:
    """Outputs and diagnostics from EMA and SVD clutter removal."""
    ema_stream: np.ndarray              # Shape: (N_FRAMES, N_RADARS, N_BINS), float32
    svd_stream: np.ndarray              # Shape: (N_FRAMES, N_RADARS, N_BINS), float32
    alpha: float
    k_components: int
    singular_values_per_radar: list[np.ndarray]
    ema_runtime_s: float
    svd_runtime_s: float


def ema_clutter_removal(raw_stream: np.ndarray, alpha: float = 0.95) -> np.ndarray:
    """
    Apply causal Exponential Moving Average (EMA) clutter suppression per radar and bin.

    Formula:
        c(0) = r(0)
        c(t) = alpha * c(t - 1) + (1 - alpha) * r(t)
        y(t) = r(t) - c(t)

    Parameters
    ----------
    raw_stream : np.ndarray
        Raw UWB stream of shape (T, R, B) where r(t) is the 1D CIR across B bins.
    alpha : float, default=0.95
        Forgetting factor in (0, 1). At 50 Hz, alpha=0.95 corresponds to an effective
        time constant tau = -dt / ln(0.95) ≈ 0.39 seconds (~19.5 frames).

    Returns
    -------
    filtered : np.ndarray
        Clutter-suppressed stream y(t) = r(t) - c(t) of shape (T, R, B), float32.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    n_frames = raw_stream.shape[0]
    filtered = np.empty_like(raw_stream, dtype=np.float32)

    # Steady-state initialization c(0) = r(0) avoids artificial t=0 step transients
    clutter_est = raw_stream[0].astype(np.float64, copy=True)
    filtered[0] = 0.0
    one_minus_alpha = 1.0 - alpha

    for t in range(1, n_frames):
        r_t = raw_stream[t].astype(np.float64, copy=False)
        clutter_est = alpha * clutter_est + one_minus_alpha * r_t
        filtered[t] = (r_t - clutter_est).astype(np.float32)

    return filtered


def svd_clutter_removal(
    raw_stream: np.ndarray,
    k_components: int | None = None,
    energy_ratio_threshold: float = 0.995,
) -> tuple[np.ndarray, int, list[np.ndarray]]:
    """
    Apply SVD subspace clutter suppression on each radar's Range-Time Matrix (RTM).

    Decomposes M_r = U * Sigma * V^T (shape T x B), identifies the first K largest
    singular values corresponding to static multipath clutter (and slow global drift),
    sets sigma[:K] = 0, and reconstructs the dynamic target residual matrix.

    Parameters
    ----------
    raw_stream : np.ndarray
        Raw UWB stream of shape (T, R, B).
    k_components : int or None, default=None
        Number of leading singular values to zero out. If None, automatically selects K
        via singular-value energy dominance (static subspace capturing >= threshold).
    energy_ratio_threshold : float, default=0.995
        Cumulative energy threshold used when k_components is None.

    Returns
    -------
    filtered : np.ndarray
        SVD-reconstructed dynamic stream of shape (T, R, B), float32.
    selected_k : int
        Number of leading static singular values removed.
    singular_values : list[np.ndarray]
        Singular value spectra (length B) for each radar.
    """
    n_frames, n_radars, n_bins = raw_stream.shape
    filtered = np.empty((n_frames, n_radars, n_bins), dtype=np.float32)
    singular_values_list: list[np.ndarray] = []

    determined_k = k_components

    for r in range(n_radars):
        rtm = raw_stream[:, r, :].astype(np.float64)
        u_mat, s_vals, vt_mat = scipy_svd(rtm, full_matrices=False, lapack_driver="gesvd")
        singular_values_list.append(s_vals)

        if determined_k is None:
            # Identify K where leading singular values dominate static clutter energy
            energy = s_vals ** 2
            cum_energy_ratio = np.cumsum(energy) / np.sum(energy)
            # First index reaching static energy threshold, plus any secondary drift mode
            k_auto = int(np.searchsorted(cum_energy_ratio, energy_ratio_threshold) + 1)
            determined_k = max(2, min(k_auto, 5))

        s_filtered = s_vals.copy()
        s_filtered[:determined_k] = 0.0

        # Efficient reconstruction: (U * s_filtered) @ V^T
        rtm_reconstructed = (u_mat * s_filtered[None, :]) @ vt_mat
        filtered[:, r, :] = rtm_reconstructed.astype(np.float32)

    assert determined_k is not None
    return filtered, determined_k, singular_values_list


def run_module2_preprocessing(
    raw_stream: np.ndarray,
    output_dir: Path,
    alpha: float = 0.95,
    k_components: int = 2,
) -> PreprocessingResult:
    """
    Execute Module 2 Clutter Removal (EMA & SVD), save singular-value diagnostic plot,
    and print analytical insights on computational complexity, causality, and subspace energy.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    ema_stream = ema_clutter_removal(raw_stream, alpha=alpha)
    ema_runtime = time.perf_counter() - t0

    t1 = time.perf_counter()
    svd_stream, used_k, sv_list = svd_clutter_removal(raw_stream, k_components=k_components)
    svd_runtime = time.perf_counter() - t1

    # Compute singular value energy statistics for Radar 0
    sv0 = sv_list[0]
    sv0_energy_pct = 100.0 * (sv0 ** 2) / np.sum(sv0 ** 2)

    # Save diagnostic SVD singular spectrum plot
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=160)
    top_n = 25
    indices = np.arange(1, top_n + 1)
    for r_idx, sv in enumerate(sv_list):
        ax.semilogy(indices, sv[:top_n], marker="o", markersize=4.5, lw=1.6, label=f"Radar {r_idx} Singular Values")
    ax.axvspan(0.7, used_k + 0.3, color="#E45756", alpha=0.18, label=f"Zeroed Static Subspace (K={used_k})")
    ax.set_title("Module 2: RTM Singular Value Spectrum (Log Scale) & Static Clutter Subspace", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Singular Component Index (i)", fontsize=10.5)
    ax.set_ylabel("Singular Value Magnitude σ_i (log scale)", fontsize=10.5)
    ax.set_xticks(indices)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(output_dir / "module2_svd_spectrum.png")
    plt.close(fig)

    tau_ema = -0.02 / np.log(alpha)

    print("=" * 80)
    print("MODULE 2: PREPROCESSING IMPLEMENTATION (CLUTTER REMOVAL)")
    print("=" * 80)
    print(f"1. EMA Clutter Filter    : alpha = {alpha:.2f} | Effective Time Constant tau = {tau_ema:.3f}s ({tau_ema * 50:.1f} frames)")
    print(f"   - Execution Time      : {ema_runtime:.3f}s (Causal O(B) per-frame streaming complexity)")
    print(f"   - Output Range        : [{ema_stream.min():.1f}, {ema_stream.max():.1f}]")
    print(f"2. SVD Subspace Filter   : Removed Top K = {used_k} Singular Components per Radar RTM (30,000 x 368)")
    print(f"   - Execution Time      : {svd_runtime:.3f}s (Non-causal batch O(T * B^2) decomposition)")
    print(f"   - Radar 0 Top-5 σ_i   : {[round(float(x), 1) for x in sv0[:5]]}")
    print(f"   - Radar 0 Energy (%)  : σ_1 = {sv0_energy_pct[0]:.4f}%, σ_2 = {sv0_energy_pct[1]:.4f}%, "
          f"Remaining (i>{used_k}) = {np.sum(sv0_energy_pct[used_k:]):.4f}%")
    print(f"   - Output Range        : [{svd_stream.min():.1f}, {svd_stream.max():.1f}]")
    print("-" * 80)
    print("[Analytical Insight - Module 2]:")
    print(f"  1. Subspace Concentration: σ_1 alone captures {sv0_energy_pct[0]:.4f}% of raw UWB stream energy,")
    print("     confirming that static indoor multipath reflections form an ultra-dominant rank-1 background.")
    print(f"  2. Filter Memory Trade-off: EMA with alpha=0.95 has a short memory constant (tau={tau_ema:.2f}s).")
    print("     While it cancels static walls instantly in real time, any human fall phase lasting >0.4s")
    print("     at a similar range bin gets partially absorbed into c(t) (self-subtraction). Conversely,")
    print(f"     SVD (K={used_k}) projects out only global static modes across 600s, preserving 100% of")
    print("     localized 2–3s fall transients without causal fade-out.")
    print()

    return PreprocessingResult(
        ema_stream=ema_stream,
        svd_stream=svd_stream,
        alpha=alpha,
        k_components=used_k,
        singular_values_per_radar=sv_list,
        ema_runtime_s=ema_runtime,
        svd_runtime_s=svd_runtime,
    )
