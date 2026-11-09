"""
Automated Unit & Mathematical Verification Suite
9th ABC Challenge - UWB Fall Track Pipeline

Validates all 4 strict modules against physical, mathematical, and specification constraints.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
from scipy.signal import lfilter, lfilter_zi

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from module1_eda import generate_mock_uwb_stream, N_FRAMES, N_RADARS, N_BINS, ADC_MIN, ADC_MAX
from module2_preprocessing import ema_clutter_removal, svd_clutter_removal
from module3_evaluation import compute_scr_db, compute_empty_frame_l2_energy
from module4_ml_baseline import extract_window_features, WINDOW_FRAMES, STRIDE_FRAMES, FEATURE_NAMES


def test_module1_constraints() -> None:
    ds = generate_mock_uwb_stream(seed=42)
    assert ds.raw_stream.shape == (N_FRAMES, N_RADARS, N_BINS), f"Unexpected shape: {ds.raw_stream.shape}"
    assert ds.raw_stream.min() >= ADC_MIN and ds.raw_stream.max() <= ADC_MAX, "ADC range [0, 65535] violated"

    falls = ds.events_df[ds.events_df["event_type"] == "Fall"]
    adls = ds.events_df[ds.events_df["event_type"] == "ADL"]
    assert len(falls) == 3, f"Expected exactly 3 Fall events, got {len(falls)}"
    assert np.all((falls["duration_s"] >= 2.0) & (falls["duration_s"] <= 3.0)), "Fall durations must be 2–3s"
    assert len(adls) >= 10, "Expected realistic background ADL population"
    assert int(np.sum(ds.empty_frame_mask)) > 5000, "Expected substantial empty frames for L2 residual check"
    print("[PASS] Module 1: Stream tensor (30000, 3, 368), [0, 65535] bounds, 3 Falls (2-3s), & ADLs verified.")


def test_module2_math_exactness() -> None:
    rng = np.random.default_rng(7)
    small_stream = rng.uniform(1000.0, 40000.0, size=(500, 3, 368)).astype(np.float32)

    # Verify EMA recurrence matches scipy.signal.lfilter IIR transfer function
    alpha = 0.95
    ema_out = ema_clutter_removal(small_stream, alpha=alpha)
    b = [1.0 - alpha]
    a = [1.0, -alpha]
    zi = lfilter_zi(b, a)
    ref_trace, _ = lfilter(b, a, small_stream[:, 0, 10].astype(np.float64), zi=small_stream[0, 0, 10] * zi)
    expected_res = small_stream[:, 0, 10].astype(np.float64) - ref_trace
    np.testing.assert_allclose(ema_out[:, 0, 10], expected_res, atol=1e-2)

    # Verify SVD removes top-K static singular modes cleanly
    svd_out, used_k, _ = svd_clutter_removal(small_stream, k_components=2)
    assert used_k == 2
    _, s_after, _ = np.linalg.svd(svd_out[:, 0, :].astype(np.float64), full_matrices=False)
    # Largest remaining singular value of svd_out must equal original 3rd singular value sigma_3
    _, s_orig, _ = np.linalg.svd(small_stream[:, 0, :].astype(np.float64), full_matrices=False)
    np.testing.assert_allclose(s_after[0], s_orig[2], rtol=1e-4)
    print("[PASS] Module 2: EMA recurrence matches IIR reference & SVD K-subspace projection verified.")


def test_module3_and_module4_end_to_end() -> None:
    ds = generate_mock_uwb_stream(seed=42)
    ema_s = ema_clutter_removal(ds.raw_stream, alpha=0.95)
    svd_s, _, _ = svd_clutter_removal(ds.raw_stream, k_components=2)

    fall0 = ds.events_df[ds.events_df["event_type"] == "Fall"].iloc[0]
    scr_raw = compute_scr_db(
        ds.raw_stream, int(fall0["start_frame"]), int(fall0["end_frame"]),
        int(fall0["target_bin_min"]), int(fall0["target_bin_max"])
    )
    scr_ema = compute_scr_db(
        ema_s, int(fall0["start_frame"]), int(fall0["end_frame"]),
        int(fall0["target_bin_min"]), int(fall0["target_bin_max"])
    )
    scr_svd = compute_scr_db(
        svd_s, int(fall0["start_frame"]), int(fall0["end_frame"]),
        int(fall0["target_bin_min"]), int(fall0["target_bin_max"])
    )
    assert scr_svd > scr_ema > scr_raw + 15.0, f"SCR ordering unexpected: Raw={scr_raw}, EMA={scr_ema}, SVD={scr_svd}"

    l2_raw, _ = compute_empty_frame_l2_energy(ds.raw_stream, ds.empty_frame_mask)
    l2_ema, _ = compute_empty_frame_l2_energy(ema_s, ds.empty_frame_mask)
    l2_svd, _ = compute_empty_frame_l2_energy(svd_s, ds.empty_frame_mask)
    assert l2_ema < 0.005 * l2_raw and l2_svd < 0.005 * l2_raw, "Empty-frame L2 energy did not approach 0"

    X, y, bounds, _ = extract_window_features(svd_s, ds.events_df)
    expected_windows = (N_FRAMES - WINDOW_FRAMES) // STRIDE_FRAMES + 1
    assert X.shape == (expected_windows, len(FEATURE_NAMES)), f"Unexpected feature shape {X.shape}"
    assert not np.any(np.isnan(X)) and not np.any(np.isinf(X)), "NaN/Inf detected in window features"
    assert int(np.sum(y == 1)) >= 15, "Expected minority Fall windows across 3 Fall events"
    print("[PASS] Module 3 & 4: SCR hierarchy (SVD > EMA >> Raw), L2 suppression (>99.5%), & 1,196x5 ML matrix verified.")


if __name__ == "__main__":
    test_module1_constraints()
    test_module2_math_exactness()
    test_module3_and_module4_end_to_end()
    print("ALL VERIFICATION TESTS PASSED SUCCESSFULLY.")
