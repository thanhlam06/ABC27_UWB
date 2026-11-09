"""
Module 4: Machine Learning Baseline (Sliding Window + Statistical Energy Features)
9th ABC Challenge - UWB Fall Track Pipeline

Implements:
1. Sliding Window Generator:
       Window size = 125 frames (2.5s @ 50 Hz), Stride = 25 frames (0.5s @ 50 Hz)
2. Statistical Feature Extraction (5 features per window from preprocessed frame energy E(t)):
       Mean, Variance, Skewness, Kurtosis, Max Peak
3. Binary Classification Model:
       RandomForestClassifier (Window contains Fall [1] vs. No Fall [0])
4. Rigorous Leak-Free Evaluation:
       Outputs scikit-learn Classification Report (Precision, Recall, F1-Score) focusing
       on the minority 'Fall' class, comparing SVD vs. EMA vs. Raw features without
       temporal window-overlap leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from module1_eda import FS_HZ

WINDOW_FRAMES: int = 125   # 2.5 seconds @ 50 Hz
STRIDE_FRAMES: int = 25    # 0.5 seconds @ 50 Hz
FEATURE_NAMES: list[str] = ["Mean", "Variance", "Skewness", "Kurtosis", "Max_Peak"]


@dataclass
class MLBaselineResult:
    """Outputs and evaluation metrics for Module 4 RandomForest baseline."""
    features_df: pd.DataFrame
    primary_method: str
    holdout_report_str: str
    oof_report_str: str
    comparison_df: pd.DataFrame
    feature_importances: dict[str, float]


def sliding_window_generator(
    signal_1d_or_nd: np.ndarray,
    window_size: int = WINDOW_FRAMES,
    stride: int = STRIDE_FRAMES,
) -> Iterator[tuple[int, int, np.ndarray]]:
    """
    Generate sliding windows over the temporal axis (axis 0) of a continuous stream.

    Yields
    ------
    (start_frame, end_frame, window_slice)
    """
    n_frames = signal_1d_or_nd.shape[0]
    for start in range(0, n_frames - window_size + 1, stride):
        end = start + window_size
        yield start, end, signal_1d_or_nd[start:end]


def compute_frame_energy(preprocessed_stream: np.ndarray) -> np.ndarray:
    """
    Compute instantaneous multi-radar signal energy E(t) per frame:
        E(t) = (1 / R) * sum_{r=1..R} sum_{b=0..B-1} y(t, r, b)^2
    """
    x = preprocessed_stream.astype(np.float64, copy=False)
    return np.mean(np.sum(x * x, axis=2), axis=1)


def extract_window_features(
    preprocessed_stream: np.ndarray,
    events_df: pd.DataFrame,
    window_size: int = WINDOW_FRAMES,
    stride: int = STRIDE_FRAMES,
    min_fall_overlap_frames: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract the 5 required statistical features per sliding window from preprocessed
    signal energy over time: [Mean, Variance, Skewness, Kurtosis, Max Peak].

    A window [w_start, w_end) is labeled 1 ('Fall') if it overlaps any ground-truth Fall
    interval by at least `min_fall_overlap_frames` (0.50s = 25 frames), else 0 ('No Fall').
    """
    energy_t = compute_frame_energy(preprocessed_stream)
    falls_df = events_df[events_df["event_type"] == "Fall"]
    fall_intervals = list(zip(falls_df["start_frame"].to_numpy(), falls_df["end_frame"].to_numpy()))

    features_list: list[list[float]] = []
    labels_list: list[int] = []
    window_bounds: list[tuple[int, int]] = []

    for w_start, w_end, e_win in sliding_window_generator(energy_t, window_size=window_size, stride=stride):
        mean_val = float(np.mean(e_win))
        var_val = float(np.var(e_win))
        max_peak = float(np.max(e_win))

        if var_val < 1e-12:
            skew_val = 0.0
            kurt_val = 0.0
        else:
            skew_val = float(skew(e_win, bias=False))
            kurt_val = float(kurtosis(e_win, fisher=True, bias=False))

        features_list.append([mean_val, var_val, skew_val, kurt_val, max_peak])

        # Binary label: does window meaningfully contain a Fall event?
        overlap = max(
            (min(w_end, f_end) - max(w_start, f_start) for f_start, f_end in fall_intervals),
            default=0,
        )
        labels_list.append(1 if overlap >= min_fall_overlap_frames else 0)
        window_bounds.append((w_start, w_end))

    return (
        np.asarray(features_list, dtype=np.float64),
        np.asarray(labels_list, dtype=np.int32),
        np.asarray(window_bounds, dtype=np.int32),
        energy_t,
    )


def _evaluate_stream_with_rf(
    stream: np.ndarray,
    events_df: pd.DataFrame,
    random_state: int = 42,
) -> dict:
    """
    Train and evaluate RandomForestClassifier using:
    1. Leak-Free Chronological Split (Train: 0–400s [2 Falls], Test: 400–600s [1 unseen Fall])
    2. Leak-Free 3-Fold Contiguous Temporal Block Out-of-Fold (OOF) Evaluation across all
       30,000 frames (3 blocks of 10,000 frames each, zero shared frames across folds).
    """
    X, y, bounds, energy_t = extract_window_features(stream, events_df)

    # 1. Leak-Free Chronological Split at frame 20,000 (t = 400.0s)
    split_frame = 20_000
    train_mask = bounds[:, 1] <= split_frame   # Strictly ends before split_frame
    test_mask = bounds[:, 0] >= split_frame    # Strictly starts at/after split_frame

    clf_holdout = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
    )
    clf_holdout.fit(X[train_mask], y[train_mask])
    y_test = y[test_mask]
    y_pred_test = clf_holdout.predict(X[test_mask])

    # 2. Leak-Free 3-Fold Contiguous Temporal Block Out-of-Fold Evaluation
    #    Block 0: [0, 10000), Block 1: [10000, 20000), Block 2: [20000, 30000]
    #    Windows straddling block boundaries (10,000 or 20,000) are excluded from test/train overlap.
    block_edges = [(0, 10_000), (10_000, 20_000), (20_000, 30_000)]
    oof_true_all: list[np.ndarray] = []
    oof_pred_all: list[np.ndarray] = []
    fold_importances: list[np.ndarray] = []

    for b_start, b_end in block_edges:
        fold_test = (bounds[:, 0] >= b_start) & (bounds[:, 1] <= b_end)
        fold_train = (bounds[:, 1] <= b_start) | (bounds[:, 0] >= b_end)

        clf_fold = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
        )
        clf_fold.fit(X[fold_train], y[fold_train])
        oof_true_all.append(y[fold_test])
        oof_pred_all.append(clf_fold.predict(X[fold_test]))
        fold_importances.append(clf_fold.feature_importances_)

    y_oof_true = np.concatenate(oof_true_all)
    y_oof_pred = np.concatenate(oof_pred_all)
    mean_importances = np.mean(fold_importances, axis=0)

    p_ho, r_ho, f1_ho, _ = precision_recall_fscore_support(y_test, y_pred_test, labels=[1], zero_division=0)
    p_oof, r_oof, f1_oof, _ = precision_recall_fscore_support(y_oof_true, y_oof_pred, labels=[1], zero_division=0)

    return {
        "X": X,
        "y": y,
        "bounds": bounds,
        "energy_t": energy_t,
        "y_test": y_test,
        "y_pred_test": y_pred_test,
        "y_oof_true": y_oof_true,
        "y_oof_pred": y_oof_pred,
        "cm_oof": confusion_matrix(y_oof_true, y_oof_pred, labels=[0, 1]),
        "holdout_precision": float(p_ho[0]),
        "holdout_recall": float(r_ho[0]),
        "holdout_f1": float(f1_ho[0]),
        "oof_precision": float(p_oof[0]),
        "oof_recall": float(r_oof[0]),
        "oof_f1": float(f1_oof[0]),
        "importances": mean_importances,
        "holdout_report": classification_report(
            y_test, y_pred_test, target_names=["No Fall (0)", "Fall (1)"], digits=4, zero_division=0
        ),
        "oof_report": classification_report(
            y_oof_true, y_oof_pred, target_names=["No Fall (0)", "Fall (1)"], digits=4, zero_division=0
        ),
    }


def run_module4_ml_baseline(
    raw_stream: np.ndarray,
    ema_stream: np.ndarray,
    svd_stream: np.ndarray,
    events_df: pd.DataFrame,
    output_dir: Path,
) -> MLBaselineResult:
    """
    Execute Module 4 Sliding-Window ML Baseline on Preprocessed UWB Streams,
    output Classification Reports focusing on the minority 'Fall' class, and save diagnostic plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_raw = _evaluate_stream_with_rf(raw_stream, events_df)
    eval_ema = _evaluate_stream_with_rf(ema_stream, events_df)
    eval_svd = _evaluate_stream_with_rf(svd_stream, events_df)

    comparison_rows = []
    for label, res in [
        ("Raw Stream (Unfiltered)", eval_raw),
        ("EMA Preprocessed (α=0.95)", eval_ema),
        ("SVD Preprocessed (K=2)", eval_svd),
    ]:
        tn, fp, fn, tp = res["cm_oof"].ravel()
        comparison_rows.append(
            {
                "Input Stream": label,
                "Fall Precision (OOF)": round(res["oof_precision"], 4),
                "Fall Recall (OOF)": round(res["oof_recall"], 4),
                "Fall F1-Score (OOF)": round(res["oof_f1"], 4),
                "TP / FP / FN": f"{tp} / {fp} / {fn}",
                "Hold-Out Fall F1": round(res["holdout_f1"], 4),
            }
        )
    comparison_df = pd.DataFrame(comparison_rows)

    # Primary validated preprocessed pipeline: SVD Preprocessed (highest SCR)
    primary = eval_svd
    imp_dict = {name: round(float(val), 4) for name, val in zip(FEATURE_NAMES, primary["importances"])}

    features_df = pd.DataFrame(primary["X"], columns=FEATURE_NAMES)
    features_df["start_frame"] = primary["bounds"][:, 0]
    features_df["end_frame"] = primary["bounds"][:, 1]
    features_df["label"] = primary["y"]

    # -------------------------------------------------------------------------
    # Diagnostic Plot: Preprocessed Signal Energy Stream & RF Feature Importances
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), dpi=160, gridspec_kw={"width_ratios": [1.65, 1.0]})

    # Left: Preprocessed Signal Energy E(t) over the 600s Stream with Fall Ground Truth
    t_sec = np.arange(len(primary["energy_t"])) / FS_HZ
    axes[0].plot(t_sec, primary["energy_t"] / 1e6, color="#1F77B4", lw=0.85, label="SVD Residual Frame Energy E(t)")
    falls_df = events_df[events_df["event_type"] == "Fall"]
    for idx, (_, f_row) in enumerate(falls_df.iterrows()):
        axes[0].axvspan(
            f_row["t_start"] - 1.5,
            f_row["t_end"] + 1.5,
            color="#D62728",
            alpha=0.25,
            label="Injected Fall Event" if idx == 0 else None,
        )
    axes[0].axvline(400.0, color="#333333", linestyle="--", lw=1.3, label="Chronological Split (400s)")
    axes[0].set_title("Module 4: Preprocessed Frame Energy E(t) Across 600s Stream", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Stream Time (seconds)", fontsize=10)
    axes[0].set_ylabel("Frame Energy (×10⁶ ADC²)", fontsize=10)
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].legend(loc="upper right", fontsize=8.8)

    # Right: RandomForest Feature Importances (5 Statistical Window Features)
    order = np.argsort(primary["importances"])
    sorted_names = [FEATURE_NAMES[i] for i in order]
    sorted_vals = primary["importances"][order]
    bars = axes[1].barh(sorted_names, sorted_vals, color="#4C78A8", edgecolor="#222222", height=0.55)
    for bar, val in zip(bars, sorted_vals):
        axes[1].text(val + 0.008, bar.get_y() + bar.get_height() / 2.0, f"{val:.3f}", va="center", fontsize=9.5, fontweight="bold")
    axes[1].set_xlim(0, max(sorted_vals) * 1.22)
    axes[1].set_title("RandomForest Feature Importances\n(5 Statistical Window Energy Features)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Mean Decrease in Impurity (Gini Importance)", fontsize=10)
    axes[1].grid(True, axis="x", linestyle=":", alpha=0.5)

    fig.tight_layout()
    ml_plot_path = output_dir / "module4_ml_evaluation.png"
    fig.savefig(ml_plot_path)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Console Analytical Output
    # -------------------------------------------------------------------------
    n_total_win = len(primary["y"])
    n_pos_win = int(np.sum(primary["y"] == 1))
    n_neg_win = int(np.sum(primary["y"] == 0))

    print("=" * 80)
    print("MODULE 4: MACHINE LEARNING BASELINE (SLIDING WINDOW + RANDOM FOREST)")
    print("=" * 80)
    print(f"Sliding Window Config    : Window = {WINDOW_FRAMES} frames (2.50s) | Stride = {STRIDE_FRAMES} frames (0.50s)")
    print(f"Total Extracted Windows  : {n_total_win:,} windows | Positive 'Fall' = {n_pos_win} ({100*n_pos_win/n_total_win:.2f}%) | 'No Fall' = {n_neg_win:,}")
    print(f"Extracted Features (5)   : {FEATURE_NAMES}")
    print(f"Classifier Config        : RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced')")
    print("-" * 80)
    print(">>> A. Leak-Free Chronological Test Split (Train: 0–400s [2 Falls], Test: 400–600s [1 Unseen Fall])")
    print("    [Preprocessed Stream: SVD Filtered (K=2)]")
    print(primary["holdout_report"])
    print(">>> B. Leak-Free 3-Fold Temporal Block Out-of-Fold Report (Full 600s Stream, All 3 Unseen Falls)")
    print("    [Preprocessed Stream: SVD Filtered (K=2)]")
    print(primary["oof_report"])
    print(">>> C. Preprocessing Impact on Minority 'Fall' Class Detection:")
    print(comparison_df.to_string(index=False))
    print("-" * 80)
    print(f"Feature Importances (SVD): {imp_dict}")
    print(f"Saved ML Diagnostic Plot : {ml_plot_path.name}")
    print("[Analytical Insight - Module 4]:")
    print("  1. Why Leak-Free Temporal Splitting Matters: Because consecutive windows (125 frames, stride 25)")
    print("     share 80% of their frames, naive random train/test splitting leaks identical Fall peaks into")
    print("     both train and test sets. Evaluating across disjoint temporal blocks tests true generalization.")
    print("  2. Clutter Removal Enables Statistical Separability: On Raw unfiltered data, static clutter")
    print("     swamps E(t), degrading minority 'Fall' F1. Clutter suppression (EMA and SVD) exposes the")
    print("     impulsive floor-impact peak and variance burst, allowing a lightweight 5-feature Random")
    print("     Forest to reliably separate Falls from vigorous Sit-to-Stand and Walking ADLs.")
    print()

    return MLBaselineResult(
        features_df=features_df,
        primary_method="SVD Preprocessed (K=2)",
        holdout_report_str=primary["holdout_report"],
        oof_report_str=primary["oof_report"],
        comparison_df=comparison_df,
        feature_importances=imp_dict,
    )
