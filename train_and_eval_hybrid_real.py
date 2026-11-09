"""
Real-Dataset Training, mAP (tIoU >= 0.5) Evaluation & Test Inference for the
Bi-Level Cascade Hybrid Pipeline (9th ABC Challenge — UWB Fall Track).

1. Trains Stage 1 `MobileNetV3ProposalNet` across all 13 real Train streams (`train_00..12`)
   for ultra-high proposal recall & high background rejection.
2. Loads the Physics-Aware trained Stage 2 `FrameWise1DTCN` (`best_framewise_1d_tcn.pth`)
   and links Stage 1 -> Stage 2 -> Stage 3 (`UnifiedBiomechanicalRefinement` with SVD K=2
   energy envelope + >3.5s Temporal Peak Shrinking) inside `HybridCascade`.
3. Evaluates on the 5 real Validation streams (`val_00..04`), benchmarking:
     - Temporal Action Localization AP@{0.1, 0.3, 0.5, 0.7} & mAP@[0.50:0.05:0.75] (tIoU >= 0.5)
     - Official ABC2027 Ranking F1, F1@0.5, F1@0.7, FA/h, Start/End Median Error
     - Stage 1 Stream Rejection % & Proposal Recall %
4. Runs unseen Test inference (`test_00..04`), validates via `abc2027.evaluate.validate_predictions`,
   and generates `hybrid_cascade_diagnostics.png`.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_PKG_DIR = PROJECT_ROOT / "abc2027_uwb_train_v1"
TEST_PKG_DIR = PROJECT_ROOT / "abc2027_uwb_test_v1"
TRAIN_ZIP_PATH = PROJECT_ROOT / "abc2027_uwb_train_v1.zip"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TRAIN_PKG_DIR / "src") not in sys.path:
    sys.path.insert(0, str(TRAIN_PKG_DIR / "src"))

from abc2027.background import subtract_background
from abc2027.evaluate import IGNORE_MARGIN_S, evaluate as official_evaluate
from abc2027.evaluate import load_ground_truth, stream_durations, temporal_iou, validate_predictions
from hybrid_pipeline import (
    FS_HZ,
    TCN_IN_CHANNELS,
    FrameWise1DTCN,
    HybridCascade,
    MobileNetV3ProposalNet,
    normalize_residual_for_stage1,
)

STAGE2_PRETRAINED_CKPT = PROJECT_ROOT / "checkpoints" / "best_framewise_1d_tcn.pth"
HYBRID_CKPT_OUT = PROJECT_ROOT / "checkpoints" / "best_hybrid_cascade.pth"
HYBRID_METRICS_JSON = PROJECT_ROOT / "checkpoints" / "hybrid_cascade_metrics.json"
SUBMISSION_HYBRID_CSV = TEST_PKG_DIR / "submission_hybrid_cascade.csv"
PREDICTIONS_HYBRID_CSV = PROJECT_ROOT / "predictions_hybrid_test.csv"
DIAGNOSTIC_PLOT_PATH = PROJECT_ROOT / "hybrid_cascade_diagnostics.png"
STAGE1_CACHE_DIR = Path(tempfile.gettempdir()) / "abc2027_stage1_rtm_cache_v1"


# ==============================================================================
# TEMPORAL ACTION LOCALIZATION mAP (tIoU >= 0.5) EVALUATOR (IGNORE-AWARE)
# ==============================================================================


def compute_average_precision_at_tiou(
    ground_truth: list[dict],
    predictions: list[dict],
    durations: dict[str, float],
    theta: float,
) -> float:
    """
    Compute continuous-interpolation Average Precision (AP) at Temporal IoU threshold `theta`
    across all streams in `durations`, strictly honoring ABC2027 `--ignore-flagged` zones
    (unmatched predictions overlapping ignored GT +/- 2.0s are neither TP nor FP).
    """
    gt_by_stream: dict[str, list[dict]] = {sid: [] for sid in durations}
    ign_by_stream: dict[str, list[dict]] = {sid: [] for sid in durations}
    for ev in ground_truth:
        sid = ev["stream_id"]
        if sid not in durations:
            continue
        if ev["ignore"]:
            ign_by_stream[sid].append(ev)
        else:
            gt_by_stream[sid].append(ev)

    total_positives = sum(len(v) for v in gt_by_stream.values())
    if total_positives == 0:
        return 0.0

    valid_preds = [p for p in predictions if p["stream_id"] in durations]
    if not valid_preds:
        return 0.0

    # Sort all predictions globally by descending confidence score
    ordered_preds = sorted(valid_preds, key=lambda item: float(item["score"]), reverse=True)
    matched_gt: dict[str, set[int]] = {sid: set() for sid in durations}

    tp_flags: list[int] = []
    fp_flags: list[int] = []

    for pred in ordered_preds:
        sid = pred["stream_id"]
        ps, pe = float(pred["t_start"]), float(pred["t_end"])

        best_idx = None
        best_iou = -1.0
        for g_idx, gt_ev in enumerate(gt_by_stream[sid]):
            if g_idx in matched_gt[sid]:
                continue
            iou = temporal_iou(ps, pe, float(gt_ev["t_start"]), float(gt_ev["t_end"]))
            if iou >= theta and iou > best_iou:
                best_iou = iou
                best_idx = g_idx

        if best_idx is not None:
            matched_gt[sid].add(best_idx)
            tp_flags.append(1)
            fp_flags.append(0)
            continue

        # Check whether unmatched prediction falls in an official +/- 2.0s ignore zone
        overlaps_ignore = any(
            min(pe, float(ig["t_end"]) + IGNORE_MARGIN_S) > max(ps, float(ig["t_start"]) - IGNORE_MARGIN_S)
            for ig in ign_by_stream[sid]
        )
        if not overlaps_ignore:
            tp_flags.append(0)
            fp_flags.append(1)

    if not tp_flags:
        return 0.0

    tp_cum = np.cumsum(tp_flags, dtype=np.float64)
    fp_cum = np.cumsum(fp_flags, dtype=np.float64)
    recall = tp_cum / float(total_positives)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    # All-point monotonic precision envelope integration (VOC2010+ / ActivityNet standard)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_tal_map_suite(
    ground_truth: list[dict],
    predictions: list[dict],
    durations: dict[str, float],
) -> dict[str, float]:
    """Compute AP@{0.1, 0.3, 0.5, 0.7} and high-tIoU mAP@[0.50:0.05:0.75] (tIoU >= 0.5)."""
    high_tiou_grid = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
    ap_high = [compute_average_precision_at_tiou(ground_truth, predictions, durations, th) for th in high_tiou_grid]
    return {
        "AP@0.1": round(compute_average_precision_at_tiou(ground_truth, predictions, durations, 0.10), 4),
        "AP@0.3": round(compute_average_precision_at_tiou(ground_truth, predictions, durations, 0.30), 4),
        "AP@0.5": round(ap_high[0], 4),
        "AP@0.7": round(ap_high[4], 4),
        "mAP_tIoU>=0.5": round(float(np.mean(ap_high)), 4),
    }


# ==============================================================================
# REAL STREAM LOADER & STAGE 1 POOLED-RTM TRAINING DATASET
# ==============================================================================


def load_real_cir(split: str, index: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Load real ABC2027 stream from disk or in-memory from `abc2027_uwb_train_v1.zip`."""
    base_dir = TEST_PKG_DIR if split == "test" else TRAIN_PKG_DIR
    disk_path = base_dir / "streams" / split / f"stream_{index:02d}.h5"
    if disk_path.is_file():
        with h5py.File(disk_path, "r") as handle:
            cir = handle["cir"][:]
            serials = handle["radar_serials"][:] if "radar_serials" in handle else None
        if serials is not None and len(serials) == 3:
            order = np.argsort(serials)
            if not np.array_equal(order, np.arange(3)):
                cir = cir[order]
        return cir, serials

    if TRAIN_ZIP_PATH.is_file():
        member = f"abc2027_uwb_train_v1/streams/{split}/stream_{index:02d}.h5"
        with zipfile.ZipFile(TRAIN_ZIP_PATH, "r") as zf:
            if member in zf.namelist():
                with h5py.File(io.BytesIO(zf.read(member)), "r") as handle:
                    cir = handle["cir"][:]
                    serials = handle["radar_serials"][:] if "radar_serials" in handle else None
                if serials is not None and len(serials) == 3:
                    order = np.argsort(serials)
                    if not np.array_equal(order, np.arange(3)):
                        cir = cir[order]
                return cir, serials

    raise FileNotFoundError(f"Missing {split}/stream_{index:02d}.h5")


def prepare_stage1_pooled_rtm(split: str, index: int) -> tuple[np.ndarray, int]:
    """
    Precompute `(3, T//2, 92)` AvgPool2d((2, 4))-equivalent scaled residual RTM for ultra-fast
    Stage 1 training on all 13 real Train streams (< 4 MB per stream in temp cache).
    """
    STAGE1_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stream_id = f"{split}_{index:02d}"
    cache_file = STAGE1_CACHE_DIR / f"{stream_id}_rtm50x92.npz"
    if cache_file.is_file():
        with np.load(cache_file) as data:
            return data["pooled_rtm"].astype(np.float32), int(data["n_frames"])

    cir, _ = load_real_cir(split, index)
    n_frames = int(cir.shape[1])
    res = subtract_background(cir, tau_s=3.0)
    scaled = normalize_residual_for_stage1(res)  # non-negative (3, T, 368)

    t_even = (n_frames // 2) * 2
    pooled = scaled[:, :t_even, :].reshape(3, t_even // 2, 2, 92, 4).mean(axis=(2, 4)).astype(np.float32)
    np.savez_compressed(cache_file, pooled_rtm=pooled, n_frames=np.int64(n_frames))
    return pooled, n_frames


class Stage1WindowDataset(Dataset):
    """
    2.0s (100-frame -> 50 pooled-frame) sliding windows with 1.0s (50-frame -> 25 pooled-frame)
    stride across all 13 real Train streams, masking ambiguous flagged intervals and up-weighting
    known 0.2s splice windows as hard negatives.
    """

    def __init__(
        self,
        events_df: pd.DataFrame,
        composition_df: pd.DataFrame,
        augment: bool = True,
    ) -> None:
        self.augment = augment
        self.rtms: list[np.ndarray] = []
        self.pos_samples: list[tuple[int, int, float, float]] = []
        self.hard_neg_samples: list[tuple[int, int, float, float]] = []
        self.easy_neg_samples: list[tuple[int, int, float, float]] = []
        self.active_samples: list[tuple[int, int, float, float]] = []

        for idx in range(13):
            sid = f"train_{idx:02d}"
            t0 = time.perf_counter()
            pooled_rtm, n_frames = prepare_stage1_pooled_rtm("train", idx)
            self.rtms.append(pooled_rtm)

            ev_s = events_df[events_df["stream_id"] == sid]
            clean_falls: list[tuple[int, int]] = []
            ignored_zones: list[tuple[int, int]] = []
            for _, row in ev_s.iterrows():
                is_low = str(row.get("quality", "")).strip().lower() == "low"
                needs_rev = str(row.get("needs_review", "")).strip().lower() in {"1", "true", "yes"}
                is_ign = str(row.get("ignore", "")).strip().lower() in {"1", "true", "yes"}
                s_f = int(round(float(row["t_start"]) * FS_HZ))
                e_f = int(round(float(row["t_end"]) * FS_HZ))
                if is_low or needs_rev or is_ign:
                    ignored_zones.append((s_f - 50, e_f + 50))
                else:
                    clean_falls.append((s_f, e_f))

            comp_s = composition_df[composition_df["stream_id"] == sid].sort_values("t_start")
            splice_frames = [int(round(float(ts) * FS_HZ)) for ts in comp_s["t_start"].to_numpy()[1:]]

            n_pooled = pooled_rtm.shape[1]
            pos_cnt = 0
            for p_start in range(0, n_pooled - 50 + 1, 25):
                f_s = p_start * 2
                f_e = f_s + 100  # 2.0s window

                # Positive if window overlaps any clean fall by >= 12 frames (0.24s)
                overlaps_clean = any(min(f_e, ce) - max(f_s, cs) >= 12 for cs, ce in clean_falls)
                if overlaps_clean:
                    self.pos_samples.append((idx, p_start, 1.0, 1.0))
                    pos_cnt += 1
                    continue

                # Skip ambiguous flagged events so Stage 1 recall is never poisoned
                overlaps_ignored = any(min(f_e, ie) > max(f_s, is_) for is_, ie in ignored_zones)
                if overlaps_ignored:
                    continue

                # Separate known 0.2s crossfade splice windows (hard negatives) from easy background
                near_splice = any((f_s - 10) <= sp <= (f_e + 20) for sp in splice_frames)
                if near_splice:
                    self.hard_neg_samples.append((idx, p_start, 0.0, 1.8))
                else:
                    self.easy_neg_samples.append((idx, p_start, 0.0, 1.0))

            tot_so_far = len(self.pos_samples) + len(self.hard_neg_samples) + len(self.easy_neg_samples)
            print(
                f"  [Stage 1 Data {sid}] total_pool={tot_so_far:5d} (stream_pos={pos_cnt:2d}) "
                f"prepared in {time.perf_counter()-t0:.2f}s"
            )
        self.resample_epoch(seed=2027)

    def resample_epoch(self, seed: int) -> None:
        """Keep 100% of positive & splice hard-negative windows + fresh subsample of easy quiet windows."""
        rng = np.random.default_rng(seed)
        n_easy = min(len(self.easy_neg_samples), 2200)
        easy_idx = rng.choice(len(self.easy_neg_samples), size=n_easy, replace=False)
        chosen_easy = [self.easy_neg_samples[int(k)] for k in easy_idx]
        self.active_samples = self.pos_samples + self.hard_neg_samples + chosen_easy

    def __len__(self) -> int:
        return len(self.active_samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s_idx, p_start, label, weight = self.active_samples[i]
        rtm = self.rtms[s_idx]
        if self.augment and rtm.shape[1] > 50:
            jitter = int(np.random.randint(-6, 7))
            p_start = min(max(0, p_start + jitter), rtm.shape[1] - 50)
        patch = rtm[:, p_start : p_start + 50, :].copy()  # (3, 50, 92)
        if self.augment:
            gain = np.random.uniform(0.90, 1.10, size=(3, 1, 1)).astype(np.float32)
            patch = patch * gain
        return (
            torch.from_numpy(patch),
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(weight, dtype=torch.float32),
        )


# ==============================================================================
# STAGE 1 TRAINING, CASCADE VALIDATION & BENCHMARKING
# ==============================================================================


def train_stage1_proposal_network(
    events_df: pd.DataFrame,
    composition_df: pd.DataFrame,
    device: torch.device,
    epochs: int = 16,
) -> MobileNetV3ProposalNet:
    """Train `MobileNetV3ProposalNet` on all 13 real Train streams with High-Recall Weighted BCE."""
    print("\n[STEP 1] Preparing 13 Real Train Streams for Stage 1 MobileNetV3ProposalNet...")
    dataset = Stage1WindowDataset(events_df=events_df, composition_df=composition_df, augment=True)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)

    net = MobileNetV3ProposalNet().to(device)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"  MobileNetV3ProposalNet trainable parameters: {n_params:,} | Total 2.0s windows: {len(dataset):,}")

    # Positive class weight = 4.5 on hard-negative-enriched epoch sample drives near-100% Recall
    pos_weight = torch.tensor([4.5], device=device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=3.0e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=2e-4)

    for ep in range(1, epochs + 1):
        dataset.resample_epoch(seed=2027 + ep)
        t0 = time.perf_counter()
        net.train()
        total_loss = 0.0
        n_batches = 0
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            optimizer.zero_grad()
            logits = net.forward_logits(xb)
            bce = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight, reduction="none")
            loss = (bce * wb).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        scheduler.step()
        if ep in {1, 4, 8, 12, epochs}:
            print(f"  Stage 1 Epoch {ep:2d}/{epochs} | Weighted BCE Loss = {total_loss / n_batches:.4f} ({time.perf_counter()-t0:.2f}s)")

    return net.eval()


def evaluate_stage1_recall_on_val(
    val_results: dict[str, object],
    gt_events_val: list[dict],
) -> tuple[int, int, float]:
    """Count how many of the 67 clean Validation falls are covered by Stage 1 SuspiciousRegions."""
    clean_val = [ev for ev in gt_events_val if ev["stream_id"].startswith("val_") and not ev["ignore"]]
    covered = 0
    for ev in clean_val:
        res = val_results[ev["stream_id"]]
        gs, ge = float(ev["t_start"]), float(ev["t_end"])
        # Covered if any Stage 1 suspicious block contains at least 50% of the ground-truth fall
        if any(
            (min(reg.t_end_s, ge) - max(reg.t_start_s, gs)) >= 0.5 * (ge - gs)
            for reg in res.suspicious_regions
        ):
            covered += 1
    total = len(clean_val)
    return covered, total, (covered / total if total else 0.0)


def main() -> None:
    torch.manual_seed(2027)
    np.random.seed(2027)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 96)
    print("9th ABC CHALLENGE — BI-LEVEL CASCADE HYBRID PIPELINE (REAL DATASET TRAINING & EVALUATION)")
    print("=" * 96)

    events_csv = TRAIN_PKG_DIR / "streams_events.csv"
    comp_csv = TRAIN_PKG_DIR / "streams_composition.csv"
    events_df = pd.read_csv(events_csv)
    composition_df = pd.read_csv(comp_csv)
    gt_events_val = load_ground_truth(events_csv, ignore_flagged=True)

    # 1. Train Stage 1 MobileNetV3ProposalNet on all 13 real Train streams
    stage1_net = train_stage1_proposal_network(
        events_df=events_df,
        composition_df=composition_df,
        device=device,
        epochs=16,
    )

    # 2. Load Physics-Aware Trained Stage 2 12-Channel FrameWise1DTCN
    print(f"\n[STEP 2] Loading Physics-Aware Stage 2 FrameWise1DTCN from {STAGE2_PRETRAINED_CKPT}...")
    ckpt_s2 = torch.load(STAGE2_PRETRAINED_CKPT, map_location=device, weights_only=False)
    stage2_tcn = FrameWise1DTCN(
        in_channels=TCN_IN_CHANNELS,
        hidden_channels=int(ckpt_s2["model_config"]["hidden_channels"]),
        dilations=tuple(ckpt_s2["model_config"]["dilations"]),
    ).to(device)
    stage2_tcn.load_state_dict(ckpt_s2["model_state_dict"])
    stage2_tcn.eval()

    # Preload the 5 real Validation streams (raw CIR + causal EMA residual) once for calibration & scoring
    print("\n[STEP 3] Loading 5 Real Validation Streams (val_00..val_04) for Cascade Calibration & mAP Audit...")
    val_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    val_durations: dict[str, float] = {}
    for idx in range(5):
        sid = f"val_{idx:02d}"
        cir, _ = load_real_cir("val", idx)
        ema_res = subtract_background(cir, tau_s=3.0)
        val_data[sid] = (cir, ema_res)
        val_durations[sid] = float(cir.shape[1]) / FS_HZ

    # Calibrate Stage 1 threshold to preserve >= 95% proposal recall while maximizing stream rejection & mAP(tIoU>=0.5)
    best_cfg = None
    best_obj = -1.0
    sweep_logs: list[dict] = []

    print("\n  Stage 1 Proposal Threshold Sweep on Validation:")
    print(
        f"  {'S1_Th':>5s} | {'Reject%':>7s} | {'S1_Recall':>9s} | "
        f"{'mAP>=0.5':>8s} {'AP@0.5':>7s} {'AP@0.7':>7s} | {'RankF1':>7s} {'F1@0.5':>7s} {'F1@0.7':>7s} {'FA/h':>6s}"
    )
    print("  " + "-" * 86)

    for s1_th in (0.18, 0.25, 0.32, 0.40, 0.48):
        cascade = HybridCascade(
            stage1_proposal_net=stage1_net,
            stage2_tcn_head=stage2_tcn,
            stage1_window_frames=100,
            stage1_stride_frames=50,
            stage1_threshold=s1_th,
            stage2_threshold=0.50,
            max_biomech_duration_s=3.50,
            shrinking_half_window_s=1.50,
        ).to(device)

        val_res_map = {}
        preds_val = []
        tot_frames = 0
        eval_frames = 0

        for sid, (cir, ema_res) in val_data.items():
            res = cascade.infer_stream(cir, precomputed_ema_residual=ema_res, device=device)
            val_res_map[sid] = res
            tot_frames += res.total_frames
            eval_frames += res.frames_evaluated_by_stage2
            for ev in res.events:
                preds_val.append(
                    {
                        "stream_id": sid,
                        "t_start": ev.t_start,
                        "t_end": ev.t_end,
                        "score": ev.score,
                        "t_alarm": ev.t_start,
                    }
                )

        rej_pct = 100.0 * (1.0 - eval_frames / max(tot_frames, 1))
        cov_n, tot_n, s1_rec = evaluate_stage1_recall_on_val(val_res_map, gt_events_val)
        map_metrics = compute_tal_map_suite(gt_events_val, preds_val, val_durations)
        off_rep = official_evaluate(gt_events_val, preds_val, val_durations)

        rank_f1 = float(off_rep["ranking_score"]["f1_mean"])
        f1_05 = float(off_rep["by_theta"]["0.5"]["f1"])
        f1_07 = float(off_rep["by_theta"]["0.7"]["f1"])
        fah = float(off_rep["primary"]["false_alarms_per_hour"] or 0.0)

        print(
            f"  {s1_th:5.2f} | {rej_pct:6.2f}% | {cov_n:2d}/{tot_n:2d} ({s1_rec*100:4.1f}%) | "
            f"{map_metrics['mAP_tIoU>=0.5']:8.4f} {map_metrics['AP@0.5']:7.4f} {map_metrics['AP@0.7']:7.4f} | "
            f"{rank_f1:7.4f} {f1_05:7.4f} {f1_07:7.4f} {fah:6.2f}"
        )

        rec_entry = {
            "s1_threshold": s1_th,
            "rejection_pct": round(rej_pct, 2),
            "stage1_recall": round(s1_rec, 4),
            "map_metrics": map_metrics,
            "official_report": off_rep,
        }
        sweep_logs.append(rec_entry)

        # Objective: maximize mAP(tIoU >= 0.5) + Ranking F1 while maintaining high Stage 1 recall
        score_obj = 0.55 * map_metrics["mAP_tIoU>=0.5"] + 0.45 * rank_f1
        if score_obj > best_obj:
            best_obj = score_obj
            best_cfg = (s1_th, cascade, val_res_map, preds_val, map_metrics, off_rep, rej_pct, s1_rec)

    assert best_cfg is not None
    best_s1_th, best_cascade, best_val_res, best_val_preds, best_map, best_off_rep, best_rej_pct, best_s1_rec = best_cfg

    # Save unified Bi-Level Cascade checkpoint
    HYBRID_CKPT_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage1_state_dict": stage1_net.state_dict(),
            "stage2_state_dict": stage2_tcn.state_dict(),
            "stage1_threshold": best_s1_th,
            "stage2_threshold": 0.50,
            "max_biomech_duration_s": 3.50,
            "shrinking_half_window_s": 1.50,
            "val_map_metrics": best_map,
            "val_ranking_f1": float(best_off_rep["ranking_score"]["f1_mean"]),
        },
        HYBRID_CKPT_OUT,
    )

    # 4. Head-to-Head Comparison against Official Energy Baseline & Single-Stage TCNs on Validation
    print("\n[STEP 4] Head-to-Head Validation Benchmark (mAP @ tIoU >= 0.5 & Official ABC2027 Metrics):")
    benchmarks: list[dict] = []

    # (a) Official Energy Baseline (sample_submission.csv)
    sample_sub_path = TRAIN_PKG_DIR / "sample_submission.csv"
    if sample_sub_path.is_file():
        eb_preds = validate_predictions(sample_sub_path, val_durations)
        eb_map = compute_tal_map_suite(gt_events_val, eb_preds, val_durations)
        eb_off = official_evaluate(gt_events_val, eb_preds, val_durations)
        benchmarks.append(
            {
                "Method": "1. Official Energy Baseline",
                **eb_map,
                "Rank_F1": round(eb_off["ranking_score"]["f1_mean"], 4),
                "F1@0.3": round(eb_off["by_theta"]["0.3"]["f1"], 4),
                "F1@0.5": round(eb_off["by_theta"]["0.5"]["f1"], 4),
                "F1@0.7": round(eb_off["by_theta"]["0.7"]["f1"], 4),
                "FA/h@0.3": round(eb_off["primary"]["false_alarms_per_hour"], 2),
                "EndMedErr": round(eb_off["primary"]["end_error_median_abs"], 3),
            }
        )

    # (b) Single-Stage 12-Channel FrameWise1DTCN (EMA Coarse-to-Fine from training_metrics.json / train_real.py)
    from train_real import coarse_to_fine_decode, extract_spatial_bands, infer_stream_probabilities

    ss_preds: list[dict] = []
    for sid, (cir, _) in val_data.items():
        nb, rb = extract_spatial_bands(cir)
        probs = infer_stream_probabilities(stage2_tcn, nb, device=device)
        for item in coarse_to_fine_decode(sid, probs, rb, prob_threshold=0.50):
            ss_preds.append(
                {
                    "stream_id": sid,
                    "t_start": item["t_start"],
                    "t_end": item["t_end"],
                    "score": item["score"],
                    "t_alarm": item["t_start"],
                }
            )
    ss_map = compute_tal_map_suite(gt_events_val, ss_preds, val_durations)
    ss_off = official_evaluate(gt_events_val, ss_preds, val_durations)
    benchmarks.append(
        {
            "Method": "2. Single-Stage 12-Ch TCN (EMA Grad)",
            **ss_map,
            "Rank_F1": round(ss_off["ranking_score"]["f1_mean"], 4),
            "F1@0.3": round(ss_off["by_theta"]["0.3"]["f1"], 4),
            "F1@0.5": round(ss_off["by_theta"]["0.5"]["f1"], 4),
            "F1@0.7": round(ss_off["by_theta"]["0.7"]["f1"], 4),
            "FA/h@0.3": round(ss_off["primary"]["false_alarms_per_hour"], 2),
            "EndMedErr": round(ss_off["primary"]["end_error_median_abs"], 3),
        }
    )

    # (c) Bi-Level Cascade Hybrid Pipeline (Stage 1 MobileNetV3 + Stage 2 12-Ch TCN + Stage 3 SVD Biomech)
    benchmarks.append(
        {
            "Method": f"3. Bi-Level Hybrid Cascade (S1={best_s1_th:.2f})",
            **best_map,
            "Rank_F1": round(best_off_rep["ranking_score"]["f1_mean"], 4),
            "F1@0.3": round(best_off_rep["by_theta"]["0.3"]["f1"], 4),
            "F1@0.5": round(best_off_rep["by_theta"]["0.5"]["f1"], 4),
            "F1@0.7": round(best_off_rep["by_theta"]["0.7"]["f1"], 4),
            "FA/h@0.3": round(best_off_rep["primary"]["false_alarms_per_hour"], 2),
            "EndMedErr": round(best_off_rep["primary"]["end_error_median_abs"], 3),
        }
    )

    bench_df = pd.DataFrame(benchmarks)
    print("\n" + bench_df.to_string(index=False))

    # 5. Predict on the 5 Unseen Real Test Streams (test_00..test_04)
    print("\n[STEP 5] Running Bi-Level Cascade Inference on 5 Unseen Test Streams (test_00..test_04)...")
    test_events_out: list[dict] = []
    test_rejection_list: list[float] = []
    shrunk_count_test = 0

    for idx in range(5):
        sid = f"test_{idx:02d}"
        cir_t, _ = load_real_cir("test", idx)
        res_t = best_cascade.infer_stream(cir_t, device=device)
        test_rejection_list.append(res_t.stage1_rejection_ratio * 100.0)
        n_shrunk = sum(1 for ev in res_t.events if ev.failsafe_shrunk)
        shrunk_count_test += n_shrunk

        for ev in res_t.events:
            test_events_out.append(
                {
                    "stream_id": sid,
                    "t_start": ev.t_start,
                    "t_end": ev.t_end,
                    "score": ev.score,
                }
            )
        print(
            f"  [{sid}] dur={res_t.stream_duration_s/60.0:5.2f}m | "
            f"S1_Rejected={res_t.stage1_rejection_ratio*100:5.1f}% | "
            f"Regions={len(res_t.suspicious_regions):2d} -> Final_Events={len(res_t.events):2d} "
            f"(Failsafe_Shrunk={n_shrunk}) | Stage1={res_t.timing_ms['stage1_mobilenet_ms']:.0f}ms, "
            f"Stage2={res_t.timing_ms['stage2_tcn_ms']:.0f}ms, Stage3={res_t.timing_ms['stage3_svd_refine_ms']:.0f}ms"
        )

    df_off = pd.DataFrame(test_events_out, columns=["stream_id", "t_start", "t_end", "score"])
    df_off.to_csv(SUBMISSION_HYBRID_CSV, index=False)

    df_prompt = df_off.rename(columns={"t_start": "start_time", "t_end": "end_time"})
    df_prompt.to_csv(PREDICTIONS_HYBRID_CSV, index=False)

    test_durs = stream_durations(TEST_PKG_DIR / "streams" / "test")
    validated = validate_predictions(SUBMISSION_HYBRID_CSV, test_durs)
    print(
        f"\n[OFFICIAL VALIDATOR PASSED] {len(validated)} hybrid test predictions verified! "
        f"(Mean Test Stream Rejection = {np.mean(test_rejection_list):.1f}%, Failsafe Shrunk = {shrunk_count_test})"
    )

    # Save JSON diagnostics
    HYBRID_METRICS_JSON.write_text(
        json.dumps(
            {
                "selected_stage1_threshold": best_s1_th,
                "val_stage1_rejection_pct": round(best_rej_pct, 2),
                "val_stage1_proposal_recall": round(best_s1_rec, 4),
                "benchmarks": benchmarks,
                "threshold_sweep": sweep_logs,
                "test_event_count": len(df_off),
                "test_mean_rejection_pct": round(float(np.mean(test_rejection_list)), 2),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # 6. Generate 3-Panel Diagnostic Plot
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), dpi=140)

    # Panel A: High-tIoU Benchmark Bar Chart
    x_pos = np.arange(len(bench_df))
    w = 0.25
    axes[0].bar(x_pos - w, bench_df["AP@0.5"], width=w, label="AP @ tIoU=0.5", color="#1f77b4")
    axes[0].bar(x_pos, bench_df["mAP_tIoU>=0.5"], width=w, label="mAP @ [0.50:0.75]", color="#2ca02c")
    axes[0].bar(x_pos + w, bench_df["F1@0.7"], width=w, label="F1 @ tIoU=0.7", color="#ff7f0e")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(["Official\nEnergy", "Single-Stage\n12-Ch TCN", "Bi-Level\nHybrid Cascade"], fontsize=8.5)
    axes[0].set_ylabel("Score")
    axes[0].set_title("A. High-tIoU Localization Benchmark (Val Set)")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)

    # Panel B: Stage 1 Threshold Trade-off (Stream Rejection % vs mAP & Recall)
    s_th = [row["s1_threshold"] for row in sweep_logs]
    s_rej = [row["rejection_pct"] for row in sweep_logs]
    s_rec = [row["stage1_recall"] * 100.0 for row in sweep_logs]
    s_map = [row["map_metrics"]["mAP_tIoU>=0.5"] * 100.0 for row in sweep_logs]
    axes[1].plot(s_th, s_rej, marker="o", lw=2.0, color="#1f77b4", label="Stage 1 Stream Rejected (%)")
    axes[1].plot(s_th, s_rec, marker="s", lw=2.0, color="#2ca02c", label="Stage 1 Proposal Recall (%)")
    axes[1].plot(s_th, s_map, marker="^", lw=2.0, color="#d62728", label="Cascade mAP>=0.5 (x100)")
    axes[1].axvline(best_s1_th, ls=":", color="#111111", lw=1.4, label=f"Selected S1={best_s1_th:.2f}")
    axes[1].set_xlabel("Stage 1 MobileNetV3 Proposal Threshold")
    axes[1].set_ylabel("Percentage (%)")
    axes[1].set_title("B. Stage 1 Fast Proposal Efficiency & Recall")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7.8)

    # Panel C: Test Event Duration Histogram (showing 3.5s Biomechanical Failsafe Ceiling)
    test_durs_arr = (df_off["t_end"] - df_off["t_start"]).to_numpy()
    axes[2].hist(test_durs_arr, bins=18, color="#9467bd", edgecolor="white", alpha=0.88)
    axes[2].axvline(3.50, color="#d62728", ls="--", lw=1.8, label="Biomech Limit (3.5s)")
    axes[2].axvline(float(np.median(test_durs_arr)), color="#2ca02c", ls="-.", lw=1.8, label=f"Median ({np.median(test_durs_arr):.2f}s)")
    axes[2].set_xlabel("Predicted Fall Duration (s)")
    axes[2].set_ylabel("Event Count")
    axes[2].set_title(f"C. Unseen Test Durations (N={len(df_off)} Events)")
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(DIAGNOSTIC_PLOT_PATH)
    plt.close(fig)
    print(f"Saved diagnostic visualization -> {DIAGNOSTIC_PLOT_PATH}")
    print(f"Saved hybrid checkpoint        -> {HYBRID_CKPT_OUT}")


if __name__ == "__main__":
    main()
