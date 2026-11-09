"""
Module 3: Inference & Coarse-to-Fine Post-Processing Pipeline for ABC2027 Test Set.

Loads the trained 12-channel FrameWise1DTCN checkpoint, processes unseen Test streams
(test_00..test_04) with identical 12-channel spatial banding and per-channel IQR
normalization, extracts coarse proposals (p(t) >= 0.50, merge <= 0.60s, drop < 0.60s),
refines [t_start, t_end] via 1st-derivative dE/dt peak onset and post-fall baseline+10%
energy stabilization, and exports validated CSV predictions.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_PKG_DIR = PROJECT_ROOT / "abc2027_uwb_train_v1"
TEST_PKG_DIR = PROJECT_ROOT / "abc2027_uwb_test_v1"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TRAIN_PKG_DIR / "src") not in sys.path:
    sys.path.insert(0, str(TRAIN_PKG_DIR / "src"))

from abc2027.evaluate import stream_durations, validate_predictions
from train_real import (
    DEFAULT_CKPT_PATH,
    FS_HZ,
    IN_CHANNELS,
    FrameWise1DTCN,
    extract_spatial_bands,
    infer_stream_probabilities,
)

DEFAULT_TEST_STREAMS_DIR = TEST_PKG_DIR / "streams" / "test"
DEFAULT_PROMPT_CSV = PROJECT_ROOT / "predictions_test.csv"
DEFAULT_OFFICIAL_CSV = TEST_PKG_DIR / "submission_final_1d_tcn.csv"


def load_trained_model(
    checkpoint_path: Path = DEFAULT_CKPT_PATH,
    device: torch.device | str = "cpu",
) -> FrameWise1DTCN:
    """Load trained 12-channel FrameWise1DTCN weights from checkpoint."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}. Run `python train_real.py` first."
        )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("model_config", {})
    model = FrameWise1DTCN(
        in_channels=int(cfg.get("in_channels", IN_CHANNELS)),
        hidden_channels=int(cfg.get("hidden_channels", 32)),
        dilations=tuple(cfg.get("dilations", [1, 2, 4, 8, 16])),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def extract_coarse_proposals(
    frame_probs: np.ndarray,
    prob_threshold: float = 0.50,
    merge_gap_s: float = 0.60,
    min_duration_s: float = 0.60,
    fs_hz: float = FS_HZ,
) -> list[tuple[int, int]]:
    """
    Step 1 — Coarse Proposal:
      - Binarize frame-wise probability p(t) >= 0.50
      - Merge inter-event gaps <= 0.60s (30 frames at 50 Hz)
      - Drop short events < 0.60s (30 frames at 50 Hz)
    """
    pos_indices = np.flatnonzero(frame_probs >= prob_threshold)
    if len(pos_indices) == 0:
        return []

    cuts = np.flatnonzero(np.diff(pos_indices) > 1)
    starts = np.concatenate([[pos_indices[0]], pos_indices[cuts + 1]])
    ends = np.concatenate([pos_indices[cuts] + 1, [pos_indices[-1] + 1]])

    max_gap_frames = int(round(merge_gap_s * fs_hz))
    merged: list[list[int]] = [[int(starts[0]), int(ends[0])]]
    for s_idx, e_idx in zip(starts[1:], ends[1:]):
        if int(s_idx) - merged[-1][1] <= max_gap_frames:
            merged[-1][1] = int(e_idx)
        else:
            merged.append([int(s_idx), int(e_idx)])

    min_frames = int(round(min_duration_s * fs_hz))
    return [(s_idx, e_idx) for s_idx, e_idx in merged if (e_idx - s_idx) >= min_frames]


def refine_events_with_energy_gradient(
    stream_id: str,
    coarse_proposals: list[tuple[int, int]],
    raw_bands: np.ndarray,
    frame_probs: np.ndarray,
    prob_threshold: float = 0.50,
    min_duration_s: float = 0.60,
    fs_hz: float = FS_HZ,
) -> list[dict]:
    """
    Step 2 — Fine Gradient Refinement:
      Computes total energy envelope E(t) = sum_{c=1..12} e_c(t) across all 12 spatial bands
      and its 1st derivative dE/dt:
        - t_start: pinned to the frame where dE/dt hits its first major positive peak (fall onset).
        - t_end  : pinned to the frame where E(t) stabilizes back to baseline + 10% dynamic span
                   (post-fall immobilization).
    """
    if not coarse_proposals:
        return []

    n_frames = len(frame_probs)
    stream_duration_s = n_frames / fs_hz
    min_frames = int(round(min_duration_s * fs_hz))

    # Total multi-radar spatial energy envelope E(t) and 1st temporal derivative dE/dt
    total_energy = np.sum(raw_bands.astype(np.float64), axis=0)
    smooth_energy = gaussian_filter1d(total_energy, sigma=2.5, mode="nearest")
    grad_energy = np.gradient(smooth_energy, 1.0 / fs_hz)

    refined_events: list[dict] = []
    for s_coarse, e_coarse in coarse_proposals:
        # Estimate local static baseline from flanking quiet frames outside the coarse proposal
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

        # Locate main fall impact peak within the high-confidence core of the proposal
        ev_probs = frame_probs[s_coarse:e_coarse]
        core_thresh = max(prob_threshold, 0.75 * float(np.max(ev_probs)))
        core_rel = np.flatnonzero(ev_probs >= core_thresh)
        if len(core_rel) > 0:
            c_s = s_coarse + int(core_rel[0])
            c_e = s_coarse + int(core_rel[-1]) + 1
            impact_idx = c_s + int(np.argmax(smooth_energy[c_s:c_e]))
        else:
            impact_idx = s_coarse + int(np.argmax(smooth_energy[s_coarse:e_coarse]))

        peak_energy = float(smooth_energy[impact_idx])
        dynamic_span = max(peak_energy - baseline, 1e-8)

        # Pin t_start to first major positive peak of dE/dt in the onset zone
        search_s = max(0, s_coarse - int(0.20 * fs_hz))
        search_e = max(search_s + 3, impact_idx + 1)
        onset_grad = grad_energy[search_s:search_e]
        max_pos_grad = float(np.max(onset_grad)) if len(onset_grad) > 0 else 0.0

        s_fine = s_coarse
        if max_pos_grad > 1e-8 and len(onset_grad) >= 3:
            peaks, _ = find_peaks(onset_grad, height=0.25 * max_pos_grad, distance=4)
            valid_peaks = [
                int(p)
                for p in peaks
                if float(np.max(smooth_energy[search_s + int(p) : min(n_frames, search_s + int(p) + 8)]))
                >= baseline + 0.05 * dynamic_span
            ]
            if valid_peaks:
                cand_s = search_s + valid_peaks[0]
                s_fine = int(np.clip(cand_s, max(0, s_coarse - int(0.20 * fs_hz)), min(impact_idx, s_coarse + int(0.55 * fs_hz))))

        # Pin t_end to where E(t) stabilizes back to baseline + 10% dynamic span
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
        t_end = round(min(stream_duration_s, e_fine / fs_hz), 3)
        if t_end <= t_start:
            t_end = min(stream_duration_s, round(t_start + min_duration_s, 3))
        if t_end <= t_start:
            continue

        score = float(0.75 * np.max(ev_probs) + 0.25 * np.mean(ev_probs))
        refined_events.append(
            {
                "stream_id": stream_id,
                "t_start": t_start,
                "t_end": t_end,
                "score": round(score, 6),
            }
        )

    return refined_events


def infer_stream_id_from_path(h5_path: Path) -> str:
    """Resolve official ABC2027 stream_id (e.g., stream_00.h5 under test/ -> test_00)."""
    stem = h5_path.stem
    if stem.startswith(("test_", "val_", "train_")):
        return stem
    parts = stem.split("_")
    idx = int(parts[-1]) if parts[-1].isdigit() else 0
    parent_name = h5_path.parent.name.lower()
    split = parent_name if parent_name in {"test", "val", "train"} else "test"
    return f"{split}_{idx:02d}"


def process_test_stream(
    h5_path: Path,
    model: FrameWise1DTCN,
    prob_threshold: float = 0.50,
    merge_gap_s: float = 0.60,
    min_duration_s: float = 0.60,
    chunk_size: int = 10_000,
    overlap: int = 500,
    device: torch.device | str = "cpu",
) -> tuple[list[dict], np.ndarray]:
    """Run full 12-band IQR preprocessing, TCN inference, and Coarse-to-Fine refinement on one .h5 stream."""
    t0 = time.perf_counter()
    stream_id = infer_stream_id_from_path(h5_path)

    with h5py.File(h5_path, "r") as handle:
        cir = handle["cir"][:]
        serials = handle["radar_serials"][:] if "radar_serials" in handle else None

    norm_bands, raw_bands = extract_spatial_bands(cir, radar_serials=serials)
    frame_probs = infer_stream_probabilities(
        model=model,
        norm_bands=norm_bands,
        chunk_size=chunk_size,
        overlap=overlap,
        device=device,
    )

    coarse_proposals = extract_coarse_proposals(
        frame_probs=frame_probs,
        prob_threshold=prob_threshold,
        merge_gap_s=merge_gap_s,
        min_duration_s=min_duration_s,
        fs_hz=FS_HZ,
    )
    refined_events = refine_events_with_energy_gradient(
        stream_id=stream_id,
        coarse_proposals=coarse_proposals,
        raw_bands=raw_bands,
        frame_probs=frame_probs,
        prob_threshold=prob_threshold,
        min_duration_s=min_duration_s,
        fs_hz=FS_HZ,
    )

    dt = time.perf_counter() - t0
    dur_min = (norm_bands.shape[1] / FS_HZ) / 60.0
    print(
        f"  [{stream_id}] {h5_path.name} ({dur_min:5.2f} min, {norm_bands.shape[1]:6d} frames) -> "
        f"coarse={len(coarse_proposals):2d}, refined={len(refined_events):2d} events | "
        f"max_p={frame_probs.max():.4f} | {dt:.2f}s"
    )
    return refined_events, frame_probs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict ABC2027 Test Streams with 12-Channel FrameWise1DTCN")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT_PATH, help="Trained model checkpoint path")
    parser.add_argument(
        "--streams",
        type=Path,
        default=DEFAULT_TEST_STREAMS_DIR,
        help="Directory of test stream_XX.h5 files or a single .h5 file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PROMPT_CSV,
        help="Output CSV path with [stream_id, start_time, end_time, score]",
    )
    parser.add_argument(
        "--official-output",
        type=Path,
        default=DEFAULT_OFFICIAL_CSV,
        help="Official ABC2027 submission CSV path with [stream_id, t_start, t_end, score]",
    )
    parser.add_argument("--threshold", type=float, default=0.50, help="Coarse proposal probability threshold")
    parser.add_argument("--merge-gap", type=float, default=0.60, help="Max gap (seconds) to merge coarse events")
    parser.add_argument("--min-duration", type=float, default=0.60, help="Min duration (seconds) to keep coarse events")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 88)
    print("ABC2027 TEST INFERENCE & COARSE-TO-FINE GRADIENT REFINEMENT PIPELINE")
    print("=" * 88)
    print(f"Loading model checkpoint : {args.checkpoint}")
    model = load_trained_model(args.checkpoint, device=device)
    print(
        f"Loaded FrameWise1DTCN    : in_channels={model.in_channels}, "
        f"RF={model.receptive_field_frames} frames ({model.receptive_field_seconds:.2f}s)"
    )

    if args.streams.is_file():
        stream_files = [args.streams]
    else:
        stream_files = sorted(args.streams.glob("*.h5"))
    if not stream_files:
        raise FileNotFoundError(f"No .h5 stream files found at {args.streams}")

    print(f"\nProcessing {len(stream_files)} test stream(s)...")
    all_events: list[dict] = []
    for h5_file in stream_files:
        events, _ = process_test_stream(
            h5_path=h5_file,
            model=model,
            prob_threshold=args.threshold,
            merge_gap_s=args.merge_gap,
            min_duration_s=args.min_duration,
            device=device,
        )
        all_events.extend(events)

    # 1. Save official ABC2027 validator-compliant CSV: [stream_id, t_start, t_end, score]
    df_official = pd.DataFrame(all_events, columns=["stream_id", "t_start", "t_end", "score"])
    args.official_output.parent.mkdir(parents=True, exist_ok=True)
    df_official.to_csv(args.official_output, index=False)

    # 2. Save requested [stream_id, start_time, end_time, score] CSV
    df_prompt = df_official.rename(columns={"t_start": "start_time", "t_end": "end_time"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_prompt.to_csv(args.output, index=False)

    print("\n" + "=" * 88)
    print("PREDICTION SUMMARY")
    print("=" * 88)
    print(f"Total predicted fall events : {len(df_official)}")
    if len(df_official) > 0:
        durations_s = df_official["t_end"] - df_official["t_start"]
        print("\nEvents per stream:")
        for sid, cnt in df_official.groupby("stream_id").size().items():
            print(f"  {sid}: {cnt} events")
        print(
            f"\nEvent duration (s) : mean={durations_s.mean():.2f}s, "
            f"median={durations_s.median():.2f}s, min={durations_s.min():.2f}s, max={durations_s.max():.2f}s"
        )
        print(
            f"Prediction score   : mean={df_official['score'].mean():.4f}, "
            f"median={df_official['score'].median():.4f}, min={df_official['score'].min():.4f}, "
            f"max={df_official['score'].max():.4f}"
        )

    print(f"\nSaved [stream_id, start_time, end_time, score] -> {args.output}")
    print(f"Saved [stream_id, t_start, t_end, score]       -> {args.official_output}")

    # Run official ABC2027 format & boundary validation when test streams directory is available
    streams_root = args.streams.parent if args.streams.is_file() else args.streams
    durations_map = stream_durations(streams_root)
    if durations_map:
        validated = validate_predictions(args.official_output, durations_map)
        print(f"[OFFICIAL VALIDATOR PASSED] {len(validated)} predictions verified against stream durations!")


if __name__ == "__main__":
    main()
