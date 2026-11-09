"""
Inference & Prediction Pipeline for ABC2027 Test Set using Best Trained 1D-TCN Model + Coarse-to-Fine Refinement
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import uniform_filter1d

# Setup paths
PROJECT_ROOT = Path(r"D:\Desktop\ABC27")
TRAIN_PKG = PROJECT_ROOT / "abc2027_uwb_train_v1"
TEST_PKG = PROJECT_ROOT / "abc2027_uwb_test_v1"
TAL_DIR = PROJECT_ROOT / "uwb_tal_pipeline"

if str(TRAIN_PKG / "src") not in sys.path:
    sys.path.insert(0, str(TRAIN_PKG / "src"))
if str(TAL_DIR) not in sys.path:
    sys.path.insert(0, str(TAL_DIR))

from abc2027.background import subtract_background
from abc2027.evaluate import stream_durations, validate_predictions
from module2_tcn_focal import FrameWise1DTCN
from module3_boundary_refinement import extract_coarse_proposals, refine_event_boundaries

FS_HZ = 50.0
BACKGROUND_TAU_S = 3.0


def load_model(weights_path: Path) -> FrameWise1DTCN:
    """Load the pre-trained FrameWise1DTCN model checkpoint."""
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    model = FrameWise1DTCN(in_channels=3, hidden_channels=24, dilations=(1, 2, 4, 8, 16))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def process_stream(
    h5_path: Path,
    model: FrameWise1DTCN,
    prob_threshold: float = 0.50,
) -> list[dict]:
    """Process a single test h5 stream through Background subtraction, 1D-TCN, and Boundary Refinement."""
    stream_id = "test_" + h5_path.stem.split("_")[1]
    print(f"--> Processing {stream_id} ({h5_path.name})...")

    with h5py.File(h5_path, "r") as f:
        cir = f["cir"][:]  # Shape (3, T, 368)

    # 1. Background subtraction (EMA 3s)
    res = subtract_background(cir, BACKGROUND_TAU_S)

    # 2. Extract 1D L2 Energy Envelope across 368 bins for each radar -> Shape (3, T)
    env_l2 = np.linalg.norm(res.astype(np.float64), ord=2, axis=2).astype(np.float32)

    # 3. Robust scaling for model input
    med = np.median(env_l2, axis=1, keepdims=True)
    scale = np.percentile(env_l2, 99.2, axis=1, keepdims=True) - med + 1e-6
    norm_env = np.maximum((env_l2 - med) / scale, 0.0).astype(np.float32)

    # 4. Multi-radar fused envelope for gradient boundary refinement E(t)
    fused_env = uniform_filter1d(env_l2.mean(axis=0), size=7).astype(np.float32)

    # 5. Model Inference in chunked/streaming windows to prevent memory exhaustion
    total_len = norm_env.shape[1]
    chunk_size = 10_000
    overlap = 500
    frame_probs = np.zeros(total_len, dtype=np.float32)

    with torch.no_grad():
        for start_idx in range(0, total_len, chunk_size):
            end_idx = min(start_idx + chunk_size + overlap, total_len)
            chunk = norm_env[:, start_idx:end_idx]
            x_tensor = torch.from_numpy(chunk[None, :, :])
            probs = model(x_tensor).squeeze(0).cpu().numpy()

            actual_end = min(start_idx + chunk_size, total_len)
            valid_len = actual_end - start_idx
            frame_probs[start_idx:actual_end] = probs[:valid_len]

    # 6. Coarse Proposals from p(t)
    coarse_events = extract_coarse_proposals(
        frame_probs=frame_probs,
        threshold=prob_threshold,
        merge_gap_s=0.60,
        min_duration_s=0.60,
        fs_hz=FS_HZ,
    )

    # 7. Fine Boundary Refinement using gradient dE/dt
    refined_events = refine_event_boundaries(
        coarse_events=coarse_events,
        energy_envelope_1d=fused_env,
        fs_hz=FS_HZ,
    )

    print(f"    Found {len(refined_events)} refined events for {stream_id}.")

    # Format events to submission schema
    output_events = []
    for ev in refined_events:
        t_s = float(ev["t_start_fine"])
        t_e = float(ev["t_end_fine"])
        score = float(ev["score"])
        # Ensure duration is plausible (between 0.6s and 3.5s)
        if t_e - t_s < 0.6:
            t_e = t_s + 1.8
        elif t_e - t_s > 3.5:
            t_e = t_s + 2.5

        output_events.append({
            "stream_id": stream_id,
            "t_start": round(t_s, 2),
            "t_end": round(t_e, 2),
            "score": round(score, 4),
            "t_alarm": round(t_s, 2),
        })

    return output_events


def main() -> None:
    weights_path = TAL_DIR / "outputs" / "best_1d_tcn_model.pth"
    test_streams_dir = TEST_PKG / "streams" / "test"
    output_csv = TEST_PKG / "submission_best_1d_tcn.csv"

    print("=" * 80)
    print("PREDICTING TEST SET WITH BEST 1D-TCN TEMPORAL ACTION LOCALIZATION MODEL")
    print("=" * 80)
    print(f"Loading checkpoint from: {weights_path}")
    model = load_model(weights_path)

    all_test_events = []
    stream_files = sorted(test_streams_dir.glob("stream_*.h5"))
    print(f"Found {len(stream_files)} test streams to predict.")

    for sf in stream_files:
        events = process_stream(sf, model=model, prob_threshold=0.50)
        all_test_events.extend(events)

    df_submission = pd.DataFrame(all_test_events)
    df_submission.to_csv(output_csv, index=False)
    print("=" * 80)
    print(f"Successfully saved predictions to: {output_csv}")
    print(f"Total detected events: {len(df_submission)}")
    print(df_submission.groupby("stream_id").size())
    print("\nDuration statistics:")
    print((df_submission["t_end"] - df_submission["t_start"]).describe())
    print("\nScore statistics:")
    print(df_submission["score"].describe())

    # Official validation check
    durations = stream_durations(TEST_PKG / "streams")
    valid_records = validate_predictions(output_csv, durations)
    print(f"\n[VALIDATION PASSED] All {len(valid_records)} records strictly conform to ABC2027 rules!")


if __name__ == "__main__":
    main()
