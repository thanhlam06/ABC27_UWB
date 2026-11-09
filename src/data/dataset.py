"""
UWB CIR Stream Data Loader & Exploratory Data Analysis (EDA) Utilities
9th ABC Challenge — UWB Fall Detection Track

Provides:
1. Unified loading of continuous UWB Channel Impulse Response (CIR) tensors
   R in R^{3 x T x 368} @ 50.0 Hz from extracted HDF5 files or zip archives.
2. Ground-truth event (`streams_events.csv`) and stream concatenation
   (`streams_composition.csv`) metadata loaders.
3. Quantitative EDA statistics: fall duration distributions across scenarios
   (`Chute01`..`Chute04`), foreground-to-background frame imbalance, mattress
   position quality breakdown (`Dataset 1..3`), and 0.2s crossfade splice density.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

# Physical Radar Constants (Novelda XeThru X4M200 @ 50 Hz)
FS_HZ: float = 50.0
DT_SEC: float = 1.0 / FS_HZ
N_RADARS: int = 3
N_RAW_BINS: int = 368
N_ACTIVE_BINS: int = 100
RADAR_SERIALS: tuple[int, int, int] = (29171, 29444, 30722)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_PKG = PROJECT_ROOT / "abc2027_uwb_train_v1"
DEFAULT_TEST_PKG = PROJECT_ROOT / "abc2027_uwb_test_v1"
DEFAULT_TRAIN_ZIP = PROJECT_ROOT / "abc2027_uwb_train_v1.zip"
DEFAULT_TEST_ZIP = PROJECT_ROOT / "abc2027_uwb_test_v1.zip"


@dataclass
class UWBStreamRecord:
    """Container for a single continuous multi-radar UWB CIR stream."""

    stream_id: str
    cir: np.ndarray  # Shape: (3, T, 368), float32
    timestamps: np.ndarray  # Shape: (T,), float64 seconds
    radar_serials: tuple[int, ...] = RADAR_SERIALS

    @property
    def n_frames(self) -> int:
        return int(self.cir.shape[1])

    @property
    def duration_s(self) -> float:
        return float(self.n_frames) / FS_HZ


def ensure_radar_first_layout(cir: np.ndarray) -> np.ndarray:
    """Ensure UWB CIR tensor has shape (3, T, 368) in contiguous float32."""
    if cir.ndim != 3:
        raise ValueError(f"Expected 3D UWB CIR array, got shape {cir.shape}")
    if cir.shape[0] == N_RADARS and cir.shape[2] == N_RAW_BINS:
        return np.ascontiguousarray(cir, dtype=np.float32)
    if cir.shape[1] == N_RADARS and cir.shape[2] == N_RAW_BINS:
        return np.ascontiguousarray(np.transpose(cir, (1, 0, 2)), dtype=np.float32)
    raise ValueError(f"Unrecognized UWB CIR shape {cir.shape}; expected (3, T, 368).")


def load_uwb_stream(
    split: str,
    index: int,
    train_pkg_dir: Path = DEFAULT_TRAIN_PKG,
    test_pkg_dir: Path = DEFAULT_TEST_PKG,
    train_zip_path: Path = DEFAULT_TRAIN_ZIP,
    test_zip_path: Path = DEFAULT_TEST_ZIP,
) -> UWBStreamRecord:
    """
    Load a continuous UWB stream (`train`, `val`, or `test`) by split and index.
    Automatically falls back to reading from the `.zip` archive if not extracted.
    """
    stream_id = f"{split}_{index:02d}"
    pkg_dir = test_pkg_dir if split == "test" else train_pkg_dir
    zip_path = test_zip_path if split == "test" else train_zip_path
    pkg_prefix = "abc2027_uwb_test_v1" if split == "test" else "abc2027_uwb_train_v1"

    disk_path = pkg_dir / "streams" / split / f"stream_{index:02d}.h5"
    if disk_path.is_file():
        with h5py.File(disk_path, "r") as handle:
            cir = ensure_radar_first_layout(handle["cir"][:])
            t = (
                handle["t"][:].astype(np.float64)
                if "t" in handle
                else np.arange(cir.shape[1], dtype=np.float64) * DT_SEC
            )
            serials = (
                tuple(int(x) for x in handle["radar_serials"][:])
                if "radar_serials" in handle
                else RADAR_SERIALS
            )
        return UWBStreamRecord(stream_id=stream_id, cir=cir, timestamps=t, radar_serials=serials)

    if zip_path.is_file():
        member = f"{pkg_prefix}/streams/{split}/stream_{index:02d}.h5"
        with zipfile.ZipFile(zip_path, "r") as zf:
            if member in zf.namelist():
                raw_bytes = zf.read(member)
                with h5py.File(io.BytesIO(raw_bytes), "r") as handle:
                    cir = ensure_radar_first_layout(handle["cir"][:])
                    t = (
                        handle["t"][:].astype(np.float64)
                        if "t" in handle
                        else np.arange(cir.shape[1], dtype=np.float64) * DT_SEC
                    )
                    serials = (
                        tuple(int(x) for x in handle["radar_serials"][:])
                        if "radar_serials" in handle
                        else RADAR_SERIALS
                    )
                return UWBStreamRecord(stream_id=stream_id, cir=cir, timestamps=t, radar_serials=serials)

    raise FileNotFoundError(
        f"Stream {stream_id} not found at {disk_path} or inside {zip_path}."
    )


def load_events_dataframe(train_pkg_dir: Path = DEFAULT_TRAIN_PKG) -> pd.DataFrame:
    """Load `streams_events.csv` and enrich with computed duration and evaluation flags."""
    csv_path = train_pkg_dir / "streams_events.csv"
    df = pd.read_csv(csv_path)
    df["duration_s"] = df["t_end"] - df["t_start"]
    df["split"] = df["stream_id"].str.split("_").str[0]
    needs_rev = df["needs_review"].astype(str).str.lower().isin(["true", "1", "yes"])
    low_qual = df["quality"].astype(str).str.lower() == "low"
    df["is_ignored"] = needs_rev | low_qual
    df["is_evaluable"] = ~df["is_ignored"]
    return df


def load_composition_dataframe(train_pkg_dir: Path = DEFAULT_TRAIN_PKG) -> pd.DataFrame:
    """Load `streams_composition.csv` with segment durations and split tags."""
    csv_path = train_pkg_dir / "streams_composition.csv"
    df = pd.read_csv(csv_path)
    df["duration_s"] = df["t_end"] - df["t_start"]
    df["split"] = df["stream_id"].str.split("_").str[0]
    return df


def compute_eda_summary_statistics(
    train_pkg_dir: Path = DEFAULT_TRAIN_PKG,
) -> dict[str, Any]:
    """
    Compute key Exploratory Data Analysis (EDA) metrics across the ABC2027 corpus:
    - Fall duration statistics (overall and per subtype `Chute01`..`Chute04`).
    - Evaluable vs. Ignored (`quality=low` / `needs_review`) breakdown per split.
    - Signal quality distribution across mattress datasets (`Dataset 1, 2, 3`).
    - Foreground-to-background class imbalance ratio.
    - Segment splicing frequency (0.2s crossfades per stream).
    """
    events_df = load_events_dataframe(train_pkg_dir)
    comp_df = load_composition_dataframe(train_pkg_dir)

    clean_falls = events_df[events_df["is_evaluable"]]
    adl_segs = comp_df[comp_df["kind"] == "adl"]

    # Stream total durations from composition table
    stream_totals = comp_df.groupby(["split", "stream_id"])["t_end"].max()
    val_total_s = float(stream_totals.loc["val"].sum()) if "val" in stream_totals.index else 7375.80
    val_clean_fall_s = float(clean_falls[clean_falls["split"] == "val"]["duration_s"].sum())
    val_fall_ratio = val_clean_fall_s / max(val_total_s, 1e-6)

    # Quality breakdown by Dataset (Mattress Position 1, 2, 3)
    dataset_col = "dataset" if "dataset" in events_df.columns else "source_path"
    events_df["dataset_pos"] = (
        events_df[dataset_col].astype(str).str.extract(r"(Dataset\s*\d+)", expand=False).fillna("Unknown")
    )
    quality_by_dataset = (
        events_df.groupby(["dataset_pos", "quality"]).size().unstack(fill_value=0).to_dict(orient="index")
    )

    subtype_stats = {}
    col_sub = "fall_type" if "fall_type" in clean_falls.columns else "source_path"
    for st, grp in clean_falls.groupby(col_sub):
        subtype_stats[str(st)] = {
            "count": int(len(grp)),
            "mean_s": round(float(grp["duration_s"].mean()), 3),
            "std_s": round(float(grp["duration_s"].std()), 3),
            "median_s": round(float(grp["duration_s"].median()), 3),
        }

    splices_per_stream = comp_df.groupby("stream_id").size() - 1

    return {
        "total_fall_annotations": int(len(events_df)),
        "evaluable_falls_total": int(len(clean_falls)),
        "ignored_falls_total": int(events_df["is_ignored"].sum()),
        "evaluable_falls_train": int((clean_falls["split"] == "train").sum()),
        "evaluable_falls_val": int((clean_falls["split"] == "val").sum()),
        "clean_fall_duration_mean_s": round(float(clean_falls["duration_s"].mean()), 3),
        "clean_fall_duration_std_s": round(float(clean_falls["duration_s"].std()), 3),
        "clean_fall_duration_median_s": round(float(clean_falls["duration_s"].median()), 3),
        "clean_fall_duration_min_s": round(float(clean_falls["duration_s"].min()), 3),
        "clean_fall_duration_max_s": round(float(clean_falls["duration_s"].max()), 3),
        "adl_segment_duration_mean_s": round(float(adl_segs["duration_s"].mean()), 3),
        "val_total_duration_s": round(val_total_s, 2),
        "val_foreground_fraction": round(val_fall_ratio, 5),
        "val_imbalance_ratio_1_to": round((1.0 - val_fall_ratio) / max(val_fall_ratio, 1e-8), 1),
        "mean_splices_per_stream": round(float(splices_per_stream.mean()), 1),
        "subtype_duration_stats": subtype_stats,
        "quality_by_dataset": quality_by_dataset,
    }
