import os, sys, gc, io, math, time, json, random, warnings, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------------------------------------------------------
# 1. Google Drive
# ----------------------------------------------------------------------------
try:
    from google.colab import drive
    drive.mount("/content/drive")
    IN_COLAB = True
except ImportError:                       # running locally (smoke tests / VS Code)
    IN_COLAB = False

ROOT_DIR = (Path("/content/drive/MyDrive/ABC27") if IN_COLAB
            else Path(os.environ.get("ABC_ROOT", ".")).resolve())

# ----------------------------------------------------------------------------
# 2. Paths (supports extracted folders AND in-memory reading from .zip archives)
# ----------------------------------------------------------------------------
TRAIN_PKG_DIR     = ROOT_DIR / "abc2027_uwb_train_v1"
TEST_PKG_DIR      = ROOT_DIR / "abc2027_uwb_test_v1"
TRAIN_ZIP_PATH    = ROOT_DIR / "abc2027_uwb_train_v1.zip"
TEST_ZIP_PATH     = ROOT_DIR / "abc2027_uwb_test_v1.zip"
EVENTS_CSV        = TRAIN_PKG_DIR / "streams_events.csv"
COMPOSITION_CSV   = TRAIN_PKG_DIR / "streams_composition.csv"
TRAIN_STREAMS_DIR = TRAIN_PKG_DIR / "streams" / "train"
VAL_STREAMS_DIR   = TRAIN_PKG_DIR / "streams" / "val"
TEST_STREAMS_DIR  = TEST_PKG_DIR / "streams" / "test"

CKPT_DIR       = ROOT_DIR / "checkpoints_cascade"          # persistent (Drive)
CACHE_DIR      = (Path("/content/cache_cascade") if IN_COLAB else ROOT_DIR / "_cache_cascade")  # fast local disk
SUBMISSION_CSV = ROOT_DIR / "submission_colab.csv"
for _d in (CKPT_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def ensure_file_from_zip(disk_path: Path, zip_path: Path, zip_member: str) -> Path:
    """Return disk_path if it exists; otherwise extract zip_member into CACHE_DIR and return that path."""
    if disk_path.is_file():
        return disk_path
    if zip_path.is_file():
        cached = CACHE_DIR / Path(zip_member).name
        if not cached.is_file():
            with zipfile.ZipFile(zip_path, "r") as zf:
                matches = [m for m in zf.namelist() if m == zip_member or m.endswith("/" + Path(zip_member).name)]
                if matches:
                    cached.write_bytes(zf.read(matches[0]))
        if cached.is_file():
            return cached
    return disk_path


EVENTS_CSV = ensure_file_from_zip(EVENTS_CSV, TRAIN_ZIP_PATH, "abc2027_uwb_train_v1/streams_events.csv")
COMPOSITION_CSV = ensure_file_from_zip(COMPOSITION_CSV, TRAIN_ZIP_PATH, "abc2027_uwb_train_v1/streams_composition.csv")

for _name, _p in [("TRAIN_PKG_DIR", TRAIN_PKG_DIR), ("TEST_PKG_DIR", TEST_PKG_DIR),
                  ("TRAIN_ZIP_PATH", TRAIN_ZIP_PATH), ("TEST_ZIP_PATH", TEST_ZIP_PATH),
                  ("EVENTS_CSV", EVENTS_CSV), ("COMPOSITION_CSV", COMPOSITION_CSV)]:
    print(f"[{'OK' if _p.exists() else 'MISSING'}] {_name:16s}: {_p}")


def stream_h5_path(stream_id: str) -> Path | None:
    """Resolve an extracted HDF5 path if present on disk; returns None when the stream lives inside a .zip."""
    split, idx = stream_id.split("_")
    base = {"train": TRAIN_STREAMS_DIR, "val": VAL_STREAMS_DIR, "test": TEST_STREAMS_DIR}[split]
    pkg = TEST_PKG_DIR if split == "test" else TRAIN_PKG_DIR
    candidates = [base / f"stream_{idx}.h5", base / f"{split}_{idx}.h5",
                  pkg / f"{split}_{idx}.h5", ROOT_DIR / f"{split}_{idx}.h5",
                  ROOT_DIR / "streams" / split / f"stream_{idx}.h5"]
    for c in candidates:
        if c.is_file():
            return c
    return None


def load_stream_cir(stream_id: str, h5_path: Path | None = None) -> np.ndarray:
    """Load `cir` (3, T, 368) float32 from an extracted .h5 file OR directly in-memory from the .zip archive."""
    disk = Path(h5_path) if h5_path is not None else stream_h5_path(stream_id)
    if disk is not None and disk.is_file():
        with h5py.File(disk, "r") as h:
            cir = np.nan_to_num(h["cir"][...].astype(np.float32, copy=False))
            serials = h["radar_serials"][:] if "radar_serials" in h else None
    else:
        split, idx = stream_id.split("_")
        zpath = TEST_ZIP_PATH if split == "test" else TRAIN_ZIP_PATH
        pkg_name = "abc2027_uwb_test_v1" if split == "test" else "abc2027_uwb_train_v1"
        if not zpath.is_file():
            raise FileNotFoundError(
                f"No HDF5 file on disk for {stream_id} and zip archive not found at {zpath}"
            )
        target_suffix = f"streams/{split}/stream_{idx}.h5"
        member = f"{pkg_name}/{target_suffix}"
        with zipfile.ZipFile(zpath, "r") as zf:
            names = zf.namelist()
            if member not in names:
                matches = [m for m in names if m.endswith(target_suffix) or m.endswith(f"{split}_{idx}.h5")]
                if not matches:
                    raise FileNotFoundError(f"Member {member} not found inside {zpath}")
                member = matches[0]
            with h5py.File(io.BytesIO(zf.read(member)), "r") as h:
                cir = np.nan_to_num(h["cir"][...].astype(np.float32, copy=False))
                serials = h["radar_serials"][:] if "radar_serials" in h else None

    if serials is not None and len(serials) == 3:
        order = np.argsort(serials)
        if not np.array_equal(order, np.arange(3)):
            cir = cir[order]
    return cir


# ----------------------------------------------------------------------------
# 3. Library imports for the official evaluator (optional, used for validation / format check)
# ----------------------------------------------------------------------------
if (TRAIN_PKG_DIR / "src").exists() and str(TRAIN_PKG_DIR / "src") not in sys.path:
    sys.path.insert(0, str(TRAIN_PKG_DIR / "src"))
try:
    from abc2027 import evaluate as abc_eval
except Exception as _e:                    # evaluator not available -> validation cell degrades gracefully
    abc_eval = None
    print("[WARN] abc2027.evaluate not importable:", _e)

# ----------------------------------------------------------------------------
# 4. Radar constants
# ----------------------------------------------------------------------------
FS_HZ      = 50.0
NUM_RADARS = 3
NUM_BINS   = 368

# 4 spatial bands per radar (inclusive bin ranges 0-65, 75-180, 183-225, 260-367) -> 12 channels
SPATIAL_BANDS = ((0, 66), (75, 181), (183, 226), (260, 368))      # python slice bounds
NUM_BANDS     = len(SPATIAL_BANDS)
NUM_CHANNELS  = NUM_RADARS * NUM_BANDS                             # 12

# ----------------------------------------------------------------------------
# 5. Device / reproducibility / dtype policy (strict float32 everywhere)
# ----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float32
torch.set_default_dtype(torch.float32)
SEED = 1234
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
print("DEVICE:", DEVICE, "| torch", torch.__version__)

# ----------------------------------------------------------------------------
# 6. Pipeline hyper-parameters (single place to tune)
# ----------------------------------------------------------------------------
TRAIN_STREAM_IDS = [f"train_{i:02d}" for i in range(13)]
VAL_STREAM_IDS   = [f"val_{i:02d}" for i in range(5)]
TEST_STREAM_IDS  = [f"test_{i:02d}" for i in range(5)]
USE_VAL_FOR_TRAINING = False      # True -> final fit on train+val (val-tuned thresholds are then optimistic)
FORCE_RETRAIN        = False      # False -> reuse checkpoints found in CKPT_DIR

# Stage 1 (MobileNetV3 proposal)
S1_WINDOW_FRAMES = int(2.0 * FS_HZ)      # 100 frames = 2.0 s
S1_STRIDE_FRAMES = int(0.5 * FS_HZ)      # inference stride 0.5 s
S1_THRESHOLD     = 0.25
S1_STEPS, S1_BATCH, S1_LR = 3000, 64, 2e-3

# Stage 2 (1D-FPN decoder)
S2_STEPS, S2_BATCH, S2_LR = 4000, 16, 1e-3
S2_MIN_BLOCK_S, S2_MAX_BLOCK_S = 10.0, 15.0

# Inference / post-processing
P_FALL_THRESHOLD   = 0.5
SOFT_NMS_IOU       = 0.3
SOFT_NMS_FLOOR     = 0.3      # overwritten by the validation sweep in Cell 6C
