"""Assemble ABC2027_Cascade_Colab.ipynb from the cell sources in ./cells (run: python build_notebook.py)."""
import json
from pathlib import Path

HERE = Path(__file__).parent
CELLS = HERE / "cells"

CELL_SPECS = [
    ("md", """# 9th ABC Challenge - UWB Fall Track: Bi-Level Hybrid Cascade Pipeline
**MobileNetV3 (fast proposal) → 1D-FPN ActionFormer-style decoder (boundary regression) → Temporal DIoU loss → 1D Soft-NMS**

Continuous, unsegmented UWB radar streams: 3 radars · 50 Hz · 368 range bins.

| Cell | Content |
|---|---|
| 1 | Environment, Drive, paths, constants, hyper-parameters |
| 2 | SVD clutter removal · 12 spatial-band features · robust IQR normalisation |
| 3 | Stage 1 - `MobileNetV3ProposalNet` + `extract_suspicious_regions` |
| 4 | Stage 2 - `TemporalFPNDecoder` (stem + 5 dilated residual blocks, FPN, per-frame heads) |
| 5 | `create_regression_targets` + `TemporalDIoULoss` (Focal + 5× splice penalty + 2·DIoU) |
| 6 | Inference engine: `decode_and_nms` (1D Soft-NMS) + cascade runner |
| 6B | Training of both stages (needed to obtain weights; skipped when checkpoints exist) |
| 6C | Validation: Stage-1 recall, Soft-NMS floor sweep, official evaluator |
| 7 | Test streams `test_00 … test_04` → `submission_colab.csv` |

> Expected Drive layout: `MyDrive/ABC27/abc2027_uwb_train_v1/` and `MyDrive/ABC27/abc2027_uwb_test_v1/` (extracted packages). Select a **GPU runtime** (Runtime → Change runtime type)."""),
    ("md", "## Cell 1 - Environment Setup & Paths"),
    ("code", "cell01_setup.py"),
    ("md", """## Cell 2 - Signal Preprocessing & Feature Engineering
* `SVDClutterRemoval(k=2)`: eigen-decomposition (`np.linalg.eigh`) of the Gram matrix `XᵀX` of the RTM, removal of the 2 dominant (static) components. Applied per 10-s block because streams are concatenations of different recordings.
* `extract_12_spatial_bands`: L2 norm of the residual over bins 0-65, 75-180, 183-225, 260-367 for 3 radars → `(12, Seq_Len)`.
* `RobustIQRNormalization`: `(x − median) / IQR` per channel.
* `preprocess_stream`: runs the above for a whole stream and caches the Stage-2 bands plus a float16 log-amplitude RTM (Stage-1 input) on local disk."""),
    ("code", "cell02_preprocessing.py"),
    ("md", """## Cell 3 - Stage 1: Fast Proposal Network (MobileNetV3)
Pure-PyTorch MobileNetV3-Small (inverted residuals, squeeze-and-excitation, h-swish/h-sigmoid) on 2-s RTM chunks `(B, 3, 100, 368)` → fall probability.
`extract_suspicious_regions` binarises `p > 0.25`, merges nearby hits and pads them into 10-15 s blocks."""),
    ("code", "cell03_stage1.py"),
    ("md", """## Cell 4 - Stage 2: Frame-wise 1D-FPN Temporal Decoder (ActionFormer style)
Stem + 5 dilated residual blocks (d = 1, 2, 4, 8, 16; receptive field 377 frames ≈ 7.54 s), a top-down FPN over the block outputs and two per-frame heads:
`p_fall(t)` `(B, L)` (bias init −4.6) and `reg_bounds(t) = (Δs, Δe)` `(B, 2, L)` with a ReLU."""),
    ("code", "cell04_stage2.py"),
    ("md", """## Cell 5 - Ground-Truth Generator & Temporal DIoU Loss
`Total = Focal(α=0.85, γ=2) + Splice_Penalty (5× on false positives in 0.2-s crossfade junctions) + 2.0 · DIoU`,
with `DIoU = 1 − (IoU − CenterDist² / EnclosingWidth²)` computed only where `cls_target == 1`."""),
    ("code", "cell05_loss.py"),
    ("md", """## Cell 6 - Inference Engine & 1D Soft-NMS
`decode_and_nms(p_fall, reg_bounds, t_offset)`: keep `p_fall > 0.5`, `start = t_offset + t − Δs`, `end = t_offset + t + Δe`, linear Soft-NMS (IoU threshold 0.3) → `{stream_id, start_time, end_time, score}`."""),
    ("code", "cell06_inference.py"),
    ("md", """## Cell 6B - Training (Stage 1 then Stage 2)
Needed to produce weights: Stage 1 is trained with BCE on balanced 2-s windows (positives overlap a fall ≥ 0.5 s, 40 % hard negatives); Stage 2 with `TemporalDIoULoss` on random 10-15 s blocks (70 % contain a fall, 15 % hard negatives, 15 % random), splice masks from `streams_composition.csv`.
Checkpoints go to `ABC27/checkpoints_cascade/` and are re-used on the next run (`FORCE_RETRAIN = True` to retrain)."""),
    ("code", "cell06b_training.py"),
    ("md", """## Cell 6C - Validation (optional but recommended)
Runs the full cascade on `val_00 … val_04`, reports Stage-1 recall, sweeps the Soft-NMS score floor and prints the official evaluator report."""),
    ("code", "cell06c_validation.py"),
    ("md", """## Cell 7 - Test Set Evaluation & CSV Generation
`test_00 … test_04` → preprocessing → Stage 1 proposals → Stage 2 FPN decoder → Soft-NMS → `/content/drive/MyDrive/ABC27/submission_colab.csv`."""),
    ("code", "cell07_test_submission.py"),
]


def to_source(text: str):
    lines = text.rstrip("\n").split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def build():
    cells = []
    for kind, payload in CELL_SPECS:
        if kind == "md":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": to_source(payload)})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": to_source((CELLS / payload).read_text(encoding="utf-8"))})
    nb = {"cells": cells,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = HERE.parent / "ABC2027_Cascade_Colab.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", out, f"({len(cells)} cells)")


if __name__ == "__main__":
    build()
