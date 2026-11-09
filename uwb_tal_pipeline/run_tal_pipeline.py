"""
End-to-End Orchestrator: Frame-wise Segmentation (1D-TCN) & Coarse-to-Fine Boundary Refinement
9th ABC Challenge - UWB Fall Track (3-Module TAL Pipeline)

Executes sequentially:
- Module 1: Data Representation & SVD(K=2) Clutter Removal -> (3, 30000) 1D Energy Envelope
- Module 2: Frame-wise 1D-TCN in PyTorch with Binary Focal Loss -> p(t) in [0, 1]
- Module 3: Coarse-to-Fine Boundary Refinement via 1st Derivative dE(t)/dt & tIoU Evaluation
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

TAL_DIR = Path(__file__).resolve().parent
if str(TAL_DIR) not in sys.path:
    sys.path.insert(0, str(TAL_DIR))

from module1_svd_envelope import run_module1_svd_representation
from module2_tcn_focal import run_module2_tcn_segmentation
from module3_boundary_refinement import (
    extract_coarse_proposals,
    refine_event_boundaries,
    evaluate_tal_predictions,
)


def main() -> None:
    output_dir = TAL_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Module 1: (30000, 3, 368) -> SVD(K=2) -> (3, 30000) 1D Energy Envelope
    m1_data = run_module1_svd_representation(seed=42, k_components=2)

    # Module 2: PyTorch 1D-TCN + Binary Focal Loss -> Frame-wise p(t) in [0, 1]
    m2_out = run_module2_tcn_segmentation(
        norm_envelope=m1_data.norm_envelope,
        frame_targets=m1_data.frame_targets,
        seed=42,
        epochs=65,
        coarse_margin_frames=22,
    )

    # Module 3: Coarse Proposal (p(t) >= 0.5) -> Fine Gradient Refinement (dE/dt)
    coarse_events = extract_coarse_proposals(
        frame_probs=m2_out.frame_probs,
        threshold=0.50,
        merge_gap_s=0.50,
        min_duration_s=0.80,
    )
    refined_events = refine_event_boundaries(
        coarse_events=coarse_events,
        energy_envelope_1d=m1_data.fused_envelope,
    )

    per_event_df, summary_df = evaluate_tal_predictions(
        refined_events=refined_events,
        events_df=m1_data.events_df,
        output_dir=output_dir,
        frame_probs=m2_out.frame_probs,
        fused_envelope=m1_data.fused_envelope,
    )

    # Verify assertions
    assert m1_data.energy_envelope.shape == (3, 30000), "Module 1 shape must be (3, 30000)"
    assert m2_out.frame_probs.shape == (30000,), "Module 2 p(t) shape must be (30000,)"
    assert float(summary_df.iloc[2]["F1 @ θ=0.5"]) >= 0.99, "Fine refinement must solve F1 @ tIoU >= 0.5"
    assert float(summary_df.iloc[2]["Mean tIoU"]) >= 0.85, "Mean tIoU after refinement must be >= 0.85"

    payload = {
        "per_event_comparison": per_event_df.to_dict(orient="records"),
        "benchmark_summary": summary_df.to_dict(orient="records"),
        "refined_events": refined_events,
    }
    (output_dir / "tal_metrics_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
