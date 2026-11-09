"""
Deep Learning Model Architectures for UWB Fall Detection:
- `ResNetSequenceClassifier`: ResNet-18 2D backbone + Range Pooling + 1D Temporal Head (w1 = 0.45).
- `ResNetV1Standard`: ResNet-18 2D backbone + Global Spatio-Temporal GAP + FC Head (w2 = 0.40).
- `EfficientNetSequenceClassifier`: EfficientNet-B0 MBConv/SE backbone + 1D Temporal Head (w3 = 0.15).
- `TripleEnsembleModel`: Unified SOTA Triple-Ensemble container & checkpoint loader.
- `FrameWise1DTCN` & `MobileNetV3ProposalNet`: 1D-TCN and Cascade Proposal architectures.
"""

from .tcn_cascade import FrameWise1DTCN, MobileNetV3ProposalNet
from .triple_ensemble import (
    ENSEMBLE_WEIGHTS,
    EfficientNetSequenceClassifier,
    ResNetSequenceClassifier,
    ResNetV1Standard,
    TripleEnsembleModel,
    load_triple_ensemble,
)

__all__ = [
    "ENSEMBLE_WEIGHTS",
    "ResNetSequenceClassifier",
    "ResNetV1Standard",
    "EfficientNetSequenceClassifier",
    "TripleEnsembleModel",
    "load_triple_ensemble",
    "FrameWise1DTCN",
    "MobileNetV3ProposalNet",
]
