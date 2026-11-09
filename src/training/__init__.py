"""
Physics-Aware Loss Functions & Supervised Training Utilities for UWB Fall Detection.
"""

from .losses import (
    PhysicsAwareCompositeLoss,
    TemporalDIoULoss,
    build_physics_targets_and_masks,
    temporal_diou,
)
from .trainer import SlidingWindowFallDataset, train_sequence_classifier_epoch

__all__ = [
    "PhysicsAwareCompositeLoss",
    "TemporalDIoULoss",
    "build_physics_targets_and_masks",
    "temporal_diou",
    "SlidingWindowFallDataset",
    "train_sequence_classifier_epoch",
]
