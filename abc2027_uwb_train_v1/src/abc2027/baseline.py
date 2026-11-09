"""Two lightweight fall detectors. Background removal is internal to the baseline."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from abc2027.background import subtract_background
from abc2027.evaluate import temporal_iou

RATE_HZ = 50.0
STEP_S = 1.0 / RATE_HZ
BACKGROUND_TAU_S = 3.0
WINDOW = 200
HOP = 50


def frame_energy(cir: np.ndarray) -> np.ndarray:
    """Mean radar energy of squared CIR, one value per time step."""
    return np.mean(np.sum(cir.astype(np.float64) ** 2, axis=2), axis=0)


def runs_to_events(mask: np.ndarray, scores: np.ndarray, min_duration: float, merge_gap: float) -> list[dict]:
    """Turn a boolean mask into events. Nearby runs are merged. Short runs are dropped."""
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return []
    cuts = np.flatnonzero(np.diff(indices) > 1)
    starts = np.concatenate([[indices[0]], indices[cuts + 1]])
    stops = np.concatenate([indices[cuts], [indices[-1]]])
    merged: list[list[float]] = []
    for start, stop in zip(starts, stops):
        if merged and (start - merged[-1][1]) * STEP_S <= merge_gap:
            merged[-1][1] = float(stop)
            merged[-1][2] = max(merged[-1][2], float(np.max(scores[int(merged[-1][0]) : int(stop) + 1])))
        else:
            piece = scores[int(start) : int(stop) + 1]
            merged.append([float(start), float(stop), float(np.max(piece)) if len(piece) else 0.0])
    events = []
    for start, stop, score in merged:
        if (stop - start + 1) * STEP_S < min_duration:
            continue
        t_start = start * STEP_S
        events.append(
            {
                "t_start": t_start,
                "t_end": (stop + 1) * STEP_S,
                "score": score,
                "t_alarm": t_start,
            }
        )
    return events


def energy_events(cir: np.ndarray, percentile: float, min_duration: float, merge_gap: float) -> list[dict]:
    """Threshold residual energy after a 3 s exponential background removal."""
    residual = subtract_background(cir, BACKGROUND_TAU_S)
    energy = frame_energy(residual)
    level = float(np.percentile(energy, percentile))
    if level <= 0.0:
        return []
    return runs_to_events(energy >= level, energy / level, min_duration, merge_gap)


def window_features(block: np.ndarray) -> np.ndarray:
    """Motion features after removing the window mean, so a static scene cancels."""
    residual = block.astype(np.float64)
    residual -= np.mean(residual, axis=1, keepdims=True)
    frame = np.mean(residual * residual, axis=2)
    return np.concatenate([frame.mean(axis=1), frame.std(axis=1), frame.max(axis=1)])


def iter_windows(count: int) -> list[tuple[int, int]]:
    if count < WINDOW:
        return []
    return [(start, start + WINDOW) for start in range(0, count - WINDOW + 1, HOP)]


def labels_for_windows(times: list[tuple[float, float]], events: list[tuple[float, float]]) -> np.ndarray:
    labels = np.zeros(len(times), dtype=np.float64)
    for index, (start, end) in enumerate(times):
        if any(temporal_iou(start, end, event_start, event_end) >= 0.1 for event_start, event_end in events):
            labels[index] = 1.0
    return labels


def fit_window_model(features: np.ndarray, labels: np.ndarray) -> Pipeline:
    """Scale first. Raw CIR features are tiny and otherwise stop the fit immediately."""
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200, class_weight="balanced")),
        ]
    )
    model.fit(features, labels)
    return model


def events_from_windows(
    times: list[tuple[float, float]], scores: np.ndarray, threshold: float
) -> list[dict]:
    active = [
        (start, end, float(score))
        for (start, end), score in zip(times, scores)
        if score >= threshold
    ]
    if not active:
        return []
    groups = [list(active[0])]
    for start, end, score in active[1:]:
        if start - groups[-1][1] <= HOP * STEP_S:
            groups[-1][1] = end
            groups[-1][2] = max(groups[-1][2], score)
        else:
            groups.append([start, end, score])
    return [
        {"t_start": start, "t_end": end, "score": score, "t_alarm": end if end - start <= WINDOW * STEP_S else start + WINDOW * STEP_S}
        for start, end, score in groups
    ]
