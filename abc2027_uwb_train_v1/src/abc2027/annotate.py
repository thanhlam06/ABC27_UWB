"""Proposed fall-interval detector for L2 step A, revision 2.

The fall is the last continuous motion burst that precedes the new static
profile. Earlier bursts are not chained. Times are seconds from the first CIR
timestamp. Frames that share an integer second are placed at the centres of
equal slots inside that second. The half-slot width is the timing uncertainty,
provided the file order within the second is the acquisition order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

_CIR_LINE = re.compile(
    rb'^\{"serial_number":(?P<serial>\d+),"timestamp":(?P<ts>\d+),"data":\[(?P<data>.*)]\}\s*$'
)

GRID_STEP_S = 0.02
SMOOTH_S = 0.4
SPIKE_MAX_S = 0.3
EDGE_FRACTION = 0.15
STATE_HIGH = 0.65
ONSET_VISIBLE_S = 1.0
FLOOR_VISIBLE_S = 1.0
MIN_EVENT_S = 1.0
SINGLE_RADAR_SHARE = 0.70
ONSET_SPREAD_S = 1.0
# Strongest-radar residual below this value is low quality.
# 5% of the dataset 1+2 median strongest residual (1.024e-4).
QUALITY_RESIDUAL_GATE = 5.12e-6
LOW_QUALITY_RATIO = 0.05


@dataclass
class RadarTrack:
    serial: int
    time_s: np.ndarray
    cir: np.ndarray
    motion: np.ndarray
    origin_unix: float = 0.0
    radar_origin_unix: float = 0.0
    median_frames_per_second: float = 0.0


@dataclass
class FallProposal:
    t_start: float
    t_end: float
    t_peak: float
    t_settle: float
    t_last: float
    start_visible: bool
    floor_visible: bool
    review: str
    reasons: tuple[str, ...] = field(default_factory=tuple)
    anchor_serial: int = 0
    single_radar_share: float = 0.0
    onset_spread_s: float = 0.0
    quality: str = "normal"
    n_qualified_radars: int = 0
    grid: np.ndarray | None = None
    fused_motion: np.ndarray | None = None
    per_radar_motion: dict | None = None
    fused_state: np.ndarray | None = None


def frame_times(timestamps: np.ndarray, t0: float) -> np.ndarray:
    """Seconds from ``t0``. Ties share a second and sit at slot centres."""
    timestamps = np.asarray(timestamps)
    out = np.empty(len(timestamps), dtype=np.float64)
    index = 0
    while index < len(timestamps):
        end = index + 1
        while end < len(timestamps) and timestamps[end] == timestamps[index]:
            end += 1
        count = end - index
        slots = (np.arange(count) + 0.5) / count
        out[index:end] = (float(timestamps[index]) - t0) + slots
        index = end
    return out


def _frames_per_second(timestamps: np.ndarray) -> float:
    counts = []
    index = 0
    while index < len(timestamps):
        end = index + 1
        while end < len(timestamps) and timestamps[end] == timestamps[index]:
            end += 1
        counts.append(end - index)
        index = end
    if not counts:
        return 0.0
    return float(np.median(counts))


def slot_uncertainty_s(frames_in_second: int) -> float:
    """Half a slot, in seconds. About 0.01 s when a second holds ~50 frames."""
    if frames_in_second <= 0:
        return 0.5
    return 0.5 / frames_in_second


def load_radar_tracks(path) -> list[RadarTrack]:
    """Stream one JSONL file and keep CIR frames, per radar, in file order."""
    grouped: dict[int, list[tuple[float, np.ndarray]]] = {}
    with open(path, "rb") as handle:
        for line in handle:
            match = _CIR_LINE.match(line.strip())
            if match is None:
                continue
            serial = int(match.group("serial"))
            timestamp = float(match.group("ts"))
            cir = np.fromstring(match.group("data"), sep=",")
            grouped.setdefault(serial, []).append((timestamp, cir))
    if not grouped:
        raise ValueError(f"no CIR frames in {path}")
    t0 = min(samples[0][0] for samples in grouped.values())
    tracks = []
    for serial, samples in sorted(grouped.items()):
        timestamps = np.array([item[0] for item in samples], dtype=np.float64)
        cir = np.vstack([item[1] for item in samples])
        times = frame_times(timestamps, t0)
        delta = np.diff(cir, axis=0)
        motion = np.einsum("ij,ij->i", delta, delta)
        tracks.append(
            RadarTrack(
                serial=serial,
                time_s=times,
                cir=cir,
                motion=motion,
                origin_unix=t0,
                radar_origin_unix=float(timestamps[0]),
                median_frames_per_second=_frames_per_second(timestamps),
            )
        )
    return tracks


def signal_quality(tracks: list[RadarTrack]) -> dict[int, dict[str, float]]:
    """Absolute residual energy and motion energy, per radar."""
    quality = {}
    for track in tracks:
        centre = np.median(track.cir, axis=0)
        residual = np.mean(np.sum((track.cir - centre) ** 2, axis=1))
        motion = float(np.mean(track.motion)) if len(track.motion) else 0.0
        quality[track.serial] = {"residual": float(residual), "motion": motion}
    return quality


def _smooth(values: np.ndarray, step_s: float, window_s: float) -> np.ndarray:
    width = max(1, int(round(window_s / step_s)))
    if width % 2 == 0:
        width += 1
    kernel = np.ones(width, dtype=np.float64) / width
    padded = np.pad(values, width // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _grid_for(tracks: list[RadarTrack], step_s: float) -> np.ndarray:
    t_last = max(float(track.time_s[-1]) for track in tracks)
    grid = np.arange(0.0, max(t_last, step_s), step_s)
    if len(grid) < 3:
        grid = np.linspace(0.0, max(t_last, step_s), 3)
    return grid


def _resample(track: RadarTrack, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-frame CIR and motion on ``grid``."""
    index = np.searchsorted(track.time_s, grid, side="left")
    index = np.clip(index, 1, len(track.time_s) - 1)
    previous = index - 1
    take_prev = np.abs(grid - track.time_s[previous]) <= np.abs(grid - track.time_s[index])
    index = np.where(take_prev, previous, index)
    cir = track.cir[index]
    motion_t = track.time_s[1:]
    if len(motion_t) < 2:
        motion = np.zeros(len(grid))
    else:
        motion = np.interp(grid, motion_t, track.motion, left=float(track.motion[0]), right=float(track.motion[-1]))
    return cir, motion


def _suppress_spikes(
    motions: dict[int, np.ndarray],
    states: dict[int, np.ndarray],
    step_s: float,
) -> dict[int, np.ndarray]:
    """Drop brief peaks that live on one radar and do not change the static profile."""
    cleaned = {serial: values.copy() for serial, values in motions.items()}
    max_len = max(1, int(round(SPIKE_MAX_S / step_s)))
    gates = {serial: float(np.percentile(values, 90)) for serial, values in motions.items()}
    for serial, values in motions.items():
        gate = gates[serial]
        above = values >= gate
        start = 0
        while start < len(values):
            if not above[start]:
                start += 1
                continue
            stop = start + 1
            while stop < len(values) and above[stop]:
                stop += 1
            others = False
            for other, other_values in motions.items():
                if other == serial:
                    continue
                if float(np.max(other_values[start:stop])) >= gates[other]:
                    others = True
                    break
            state_change = float(np.ptp(states[serial][start:stop])) if stop > start else 0.0
            if (stop - start) <= max_len and not others and state_change < 0.15:
                floor = float(np.percentile(values, 20))
                cleaned[serial][start:stop] = floor
            start = stop
    return cleaned


def _profiles(cir: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standing profile from the first second, floor profile from the last 1.5 s."""
    before_mask = grid <= 1.0
    after_mask = grid >= float(grid[-1]) - 1.5
    if int(before_mask.sum()) < 5:
        before_mask[: max(5, len(grid) // 5)] = True
    if int(after_mask.sum()) < 5:
        after_mask[-max(5, len(grid) // 5) :] = True
    return np.median(cir[before_mask], axis=0), np.median(cir[after_mask], axis=0)


def _state(cir: np.ndarray, before: np.ndarray, after: np.ndarray) -> np.ndarray:
    before_dist = np.sum((cir - before) ** 2, axis=1)
    after_dist = np.sum((cir - after) ** 2, axis=1)
    return before_dist / (before_dist + after_dist + 1e-18)


def _transition_bounds(state: np.ndarray, motion: np.ndarray) -> tuple[int, int, int]:
    """Bounds of the state step that coincides with motion. A dip stops the expansion."""
    width = max(1, int(round(1.5 / GRID_STEP_S)))
    if len(state) <= width + 2:
        peak = int(np.argmax(motion))
        return peak, peak, peak
    rise = state[width:] - state[:-width]
    motion_sum = np.convolve(motion, np.ones(width), mode="valid")
    count = min(len(rise), len(motion_sum))
    rise = rise[:count]
    motion_sum = motion_sum[:count]
    scale = float(np.median(motion_sum)) + 1e-18
    score = rise * np.log1p(motion_sum / scale)
    eligible = rise >= 0.12
    if np.any(eligible):
        score = np.where(eligible, score, -np.inf)
    index = int(np.argmax(score))
    span_end = min(len(state) - 1, index + width)
    start_level = float(state[index])
    end_level = float(state[span_end])
    delta = end_level - start_level
    left = index
    right = span_end
    if delta > 0.05:
        low = start_level + 0.2 * delta
        high = start_level + 0.8 * delta
        above_low = np.flatnonzero(state[index : span_end + 1] >= low)
        below_high = np.flatnonzero(state[index : span_end + 1] <= high)
        if above_low.size:
            left = index + int(above_low[0])
        if below_high.size:
            right = index + int(below_high[-1])
        right = max(right, left)
    height = float(np.max(motion[left : right + 1]))
    baseline = float(np.percentile(motion, 30))
    level = baseline + EDGE_FRACTION * max(height - baseline, 0.0)
    left_cap = int(round(3.0 / GRID_STEP_S))
    right_cap = int(round(2.0 / GRID_STEP_S))
    origin = left
    while left > 0 and motion[left - 1] >= level and (origin - left) < left_cap:
        left -= 1
    while right < len(motion) - 1 and motion[right + 1] >= level and (right - origin) < right_cap:
        right += 1
    peak = left + int(np.argmax(motion[left : right + 1]))
    return left, right, peak


def is_low_quality(residuals: dict[int, float], gate: float = QUALITY_RESIDUAL_GATE) -> bool:
    """True when the strongest radar stays under the residual gate."""
    if not residuals or gate <= 0:
        return False
    return max(residuals.values()) < gate


def propose_fall(tracks: list[RadarTrack], duration_s: float) -> FallProposal:
    """Place the fall at the last motion burst before the new static profile."""
    grid = _grid_for(tracks, GRID_STEP_S)
    cirs = {}
    motions = {}
    for track in tracks:
        cir, motion = _resample(track, grid)
        cirs[track.serial] = cir
        motions[track.serial] = motion
    states = {}
    for serial, cir in cirs.items():
        before, after = _profiles(cir, grid)
        states[serial] = _state(cir, before, after)
    cleaned = _suppress_spikes(motions, states, GRID_STEP_S)
    for serial in list(states):
        before, after = _profiles(cirs[serial], grid)
        states[serial] = _smooth(_state(cirs[serial], before, after), GRID_STEP_S, SMOOTH_S)
        cleaned[serial] = _smooth(cleaned[serial], GRID_STEP_S, SMOOTH_S)
    changes = {}
    for serial, cir in cirs.items():
        before, after = _profiles(cir, grid)
        changes[serial] = float(np.sum((after - before) ** 2))
    anchor = max(changes, key=changes.get)
    fused_state = states[anchor]
    fused_motion = np.median(
        np.vstack([values / (float(np.percentile(values, 75)) + 1e-18) for values in cleaned.values()]),
        axis=0,
    )
    left, right, peak = _transition_bounds(states[anchor], cleaned[anchor])
    t_start = float(grid[left])
    t_end = float(grid[right])
    t_last = float(grid[-1])
    residuals = {serial: values["residual"] for serial, values in signal_quality(tracks).items()}
    qualified = [serial for serial, residual in residuals.items() if residual >= QUALITY_RESIDUAL_GATE]
    onsets = []
    for serial in qualified:
        other_left, _other_right, _other_peak = _transition_bounds(states[serial], cleaned[serial])
        onsets.append(float(grid[other_left]))
    spread = (max(onsets) - min(onsets)) if len(onsets) >= 2 else 0.0
    low_quality = is_low_quality(residuals)
    floor_visible = (t_last - t_end) >= FLOOR_VISIBLE_S and fused_state[-1] >= STATE_HIGH
    start_visible = t_start >= ONSET_VISIBLE_S
    reasons = []
    if not start_visible:
        reasons.append("start_not_visible")
    if not floor_visible:
        reasons.append("floor_not_visible")
    if (t_end - t_start) < MIN_EVENT_S:
        reasons.append("event_under_1s")
    if spread > ONSET_SPREAD_S:
        reasons.append("radar_onset_disagree")
    return FallProposal(
        t_start=t_start,
        t_end=t_end,
        t_peak=float(grid[peak]),
        t_settle=float(grid[right]),
        t_last=t_last,
        start_visible=start_visible,
        floor_visible=floor_visible,
        review="needs_review" if reasons else "keep",
        reasons=tuple(reasons),
        anchor_serial=anchor,
        onset_spread_s=spread,
        quality="low" if low_quality else "normal",
        n_qualified_radars=len(qualified),
        grid=grid,
        fused_motion=fused_motion,
        per_radar_motion=cleaned,
        fused_state=fused_state,
    )
