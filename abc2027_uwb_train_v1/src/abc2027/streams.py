"""Plan continuous streams from one split. Does not read radar samples."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SEED = 2027
STREAM_S = 30.0 * 60.0
FADE_S = 0.2
MARGIN_BEFORE = (3.0, 8.0)
MARGIN_AFTER = (5.0, 10.0)
ADL_CHUNK = (8.0, 20.0)
ADL_USES = {"train": 3, "val": 2, "test": 2}
SPLIT_ORDER = ("train", "val", "test")


@dataclass
class Segment:
    kind: str
    path: str
    label: str
    dataset: str
    source_t0: float
    source_t1: float
    event_t0: float | None = None
    event_t1: float | None = None
    fall_type: str = ""
    quality: str = ""
    needs_review: bool = False
    ignore: bool = False
    segment_id: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.source_t1 - self.source_t0)


@dataclass
class StreamPlan:
    split: str
    segments: list[Segment] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return sum(segment.duration for segment in self.segments)


@dataclass
class SplitPlan:
    split: str
    density_per_hour: float
    streams: list[StreamPlan]
    falls: list[Segment]
    adl_seconds: float

    @property
    def duration(self) -> float:
        return sum(stream.duration for stream in self.streams)


def _clip(draw_low: float, draw_high: float, available: float, rng: np.random.Generator) -> float:
    draw = float(rng.uniform(draw_low, draw_high))
    return float(min(max(available, 0.0), max(draw, 0.0)))


def fall_segment(row: dict, rng: np.random.Generator, split: str) -> Segment:
    t_start = float(row["t_start"])
    t_end = float(row["t_end"])
    t_last = float(row["t_last"])
    before = _clip(*MARGIN_BEFORE, t_start, rng)
    after = _clip(*MARGIN_AFTER, t_last - t_end, rng)
    ignore = split == "test" and (row["quality"] == "low" or str(row["needs_review"]).lower() == "true")
    return Segment(
        kind="fall",
        path=row["path"],
        label=row["fall_type"],
        dataset=row["dataset"],
        source_t0=t_start - before,
        source_t1=t_end + after,
        event_t0=t_start,
        event_t1=t_end,
        fall_type=row["fall_type"],
        quality=row["quality"],
        needs_review=str(row["needs_review"]).lower() == "true",
        ignore=ignore,
        segment_id=row["path"],
    )


def adl_segments(row: dict, uses: int, rng: np.random.Generator) -> list[Segment]:
    duration = float(row["duration_s"])
    pieces: list[Segment] = []
    seen: set[tuple[int, int]] = set()
    for use in range(uses):
        if duration <= 0:
            continue
        low, high = ADL_CHUNK
        offset = float(rng.uniform(0.0, min(duration, high)))
        edges = [0.0]
        cursor = 0.0
        while cursor < duration - 1e-6:
            cursor = min(duration, cursor + float(rng.uniform(low, high)))
            edges.append(cursor)
        shifted = sorted({(edge + offset) % duration for edge in edges[:-1]})
        points = [0.0] + [point for point in shifted if 1e-3 < point < duration - 1e-3] + [duration]
        for start, stop in zip(points, points[1:]):
            if stop - start < 0.5:
                continue
            key = (int(round(start * 1000)), int(round(stop * 1000)))
            if key in seen:
                continue
            seen.add(key)
            pieces.append(
                Segment(
                    kind="adl",
                    path=row["path"],
                    label=row["label"],
                    dataset="",
                    source_t0=start,
                    source_t1=stop,
                    segment_id=f"{row['path']}#{use}:{key[0]}-{key[1]}",
                )
            )
    return pieces


def other_segment(row: dict) -> Segment:
    return Segment(
        kind="other",
        path=row["path"],
        label=row["label"],
        dataset=f"Dataset {row['position']}" if row.get("position") else "",
        source_t0=0.0,
        source_t1=float(row["duration_s"] or 0.0),
        segment_id=row["path"],
    )


def _select_adl(pieces: list[Segment], target_s: float, rng: np.random.Generator) -> list[Segment]:
    order = list(pieces)
    rng.shuffle(order)
    chosen: list[Segment] = []
    total = 0.0
    for piece in order:
        if total >= target_s:
            break
        chosen.append(piece)
        total += piece.duration
    return chosen


def _pack(split: str, segments: list[Segment]) -> list[StreamPlan]:
    streams: list[StreamPlan] = []
    current = StreamPlan(split=split)
    for segment in segments:
        if current.segments and current.duration + segment.duration > STREAM_S:
            streams.append(current)
            current = StreamPlan(split=split)
        current.segments.append(segment)
    if current.segments:
        streams.append(current)
    return streams


def _rng(split: str) -> np.random.Generator:
    index = SPLIT_ORDER.index(split)
    sequence = np.random.SeedSequence(SEED).spawn(len(SPLIT_ORDER))[index]
    return np.random.Generator(np.random.PCG64(sequence))


def prepare_split(
    split: str,
    falls: list[dict],
    adl: list[dict],
    others: list[dict],
    rng: np.random.Generator | None = None,
) -> dict:
    """Sample clips and the full ADL budget. Density is computed afterwards.

    ``rng`` overrides the public seed. The official test plan uses a generator
    whose seed is kept outside the public package.
    """
    rng = _rng(split) if rng is None else rng
    fall_rows = sorted(falls, key=lambda row: row["path"])
    adl_rows = sorted(adl, key=lambda row: row["path"])
    other_rows = sorted(others, key=lambda row: row["path"])
    fall_segments = [fall_segment(row, rng, split) for row in fall_rows]
    adl_pool = [piece for row in adl_rows for piece in adl_segments(row, ADL_USES[split], rng)]
    other_segments = [other_segment(row) for row in other_rows]
    return {
        "split": split,
        "rng_state": rng.bit_generator.state,
        "falls": fall_segments,
        "adl_pool": adl_pool,
        "others": other_segments,
    }


def binding_density(prepared: dict[str, dict]) -> float:
    """Highest falls/hour that every split can reach without passing its ADL cap."""
    floors = []
    for split, bundle in prepared.items():
        fall_s = sum(segment.duration for segment in bundle["falls"])
        other_s = sum(segment.duration for segment in bundle["others"])
        adl_s = sum(segment.duration for segment in bundle["adl_pool"])
        hours = (fall_s + other_s + adl_s) / 3600.0
        floors.append(len(bundle["falls"]) / hours)
    return float(max(floors))


def finish_plan(bundle: dict, density_per_hour: float) -> SplitPlan:
    falls = bundle["falls"]
    others = bundle["others"]
    fall_s = sum(segment.duration for segment in falls)
    other_s = sum(segment.duration for segment in others)
    target_total = len(falls) / density_per_hour * 3600.0
    target_adl = max(0.0, target_total - fall_s - other_s)
    rng = np.random.Generator(np.random.PCG64())
    rng.bit_generator.state = bundle["rng_state"]
    adl = _select_adl(bundle["adl_pool"], target_adl, rng)
    segments = falls + others + adl
    rng.shuffle(segments)
    streams = _pack(bundle["split"], segments)
    return SplitPlan(
        split=bundle["split"],
        density_per_hour=len(falls) / (sum(stream.duration for stream in streams) / 3600.0),
        streams=streams,
        falls=falls,
        adl_seconds=sum(segment.duration for segment in adl),
    )


def junction_scores(detection_times: list[float], events: list[tuple[float, float, bool]], tolerance: float = 0.5) -> dict:
    """Score a discontinuity detector against fall intervals. Ignored events are not falls to find."""
    scored = [(start, end) for start, end, ignore in events if not ignore]
    matched: set[int] = set()
    true_positive = 0
    for detection in detection_times:
        for index, (start, end) in enumerate(scored):
            if index in matched:
                continue
            if start - tolerance <= detection <= end + tolerance:
                matched.add(index)
                true_positive += 1
                break
    false_positive = len(detection_times) - true_positive
    false_negative = len(scored) - len(matched)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def plan_all(by_split: dict[str, dict]) -> tuple[float, dict[str, SplitPlan]]:
    prepared = {
        split: prepare_split(split, groups["falls"], groups["adl"], groups["others"])
        for split, groups in by_split.items()
    }
    density = binding_density(prepared)
    return density, {split: finish_plan(prepared[split], density) for split in prepared}
