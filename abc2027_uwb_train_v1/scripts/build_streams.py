"""Write the continuous streams for train, val and test."""

from __future__ import annotations

import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abc2027.align import RATE_HZ, SERIALS, STEP_S, load_aligned  # noqa: E402
from abc2027.streams import FADE_S, SEED, junction_scores, plan_all  # noqa: E402

FADE_N = int(round(FADE_S * RATE_HZ))
PUBLIC = {"train", "val"}


def _groups() -> dict:
    events_list = list(csv.DictReader((ROOT / "streams_events.csv").open(encoding="utf-8")))
    comp_list = list(csv.DictReader((ROOT / "streams_composition.csv").open(encoding="utf-8")))
    
    events = {row["source_path"]: row for row in events_list}
    groups = {name: {"falls": [], "adl": [], "others": []} for name in ("train", "val")}
    
    seen_paths = set()
    for row in comp_list:
        path = row["path"]
        if path in seen_paths:
            continue
        seen_paths.add(path)
        
        split = row["stream_id"].split("_")[0]
        if split not in groups:
            continue
            
        if row["kind"] == "adl":
            dest = ROOT / "segments" / (path.replace("/", "__") + ".npy")
            if not dest.exists(): continue
            cir = np.load(dest, mmap_mode="r")
            row["duration_s"] = cir.shape[1] * 0.02
            groups[split]["adl"].append(row)
        elif row["kind"] == "fall":
            event = events.get(path)
            if event is not None:
                event["t_start"] = event["source_t_start"]
                event["t_end"] = event["source_t_end"]
                dest = ROOT / "segments" / (path.replace("/", "__") + ".npy")
                if not dest.exists(): continue
                cir = np.load(dest, mmap_mode="r")
                event["t_last"] = cir.shape[1] * 0.02
                event["path"] = path
                groups[split]["falls"].append(event)
        elif row["kind"] == "other":
            dest = ROOT / "segments" / (path.replace("/", "__") + ".npy")
            if not dest.exists(): continue
            cir = np.load(dest, mmap_mode="r")
            row["duration_s"] = cir.shape[1] * 0.02
            groups[split]["others"].append(row)
            
    return groups


def _cache(relative: str) -> Path:
    """Aligned segment, next to this package or in the organizer cache."""
    name = relative.replace("/", "__") + ".npy"
    return ROOT / "segments" / name


def _align_one(relative: str) -> tuple[str, float]:
    dest = _cache(relative)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return relative, -1.0
    cir, _grid, missing = load_aligned(ROOT / relative)
    np.save(dest, cir)
    return relative, float(missing)


def _slice(relative: str, source_t0: float, source_t1: float) -> tuple[np.ndarray, int, int]:
    cir = np.load(_cache(relative), mmap_mode="r")
    i0 = int(np.clip(np.round(source_t0 / STEP_S), 0, cir.shape[1] - 1))
    i1 = int(np.clip(np.round(source_t1 / STEP_S), i0 + 1, cir.shape[1]))
    return np.array(cir[:, i0:i1], dtype=np.float32), i0, i1


def _write_stream(path: Path, split: str, index: int, segments) -> tuple[list[dict], list[dict]]:
    chunks: list[np.ndarray] = []
    events = []
    composition = []
    cursor = 0
    last_event_end = -1
    stream_id = f"{split}_{index:02d}"
    for order, segment in enumerate(segments):
        block, i0, _i1 = _slice(segment.path, segment.source_t0, segment.source_t1)
        blend = 0
        if chunks:
            protect = block.shape[1]
            if segment.event_t0 is not None:
                event_index = int(np.round(segment.event_t0 / STEP_S)) - i0
                protect = max(0, event_index)
            tail_room = cursor - last_event_end
            blend = min(FADE_N, block.shape[1], chunks[-1].shape[1], protect, max(tail_room, 0))
            if blend > 0:
                weight = np.linspace(0.0, 1.0, blend, dtype=np.float32)[None, :, None]
                head = chunks[-1][:, -blend:]
                chunks[-1][:, -blend:] = head * (1.0 - weight) + block[:, :blend] * weight
        origin = cursor - blend
        if block.shape[1] > blend:
            chunks.append(np.array(block[:, blend:], dtype=np.float32))
        cursor = origin + block.shape[1]
        composition.append(
            {
                "stream_id": stream_id,
                "segment_id": f"{stream_id}_{order:04d}",
                "kind": segment.kind,
                "path": segment.path,
                "label": segment.label,
                "dataset": segment.dataset,
                "t_start": f"{origin * STEP_S:.3f}",
                "t_end": f"{cursor * STEP_S:.3f}",
                "source_t0": f"{segment.source_t0:.3f}",
                "source_t1": f"{segment.source_t1:.3f}",
            }
        )
        if segment.kind == "fall":
            event_a = origin + int(np.round(segment.event_t0 / STEP_S)) - i0
            event_b = origin + int(np.round(segment.event_t1 / STEP_S)) - i0
            last_event_end = max(last_event_end, event_b)
            events.append(
                {
                    "stream_id": stream_id,
                    "event_id": f"{stream_id}_{order:04d}",
                    "t_start": f"{event_a * STEP_S:.3f}",
                    "t_end": f"{event_b * STEP_S:.3f}",
                    "fall_type": segment.fall_type,
                    "ignore": str(segment.ignore).lower(),
                    "quality": segment.quality,
                    "needs_review": str(segment.needs_review).lower(),
                    "source_path": segment.path,
                    "source_t_start": f"{segment.event_t0:.3f}",
                    "source_t_end": f"{segment.event_t1:.3f}",
                    "dataset": segment.dataset,
                }
            )
    cir = np.concatenate(chunks, axis=1) if chunks else np.zeros((3, 1, 368), dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    for event in events:
        event.pop("_unused", None)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("cir", data=cir, chunks=(3, min(cir.shape[1], 5000), 368))
        handle.create_dataset("t", data=np.arange(cir.shape[1], dtype=np.float64) * STEP_S)
        handle.create_dataset("radar_serials", data=np.array(SERIALS, dtype=np.int32))
        handle.attrs["split"] = split
        if split != "test":
            handle.attrs["seed"] = SEED
        handle.attrs["stream_index"] = index
        handle.attrs["fade_s"] = FADE_S
    return events, composition


def _csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _discontinuity_times(cir: np.ndarray) -> list[float]:
    energy = np.mean(np.sum(cir.astype(np.float64) ** 2, axis=2), axis=0)
    window = int(round(RATE_HZ))
    smooth = np.convolve(energy, np.ones(window) / window, mode="same")
    delta = np.abs(np.diff(smooth, prepend=smooth[0]))
    level = float(np.percentile(delta, 99))
    times = []
    last = -1e9
    for index in np.flatnonzero(delta >= max(level, 1e-18)):
        stamp = float(index * STEP_S)
        if stamp - last < 1.0:
            continue
        times.append(stamp)
        last = stamp
    return times


def main() -> None:
    density, plans = plan_all(_groups())
    print(f"density={density:.2f} " + " ".join(
        f"{name}={plans[name].duration / 3600:.2f}h" for name in plans
    ), flush=True)
    if not 40.0 <= density <= 70.0:
        print("density outside 40-70, stopped", flush=True)
        return
    paths = sorted({segment.path for plan in plans.values() for stream in plan.streams for segment in stream.segments})
    with ProcessPoolExecutor(max_workers=4) as pool:
        aligned = list(pool.map(_align_one, paths, chunksize=4))
    missing = [value for _path, value in aligned if value >= 0]
    public_events = []
    public_parts = []
    private_events = []
    private_parts = []
    kept_identity = []
    for name, plan in plans.items():
        if name == "test":
            # Official test streams are written by scripts/rebuild_test.py.
            print("skipped test streams; use scripts/rebuild_test.py", flush=True)
            continue
        folder = ROOT / "streams" / name
        for index, stream in enumerate(plan.streams):
            events, parts = _write_stream(folder / f"stream_{index:02d}.h5", name, index, stream.segments)
            print(f"wrote {name} stream {index} events={len(events)}", flush=True)
            if name in PUBLIC:
                public_events.extend(events)
                public_parts.extend(parts)
            else:
                private_events.extend(events)
                private_parts.extend(parts)
            if name == "train" and index == 0:
                kept_identity.append(folder / f"stream_{index:02d}.h5")
    _csv(ROOT / "streams_events.csv", public_events)
    _csv(ROOT / "streams_composition.csv", public_parts)
    if private_events:
        _csv(ROOT / "private_streams_events.csv", private_events)
        _csv(ROOT / "private_streams_composition.csv", private_parts)
    _shortcut(plans, public_events, public_parts)
    _figure(ROOT / "streams" / "train" / "stream_00.h5", public_events)
    _identity(public_events + private_events)
    summary = {
        "density_per_hour": density,
        "durations_h": {name: plans[name].duration / 3600 for name in plans},
        "streams": {name: len(plans[name].streams) for name in plans},
        "alignment_files": len(paths),
        "alignment_missing_mean": float(np.mean(missing)) if missing else 0.0,
    }
    (ROOT / "reports" / "L1_build.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("build done", flush=True)


def _shortcut(plans, events, parts) -> None:
    detections = []
    gt = []
    boundaries = []
    folder = ROOT / "streams" / "val"
    for index in range(len(plans["val"].streams)):
        path = folder / f"stream_{index:02d}.h5"
        stream_id = f"val_{index:02d}"
        with h5py.File(path, "r") as handle:
            detections.extend((stream_id, stamp) for stamp in _discontinuity_times(handle["cir"][:]))
        for row in events:
            if row["stream_id"] == stream_id:
                gt.append((stream_id, float(row["t_start"]), float(row["t_end"]), row["ignore"] == "true"))
        for row in parts:
            if row["stream_id"] == stream_id and row["dataset"] == "Dataset 3":
                boundaries.append((stream_id, float(row["t_start"]), float(row["t_end"])))
    by_stream: dict[str, list] = {}
    for stream_id, stamp in detections:
        by_stream.setdefault(stream_id, []).append(stamp)
    scores = {"tp": 0, "fp": 0, "fn": 0}
    near_dataset3 = 0
    for stream_id, stamps in by_stream.items():
        stream_events = [(a, b, ignore) for sid, a, b, ignore in gt if sid == stream_id]
        part = junction_scores(stamps, stream_events)
        for key in scores:
            scores[key] += part[key]
        edges = [edge for sid, start, end in boundaries if sid == stream_id for edge in (start, end)]
        for stamp in stamps:
            if any(abs(stamp - edge) <= 1.0 for edge in edges):
                near_dataset3 += 1
    precision = scores["tp"] / (scores["tp"] + scores["fp"]) if scores["tp"] + scores["fp"] else 0.0
    recall = scores["tp"] / (scores["tp"] + scores["fn"]) if scores["tp"] + scores["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    report = {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        **scores,
        "detections": len(detections),
        "detections_near_dataset3_boundary": near_dataset3,
    }
    (ROOT / "reports" / "L1_shortcut.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"shortcut f1={f1:.3f}", flush=True)


def _figure(path: Path, events: list[dict]) -> None:
    with h5py.File(path, "r") as handle:
        cir = handle["cir"][:, ::10, :]
        times = handle["t"][::10]
    energy = np.mean(np.sum(cir.astype(np.float64) ** 2, axis=2), axis=0)
    fig, axis = plt.subplots(figsize=(14, 3.5))
    axis.plot(times, energy, color="black", linewidth=0.6)
    stream_id = path.stem.replace("stream_", path.parent.name + "_")
    for event in events:
        if event["stream_id"] != stream_id:
            continue
        axis.axvspan(float(event["t_start"]), float(event["t_end"]), color="#b00020", alpha=0.25)
    axis.set_xlabel("seconds")
    axis.set_ylabel("mean CIR energy")
    axis.set_title(path.relative_to(ROOT).as_posix())
    fig.tight_layout()
    dest = ROOT / "reports" / "figures" / "L1_stream_example.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=110)
    plt.close(fig)


def _identity(events: list[dict]) -> None:
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(events))
    checked = 0
    mismatches = 0
    for index in order:
        event = events[int(index)]
        i0 = int(np.round(float(event["source_t_start"]) / STEP_S))
        i1 = int(np.round(float(event["source_t_end"]) / STEP_S))
        source = np.load(_cache(event["source_path"]), mmap_mode="r")
        i0 = int(np.clip(i0, 0, source.shape[1] - 1))
        i1 = int(np.clip(i1, i0 + 1, source.shape[1]))
        split, number = event["stream_id"].rsplit("_", 1)
        folder = ROOT / "streams" / split
        with h5py.File(folder / f"stream_{int(number):02d}.h5", "r") as handle:
            a = int(np.round(float(event["t_start"]) / STEP_S))
            b = int(np.round(float(event["t_end"]) / STEP_S))
            b = min(b, handle["cir"].shape[1])
            got = np.array(handle["cir"][:, a:b])
        expected = np.array(source[:, i0:i1])
        if got.shape != expected.shape or not np.array_equal(got, expected):
            mismatches += 1
        checked += 1
        if checked == 10:
            break
    (ROOT / "reports" / "L1_identity.txt").write_text(
        f"checked={checked} mismatches={mismatches}\n", encoding="utf-8"
    )
    print(f"identity checked={checked} mismatches={mismatches}", flush=True)


if __name__ == "__main__":
    main()
