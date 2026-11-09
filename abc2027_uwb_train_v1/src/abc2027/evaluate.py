"""Official event-level evaluation for the ABC2027 UWB challenge.

Submission columns: ``stream_id, t_start, t_end, score``, optional ``t_alarm``.
Ground truth uses the same time columns and an ``ignore`` flag.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import h5py

RATE_HZ = 50.0
PRIMARY_THETA = 0.3
REPORT_THETAS = (0.1, 0.3, 0.5, 0.7)
IGNORE_MARGIN_S = 2.0
TOLERANCES_S = (0.5, 1.0)
REQUIRED_PREDICTION_COLUMNS = ("stream_id", "t_start", "t_end", "score")


class ValidationError(ValueError):
    """A submission failed the public format checks."""


def temporal_iou(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    overlap = min(end_a, end_b) - max(start_a, start_b)
    if overlap <= 0.0:
        return 0.0
    union = (end_a - start_a) + (end_b - start_b) - overlap
    return overlap / union if union > 0.0 else 0.0


def stream_durations(folder: Path) -> dict[str, float]:
    """Seconds covered by each ``stream_XX.h5`` under ``folder``.

    A folder named ``train``, ``val`` or ``test`` yields ids ``{name}_XX``.
    A parent of those folders is scanned one level down.
    """
    folder = Path(folder)
    durations: dict[str, float] = {}
    files = sorted(folder.glob("stream_*.h5"))
    if not files:
        for child in sorted(path for path in folder.iterdir() if path.is_dir()):
            durations.update(stream_durations(child))
        return durations
    split = folder.name if folder.name in {"train", "val", "test"} else ""
    for path in files:
        index = int(path.stem.split("_")[1])
        stream_id = f"{split}_{index:02d}" if split else f"stream_{index:02d}"
        with h5py.File(path, "r") as handle:
            durations[stream_id] = float(handle["cir"].shape[1]) / RATE_HZ
    return durations


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _finite(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("not finite")
    return value


def load_ground_truth(path: Path, ignore_flagged: bool = False) -> list[dict]:
    """Load events. With ``ignore_flagged``, ``quality = low`` and ``needs_review`` are ignore zones.

    Participants can then score validation under the same rule as the official test.
    """
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    events = []
    for row in rows:
        flagged = str(row.get("quality", "")).strip().lower() == "low" or _as_bool(row.get("needs_review", ""))
        events.append(
            {
                "stream_id": row["stream_id"],
                "t_start": float(row["t_start"]),
                "t_end": float(row["t_end"]),
                "ignore": _as_bool(row.get("ignore", "")) or (ignore_flagged and flagged),
            }
        )
    return events


def validate_predictions(path: Path, durations: dict[str, float]) -> list[dict]:
    """Return prediction records, or raise ``ValidationError`` with row-level messages."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValidationError(f"{path}: file is missing or empty")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        missing = [name for name in REQUIRED_PREDICTION_COLUMNS if name not in fields]
        if missing:
            raise ValidationError(f"{path}: missing columns: {', '.join(missing)}")
        has_alarm = "t_alarm" in fields
        records = []
        errors = []
        for number, row in enumerate(reader, start=2):
            stream_id = (row.get("stream_id") or "").strip()
            if stream_id not in durations:
                errors.append(f"row {number}: unknown stream_id {stream_id!r}")
                continue
            try:
                start = _finite(row["t_start"])
                end = _finite(row["t_end"])
                score = _finite(row["score"])
            except (TypeError, ValueError):
                errors.append(f"row {number}: t_start, t_end and score must be finite numbers")
                continue
            duration = durations[stream_id]
            if not start < end:
                errors.append(f"row {number}: t_start must be < t_end")
            elif start < 0.0 or end > duration + 1e-6:
                errors.append(f"row {number}: times must lie inside [0, {duration:.3f}]")
            alarm = None
            if has_alarm and str(row.get("t_alarm") or "").strip():
                try:
                    alarm = _finite(row["t_alarm"])
                except (TypeError, ValueError):
                    errors.append(f"row {number}: t_alarm must be a finite number")
                    continue
                if alarm < 0.0 or alarm > duration + 1e-6:
                    errors.append(f"row {number}: t_alarm must lie inside [0, {duration:.3f}]")
            if start < end and 0.0 <= start and end <= duration + 1e-6:
                records.append(
                    {
                        "stream_id": stream_id,
                        "t_start": start,
                        "t_end": end,
                        "score": score,
                        "t_alarm": alarm,
                    }
                )
        if errors:
            raise ValidationError("; ".join(errors))
    return records


def _intervals_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
    return min(end_a, end_b) > max(start_a, start_b)


def _match_stream(ground: list[dict], predictions: list[dict], theta: float) -> dict:
    """Match predictions to evaluable events, then drop unmatched hits on ignore zones.

    A prediction already matched to an evaluable event stays a true positive even
    when it also overlaps a neighbouring ignore zone widened by ±2 s.
    """
    scored = [event for event in ground if not event["ignore"]]
    ignored = [event for event in ground if event["ignore"]]
    order = sorted(range(len(predictions)), key=lambda index: predictions[index]["score"], reverse=True)
    used: set[int] = set()
    pairs = []
    for pred_index in order:
        prediction = predictions[pred_index]
        best_index = None
        best_iou = -1.0
        for gt_index, event in enumerate(scored):
            if gt_index in used:
                continue
            iou = temporal_iou(prediction["t_start"], prediction["t_end"], event["t_start"], event["t_end"])
            if iou >= theta and iou > best_iou:
                best_index = gt_index
                best_iou = iou
        if best_index is None:
            continue
        used.add(best_index)
        pairs.append((pred_index, best_index, best_iou))
    matched = {pred_index for pred_index, _gt, _iou in pairs}
    ignored_predictions = 0
    false_positive = 0
    for pred_index, prediction in enumerate(predictions):
        if pred_index in matched:
            continue
        overlaps_ignore = any(
            _intervals_overlap(
                prediction["t_start"],
                prediction["t_end"],
                event["t_start"] - IGNORE_MARGIN_S,
                event["t_end"] + IGNORE_MARGIN_S,
            )
            for event in ignored
        )
        if overlaps_ignore:
            ignored_predictions += 1
        else:
            false_positive += 1
    return {
        "pairs": pairs,
        "predictions": predictions,
        "scored": scored,
        "false_positive": false_positive,
        "ignored_predictions": ignored_predictions,
        "false_negative": len(scored) - len(pairs),
    }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def score_events(ground_truth: list[dict], predictions: list[dict], durations: dict[str, float], theta: float) -> dict:
    """Aggregate one theta over every stream in ``durations``."""
    ground_truth = [event for event in ground_truth if event["stream_id"] in durations]
    predictions = [prediction for prediction in predictions if prediction["stream_id"] in durations]
    by_stream_gt: dict[str, list] = {stream_id: [] for stream_id in durations}
    by_stream_pred: dict[str, list] = {stream_id: [] for stream_id in durations}
    for event in ground_truth:
        by_stream_gt[event["stream_id"]].append(event)
    for prediction in predictions:
        by_stream_pred[prediction["stream_id"]].append(prediction)
    true_positive = 0
    false_positive = 0
    false_negative = 0
    ignored_predictions = 0
    ignored_ground = sum(1 for event in ground_truth if event["ignore"])
    start_abs: list[float] = []
    end_abs: list[float] = []
    ious: list[float] = []
    latency: list[float] = []
    for stream_id in sorted(set(by_stream_gt) | set(by_stream_pred)):
        result = _match_stream(by_stream_gt.get(stream_id, []), by_stream_pred.get(stream_id, []), theta)
        true_positive += len(result["pairs"])
        false_positive += result["false_positive"]
        false_negative += result["false_negative"]
        ignored_predictions += result["ignored_predictions"]
        scored = result["scored"]
        predictions_here = result["predictions"]
        for pred_index, gt_index, iou in result["pairs"]:
            prediction = predictions_here[pred_index]
            event = scored[gt_index]
            start_abs.append(abs(prediction["t_start"] - event["t_start"]))
            end_abs.append(abs(prediction["t_end"] - event["t_end"]))
            ious.append(iou)
            alarm = prediction["t_alarm"] if prediction["t_alarm"] is not None else prediction["t_start"]
            latency.append(alarm - event["t_start"])
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    hours = sum(durations.values()) / 3600.0
    within = {}
    for tolerance in TOLERANCES_S:
        label = f"{tolerance:.1f}"
        within[f"start_within_{label}"] = (
            sum(value <= tolerance for value in start_abs) / len(start_abs) if start_abs else None
        )
        within[f"end_within_{label}"] = (
            sum(value <= tolerance for value in end_abs) / len(end_abs) if end_abs else None
        )
    return {
        "theta": theta,
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarms_per_hour": (false_positive / hours) if hours else None,
        "start_error_mean_abs": _mean(start_abs),
        "start_error_median_abs": _median(start_abs),
        "end_error_mean_abs": _mean(end_abs),
        "end_error_median_abs": _median(end_abs),
        "mean_tiou": _mean(ious),
        "latency_mean": _mean(latency),
        "latency_median": _median(latency),
        "n_ignored_ground_truth": ignored_ground,
        "n_ignored_predictions": ignored_predictions,
        **within,
    }


def ranking_score(by_theta: dict) -> dict:
    """Official ordering key. Lower tie-breaks win."""
    f1_values = [by_theta[theta]["f1"] for theta in ("0.1", "0.3", "0.5")]
    at_main = by_theta["0.3"]
    return {
        "f1_mean": sum(f1_values) / len(f1_values),
        "start_error_median_abs": at_main["start_error_median_abs"],
        "false_alarms_per_hour": at_main["false_alarms_per_hour"],
    }


def evaluate(ground_truth: list[dict], predictions: list[dict], durations: dict[str, float], theta: float = PRIMARY_THETA) -> dict:
    by_theta = {
        f"{item:.1f}": score_events(ground_truth, predictions, durations, item) for item in REPORT_THETAS
    }
    headline = f"{theta:.1f}"
    if headline not in by_theta:
        by_theta[headline] = score_events(ground_truth, predictions, durations, theta)
    at_main = by_theta["0.3"]
    return {
        "theta": theta,
        "tolerances_s": list(TOLERANCES_S),
        "ranking_score": ranking_score(by_theta),
        "latency": {
            "theta": PRIMARY_THETA,
            "mean": at_main["latency_mean"],
            "median": at_main["latency_median"],
        },
        "primary": by_theta[headline],
        "by_theta": by_theta,
        "ranking": {
            "primary": "mean event F1 at theta 0.1, 0.3 and 0.5",
            "tie_break_1": "median absolute start error on true positives at theta 0.3, lower is better",
            "tie_break_2": "false alarms per hour at theta 0.3, lower is better",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a fall-event submission.")
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--streams", required=True, type=Path, help="Directory of stream_XX.h5 files, or their parent.")
    parser.add_argument(
        "--ignore-flagged",
        action="store_true",
        help="Treat quality=low and needs_review events as ignore zones, as on the official test.",
    )
    parser.add_argument("--theta", type=float, default=PRIMARY_THETA)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    durations = stream_durations(args.streams)
    if not durations:
        raise ValidationError(f"no streams found under {args.streams}")
    predictions = validate_predictions(args.pred, durations)
    report = evaluate(load_ground_truth(args.gt, ignore_flagged=args.ignore_flagged), predictions, durations, args.theta)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    primary = report["primary"]
    score = report["ranking_score"]
    print(
        f"ranking_f1_mean={score['f1_mean']:.4f} "
        f"theta={args.theta:.1f} f1={primary['f1']:.4f} "
        f"tp={primary['tp']} fp={primary['fp']} fn={primary['fn']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
