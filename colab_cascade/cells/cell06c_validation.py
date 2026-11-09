def _tiou(a0, a1, b0, b1):
    ov = min(a1, b1) - max(a0, b0)
    return 0.0 if ov <= 0 else ov / ((a1 - a0) + (b1 - b0) - ov)


def event_f1(gt_by_stream: dict, preds: list, thetas=(0.1, 0.3, 0.5)):
    """Event-level F1 (greedy one-to-one matching by descending score, same rule as the official scorer)."""
    out = {}
    for th in thetas:
        tp = fp = fn = 0
        for sid, gts in gt_by_stream.items():
            ps = sorted([p for p in preds if p["stream_id"] == sid], key=lambda p: -p["score"])
            used = set()
            for p in ps:
                best, best_iou = None, -1.0
                for j, (g0, g1) in enumerate(gts):
                    if j in used:
                        continue
                    iou = _tiou(p["t_start"], p["t_end"], g0, g1)
                    if iou >= th and iou > best_iou:
                        best, best_iou = j, iou
                if best is None:
                    fp += 1
                else:
                    used.add(best); tp += 1
            fn += len(gts) - len(used)
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        out[th] = 2 * prec * rec / max(prec + rec, 1e-9)
    return out


def to_records(dets):
    return [{"stream_id": d["stream_id"], "t_start": d["start_time"], "t_end": d["end_time"],
             "score": d["score"], "t_alarm": None} for d in dets]


# ---- run the cascade once per validation stream and keep the raw frame-wise outputs ----
val_results, val_durations, val_gt = {}, {}, {}
for sid in VAL_STREAM_IDS:
    st = LabeledStream(sid)
    val_results[sid] = run_cascade(st.sd, stage1_model, stage2_model)
    val_durations[sid] = st.sd.duration
    val_gt[sid] = [tuple(e) for e in st.events]
    regs = val_results[sid]["regions"]
    cover = np.mean([any(a <= (g0 + g1) / 2 <= b for a, b in regs) for g0, g1 in val_gt[sid]]) if val_gt[sid] else float("nan")
    print(f"{sid}: {len(regs)} suspicious blocks ({sum(b - a for a, b in regs) / st.sd.duration:.0%} of stream), "
          f"Stage-1 fall recall {cover:.1%}")
    del st; gc.collect()

# ---- sweep the Soft-NMS score floor on validation (mean F1 @ 0.1 / 0.3 / 0.5 = ranking score) ----
sweep = {}
for floor in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
    preds = []
    for sid in VAL_STREAM_IDS:
        preds += to_records(detections_from_outputs(val_results[sid], sid, val_durations[sid], score_floor=floor))
    f1 = event_f1(val_gt, preds)
    sweep[floor] = (float(np.mean(list(f1.values()))), f1, len(preds))
    print(f"  soft-NMS floor {floor:.2f}: #pred {len(preds):4d}  F1@0.1/0.3/0.5 = "
          f"{f1[0.1]:.3f}/{f1[0.3]:.3f}/{f1[0.5]:.3f}  mean {sweep[floor][0]:.3f}")
SOFT_NMS_FLOOR = max(sweep, key=lambda k: (round(sweep[k][0], 4), k))     # ties -> larger floor (fewer duplicates)
print(f"==> SOFT_NMS_FLOOR = {SOFT_NMS_FLOOR} (val ranking score {sweep[SOFT_NMS_FLOOR][0]:.3f})")

# ---- official evaluator report (if the package is available) ----
if abc_eval is not None:
    val_preds = []
    for sid in VAL_STREAM_IDS:
        val_preds += to_records(detections_from_outputs(val_results[sid], sid, val_durations[sid], score_floor=SOFT_NMS_FLOOR))
    gt_all = [e for e in abc_eval.load_ground_truth(EVENTS_CSV, ignore_flagged=True) if e["stream_id"] in val_durations]
    rep = abc_eval.evaluate(gt_all, val_preds, val_durations)
    print("Official ranking score:", json.dumps(rep["ranking_score"], default=float))
    for th, r in rep["by_theta"].items():
        print(f"  theta {th}: P {r['precision']:.3f} R {r['recall']:.3f} F1 {r['f1']:.3f}  FP/h {r['false_alarms_per_hour']:.1f}")
