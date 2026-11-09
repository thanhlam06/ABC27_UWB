def _iou_1d(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    inter = np.clip(np.minimum(box[1], boxes[:, 1]) - np.maximum(box[0], boxes[:, 0]), 0.0, None)
    union = (box[1] - box[0]) + (boxes[:, 1] - boxes[:, 0]) - inter
    return inter / np.maximum(union, 1e-9)


def soft_nms_1d(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = SOFT_NMS_IOU, method: str = "linear",
                sigma: float = 0.5, score_floor: float = None):
    """1-D Soft-NMS (Bodla et al.). boxes (N,2) [start,end], scores (N,).

    Repeatedly keeps the highest-scoring box and decays the score of the remaining boxes:
      linear  : s <- s * (1 - IoU)            if IoU > iou_thr
      gaussian: s <- s * exp(-IoU^2 / sigma)  for every box
    Boxes whose decayed score falls below `score_floor` are dropped. Returns (kept_boxes, kept_scores)."""
    score_floor = SOFT_NMS_FLOOR if score_floor is None else score_floor
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 2).copy()
    scores = np.asarray(scores, dtype=np.float64).reshape(-1).copy()
    kept_b, kept_s = [], []
    while len(scores) > 0:
        m = int(np.argmax(scores))
        best_b, best_s = boxes[m].copy(), float(scores[m])
        kept_b.append(best_b); kept_s.append(best_s)
        boxes, scores = np.delete(boxes, m, axis=0), np.delete(scores, m)
        if len(scores) == 0:
            break
        iou = _iou_1d(best_b, boxes)
        if method == "gaussian":
            decay = np.exp(-(iou ** 2) / sigma)
        else:
            decay = np.where(iou > iou_thr, 1.0 - iou, 1.0)
        scores = scores * decay
        alive = scores >= score_floor
        boxes, scores = boxes[alive], scores[alive]
    if not kept_b:
        return np.zeros((0, 2)), np.zeros(0)
    return np.stack(kept_b), np.asarray(kept_s)


@torch.no_grad()
def decode_and_nms(p_fall, reg_bounds, t_offset: float, stream_id: str = "stream", fs: float = FS_HZ,
                   score_thresh: float = P_FALL_THRESHOLD, iou_thr: float = SOFT_NMS_IOU,
                   score_floor: float = None, method: str = "linear", min_duration_s: float = 0.2,
                   t_max: float = None):
    """Frame-wise predictions of one block -> list of {stream_id, start_time, end_time, score}.

    p_fall (L,) | reg_bounds (2, L) [seconds] | t_offset = absolute time [s] of the block's first frame.
      keep frames with p_fall > score_thresh;  start = t_offset + t - ds,  end = t_offset + t + de;
      then 1-D Soft-NMS (IoU threshold 0.3)."""
    p = torch.as_tensor(p_fall).detach().to("cpu", torch.float32).reshape(-1).numpy()
    reg = torch.as_tensor(reg_bounds).detach().to("cpu", torch.float32).reshape(2, -1).numpy()
    t = np.arange(p.shape[0], dtype=np.float64) / fs
    keep = p > score_thresh
    if not keep.any():
        return []
    start = t_offset + t[keep] - reg[0, keep]
    end = t_offset + t[keep] + reg[1, keep]
    start = np.maximum(start, 0.0)
    if t_max is not None:
        end = np.minimum(end, t_max)
    valid = (end - start) >= min_duration_s
    if not valid.any():
        return []
    boxes = np.stack([start[valid], end[valid]], axis=1)
    kb, ks = soft_nms_1d(boxes, p[keep][valid], iou_thr, method, score_floor=score_floor)
    return [{"stream_id": stream_id, "start_time": float(b[0]), "end_time": float(b[1]), "score": float(s)}
            for b, s in zip(kb, ks)]


def merge_detections(dets, iou_thr: float = SOFT_NMS_IOU, score_floor: float = None, method: str = "linear"):
    """Soft-NMS across the detections of all blocks of a stream (blocks may overlap)."""
    if not dets:
        return []
    boxes = np.array([[d["start_time"], d["end_time"]] for d in dets])
    scores = np.array([d["score"] for d in dets])
    kb, ks = soft_nms_1d(boxes, scores, iou_thr, method, score_floor=score_floor)
    sid = dets[0]["stream_id"]
    out = [{"stream_id": sid, "start_time": float(b[0]), "end_time": float(b[1]), "score": float(s)}
           for b, s in zip(kb, ks)]
    return sorted(out, key=lambda d: d["start_time"])


# ----------------------------------------------------------------------------
# Cascade runner:  Stage 1 proposals -> Stage 2 frame-wise decoding (raw outputs are kept so that the
# post-processing thresholds can be swept cheaply on validation data).
# ----------------------------------------------------------------------------
@torch.no_grad()
def run_cascade(sd: StreamData, stage1: nn.Module, stage2: nn.Module):
    stage1.eval(); stage2.eval()
    probs, starts = stage1_scan(stage1, sd.rtm)
    regions = extract_suspicious_regions(probs, starts / FS_HZ, window_s=S1_WINDOW_FRAMES / FS_HZ,
                                         total_duration_s=sd.duration)
    outputs = []
    for (a_s, b_s) in regions:
        a, b = int(round(a_s * FS_HZ)), min(int(round(b_s * FS_HZ)), sd.n_frames)
        if b - a < 50:
            continue
        x = torch.from_numpy(np.ascontiguousarray(sd.bands[:, a:b])).to(device=DEVICE, dtype=torch.float32).unsqueeze(0)
        p, reg = stage2(x)
        outputs.append({"t_offset": a / FS_HZ, "p": p[0].float().cpu(), "reg": reg[0].float().cpu()})
    return {"s1_probs": probs, "s1_starts": starts, "regions": regions, "blocks": outputs}


def detections_from_outputs(result: dict, stream_id: str, duration: float, score_floor: float = None,
                            score_thresh: float = P_FALL_THRESHOLD):
    dets = []
    for blk in result["blocks"]:
        dets += decode_and_nms(blk["p"], blk["reg"], blk["t_offset"], stream_id, score_thresh=score_thresh,
                               score_floor=score_floor, t_max=duration)
    return merge_detections(dets, score_floor=score_floor)


# ---- unit check: two overlapping duplicates + one distinct event ----
_kb, _ks = soft_nms_1d(np.array([[1.0, 3.0], [1.1, 3.1], [10.0, 12.0]]), np.array([0.9, 0.8, 0.7]), 0.3, score_floor=0.3)
assert len(_kb) == 2 and abs(_ks[0] - 0.9) < 1e-9, (_kb, _ks)
print("Soft-NMS unit check passed.")
