def create_regression_targets(gt_events, n_frames: int, fs: float = FS_HZ):
    """Frame-wise targets for one block.

    gt_events: iterable of (t_start, t_end) [s] (or dicts with those keys) relative to the block start.
    Returns cls_target (n_frames,) and reg_target (2, n_frames), both float32 tensors:
      inside [t_start, t_end] : cls = 1, reg[0] = t - t_start, reg[1] = t_end - t   (seconds)
      elsewhere               : cls = 0, reg = 0
    """
    t = torch.arange(n_frames, dtype=torch.float32) / fs
    cls_target = torch.zeros(n_frames, dtype=torch.float32)
    reg_target = torch.zeros(2, n_frames, dtype=torch.float32)
    for ev in gt_events:
        ts, te = (ev["t_start"], ev["t_end"]) if isinstance(ev, dict) else (ev[0], ev[1])
        inside = (t >= ts) & (t <= te)
        cls_target[inside] = 1.0
        reg_target[0, inside] = t[inside] - float(ts)
        reg_target[1, inside] = float(te) - t[inside]
    return cls_target, reg_target


def build_splice_mask(composition_df: pd.DataFrame, stream_id: str, n_frames: int,
                      fs: float = FS_HZ, crossfade_s: float = 0.2, pad_s: float = 0.1) -> np.ndarray:
    """Boolean (n_frames,) mask of the 0.2-s crossfade zones where consecutive source segments are joined
    (segment k starts at its t_start; the cross-fade covers [t_start, t_start + 0.2 s]), +-pad_s margin."""
    mask = np.zeros(n_frames, dtype=bool)
    seg = composition_df[composition_df["stream_id"] == stream_id].sort_values("t_start")
    for t_start in seg["t_start"].to_numpy()[1:]:                   # the first segment has no junction
        a = max(int(round((t_start - pad_s) * fs)), 0)
        b = min(int(round((t_start + crossfade_s + pad_s) * fs)) + 1, n_frames)
        if b > a:
            mask[a:b] = True
    return mask


def focal_loss_map(p: torch.Tensor, target: torch.Tensor, alpha: float, gamma: float, eps: float = 1e-6):
    """Per-frame focal loss  FL = -alpha_t (1 - p_t)^gamma log(p_t)."""
    p = p.clamp(eps, 1.0 - eps)
    pos = target > 0.5
    p_t = torch.where(pos, p, 1.0 - p)
    alpha_t = torch.where(pos, torch.full_like(p, alpha), torch.full_like(p, 1.0 - alpha))
    return -alpha_t * (1.0 - p_t).pow(gamma) * torch.log(p_t)


def temporal_diou(pred_bounds: torch.Tensor, tgt_bounds: torch.Tensor, t: torch.Tensor, eps: float = 1e-6):
    """1-D Distance-IoU for frames at times t with (delta_start, delta_end) distances.

    pred interval [t - ds, t + de]  vs  GT interval [t - gs, t + ge];  all tensors (N,) / (N, 2).
      IoU  = I / U,   I = max(0, min(pe, ge) - max(ps, gs)),   U = (pe - ps) + (ge - gs) - I
      DIoU = IoU - (c_p - c_g)^2 / W^2        (W = width of the smallest enclosing interval)
    returns (1 - DIoU, IoU)."""
    ps, pe = t - pred_bounds[:, 0], t + pred_bounds[:, 1]
    gs, ge = t - tgt_bounds[:, 0], t + tgt_bounds[:, 1]
    inter = (torch.minimum(pe, ge) - torch.maximum(ps, gs)).clamp(min=0.0)
    union = (pe - ps) + (ge - gs) - inter
    iou = inter / (union + eps)
    center_dist2 = (0.5 * (ps + pe) - 0.5 * (gs + ge)).pow(2)
    enclose_w = torch.maximum(pe, ge) - torch.minimum(ps, gs)
    diou = iou - center_dist2 / (enclose_w.pow(2) + eps)
    return 1.0 - diou, iou


class TemporalDIoULoss(nn.Module):
    """Total = Focal(alpha=0.85, gamma=2) + Splice_Penalty + 2.0 * DIoU

    * Focal      : all frames, normalised by the number of positive frames (>= 1).
    * Splice     : splice_weight (5) x focal loss of *negative* frames inside crossfade junction zones
                   (added on top of the plain focal term), same normalisation -> punishes FPs at splices.
    * DIoU       : only frames with cls_target == 1, mean over those frames.
    """

    def __init__(self, alpha: float = 0.85, gamma: float = 2.0, splice_weight: float = 5.0,
                 diou_weight: float = 2.0, fs: float = FS_HZ):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
        self.splice_weight, self.diou_weight, self.fs = splice_weight, diou_weight, fs

    def forward(self, p_fall, reg_bounds, cls_target, reg_target, splice_mask=None):
        """p_fall (B,L) | reg_bounds (B,2,L) | cls_target (B,L) | reg_target (B,2,L) | splice_mask (B,L) bool."""
        p_fall, reg_bounds = p_fall.to(torch.float32), reg_bounds.to(torch.float32)
        pos = cls_target > 0.5
        n_pos = pos.sum().clamp(min=1).to(torch.float32)

        fl = focal_loss_map(p_fall, cls_target, self.alpha, self.gamma)
        focal = fl.sum() / n_pos

        if splice_mask is not None:
            junction_neg = splice_mask.to(torch.bool) & (~pos)
            splice = self.splice_weight * (fl * junction_neg.to(torch.float32)).sum() / n_pos
        else:
            splice = fl.new_zeros(())

        if pos.any():
            B, L = p_fall.shape
            t = (torch.arange(L, device=p_fall.device, dtype=torch.float32) / self.fs).expand(B, L)
            pred = reg_bounds.permute(0, 2, 1)[pos]               # (N, 2)
            tgt = reg_target.permute(0, 2, 1)[pos]                # (N, 2)
            diou_l, iou = temporal_diou(pred, tgt, t[pos])
            diou_loss, mean_iou = diou_l.mean(), iou.mean().detach()
        else:
            diou_loss, mean_iou = reg_bounds.sum() * 0.0, reg_bounds.new_zeros(())

        total = focal + splice + self.diou_weight * diou_loss
        return total, {"focal": float(focal.detach()), "splice": float(splice.detach()),
                       "diou": float(diou_loss.detach()), "iou": float(mean_iou), "n_pos": int(n_pos)}


# ---- unit checks of the exact math ----
_c, _r = create_regression_targets([(1.0, 2.0)], 150)
assert _c.dtype == torch.float32 and _r.dtype == torch.float32
assert int(_c.sum()) == 51 and abs(float(_r[0, 75]) - 0.5) < 1e-6 and abs(float(_r[1, 75]) - 0.5) < 1e-6
_t = torch.tensor([1.5]); _g = torch.tensor([[0.5, 0.5]])
_l, _i = temporal_diou(torch.tensor([[0.5, 0.5]]), _g, _t); assert abs(float(_l)) < 1e-4 and abs(float(_i) - 1) < 1e-4
_l, _i = temporal_diou(torch.tensor([[0.25, 0.25]]), _g, _t); assert abs(float(_i) - 0.5) < 1e-4     # IoU = 0.5/1.0
_l, _i = temporal_diou(torch.tensor([[0.0, 0.5]]), _g, _t)   # pred [1.5,2.0] vs GT [1.0,2.0]: IoU .5, d=.25, W=1
assert abs(float(_l) - (1 - (0.5 - 0.0625))) < 1e-4, float(_l)
print("Loss unit checks passed.")
