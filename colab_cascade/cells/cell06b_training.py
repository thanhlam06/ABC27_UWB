EVENTS_DF = pd.read_csv(EVENTS_CSV)
COMPOSITION_DF = pd.read_csv(COMPOSITION_CSV)


class LabeledStream:
    """Preprocessed stream + fall events + splice-junction mask + hard-negative candidates."""

    def __init__(self, stream_id: str):
        self.sd = preprocess_stream(stream_id)
        self.stream_id, self.n_frames, self.bands, self.rtm = stream_id, self.sd.n_frames, self.sd.bands, self.sd.rtm
        ev = EVENTS_DF[(EVENTS_DF["stream_id"] == stream_id)]
        self.events = ev[["t_start", "t_end"]].to_numpy(dtype=np.float64)          # (E, 2) seconds
        self.splice = build_splice_mask(COMPOSITION_DF, stream_id, self.n_frames)
        # hard negatives: 2-s windows (stride 0.5 s) with the highest motion energy and no fall nearby
        energy = np.log1p(np.clip(self.bands, 0, None)).mean(0)
        csum = np.concatenate([[0.0], np.cumsum(energy, dtype=np.float64)])
        W, S = S1_WINDOW_FRAMES, S1_STRIDE_FRAMES
        starts = np.arange(0, self.n_frames - W, S)
        win_e = (csum[starts + W] - csum[starts]) / W
        ok = np.array([self.overlap_s(s, W) <= 0.0 for s in starts])
        cand = starts[ok & (win_e >= np.quantile(win_e, 0.85))]
        self.hard_starts = cand if len(cand) else starts

    def overlap_s(self, start_frame: int, length: int) -> float:
        """Max overlap [s] between [start, start+length) and any fall event."""
        if len(self.events) == 0:
            return 0.0
        a, b = start_frame / FS_HZ, (start_frame + length) / FS_HZ
        ov = np.minimum(b, self.events[:, 1]) - np.maximum(a, self.events[:, 0])
        return float(max(ov.max(), 0.0))


# ----------------------------------------------------------------------------
# Stage 1 sampler / trainer
# ----------------------------------------------------------------------------
def sample_stage1_batch(stores, batch_size: int, rng: np.random.Generator):
    W = S1_WINDOW_FRAMES
    xs, ys = [], []
    weights = np.array([max(len(s.events), 1) for s in stores], dtype=np.float64); weights /= weights.sum()
    while len(xs) < batch_size:
        positive = len(xs) % 2 == 0                                  # balanced 1:1
        st = stores[rng.choice(len(stores), p=weights)] if positive else stores[rng.integers(len(stores))]
        if positive and len(st.events):
            ts, te = st.events[rng.integers(len(st.events))]
            min_ov = min(0.5, 0.5 * (te - ts))                       # window must contain >= 0.5 s of the fall
            lo, hi = ts + min_ov - W / FS_HZ, te - min_ov
            s = int(round(rng.uniform(lo, hi) * FS_HZ))
        elif (not positive) and rng.random() < 0.4:
            s = int(rng.choice(st.hard_starts)) + int(rng.integers(-10, 11))
        else:
            s = int(rng.integers(0, st.n_frames - W))
        s = int(np.clip(s, 0, st.n_frames - W))
        ov = st.overlap_s(s, W)
        if positive and ov < 0.25:
            continue
        if (not positive) and ov > 0.0:
            continue
        xs.append(np.asarray(st.rtm[:, s:s + W, :], dtype=np.float32)); ys.append(1.0 if positive else 0.0)
    x = torch.from_numpy(np.stack(xs)).to(device=DEVICE, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32, device=DEVICE)
    return x, y


def train_stage1(stores, steps=S1_STEPS, batch_size=S1_BATCH, lr=S1_LR, log_every=250):
    model = MobileNetV3ProposalNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(SEED)
    model.train(); run, t0 = 0.0, time.time()
    for it in range(1, steps + 1):
        x, y = sample_stage1_batch(stores, batch_size, rng)
        if rng.random() < 0.5:                                        # light augmentation: gain jitter
            x = x * float(rng.uniform(0.9, 1.1))
        loss = F.binary_cross_entropy_with_logits(model.logits(x), y)
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); sched.step()
        run = loss.item() if it == 1 else 0.98 * run + 0.02 * loss.item()
        if it % log_every == 0 or it == steps:
            print(f"  [Stage1] step {it:5d}/{steps}  bce {run:.4f}  ({time.time() - t0:.0f}s)")
    return model.eval()


@torch.no_grad()
def stage1_recall(model, stores):
    """Fraction of falls covered by at least one hit window (p > 0.25) + fraction of the stream flagged."""
    hit_ev = tot_ev = 0; flagged = total = 0.0
    for st in stores:
        probs, starts = stage1_scan(model, st.rtm)
        regions = extract_suspicious_regions(probs, starts / FS_HZ, window_s=S1_WINDOW_FRAMES / FS_HZ,
                                             total_duration_s=st.sd.duration)
        for ts, te in st.events:
            tot_ev += 1
            hit_ev += any(a <= (ts + te) / 2 <= b for a, b in regions)
        flagged += sum(b - a for a, b in regions); total += st.sd.duration
    return hit_ev / max(tot_ev, 1), flagged / max(total, 1e-9)


# ----------------------------------------------------------------------------
# Stage 2 sampler / trainer
# ----------------------------------------------------------------------------
def sample_stage2_batch(stores, batch_size: int, rng: np.random.Generator):
    L = int(rng.integers(int(S2_MIN_BLOCK_S * FS_HZ), int(S2_MAX_BLOCK_S * FS_HZ) + 1))   # one length per batch
    xs, cls_l, reg_l, spl_l = [], [], [], []
    weights = np.array([max(len(s.events), 1) for s in stores], dtype=np.float64); weights /= weights.sum()
    for _ in range(batch_size):
        r = rng.random()
        st = stores[rng.choice(len(stores), p=weights)] if r < 0.7 else stores[rng.integers(len(stores))]
        if r < 0.7 and len(st.events):                                # block containing a fall (random position)
            ts, te = st.events[rng.integers(len(st.events))]
            lo = int(te * FS_HZ) - L + 25
            hi = int(ts * FS_HZ) - 25
            start = int(rng.integers(lo, hi + 1)) if hi >= lo else int(ts * FS_HZ) - L // 2
        elif r < 0.85:                                                # hard negative (high-energy, fall-free)
            start = int(rng.choice(st.hard_starts)) - int(rng.integers(0, L))
        else:                                                         # random block
            start = int(rng.integers(0, st.n_frames - L + 1))
        start = int(np.clip(start, 0, st.n_frames - L))
        a, b = start, start + L
        x = st.bands[:, a:b].copy()
        x *= rng.uniform(0.85, 1.15)                                  # gain jitter
        x += rng.normal(0.0, 0.02, size=x.shape).astype(np.float32)
        t0 = start / FS_HZ
        evs = [(ts - t0, te - t0) for ts, te in st.events if te > t0 and ts < t0 + L / FS_HZ]
        c, rg = create_regression_targets(evs, L)
        xs.append(torch.from_numpy(x)); cls_l.append(c); reg_l.append(rg)
        spl_l.append(torch.from_numpy(st.splice[a:b].copy()))
    to = lambda t: torch.stack(t).to(DEVICE)
    return (to(xs).to(torch.float32), to(cls_l).to(torch.float32), to(reg_l).to(torch.float32), to(spl_l))


def train_stage2(stores, steps=S2_STEPS, batch_size=S2_BATCH, lr=S2_LR, log_every=250):
    model = TemporalFPNDecoder().to(DEVICE)
    criterion = TemporalDIoULoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(SEED + 1)
    model.train(); t0 = time.time(); agg = {}
    for it in range(1, steps + 1):
        x, cls_t, reg_t, spl = sample_stage2_batch(stores, batch_size, rng)
        p, reg = model(x)
        loss, parts = criterion(p, reg, cls_t, reg_t, spl)
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); sched.step()
        parts["total"] = float(loss.detach())
        for k, v in parts.items():
            agg[k] = v if k not in agg else 0.98 * agg[k] + 0.02 * v
        if it % log_every == 0 or it == steps:
            print(f"  [Stage2] step {it:5d}/{steps}  total {agg['total']:.3f}  focal {agg['focal']:.3f}  "
                  f"splice {agg['splice']:.3f}  diou {agg['diou']:.3f}  mIoU {agg['iou']:.3f}  ({time.time() - t0:.0f}s)")
    return model.eval()


# ----------------------------------------------------------------------------
# Run (or resume from checkpoints)
# ----------------------------------------------------------------------------
S1_CKPT, S2_CKPT = CKPT_DIR / "stage1_mobilenetv3.pt", CKPT_DIR / "stage2_fpn_decoder.pt"
need_train = FORCE_RETRAIN or not (S1_CKPT.exists() and S2_CKPT.exists())

stage1_model = MobileNetV3ProposalNet().to(DEVICE)
stage2_model = TemporalFPNDecoder().to(DEVICE)

if need_train:
    fit_ids = TRAIN_STREAM_IDS + (VAL_STREAM_IDS if USE_VAL_FOR_TRAINING else [])
    print(f"Preprocessing {len(fit_ids)} streams (SVD clutter removal + band features, cached in {CACHE_DIR}) ...")
    train_stores = [LabeledStream(s) for s in fit_ids]
    print(f"Training streams: {len(train_stores)}, fall events: {sum(len(s.events) for s in train_stores)}")
    stage1_model = train_stage1(train_stores)
    torch.save(stage1_model.state_dict(), S1_CKPT)
    stage2_model = train_stage2(train_stores)
    torch.save(stage2_model.state_dict(), S2_CKPT)
    del train_stores; gc.collect()
else:
    stage1_model.load_state_dict(torch.load(S1_CKPT, map_location=DEVICE))
    stage2_model.load_state_dict(torch.load(S2_CKPT, map_location=DEVICE))
    print("Loaded checkpoints from", CKPT_DIR)
stage1_model.eval(); stage2_model.eval()
