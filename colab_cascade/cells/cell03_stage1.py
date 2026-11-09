class HSwish(nn.Module):
    """h-swish(x) = x * ReLU6(x + 3) / 6"""
    def forward(self, x):
        return x * F.relu6(x + 3.0) / 6.0


class HSigmoid(nn.Module):
    """h-sigmoid(x) = ReLU6(x + 3) / 6"""
    def forward(self, x):
        return F.relu6(x + 3.0) / 6.0


def _make_divisible(v: float, divisor: int = 8) -> int:
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    return new_v + divisor if new_v < 0.9 * v else new_v


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation:  x * h-sigmoid(W2 ReLU(W1 GAP(x)))   (reduction 4, MobileNetV3)."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        squeeze = _make_divisible(channels // reduction, 8)
        self.fc1 = nn.Conv2d(channels, squeeze, 1)
        self.fc2 = nn.Conv2d(squeeze, channels, 1)
        self.gate = HSigmoid()

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = self.gate(self.fc2(F.relu(self.fc1(s), inplace=True)))
        return x * s


class InvertedResidualSE(nn.Module):
    """MobileNetV3 bneck: 1x1 expand -> kxk depthwise -> [SE] -> 1x1 linear projection (+ skip)."""
    def __init__(self, c_in, k, c_exp, c_out, use_se, use_hs, stride):
        super().__init__()
        act = HSwish if use_hs else (lambda: nn.ReLU(inplace=True))
        layers = []
        if c_exp != c_in:
            layers += [nn.Conv2d(c_in, c_exp, 1, bias=False), nn.BatchNorm2d(c_exp), act()]
        layers += [nn.Conv2d(c_exp, c_exp, k, stride, padding=k // 2, groups=c_exp, bias=False),
                   nn.BatchNorm2d(c_exp), act()]
        if use_se:
            layers.append(SqueezeExcite(c_exp))
        layers += [nn.Conv2d(c_exp, c_out, 1, bias=False), nn.BatchNorm2d(c_out)]
        self.block = nn.Sequential(*layers)
        self.use_skip = stride == 1 and c_in == c_out

    def forward(self, x):
        y = self.block(x)
        return x + y if self.use_skip else y


class MobileNetV3ProposalNet(nn.Module):
    """MobileNetV3-Small on 2-s RTM chunks.

    Input : (B, 3, 100, 368) float32 - 3 radars as channels, 100 frames (2.0 s) x 368 range bins.
    Output: (B,) probability that a fall overlaps the window (sigmoid); `logits()` gives the raw score.
    """
    # (kernel, expansion, out, SE, hard-swish, stride) - MobileNetV3-Small table (Howard et al., 2019)
    CFG = [(3, 16, 16, True, False, 2),
           (3, 72, 24, False, False, 2),
           (3, 88, 24, False, False, 1),
           (5, 96, 40, True, True, 2),
           (5, 240, 40, True, True, 1),
           (5, 240, 40, True, True, 1),
           (5, 120, 48, True, True, 1),
           (5, 144, 48, True, True, 1),
           (5, 288, 96, True, True, 2),
           (5, 576, 96, True, True, 1),
           (5, 576, 96, True, True, 1)]

    def __init__(self, in_channels: int = NUM_RADARS, dropout: float = 0.2):
        super().__init__()
        stem_c = 16
        layers = [nn.Conv2d(in_channels, stem_c, 3, 2, 1, bias=False), nn.BatchNorm2d(stem_c), HSwish()]
        c_in = stem_c
        for k, exp, out, se, hs, s in self.CFG:
            layers.append(InvertedResidualSE(c_in, k, exp, out, se, hs, s))
            c_in = out
        last_c = 6 * c_in                                                   # 576
        layers += [nn.Conv2d(c_in, last_c, 1, bias=False), nn.BatchNorm2d(last_c), HSwish()]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Linear(last_c, 1024), HSwish(), nn.Dropout(dropout), nn.Linear(1024, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01); nn.init.zeros_(m.bias)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        z = F.adaptive_avg_pool2d(self.features(x), 1).flatten(1)
        return self.classifier(z).squeeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(x))


@torch.no_grad()
def stage1_scan(model: nn.Module, rtm, window: int = S1_WINDOW_FRAMES, stride: int = S1_STRIDE_FRAMES,
                batch_size: int = 128):
    """Slide the 2-s window over a whole stream. rtm: (3, T, 368) array/memmap.

    Returns (probs (N,) float32, starts_frames (N,) int64)."""
    model.eval()
    T = rtm.shape[1]
    starts = np.arange(0, max(T - window, 0) + 1, stride, dtype=np.int64)
    if starts[-1] + window < T:
        starts = np.append(starts, T - window)
    probs = np.empty(len(starts), dtype=np.float32)
    for i in range(0, len(starts), batch_size):
        sl = starts[i:i + batch_size]
        x = np.stack([np.asarray(rtm[:, s:s + window, :]) for s in sl]).astype(np.float32)
        x = torch.from_numpy(x).to(device=DEVICE, dtype=torch.float32)
        probs[i:i + len(sl)] = model(x).float().cpu().numpy()
    return probs, starts


def extract_suspicious_regions(probs, window_starts_s=None, window_s: float = 2.0, stride_s: float = 0.5,
                               threshold: float = S1_THRESHOLD, merge_gap_s: float = 2.0,
                               margin_s: float = 3.0, min_block_s: float = S2_MIN_BLOCK_S,
                               max_block_s: float = S2_MAX_BLOCK_S, total_duration_s: float = None):
    """Binarise Stage-1 window probabilities (p > 0.25), merge nearby hits, pad to 10-15 s blocks.

    probs: (N,) window probabilities. window_starts_s: (N,) window start times [s]
           (default: i * stride_s).
    Returns a sorted list of (block_start_s, block_end_s) with 10 <= length <= 15 s.
    """
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    starts = (np.arange(len(probs)) * stride_s if window_starts_s is None
              else np.asarray(window_starts_s, dtype=np.float64))
    hits = np.flatnonzero(probs > threshold)
    if hits.size == 0:
        return []

    merged = []                                           # 1) merge hits closer than merge_gap_s
    for i in hits:
        s, e = float(starts[i]), float(starts[i]) + window_s
        if merged and s - merged[-1][1] <= merge_gap_s:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    blocks = []                                           # 2) pad into [min_block_s, max_block_s]
    for s, e in merged:
        s, e = s - margin_s, e + margin_s
        if e - s < min_block_s:
            c = 0.5 * (s + e); s, e = c - 0.5 * min_block_s, c + 0.5 * min_block_s
        if e - s <= max_block_s:
            blocks.append([s, e])
        else:                                             # long activity -> overlapping 15-s tiles
            cur, step = s, max_block_s - margin_s
            while cur + max_block_s < e:
                blocks.append([cur, cur + max_block_s]); cur += step
            blocks.append([e - max_block_s, e])

    if total_duration_s is not None:                      # 3) clip to the stream, keep block length
        clipped = []
        for s, e in blocks:
            L = min(e - s, total_duration_s)
            if s < 0.0:
                s, e = 0.0, L
            if e > total_duration_s:
                s, e = total_duration_s - L, total_duration_s
            clipped.append([max(s, 0.0), e])
        blocks = clipped

    blocks.sort()                                         # 4) fuse overlapping blocks if still <= 15 s
    out = [blocks[0]]
    for s, e in blocks[1:]:
        if s <= out[-1][1] and e - out[-1][0] <= max_block_s:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(float(s), float(e)) for s, e in out]


# quick self-check
_net = MobileNetV3ProposalNet().to(DEVICE)
with torch.no_grad():
    _y = _net.eval()(torch.zeros(2, NUM_RADARS, S1_WINDOW_FRAMES, NUM_BINS, dtype=torch.float32, device=DEVICE))
print(f"MobileNetV3ProposalNet: {sum(p.numel() for p in _net.parameters()) / 1e6:.2f} M params, out {tuple(_y.shape)}")
del _net, _y
