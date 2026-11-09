class ChannelNorm(nn.Module):
    """LayerNorm over the channel axis of a (B, C, L) tensor, independently for every frame.

    Unlike GroupNorm/InstanceNorm it does not normalise over time, so the absolute motion energy of a
    block (what separates a fall from a quiet block) is preserved."""
    def __init__(self, channels: int):
        super().__init__()
        self.ln = nn.LayerNorm(channels)

    def forward(self, x):
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class DilatedResidualBlock(nn.Module):
    """x + Conv(k, d) -> Norm -> ReLU -> Dropout -> Conv(k, d) -> Norm,  followed by ReLU."""
    def __init__(self, ch: int, dilation: int, kernel: int = 7, dropout: float = 0.1):
        super().__init__()
        pad = dilation * (kernel - 1) // 2                       # 'same' length
        self.conv1 = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.norm1 = ChannelNorm(ch)
        self.conv2 = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.norm2 = ChannelNorm(ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.drop(F.relu(self.norm1(self.conv1(x))))
        y = self.norm2(self.conv2(y))
        return F.relu(x + y)


class TemporalFPNDecoder(nn.Module):
    """Frame-wise 1D-FPN temporal decoder (ActionFormer-style heads).

    x: (B, 12, L)  ->  p_fall (B, L) in (0,1)  and  reg_bounds (B, 2, L) = (delta_start, delta_end) >= 0 [s]

    Backbone: Conv1d stem (k=5) + 5 dilated residual blocks (d = 1, 2, 4, 8, 16, two k=7 convs each)
              receptive field = 1 + 4 + 2 * 6 * (1+2+4+8+16) = 377 frames = 7.54 s.
    FPN     : the 5 block outputs are the pyramid levels (increasing temporal context). A top-down
              pathway P_i = smooth(lateral(C_i) + P_{i+1}) injects wide-context features into the
              fine levels, and all levels are fused into one per-frame feature (full 50 Hz resolution).
    Heads   : classification (bias init -4.6 -> p0 ~ 0.01) and regression (final ReLU, distances >= 0).
    """

    def __init__(self, in_channels: int = NUM_CHANNELS, channels: int = 64,
                 dilations=(1, 2, 4, 8, 16), kernel: int = 7, dropout: float = 0.1):
        super().__init__()
        self.dilations, self.kernel = tuple(dilations), kernel
        self.stem = nn.Sequential(nn.Conv1d(in_channels, channels, 5, padding=2), ChannelNorm(channels), nn.ReLU())
        self.blocks = nn.ModuleList([DilatedResidualBlock(channels, d, kernel, dropout) for d in self.dilations])
        n = len(self.dilations)
        self.lateral = nn.ModuleList([nn.Conv1d(channels, channels, 1) for _ in range(n)])
        self.smooth = nn.ModuleList([nn.Conv1d(channels, channels, 3, padding=1) for _ in range(n)])
        self.fuse = nn.Sequential(nn.Conv1d(n * channels, channels, 1), ChannelNorm(channels), nn.ReLU())

        self.cls_head = nn.Sequential(nn.Conv1d(channels, channels, 3, padding=1), nn.ReLU(),
                                      nn.Conv1d(channels, 1, 1))
        self.reg_head = nn.Sequential(nn.Conv1d(channels, channels, 3, padding=1), nn.ReLU(),
                                      nn.Conv1d(channels, 2, 1))
        nn.init.constant_(self.cls_head[-1].bias, -4.6)           # focal-loss prior: p(fall) ~ 0.01
        nn.init.normal_(self.cls_head[-1].weight, 0.0, 0.01)
        nn.init.normal_(self.reg_head[-1].weight, 0.0, 0.01)
        nn.init.constant_(self.reg_head[-1].bias, 1.0)            # ~1 s so the output ReLU starts alive

    @property
    def receptive_field_frames(self) -> int:
        return 1 + 4 + 2 * (self.kernel - 1) * sum(self.dilations)

    def forward_logits(self, x: torch.Tensor):
        h = self.stem(x)
        feats = []
        for blk in self.blocks:
            h = blk(h)
            feats.append(h)
        top, pyramid = None, [None] * len(feats)
        for i in reversed(range(len(feats))):                     # top-down pathway
            lat = self.lateral[i](feats[i])
            top = self.smooth[i](lat if top is None else lat + top)
            pyramid[i] = top
        z = self.fuse(torch.cat(pyramid, dim=1))
        cls_logit = self.cls_head(z).squeeze(1)                   # (B, L)
        reg = F.relu(self.reg_head(z))                            # (B, 2, L), distances >= 0
        return cls_logit, reg

    def forward(self, x: torch.Tensor):
        cls_logit, reg = self.forward_logits(x)
        return torch.sigmoid(cls_logit), reg


# quick self-check
_dec = TemporalFPNDecoder().to(DEVICE)
with torch.no_grad():
    _p, _r = _dec.eval()(torch.zeros(2, NUM_CHANNELS, 600, dtype=torch.float32, device=DEVICE))
print(f"TemporalFPNDecoder: {sum(p.numel() for p in _dec.parameters()) / 1e6:.2f} M params | "
      f"p_fall {tuple(_p.shape)} reg {tuple(_r.shape)} | receptive field {_dec.receptive_field_frames} frames "
      f"= {_dec.receptive_field_frames / FS_HZ:.2f} s | init p ~ {_p.mean().item():.4f}")
del _dec, _p, _r
