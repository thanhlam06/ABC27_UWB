class SVDClutterRemoval:
    """Static-clutter removal of a Range-Time Map (RTM) with a rank-k SVD.

    For an RTM X in R^{T x 368} the right singular vectors are the eigenvectors of the Gram matrix
    G = X^T X (368 x 368).  The k dominant eigenvectors V_k span the static background (furniture,
    walls) so the residual is  R = X - X V_k V_k^T.
    `process_stream` applies this independently on consecutive blocks (default 10 s) because the
    published streams are concatenations of different recordings with different clutter.
    """

    def __init__(self, k: int = 2, block_frames: int = 500):
        self.k = int(k)
        self.block_frames = int(block_frames)

    def __call__(self, rtm: np.ndarray) -> np.ndarray:
        """rtm: (T, 368) -> residual (T, 368), float32."""
        X = np.asarray(rtm, dtype=np.float64)
        gram = X.T @ X                                    # (368, 368), symmetric PSD
        _, eigvecs = np.linalg.eigh(gram)                 # eigenvalues ascending
        Vk = eigvecs[:, -self.k:]                         # k dominant (static) directions
        residual = X - (X @ Vk) @ Vk.T
        return residual.astype(np.float32)

    def _radar_blocks(self, T: int):
        B = self.block_frames
        edges = list(range(0, T, B)) + [T]
        if len(edges) > 2 and edges[-1] - edges[-2] < B // 2:   # merge a too-short tail block
            edges.pop(-2)
        return list(zip(edges[:-1], edges[1:]))

    def process_radar(self, rtm: np.ndarray, out: np.ndarray = None) -> np.ndarray:
        out = np.empty(rtm.shape, dtype=np.float32) if out is None else out
        for a, b in self._radar_blocks(rtm.shape[0]):
            out[a:b] = self(rtm[a:b])
        return out

    def process_stream(self, cir: np.ndarray, inplace: bool = True) -> np.ndarray:
        """cir: (3, T, 368) float32 -> residual (3, T, 368) float32.

        Radars are processed sequentially: LAPACK already multi-threads the 368x368 eigh and Python-level
        threads around it proved unstable on some platforms."""
        out = cir if inplace else np.empty_like(cir)
        for r in range(cir.shape[0]):
            self.process_radar(cir[r], out[r])
        return out


def extract_12_spatial_bands(residual_cir: np.ndarray) -> np.ndarray:
    """L2 norm over 4 range bands (bins 0-65, 75-180, 183-225, 260-367) for each of 3 radars.

    residual_cir: (3, Seq_Len, 368) -> (12, Seq_Len) float32, channel = radar * 4 + band.
    """
    assert residual_cir.ndim == 3 and residual_cir.shape[2] == NUM_BINS, residual_cir.shape
    R, T, _ = residual_cir.shape
    bands = np.empty((R * NUM_BANDS, T), dtype=np.float32)
    for r in range(R):
        for b, (lo, hi) in enumerate(SPATIAL_BANDS):
            seg = residual_cir[r, :, lo:hi].astype(np.float64, copy=False)
            bands[r * NUM_BANDS + b] = np.sqrt(np.einsum("tb,tb->t", seg, seg))
    return bands


class RobustIQRNormalization:
    """Per-channel robust scaling  (x - median) / IQR   with IQR = Q75 - Q25.

    Fitted once per stream (statistics are dominated by background, falls are rare), then applied to
    any slice of that stream, so training crops and inference blocks see identical scaling.
    """

    def __init__(self, eps: float = 1e-12, clip: float = 50.0):
        self.eps, self.clip = eps, clip
        self.median = None
        self.iqr = None

    def fit(self, x: np.ndarray):
        """x: (C, T)."""
        x64 = np.asarray(x, dtype=np.float64)
        q25, med, q75 = np.percentile(x64, [25, 50, 75], axis=1)
        iqr = q75 - q25
        fallback = x64.std(axis=1)                                   # degenerate (flat) channels
        self.iqr = np.where(iqr > self.eps, iqr, np.maximum(fallback, self.eps))
        self.median = med
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        y = (np.asarray(x, dtype=np.float64) - self.median[:, None]) / self.iqr[:, None]
        if self.clip is not None:
            y = np.clip(y, -self.clip, self.clip)
        return y.astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


# ----------------------------------------------------------------------------
# Whole-stream preprocessing with an on-disk cache (bands for Stage 2, log-amplitude RTM for Stage 1)
# ----------------------------------------------------------------------------
class StreamData:
    """Container: bands (12, T) float32 | rtm (3, T, 368) float16 memmap | n_frames | duration."""

    def __init__(self, stream_id, bands, rtm):
        self.stream_id, self.bands, self.rtm = stream_id, bands, rtm
        self.n_frames = int(bands.shape[1])
        self.duration = self.n_frames / FS_HZ


def preprocess_stream(stream_id: str, h5_path: Path = None, cache_dir: Path = CACHE_DIR,
                      svd: SVDClutterRemoval = None, force: bool = False) -> StreamData:
    cache_dir = Path(cache_dir)
    bands_p, rtm_p = cache_dir / f"{stream_id}_bands.npy", cache_dir / f"{stream_id}_rtm.npy"
    if bands_p.exists() and rtm_p.exists() and not force:
        return StreamData(stream_id, np.load(bands_p), np.load(rtm_p, mmap_mode="r"))

    svd = svd or SVDClutterRemoval(k=2)
    t0 = time.time()
    cir = load_stream_cir(stream_id, h5_path=h5_path)                         # (3, T, 368) from disk or .zip
    assert cir.shape[0] == NUM_RADARS and cir.shape[2] == NUM_BINS, cir.shape
    residual = svd.process_stream(cir, inplace=True)                          # static clutter removed

    bands = extract_12_spatial_bands(residual)                                # (12, T)
    bands = RobustIQRNormalization().fit_transform(bands)                     # (x - median) / IQR

    # Stage-1 image: log1p(|residual| / robust per-radar scale) stored as float16 on disk
    T = residual.shape[1]
    rtm = np.lib.format.open_memmap(rtm_p, mode="w+", dtype=np.float16, shape=(NUM_RADARS, T, NUM_BINS))
    for r in range(NUM_RADARS):
        scale = float(np.median(np.abs(residual[r, ::25]))) + 1e-12
        for a in range(0, T, 20000):
            rtm[r, a:a + 20000] = np.log1p(np.abs(residual[r, a:a + 20000]) / scale).astype(np.float16)
    rtm.flush(); del rtm
    np.save(bands_p, bands)
    del residual, cir; gc.collect()
    print(f"  [preprocess] {stream_id}: T={T} ({T / FS_HZ:.0f}s) in {time.time() - t0:.0f}s")
    return StreamData(stream_id, bands, np.load(rtm_p, mmap_mode="r"))
