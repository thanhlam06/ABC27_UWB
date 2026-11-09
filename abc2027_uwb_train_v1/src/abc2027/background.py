"""Causal exponential-moving-average background removal, per radar and per bin."""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter, lfilter_zi

from abc2027.align import RATE_HZ


def ema_alpha(tau_s: float, rate_hz: float = RATE_HZ) -> float:
    """Smoothing factor of an exponential average with time constant ``tau_s``."""
    if tau_s <= 0:
        raise ValueError("tau_s must be positive")
    return float(np.exp(-1.0 / (rate_hz * tau_s)))


def subtract_background(cir: np.ndarray, tau_s: float, rate_hz: float = RATE_HZ) -> np.ndarray:
    """Subtract a causal exponential average from one stream.

    ``cir`` has shape ``(n_radars, T, n_bins)``. The average runs along time,
    independently for every radar and every range bin. Segment boundaries are
    not used. With ``alpha = exp(-dt / tau)`` and ``dt = 1 / rate_hz``:

        s_0 = x_0
        s_t = alpha * s_{t-1} + (1 - alpha) * x_t
        y_t = x_t - s_t

    The state is started as though ``x_0`` had been held forever, which gives
    ``s_0 = x_0``. The result is float32.
    """
    if cir.ndim != 3:
        raise ValueError("cir must have shape (n_radars, T, n_bins)")
    alpha = ema_alpha(tau_s, rate_hz)
    radars, count, bins = cir.shape
    frames = np.ascontiguousarray(np.moveaxis(cir, 1, -1), dtype=np.float64).reshape(radars * bins, count)
    gain = 1.0 - alpha
    numerator = [gain]
    denominator = [1.0, -alpha]
    steady = lfilter_zi(numerator, denominator)
    residual = np.empty_like(frames)
    for index in range(frames.shape[0]):
        average, _state = lfilter(numerator, denominator, frames[index], zi=frames[index, 0] * steady)
        residual[index] = frames[index] - average
    restored = np.moveaxis(residual.reshape(radars, bins, count), -1, 1)
    return restored.astype(np.float32, copy=False)
