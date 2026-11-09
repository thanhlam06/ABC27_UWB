"""Align the three radars of one recording on a common 50 Hz grid."""

from __future__ import annotations

import numpy as np

from abc2027.annotate import RadarTrack, load_radar_tracks

RATE_HZ = 50.0
STEP_S = 1.0 / RATE_HZ
SERIALS = (29171, 29444, 30722)
CIR_BINS = 368


def align_tracks(tracks: list[RadarTrack]) -> tuple[np.ndarray, np.ndarray, float]:
    """Nearest-frame CIR on a 50 Hz grid.

    Returns ``cir`` with shape ``(3, T, 368)`` float32, ``t`` in seconds from the
    common origin, and the fraction of grid samples whose nearest frame is more
    than one step away.
    """
    by_serial = {track.serial: track for track in tracks}
    t_last = max(float(track.time_s[-1]) for track in tracks)
    count = max(1, int(np.floor(t_last / STEP_S)) + 1)
    grid = np.arange(count, dtype=np.float64) * STEP_S
    cir = np.zeros((len(SERIALS), count, CIR_BINS), dtype=np.float32)
    missing = 0
    checked = 0
    for index, serial in enumerate(SERIALS):
        track = by_serial.get(serial)
        if track is None or len(track.time_s) == 0:
            missing += count
            checked += count
            continue
        places = np.searchsorted(track.time_s, grid, side="left")
        places = np.clip(places, 1, len(track.time_s) - 1)
        previous = places - 1
        choose_prev = np.abs(grid - track.time_s[previous]) <= np.abs(grid - track.time_s[places])
        places = np.where(choose_prev, previous, places)
        distance = np.abs(grid - track.time_s[places])
        missing += int(np.sum(distance > STEP_S))
        checked += count
        cir[index] = track.cir[places].astype(np.float32, copy=False)
    return cir, grid.astype(np.float64), missing / max(checked, 1)


def load_aligned(path) -> tuple[np.ndarray, np.ndarray, float]:
    return align_tracks(load_radar_tracks(path))
