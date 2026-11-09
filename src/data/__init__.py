"""
Data loading, corpus metadata analysis, and synthetic UWB CIR stream utilities.
"""

from .dataset import (
    DT_SEC,
    FS_HZ,
    N_ACTIVE_BINS,
    N_RADARS,
    N_RAW_BINS,
    RADAR_SERIALS,
    UWBStreamRecord,
    compute_eda_summary_statistics,
    load_composition_dataframe,
    load_events_dataframe,
    load_uwb_stream,
)

__all__ = [
    "FS_HZ",
    "DT_SEC",
    "N_RADARS",
    "N_RAW_BINS",
    "N_ACTIVE_BINS",
    "RADAR_SERIALS",
    "UWBStreamRecord",
    "load_uwb_stream",
    "load_events_dataframe",
    "load_composition_dataframe",
    "compute_eda_summary_statistics",
]
