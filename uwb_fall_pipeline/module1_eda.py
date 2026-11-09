"""
Module 1: Advanced Exploratory Data Analysis (EDA) & Realistic UWB Stream Synthesis
9th ABC Challenge - UWB Fall Track Pipeline

Generates a physically realistic continuous UWB radar stream:
- 30,000 frames @ 50 Hz (600.0 seconds = 10 minutes)
- 3 UWB radars (mimicking Novelda XeThru X4M200 multi-view geometry)
- 368 fast-time range bins, 16-bit unsigned ADC scale [0, 65535]
- 3 simulated Fall events (2.0 - 3.0 seconds duration)
- 23 background Activities of Daily Living (ADL) events + Empty Room intervals

Outputs analytical console insights and 3 diagnostic EDA plots:
1. Temporal Duration Distribution (KDE plot) of Falls vs. ADLs
2. Class Imbalance Pie Chart (Binary Frame Imbalance & 3-Way Activity Breakdown)
3. Raw Range-Time Matrix (RTM) Heat Map for a 10-second window containing a Fall
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Stream Constants matching 9th ABC Challenge Format
N_FRAMES: int = 30_000
N_RADARS: int = 3
N_BINS: int = 368
FS_HZ: float = 50.0
DT_SEC: float = 1.0 / FS_HZ
ADC_MIN: float = 0.0
ADC_MAX: float = 65535.0


@dataclass
class UWBStreamDataset:
    """Container for simulated continuous UWB stream and ground-truth annotations."""
    raw_stream: np.ndarray          # Shape: (N_FRAMES, N_RADARS, N_BINS), float32 in [0, 65535]
    timestamps: np.ndarray          # Shape: (N_FRAMES,), seconds
    events_df: pd.DataFrame         # Ground-truth table of Fall and ADL intervals
    frame_labels: np.ndarray        # Shape: (N_FRAMES,), 0=Empty, 1=ADL, 2=Fall
    empty_frame_mask: np.ndarray    # Shape: (N_FRAMES,), True when room is empty & static


def _uwb_bandpass_wavelet(delta_bins: np.ndarray, width_bins: float, carrier_rad: float = 1.35) -> np.ndarray:
    """
    Zero-mean UWB impulse wavelet (Gaussian-windowed cosine / Gabor doublet).
    Models coherent constructive/destructive multipath modulation around static clutter.
    """
    envelope = np.exp(-0.5 * (delta_bins / width_bins) ** 2)
    carrier = np.cos(carrier_rad * delta_bins)
    wavelet = envelope * carrier
    # Enforce zero spatial mean per frame so human scattering carries no unphysical DC bias
    wavelet -= np.mean(wavelet, axis=-1, keepdims=True)
    return wavelet


def generate_mock_uwb_stream(seed: int = 42) -> UWBStreamDataset:
    """
    Synthesize a 30,000-frame multi-radar UWB CIR stream with realistic rank-2 static
    multipath clutter (room reflections + slow transceiver thermal drift), receiver noise,
    background ADL segments, and 3 Fall events (2-3s each).
    """
    rng = np.random.default_rng(seed)
    timestamps = np.arange(N_FRAMES, dtype=np.float64) * DT_SEC
    bins = np.arange(N_BINS, dtype=np.float64)

    # -------------------------------------------------------------------------
    # 1. Rank-2 Static Indoor Clutter (Static Room Profile + Slow Thermal Drift)
    # -------------------------------------------------------------------------
    # Mode 1: Static room walls, furniture, floor, and Tx-Rx antenna coupling [9,000 - 46,000]
    # Mode 2: Slow hardware transceiver warm-up & wall multipath thermal drift over 600s
    raw_stream = np.zeros((N_FRAMES, N_RADARS, N_BINS), dtype=np.float32)

    static_reflectors = [
        # (bin_center, width, amplitude_base) — walls, furniture, & Tx-Rx coupling
        (18.0, 7.0, 35000.0),   # Direct-path Tx-Rx antenna crosstalk
        (66.0, 9.0, 24000.0),   # Near room furniture / cabinet echo
        (232.0, 13.0, 26500.0), # Mid-far concrete partition wall
        (308.0, 15.0, 23500.0), # Far apartment boundary wall
    ]

    # Rank-2 static background temporal drift (transceiver warm-up + slow multipath phase wander)
    thermal_drift_t = (
        np.cos(2.0 * np.pi * timestamps / 600.0)
        + 0.55 * np.sin(2.0 * np.pi * timestamps / 19.0)
        + 0.40 * np.cos(2.0 * np.pi * timestamps / 11.5)
    )

    for r in range(N_RADARS):
        clutter_profile = np.full(N_BINS, 11500.0 + 400.0 * r, dtype=np.float64)
        for b_center, b_width, amp in static_reflectors:
            shifted_center = b_center + rng.uniform(-5.0, 5.0)
            shifted_amp = amp * rng.uniform(0.90, 1.10)
            clutter_profile += shifted_amp * np.exp(-0.5 * ((bins - shifted_center) / b_width) ** 2)

        ripple = 750.0 * np.sin(2.0 * np.pi * bins / (22.0 + 2.0 * r) + r)
        clutter_profile = np.clip(clutter_profile + ripple, 7500.0, 48000.0)

        # Secondary spatial mode for slow transceiver coupling & far-wall drift (~950 ADC counts)
        drift_spatial = (
            1150.0 * np.exp(-0.5 * ((bins - (18.0 + r)) / 14.0) ** 2)
            - 650.0 * np.exp(-0.5 * ((bins - (216.0 - 3.0 * r)) / 24.0) ** 2)
        )

        rtm_bg = clutter_profile[None, :] + thermal_drift_t[:, None] * drift_spatial[None, :]
        raw_stream[:, r, :] = rtm_bg.astype(np.float32)

    # Add realistic UWB pulse-integrated receiver thermal noise (sigma = 3.5 ADC counts)
    raw_stream += rng.normal(0.0, 3.5, size=raw_stream.shape).astype(np.float32)

    # -------------------------------------------------------------------------
    # 2. Ground-Truth Event Schedule: 3 Falls (2-3s) & Background ADLs
    # -------------------------------------------------------------------------
    fall_specs = [
        {
            "event_id": "FALL_01",
            "event_type": "Fall",
            "subtype": "Chute01_Forward",
            "t_start": 112.00,
            "t_end": 114.40,   # 2.40s (120 frames)
            "bin_start": 148.0,
            "bin_end": 120.0,
            "target_bin_min": 110,
            "target_bin_max": 158,
            "peak_amp": 4300.0,
        },
        {
            "event_id": "FALL_02",
            "event_type": "Fall",
            "subtype": "Chute02_Lateral",
            "t_start": 295.00,
            "t_end": 297.80,   # 2.80s (140 frames)
            "bin_start": 196.0,
            "bin_end": 168.0,
            "target_bin_min": 158,
            "target_bin_max": 206,
            "peak_amp": 4150.0,
        },
        {
            "event_id": "FALL_03",
            "event_type": "Fall",
            "subtype": "Chute03_Backward",
            "t_start": 478.00,
            "t_end": 480.20,   # 2.20s (110 frames)
            "bin_start": 138.0,
            "bin_end": 112.0,
            "target_bin_min": 102,
            "target_bin_max": 148,
            "peak_amp": 4450.0,
        },
    ]

    safe_zones = [
        (8.0, 98.0),
        (128.0, 281.0),
        (312.0, 464.0),
        (494.0, 592.0),
    ]
    adl_subtypes = [
        ("Walking", 5.0, 12.0, 1050.0, 1700.0),
        ("Cooking_Kitchen", 9.0, 19.0, 600.0, 1050.0),
        ("Computer_Desk", 11.0, 21.0, 420.0, 760.0),
        ("Sit_Stand_Transition", 3.6, 6.2, 1650.0, 2650.0),  # Vigorous hard-negative confuser
        ("Dishwashing", 7.5, 15.0, 680.0, 1150.0),
    ]

    adl_specs = []
    adl_idx = 1
    for z_start, z_end in safe_zones:
        cursor = z_start + rng.uniform(2.0, 5.0)
        while cursor < z_end - 8.0:
            subtype_name, d_min, d_max, a_min, a_max = adl_subtypes[(adl_idx - 1) % len(adl_subtypes)]
            duration = round(float(rng.uniform(d_min, d_max)) / DT_SEC) * DT_SEC
            if cursor + duration > z_end - 2.0:
                break
            b0 = float(rng.uniform(70.0, 275.0))
            b1 = float(np.clip(b0 + rng.uniform(-38.0, 38.0), 55.0, 300.0))
            adl_specs.append(
                {
                    "event_id": f"ADL_{adl_idx:02d}",
                    "event_type": "ADL",
                    "subtype": subtype_name,
                    "t_start": round(cursor, 2),
                    "t_end": round(cursor + duration, 2),
                    "bin_start": b0,
                    "bin_end": b1,
                    "target_bin_min": int(min(b0, b1) - 12),
                    "target_bin_max": int(max(b0, b1) + 12),
                    "peak_amp": float(rng.uniform(a_min, a_max)),
                }
            )
            adl_idx += 1
            cursor += duration + round(float(rng.uniform(5.5, 13.5)) / DT_SEC) * DT_SEC

    frame_labels = np.zeros(N_FRAMES, dtype=np.int32)  # 0=Empty, 1=ADL, 2=Fall
    radar_aspect_gains = np.array([1.00, 0.87, 0.79], dtype=np.float32)

    # -------------------------------------------------------------------------
    # 3. Inject Background ADL UWB Wavelet Signatures
    # -------------------------------------------------------------------------
    for adl in adl_specs:
        f_start = int(round(adl["t_start"] * FS_HZ))
        f_end = int(round(adl["t_end"] * FS_HZ))
        n_ev = f_end - f_start
        frame_labels[f_start:f_end] = 1

        tau = np.linspace(0.0, 1.0, n_ev, dtype=np.float64)
        gait_freq = 1.65 if adl["subtype"] == "Walking" else 0.70
        bin_traj = (
            adl["bin_start"]
            + (adl["bin_end"] - adl["bin_start"]) * tau
            + 2.2 * np.sin(2.0 * np.pi * gait_freq * tau * (n_ev * DT_SEC))
        )
        env = np.sin(np.pi * tau) ** 0.65
        if adl["subtype"] == "Sit_Stand_Transition":
            env = np.exp(-0.5 * ((tau - 0.5) / 0.20) ** 2)

        for r in range(N_RADARS):
            r_shift = (r - 1) * 6.0
            delta_b = bins[None, :] - (bin_traj[:, None] + r_shift)
            wavelet = _uwb_bandpass_wavelet(delta_b, width_bins=4.2, carrier_rad=1.25 + 0.1 * r)
            doppler_mod = np.cos(2.0 * np.pi * (1.5 * tau * n_ev * DT_SEC) + r)[:, None]
            adl_wave = adl["peak_amp"] * radar_aspect_gains[r] * env[:, None] * (0.75 + 0.25 * doppler_mod) * wavelet
            raw_stream[f_start:f_end, r, :] += adl_wave.astype(np.float32)

    # -------------------------------------------------------------------------
    # 4. Inject 3 Biomechanical Fall Signatures (2.2s - 2.8s)
    # -------------------------------------------------------------------------
    # Phases: (a) parabolic downward acceleration (0 -> 0.42),
    #         (b) high-energy floor impact burst (around 0.42),
    #         (c) post-impact mat stabilization / torso settling (0.42 -> 1.0, ~1.4s).
    for fall in fall_specs:
        f_start = int(round(fall["t_start"] * FS_HZ))
        f_end = int(round(fall["t_end"] * FS_HZ))
        n_ev = f_end - f_start
        frame_labels[f_start:f_end] = 2

        tau = np.linspace(0.0, 1.0, n_ev, dtype=np.float64)

        accel_phase = np.clip(tau / 0.36, 0.0, 1.0) ** 2
        damped_bounce = 0.8 * np.exp(-12.0 * np.maximum(tau - 0.36, 0.0)) * np.sin(20.0 * np.maximum(tau - 0.36, 0.0))
        bin_traj = fall["bin_start"] + (fall["bin_end"] - fall["bin_start"]) * accel_phase + damped_bounce

        descent_env = 0.50 * np.sin(np.pi * np.clip(tau / 0.45, 0.0, 1.0))
        impact_spike = 0.95 * np.exp(-0.5 * ((tau - 0.36) / 0.055) ** 2)
        # Quasi-stationary torso reflection on floor mat during stabilization (~1.5s > EMA tau=0.39s)
        settling_plateau = 0.88 * (tau >= 0.36) * np.sin(np.pi * np.clip((1.0 - tau) / 0.08, 0.0, 1.0))
        total_env = descent_env + impact_spike + settling_plateau

        for r in range(N_RADARS):
            r_shift = (r - 1) * 5.0
            delta_b = bins[None, :] - (bin_traj[:, None] + r_shift)
            spatial_sigma = (4.8 + 3.8 * np.exp(-0.5 * ((tau - 0.42) / 0.08) ** 2))[:, None]
            wavelet = _uwb_bandpass_wavelet(delta_b, width_bins=spatial_sigma, carrier_rad=1.20 + 0.08 * r)
            fall_wave = fall["peak_amp"] * radar_aspect_gains[r] * total_env[:, None] * wavelet
            raw_stream[f_start:f_end, r, :] += fall_wave.astype(np.float32)

    # Enforce strict 16-bit ADC dynamic range [0, 65535]
    np.clip(raw_stream, ADC_MIN, ADC_MAX, out=raw_stream)

    # Build Ground-Truth DataFrame
    all_events = fall_specs + adl_specs
    events_df = pd.DataFrame(all_events).sort_values("t_start").reset_index(drop=True)
    events_df["duration_s"] = (events_df["t_end"] - events_df["t_start"]).round(2)
    events_df["start_frame"] = (events_df["t_start"] * FS_HZ).round().astype(int)
    events_df["end_frame"] = (events_df["t_end"] * FS_HZ).round().astype(int)

    # Empty frames: frames with no event AND >=2.5s after any preceding event
    occupied = frame_labels > 0
    empty_frame_mask = ~occupied.copy()
    settling_guard = int(2.5 * FS_HZ)
    for ef in events_df["end_frame"].to_numpy():
        empty_frame_mask[ef : min(N_FRAMES, ef + settling_guard)] = False
    empty_frame_mask[:settling_guard] = False

    return UWBStreamDataset(
        raw_stream=raw_stream,
        timestamps=timestamps,
        events_df=events_df,
        frame_labels=frame_labels,
        empty_frame_mask=empty_frame_mask,
    )


def run_module1_eda(dataset: UWBStreamDataset, output_dir: Path) -> dict:
    """
    Execute Module 1 EDA: compute summary statistics, print analytical insights,
    and save the 3 required EDA plots (KDE Duration, Class Imbalance Pie, Raw Fall RTM).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = dataset.events_df
    falls_df = df[df["event_type"] == "Fall"]
    adls_df = df[df["event_type"] == "ADL"]

    fall_durations = falls_df["duration_s"].to_numpy()
    adl_durations = adls_df["duration_s"].to_numpy()

    n_fall_frames = int(np.sum(dataset.frame_labels == 2))
    n_adl_frames = int(np.sum(dataset.frame_labels == 1))
    n_empty_frames = int(np.sum(dataset.frame_labels == 0))
    n_nonfall_frames = n_adl_frames + n_empty_frames

    # -------------------------------------------------------------------------
    # Plot 1: Temporal Duration Distribution (KDE Plot) of Falls vs. ADLs
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    x_grid = np.linspace(0.5, max(adl_durations.max() + 3.0, 24.0), 500)

    kde_fall = gaussian_kde(fall_durations, bw_method=0.45)
    kde_adl = gaussian_kde(adl_durations, bw_method="scott")

    y_fall = kde_fall(x_grid)
    y_adl = kde_adl(x_grid)

    ax.plot(x_grid, y_fall, color="#D62728", lw=2.5, label=f"Fall Events (n={len(fall_durations)}, μ={fall_durations.mean():.2f}s)")
    ax.fill_between(x_grid, 0, y_fall, color="#D62728", alpha=0.22)

    ax.plot(x_grid, y_adl, color="#1F77B4", lw=2.5, label=f"Background ADLs (n={len(adl_durations)}, μ={adl_durations.mean():.2f}s)")
    ax.fill_between(x_grid, 0, y_adl, color="#1F77B4", alpha=0.20)

    ax.plot(fall_durations, np.full_like(fall_durations, 0.025), "|", color="#D62728", markersize=14, markeredgewidth=2.2)
    ax.plot(adl_durations, np.full_like(adl_durations, 0.012), "|", color="#1F77B4", markersize=10, markeredgewidth=1.5)

    ax.axvline(2.5, color="#444444", linestyle="--", lw=1.2, label="ML Sliding Window (2.50s / 125 frames)")
    ax.set_title("Module 1: Temporal Duration Distribution (KDE) — Falls vs. Background ADLs", fontsize=12, fontweight="bold")
    ax.set_xlabel("Event Duration (seconds)", fontsize=11)
    ax.set_ylabel("Probability Density (KDE)", fontsize=11)
    ax.set_xlim(0.5, x_grid.max())
    ax.set_ylim(bottom=0.0)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, fontsize=9.5)
    fig.tight_layout()
    kde_path = output_dir / "module1_duration_kde.png"
    fig.savefig(kde_path)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Plot 2: Class Imbalance Pie Chart (Clean Non-Overlapping Labels)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), dpi=160)

    def _autopct_large_only(pct: float) -> str:
        return f"{pct:.2f}%" if pct >= 5.0 else ""

    axes[0].pie(
        [n_nonfall_frames, n_fall_frames],
        explode=(0.0, 0.16),
        labels=[
            f"No Fall (ADL + Empty)\n{n_nonfall_frames:,} frames",
            f"Fall Event (Minority)\n{n_fall_frames:,} frames ({100.0 * n_fall_frames / N_FRAMES:.2f}%)",
        ],
        colors=["#4C78A8", "#E45756"],
        autopct=_autopct_large_only,
        pctdistance=0.55,
        startangle=18,
        textprops={"fontsize": 9.5, "fontweight": "bold"},
    )
    axes[0].set_title("Binary Frame Class Imbalance\n(1 : 80.1 Minority Target Ratio)", fontsize=11, fontweight="bold")

    axes[1].pie(
        [n_adl_frames, n_empty_frames, n_fall_frames],
        explode=(0.0, 0.0, 0.16),
        labels=[
            f"Active ADL\n({n_adl_frames:,} fr)",
            f"Empty Static Room\n({n_empty_frames:,} fr)",
            f"Fall Events\n({n_fall_frames:,} fr, {100.0 * n_fall_frames / N_FRAMES:.2f}%)",
        ],
        colors=["#72B7B2", "#BAB0AC", "#E45756"],
        autopct=_autopct_large_only,
        pctdistance=0.58,
        startangle=25,
        textprops={"fontsize": 9.5},
    )
    axes[1].set_title("3-Way Continuous Stream Breakdown\n(30,000 Frames = 600.0s @ 50 Hz)", fontsize=11, fontweight="bold")

    fig.suptitle("Module 1: UWB Radar Stream Class Imbalance Analysis", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    pie_path = output_dir / "module1_class_imbalance.png"
    fig.savefig(pie_path, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Plot 3: Raw Range-Time Matrix (RTM) Heat Map for 10s Window around FALL_01
    # -------------------------------------------------------------------------
    fall_row = falls_df.iloc[0]
    fall_mid = 0.5 * (fall_row["t_start"] + fall_row["t_end"])
    win_t0 = fall_mid - 5.0
    win_t1 = fall_mid + 5.0
    idx0 = int(round(win_t0 * FS_HZ))
    idx1 = int(round(win_t1 * FS_HZ))

    raw_rtm_10s = dataset.raw_stream[idx0:idx1, 0, :].T

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=160)
    im = ax.imshow(
        raw_rtm_10s,
        aspect="auto",
        origin="lower",
        extent=[win_t0, win_t1, 0, N_BINS - 1],
        cmap="inferno",
        vmin=ADC_MIN,
        vmax=52000.0,
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Raw CIR Amplitude (16-bit ADC Counts [0–65535])", fontsize=10)

    ax.axvline(fall_row["t_start"], color="#00FFCC", linestyle="--", lw=1.8, label=f"Fall Start ({fall_row['t_start']:.2f}s)")
    ax.axvline(fall_row["t_end"], color="#00FFCC", linestyle="-.", lw=1.8, label=f"Fall End ({fall_row['t_end']:.2f}s)")
    ax.axhspan(fall_row["target_bin_min"], fall_row["target_bin_max"], color="#00FFCC", alpha=0.08, label="Target Range Bins")

    ax.set_title(
        f"Module 1: Raw Range-Time Matrix (RTM) — 10s Window Around {fall_row['event_id']} (Radar 0)\n"
        "Dominant Horizontal Static Clutter Bands Obscure the Transient Human Fall Signature",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Stream Time (seconds)", fontsize=10.5)
    ax.set_ylabel("Fast-Time Range Bin (0 – 367)", fontsize=10.5)
    ax.legend(loc="upper right", frameon=True, facecolor="#111111", edgecolor="white", labelcolor="white", fontsize=9)
    fig.tight_layout()
    rtm_path = output_dir / "module1_raw_rtm_fall.png"
    fig.savefig(rtm_path)
    plt.close(fig)

    imbalance_ratio = n_nonfall_frames / max(n_fall_frames, 1)
    print("=" * 80)
    print("MODULE 1: ADVANCED EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 80)
    print(f"Stream Tensor Shape      : {dataset.raw_stream.shape} (Frames, Radars, Fast-Time Bins)")
    print(f"Sampling Rate & Duration : {FS_HZ:.1f} Hz | {N_FRAMES / FS_HZ:.1f} seconds (10.0 minutes)")
    print(f"ADC Dynamic Range        : [{dataset.raw_stream.min():.1f}, {dataset.raw_stream.max():.1f}] (Bounds: [0, 65535])")
    print(f"Injected Fall Events     : {len(falls_df)} events | Durations: {fall_durations.tolist()} s (Mean: {fall_durations.mean():.2f}s ± {fall_durations.std():.2f}s)")
    print(f"Injected ADL Events      : {len(adls_df)} events | Duration Range: [{adl_durations.min():.2f}s, {adl_durations.max():.2f}s] (Mean: {adl_durations.mean():.2f}s ± {adl_durations.std():.2f}s)")
    print(f"Frame Class Breakdown    : Fall = {n_fall_frames:,} ({100*n_fall_frames/N_FRAMES:.2f}%) | "
          f"ADL = {n_adl_frames:,} ({100*n_adl_frames/N_FRAMES:.2f}%) | Empty = {n_empty_frames:,} ({100*n_empty_frames/N_FRAMES:.2f}%)")
    print(f"Binary Imbalance Ratio   : 1 : {imbalance_ratio:.1f} (Fall : No-Fall)")
    print(f"Saved EDA Plots          : {kde_path.name}, {pie_path.name}, {rtm_path.name}")
    print("-" * 80)
    print("[Analytical Insight - Module 1]:")
    print("  1. Biomechanical Duration Separation: KDE analysis confirms Falls are tightly concentrated")
    print("     between 2.20s–2.80s, motivating a 2.50s (125-frame) sliding window in Module 4, whereas")
    print("     ADLs exhibit a broad right-skewed distribution (4.0s–21.0s).")
    print("  2. Severe Class Imbalance: Fall frames occupy only 1.23% of continuous streaming time")
    print(f"     (1:{imbalance_ratio:.1f} imbalance), meaning naive accuracy (>98.7% by predicting all No-Fall)")
    print("     is misleading; class-balanced training and minority-class Precision/Recall/F1 are mandatory.")
    print("  3. Clutter Masking on Raw RTM: Static antenna crosstalk and wall reflections reach ~46,000 ADC")
    print("     counts—over 10x stronger than human UWB wavelet scattering—visually burying FALL_01")
    print("     under bright horizontal clutter bands and proving clutter suppression is essential.")
    print()

    return {
        "n_frames": N_FRAMES,
        "n_radars": N_RADARS,
        "n_bins": N_BINS,
        "adc_min": float(dataset.raw_stream.min()),
        "adc_max": float(dataset.raw_stream.max()),
        "n_falls": int(len(falls_df)),
        "n_adls": int(len(adls_df)),
        "fall_frames": n_fall_frames,
        "adl_frames": n_adl_frames,
        "empty_frames": n_empty_frames,
        "imbalance_ratio": float(imbalance_ratio),
        "plots": [str(kde_path), str(pie_path), str(rtm_path)],
    }
