from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import analyze_phrag_bucket_highq as analysis
from photo26.curves import read_licor_export


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "2026-07-21_phrag_crc_fixed_h2or_async_mpf"
PREFIX = "phrag_crc_fixed_h2or_async"

analysis.DATASET_DIR = DATASET_DIR
analysis.LOG_PATH = DATASET_DIR / "2026-07-21-1033_logdata_aqci_phrag_crc"
analysis.FLASH_DIR = DATASET_DIR / "flashes"
analysis.RUN_LABEL = "Phragmites CRC, Fixed H2O_r + Async MPF"
analysis.VPD_SANE_RANGE = (0.75, 1.30)
analysis.FLOW_SANE_RANGE = (580, 620)

analysis.POINTS_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_points.csv"
analysis.FLASH_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_flash_summary.csv"
analysis.JOINED_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_points_with_flash.csv"
analysis.PHASE_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_phase_summary.csv"
analysis.FLASH_JOIN_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_flash_join_report.csv"

analysis.TIMELINE_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_timeline_raw.png"
analysis.STOMATAL_TS_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_stomatal_time_series.png"
analysis.RAMP_QUALITY_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_ramp_quality.png"
analysis.ACI_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_aci_filtered.png"
analysis.SURFACE_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_a_surface_3d.png"
analysis.AQC_SURFACE_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_aqc_surface_3d.png"
analysis.FLUORO_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_fluorescence.png"

CONTROLLER_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_fixed_h2or_diagnostic.png"
CONTROLLER_SUMMARY_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_controller_summary.csv"
RAW_ACI_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_aci_raw_all_rows.png"

FLASH_NAME = re.compile(
    r"MPF-\d+-(?P<date>\d{8})-(?P<hour>\d{2})_(?P<minute>\d{2})_(?P<second>\d{2})"
)


def flash_time_from_name(path: Path) -> pd.Timestamp:
    match = FLASH_NAME.fullmatch(path.stem)
    if match is None:
        raise ValueError(f"Unrecognized MPF filename: {path.name}")
    values = match.groupdict()
    return pd.to_datetime(
        f"{values['date']} {values['hour']}:{values['minute']}:{values['second']}",
        format="%Y%m%d %H:%M:%S",
    )


def active_flash_events() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(analysis.FLASH_DIR.glob("MPF-*.json")):
        payload = json.loads(path.read_text())
        light_id = str(payload.get("FLR", {}).get("LightAdaptedID", "-"))
        if light_id in {"", "-", "None"}:
            continue
        rows.append(
            {
                "flash_id": path.stem,
                "flash_datetime": flash_time_from_name(path),
                "pre_qin": float(payload.get("Pre_Qin", np.nan)),
            }
        )
    return pd.DataFrame(rows).sort_values("flash_datetime").reset_index(drop=True)


def load_points_async() -> pd.DataFrame:
    points = read_licor_export(analysis.LOG_PATH).sort_values("elapsed").reset_index(drop=True).copy()
    points["minute"] = points["elapsed"] / 60
    points["datetime"] = pd.to_datetime(points["date"], format="%Y%m%d %H:%M:%S")
    points["LightAdaptedID_raw"] = points["LightAdaptedID"].astype(str)
    points["DarkAdaptedID_raw"] = points["DarkAdaptedID"].astype(str)

    # MPFs briefly drive Qin above 10,000. Carry the surrounding chamber-light
    # value through those pulses to recover the intended Q level of each ramp.
    points["Q_nominal"] = points["Q"].where(points["Q"].le(2500)).ffill().bfill()
    points["Q_level"] = analysis.nearest_q(points["Q_nominal"])
    phase_break = points["Q_level"].ne(points["Q_level"].shift()) | points["minute"].diff().gt(5)
    points["phase_id"] = phase_break.cumsum()
    points["phase_min"] = points["minute"] - points.groupby("phase_id")["minute"].transform("min")
    points["co2_command_jump_back"] = points.groupby("phase_id")["CO2_r_sp"].diff().lt(-20)

    points["LightAdaptedID"] = "-"
    points["light_flash_logged"] = False
    events = active_flash_events()
    point_ns = points["datetime"].to_numpy(dtype="datetime64[ns]").astype("int64")
    high_q_times = points.loc[points["Q"].gt(3000), "datetime"]
    observed_offsets: list[float] = []
    for event_time in events["flash_datetime"]:
        offsets = (high_q_times - event_time).dt.total_seconds()
        plausible = offsets.loc[offsets.between(15, 50)]
        if not plausible.empty:
            observed_offsets.append(float(plausible.iloc[np.argmin((plausible - 32).abs())]))
    flash_clock_offset = float(np.median(observed_offsets))
    events["flash_datetime"] = events["flash_datetime"] + pd.to_timedelta(flash_clock_offset, unit="s")
    points["mpf_filename_to_gas_offset_s"] = flash_clock_offset
    event_ns = events["flash_datetime"].to_numpy(dtype="datetime64[ns]").astype("int64")
    for event_index, event_time_ns in enumerate(event_ns):
        row_index = int(np.argmin(np.abs(point_ns - event_time_ns)))
        if abs(point_ns[row_index] - event_time_ns) <= 3_000_000_000:
            points.at[row_index, "LightAdaptedID"] = events.at[event_index, "flash_id"]
            points.at[row_index, "light_flash_logged"] = True

    points["dark_flash_logged"] = points["DarkAdaptedID"].astype(str).ne("-")
    points["seconds_after_mpf"] = np.nan
    if len(event_ns):
        latest = np.searchsorted(event_ns, point_ns, side="right") - 1
        valid = latest >= 0
        seconds = np.full(len(points), np.nan)
        seconds[valid] = (point_ns[valid] - event_ns[latest[valid]]) / 1e9
        points["seconds_after_mpf"] = seconds
    points["mpf_or_recovery"] = points["seconds_after_mpf"].between(0, 25, inclusive="both")

    points["A_minus_Asty"] = points["A"] - points["Asty"]
    points["Ci_Ca"] = points["Ci"] / points["Ca"]
    points["sane_for_curve"] = (
        points["Q_level"].isin(analysis.ACTIVE_Q_LEVELS)
        & points["Ci"].between(0, 2200)
        & points["A"].between(-20, 40)
        & points["gsw"].between(0.005, 1.0)
        & points["CO2_r"].between(40, 2050)
        & points["CO2_s"].between(40, 2050)
        & points["Flow"].between(*analysis.FLOW_SANE_RANGE)
        & points["VPDleaf"].between(*analysis.VPD_SANE_RANGE)
        & points["Q"].sub(points["Q_level"]).abs().lt(75)
        & ~points["mpf_or_recovery"]
    )
    return points


analysis.load_points = load_points_async


def controller_summary(points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for phase_id, group in points.groupby("phase_id", sort=True):
        clean = group.loc[~group["mpf_or_recovery"]]
        rows.append(
            {
                "phase_id": int(phase_id),
                "Q": int(group["Q_level"].median()),
                "n_rows": int(len(group)),
                "duration_min": float(group["minute"].max() - group["minute"].min()),
                "n_mpf": int(group["light_flash_logged"].sum()),
                "A_max_clean": float(clean["A"].max()),
                "Ci_max_clean": float(clean["Ci"].max()),
                "gsw_median_clean": float(clean["gsw"].median()),
                "gsw_max_clean": float(clean["gsw"].max()),
                "VPDleaf_median": float(group["VPDleaf"].median()),
                "VPDleaf_min": float(group["VPDleaf"].min()),
                "VPDleaf_max": float(group["VPDleaf"].max()),
                "Tleaf_median": float(group["TleafCnd"].median()),
                "Tleaf_max": float(group["TleafCnd"].max()),
                "H2O_r_sp_median": float(group["H2O_r_sp"].median()),
                "H2O_r_median": float(group["H2O_r"].median()),
                "H2O_s_median": float(group["H2O_s"].median()),
                "Humidifier_pct_median": float(group["Humidifier_%"].median()),
                "Desiccant_pct_median": float(group["Desiccant_%"].median()),
                "Flow_median": float(group["Flow"].median()),
            }
        )
    return pd.DataFrame(rows)


def add_phase_bands(axes: np.ndarray, points: pd.DataFrame) -> None:
    for _, group in points.groupby("phase_id", sort=True):
        q_level = float(group["Q_level"].median())
        color = analysis.q_color(q_level)
        for ax in axes:
            ax.axvspan(group["minute"].min(), group["minute"].max(), color=color, alpha=0.075, lw=0)
        axes[0].text(
            group["minute"].mean(),
            0.88,
            f"{int(q_level)}",
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="top",
            color=color,
            fontsize=8.5,
            fontweight="bold",
        )


def plot_fixed_h2or_diagnostic(points: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.72)
    fig, axes = plt.subplots(
        6,
        1,
        figsize=(16, 13.4),
        sharex=True,
        gridspec_kw={"height_ratios": [0.72, 1.0, 1.0, 1.05, 1.0, 1.0]},
    )
    add_phase_bands(axes, points)

    axes[0].plot(points["minute"], points["Q_nominal"], color="#333333", lw=1.35, label="Nominal Q")
    flash_rows = points.loc[points["light_flash_logged"]]
    axes[0].scatter(flash_rows["minute"], flash_rows["Q_level"], marker="|", s=65, color="#d1495b", label="MPF")
    axes[0].set_ylabel("Q")
    axes[0].legend(frameon=False, ncol=2, loc="upper left", fontsize=8.5)

    axes[1].plot(points["minute"], points["A"], color="#2f8f46", lw=1.1)
    axes[1].axhline(0, color="#555555", lw=0.8, ls=":")
    axes[1].set_ylabel("A")

    axes[2].plot(points["minute"], points["VPDleaf"], color="#7b4a3b", lw=1.15)
    axes[2].axhline(1.0, color="#333333", lw=1.0, ls="--", label="1.0 kPa reference")
    axes[2].set_ylabel("VPDleaf\n(kPa)")
    axes[2].legend(frameon=False, loc="upper left", fontsize=8.5)

    axes[3].plot(points["minute"], points["H2O_r_sp"], color="#333333", lw=1.1, label="H2O_r setpoint")
    axes[3].plot(points["minute"], points["H2O_r"], color="#2a9d8f", lw=1.0, label="H2O_r")
    axes[3].plot(points["minute"], points["H2O_s"], color="#4c78a8", lw=1.0, label="H2O_s")
    axes[3].set_ylabel("H2O\n(mmol mol-1)")
    axes[3].legend(frameon=False, ncol=3, loc="upper left", fontsize=8.5)

    axes[4].plot(points["minute"], points["Humidifier_%"], color="#2a9d8f", lw=1.0, label="Humidifier")
    axes[4].plot(points["minute"], points["Desiccant_%"], color="#e76f51", lw=1.0, label="Desiccant")
    axes[4].set_ylabel("Actuator\n(%)")
    axes[4].legend(frameon=False, ncol=2, loc="upper left", fontsize=8.5)

    axes[5].plot(points["minute"], points["TleafCnd"], color="#d95f70", lw=1.1, label="Tleaf")
    axes[5].axhline(25.0, color="#333333", lw=1.0, ls="--", label="25 C target")
    axes[5].plot(points["minute"], points["Txchg_sp"], color="#2475a8", lw=1.0, label="Txchg setpoint")
    axes[5].set_ylabel("Temperature\n(C)")
    axes[5].set_xlabel("Elapsed time after first logged point (min)")
    axes[5].legend(frameon=False, ncol=3, loc="upper left", fontsize=8.5)

    for ax in axes:
        analysis.style_axes(ax)

    fig.text(0.06, 0.985, "Phragmites CRC: Fixed Incoming Moisture", ha="left", va="top", fontsize=21, fontweight="bold")
    fig.text(
        0.06,
        0.955,
        "H2O_r stayed fixed at 21 mmol mol-1 while VPDleaf remained near 1 kPa across the complete Q ladder.",
        ha="left",
        va="top",
        fontsize=10.5,
        color="#555555",
    )
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.065, top=0.92, hspace=0.17)
    fig.savefig(CONTROLLER_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_raw_aci(points: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.82)
    fig, ax = plt.subplots(figsize=(15, 9))
    active = points.loc[points["Q_level"].isin(analysis.ACTIVE_Q_LEVELS)]
    for q_level, group in active.groupby("Q_level", sort=True):
        color = analysis.q_color(float(q_level))
        ax.plot(group["Ci"], group["A"], color=color, lw=1.1, alpha=0.82)
        ax.scatter(group["Ci"], group["A"], color=color, s=8, alpha=0.34, linewidth=0)
        endpoint = group.loc[group["Ci"].idxmax()]
        ax.annotate(
            f"{int(q_level)}",
            (endpoint["Ci"], endpoint["A"]),
            xytext=(12, 0),
            textcoords="offset points",
            va="center",
            color=color,
            fontsize=9,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
        )
    ax.axhline(0, color="#555555", lw=0.9, ls=":")
    ax.set_xlim(0, 2200)
    ax.set_ylim(0, max(19.0, float(active["A"].max()) + 1))
    ax.set_xlabel("Ci (umol mol-1)")
    ax.set_ylabel("DAT Adyn (umol CO2 m-2 s-1)")
    analysis.style_axes(ax)
    fig.text(0.07, 0.975, "Phragmites CRC A/Ci Ramps: Every Logged Row", ha="left", va="top", fontsize=21, fontweight="bold")
    fig.text(
        0.07,
        0.938,
        "Raw acquisition order is retained, including MPF and post-MPF points; the filtered companion plot flags those windows explicitly.",
        ha="left",
        va="top",
        fontsize=10.5,
        color="#555555",
    )
    fig.subplots_adjust(left=0.085, right=0.95, bottom=0.09, top=0.89)
    fig.savefig(RAW_ACI_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    analysis.main()
    run_points = load_points_async()
    summary = controller_summary(run_points)
    summary.to_csv(CONTROLLER_SUMMARY_OUT, index=False)
    plot_fixed_h2or_diagnostic(run_points)
    plot_raw_aci(run_points)
    print(f"  {CONTROLLER_OUT}")
    print(f"  {CONTROLLER_SUMMARY_OUT}")
    print(f"  {RAW_ACI_OUT}")
    print("\nFixed-H2O_r summary:")
    print(summary.to_string(index=False))
