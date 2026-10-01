from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import analyze_phrag_bucket_highq as analysis


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "2026-07-20_phrag_crc_indoor_vpd_flow600"
PREFIX = "phrag_crc_indoor_vpd_flow600"

analysis.DATASET_DIR = DATASET_DIR
analysis.LOG_PATH = DATASET_DIR / "2026-07-20-1029_logdata_aq4_phrag_crc"
analysis.FLASH_DIR = DATASET_DIR / "flashes"
analysis.RUN_LABEL = "Phragmites CRC Indoor, VPD-Gated Flow-600"
analysis.VPD_SANE_RANGE = (0.55, 3.30)
analysis.FLOW_SANE_RANGE = (580, 620)

analysis.POINTS_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_points.csv"
analysis.FLASH_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_flash_summary.csv"
analysis.JOINED_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_points_with_flash.csv"
analysis.PHASE_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_phase_summary.csv"
analysis.FLASH_JOIN_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_flash_join_report.csv"

analysis.TIMELINE_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_timeline.png"
analysis.STOMATAL_TS_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_stomatal_time_series.png"
analysis.RAMP_QUALITY_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_ramp_quality.png"
analysis.ACI_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_aci_by_q.png"
analysis.SURFACE_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_a_surface_3d.png"
analysis.AQC_SURFACE_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_aqc_surface_3d.png"
analysis.FLUORO_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_fluorescence.png"

CONTROLLER_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_controller_diagnostic.png"
CONTROLLER_SUMMARY_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_controller_summary_by_q.csv"
PREFLASH_ACI_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_aci_preflash_anchors.png"
PREFLASH_POINTS_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_preflash_anchor_points.csv"
RAW_ACI_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_aci_all_logged_rows.png"


def controller_summary(points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for phase_id, group in points.groupby("phase_id", sort=True):
        rows.append(
            {
                "phase_id": int(phase_id),
                "Q": float(group["Q_level"].median()),
                "n": int(len(group)),
                "start_min": float(group["minute"].min()),
                "end_min": float(group["minute"].max()),
                "A_max": float(group["A"].max()),
                "Ci_max": float(group["Ci"].max()),
                "gsw_median": float(group["gsw"].median()),
                "gsw_max": float(group["gsw"].max()),
                "Tleaf_median": float(group["TleafCnd"].median()),
                "Tleaf_min": float(group["TleafCnd"].min()),
                "Tleaf_max": float(group["TleafCnd"].max()),
                "VPDleaf_median": float(group["VPDleaf"].median()),
                "VPDleaf_min": float(group["VPDleaf"].min()),
                "VPDleaf_max": float(group["VPDleaf"].max()),
                "VPDleaf_abs_error_median": float((group["VPDleaf"] - 1.0).abs().median()),
                "H2O_r_sp_median": float(group["H2O_r_sp"].median()),
                "H2O_r_median": float(group["H2O_r"].median()),
                "H2O_s_median": float(group["H2O_s"].median()),
                "Humidifier_pct_median": float(group["Humidifier_%"].median()),
                "Humidifier_pct_max": float(group["Humidifier_%"].max()),
                "Desiccant_pct_median": float(group["Desiccant_%"].median()),
                "Desiccant_pct_max": float(group["Desiccant_%"].max()),
                "Flow_median": float(group["Flow"].median()),
                "Fan_speed_median": float(group["Fan_speed"].median()),
            }
        )
    return pd.DataFrame(rows)


def plot_controller_diagnostic(points: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.72)
    fig, axes = plt.subplots(
        6,
        1,
        figsize=(15.8, 13.2),
        sharex=True,
        gridspec_kw={"height_ratios": [0.70, 1.0, 1.0, 1.05, 1.0, 1.0]},
    )

    for _, group in points.groupby("phase_id", sort=True):
        q_level = float(group["Q_level"].median())
        color = analysis.q_color(q_level)
        for ax in axes:
            ax.axvspan(group["minute"].min(), group["minute"].max(), color=color, alpha=0.08, lw=0)
        axes[0].text(
            group["minute"].mean(),
            0.84,
            f"{int(q_level)}",
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="top",
            color=color,
            fontsize=8.5,
            fontweight="bold",
        )

    axes[0].plot(points["minute"], points["Q"], color="#333333", lw=1.4)
    axes[0].set_ylabel("Q")

    axes[1].plot(points["minute"], points["A"], color="#2f8f46", lw=1.15)
    axes[1].axhline(0, color="#555555", lw=0.8, ls=":")
    axes[1].set_ylabel("A")

    axes[2].plot(points["minute"], points["VPDleaf"], color="#7b4a3b", lw=1.2)
    axes[2].axhline(1.0, color="#333333", lw=1.05, ls="--", label="VPDleaf target")
    axes[2].set_ylabel("VPDleaf\n(kPa)")
    axes[2].set_ylim(0.75, 3.25)
    axes[2].legend(frameon=False, loc="upper left", fontsize=8.5)

    axes[3].plot(points["minute"], points["H2O_r_sp"], color="#333333", lw=1.0, label="H2O_r_sp")
    axes[3].plot(points["minute"], points["H2O_r"], color="#2a9d8f", lw=1.0, label="H2O_r")
    axes[3].plot(points["minute"], points["H2O_s"], color="#4c78a8", lw=1.0, label="H2O_s")
    axes[3].set_ylabel("H2O\n(mmol mol-1)")
    axes[3].legend(frameon=False, ncol=3, loc="upper left", fontsize=8.5)

    axes[4].plot(points["minute"], points["Humidifier_%"], color="#2a9d8f", lw=1.0, label="Humidifier")
    axes[4].plot(points["minute"], points["Desiccant_%"], color="#e76f51", lw=1.0, label="Desiccant")
    axes[4].set_ylabel("Actuator\n(%)")
    axes[4].legend(frameon=False, ncol=2, loc="upper left", fontsize=8.5)

    axes[5].plot(points["minute"], points["TleafCnd"], color="#d95f70", lw=1.15, label="Tleaf")
    axes[5].axhline(25.0, color="#333333", lw=1.05, ls="--", label="Tleaf target")
    axes[5].plot(points["minute"], points["Txchg_sp"], color="#2475a8", lw=1.0, label="Txchg_sp")
    axes[5].set_ylabel("Temperature\n(C)")
    axes[5].set_xlabel("Elapsed time after first logged point (min)")
    axes[5].legend(frameon=False, ncol=3, loc="upper left", fontsize=8.5)

    for ax in axes:
        analysis.style_axes(ax)

    fig.text(
        0.06,
        0.985,
        "Phragmites CRC Indoor: Flow-600 VPD Controller Test",
        ha="left",
        va="top",
        fontsize=21,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.956,
        "VPD held at Q=0 and Q>=1700, but the conditioner actively dried the reference stream during Q=200-1300.",
        ha="left",
        va="top",
        fontsize=10.5,
        color="#555555",
    )
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.065, top=0.92, hspace=0.17)
    fig.savefig(CONTROLLER_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def preflash_anchor_points(points: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "elapsed",
        "minute",
        "Q_level",
        "Q",
        "A",
        "Asty",
        "Ci",
        "Ca",
        "gsw",
        "VPDleaf",
        "TleafCnd",
        "CO2_r_sp",
        "CO2_r",
        "CO2_s",
        "H2O_r_sp",
        "H2O_r",
        "H2O_s",
    ]
    rows: list[dict[str, float | str]] = []
    events = points.loc[points["light_flash_logged"] & points["Q_level"].gt(0)]
    for _, event in events.iterrows():
        window = points.loc[
            points["phase_id"].eq(event["phase_id"])
            & points["elapsed"].between(event["elapsed"] - 12.0, event["elapsed"] - 2.0)
        ]
        if window.empty:
            window = event.to_frame().T
        row = {
            column: float(pd.to_numeric(window[column], errors="coerce").median())
            for column in numeric_columns
        }
        row["LightAdaptedID"] = str(event["LightAdaptedID"])
        row["source_n"] = float(len(window))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_preflash_aci(anchors: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.80)
    fig, ax = plt.subplots(figsize=(14.8, 8.8))
    label_rows = []
    for q_level, group in anchors.groupby("Q_level", sort=True):
        color = analysis.q_color(float(q_level))
        group = group.sort_values("Ci")
        ax.plot(group["Ci"], group["A"], color=color, lw=2.3)
        ax.scatter(group["Ci"], group["A"], s=54, facecolor="white", edgecolor=color, linewidth=1.8, zorder=3)
        end = group.loc[group["Ci"].idxmax()]
        label_rows.append((float(end["Ci"]), float(end["A"]), int(q_level), color))

    for x_value, y_value, q_label, color in label_rows:
        ax.annotate(
            f"{q_label}",
            xy=(x_value, y_value),
            xytext=(14, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            color=color,
            fontsize=9,
            fontweight="bold",
            clip_on=False,
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.86},
        )

    ax.axhline(0, color="#555555", lw=0.9, ls=":")
    ax.set_xlim(0, min(1900, float(anchors["Ci"].max()) + 160))
    ax.set_ylim(0, max(18.5, float(anchors["A"].max()) + 1.0))
    ax.set_xlabel("Ci (umol mol-1)")
    ax.set_ylabel("DAT Adyn (umol CO2 m-2 s-1)")
    analysis.style_axes(ax)

    norm = Normalize(vmin=min(analysis.ACTIVE_Q_LEVELS), vmax=max(analysis.ACTIVE_Q_LEVELS))
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=ax, pad=0.025)
    colorbar.set_label("Q (umol photons m-2 s-1)")
    fig.text(
        0.075,
        0.965,
        "Phragmites CRC Indoor Flow-600: Pre-Flash A/Ci Anchors",
        ha="left",
        va="top",
        fontsize=21,
        fontweight="bold",
    )
    fig.text(
        0.075,
        0.925,
        "Each point is the median from 12 to 2 s before its MPF, excluding the flash-driven gas and thermal pulse.",
        ha="left",
        va="top",
        fontsize=10.5,
        color="#555555",
    )
    fig.subplots_adjust(left=0.09, right=0.90, bottom=0.10, top=0.86)
    fig.savefig(PREFLASH_ACI_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_raw_aci(points: pd.DataFrame) -> None:
    raw = points.loc[points["Q_level"].isin(analysis.ACTIVE_Q_LEVELS)].copy()
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.80)
    fig, ax = plt.subplots(figsize=(14.8, 8.8))

    for q_level, group in raw.groupby("Q_level", sort=True):
        color = analysis.q_color(float(q_level))
        group = group.sort_values("elapsed")
        ax.plot(group["Ci"], group["A"], color=color, lw=0.9, alpha=0.72)
        ax.scatter(group["Ci"], group["A"], color=color, s=7, alpha=0.34, linewidth=0)

    ax.axhline(0, color="#555555", lw=0.9, ls=":")
    ax.set_xlabel("Ci (umol mol-1)")
    ax.set_ylabel("DAT Adyn (umol CO2 m-2 s-1)")
    analysis.style_axes(ax)

    norm = Normalize(vmin=min(analysis.ACTIVE_Q_LEVELS), vmax=max(analysis.ACTIVE_Q_LEVELS))
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=ax, pad=0.025)
    colorbar.set_label("Q (umol photons m-2 s-1)")
    fig.text(
        0.075,
        0.965,
        "Phragmites CRC Indoor Flow-600: Every Logged A/Ci Row",
        ha="left",
        va="top",
        fontsize=21,
        fontweight="bold",
    )
    fig.text(
        0.075,
        0.925,
        "No post-flash removal, binning, smoothing, or curve-quality filtering; lines follow acquisition order.",
        ha="left",
        va="top",
        fontsize=10.5,
        color="#555555",
    )
    fig.subplots_adjust(left=0.09, right=0.90, bottom=0.10, top=0.86)
    fig.savefig(RAW_ACI_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    analysis.main()
    run_points = analysis.load_points()
    summary = controller_summary(run_points)
    anchors = preflash_anchor_points(run_points)
    summary.to_csv(CONTROLLER_SUMMARY_OUT, index=False)
    anchors.to_csv(PREFLASH_POINTS_OUT, index=False)
    plot_controller_diagnostic(run_points)
    plot_raw_aci(run_points)
    plot_preflash_aci(anchors)
    print(f"  {CONTROLLER_OUT}")
    print(f"  {CONTROLLER_SUMMARY_OUT}")
    print(f"  {PREFLASH_ACI_OUT}")
    print(f"  {PREFLASH_POINTS_OUT}")
    print(f"  {RAW_ACI_OUT}")
    print("\nController summary:")
    print(summary.to_string(index=False))
