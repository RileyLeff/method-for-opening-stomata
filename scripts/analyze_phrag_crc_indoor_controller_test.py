from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import analyze_phrag_bucket_highq as analysis


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "2026-07-13_phrag_crc_indoor_controller_test"
PREFIX = "phrag_crc_indoor_controller_test"

analysis.DATASET_DIR = DATASET_DIR
analysis.LOG_PATH = DATASET_DIR / "2026-07-13-1217_logdata_phrag_test3_crc_indoor"
analysis.FLASH_DIR = DATASET_DIR / "flashes"
analysis.RUN_LABEL = "Phragmites CRC Indoor, Controller Test"
analysis.VPD_SANE_RANGE = (0.55, 1.50)

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

ENV_STOMATAL_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_vpd_q_ci_ca_gsw_time_series.png"
CONTROLLER_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_controller_diagnostic.png"
CONTROLLER_SUMMARY_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_controller_summary_by_q.csv"
PREFLASH_ACI_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_aci_preflash_anchors.png"
PREFLASH_POINTS_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_preflash_anchor_points.csv"
RAW_ACI_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_aci_all_raw_rows.png"


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
                "Tleaf_median": float(group["TleafCnd"].median()),
                "Tleaf_min": float(group["TleafCnd"].min()),
                "Tleaf_max": float(group["TleafCnd"].max()),
                "Tleaf_abs_error_median": float((group["TleafCnd"] - 25.0).abs().median()),
                "VPDleaf_median": float(group["VPDleaf"].median()),
                "VPDleaf_min": float(group["VPDleaf"].min()),
                "VPDleaf_max": float(group["VPDleaf"].max()),
                "VPDleaf_abs_error_median": float((group["VPDleaf"] - 1.0).abs().median()),
                "Txchg_sp_median": float(group["Txchg_sp"].median()),
                "H2O_r_sp_median": float(group["H2O_r_sp"].median()),
                "H2O_r_median": float(group["H2O_r"].median()),
                "H2O_s_median": float(group["H2O_s"].median()),
                "Humidifier_pct_max": float(group["Humidifier_%"].max()),
                "Desiccant_pct_max": float(group["Desiccant_%"].max()),
            }
        )
    return pd.DataFrame(rows)


def plot_controller_diagnostic(points: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.72)
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(15.8, 11.8),
        sharex=True,
        gridspec_kw={"height_ratios": [0.75, 1.15, 1.0, 1.1, 1.0]},
    )

    for _, group in points.groupby("phase_id", sort=True):
        q_level = float(group["Q_level"].median())
        color = analysis.q_color(q_level)
        for ax in axes:
            ax.axvspan(group["minute"].min(), group["minute"].max(), color=color, alpha=0.08, lw=0)
        axes[0].text(
            group["minute"].mean(),
            0.82,
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

    axes[1].plot(points["minute"], points["TleafCnd"], color="#d95f70", lw=1.15, label="Tleaf")
    axes[1].axhline(25.0, color="#333333", lw=1.05, ls="--", label="Tleaf target")
    axes[1].plot(points["minute"], points["Txchg_sp"], color="#2475a8", lw=1.0, label="Txchg_sp")
    axes[1].set_ylabel("Temperature\n(C)")
    axes[1].legend(frameon=False, ncol=3, loc="upper left", fontsize=8.5)

    axes[2].plot(points["minute"], points["VPDleaf"], color="#7b4a3b", lw=1.2)
    axes[2].axhline(1.0, color="#333333", lw=1.05, ls="--", label="VPD target")
    axes[2].set_ylabel("VPDleaf\n(kPa)")
    axes[2].legend(frameon=False, loc="upper left", fontsize=8.5)

    axes[3].plot(points["minute"], points["H2O_r_sp"], color="#333333", lw=1.0, label="H2O_r_sp")
    axes[3].plot(points["minute"], points["H2O_r"], color="#2a9d8f", lw=1.0, label="H2O_r")
    axes[3].plot(points["minute"], points["H2O_s"], color="#4c78a8", lw=1.0, label="H2O_s")
    axes[3].set_ylabel("H2O\n(mmol mol-1)")
    axes[3].legend(frameon=False, ncol=3, loc="upper left", fontsize=8.5)

    axes[4].plot(points["minute"], points["Humidifier_%"], color="#2a9d8f", lw=1.0, label="Humidifier")
    axes[4].plot(points["minute"], points["Desiccant_%"], color="#e76f51", lw=1.0, label="Desiccant")
    axes[4].axhline(-30, color="#777777", lw=0.8, ls=":", label="Bypass")
    axes[4].set_ylabel("Actuator\n(%)")
    axes[4].set_xlabel("Elapsed time after first logged point (min)")
    axes[4].legend(frameon=False, ncol=3, loc="upper left", fontsize=8.5)

    for ax in axes:
        analysis.style_axes(ax)

    fig.text(
        0.06,
        0.982,
        "Phragmites CRC Indoor: Temperature and Humidity Control",
        ha="left",
        va="top",
        fontsize=21,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.951,
        "Tleaf is actively held near 25 C; VPD improves substantially but remains about 0.2 kPa above its requested target during most ramps.",
        ha="left",
        va="top",
        fontsize=10.5,
        color="#555555",
    )
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.07, top=0.91, hspace=0.18)
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
        row = {column: float(pd.to_numeric(window[column], errors="coerce").median()) for column in numeric_columns}
        row["LightAdaptedID"] = str(event["LightAdaptedID"])
        row["source_n"] = float(len(window))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_preflash_aci(points: pd.DataFrame) -> None:
    anchors = preflash_anchor_points(points)
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.80)
    fig, ax = plt.subplots(figsize=(14.8, 8.8))
    for q_level, group in anchors.groupby("Q_level", sort=True):
        color = analysis.q_color(float(q_level))
        group = group.sort_values("Ci")
        ax.plot(group["Ci"], group["A"], color=color, lw=2.3)
        ax.scatter(group["Ci"], group["A"], s=54, facecolor="white", edgecolor=color, linewidth=1.8, zorder=3)

    ax.axhline(0, color="#555555", lw=0.9, ls=":")
    ax.set_xlim(0, 1950)
    ax.set_ylim(0, 17.2)
    ax.set_xlabel("Ci (umol mol-1)")
    ax.set_ylabel("DAT Adyn (umol CO2 m-2 s-1)")
    analysis.style_axes(ax)

    norm = Normalize(vmin=min(analysis.ACTIVE_Q_LEVELS), vmax=max(analysis.ACTIVE_Q_LEVELS))
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=ax, pad=0.025)
    colorbar.set_label("Q (umol photons m-2 s-1)")
    fig.text(
        0.075,
        0.965,
        "Phragmites CRC Indoor: Pre-Flash A/Ci Anchors",
        ha="left",
        va="top",
        fontsize=21,
        fontweight="bold",
    )
    fig.text(
        0.075,
        0.925,
        "Each point is the median from 12 to 2 s before its MPF, avoiding the post-flash thermal and gas-exchange pulse.",
        ha="left",
        va="top",
        fontsize=10.5,
        color="#555555",
    )
    fig.subplots_adjust(left=0.09, right=0.90, bottom=0.10, top=0.86)
    fig.savefig(PREFLASH_ACI_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_raw_aci(points: pd.DataFrame) -> None:
    raw = points.loc[
        points["Q_level"].isin(analysis.ACTIVE_Q_LEVELS)
        & points["Ci"].notna()
        & points["A"].notna()
    ].copy()
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.76)
    fig, (main_ax, full_ax) = plt.subplots(
        1,
        2,
        figsize=(16.2, 8.7),
        sharey=True,
        gridspec_kw={"width_ratios": [3.2, 1.25], "wspace": 0.10},
    )

    for q_level, group in raw.groupby("Q_level", sort=True):
        group = group.sort_values("elapsed")
        color = analysis.q_color(float(q_level))
        for ax in (main_ax, full_ax):
            ax.plot(group["Ci"], group["A"], color=color, lw=0.72, alpha=0.58)
            ax.scatter(
                group["Ci"],
                group["A"],
                s=8,
                color=color,
                alpha=0.42,
                linewidth=0,
            )
        flash_rows = group.loc[group["light_flash_logged"]]
        main_ax.scatter(
            flash_rows["Ci"],
            flash_rows["A"],
            s=42,
            facecolor="white",
            edgecolor=color,
            linewidth=1.2,
            zorder=4,
        )

    main_ax.set_xlim(-250, 2100)
    main_ax.set_xlabel("Ci (umol mol-1)")
    main_ax.set_ylabel("DAT Adyn (umol CO2 m-2 s-1)")
    main_ax.set_title("Chamber range", loc="left", fontsize=13, fontweight="bold")

    full_ax.set_xscale("symlog", linthresh=250, linscale=0.8)
    full_ax.set_xlim(float(raw["Ci"].min()) * 1.08, float(raw["Ci"].max()) * 1.08)
    full_ax.set_xlabel("Ci, symlog scale")
    full_ax.set_title("All derived Ci values", loc="left", fontsize=13, fontweight="bold")

    y_pad = max(0.8, float(raw["A"].max() - raw["A"].min()) * 0.05)
    main_ax.set_ylim(float(raw["A"].min()) - y_pad, float(raw["A"].max()) + y_pad)
    for ax in (main_ax, full_ax):
        ax.axhline(0, color="#555555", lw=0.9, ls=":")
        analysis.style_axes(ax)

    norm = Normalize(vmin=min(analysis.ACTIVE_Q_LEVELS), vmax=max(analysis.ACTIVE_Q_LEVELS))
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap="viridis"),
        ax=[main_ax, full_ax],
        pad=0.018,
        fraction=0.025,
    )
    colorbar.set_label("Q (umol photons m-2 s-1)")
    fig.text(
        0.065,
        0.975,
        "Phragmites CRC Indoor: Raw A/Ci Output",
        ha="left",
        va="top",
        fontsize=21,
        fontweight="bold",
    )
    fig.text(
        0.065,
        0.937,
        "Every active-Q log row in acquisition order; no filtering, binning, smoothing, or post-flash exclusion. White rings mark MPF log rows.",
        ha="left",
        va="top",
        fontsize=10.5,
        color="#555555",
    )
    fig.subplots_adjust(left=0.075, right=0.925, bottom=0.10, top=0.88)
    fig.savefig(RAW_ACI_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    analysis.main()
    points = analysis.load_points()
    analysis.plot_vpd_q_ci_ca_gsw_time_series(points, ENV_STOMATAL_OUT)
    plot_controller_diagnostic(points)
    anchors = preflash_anchor_points(points)
    plot_preflash_aci(points)
    plot_raw_aci(points)
    anchors.to_csv(PREFLASH_POINTS_OUT, index=False)
    controller_summary(points).to_csv(CONTROLLER_SUMMARY_OUT, index=False)
    print(f"  {ENV_STOMATAL_OUT}")
    print(f"  {RAW_ACI_OUT}")
    print(f"  {CONTROLLER_OUT}")
    print(f"  {CONTROLLER_SUMMARY_OUT}")
    print(f"  {PREFLASH_ACI_OUT}")
    print(f"  {PREFLASH_POINTS_OUT}")
