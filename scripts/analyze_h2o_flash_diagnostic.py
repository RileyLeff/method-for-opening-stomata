from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from photo26.curves import read_licor_export


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "2026-07-20_phrag_crc_h2o_flash_diagnostic"
LOG_PATH = DATA_DIR / "2026-07-20-1615_logdata_aqh2o_phrag_crc"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TIMELINE_OUT = OUTPUT_DIR / "phrag_h2o_flash_diagnostic_timeline.png"
AB_OUT = OUTPUT_DIR / "phrag_h2o_flash_diagnostic_ab_comparison.png"
POINTS_OUT = OUTPUT_DIR / "phrag_h2o_flash_diagnostic_points.csv"
SUMMARY_OUT = OUTPUT_DIR / "phrag_h2o_flash_diagnostic_phase_summary.csv"

PHASES = [
    ("Equilibrate", 0.0, 240.0, "#e8ece9"),
    ("A: no-flash ramp", 240.0, 421.0, "#d9e9f2"),
    ("Recover 1", 421.0, 542.0, "#e7f0df"),
    ("B: Default MPF ramp", 542.0, 723.0, "#f5e5ba"),
    ("Recover 2", 723.0, 843.0, "#e7f0df"),
    ("C: one MPF", 843.0, 966.0, "#efd9d4"),
]


def load_points() -> pd.DataFrame:
    points = read_licor_export(LOG_PATH).sort_values("elapsed").reset_index(drop=True).copy()
    points["minute"] = points["elapsed"] / 60
    points["phase"] = "Unassigned"
    points["phase_second"] = np.nan
    for label, start, end, _ in PHASES:
        mask = points["elapsed"].ge(start) & points["elapsed"].lt(end)
        points.loc[mask, "phase"] = label
        points.loc[mask, "phase_second"] = points.loc[mask, "elapsed"] - start
    points["is_flash"] = points["LightAdaptedID"].notna() & points["LightAdaptedID"].ne("-")
    return points


def summarize(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, start, end, _ in PHASES:
        group = points.loc[points["phase"].eq(label)]
        rows.append(
            {
                "phase": label,
                "start_s": start,
                "end_s": float(group["elapsed"].max()),
                "n": len(group),
                "Q_mean": group["Qin"].mean(),
                "CO2_r_sp_min": group["CO2_r_sp"].min(),
                "CO2_r_sp_max": group["CO2_r_sp"].max(),
                "VPDleaf_mean": group["VPDleaf"].mean(),
                "VPDleaf_min": group["VPDleaf"].min(),
                "VPDleaf_max": group["VPDleaf"].max(),
                "VPD_abs_error_median": (group["VPDleaf"] - 1.0).abs().median(),
                "H2O_r_mean": group["H2O_r"].mean(),
                "H2O_r_sp_mean": group["H2O_r_sp"].mean(),
                "Humidifier_min": group["Humidifier_%"].min(),
                "Humidifier_max": group["Humidifier_%"].max(),
                "Desiccant_min": group["Desiccant_%"].min(),
                "Desiccant_max": group["Desiccant_%"].max(),
                "Tleaf_mean": group["Tleaf"].mean(),
                "Tleaf_min": group["Tleaf"].min(),
                "Tleaf_max": group["Tleaf"].max(),
                "flash_count": group.loc[group["is_flash"], "LightAdaptedID"].nunique(),
                "full_dry_rows": int(
                    (group["Humidifier_%"].eq(0) & group["Desiccant_%"].eq(100)).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(True, color="#d8ddd9", linewidth=0.7)


def shade_phases(axes: list[plt.Axes]) -> None:
    for label, start, end, color in PHASES:
        x0, x1 = start / 60, end / 60
        for axis in axes:
            axis.axvspan(x0, x1, color=color, alpha=0.48, linewidth=0)
        axes[0].text(
            (x0 + x1) / 2,
            0.96,
            label.replace(" ", "\n", 1),
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8.1,
            color="#303633",
        )


def mark_flashes(axes: list[plt.Axes], points: pd.DataFrame) -> None:
    flashes = points.loc[points["is_flash"]].drop_duplicates("LightAdaptedID")
    for elapsed in flashes["elapsed"]:
        minute = elapsed / 60
        for axis in axes:
            axis.axvline(minute, color="#bd3f34", lw=0.9, ls="--", alpha=0.72)


def plot_timeline(points: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.70)
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(16, 12.2),
        sharex=True,
        gridspec_kw={"height_ratios": [0.85, 1.15, 1.0, 1.0, 0.9]},
    )
    shade_phases(list(axes))
    mark_flashes(list(axes), points)

    axes[0].plot(points["minute"], points["CO2_r_sp"], color="#d66a2c", lw=1.25)
    axes[0].set_ylabel("CO2 setpoint\n(ppm)")
    q_axis = axes[0].twinx()
    q_axis.plot(points["minute"], points["Qin"], color="#343936", lw=0.85, alpha=0.75)
    q_axis.set_ylim(0, 930)
    q_axis.set_ylabel("Q")

    axes[1].plot(points["minute"], points["VPDleaf"], color="#7754a6", lw=1.15)
    axes[1].axhline(1.0, color="#303633", lw=1.0, ls=":", label="1.0-kPa target")
    axes[1].set_ylim(0.84, 1.20)
    axes[1].set_ylabel("VPDleaf\n(kPa)")
    axes[1].legend(frameon=False, loc="upper left", fontsize=8.5)

    axes[2].plot(points["minute"], points["H2O_r_sp"], color="#303633", lw=1.1, label="H2O_r setpoint")
    axes[2].plot(points["minute"], points["H2O_r"], color="#218c75", lw=1.15, label="H2O_r")
    axes[2].plot(points["minute"], points["H2O_s"], color="#3178a8", lw=1.0, label="H2O_s")
    axes[2].set_ylabel("H2O\n(mmol mol-1)")
    axes[2].legend(frameon=False, ncol=3, loc="upper left", fontsize=8.2)

    axes[3].plot(points["minute"], points["Humidifier_%"], color="#218c75", lw=1.2, label="Humidifier")
    axes[3].plot(points["minute"], points["Desiccant_%"], color="#d05247", lw=1.2, label="Desiccant")
    axes[3].set_ylim(0, 50)
    axes[3].set_ylabel("Actuator\n(%)")
    axes[3].legend(frameon=False, ncol=2, loc="upper left", fontsize=8.2)

    axes[4].plot(points["minute"], points["Tleaf"], color="#b73f69", lw=1.15)
    axes[4].axhline(25.0, color="#303633", lw=1.0, ls=":")
    axes[4].set_ylim(24.75, 25.65)
    axes[4].set_ylabel("Tleaf (C)")
    axes[4].set_xlabel("Elapsed logged time (min)")

    for axis in axes:
        style_axis(axis)

    fig.text(
        0.065,
        0.985,
        "Phragmites H2O / Fluorometer Diagnostic",
        ha="left",
        va="top",
        fontsize=21,
        fontweight="bold",
    )
    fig.text(
        0.065,
        0.957,
        "Raw one-second observations. Red dashed lines mark MPFs; neither flash condition triggered the previous full-dry actuator lock.",
        ha="left",
        va="top",
        fontsize=10.5,
        color="#555b57",
    )
    fig.subplots_adjust(left=0.09, right=0.92, bottom=0.065, top=0.91, hspace=0.17)
    fig.savefig(TIMELINE_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ab(points: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.76)
    fig, axes = plt.subplots(4, 1, figsize=(13.8, 10.5), sharex=True)
    colors = {"A: no-flash ramp": "#2878a5", "B: Default MPF ramp": "#d28a24"}

    for phase, color in colors.items():
        group = points.loc[points["phase"].eq(phase)]
        x = group["phase_second"]
        axes[0].plot(x, group["CO2_r_sp"], color=color, lw=1.35, label=phase)
        axes[1].plot(x, group["VPDleaf"], color=color, lw=1.2)
        axes[2].plot(x, group["H2O_r"], color=color, lw=1.2)
        axes[3].plot(x, group["Desiccant_%"], color=color, lw=1.2)

    flash_seconds = points.loc[
        points["phase"].eq("B: Default MPF ramp") & points["is_flash"],
        "phase_second",
    ]
    for second in flash_seconds:
        for axis in axes:
            axis.axvline(second, color="#bd3f34", lw=0.9, ls="--", alpha=0.7)

    axes[0].set_ylabel("CO2 setpoint\n(ppm)")
    axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    axes[1].axhline(1.0, color="#303633", lw=1.0, ls=":")
    axes[1].set_ylabel("VPDleaf\n(kPa)")
    axes[1].set_ylim(0.94, 1.18)
    axes[2].set_ylabel("H2O_r\n(mmol mol-1)")
    axes[3].set_ylabel("Desiccant\n(%)")
    axes[3].set_xlabel("Seconds from ramp start")

    for axis in axes:
        style_axis(axis)

    fig.text(
        0.075,
        0.985,
        "Matched Ramp Comparison: No Flash vs Default MPF",
        ha="left",
        va="top",
        fontsize=20,
        fontweight="bold",
    )
    fig.text(
        0.075,
        0.952,
        "The MPFs generate short thermal/VPD pulses, but reference humidity and actuator feedback remain controlled.",
        ha="left",
        va="top",
        fontsize=10.5,
        color="#555b57",
    )
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.075, top=0.90, hspace=0.18)
    fig.savefig(AB_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    points = load_points()
    summary = summarize(points)
    points.to_csv(POINTS_OUT, index=False)
    summary.to_csv(SUMMARY_OUT, index=False)
    plot_timeline(points)
    plot_ab(points)
    print(summary.to_string(index=False))
    print(TIMELINE_OUT)
    print(AB_OUT)


if __name__ == "__main__":
    main()
