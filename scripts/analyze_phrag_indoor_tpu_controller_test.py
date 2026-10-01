from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from photo26.curves import read_licor_export


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "2026-07-14_phrag_indoor_tpu_controller_test"
LOG_PATH = DATA_DIR / "2026-07-14-1225_logdata_phrag_crc_tpu_indoor"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

POINTS_OUT = OUTPUT_DIR / "phrag_indoor_tpu_controller_points.csv"
PHASE_OUT = OUTPUT_DIR / "phrag_indoor_tpu_controller_phase_summary.csv"
PLOT_OUT = OUTPUT_DIR / "phrag_indoor_tpu_vpd_controller.png"

PHASES = [
    ("Opening: 120", 20.0, 1098.0, "#dceaf5"),
    ("420 pre", 1103.5, 1460.0, "#dcefe2"),
    ("750 pre", 1460.5, 1943.1, "#f4e6b8"),
    ("1800 challenge", 1943.6, 2852.1, "#f3d1ca"),
    ("750 return", 2852.6, 3458.5, "#f4e6b8"),
    ("420 return", 3459.0, 3941.5, "#dcefe2"),
]


def load_points() -> pd.DataFrame:
    points = read_licor_export(LOG_PATH).sort_values("elapsed").reset_index(drop=True).copy()
    points["minute"] = points["elapsed"] / 60
    for column in ["VPDleaf", "TleafCnd", "H2O_r_sp", "H2O_r", "H2O_s", "gsw"]:
        points[f"{column}_smooth"] = points[column].rolling(15, center=True, min_periods=5).median()
    return points


def summarize_phases(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, start, end, _ in PHASES:
        group = points.loc[points["elapsed"].between(start, end)]
        tail = group.loc[group["elapsed"].ge(group["elapsed"].max() - 60)]
        rows.append(
            {
                "phase": label,
                "start_min": start / 60,
                "end_min": end / 60,
                "duration_min": (end - start) / 60,
                "Q_median": group["Q"].median(),
                "CO2_r_sp_median": group["CO2_r_sp"].median(),
                "VPDleaf_median": group["VPDleaf"].median(),
                "VPDleaf_p05": group["VPDleaf"].quantile(0.05),
                "VPDleaf_p95": group["VPDleaf"].quantile(0.95),
                "VPDleaf_last60": tail["VPDleaf"].median(),
                "Tleaf_median": group["TleafCnd"].median(),
                "Tleaf_p05": group["TleafCnd"].quantile(0.05),
                "Tleaf_p95": group["TleafCnd"].quantile(0.95),
                "H2O_r_sp_median": group["H2O_r_sp"].median(),
                "H2O_r_median": group["H2O_r"].median(),
                "H2O_s_median": group["H2O_s"].median(),
                "Humidifier_median": group["Humidifier_%"].median(),
                "Humidifier_max": group["Humidifier_%"].max(),
                "gsw_median": group["gsw"].median(),
                "gsw_last60": tail["gsw"].median(),
            }
        )
    return pd.DataFrame(rows)


def style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(True, color="#d8ddd4", linewidth=0.75)


def shade_phases(axes: list[plt.Axes]) -> None:
    for label, start, end, color in PHASES:
        start_min = start / 60
        end_min = end / 60
        for axis in axes:
            axis.axvspan(start_min, end_min, color=color, alpha=0.48, linewidth=0)
        axes[0].text(
            (start_min + end_min) / 2,
            0.96,
            label,
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8.2,
        )


def plot_controller(points: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.68)
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(16, 11.8),
        sharex=True,
        gridspec_kw={"height_ratios": [0.8, 1.35, 1.15, 0.95, 0.95]},
    )
    shade_phases(list(axes))

    axes[0].plot(points["minute"], points["Q"], color="#333333", lw=1.2)
    co2_axis = axes[0].twinx()
    co2_axis.plot(points["minute"], points["CO2_r_sp"], color="#d97925", lw=1.15)
    axes[0].set_ylabel("Q")
    co2_axis.set_ylabel("CO2 setpoint")

    axes[1].plot(points["minute"], points["VPDleaf"], color="#8f5a4a", alpha=0.18, lw=0.7)
    axes[1].plot(points["minute"], points["VPDleaf_smooth"], color="#8f5a4a", lw=1.8, label="VPDleaf")
    axes[1].axhline(1.0, color="#333333", lw=1.0, ls="--", label="target")
    axes[1].set_ylim(0.7, 1.6)
    axes[1].set_ylabel("Leaf VPD (kPa)")
    temperature_axis = axes[1].twinx()
    temperature_axis.plot(points["minute"], points["TleafCnd_smooth"], color="#1b8fa7", lw=1.25)
    temperature_axis.axhline(25.0, color="#1b8fa7", lw=0.8, ls=":")
    temperature_axis.set_ylim(24.0, 26.0)
    temperature_axis.set_ylabel("Tleaf (C)")
    axes[1].legend(frameon=False, loc="upper right", fontsize=8.5)

    axes[2].plot(points["minute"], points["H2O_r_sp_smooth"], color="#333333", lw=1.3, label="H2O_r setpoint")
    axes[2].plot(points["minute"], points["H2O_r_smooth"], color="#2a9d8f", lw=1.3, label="H2O_r")
    axes[2].plot(points["minute"], points["H2O_s_smooth"], color="#4c78a8", lw=1.3, label="H2O_s")
    axes[2].set_ylabel("H2O (mmol mol-1)")
    axes[2].legend(frameon=False, ncol=3, loc="upper right", fontsize=8.5)

    axes[3].plot(points["minute"], points["Humidifier_%"], color="#2a9d8f", lw=1.1, label="Humidifier")
    axes[3].plot(points["minute"], points["Desiccant_%"], color="#e76f51", lw=1.1, label="Desiccant")
    axes[3].axhline(-30, color="#555555", lw=0.8, ls=":")
    axes[3].set_ylabel("Conditioner (%)")
    axes[3].legend(frameon=False, ncol=2, loc="upper right", fontsize=8.5)

    axes[4].plot(points["minute"], points["gsw"], color="#2475a6", alpha=0.16, lw=0.7)
    axes[4].plot(points["minute"], points["gsw_smooth"], color="#2475a6", lw=1.6)
    axes[4].set_ylim(0, 0.16)
    axes[4].set_ylabel("gsw")
    axes[4].set_xlabel("Elapsed logged time (min)")

    for axis in axes:
        style_axis(axis)

    fig.text(
        0.06,
        0.985,
        "Phragmites Indoor TPU Test: Leaf-VPD Control",
        ha="left",
        va="top",
        fontsize=21,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.952,
        "Flow 600 and fan 10,000 rpm. VPD control engages late in the first 750-ppm phase, leaving pre- and post-challenge baselines environmentally unmatched.",
        ha="left",
        va="top",
        fontsize=10.5,
        color="#555555",
    )
    fig.subplots_adjust(left=0.085, right=0.92, bottom=0.07, top=0.91, hspace=0.16)
    fig.savefig(PLOT_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    points = load_points()
    summary = summarize_phases(points)
    points.to_csv(POINTS_OUT, index=False)
    summary.to_csv(PHASE_OUT, index=False)
    plot_controller(points)
    print(summary.to_string(index=False))
    print(PLOT_OUT)


if __name__ == "__main__":
    main()
