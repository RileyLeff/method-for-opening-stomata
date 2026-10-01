from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

from .curves import DATA_DIR, OUTPUT_DIR, read_licor_export


CONTROL_FILE_HINT = "morcer_sun_control"

STAGE_COLORS = ["#e9f5ea", "#eef3fb", "#fff7d6", "#e9f5ea"]
LINE_COLORS = {
    "A": "#2f8f46",
    "gsw": "#2475a8",
    "Ci": "#805bb7",
    "CO2_setpoint": "#333333",
    "CO2_reference": "#f28e2b",
    "CO2_sample": "#cc6677",
    "CO2_chamber": "#d64b3b",
    "H2O_setpoint": "#333333",
    "H2O_reference": "#44aa99",
    "H2O_sample": "#117733",
    "Q": "#2d2d2d",
    "Q_red": "#c43c39",
    "Q_blue": "#377eb8",
    "VPD": "#8c564b",
    "Tleaf": "#15aabf",
}


def find_control_file() -> Path:
    matches = sorted(DATA_DIR.glob(f"*{CONTROL_FILE_HINT}*"))
    if not matches:
        raise FileNotFoundError(f"No control run in {DATA_DIR} matches {CONTROL_FILE_HINT!r}")
    return matches[-1]


def load_control(path: Path | None = None) -> pd.DataFrame:
    path = path or find_control_file()
    frame = read_licor_export(path).copy()
    frame["minute"] = frame["elapsed"] / 60
    return frame


def co2_stages(frame: pd.DataFrame) -> pd.DataFrame:
    starts = frame.loc[
        frame["CO2_r_sp"].ne(frame["CO2_r_sp"].shift()), ["minute", "CO2_r_sp"]
    ].reset_index(drop=True)
    rows = []
    for idx, row in starts.iterrows():
        start = float(row["minute"])
        end = float(starts.loc[idx + 1, "minute"]) if idx + 1 < len(starts) else float(frame["minute"].max())
        rows.append(
            {
                "start": start,
                "end": end,
                "co2": int(round(float(row["CO2_r_sp"]))),
            }
        )
    return pd.DataFrame(rows)


def stage_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby("CO2_r_sp")
        .agg(
            n=("A", "size"),
            start_min=("minute", "min"),
            end_min=("minute", "max"),
            A_mean=("A", "mean"),
            A_last5=("A", lambda values: values.tail(5).mean()),
            A_max=("A", "max"),
            gsw_mean=("gsw", "mean"),
            gsw_last5=("gsw", lambda values: values.tail(5).mean()),
            gsw_max=("gsw", "max"),
            Ci_mean=("Ci", "mean"),
            Ci_last5=("Ci", lambda values: values.tail(5).mean()),
            Ca_mean=("Ca", "mean"),
            VPD_mean=("VPDleaf", "mean"),
            Tleaf_mean=("Tleaf", "mean"),
        )
        .reset_index()
        .rename(columns={"CO2_r_sp": "co2_setpoint"})
    )
    return summary


def shade_stages(ax: plt.Axes, stages: pd.DataFrame) -> None:
    for index, stage in stages.iterrows():
        ax.axvspan(stage["start"], stage["end"], color=STAGE_COLORS[index % len(STAGE_COLORS)], alpha=0.75, lw=0)
    for _, stage in stages.iloc[1:].iterrows():
        ax.axvline(stage["start"], color="#8a8a8a", lw=1, ls="--", alpha=0.75)


def style_timeseries_axis(ax: plt.Axes, stages: pd.DataFrame) -> None:
    shade_stages(ax, stages)
    ax.grid(True, color="#d7d7d2", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", colors="#333333")


def add_stage_header(ax: plt.Axes, stages: pd.DataFrame, x_max: float) -> None:
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for index, stage in stages.iterrows():
        color = STAGE_COLORS[index % len(STAGE_COLORS)]
        width = stage["end"] - stage["start"]
        ax.add_patch(Rectangle((stage["start"], 0.08), width, 0.78, facecolor=color, edgecolor="#c7c7c2", lw=0.8))
        ax.text(
            stage["start"] + width / 2,
            0.47,
            f"{int(stage['co2'])} ppm CO2\n{stage['start']:.0f}-{stage['end']:.0f} min",
            ha="center",
            va="center",
            fontsize=9.5,
            color="#333333",
        )


def line_legend(ax: plt.Axes, *, columns: int = 1) -> None:
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.02),
        borderaxespad=0,
        fontsize=9,
        ncol=columns,
    )


def plot_timeseries(frame: pd.DataFrame, stages: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="ticks", context="notebook", font_scale=1.0)
    plt.rcParams.update({"axes.titleweight": "bold", "figure.facecolor": "white"})

    fig = plt.figure(figsize=(13.5, 13.5))
    grid = GridSpec(
        7,
        1,
        height_ratios=[0.45, 1.35, 1.05, 1.05, 1.05, 0.9, 1.0],
        hspace=0.15,
        figure=fig,
    )
    header = fig.add_subplot(grid[0])
    axes = [fig.add_subplot(grid[i], sharex=header) for i in range(1, 7)]
    x_max = float(frame["minute"].max())

    fig.suptitle("Morella Sun Leaf Control", fontsize=19, y=0.985, fontweight="bold")
    fig.text(
        0.5,
        0.962,
        "Manual CO2 step run at constant Q = 800 and control spectrum R90B40",
        ha="center",
        va="top",
        fontsize=11,
        color="#555555",
    )
    add_stage_header(header, stages, x_max)

    for ax in axes:
        style_timeseries_axis(ax, stages)

    axes[0].plot(frame["minute"], frame["A"], color=LINE_COLORS["A"], marker="o", markersize=3.2, lw=1.8)
    axes[0].axhline(0, color="#555555", ls=":", lw=1)
    axes[0].set_ylabel("A\numol CO2\nm$^{-2}$ s$^{-1}$")
    peak = frame.loc[frame["A"].idxmax()]
    axes[0].annotate(
        f"spike {peak['A']:.1f}",
        xy=(peak["minute"], peak["A"]),
        xytext=(peak["minute"] + 5, peak["A"] - 1.2),
        arrowprops={"arrowstyle": "->", "color": "#4a4a4a", "lw": 1},
        fontsize=9,
        color="#333333",
    )

    axes[1].plot(frame["minute"], frame["gsw"], color=LINE_COLORS["gsw"], marker="o", markersize=3.2, lw=1.8)
    axes[1].axhline(0, color="#555555", ls=":", lw=1)
    axes[1].set_ylabel("gsw\nmol H2O\nm$^{-2}$ s$^{-1}$")

    ci_bad = (frame["Ci"] < 0) | (frame["Ci"] > 500)
    ci_display = frame["Ci"].mask(ci_bad)
    axes[2].plot(frame["minute"], ci_display, color=LINE_COLORS["Ci"], marker="o", markersize=3.2, lw=1.6)
    axes[2].scatter(
        frame.loc[ci_bad, "minute"],
        np.full(int(ci_bad.sum()), 330),
        color="#d64b3b",
        marker="^",
        s=42,
        zorder=4,
        label=f"{int(ci_bad.sum())} flagged Ci points",
    )
    axes[2].set_ylim(40, 350)
    axes[2].set_ylabel("Ci\numol mol$^{-1}$")
    line_legend(axes[2])

    axes[3].plot(
        frame["minute"],
        frame["CO2_r_sp"],
        color=LINE_COLORS["CO2_setpoint"],
        lw=2,
        drawstyle="steps-post",
        label="Setpoint",
    )
    axes[3].plot(frame["minute"], frame["CO2_r"], color=LINE_COLORS["CO2_reference"], lw=1.5, label="Reference")
    axes[3].plot(frame["minute"], frame["CO2_s"], color=LINE_COLORS["CO2_sample"], lw=1.5, label="Sample")
    axes[3].plot(frame["minute"], frame["Ca"], color=LINE_COLORS["CO2_chamber"], lw=1.5, label="Chamber Ca")
    axes[3].set_ylabel("CO2\numol mol$^{-1}$")
    line_legend(axes[3])

    axes[4].plot(frame["minute"], frame["Q"], color=LINE_COLORS["Q"], lw=1.8, label="Total Q")
    axes[4].plot(frame["minute"], frame["Q_red"], color=LINE_COLORS["Q_red"], lw=1.5, label="Red")
    axes[4].plot(frame["minute"], frame["Q_blue"], color=LINE_COLORS["Q_blue"], lw=1.5, label="Blue")
    axes[4].set_ylabel("Light\numol photons\nm$^{-2}$ s$^{-1}$")
    axes[4].set_ylim(0, 850)
    line_legend(axes[4])

    axes[5].plot(frame["minute"], frame["VPDleaf"], color=LINE_COLORS["VPD"], marker="o", markersize=2.8, lw=1.5, label="VPDleaf")
    temp_axis = axes[5].twinx()
    temp_axis.plot(frame["minute"], frame["Tleaf"], color=LINE_COLORS["Tleaf"], lw=1.6, label="Tleaf")
    axes[5].set_ylabel("VPDleaf\nkPa")
    temp_axis.set_ylabel("Tleaf (C)")
    temp_axis.spines[["top"]].set_visible(False)
    lines_1, labels_1 = axes[5].get_legend_handles_labels()
    lines_2, labels_2 = temp_axis.get_legend_handles_labels()
    axes[5].legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.02),
        borderaxespad=0,
        fontsize=9,
    )
    axes[5].set_xlabel("Elapsed time (min)")
    axes[5].set_xlim(0, x_max)
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
        ax.set_xlabel("")

    fig.subplots_adjust(top=0.93, left=0.105, right=0.82, bottom=0.055)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_gas_streams(frame: pd.DataFrame, stages: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="ticks", context="notebook", font_scale=1.0)
    fig = plt.figure(figsize=(13.5, 10.5))
    grid = GridSpec(5, 1, height_ratios=[0.42, 1.1, 1.35, 1.35, 1.15], hspace=0.17, figure=fig)
    header = fig.add_subplot(grid[0])
    axes = [fig.add_subplot(grid[i], sharex=header) for i in range(1, 5)]
    x_max = float(frame["minute"].max())

    fig.suptitle("Morella Control Gas Streams", fontsize=18, y=0.985, fontweight="bold")
    fig.text(
        0.5,
        0.958,
        "Reference and sample analyzer streams during the manual CO2 step run",
        ha="center",
        va="top",
        fontsize=11,
        color="#555555",
    )
    add_stage_header(header, stages, x_max)

    for ax in axes:
        style_timeseries_axis(ax, stages)

    axes[0].plot(frame["minute"], frame["A"], color=LINE_COLORS["A"], lw=1.8, label="A")
    conductance_axis = axes[0].twinx()
    conductance_axis.plot(frame["minute"], frame["gsw"], color=LINE_COLORS["gsw"], lw=1.6, label="gsw")
    axes[0].set_ylabel("A\numol CO2\nm$^{-2}$ s$^{-1}$")
    conductance_axis.set_ylabel("gsw")
    conductance_axis.spines[["top"]].set_visible(False)
    handles_1, labels_1 = axes[0].get_legend_handles_labels()
    handles_2, labels_2 = conductance_axis.get_legend_handles_labels()
    axes[0].legend(
        handles_1 + handles_2,
        labels_1 + labels_2,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.02),
        fontsize=9,
        borderaxespad=0,
    )

    axes[1].plot(
        frame["minute"],
        frame["CO2_r_sp"],
        color=LINE_COLORS["CO2_setpoint"],
        lw=2.0,
        drawstyle="steps-post",
        label="CO2 setpoint",
    )
    axes[1].plot(frame["minute"], frame["CO2_r"], color=LINE_COLORS["CO2_reference"], lw=1.7, label="CO2_R")
    axes[1].plot(frame["minute"], frame["CO2_s"], color=LINE_COLORS["CO2_sample"], lw=1.7, label="CO2_S")
    axes[1].set_ylabel("CO2\numol mol$^{-1}$")
    line_legend(axes[1])

    axes[2].plot(
        frame["minute"],
        frame["H2O_r_sp"],
        color=LINE_COLORS["H2O_setpoint"],
        lw=2.0,
        drawstyle="steps-post",
        label="H2O setpoint",
    )
    axes[2].plot(frame["minute"], frame["H2O_r"], color=LINE_COLORS["H2O_reference"], lw=1.7, label="H2O_R")
    axes[2].plot(frame["minute"], frame["H2O_s"], color=LINE_COLORS["H2O_sample"], lw=1.7, label="H2O_S")
    axes[2].set_ylabel("H2O\nmmol mol$^{-1}$")
    line_legend(axes[2])

    delta_co2 = frame["CO2_s"] - frame["CO2_r"]
    delta_h2o = frame["H2O_s"] - frame["H2O_r"]
    axes[3].plot(frame["minute"], delta_co2, color=LINE_COLORS["CO2_sample"], lw=1.8, label="CO2_S - CO2_R")
    axes[3].axhline(0, color="#555555", ls=":", lw=1)
    delta_axis = axes[3].twinx()
    delta_axis.plot(frame["minute"], delta_h2o, color=LINE_COLORS["H2O_sample"], lw=1.8, label="H2O_S - H2O_R")
    delta_axis.axhline(0, color="#777777", ls=":", lw=1)
    axes[3].set_ylabel("Delta CO2\numol mol$^{-1}$")
    delta_axis.set_ylabel("Delta H2O\nmmol mol$^{-1}$")
    delta_axis.spines[["top"]].set_visible(False)
    handles_1, labels_1 = axes[3].get_legend_handles_labels()
    handles_2, labels_2 = delta_axis.get_legend_handles_labels()
    axes[3].legend(
        handles_1 + handles_2,
        labels_1 + labels_2,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.02),
        fontsize=9,
        borderaxespad=0,
    )
    axes[3].set_xlabel("Elapsed time (min)")
    axes[3].set_xlim(0, x_max)

    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
        ax.set_xlabel("")

    fig.subplots_adjust(top=0.92, left=0.105, right=0.82, bottom=0.07)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_diagnostics(frame: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="ticks", context="notebook", font_scale=1.0)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6), constrained_layout=True)
    fig.suptitle("Morella Control Diagnostics", fontsize=17, fontweight="bold")

    palette = {125: "#6b3d91", 275: "#1f9e89", 420: "#f2c300"}
    setpoints = frame["CO2_r_sp"].round().astype(int)
    plot_frame = frame.assign(co2_setpoint=setpoints)

    sns.scatterplot(
        data=plot_frame,
        x="gsw",
        y="A",
        hue="co2_setpoint",
        palette=palette,
        ax=axes[0, 0],
        s=48,
        edgecolor="white",
        linewidth=0.4,
    )
    axes[0, 0].set_title("Assimilation vs Conductance")
    axes[0, 0].set_xlabel("gsw (mol H2O m$^{-2}$ s$^{-1}$)")
    axes[0, 0].set_ylabel("A (umol CO2 m$^{-2}$ s$^{-1}$)")
    axes[0, 0].legend(title="CO2 ppm", frameon=False, fontsize=8, title_fontsize=9)

    ci_ok = plot_frame[(plot_frame["Ci"] >= 0) & (plot_frame["Ci"] <= 500)]
    sns.scatterplot(
        data=ci_ok,
        x="Ci",
        y="A",
        hue="co2_setpoint",
        palette=palette,
        ax=axes[0, 1],
        s=48,
        edgecolor="white",
        linewidth=0.4,
        legend=False,
    )
    axes[0, 1].set_title("A vs Ci")
    axes[0, 1].set_xlabel("Ci (umol mol$^{-1}$)")
    axes[0, 1].set_ylabel("A (umol CO2 m$^{-2}$ s$^{-1}$)")

    axes[1, 0].plot(frame["minute"], frame["E"] * 1000, color=LINE_COLORS["CO2_reference"], lw=1.8, label="E x 1000")
    axes[1, 0].plot(frame["minute"], frame["gsw"], color=LINE_COLORS["gsw"], lw=1.8, label="gsw")
    axes[1, 0].set_title("Transpiration and Stomatal Opening")
    axes[1, 0].set_xlabel("Elapsed time (min)")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(frame["minute"], frame["RHcham"], color=LINE_COLORS["Ci"], lw=1.7, label="RH chamber")
    vpd_axis = axes[1, 1].twinx()
    vpd_axis.plot(frame["minute"], frame["VPDleaf"], color=LINE_COLORS["VPD"], lw=1.7, label="VPDleaf")
    axes[1, 1].set_title("Humidity and VPD Stability")
    axes[1, 1].set_xlabel("Elapsed time (min)")
    axes[1, 1].set_ylabel("RHcham (%)")
    vpd_axis.set_ylabel("VPDleaf (kPa)")
    handles_1, labels_1 = axes[1, 1].get_legend_handles_labels()
    handles_2, labels_2 = vpd_axis.get_legend_handles_labels()
    axes[1, 1].legend(handles_1 + handles_2, labels_1 + labels_2, frameon=False, loc="upper left")

    for ax in axes.flat:
        ax.grid(True, color="#d7d7d2", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    vpd_axis.spines[["top"]].set_visible(False)

    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def build_control_plots() -> tuple[Path, Path, Path, Path]:
    OUTPUT_DIR.mkdir(exist_ok=True)
    frame = load_control()
    stages = co2_stages(frame)

    summary_path = OUTPUT_DIR / "morella_sun_control_stage_summary.csv"
    stage_summary(frame).to_csv(summary_path, index=False)

    timeseries_path = OUTPUT_DIR / "morella_sun_control_timeseries.png"
    gas_streams_path = OUTPUT_DIR / "morella_sun_control_gas_streams.png"
    diagnostics_path = OUTPUT_DIR / "morella_sun_control_diagnostics.png"
    plot_timeseries(frame, stages, timeseries_path)
    plot_gas_streams(frame, stages, gas_streams_path)
    plot_diagnostics(frame, diagnostics_path)
    return timeseries_path, gas_streams_path, diagnostics_path, summary_path


def main() -> None:
    for output in build_control_plots():
        print(f"Wrote {output}")
