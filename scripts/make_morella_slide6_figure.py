from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "poster_figure_data" / "figure2_morella_co2_blue_comparison.csv"
OUT_PNG = ROOT / "outputs" / "morella_low_co2_blue_slide6.png"
OUT_PDF = ROOT / "outputs" / "morella_low_co2_blue_slide6.pdf"

RED = "#CF4F43"
BLUE = "#397BC4"
DARK = "#263238"
MUTED = "#5F6B70"
GRID = "#D9E0DD"
REFERENCE = "#54725A"
REFERENCE_FILL = "#E2ECE2"
LOW_CO2_FILL = "#FFF2C8"


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#748078")
    ax.spines["bottom"].set_color("#748078")
    ax.spines["left"].set_linewidth(1.3)
    ax.spines["bottom"].set_linewidth(1.3)
    ax.grid(axis="y", color=GRID, linewidth=1.0, alpha=0.8)
    ax.grid(axis="x", visible=False)


def main() -> None:
    data = pd.read_csv(DATA)
    control = data.loc[data["experiment"] == "CO2 only"].copy()
    enriched = data.loc[data["experiment"] == "CO2 + blue"].copy()

    stage_order = [1, 2, 3, 4]
    x = np.arange(len(stage_order), dtype=float)
    control_means = (
        control.groupby("stage_order", sort=True)["stage_mean_gsw"].first().reindex(stage_order).to_numpy()
    )
    enriched_means = (
        enriched.groupby("stage_order", sort=True)["stage_mean_gsw"].first().reindex(stage_order).to_numpy()
    )
    control_points = [
        control.loc[control["stage_order"] == stage, "gsw"].to_numpy(dtype=float) for stage in stage_order
    ]
    enriched_points = [
        enriched.loc[enriched["stage_order"] == stage, "gsw"].to_numpy(dtype=float) for stage in stage_order
    ]

    control_red = float(control["Q_red"].median())
    control_blue = float(control["Q_blue"].median())
    enriched_red = float(enriched["Q_red"].median())
    enriched_blue = float(enriched["Q_blue"].median())

    sns.set_theme(style="white", context="talk")
    fig = plt.figure(figsize=(14.0, 8.33), facecolor="white")
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.12, 4.3],
        left=0.14,
        right=0.975,
        bottom=0.125,
        top=0.965,
        hspace=0.24,
    )
    ax_spectrum = fig.add_subplot(grid[0])
    ax = fig.add_subplot(grid[1])

    spectra = [
        ("Control spectrum", control_red, control_blue),
        ("Blue-enriched", enriched_red, enriched_blue),
    ]
    for y_value, (label, red_q, blue_q) in zip([1, 0], spectra):
        ax_spectrum.barh(y_value, red_q, height=0.52, color=RED, edgecolor="white", linewidth=1.2)
        ax_spectrum.barh(
            y_value,
            blue_q,
            left=red_q,
            height=0.52,
            color=BLUE,
            edgecolor="white",
            linewidth=1.2,
        )
        blue_fraction = 100.0 * blue_q / (red_q + blue_q)
        ax_spectrum.text(
            825,
            y_value,
            f"{red_q:.0f} red  +  {blue_q:.0f} blue   ({blue_fraction:.0f}% blue)",
            ha="left",
            va="center",
            fontsize=15,
            fontweight="bold",
            color=DARK,
        )

    ax_spectrum.text(
        0,
        1.66,
        r"Light treatments at constant total $Q \approx 800$",
        ha="left",
        va="bottom",
        fontsize=17,
        fontweight="bold",
        color=DARK,
    )
    ax_spectrum.set_xlim(0, 1120)
    ax_spectrum.set_ylim(-0.65, 1.65)
    ax_spectrum.set_yticks([1, 0], [spectra[0][0], spectra[1][0]])
    ax_spectrum.tick_params(axis="y", length=0, pad=14, labelsize=15)
    ax_spectrum.set_xticks([])
    ax_spectrum.spines[:].set_visible(False)
    ax_spectrum.grid(False)

    ax.axvspan(1.62, 2.38, color=LOW_CO2_FILL, alpha=0.95, lw=0, zorder=0)
    ax.axhspan(0.034, 0.046, color=REFERENCE_FILL, alpha=0.95, lw=0, zorder=0)
    ax.axhline(0.040, color=REFERENCE, lw=2.4, ls=(0, (6, 4)), zorder=1)
    ax.text(
        -0.43,
        0.050,
        r"Initial Morella:  $g_{sw} \approx 0.04$",
        color=REFERENCE,
        ha="left",
        va="bottom",
        fontsize=15,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.88},
        zorder=6,
    )

    jitter = np.linspace(-0.055, 0.055, 5)
    for index in range(4):
        ax.scatter(
            x[index] + jitter,
            control_points[index],
            color=RED,
            s=42,
            alpha=0.28,
            linewidth=0,
            zorder=3,
        )
        ax.scatter(
            x[index] + jitter,
            enriched_points[index],
            color=BLUE,
            s=42,
            alpha=0.28,
            linewidth=0,
            zorder=3,
        )

    ax.plot(
        x,
        control_means,
        color=RED,
        lw=4.0,
        marker="o",
        ms=13,
        mec="white",
        mew=1.8,
        label="Low CO$_2$ only",
        zorder=5,
    )
    ax.plot(
        x,
        enriched_means,
        color=BLUE,
        lw=4.0,
        marker="o",
        ms=13,
        mec="white",
        mew=1.8,
        label="Low CO$_2$ + blue light",
        zorder=5,
    )

    ax.annotate(
        f"{control_means[2]:.3f}",
        (2, control_means[2]),
        xytext=(-34, -25),
        textcoords="offset points",
        ha="center",
        va="center",
        color=RED,
        fontsize=14,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "none", "alpha": 0.9},
        zorder=8,
    )
    ax.annotate(
        f"{enriched_means[2]:.3f}",
        (2, enriched_means[2]),
        xytext=(36, -3),
        textcoords="offset points",
        ha="center",
        va="center",
        color=BLUE,
        fontsize=14,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "none", "alpha": 0.9},
        zorder=8,
    )

    ax.set_xticks(
        x,
        [
            "420 ppm\nbefore drawdown",
            "275 ppm",
            "125 ppm",
            "420 ppm\nrecovery",
        ],
    )
    ax.tick_params(axis="x", labelsize=15, pad=10)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_xlim(-0.48, 3.48)
    ax.set_ylim(0.022, 0.205)
    ax.set_yticks([0.04, 0.08, 0.12, 0.16, 0.20])
    ax.set_ylabel(r"End-of-stage $g_{sw}$" + "\n" + r"(mol H$_2$O m$^{-2}$ s$^{-1}$)", fontsize=17)
    ax.set_xlabel("")
    ax.legend(
        frameon=False,
        ncol=2,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.005),
        borderaxespad=0,
        fontsize=15,
        handlelength=2.4,
        columnspacing=1.8,
    )
    style_axis(ax)

    fig.savefig(OUT_PNG, dpi=240, facecolor="white")
    fig.savefig(OUT_PDF, facecolor="white")
    plt.close(fig)

    print(f"Control means: {control_means}")
    print(f"Blue-enriched means: {enriched_means}")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
