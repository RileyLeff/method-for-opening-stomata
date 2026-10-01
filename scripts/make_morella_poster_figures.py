from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.interpolate import griddata

from photo26.control_plots import co2_stages, load_control
from photo26.curves import OUTPUT_DIR, read_licor_export


PROJECT_ROOT = Path(__file__).resolve().parents[1]

AQ_PATH = PROJECT_ROOT / "data" / "2026-06-15-1020_logdata_morcer_test1"
ACI_PATH = PROJECT_ROOT / "data" / "2026-06-15-1053_logdata_morcertest2_Acicurve (1)"
CONTROL_PATH = PROJECT_ROOT / "data" / "2026-06-21-0833_logdata_morcer_sun_control"
BLUE_PATH = PROJECT_ROOT / "data" / "2026-06-22-1405_logdata_morcer_afternoon_bluelight_indoors"
SURFACE_POINTS_PATH = OUTPUT_DIR / "morella_outside_regular_points.csv"
PHRAG_SURFACE_POINTS_PATH = OUTPUT_DIR / "phrag_brownsville_salinity_edge_points.csv"
PHRAG_BUCKET_POINTS_PATH = OUTPUT_DIR / "phrag_bucket_highq_points.csv"
PHRAG_BUCKET_SOURCE_PATH = (
    PROJECT_ROOT / "data" / "2026-07-08_phrag_bucket_highq" / "2026-07-08-1123_logdata_phrag_aq_dark"
)
POSTER_DATA_DIR = OUTPUT_DIR / "poster_figure_data"
POSTER_DATA_ZIP = OUTPUT_DIR / "poster_figure_data.zip"

OUT_STEM_PROBLEM = OUTPUT_DIR / "poster_morella_01_initial_low_a"
OUT_STEM_CO2 = OUTPUT_DIR / "poster_morella_02_low_co2_opening"
OUT_STEM_BLUE = OUTPUT_DIR / "poster_morella_03_blue_and_co2"
OUT_STEM_CO2_BLUE_COMPARISON = OUTPUT_DIR / "poster_morella_02_03_co2_blue_comparison"
OUT_STEM_CURVES = OUTPUT_DIR / "poster_morella_04_multi_q_aci"
OUT_STEM_SURFACE = OUTPUT_DIR / "poster_morella_05_aqc_surface"
OUT_STEM_PAIR = OUTPUT_DIR / "poster_morella_04_05_capacity_pair"
OUT_STEM_PHRAG_PAIR = OUTPUT_DIR / "poster_phrag_brownsville_salinity_edge_capacity_pair"
OUT_STEM_PHRAG_GSW_SURFACE = OUTPUT_DIR / "poster_phrag_bucket_highq_aci_gsw_surface"

GREEN = "#1b8a5a"
BLUE = "#2475a8"
SPECTRUM_RED = "#c94f45"
SPECTRUM_BLUE = "#3c78c2"
PURPLE = "#7656a8"
ORANGE = "#d97732"
DARK = "#2c3333"
MUTED = "#5d6663"
GRID = "#d9dedb"
STAGE_COLORS = ["#edf5ee", "#edf2f8", "#fff5d8", "#edf5ee"]
BLUE_STAGE_COLORS = ["#f1f3f2", "#e7f1fb", "#edf2f8", "#fff5d8", "#edf5ee"]

Q_LEVELS = [200, 500, 800, 1100, 1300, 1700, 2000]
Q_NORM = Normalize(vmin=min(Q_LEVELS), vmax=max(Q_LEVELS))
Q_CMAP = plt.get_cmap("viridis")


def set_poster_style() -> None:
    sns.set_theme(style="ticks", context="talk", font_scale=0.92)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.labelcolor": DARK,
            "axes.edgecolor": DARK,
            "axes.linewidth": 1.1,
            "xtick.color": DARK,
            "ytick.color": DARK,
            "text.color": DARK,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def style_axis(ax: plt.Axes, *, grid: bool = True) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    else:
        ax.grid(False)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def stage_windows(
    frame: pd.DataFrame,
    starts: list[tuple[float, str]],
) -> list[dict[str, float | str | pd.DataFrame]]:
    windows: list[dict[str, float | str | pd.DataFrame]] = []
    final_minute = float(frame["minute"].max())
    for index, (start, label) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else final_minute
        if index + 1 < len(starts):
            part = frame.loc[frame["minute"].between(start, end, inclusive="left")]
        else:
            part = frame.loc[frame["minute"].between(start, end, inclusive="both")]
        windows.append({"start": start, "end": end, "label": label, "data": part})
    return windows


def add_treatment_strip(
    ax: plt.Axes,
    windows: list[dict[str, float | str | pd.DataFrame]],
    colors: list[str],
) -> None:
    ax.set_ylim(0, 1)
    ax.axis("off")
    for index, window in enumerate(windows):
        start = float(window["start"])
        end = float(window["end"])
        ax.axvspan(start, end, ymin=0.08, ymax=0.92, color=colors[index], ec="#c9cfcc", lw=0.8)
        ax.text(
            (start + end) / 2,
            0.5,
            str(window["label"]),
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
        )


def shade_windows(
    axes: list[plt.Axes],
    windows: list[dict[str, float | str | pd.DataFrame]],
    colors: list[str],
) -> None:
    for index, window in enumerate(windows):
        start = float(window["start"])
        end = float(window["end"])
        for ax in axes:
            ax.axvspan(start, end, color=colors[index], alpha=0.48, lw=0, zorder=0)
            if index:
                ax.axvline(start, color="#8b9490", lw=1.0, ls="--", zorder=1)


def annotate_stage_endpoints(
    ax: plt.Axes,
    windows: list[dict[str, float | str | pd.DataFrame]],
    column: str,
    color: str,
    offsets: list[tuple[float, float]],
    precision: int = 3,
) -> None:
    for window, offset in zip(windows, offsets):
        part = window["data"]
        if not isinstance(part, pd.DataFrame) or part.empty:
            continue
        x_value = float(part["minute"].tail(5).mean())
        y_value = float(part[column].tail(5).mean())
        ax.scatter([x_value], [y_value], s=52, color=color, edgecolor="white", linewidth=0.9, zorder=5)
        ax.annotate(
            f"{y_value:.{precision}f}",
            xy=(x_value, y_value),
            xytext=offset,
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color=DARK,
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.88},
            zorder=6,
        )


def plot_initial_problem() -> None:
    aq = read_licor_export(AQ_PATH).sort_values("Q").copy()
    aci = read_licor_export(ACI_PATH).sort_values("Ci").copy()

    fig = plt.figure(figsize=(14.2, 8.1))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[2.4, 1],
        left=0.075,
        right=0.98,
        bottom=0.10,
        top=0.87,
        wspace=0.20,
        hspace=0.08,
    )
    ax_aq = fig.add_subplot(grid[0, 0])
    ax_aq_gsw = fig.add_subplot(grid[1, 0], sharex=ax_aq)
    ax_aci = fig.add_subplot(grid[0, 1])
    ax_aci_gsw = fig.add_subplot(grid[1, 1], sharex=ax_aci)

    ax_aq.plot(aq["Q"], aq["A"], color=GREEN, lw=2.8, marker="o", ms=7)
    ax_aci.plot(aci["Ci"], aci["A"], color=GREEN, lw=2.8, marker="o", ms=7)
    ax_aq_gsw.plot(aq["Q"], aq["gsw"], color=BLUE, lw=2.4, marker="o", ms=6)
    ax_aci_gsw.plot(aci["Ci"], aci["gsw"], color=BLUE, lw=2.4, marker="o", ms=6)

    for ax in [ax_aq, ax_aci]:
        ax.axhline(0, color=MUTED, lw=1.0, ls=":")
        ax.axhline(20, color=SPECTRUM_RED, lw=2.2, ls="--")
        ax.text(
            0.98,
            20.35,
            r"approx. expected $A \approx 20$",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            color=SPECTRUM_RED,
            fontsize=9.5,
            fontweight="bold",
        )
        ax.set_ylim(-1.3, 22.2)
        ax.set_ylabel(r"$A$ ($\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$)")
        ax.tick_params(labelbottom=False)
        style_axis(ax)
    for ax in [ax_aq_gsw, ax_aci_gsw]:
        ax.axhline(0.15, color=SPECTRUM_RED, lw=2.2, ls="--")
        ax.text(
            0.98,
            0.153,
            r"approx. expected $g_{sw} \approx 0.15$",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            color=SPECTRUM_RED,
            fontsize=9.5,
            fontweight="bold",
        )
        ax.set_ylim(0, 0.175)
        ax.set_ylabel(r"$g_{sw}$" + "\n" + r"(mol H$_2$O m$^{-2}$ s$^{-1}$)")
        style_axis(ax)

    ax_aq.set_title("A. Initial A/Q curve", loc="left")
    ax_aci.set_title("B. Initial A/Ci curve", loc="left")
    ax_aq_gsw.set_xlabel(r"Incident light, $Q$ ($\mu$mol photons m$^{-2}$ s$^{-1}$)")
    ax_aci_gsw.set_xlabel(r"Intercellular CO$_2$, $C_i$ ($\mu$mol mol$^{-1}$)")

    ax_aq.text(
        0.97,
        0.10,
        "max A = 5.5\n$g_{sw}$ = 0.028-0.042",
        transform=ax_aq.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#b8c0bc", "alpha": 0.94},
    )
    ax_aci.text(
        0.97,
        0.10,
        "max A = 7.0\n$g_{sw}$ = 0.042-0.082",
        transform=ax_aci.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#b8c0bc", "alpha": 0.94},
    )

    fig.text(
        0.075,
        0.955,
        "Initial Morella curves showed low assimilation and nearly closed stomata",
        ha="left",
        va="top",
        fontsize=22,
        fontweight="bold",
    )
    save_figure(fig, OUT_STEM_PROBLEM)


def plot_low_co2_opening() -> None:
    frame = load_control(CONTROL_PATH)
    stage_frame = co2_stages(frame)
    starts = [
        (float(row.start), f"{int(row.co2)} ppm CO$_2$")
        for row in stage_frame.itertuples(index=False)
    ]
    windows = stage_windows(frame, starts)

    fig = plt.figure(figsize=(13.5, 8.9))
    grid = fig.add_gridspec(
        3,
        1,
        height_ratios=[0.34, 1.25, 1.05],
        left=0.09,
        right=0.985,
        bottom=0.10,
        top=0.83,
        hspace=0.12,
    )
    strip = fig.add_subplot(grid[0])
    ax_gsw = fig.add_subplot(grid[1], sharex=strip)
    ax_a = fig.add_subplot(grid[2], sharex=strip)
    axes = [ax_gsw, ax_a]

    add_treatment_strip(strip, windows, STAGE_COLORS)
    shade_windows(axes, windows, STAGE_COLORS)

    ax_gsw.plot(frame["minute"], frame["gsw"], color=BLUE, lw=2.5, marker="o", ms=3.6)
    ax_a.plot(frame["minute"], frame["A"], color=GREEN, lw=2.5, marker="o", ms=3.6)
    annotate_stage_endpoints(
        ax_gsw,
        windows,
        "gsw",
        BLUE,
        [(0, 8), (0, 8), (0, 8), (0, 8)],
    )

    ax_gsw.set_ylabel(r"$g_{sw}$ (mol H$_2$O m$^{-2}$ s$^{-1}$)")
    ax_gsw.set_ylim(0, 0.225)
    ax_gsw.tick_params(labelbottom=False)
    ax_a.set_ylabel(r"$A$ ($\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$)")
    ax_a.set_xlabel("Elapsed time (min)")
    ax_a.axhline(0, color=MUTED, lw=1.0, ls=":")
    ax_a.set_ylim(-0.8, 16.2)
    for ax in axes:
        style_axis(ax)

    ax_gsw.annotate(
        "59% increase",
        xy=(float(windows[2]["end"]) - 2, 0.173),
        xytext=(-72, 34),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "lw": 1.3, "color": DARK},
        fontsize=11,
        fontweight="bold",
    )
    ax_a.annotate(
        "A recovered at ambient CO$_2$",
        xy=(float(windows[3]["start"]) + 4, 12.1),
        xytext=(36, 28),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "lw": 1.2, "color": DARK},
        fontsize=10.5,
    )

    fig.text(
        0.09,
        0.955,
        "Low CO$_2$ produced strong, reversible stomatal opening",
        ha="left",
        va="top",
        fontsize=22,
        fontweight="bold",
    )
    fig.text(
        0.09,
        0.905,
        r"Sun leaf; constant $Q$ = 800, spectrum R755/B40; median leaf VPD = 0.99-1.07 kPa across stages",
        ha="left",
        va="top",
        fontsize=12,
        color=MUTED,
    )
    save_figure(fig, OUT_STEM_CO2)


def load_blue_run() -> pd.DataFrame:
    frame = read_licor_export(BLUE_PATH).sort_values("elapsed").reset_index(drop=True).copy()
    frame["minute"] = frame["elapsed"] / 60
    return frame


def plot_blue_and_co2() -> None:
    frame = load_blue_run()
    starts = [
        (0.0, "Control spectrum\n420 ppm"),
        (33.0, "More blue\n420 ppm"),
        (60.0, "More blue\n275 ppm"),
        (90.0, "More blue\n125 ppm"),
        (120.0, "More blue\n420 ppm"),
    ]
    windows = stage_windows(frame, starts)

    fig = plt.figure(figsize=(14.2, 8.9))
    grid = fig.add_gridspec(
        3,
        1,
        height_ratios=[0.42, 1.25, 1.05],
        left=0.09,
        right=0.985,
        bottom=0.10,
        top=0.83,
        hspace=0.12,
    )
    strip = fig.add_subplot(grid[0])
    ax_gsw = fig.add_subplot(grid[1], sharex=strip)
    ax_a = fig.add_subplot(grid[2], sharex=strip)
    axes = [ax_gsw, ax_a]

    add_treatment_strip(strip, windows, BLUE_STAGE_COLORS)
    shade_windows(axes, windows, BLUE_STAGE_COLORS)
    ax_gsw.plot(frame["minute"], frame["gsw"], color=BLUE, lw=2.5, marker="o", ms=3.5)
    ax_a.plot(frame["minute"], frame["A"], color=GREEN, lw=2.5, marker="o", ms=3.5)
    annotate_stage_endpoints(
        ax_gsw,
        windows,
        "gsw",
        BLUE,
        [(0, 8), (0, 8), (0, 8), (0, 8), (0, 8)],
    )

    ax_gsw.set_ylabel(r"$g_{sw}$ (mol H$_2$O m$^{-2}$ s$^{-1}$)")
    ax_gsw.set_ylim(0, 0.235)
    ax_gsw.tick_params(labelbottom=False)
    ax_a.set_ylabel(r"$A$ ($\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$)")
    ax_a.set_xlabel("Elapsed time (min)")
    ax_a.axhline(0, color=MUTED, lw=1.0, ls=":")
    ax_a.set_ylim(-0.5, 10.8)
    for ax in axes:
        style_axis(ax)

    ax_gsw.annotate(
        "Blue-associated\nincrease: 28%",
        xy=(55, 0.132),
        xytext=(-34, 42),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "lw": 1.2, "color": DARK},
        fontsize=10.5,
        ha="center",
        fontweight="bold",
    )
    ax_gsw.annotate(
        "Low CO$_2$:\nfurther opening",
        xy=(114, 0.176),
        xytext=(-4, 34),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "lw": 1.2, "color": DARK},
        fontsize=10.5,
        ha="center",
        fontweight="bold",
    )

    fig.text(
        0.09,
        0.955,
        "Blue enrichment coincided with modest opening; low CO$_2$ drove a larger response",
        ha="left",
        va="top",
        fontsize=22,
        fontweight="bold",
    )
    fig.text(
        0.09,
        0.905,
        r"Cut plant in water; constant total $Q$ = 800 and leaf VPD near 1.0 kPa; treatments were sequential",
        ha="left",
        va="top",
        fontsize=12,
        color=MUTED,
    )
    save_figure(fig, OUT_STEM_BLUE)


def stage_tail_summary(
    windows: list[dict[str, float | str | pd.DataFrame]],
) -> tuple[np.ndarray, list[np.ndarray]]:
    means: list[float] = []
    tails: list[np.ndarray] = []
    for window in windows:
        part = window["data"]
        if not isinstance(part, pd.DataFrame) or part.empty:
            raise ValueError("Stage window contains no observations")
        tail = part["gsw"].tail(5).to_numpy(dtype=float)
        tails.append(tail)
        means.append(float(np.mean(tail)))
    return np.asarray(means), tails


def plot_co2_blue_comparison() -> None:
    control = load_control(CONTROL_PATH)
    control_stage_frame = co2_stages(control)
    control_starts = [
        (float(row.start), f"{int(row.co2)} ppm")
        for row in control_stage_frame.itertuples(index=False)
    ]
    control_windows = stage_windows(control, control_starts)

    blue = load_blue_run()
    blue_windows = stage_windows(
        blue,
        [
            (0.0, "Control spectrum / 420 ppm"),
            (33.0, "More blue / 420 ppm"),
            (60.0, "More blue / 275 ppm"),
            (90.0, "More blue / 125 ppm"),
            (120.0, "More blue / 420 ppm"),
        ],
    )[1:]

    control_means, control_tails = stage_tail_summary(control_windows)
    blue_means, blue_tails = stage_tail_summary(blue_windows)
    x = np.arange(4, dtype=float)

    control_red = float(control["Q_red"].median())
    control_blue = float(control["Q_blue"].median())
    blue_stage = blue.loc[blue["minute"].ge(33)]
    enriched_red = float(blue_stage["Q_red"].median())
    enriched_blue = float(blue_stage["Q_blue"].median())

    fig = plt.figure(figsize=(14.4, 8.8))
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.82, 3.4],
        left=0.12,
        right=0.97,
        bottom=0.12,
        top=0.86,
        hspace=0.25,
    )
    ax_spectrum = fig.add_subplot(grid[0])
    ax = fig.add_subplot(grid[1])

    spectra = [
        ("Control spectrum", control_red, control_blue),
        ("Blue-enriched", enriched_red, enriched_blue),
    ]
    for y_value, (label, red_q, blue_q) in zip([1, 0], spectra):
        ax_spectrum.barh(
            y_value,
            red_q,
            left=0,
            height=0.56,
            color=SPECTRUM_RED,
            edgecolor="white",
            linewidth=0.8,
        )
        ax_spectrum.barh(
            y_value,
            blue_q,
            left=red_q,
            height=0.56,
            color=SPECTRUM_BLUE,
            edgecolor="white",
            linewidth=0.8,
        )
        ax_spectrum.text(
            812,
            y_value,
            f"{red_q:.0f} red + {blue_q:.0f} blue",
            ha="left",
            va="center",
            fontsize=10.5,
            fontweight="bold",
        )
    ax_spectrum.set_xlim(0, 1050)
    ax_spectrum.set_ylim(-0.6, 1.6)
    ax_spectrum.set_yticks([1, 0], [spectra[0][0], spectra[1][0]])
    ax_spectrum.tick_params(axis="y", length=0, pad=12, labelsize=11)
    ax_spectrum.set_xticks([])
    ax_spectrum.spines[:].set_visible(False)
    ax_spectrum.grid(False)
    ax_spectrum.set_title(
        r"Chamber spectra at constant total $Q \approx 800$",
        loc="left",
        fontsize=12.5,
        pad=5,
    )
    ax_spectrum.text(
        565,
        1.52,
        "red photons",
        color=SPECTRUM_RED,
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
    )
    ax_spectrum.text(
        765,
        1.52,
        "blue photons",
        color=SPECTRUM_BLUE,
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
    )

    stage_colors = ["#edf3ee", "#f7f1dc", "#fff0bf", "#edf3ee"]
    for index, color in enumerate(stage_colors):
        ax.axvspan(index - 0.5, index + 0.5, color=color, alpha=0.72, lw=0)

    jitter = np.linspace(-0.06, 0.06, 5)
    for index in range(4):
        ax.scatter(
            x[index] + jitter,
            control_tails[index],
            color=SPECTRUM_RED,
            s=24,
            alpha=0.25,
            linewidth=0,
            zorder=3,
        )
        ax.scatter(
            x[index] + jitter,
            blue_tails[index],
            color=SPECTRUM_BLUE,
            s=24,
            alpha=0.25,
            linewidth=0,
            zorder=3,
        )

    ax.plot(
        x,
        control_means,
        color=SPECTRUM_RED,
        lw=3.0,
        marker="o",
        ms=10,
        mec="white",
        mew=1.3,
        label="Control spectrum",
        zorder=5,
    )
    ax.plot(
        x,
        blue_means,
        color=SPECTRUM_BLUE,
        lw=3.0,
        marker="o",
        ms=10,
        mec="white",
        mew=1.3,
        label="Blue-enriched",
        zorder=5,
    )

    control_offsets = [(-14, -24), (-8, 14), (-18, -24), (-10, 14)]
    blue_offsets = [(12, 12), (10, -24), (18, 12), (10, -24)]
    for index, value in enumerate(control_means):
        ax.annotate(
            f"{value:.3f}",
            (x[index], value),
            xytext=control_offsets[index],
            textcoords="offset points",
            ha="center",
            va="center",
            color=SPECTRUM_RED,
            fontsize=10,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "none", "alpha": 0.88},
            zorder=6,
        )
    for index, value in enumerate(blue_means):
        ax.annotate(
            f"{value:.3f}",
            (x[index], value),
            xytext=blue_offsets[index],
            textcoords="offset points",
            ha="center",
            va="center",
            color=SPECTRUM_BLUE,
            fontsize=10,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "none", "alpha": 0.88},
            zorder=6,
        )

    low_co2_delta = float(abs(control_means[2] - blue_means[2]))
    ax.annotate(
        rf"Nearly identical at 125 ppm: $\Delta g_{{sw}}$ = {low_co2_delta:.3f}",
        xy=(2.0, max(control_means[2], blue_means[2])),
        xytext=(0, 48),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "lw": 1.2, "color": DARK},
        bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "#c6ceca", "alpha": 0.94},
        zorder=7,
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
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(0.055, 0.202)
    ax.set_ylabel(r"End-of-stage $g_{sw}$ (mol H$_2$O m$^{-2}$ s$^{-1}$)")
    ax.set_xlabel("CO$_2$ treatment sequence")
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=10.5)
    style_axis(ax)

    fig.text(
        0.07,
        0.962,
        "Blue enrichment showed no clear additive effect under low CO$_2$",
        ha="left",
        va="top",
        fontsize=22,
        fontweight="bold",
    )
    save_figure(fig, OUT_STEM_CO2_BLUE_COMPARISON)


def load_surface_points() -> pd.DataFrame:
    return load_points_file(SURFACE_POINTS_PATH)


def load_points_file(path: Path) -> pd.DataFrame:
    points = pd.read_csv(path, low_memory=False)
    if points["sane_for_curve"].dtype != bool:
        points["sane_for_curve"] = points["sane_for_curve"].astype(str).str.lower().eq("true")
    return points


def binned_surface_points(points: pd.DataFrame, bin_width: int = 60) -> pd.DataFrame:
    data = points.loc[
        points["sane_for_curve"]
        & points["Q_level"].isin(Q_LEVELS)
        & points["Ci"].between(0, 1900)
    ].copy()
    bins = np.arange(0, 1920 + bin_width, bin_width)
    data["ci_bin"] = pd.cut(data["Ci"], bins=bins, include_lowest=True)
    binned = (
        data.groupby(["Q_level", "ci_bin"], observed=True)
        .agg(
            n=("A", "size"),
            Ci=("Ci", "median"),
            Q=("Q", "median"),
            A=("A", "median"),
            A_q25=("A", lambda values: values.quantile(0.25)),
            A_q75=("A", lambda values: values.quantile(0.75)),
            gsw=("gsw", "median"),
        )
        .reset_index()
    )
    return binned.loc[binned["n"].ge(3)].sort_values(["Q_level", "Ci"]).reset_index(drop=True)


def draw_multi_q_curves(ax: plt.Axes, binned: pd.DataFrame, *, title: str) -> None:
    for q_level, group in binned.groupby("Q_level", sort=True):
        color = Q_CMAP(Q_NORM(float(q_level)))
        ax.plot(group["Ci"], group["A"], color=color, lw=3.0)
        ax.scatter(group["Ci"], group["A"], color=color, s=12, alpha=0.48, linewidth=0)
    ax.axhline(0, color=MUTED, lw=1.0, ls=":")
    ax.set_title(title, loc="left", fontsize=17, pad=12)
    ax.set_xlabel(r"Intercellular CO$_2$, $C_i$ ($\mu$mol mol$^{-1}$)")
    ax.set_ylabel(r"$A$ ($\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$)")
    ax.set_xlim(0, 1900)
    ax.set_ylim(0, 21)
    style_axis(ax)


def add_q_colorbar(fig: plt.Figure, ax: plt.Axes) -> None:
    scalar = ScalarMappable(norm=Q_NORM, cmap=Q_CMAP)
    colorbar = fig.colorbar(scalar, ax=ax, pad=0.025, fraction=0.05)
    colorbar.set_label(r"$Q$ ($\mu$mol photons m$^{-2}$ s$^{-1}$)")
    colorbar.set_ticks(Q_LEVELS)


def q_legend_handles() -> list[plt.Line2D]:
    return [
        plt.Line2D([0], [0], color=Q_CMAP(Q_NORM(q_level)), lw=2.6, label=str(q_level))
        for q_level in Q_LEVELS
    ]


def draw_gsw_curves(ax: plt.Axes, binned: pd.DataFrame) -> None:
    for q_level, group in binned.groupby("Q_level", sort=True):
        color = Q_CMAP(Q_NORM(float(q_level)))
        ax.plot(group["Ci"], group["gsw"], color=color, lw=2.6)
        ax.scatter(group["Ci"], group["gsw"], color=color, s=10, alpha=0.42, linewidth=0)
    ax.set_title(r"B. $g_{sw}$ during A/Ci ramps", loc="left", fontsize=16, pad=9)
    ax.set_xlabel(r"Intercellular CO$_2$, $C_i$ ($\mu$mol mol$^{-1}$)")
    ax.set_ylabel(r"$g_{sw}$ (mol H$_2$O m$^{-2}$ s$^{-1}$)")
    ax.set_xlim(0, 1900)
    ax.set_ylim(0, 0.20)
    style_axis(ax)


def make_surface_grid(binned: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ci_grid = np.linspace(100, min(1850, float(binned["Ci"].max())), 95)
    q_grid = np.linspace(min(Q_LEVELS), max(Q_LEVELS), 80)
    xx, yy = np.meshgrid(ci_grid, q_grid)
    zz = griddata(
        binned[["Ci", "Q"]].to_numpy(),
        binned["A"].to_numpy(),
        (xx, yy),
        method="linear",
    )
    return xx, yy, zz


def draw_surface(
    ax: plt.Axes,
    binned: pd.DataFrame,
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    *,
    title: str,
    compact: bool = False,
) -> None:
    surface = ax.plot_surface(
        xx,
        yy,
        zz,
        cmap="viridis",
        vmin=0,
        vmax=21,
        alpha=0.88,
        linewidth=0,
        antialiased=True,
    )
    ax.scatter(
        binned["Ci"],
        binned["Q"],
        binned["A"],
        c=binned["A"],
        cmap="viridis",
        vmin=0,
        vmax=21,
        s=10,
        edgecolor="none",
        alpha=0.55,
    )
    ax.set_title(title, loc="left", fontsize=17, pad=12)
    if compact:
        ax.set_xlabel(r"$C_i$", labelpad=3)
        ax.set_ylabel(r"$Q$", labelpad=3)
        ax.set_zlabel(r"$A$", labelpad=3)
        ax.tick_params(axis="both", which="major", labelsize=8, pad=1)
        ax.zaxis.set_tick_params(labelsize=8, pad=1)
    else:
        ax.set_xlabel(r"$C_i$ ($\mu$mol mol$^{-1}$)", labelpad=9)
        ax.set_ylabel(r"$Q$ ($\mu$mol photons m$^{-2}$ s$^{-1}$)", labelpad=9)
        ax.set_zlabel(r"$A$ ($\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$)", labelpad=9)
    ax.set_xlim(0, 1900)
    ax.set_ylim(200, 2000)
    ax.set_zlim(0, 21)
    ax.view_init(elev=27, azim=-126)
    ax.grid(True)
    ax.xaxis.pane.set_facecolor((0.97, 0.98, 0.97, 1))
    ax.yaxis.pane.set_facecolor((0.97, 0.98, 0.97, 1))
    ax.zaxis.pane.set_facecolor((0.97, 0.98, 0.97, 1))
    return surface


def plot_capacity_pair(
    binned: pd.DataFrame,
    output_stem: Path,
    *,
    title: str,
) -> None:
    xx, yy, zz = make_surface_grid(binned)
    fig = plt.figure(figsize=(18.2, 7.4))
    pair_grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.05, 1],
        left=0.055,
        right=0.975,
        bottom=0.12,
        top=0.84,
        wspace=0.17,
    )
    ax_curves = fig.add_subplot(pair_grid[0, 0])
    ax_surface = fig.add_subplot(pair_grid[0, 1], projection="3d")
    draw_multi_q_curves(ax_curves, binned, title="A. A/Ci response across Q")
    legend_handles = q_legend_handles()
    ax_curves.legend(
        handles=legend_handles,
        title=r"$Q$",
        frameon=False,
        ncol=2,
        loc="lower right",
        fontsize=8.5,
        title_fontsize=9.5,
        columnspacing=1.0,
        handlelength=1.8,
    )
    surface = draw_surface(
        ax_surface,
        binned,
        xx,
        yy,
        zz,
        title="B. Joint A/Q/C response surface",
        compact=True,
    )
    colorbar = fig.colorbar(surface, ax=ax_surface, pad=0.03, shrink=0.72)
    colorbar.set_label(r"$A$")
    fig.suptitle(
        title,
        x=0.055,
        y=0.975,
        ha="left",
        fontsize=22,
        fontweight="bold",
    )
    save_figure(fig, output_stem)


def plot_capacity_outputs() -> None:
    points = load_surface_points()
    binned = binned_surface_points(points)
    xx, yy, zz = make_surface_grid(binned)

    fig_curves, ax_curves = plt.subplots(figsize=(12.6, 8.0))
    draw_multi_q_curves(
        ax_curves,
        binned,
        title="A. A/Ci curves across seven light levels",
    )
    add_q_colorbar(fig_curves, ax_curves)
    fig_curves.suptitle(
        "A high-conductance leaf reveals a broad photosynthetic response",
        x=0.08,
        y=0.985,
        ha="left",
        fontsize=22,
        fontweight="bold",
    )
    fig_curves.text(
        0.08,
        0.925,
        r"Outside Morella run, July 10; lines are median dynamic A in 60-ppm $C_i$ bins; observed max A = 19.7",
        ha="left",
        fontsize=11.5,
        color=MUTED,
    )
    fig_curves.subplots_adjust(left=0.10, right=0.91, bottom=0.13, top=0.84)
    save_figure(fig_curves, OUT_STEM_CURVES)

    fig_surface = plt.figure(figsize=(11.8, 8.7))
    ax_surface = fig_surface.add_subplot(111, projection="3d")
    surface = draw_surface(
        ax_surface,
        binned,
        xx,
        yy,
        zz,
        title="B. Assimilation across the joint Q/Ci space",
    )
    colorbar = fig_surface.colorbar(surface, ax=ax_surface, pad=0.07, shrink=0.72)
    colorbar.set_label(r"$A$ ($\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$)")
    fig_surface.suptitle(
        "The same measurements form an A/Q/C response surface",
        x=0.06,
        y=0.985,
        ha="left",
        fontsize=21,
        fontweight="bold",
    )
    fig_surface.text(
        0.06,
        0.925,
        "This surface is the measurement framework for estimating biochemical limits and environmental responses.",
        ha="left",
        fontsize=11.5,
        color=MUTED,
    )
    fig_surface.subplots_adjust(left=0.01, right=0.91, bottom=0.02, top=0.86)
    save_figure(fig_surface, OUT_STEM_SURFACE)

    plot_capacity_pair(
        binned,
        OUT_STEM_PAIR,
        title="From stomatal opening to multidimensional photosynthetic phenotyping",
    )


def plot_phrag_capacity_pair() -> None:
    points = load_points_file(PHRAG_SURFACE_POINTS_PATH)
    binned = binned_surface_points(points)
    plot_capacity_pair(
        binned,
        OUT_STEM_PHRAG_PAIR,
        title="A/Q/C phenotyping of Brownsville marsh-edge Phragmites",
    )


def plot_phrag_capacity_gsw_surface() -> None:
    points = load_points_file(PHRAG_BUCKET_POINTS_PATH)
    binned = binned_surface_points(points)
    xx, yy, zz = make_surface_grid(binned)

    fig = plt.figure(figsize=(18.2, 9.6))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.08, 1],
        height_ratios=[1.75, 0.95],
        left=0.06,
        right=0.975,
        bottom=0.10,
        top=0.84,
        wspace=0.20,
        hspace=0.14,
    )
    ax_aci = fig.add_subplot(grid[0, 0])
    ax_gsw = fig.add_subplot(grid[1, 0], sharex=ax_aci)
    ax_surface = fig.add_subplot(grid[:, 1], projection="3d")

    draw_multi_q_curves(ax_aci, binned, title="A. A/Ci response across Q")
    ax_aci.set_xlabel("")
    ax_aci.tick_params(axis="x", labelbottom=False)
    ax_aci.legend(
        handles=q_legend_handles(),
        title=r"Light level, $Q$ ($\mu$mol photons m$^{-2}$ s$^{-1}$)",
        frameon=True,
        facecolor="white",
        edgecolor="#c6ceca",
        framealpha=0.94,
        ncol=4,
        loc="upper left",
        fontsize=10,
        title_fontsize=10.5,
        columnspacing=1.2,
        handlelength=2.0,
        borderpad=0.6,
    )
    draw_gsw_curves(ax_gsw, binned)

    surface = draw_surface(
        ax_surface,
        binned,
        xx,
        yy,
        zz,
        title="C. Joint A/Q/C response surface",
        compact=True,
    )
    colorbar = fig.colorbar(surface, ax=ax_surface, pad=0.03, shrink=0.72)
    colorbar.set_label(r"$A$")

    fig.suptitle(
        "A/Q/C phenotyping with concurrent stomatal conductance in Phragmites",
        x=0.06,
        y=0.97,
        ha="left",
        fontsize=22,
        fontweight="bold",
    )
    save_figure(fig, OUT_STEM_PHRAG_GSW_SURFACE)


def export_initial_figure_data() -> Path:
    rows: list[pd.DataFrame] = []
    for panel, source_path, x_column in [
        ("A/Q", AQ_PATH, "Q"),
        ("A/Ci", ACI_PATH, "Ci"),
    ]:
        data = read_licor_export(source_path).reset_index(drop=True).copy()
        data["measurement_order"] = np.arange(1, len(data) + 1)
        data = data.sort_values(x_column).reset_index(drop=True)
        data["plot_order"] = np.arange(1, len(data) + 1)
        data["panel"] = panel
        data["x_variable"] = x_column
        data["x_value"] = data[x_column]
        data["expected_A"] = 20.0
        data["expected_gsw"] = 0.15
        data["source_log"] = str(source_path.relative_to(PROJECT_ROOT))
        rows.append(data)

    combined = pd.concat(rows, ignore_index=True)
    columns = [
        "panel",
        "source_log",
        "measurement_order",
        "plot_order",
        "x_variable",
        "x_value",
        "Q",
        "Ci",
        "Ca",
        "A",
        "gsw",
        "VPDleaf",
        "Tleaf",
        "expected_A",
        "expected_gsw",
    ]
    path = POSTER_DATA_DIR / "figure1_initial_morella_curves.csv"
    combined[columns].to_csv(path, index=False)
    return path


def export_comparison_figure_data() -> Path:
    control = load_control(CONTROL_PATH)
    control_stage_frame = co2_stages(control)
    control_windows = stage_windows(
        control,
        [
            (float(row.start), f"{int(row.co2)} ppm")
            for row in control_stage_frame.itertuples(index=False)
        ],
    )

    blue = load_blue_run()
    blue_windows = stage_windows(
        blue,
        [
            (0.0, "Control spectrum / 420 ppm"),
            (33.0, "More blue / 420 ppm"),
            (60.0, "More blue / 275 ppm"),
            (90.0, "More blue / 125 ppm"),
            (120.0, "More blue / 420 ppm"),
        ],
    )[1:]

    experiment_specs = [
        {
            "experiment": "CO2 only",
            "spectrum": "Control spectrum",
            "context": "Sun leaf",
            "frame": control,
            "windows": control_windows,
            "source_path": CONTROL_PATH,
        },
        {
            "experiment": "CO2 + blue",
            "spectrum": "Blue-enriched",
            "context": "Cut plant in water",
            "frame": blue.loc[blue["minute"].ge(33)],
            "windows": blue_windows,
            "source_path": BLUE_PATH,
        },
    ]
    co2_sequence = [420, 275, 125, 420]
    stage_names = ["420 before drawdown", "275", "125", "420 recovery"]
    rows: list[dict[str, float | int | str]] = []
    for spec in experiment_specs:
        frame = spec["frame"]
        if not isinstance(frame, pd.DataFrame):
            continue
        q_red = float(frame["Q_red"].median())
        q_blue = float(frame["Q_blue"].median())
        q_total = q_red + q_blue
        windows = spec["windows"]
        if not isinstance(windows, list):
            continue
        for stage_order, (window, co2_ppm, stage_name) in enumerate(
            zip(windows, co2_sequence, stage_names),
            start=1,
        ):
            part = window["data"]
            if not isinstance(part, pd.DataFrame) or part.empty:
                continue
            tail = part.tail(5).copy()
            mean_gsw = float(tail["gsw"].mean())
            for observation_order, row in enumerate(tail.itertuples(index=False), start=1):
                rows.append(
                    {
                        "experiment": str(spec["experiment"]),
                        "spectrum": str(spec["spectrum"]),
                        "leaf_context": str(spec["context"]),
                        "source_log": str(Path(spec["source_path"]).relative_to(PROJECT_ROOT)),
                        "stage_order": stage_order,
                        "stage_name": stage_name,
                        "CO2_ppm": co2_ppm,
                        "terminal_observation": observation_order,
                        "gsw": float(row.gsw),
                        "stage_mean_gsw": mean_gsw,
                        "A": float(row.A),
                        "Ci": float(row.Ci),
                        "Ca": float(row.Ca),
                        "VPDleaf": float(row.VPDleaf),
                        "Tleaf": float(row.Tleaf),
                        "Q_red": q_red,
                        "Q_blue": q_blue,
                        "Q_total_red_plus_blue": q_total,
                        "blue_fraction": q_blue / q_total,
                    }
                )

    path = POSTER_DATA_DIR / "figure2_morella_co2_blue_comparison.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def export_capacity_figure_data() -> tuple[Path, Path]:
    points = load_points_file(PHRAG_BUCKET_POINTS_PATH)
    binned = binned_surface_points(points).copy()
    binned["Ci_bin_label"] = binned["ci_bin"].astype(str)
    binned["Ci_bin_left"] = binned["ci_bin"].map(lambda interval: float(interval.left))
    binned["Ci_bin_right"] = binned["ci_bin"].map(lambda interval: float(interval.right))
    binned["source_log"] = str(PHRAG_BUCKET_SOURCE_PATH.relative_to(PROJECT_ROOT))
    binned_path = POSTER_DATA_DIR / "figure3_phrag_aci_gsw_surface_binned.csv"
    binned[
        [
            "source_log",
            "Q_level",
            "Ci_bin_label",
            "Ci_bin_left",
            "Ci_bin_right",
            "n",
            "Ci",
            "Q",
            "A",
            "A_q25",
            "A_q75",
            "gsw",
        ]
    ].to_csv(binned_path, index=False)

    filtered = points.loc[
        points["sane_for_curve"]
        & points["Q_level"].isin(Q_LEVELS)
        & points["Ci"].between(0, 1900)
    ].copy()
    filtered["source_log"] = str(PHRAG_BUCKET_SOURCE_PATH.relative_to(PROJECT_ROOT))
    filtered_path = POSTER_DATA_DIR / "figure3_phrag_filtered_rows.csv"
    preferred_columns = [
        "source_log",
        "elapsed",
        "minute",
        "phase_id",
        "phase_min",
        "Q_level",
        "Q",
        "Ci",
        "Ca",
        "A",
        "Asty",
        "gsw",
        "VPDleaf",
        "Tleaf",
        "CO2_r_sp",
        "CO2_r",
        "CO2_s",
        "Flow",
        "LightAdaptedID",
        "light_flash_logged",
    ]
    filtered[[column for column in preferred_columns if column in filtered.columns]].to_csv(
        filtered_path,
        index=False,
    )
    return binned_path, filtered_path


def export_poster_figure_data() -> list[Path]:
    POSTER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    figure1_path = export_initial_figure_data()
    figure2_path = export_comparison_figure_data()
    figure3_binned_path, figure3_raw_path = export_capacity_figure_data()

    manifest = pd.DataFrame(
        [
            {
                "figure": 1,
                "csv_file": figure1_path.name,
                "row_definition": "One original logged point from either the initial A/Q or A/Ci curve",
                "plot_use": "A and gsw lines; expected_A and expected_gsw provide the red reference lines",
                "units": "A: umol CO2 m-2 s-1; gsw: mol H2O m-2 s-1; Q: umol photons m-2 s-1; Ci/Ca: umol mol-1",
            },
            {
                "figure": 2,
                "csv_file": figure2_path.name,
                "row_definition": "One of the final five observations from a treatment stage",
                "plot_use": "Small points are gsw; stage_mean_gsw is the connected large point; Q_red and Q_blue build the spectrum bars",
                "units": "gsw: mol H2O m-2 s-1; Q components: umol photons m-2 s-1; CO2/Ci/Ca: umol mol-1",
            },
            {
                "figure": 3,
                "csv_file": figure3_binned_path.name,
                "row_definition": "Median within a 60-ppm Ci bin at one Q level, requiring at least three rows",
                "plot_use": "A and gsw line traces and the interpolated A/Q/C surface",
                "units": "A: umol CO2 m-2 s-1; gsw: mol H2O m-2 s-1; Q: umol photons m-2 s-1; Ci: umol mol-1",
            },
            {
                "figure": 3,
                "csv_file": figure3_raw_path.name,
                "row_definition": "One filtered dynamic-assimilation row before 60-ppm Ci binning",
                "plot_use": "Re-bin, refilter, or rebuild Figure 3 from row-level data",
                "units": "A: umol CO2 m-2 s-1; gsw: mol H2O m-2 s-1; Q: umol photons m-2 s-1; gases: umol mol-1",
            },
        ]
    )
    manifest_path = POSTER_DATA_DIR / "poster_figure_data_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    paths = [figure1_path, figure2_path, figure3_binned_path, figure3_raw_path, manifest_path]

    with ZipFile(POSTER_DATA_ZIP, "w", compression=ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, arcname=path.name)
    return paths


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    set_poster_style()
    plot_initial_problem()
    plot_low_co2_opening()
    plot_blue_and_co2()
    plot_co2_blue_comparison()
    plot_capacity_outputs()
    plot_phrag_capacity_pair()
    plot_phrag_capacity_gsw_surface()
    data_paths = export_poster_figure_data()
    for stem in [
        OUT_STEM_PROBLEM,
        OUT_STEM_CO2,
        OUT_STEM_BLUE,
        OUT_STEM_CO2_BLUE_COMPARISON,
        OUT_STEM_CURVES,
        OUT_STEM_SURFACE,
        OUT_STEM_PAIR,
        OUT_STEM_PHRAG_PAIR,
        OUT_STEM_PHRAG_GSW_SURFACE,
    ]:
        print(stem.with_suffix(".png"))
        print(stem.with_suffix(".pdf"))
    for path in data_paths:
        print(path)
    print(POSTER_DATA_ZIP)


if __name__ == "__main__":
    main()
