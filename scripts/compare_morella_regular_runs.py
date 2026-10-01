from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_OUT = OUTPUT_DIR / "morella_regular_runs_comparison.png"
SUMMARY_OUT = OUTPUT_DIR / "morella_regular_runs_comparison_summary.csv"

RUNS = {
    "July 9": {
        "path": OUTPUT_DIR / "morella_aqc_surface_points.csv",
        "color": "#666666",
    },
    "July 10 outside": {
        "path": OUTPUT_DIR / "morella_outside_regular_points.csv",
        "color": "#168f75",
    },
}
SELECTED_Q = [800, 1300, 2000]


def load_runs() -> dict[str, pd.DataFrame]:
    return {
        label: pd.read_csv(config["path"], low_memory=False)
        for label, config in RUNS.items()
    }


def binned_curve(points: pd.DataFrame, q_level: int, bin_width: int = 60) -> pd.DataFrame:
    data = points.loc[
        points["sane_for_curve"].astype(bool)
        & points["Q_level"].eq(q_level)
        & points["Ci"].between(0, 1900)
    ].copy()
    bins = np.arange(0, 1920 + bin_width, bin_width)
    data["ci_bin"] = pd.cut(data["Ci"], bins=bins, include_lowest=True)
    curve = (
        data.groupby("ci_bin", observed=True)
        .agg(
            n=("A", "size"),
            Ci=("Ci", "median"),
            A=("A", "median"),
            A_q25=("A", lambda values: values.quantile(0.25)),
            A_q75=("A", lambda values: values.quantile(0.75)),
        )
        .reset_index(drop=True)
    )
    return curve.loc[curve["n"].ge(3)].sort_values("Ci")


def comparison_summary(runs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for label, points in runs.items():
        sane = points.loc[points["sane_for_curve"].astype(bool)].copy()
        for q_level, group in sane.groupby("Q_level", sort=True):
            if q_level == 0:
                continue
            curve = binned_curve(points, int(q_level))
            peak = curve.loc[curve["A"].idxmax()]
            high_ci = curve.loc[curve["Ci"].ge(1600), "A"]
            rows.append(
                {
                    "run": label,
                    "Q_level": float(q_level),
                    "A_p95": float(group["A"].quantile(0.95)),
                    "A_peak_binned": float(peak["A"]),
                    "Ci_at_A_peak": float(peak["Ci"]),
                    "A_high_Ci": float(high_ci.median()) if not high_ci.empty else np.nan,
                    "high_Ci_decline": (
                        float(peak["A"] - high_ci.median()) if not high_ci.empty else np.nan
                    ),
                    "gsw_median": float(group["gsw"].median()),
                    "Ci_max": float(group["Ci"].max()),
                    "VPDleaf_median": float(group["VPDleaf"].median()),
                    "Tleaf_median": float(group["Tleaf"].median()),
                }
            )
    return pd.DataFrame(rows)


def style_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#d8ddd4", linewidth=0.8)


def make_figure(runs: dict[str, pd.DataFrame], summary: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.72)
    fig, axes = plt.subplots(2, 2, figsize=(14.8, 10.2))
    curve_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]

    for ax, q_level in zip(curve_axes, SELECTED_Q):
        for label, points in runs.items():
            color = RUNS[label]["color"]
            curve = binned_curve(points, q_level)
            ax.fill_between(
                curve["Ci"],
                curve["A_q25"],
                curve["A_q75"],
                color=color,
                alpha=0.12,
                linewidth=0,
            )
            ax.plot(curve["Ci"], curve["A"], color=color, lw=2.7, label=label)
        ax.axhline(0, color="#555555", lw=0.9, ls=":")
        ax.set_title(f"Q = {q_level}", loc="left", fontweight="bold")
        ax.set_xlim(0, 1900)
        ax.set_ylim(0, 21)
        ax.set_xlabel("Ci (umol mol-1)")
        ax.set_ylabel("A / DAT Adyn")
        style_axes(ax)
    curve_axes[0].legend(frameon=False, loc="lower right", fontsize=9)

    ax_summary = axes[1, 1]
    for label, group in summary.groupby("run", sort=False):
        color = RUNS[label]["color"]
        group = group.sort_values("Q_level")
        ax_summary.plot(
            group["Q_level"],
            group["A_peak_binned"],
            color=color,
            marker="o",
            lw=2.7,
            ms=6,
            label=label,
        )
        for row in group.loc[group["Q_level"].isin([800, 1300, 2000])].itertuples():
            ax_summary.annotate(
                f"{row.A_peak_binned:.1f}",
                (row.Q_level, row.A_peak_binned),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                color=color,
                fontsize=8,
            )
    ax_summary.set_title("Observed A Peak Across Q", loc="left", fontweight="bold")
    ax_summary.set_xlabel("Q (umol photons m-2 s-1)")
    ax_summary.set_ylabel("Peak 60-ppm-bin median A")
    ax_summary.set_ylim(0, 21)
    ax_summary.legend(frameon=False, loc="lower right", fontsize=9)
    style_axes(ax_summary)

    fig.text(
        0.06,
        0.982,
        "Morella Regular A/Q/C Runs: July 9 vs July 10 Outside",
        ha="left",
        va="top",
        fontsize=21,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.948,
        "Same regular ascending-ramp program; separate leaves and environments. Shading is the within-bin IQR.",
        ha="left",
        va="top",
        fontsize=10.5,
        color="#555555",
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.08,
        top=0.88,
        wspace=0.22,
        hspace=0.34,
    )
    fig.savefig(FIGURE_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    runs = load_runs()
    summary = comparison_summary(runs)
    summary.to_csv(SUMMARY_OUT, index=False)
    make_figure(runs, summary)
    print(f"Wrote {FIGURE_OUT}")
    print(f"Wrote {SUMMARY_OUT}")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
