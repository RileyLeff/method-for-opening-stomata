from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
TRACE = OUTPUTS / "phrag_brownsville_tier_c_posterior.nc"
QC = OUTPUTS / "phrag_brownsville_cohort_run_qc.csv"

ENVELOPE_PNG = OUTPUTS / "phrag_brownsville_vcmax_jmax_overlapping_envelopes.png"
ENVELOPE_PDF = OUTPUTS / "phrag_brownsville_vcmax_jmax_overlapping_envelopes.pdf"
LITERATURE_PNG = OUTPUTS / "phrag_brownsville_vcmax_jmax_type_m_comparison.png"
LITERATURE_PDF = OUTPUTS / "phrag_brownsville_vcmax_jmax_type_m_comparison.pdf"
SUMMARY_CSV = OUTPUTS / "phrag_brownsville_vcmax_jmax_figure_summary.csv"

COLORS = {"High": "#2676A8", "Low": "#D36B2A"}
LITERATURE_COLOR = "#4F555A"

PARAMETERS = {
    "Vcmax25": {
        "mu": "mu_log_vc",
        "zone": "beta_vc_high",
        "order": "beta_vc_order",
        "ratio": "Vcmax_high_low_ratio",
        "title": "Rubisco carboxylation capacity",
        "xlabel": r"$V_{cmax,25}$ ($\mu$mol m$^{-2}$ s$^{-1}$)",
        "literature_mean": 84.0,
        "literature_se": 6.0,
        "comparison_xlim": (0.0, 100.0),
    },
    "Jmax25": {
        "mu": "mu_log_j",
        "zone": "beta_j_high",
        "order": "beta_j_order",
        "ratio": "Jmax_high_low_ratio",
        "title": "Electron transport capacity",
        "xlabel": r"$J_{max,25}$ ($\mu$mol e$^{-}$ m$^{-2}$ s$^{-1}$)",
        "literature_mean": 116.0,
        "literature_se": 10.0,
        "comparison_xlim": (55.0, 175.0),
    },
}


def posterior_samples(tree: xr.DataTree, variable: str) -> np.ndarray:
    return tree.posterior[variable].stack(sample=("chain", "draw")).values.astype(float)


def retained_orders() -> np.ndarray:
    qc = pd.read_csv(QC)
    qc["start_datetime"] = pd.to_datetime(qc["start_datetime"])
    qc["date"] = qc["start_datetime"].dt.date
    qc["day_order"] = qc.groupby("date")["start_datetime"].rank(method="first").astype(int)
    keep = qc.loc[(~qc["suspect_run"]) & qc["primary_comparison"]]
    return keep["day_order"].to_numpy(dtype=float)


def standardized_center_draws(
    tree: xr.DataTree,
    spec: dict[str, object],
    zone_high: float,
    orders: np.ndarray,
) -> np.ndarray:
    mu = posterior_samples(tree, str(spec["mu"]))
    beta_zone = posterior_samples(tree, str(spec["zone"]))
    beta_order = posterior_samples(tree, str(spec["order"]))
    order_mean = orders.mean()
    order_sd = orders.std(ddof=0)
    observed_z = (orders - order_mean) / order_sd
    centers_at_orders = np.exp(
        mu[:, None]
        + beta_zone[:, None] * zone_high
        + beta_order[:, None] * observed_z[None, :]
    )
    return centers_at_orders.mean(axis=1)


def interval(values: np.ndarray) -> tuple[float, float, float]:
    low, median, high = np.quantile(values, [0.025, 0.5, 0.975])
    return float(median), float(low), float(high)


def build_draws_and_summary(tree: xr.DataTree) -> tuple[dict[str, dict[str, np.ndarray]], pd.DataFrame]:
    orders = retained_orders()
    draws: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict[str, float | str]] = []

    for parameter, spec in PARAMETERS.items():
        draws[parameter] = {}
        literature_mean = float(spec["literature_mean"])
        literature_se = float(spec["literature_se"])
        for zone, zone_high in [("High", 1.0), ("Low", 0.0)]:
            values = standardized_center_draws(tree, spec, zone_high, orders)
            draws[parameter][zone] = values
            median, low, high = interval(values)
            rows.append(
                {
                    "parameter": parameter,
                    "source": "Brownsville joint Bayesian model",
                    "group": zone,
                    "estimate": median,
                    "lower_95": low,
                    "upper_95": high,
                    "literature_mean": literature_mean,
                    "literature_se": literature_se,
                    "estimate_percent_of_literature": 100.0 * median / literature_mean,
                    "posterior_probability_above_literature_mean": float(np.mean(values > literature_mean)),
                }
            )

        ratio = posterior_samples(tree, str(spec["ratio"]))
        median, low, high = interval(ratio)
        rows.append(
            {
                "parameter": parameter,
                "source": "Brownsville joint Bayesian model",
                "group": "High/Low ratio",
                "estimate": median,
                "lower_95": low,
                "upper_95": high,
                "literature_mean": literature_mean,
                "literature_se": literature_se,
                "estimate_percent_of_literature": np.nan,
                "posterior_probability_above_literature_mean": np.nan,
            }
        )

    return draws, pd.DataFrame(rows)


def density(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return gaussian_kde(values)(grid)


def plot_overlapping_envelopes(
    tree: xr.DataTree,
    draws: dict[str, dict[str, np.ndarray]],
    summary: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.5))

    for ax, (parameter, spec) in zip(axes, PARAMETERS.items()):
        pooled = np.concatenate([draws[parameter]["High"], draws[parameter]["Low"]])
        x_min = max(0.0, float(np.quantile(pooled, 0.001)) * 0.72)
        x_max = float(np.quantile(pooled, 0.999)) * 1.05
        grid = np.linspace(x_min, x_max, 600)
        max_density = 0.0

        for zone in ["High", "Low"]:
            values = draws[parameter][zone]
            y = density(values, grid)
            max_density = max(max_density, float(y.max()))
            color = COLORS[zone]
            ax.fill_between(grid, 0, y, color=color, alpha=0.27, linewidth=0)
            ax.plot(grid, y, color=color, lw=2.6, label=zone)

        interval_y = {"High": -0.070 * max_density, "Low": -0.145 * max_density}
        for zone in ["High", "Low"]:
            values = draws[parameter][zone]
            median, low, high = interval(values)
            y = interval_y[zone]
            ax.plot([low, high], [y, y], color=COLORS[zone], lw=5.0, solid_capstyle="round")
            ax.scatter([median], [y], s=54, color=COLORS[zone], edgecolor="white", linewidth=1.1, zorder=4)

        ratio_row = summary.loc[
            (summary["parameter"] == parameter) & (summary["group"] == "High/Low ratio")
        ].iloc[0]
        ax.text(
            0.98,
            0.96,
            f"High / Low = {ratio_row['estimate']:.2f}\n"
            f"95% interval {ratio_row['lower_95']:.2f}-{ratio_row['upper_95']:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=11.5,
            color="#394046",
        )
        ax.axhline(0, color="#8B9297", lw=0.8)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.20 * max_density, 1.10 * max_density)
        ax.set_title(str(spec["title"]), loc="left", fontweight="bold", pad=12)
        ax.set_xlabel(str(spec["xlabel"]))
        ax.set_ylabel("Posterior density" if parameter == "Vcmax25" else "")
        ax.set_yticks([])
        ax.grid(axis="x", alpha=0.22)
        ax.grid(axis="y", visible=False)
        sns.despine(ax=ax, left=True)

    handles = [
        Line2D([0], [0], color=COLORS["High"], lw=5, label="High"),
        Line2D([0], [0], color=COLORS["Low"], lw=5, label="Low"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.44, -0.015), ncol=2, frameon=False)
    fig.text(
        0.67,
        0.012,
        "Point + line: median and 95% interval",
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#596168",
    )
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(ENVELOPE_PNG, dpi=260, bbox_inches="tight")
    fig.savefig(ENVELOPE_PDF, bbox_inches="tight")
    plt.close(fig)


def plot_literature_comparison(
    draws: dict[str, dict[str, np.ndarray]],
    summary: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.8), sharey=True)
    y_positions = {"Type M": 2.0, "High": 1.0, "Low": 0.0}

    for ax, (parameter, spec) in zip(axes, PARAMETERS.items()):
        literature_mean = float(spec["literature_mean"])
        literature_se = float(spec["literature_se"])
        x_limits = tuple(spec["comparison_xlim"])
        x_span = float(x_limits[1]) - float(x_limits[0])

        ax.plot(
            [literature_mean - literature_se, literature_mean + literature_se],
            [y_positions["Type M"], y_positions["Type M"]],
            color=LITERATURE_COLOR,
            lw=5.0,
            solid_capstyle="round",
        )
        ax.scatter(
            [literature_mean],
            [y_positions["Type M"]],
            s=105,
            marker="s",
            color=LITERATURE_COLOR,
            edgecolor="white",
            linewidth=1.5,
            zorder=4,
        )
        ax.text(
            literature_mean + literature_se + 0.018 * x_span,
            y_positions["Type M"],
            rf"{literature_mean:.0f} $\pm$ {literature_se:.0f}",
            ha="left",
            va="center",
            fontsize=11.5,
            color="#30363A",
        )

        for zone in ["High", "Low"]:
            values = draws[parameter][zone]
            median, low, high = interval(values)
            y = y_positions[zone]
            ax.plot([low, high], [y, y], color=COLORS[zone], lw=5.0, solid_capstyle="round")
            ax.scatter(
                [median],
                [y],
                s=105,
                color=COLORS[zone],
                edgecolor="white",
                linewidth=1.5,
                zorder=4,
            )
            ax.text(
                high + 0.018 * x_span,
                y,
                f"{median:.0f} [{low:.0f}-{high:.0f}]",
                ha="left",
                va="center",
                fontsize=11.5,
                color="#30363A",
            )

        if parameter == "Vcmax25":
            takeaway = "Brownsville below published Type M"
        else:
            takeaway = "Brownsville overlaps published Type M"
        ax.text(
            0.0,
            1.015,
            takeaway,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10.5,
            color="#596168",
        )
        ax.set_xlim(*x_limits)
        ax.set_ylim(-0.45, 2.45)
        ax.set_title(str(spec["title"]), loc="left", fontweight="bold", pad=28)
        ax.set_xlabel(str(spec["xlabel"]))
        ax.set_yticks(
            [0, 1, 2],
            labels=["Brownsville Low", "Brownsville High", "Published Type M"],
        )
        ax.grid(axis="x", alpha=0.22)
        ax.grid(axis="y", visible=False)
        sns.despine(ax=ax, left=True)

    fig.text(
        0.5,
        0.016,
        r"Brownsville: posterior median [95% interval]   |   Published Type M: mean $\pm$ SE",
        ha="center",
        va="bottom",
        fontsize=10.0,
        color="#596168",
    )
    fig.tight_layout(rect=(0, 0.095, 1, 1))
    fig.savefig(LITERATURE_PNG, dpi=260, bbox_inches="tight")
    fig.savefig(LITERATURE_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    tree = xr.open_datatree(TRACE)
    draws, summary = build_draws_and_summary(tree)
    summary.to_csv(SUMMARY_CSV, index=False)
    plot_overlapping_envelopes(tree, draws, summary)
    plot_literature_comparison(draws, summary)

    print(summary.round(4).to_string(index=False))
    for path in [ENVELOPE_PNG, ENVELOPE_PDF, LITERATURE_PNG, LITERATURE_PDF, SUMMARY_CSV]:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
