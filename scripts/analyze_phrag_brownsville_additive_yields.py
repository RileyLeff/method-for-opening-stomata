from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
FLASH_PATH = OUTPUTS / "phrag_brownsville_cohort_flashes.csv"

FLASH_OUT = OUTPUTS / "phrag_brownsville_additive_yields_flash_level.csv"
COMMON_CI_OUT = OUTPUTS / "phrag_brownsville_additive_yields_common_ci.csv"
SUMMARY_OUT = OUTPUTS / "phrag_brownsville_additive_yields_ci800_summary.csv"
FIGURE_OUT = OUTPUTS / "phrag_brownsville_additive_yields_ci800.png"
PDF_OUT = OUTPUTS / "phrag_brownsville_additive_yields_ci800.pdf"
DIFFERENCE_OUT = OUTPUTS / "phrag_brownsville_additive_yield_differences_ci800.csv"
DIFFERENCE_FIGURE = OUTPUTS / "phrag_brownsville_additive_yield_difference_evidence.png"
DIFFERENCE_PDF = OUTPUTS / "phrag_brownsville_additive_yield_difference_evidence.pdf"
SURFACE_DIFFERENCE_OUT = OUTPUTS / "phrag_brownsville_additive_yield_differences_q_ci.csv"
SURFACE_DIFFERENCE_FIGURE = OUTPUTS / "phrag_brownsville_additive_yield_difference_q_ci_maps.png"
SURFACE_DIFFERENCE_PDF = OUTPUTS / "phrag_brownsville_additive_yield_difference_q_ci_maps.pdf"
RELATIVE_YNO_FIGURE = OUTPUTS / "phrag_brownsville_yno_relative_change_q_ci.png"
RELATIVE_YNO_PDF = OUTPUTS / "phrag_brownsville_yno_relative_change_q_ci.pdf"
LEAF_SUMMARY_OUT = OUTPUTS / "phrag_brownsville_additive_yield_leaf_standardized_summary.csv"
DISTRIBUTION_FIGURE = OUTPUTS / "phrag_brownsville_additive_yield_high_low_distributions.png"
DISTRIBUTION_PDF = OUTPUTS / "phrag_brownsville_additive_yield_high_low_distributions.pdf"
BINNED_SUMMARY_OUT = OUTPUTS / "phrag_brownsville_additive_yield_binned_light_summary.csv"
BINNED_FIGURE = OUTPUTS / "phrag_brownsville_additive_yield_binned_light.png"
BINNED_PDF = OUTPUTS / "phrag_brownsville_additive_yield_binned_light.pdf"
STACKED_FIGURE = OUTPUTS / "phrag_brownsville_additive_yield_stacked_composition.png"
STACKED_PDF = OUTPUTS / "phrag_brownsville_additive_yield_stacked_composition.pdf"

VALID_RUNS = ["P03", "P06", "P10", "P11", "P12", "P13", "P14"]
HIGH_RUNS = ["P12", "P13", "P14"]
Q_LEVELS = np.array([50, 100, 200, 350, 600, 1000, 1400, 1700, 2000], dtype=float)
CI_GRID = np.arange(300, 1201, 50, dtype=float)
LIGHT_BIN_ORDER = ["Low", "Moderate", "High", "Extreme"]
LIGHT_BIN_LABELS = {
    "Low": "Low\n50-200",
    "Moderate": "Moderate\n350-600",
    "High": "High\n1000-1400",
    "Extreme": "Extreme\n1700-2000",
}
Q_TO_LIGHT_BIN = {
    50.0: "Low",
    100.0: "Low",
    200.0: "Low",
    350.0: "Moderate",
    600.0: "Moderate",
    1000.0: "High",
    1400.0: "High",
    1700.0: "Extreme",
    2000.0: "Extreme",
}
LIGHT_BIN_COLORS = {
    "Low": "#5e3c99",
    "Moderate": "#2b8cbe",
    "High": "#2ca25f",
    "Extreme": "#e6bd16",
}
ZONE_LABELS = {
    "High": "High / lower-salinity, shaded",
    "Low": "Low / saline, open-sun",
}
YIELD_LABELS = {
    "YII": "Y(II): photochemistry",
    "YNPQ": "Y(NPQ): regulated dissipation",
    "YNO": "Y(NO): unregulated loss",
}
YIELD_COLORS = {
    "YII": "#2b8c69",
    "YNPQ": "#e48b2a",
    "YNO": "#7c8790",
}
ZONE_COLORS = {"High": "#2478a6", "Low": "#d76528"}


def style() -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.90)
    plt.rcParams.update(
        {
            "axes.titleweight": "bold",
            "axes.titlesize": 16,
            "figure.titlesize": 22,
            "figure.titleweight": "bold",
            "legend.frameon": False,
        }
    )


def calculate_flash_yields() -> pd.DataFrame:
    flashes = pd.read_csv(FLASH_PATH)
    active = flashes.loc[
        flashes["run_code"].isin(VALID_RUNS)
        & flashes["flash_kind"].eq("active")
        & flashes["standard_flash"]
        & ~flashes["overlapping_flash"]
    ].dropna(subset=["PhiPS2", "NPQ", "qL", "FvFm", "gas_Ci", "assigned_Q"])
    active = active.copy()
    active["zone"] = np.where(active["run_code"].isin(HIGH_RUNS), "High", "Low")

    active["Fm_over_Fo"] = 1.0 / (1.0 - active["FvFm"])
    active["YII"] = active["PhiPS2"]
    active["YNO"] = 1.0 / (
        active["NPQ"]
        + 1.0
        + active["qL"] * (active["Fm_over_Fo"] - 1.0)
    )
    active["YNPQ"] = 1.0 - active["YII"] - active["YNO"]
    active["yield_sum"] = active[["YII", "YNPQ", "YNO"]].sum(axis=1)

    yields = active[["YII", "YNPQ", "YNO"]].to_numpy()
    if not np.all((yields >= 0.0) & (yields <= 1.0)):
        raise ValueError("At least one additive yield lies outside [0, 1].")
    if not np.allclose(active["yield_sum"], 1.0, atol=1e-12):
        raise ValueError("Additive yields do not close to one.")
    return active


def interpolate_common_ci(active: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for (run_code, q), group in active.groupby(["run_code", "assigned_Q"]):
        curve = group.sort_values("gas_Ci")
        ci = curve["gas_Ci"].to_numpy()
        zone = curve["zone"].iloc[0]
        for target in CI_GRID:
            if ci.min() <= target <= ci.max():
                row: dict[str, float | str] = {
                    "run_code": run_code,
                    "zone": zone,
                    "Q": float(q),
                    "Ci": float(target),
                }
                for column in ["YII", "YNPQ", "YNO"]:
                    row[column] = float(np.interp(target, ci, curve[column]))
                row["yield_sum"] = float(row["YII"] + row["YNPQ"] + row["YNO"])
                rows.append(row)
    common = pd.DataFrame(rows)
    if not np.allclose(common["yield_sum"], 1.0, atol=1e-12):
        raise ValueError("Interpolated additive yields do not close to one.")
    return common


def summarize_ci800(common: pd.DataFrame) -> pd.DataFrame:
    at_ci = common.loc[common["Ci"].eq(800)]
    rows: list[dict[str, float | int | str]] = []
    for (zone, q), group in at_ci.groupby(["zone", "Q"]):
        for column in ["YII", "YNPQ", "YNO"]:
            values = group[column].to_numpy()
            rows.append(
                {
                    "zone": zone,
                    "Q": float(q),
                    "yield": column,
                    "n_leaves": len(values),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "sd": float(np.std(values, ddof=1)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )
    return pd.DataFrame(rows).sort_values(["Q", "zone", "yield"])


def bootstrap_differences(common: pd.DataFrame, draws: int = 50_000) -> pd.DataFrame:
    at_ci = common.loc[common["Ci"].eq(800)]
    columns = ["YII", "YNPQ", "YNO"]
    rng = np.random.default_rng(20260803)
    rows: list[dict[str, float | str]] = []
    for q, group in at_ci.groupby("Q"):
        high = group.loc[group["zone"].eq("High"), columns].to_numpy()
        low = group.loc[group["zone"].eq("Low"), columns].to_numpy()
        high_indices = rng.integers(0, len(high), size=(draws, len(high)))
        low_indices = rng.integers(0, len(low), size=(draws, len(low)))
        boot = low[low_indices].mean(axis=1) - high[high_indices].mean(axis=1)
        boot_relative = boot / high[high_indices].mean(axis=1) * 100
        observed = low.mean(axis=0) - high.mean(axis=0)
        lower, upper = np.quantile(boot, [0.025, 0.975], axis=0)
        relative = observed / high.mean(axis=0) * 100
        relative_lower, relative_upper = np.quantile(boot_relative, [0.025, 0.975], axis=0)
        for index, column in enumerate(columns):
            rows.append(
                {
                    "Q": float(q),
                    "yield": column,
                    "high_mean": float(high[:, index].mean()),
                    "low_mean": float(low[:, index].mean()),
                    "low_minus_high": float(observed[index]),
                    "bootstrap_2.5": float(lower[index]),
                    "bootstrap_97.5": float(upper[index]),
                    "relative_change_percent": float(relative[index]),
                    "relative_bootstrap_2.5": float(relative_lower[index]),
                    "relative_bootstrap_97.5": float(relative_upper[index]),
                }
            )
    return pd.DataFrame(rows).sort_values(["yield", "Q"])


def bootstrap_surface_differences(
    common: pd.DataFrame, draws: int = 20_000
) -> pd.DataFrame:
    columns = ["YII", "YNPQ", "YNO"]
    rng = np.random.default_rng(20260804)
    rows: list[dict[str, float | int | str]] = []
    for (q, ci), group in common.groupby(["Q", "Ci"]):
        high = group.loc[group["zone"].eq("High"), columns].to_numpy()
        low = group.loc[group["zone"].eq("Low"), columns].to_numpy()
        if len(high) < 2 or len(low) < 2:
            continue
        high_indices = rng.integers(0, len(high), size=(draws, len(high)))
        low_indices = rng.integers(0, len(low), size=(draws, len(low)))
        high_boot = high[high_indices].mean(axis=1)
        low_boot = low[low_indices].mean(axis=1)
        boot = low_boot - high_boot
        boot_relative = boot / high_boot * 100
        observed = low.mean(axis=0) - high.mean(axis=0)
        lower, upper = np.quantile(boot, [0.025, 0.975], axis=0)
        relative = observed / high.mean(axis=0) * 100
        relative_lower, relative_upper = np.quantile(
            boot_relative, [0.025, 0.975], axis=0
        )
        for index, column in enumerate(columns):
            rows.append(
                {
                    "Q": float(q),
                    "Ci": float(ci),
                    "yield": column,
                    "n_high": len(high),
                    "n_low": len(low),
                    "high_mean": float(high[:, index].mean()),
                    "low_mean": float(low[:, index].mean()),
                    "low_minus_high": float(observed[index]),
                    "bootstrap_2.5": float(lower[index]),
                    "bootstrap_97.5": float(upper[index]),
                    "relative_change_percent": float(relative[index]),
                    "relative_bootstrap_2.5": float(relative_lower[index]),
                    "relative_bootstrap_97.5": float(relative_upper[index]),
                    "interval_excludes_zero": bool(
                        lower[index] > 0 or upper[index] < 0
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["yield", "Q", "Ci"])


def leaf_standardized_summary(
    common: pd.DataFrame, draws: int = 100_000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["YII", "YNPQ", "YNO"]
    balanced = common.loc[common["Ci"].ge(350)]
    leaf_means = (
        balanced.groupby(["run_code", "zone"], as_index=False)[columns].mean()
    )
    rng = np.random.default_rng(20260805)
    rows: list[dict[str, float | int | str]] = []
    for column in columns:
        high = leaf_means.loc[leaf_means["zone"].eq("High"), column].to_numpy()
        low = leaf_means.loc[leaf_means["zone"].eq("Low"), column].to_numpy()
        high_boot = high[rng.integers(0, len(high), size=(draws, len(high)))].mean(axis=1)
        low_boot = low[rng.integers(0, len(low), size=(draws, len(low)))].mean(axis=1)
        difference_boot = low_boot - high_boot
        relative_boot = difference_boot / high_boot * 100
        high_mean = high.mean()
        low_mean = low.mean()
        difference = low_mean - high_mean
        rows.append(
            {
                "yield": column,
                "n_high_leaves": len(high),
                "n_low_leaves": len(low),
                "high_mean": float(high_mean),
                "low_mean": float(low_mean),
                "low_minus_high": float(difference),
                "difference_percentage_points": float(difference * 100),
                "relative_change_percent": float(difference / high_mean * 100),
                "relative_bootstrap_2.5": float(np.quantile(relative_boot, 0.025)),
                "relative_bootstrap_97.5": float(np.quantile(relative_boot, 0.975)),
            }
        )
    return leaf_means, pd.DataFrame(rows)


def binned_light_summary(
    common: pd.DataFrame, draws: int = 50_000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["YII", "YNPQ", "YNO"]
    balanced = common.loc[common["Ci"].ge(350)].copy()
    balanced["light_bin"] = balanced["Q"].map(Q_TO_LIGHT_BIN)
    leaf_means = (
        balanced.groupby(["run_code", "zone", "light_bin"], as_index=False)[columns]
        .mean()
    )
    rng = np.random.default_rng(20260807)
    rows: list[dict[str, float | int | str | bool]] = []
    for light_bin in LIGHT_BIN_ORDER:
        for column in columns:
            subset = leaf_means.loc[leaf_means["light_bin"].eq(light_bin)]
            high = subset.loc[subset["zone"].eq("High"), column].to_numpy()
            low = subset.loc[subset["zone"].eq("Low"), column].to_numpy()
            high_boot = high[
                rng.integers(0, len(high), size=(draws, len(high)))
            ].mean(axis=1)
            low_boot = low[
                rng.integers(0, len(low), size=(draws, len(low)))
            ].mean(axis=1)
            difference_boot = low_boot - high_boot
            relative_boot = difference_boot / high_boot * 100
            high_mean = high.mean()
            low_mean = low.mean()
            difference = low_mean - high_mean
            lower = float(np.quantile(relative_boot, 0.025))
            upper = float(np.quantile(relative_boot, 0.975))
            rows.append(
                {
                    "light_bin": light_bin,
                    "yield": column,
                    "n_high_leaves": len(high),
                    "n_low_leaves": len(low),
                    "high_mean": float(high_mean),
                    "low_mean": float(low_mean),
                    "low_minus_high": float(difference),
                    "difference_percentage_points": float(difference * 100),
                    "relative_change_percent": float(difference / high_mean * 100),
                    "relative_bootstrap_2.5": lower,
                    "relative_bootstrap_97.5": upper,
                    "interval_excludes_zero": bool(lower > 0 or upper < 0),
                }
            )
    return leaf_means, pd.DataFrame(rows)


def plot_partition(common: pd.DataFrame) -> None:
    at_ci = common.loc[common["Ci"].eq(800)]
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 6.9), sharex=True, sharey=True)

    for ax, zone in zip(axes, ["High", "Low"]):
        means = (
            at_ci.loc[at_ci["zone"].eq(zone)]
            .groupby("Q", as_index=False)[["YII", "YNPQ", "YNO"]]
            .mean()
            .set_index("Q")
            .reindex(Q_LEVELS)
            .reset_index()
        )
        q = means["Q"].to_numpy()
        bottom = np.zeros(len(means))
        for column in ["YII", "YNPQ", "YNO"]:
            values = means[column].to_numpy()
            ax.fill_between(
                q,
                bottom,
                bottom + values,
                color=YIELD_COLORS[column],
                alpha=0.92,
                label=YIELD_LABELS[column],
            )
            ax.plot(q, bottom + values, color="white", lw=1.3, alpha=0.9)
            bottom += values

        n_leaves = at_ci.loc[at_ci["zone"].eq(zone), "run_code"].nunique()
        ax.set_title(f"{ZONE_LABELS[zone]}  (n = {n_leaves})")
        ax.set_xlim(0, 2050)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 500, 1000, 1500, 2000])
        ax.set_xlabel(r"Q ($\mu$mol photons m$^{-2}$ s$^{-1}$)")
        ax.grid(axis="x", color="#d8d8d8", lw=0.65, alpha=0.65)
        ax.grid(axis="y", color="#d8d8d8", lw=0.65, alpha=0.65)

    axes[0].set_ylabel("Fraction of absorbed PSII excitation")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.005),
        fontsize=11,
    )
    fig.suptitle("Complementary PSII quantum yields at Ci = 800")
    fig.tight_layout(rect=(0, 0.07, 1, 0.93), w_pad=1.25)
    fig.savefig(FIGURE_OUT, dpi=240, bbox_inches="tight")
    fig.savefig(PDF_OUT, bbox_inches="tight")
    plt.close(fig)


def plot_difference_evidence(common: pd.DataFrame, differences: pd.DataFrame) -> None:
    at_ci = common.loc[common["Ci"].eq(800)]
    fig = plt.figure(figsize=(15.5, 10.4))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.18, 1.0], hspace=0.34, wspace=0.30)
    curve_ax = fig.add_subplot(grid[0, :])
    balance_ax = fig.add_subplot(grid[1, 0])
    leaf_ax = fig.add_subplot(grid[1, 1])

    for column in ["YNPQ", "YII", "YNO"]:
        component = differences.loc[differences["yield"].eq(column)].sort_values("Q")
        q = component["Q"].to_numpy()
        estimate = component["low_minus_high"].to_numpy() * 100
        lower = component["bootstrap_2.5"].to_numpy() * 100
        upper = component["bootstrap_97.5"].to_numpy() * 100
        curve_ax.fill_between(q, lower, upper, color=YIELD_COLORS[column], alpha=0.13)
        curve_ax.plot(
            q,
            estimate,
            color=YIELD_COLORS[column],
            marker="o",
            ms=6,
            lw=2.8,
        )
        curve_ax.text(
            2035,
            estimate[-1],
            f"{YIELD_LABELS[column].split(':')[0]}  {estimate[-1]:+.1f} pp",
            color=YIELD_COLORS[column],
            fontsize=11,
            va="center",
        )

    curve_ax.axhline(0, color="#444444", ls="--", lw=1.2)
    curve_ax.set(
        title="Allocation difference across light",
        xlabel=r"Q ($\mu$mol photons m$^{-2}$ s$^{-1}$)",
        ylabel="Low - high yield (percentage points)",
        xlim=(0, 2240),
        ylim=(-12, 16),
    )
    curve_ax.text(
        55,
        14.7,
        "Shading shows 95% leaf-bootstrap intervals",
        fontsize=10.5,
        color="#555555",
        va="top",
    )

    endpoint = differences.loc[differences["Q"].eq(2000)].set_index("yield")
    order = ["YNPQ", "YII", "YNO"]
    y_positions = np.arange(len(order))[::-1]
    estimates = endpoint.loc[order, "low_minus_high"].to_numpy() * 100
    relative_changes = endpoint.loc[order, "relative_change_percent"].to_numpy()
    lower_error = estimates - endpoint.loc[order, "bootstrap_2.5"].to_numpy() * 100
    upper_error = endpoint.loc[order, "bootstrap_97.5"].to_numpy() * 100 - estimates
    balance_ax.barh(
        y_positions,
        estimates,
        color=[YIELD_COLORS[column] for column in order],
        alpha=0.90,
        height=0.58,
    )
    balance_ax.errorbar(
        estimates,
        y_positions,
        xerr=np.vstack([lower_error, upper_error]),
        fmt="none",
        ecolor="#333333",
        elinewidth=1.5,
        capsize=4,
    )
    balance_ax.axvline(0, color="#444444", ls="--", lw=1.2)
    for y, value, relative in zip(y_positions, estimates, relative_changes):
        balance_ax.text(
            value + (0.35 if value >= 0 else -0.35),
            y,
            f"{value:+.1f} pp\n({relative:+.1f}% vs high)",
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
    balance_ax.set(
        title="High-light balance at Q = 2000",
        xlabel="Low - high yield (percentage points)",
        yticks=y_positions,
        yticklabels=[YIELD_LABELS[column].split(":")[0] for column in order],
        xlim=(-10.5, 11.5),
    )

    endpoint_leaves = at_ci.loc[at_ci["Q"].eq(2000)]
    for zone in ["High", "Low"]:
        leaves = endpoint_leaves.loc[endpoint_leaves["zone"].eq(zone)]
        leaf_ax.scatter(
            leaves["YNO"],
            leaves["YNPQ"],
            s=68,
            color=ZONE_COLORS[zone],
            label=ZONE_LABELS[zone],
            zorder=4,
        )
        for row in leaves.itertuples():
            leaf_ax.annotate(
                row.run_code,
                (row.YNO, row.YNPQ),
                xytext=(5, 5),
                textcoords="offset points",
                color=ZONE_COLORS[zone],
                fontsize=9.5,
            )

    x_line = np.linspace(0.18, 0.30, 100)
    for yii in [0.08, 0.12, 0.16]:
        leaf_ax.plot(x_line, 1 - yii - x_line, color="#888888", ls=":", lw=1.0, zorder=1)
        leaf_ax.text(
            0.298,
            1 - yii - 0.298,
            f"Y(II)={yii:.2f}",
            color="#666666",
            fontsize=8.5,
            ha="right",
            va="bottom",
        )

    high_mean = endpoint_leaves.loc[endpoint_leaves["zone"].eq("High"), ["YNO", "YNPQ"]].mean()
    low_mean = endpoint_leaves.loc[endpoint_leaves["zone"].eq("Low"), ["YNO", "YNPQ"]].mean()
    leaf_ax.scatter(
        [high_mean["YNO"], low_mean["YNO"]],
        [high_mean["YNPQ"], low_mean["YNPQ"]],
        s=150,
        marker="X",
        color=[ZONE_COLORS["High"], ZONE_COLORS["Low"]],
        edgecolor="white",
        linewidth=1.2,
        zorder=6,
    )
    leaf_ax.annotate(
        "",
        xy=(low_mean["YNO"], low_mean["YNPQ"]),
        xytext=(high_mean["YNO"], high_mean["YNPQ"]),
        arrowprops={"arrowstyle": "->", "color": "#4d4d4d", "lw": 2.0},
    )
    leaf_ax.text(
        0.247,
        0.649,
        "difference of\ngroup means",
        color="#4d4d4d",
        fontsize=9.5,
        ha="center",
    )
    leaf_ax.set(
        title="Individual leaves at Q = 2000",
        xlabel="Y(NO): unregulated loss",
        ylabel="Y(NPQ): regulated dissipation",
        xlim=(0.18, 0.305),
        ylim=(0.54, 0.72),
    )
    leaf_ax.legend(loc="lower left", fontsize=9.5)

    fig.suptitle("Low-site leaves replace unregulated loss with regulated photoprotection")
    fig.text(
        0.995,
        0.008,
        "Ci = 800; percentage points preserve additive closure. Relative changes use the high-site mean as denominator and do not sum.",
        ha="right",
        va="bottom",
        fontsize=9.5,
        color="#555555",
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.94,
        bottom=0.10,
        top=0.88,
        hspace=0.38,
        wspace=0.30,
    )
    fig.savefig(DIFFERENCE_FIGURE, dpi=240, bbox_inches="tight")
    fig.savefig(DIFFERENCE_PDF, bbox_inches="tight")
    plt.close(fig)


def plot_surface_differences(surface: pd.DataFrame) -> None:
    ci_levels = np.sort(surface["Ci"].unique())
    q_levels = np.sort(surface["Q"].unique())[::-1]
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17.0, 6.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    image = None

    for ax, column in zip(axes, ["YII", "YNPQ", "YNO"]):
        component = surface.loc[surface["yield"].eq(column)]
        matrix = (
            component.pivot(index="Q", columns="Ci", values="low_minus_high")
            .reindex(index=q_levels, columns=ci_levels)
            .to_numpy()
            * 100
        )
        image = ax.imshow(
            matrix,
            cmap="RdBu_r",
            vmin=-12,
            vmax=12,
            aspect="auto",
            interpolation="nearest",
        )
        resolved = component.loc[component["interval_excludes_zero"]]
        resolved_x = [np.where(ci_levels == value)[0][0] for value in resolved["Ci"]]
        resolved_y = [np.where(q_levels == value)[0][0] for value in resolved["Q"]]
        ax.scatter(
            resolved_x,
            resolved_y,
            s=13,
            color="#222222",
            alpha=0.75,
        )
        x_indices = [0, 4, 8, 12, 16, 18]
        ax.set_title(YIELD_LABELS[column])
        ax.set_xlabel("Ci (umol mol-1)")
        ax.set_xticks(x_indices, [f"{ci_levels[index]:.0f}" for index in x_indices])
        ax.set_yticks(np.arange(len(q_levels)), [f"{q:.0f}" for q in q_levels])
    axes[0].set_ylabel(r"Q ($\mu$mol photons m$^{-2}$ s$^{-1}$)")
    assert image is not None
    colorbar = fig.colorbar(
        image,
        ax=axes,
        orientation="horizontal",
        shrink=0.80,
        pad=0.16,
        aspect=45,
    )
    colorbar.set_label("Low - high yield (percentage points)")
    fig.suptitle("Excitation-energy reallocation across the Q-Ci surface")
    fig.savefig(SURFACE_DIFFERENCE_FIGURE, dpi=240, bbox_inches="tight")
    fig.savefig(SURFACE_DIFFERENCE_PDF, bbox_inches="tight")
    plt.close(fig)


def plot_relative_yno_surface(surface: pd.DataFrame) -> None:
    component = surface.loc[surface["yield"].eq("YNO")]
    ci_levels = np.sort(component["Ci"].unique())
    q_levels = np.sort(component["Q"].unique())[::-1]
    matrix = (
        component.pivot(index="Q", columns="Ci", values="relative_change_percent")
        .reindex(index=q_levels, columns=ci_levels)
        .to_numpy()
    )

    fig, ax = plt.subplots(figsize=(12.8, 6.8))
    image = ax.imshow(
        matrix,
        cmap="RdBu_r",
        vmin=-30,
        vmax=10,
        aspect="auto",
        interpolation="nearest",
    )
    resolved = component.loc[
        (component["relative_bootstrap_2.5"] > 0)
        | (component["relative_bootstrap_97.5"] < 0)
    ]
    resolved_x = [np.where(ci_levels == value)[0][0] for value in resolved["Ci"]]
    resolved_y = [np.where(q_levels == value)[0][0] for value in resolved["Q"]]
    ax.scatter(
        resolved_x,
        resolved_y,
        s=18,
        color="#222222",
        alpha=0.75,
    )
    x_indices = [0, 4, 8, 12, 16, 18]
    ax.set(
        xlabel="Ci (umol mol-1)",
        ylabel=r"Q ($\mu$mol photons m$^{-2}$ s$^{-1}$)",
    )
    ax.set_xticks(x_indices, [f"{ci_levels[index]:.0f}" for index in x_indices])
    ax.set_yticks(np.arange(len(q_levels)), [f"{q:.0f}" for q in q_levels])
    colorbar = fig.colorbar(image, ax=ax, orientation="horizontal", pad=0.15, aspect=45)
    colorbar.set_label("Change in Y(NO) relative to high-site mean (%)")
    fig.suptitle("Low-site leaves have about 22% lower unregulated yield at high light")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RELATIVE_YNO_FIGURE, dpi=240, bbox_inches="tight")
    fig.savefig(RELATIVE_YNO_PDF, bbox_inches="tight")
    plt.close(fig)


def plot_high_low_distributions(
    common: pd.DataFrame,
    leaf_means: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    columns = ["YII", "YNPQ", "YNO"]
    panel_titles = {
        "YII": "Photochemistry",
        "YNPQ": "Regulated dissipation",
        "YNO": "Unregulated loss",
    }
    y_labels = {
        "YII": "Light going toward photosynthesis",
        "YNPQ": "Regulated dissipation (Y(NPQ))",
        "YNO": "Unregulated loss (Y(NO))",
    }
    y_limits = {
        "YII": (0.04, 0.77),
        "YNPQ": (0.02, 0.75),
        "YNO": (0.18, 0.32),
    }
    headline_colors = {
        "YII": "#2b8c69",
        "YNPQ": "#d97919",
        "YNO": "#506b7c",
    }
    positions = {"High": 0.0, "Low": 1.0}
    rng = np.random.default_rng(20260806)
    norm = plt.Normalize(Q_LEVELS.min(), Q_LEVELS.max())
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 7.8))
    for ax, column in zip(axes, columns):
        for zone in ["High", "Low"]:
            points = common.loc[common["zone"].eq(zone)]
            jitter = rng.uniform(-0.16, 0.16, len(points))
            ax.scatter(
                positions[zone] + jitter,
                points[column],
                c=points["Q"],
                cmap=cmap,
                norm=norm,
                s=13,
                alpha=0.13,
                edgecolors="none",
                zorder=1,
            )

        box_values = [
            common.loc[common["zone"].eq(zone), column].to_numpy()
            for zone in ["High", "Low"]
        ]
        boxes = ax.boxplot(
            box_values,
            positions=[0, 1],
            widths=0.36,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#222222", "linewidth": 2.2},
            whiskerprops={"color": "#444444", "linewidth": 1.6},
            capprops={"color": "#444444", "linewidth": 1.6},
            boxprops={"linewidth": 2.2},
            zorder=5,
        )
        for patch, zone in zip(boxes["boxes"], ["High", "Low"]):
            patch.set_facecolor(ZONE_COLORS[zone])
            patch.set_alpha(0.22)
            patch.set_edgecolor(ZONE_COLORS[zone])

        for zone in ["High", "Low"]:
            leaves = leaf_means.loc[leaf_means["zone"].eq(zone)]
            offsets = np.linspace(-0.075, 0.075, len(leaves))
            ax.scatter(
                positions[zone] + offsets,
                leaves[column],
                s=48,
                marker="D",
                facecolor="white",
                edgecolor=ZONE_COLORS[zone],
                linewidth=1.7,
                zorder=7,
            )

        result = summary.loc[summary["yield"].eq(column)].iloc[0]
        relative = result["relative_change_percent"]
        lower = result["relative_bootstrap_2.5"]
        upper = result["relative_bootstrap_97.5"]
        ax.set_title(panel_titles[column], pad=51)
        ax.text(
            0.5,
            1.095,
            f"{relative:+.1f}%",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=headline_colors[column],
            fontsize=22,
            fontweight="bold",
        )
        ax.text(
            0.5,
            1.045,
            f"95% interval {lower:+.1f}% to {upper:+.1f}%",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#555555",
            fontsize=10.5,
        )
        ax.set(
            ylabel=panel_titles[column].split(":")[0],
            xlim=(-0.45, 1.45),
            ylim=y_limits[column],
            xticks=[0, 1],
            xticklabels=["High\nshaded", "Low\nopen-sun"],
        )
        ax.grid(axis="x", visible=False)

    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(
        scalar,
        ax=axes,
        orientation="horizontal",
        shrink=0.72,
        pad=0.16,
        aspect=45,
    )
    colorbar.set_label(r"Light level, Q ($\mu$mol photons m$^{-2}$ s$^{-1}$)")
    colorbar.set_ticks([50, 200, 600, 1000, 1400, 2000])
    fig.text(
        0.995,
        0.008,
        "Translucent points: all available Ci 300-1200 observations. Boxes: pooled distributions. Outlined diamonds and percentages: leaf-standardized, balanced Ci 350-1200 estimates.",
        ha="right",
        va="bottom",
        fontsize=9.5,
        color="#555555",
    )
    fig.savefig(DISTRIBUTION_FIGURE, dpi=240, bbox_inches="tight")
    fig.savefig(DISTRIBUTION_PDF, bbox_inches="tight")
    plt.close(fig)


def plot_binned_light_distributions(
    common: pd.DataFrame,
    leaf_means: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    from matplotlib.patches import Patch

    raw = common.copy()
    raw["light_bin"] = raw["Q"].map(Q_TO_LIGHT_BIN)
    columns = ["YII", "YNPQ", "YNO"]
    panel_titles = {
        "YII": "Photochemistry",
        "YNPQ": "Regulated dissipation",
        "YNO": "Unregulated loss",
    }
    y_labels = {
        "YII": "Light going toward photosynthesis",
        "YNPQ": "Regulated dissipation (Y(NPQ))",
        "YNO": "Unregulated loss (Y(NO))",
    }
    y_limits = {
        "YII": (0.04, 0.78),
        "YNPQ": (0.02, 0.76),
        "YNO": (0.18, 0.32),
    }
    headline_colors = {
        "YII": "#2b8c69",
        "YNPQ": "#d97919",
        "YNO": "#506b7c",
    }
    zone_offsets = {"High": -0.19, "Low": 0.19}
    rng = np.random.default_rng(20260808)

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 8.0))
    for ax, column in zip(axes, columns):
        for bin_index, light_bin in enumerate(LIGHT_BIN_ORDER):
            for zone in ["High", "Low"]:
                center = bin_index + zone_offsets[zone]
                points = raw.loc[
                    raw["light_bin"].eq(light_bin) & raw["zone"].eq(zone)
                ]
                jitter = rng.uniform(-0.105, 0.105, len(points))
                ax.scatter(
                    center + jitter,
                    points[column],
                    color=LIGHT_BIN_COLORS[light_bin],
                    s=12,
                    alpha=0.14,
                    edgecolors="none",
                    zorder=1,
                )

                leaves = leaf_means.loc[
                    leaf_means["light_bin"].eq(light_bin)
                    & leaf_means["zone"].eq(zone),
                    column,
                ].to_numpy()
                boxes = ax.boxplot(
                    [leaves],
                    positions=[center],
                    widths=0.28,
                    patch_artist=True,
                    showfliers=False,
                    medianprops={"color": "#222222", "linewidth": 2.0},
                    whiskerprops={"color": ZONE_COLORS[zone], "linewidth": 1.6},
                    capprops={"color": ZONE_COLORS[zone], "linewidth": 1.6},
                    boxprops={"linewidth": 2.0},
                    zorder=5,
                )
                patch = boxes["boxes"][0]
                patch.set_facecolor(ZONE_COLORS[zone])
                patch.set_alpha(0.23)
                patch.set_edgecolor(ZONE_COLORS[zone])
            result = summary.loc[
                summary["light_bin"].eq(light_bin)
                & summary["yield"].eq(column)
            ].iloc[0]
            relative = result["relative_change_percent"]
            resolved = bool(result["interval_excludes_zero"])
            ax.text(
                bin_index,
                0.975,
                f"{relative:+.1f}%",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                color=headline_colors[column],
                alpha=1.0 if resolved else 0.62,
                fontsize=13,
                fontweight="bold" if resolved else "normal",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.8},
                zorder=10,
            )

        ax.set_title(panel_titles[column], pad=14)
        ax.set(
            ylabel=y_labels[column],
            xlim=(-0.55, 3.55),
            ylim=y_limits[column],
            xticks=np.arange(len(LIGHT_BIN_ORDER)),
            xticklabels=LIGHT_BIN_ORDER,
        )
        ax.grid(axis="x", visible=False)

    legend_handles = [
        Patch(
            facecolor=ZONE_COLORS["High"],
            edgecolor=ZONE_COLORS["High"],
            alpha=0.23,
            label="High",
        ),
        Patch(
            facecolor=ZONE_COLORS["Low"],
            edgecolor=ZONE_COLORS["Low"],
            alpha=0.23,
            label="Low",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.995),
        fontsize=11,
        frameon=False,
    )
    fig.subplots_adjust(top=0.89, bottom=0.26, wspace=0.30)
    light_ax = fig.add_axes([0.20, 0.07, 0.60, 0.060])
    light_ax.set_xlim(0, len(LIGHT_BIN_ORDER))
    light_ax.set_ylim(0, 1)
    light_ax.set_title(
        r"Light regime, Q ($\mu$mol photons m$^{-2}$ s$^{-1}$)",
        loc="center",
        pad=5,
        fontsize=11,
    )
    ranges = {
        "Low": "50-200",
        "Moderate": "350-600",
        "High": "1000-1400",
        "Extreme": "1700-2000",
    }
    text_colors = {
        "Low": "white",
        "Moderate": "white",
        "High": "white",
        "Extreme": "#282828",
    }
    for index, light_bin in enumerate(LIGHT_BIN_ORDER):
        light_ax.barh(
            0.5,
            1,
            left=index,
            height=1,
            color=LIGHT_BIN_COLORS[light_bin],
            edgecolor="white",
            linewidth=1.5,
        )
        light_ax.text(
            index + 0.5,
            0.5,
            f"{light_bin}  {ranges[light_bin]}",
            ha="center",
            va="center",
            color=text_colors[light_bin],
            fontsize=10.5,
            fontweight="bold",
        )
    light_ax.axis("off")
    fig.savefig(BINNED_FIGURE, dpi=240, bbox_inches="tight")
    fig.savefig(BINNED_PDF, bbox_inches="tight")
    plt.close(fig)


def plot_stacked_composition(summary: pd.DataFrame) -> None:
    from matplotlib.patches import Patch, Rectangle

    components = ["YII", "YNPQ", "YNO"]
    component_labels = {
        "YII": "Photochemistry",
        "YNPQ": "Regulated dissipation",
        "YNO": "Unregulated loss",
    }
    zone_offsets = {"High": -0.23, "Low": 0.23}
    centers = np.arange(len(LIGHT_BIN_ORDER), dtype=float) * 1.55
    width = 0.36

    fig, ax = plt.subplots(figsize=(15.8, 8.0))
    for bin_index, light_bin in enumerate(LIGHT_BIN_ORDER):
        center = centers[bin_index]
        boundaries: dict[str, list[float]] = {}
        for zone in ["High", "Low"]:
            x = center + zone_offsets[zone]
            field = "high_mean" if zone == "High" else "low_mean"
            values = {
                row["yield"]: float(row[field])
                for _, row in summary.loc[
                    summary["light_bin"].eq(light_bin)
                ].iterrows()
            }
            bottom = 0.0
            boundaries[zone] = []
            for component in components:
                value = values[component]
                ax.bar(
                    x,
                    value,
                    width=width,
                    bottom=bottom,
                    color=YIELD_COLORS[component],
                    edgecolor="white",
                    linewidth=1.4,
                    zorder=3,
                )
                ax.text(
                    x,
                    bottom + value / 2,
                    f"{value * 100:.1f}%",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=11,
                    fontweight="bold",
                    zorder=5,
                )
                bottom += value
                boundaries[zone].append(bottom)
            ax.add_patch(
                Rectangle(
                    (x - width / 2, 0),
                    width,
                    1,
                    fill=False,
                    edgecolor=ZONE_COLORS[zone],
                    linewidth=2.4,
                    zorder=6,
                )
            )
            ax.text(
                x,
                -0.035,
                zone,
                ha="center",
                va="top",
                color=ZONE_COLORS[zone],
                fontsize=11,
                fontweight="bold",
                clip_on=False,
            )

        high_x = center + zone_offsets["High"] + width / 2
        low_x = center + zone_offsets["Low"] - width / 2
        for boundary_index, color in zip([0, 1], [YIELD_COLORS["YII"], YIELD_COLORS["YNPQ"]]):
            ax.plot(
                [high_x, low_x],
                [boundaries["High"][boundary_index], boundaries["Low"][boundary_index]],
                color=color,
                lw=1.5,
                alpha=0.65,
                zorder=4,
            )

        ax.text(
            center,
            -0.125,
            f"{light_bin}\n{LIGHT_BIN_LABELS[light_bin].splitlines()[1]}",
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
            color=LIGHT_BIN_COLORS[light_bin],
            clip_on=False,
        )

    legend_handles = [
        Patch(color=YIELD_COLORS[component], label=component_labels[component])
        for component in components
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.19),
        ncol=3,
        frameon=False,
        fontsize=11,
    )
    ax.set(
        ylabel="Fraction of absorbed PSII excitation",
        xlim=(centers[0] - 0.75, centers[-1] + 0.75),
        ylim=(0, 1.09),
    )
    ax.set_xticks([])
    ax.set_yticks(np.linspace(0, 1, 6), [f"{value:.0%}" for value in np.linspace(0, 1, 6)])
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color="#d7d7d7", lw=0.8)
    fig.subplots_adjust(top=0.88, bottom=0.23, left=0.09, right=0.98)
    fig.savefig(STACKED_FIGURE, dpi=240, bbox_inches="tight")
    fig.savefig(STACKED_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    style()
    active = calculate_flash_yields()
    common = interpolate_common_ci(active)
    summary = summarize_ci800(common)
    differences = bootstrap_differences(common)
    surface_differences = bootstrap_surface_differences(common)
    leaf_means, leaf_summary = leaf_standardized_summary(common)
    binned_leaf_means, binned_summary = binned_light_summary(common)

    active.to_csv(FLASH_OUT, index=False)
    common.to_csv(COMMON_CI_OUT, index=False)
    summary.to_csv(SUMMARY_OUT, index=False)
    differences.to_csv(DIFFERENCE_OUT, index=False)
    surface_differences.to_csv(SURFACE_DIFFERENCE_OUT, index=False)
    leaf_summary.to_csv(LEAF_SUMMARY_OUT, index=False)
    binned_summary.to_csv(BINNED_SUMMARY_OUT, index=False)
    plot_partition(common)
    plot_difference_evidence(common, differences)
    plot_surface_differences(surface_differences)
    plot_relative_yno_surface(surface_differences)
    plot_high_low_distributions(common, leaf_means, leaf_summary)
    plot_binned_light_distributions(common, binned_leaf_means, binned_summary)
    plot_stacked_composition(binned_summary)

    print(f"Flash rows: {len(active)}")
    print(f"Yield range: {active[['YII', 'YNPQ', 'YNO']].min().min():.6f} to {active[['YII', 'YNPQ', 'YNO']].max().max():.6f}")
    print(f"Maximum closure error: {(active['yield_sum'] - 1.0).abs().max():.3e}")
    for path in [FLASH_OUT, COMMON_CI_OUT, SUMMARY_OUT, DIFFERENCE_OUT, SURFACE_DIFFERENCE_OUT, LEAF_SUMMARY_OUT, BINNED_SUMMARY_OUT, FIGURE_OUT, PDF_OUT, DIFFERENCE_FIGURE, DIFFERENCE_PDF, SURFACE_DIFFERENCE_FIGURE, SURFACE_DIFFERENCE_PDF, RELATIVE_YNO_FIGURE, RELATIVE_YNO_PDF, DISTRIBUTION_FIGURE, DISTRIBUTION_PDF, BINNED_FIGURE, BINNED_PDF, STACKED_FIGURE, STACKED_PDF]:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
