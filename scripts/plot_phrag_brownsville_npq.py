from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
COMMON_PATH = OUTPUTS / "phrag_brownsville_cohort_fluorescence_at_common_ci.csv"
FLASH_PATH = OUTPUTS / "phrag_brownsville_cohort_flashes.csv"

LIGHT_FIGURE = OUTPUTS / "phrag_brownsville_npq_light_response.png"
LIGHT_PDF = OUTPUTS / "phrag_brownsville_npq_light_response.pdf"
SURFACE_FIGURE = OUTPUTS / "phrag_brownsville_npq_q_ci_maps.png"
SURFACE_PDF = OUTPUTS / "phrag_brownsville_npq_q_ci_maps.pdf"
CONTEXT_FIGURE = OUTPUTS / "phrag_brownsville_npq_energy_context.png"
CONTEXT_PDF = OUTPUTS / "phrag_brownsville_npq_energy_context.pdf"
CI_GRID_FIGURE = OUTPUTS / "phrag_brownsville_npq_vs_q_by_ci_grid.png"
CI_GRID_PDF = OUTPUTS / "phrag_brownsville_npq_vs_q_by_ci_grid.pdf"
STATS_OUT = OUTPUTS / "phrag_brownsville_npq_by_q_at_ci800.csv"
SURFACE_OUT = OUTPUTS / "phrag_brownsville_npq_interpolated_q_ci.csv"

VALID_RUNS = ["P03", "P06", "P10", "P11", "P12", "P13", "P14"]
ZONE_RUNS = {
    "Low": ["P03", "P06", "P10", "P11"],
    "High": ["P12", "P13", "P14"],
}
ZONE_LABELS = {
    "Low": "Low / saline, open-sun",
    "High": "High / lower-salinity, shaded",
}
COLORS = {"Low": "#d76528", "High": "#2478a6"}
Q_LEVELS = np.array([50, 100, 200, 350, 600, 1000, 1400, 1700, 2000], dtype=float)


def style() -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.88)
    plt.rcParams.update(
        {
            "axes.titleweight": "bold",
            "axes.titlesize": 16,
            "figure.titlesize": 22,
            "figure.titleweight": "bold",
            "legend.frameon": False,
        }
    )


def load_common() -> pd.DataFrame:
    data = pd.read_csv(COMMON_PATH)
    return data.loc[
        data["run_code"].isin(VALID_RUNS) & data["fluorescence_qc_ok"]
    ].copy()


def bootstrap_by_q(data: pd.DataFrame, draws: int = 30_000) -> pd.DataFrame:
    rng = np.random.default_rng(20260803)
    rows: list[dict[str, float | int]] = []
    for q, group in data.groupby("Q_level"):
        high = group.loc[group["zone"].eq("High"), "NPQ_at_Ci800"].dropna().to_numpy()
        low = group.loc[group["zone"].eq("Low"), "NPQ_at_Ci800"].dropna().to_numpy()
        boot = np.empty(draws)
        for index in range(draws):
            boot[index] = np.median(rng.choice(low, len(low), replace=True)) - np.median(
                rng.choice(high, len(high), replace=True)
            )
        rows.append(
            {
                "Q": float(q),
                "n_high": len(high),
                "n_low": len(low),
                "high_NPQ_median": float(np.median(high)),
                "low_NPQ_median": float(np.median(low)),
                "low_minus_high": float(np.median(low) - np.median(high)),
                "bootstrap_2.5": float(np.quantile(boot, 0.025)),
                "bootstrap_97.5": float(np.quantile(boot, 0.975)),
                "P_low_gt_high": float(np.mean(boot > 0)),
            }
        )
    return pd.DataFrame(rows).sort_values("Q")


def endpoint_labels(ax: plt.Axes, data: pd.DataFrame) -> None:
    endpoints = data.loc[data["Q_level"].eq(2000), ["run_code", "zone", "NPQ_at_Ci800"]].copy()
    offsets = {"P03": -0.02, "P06": -0.03, "P10": -0.01, "P11": 0.03, "P12": -0.025, "P13": 0.10, "P14": -0.02}
    for row in endpoints.itertuples():
        ax.text(
            2035,
            row.NPQ_at_Ci800 + offsets.get(row.run_code, 0.0),
            row.run_code,
            color=COLORS[row.zone],
            fontsize=10,
            va="center",
        )


def plot_light_response(data: pd.DataFrame, stats: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 6.8),
        gridspec_kw={"width_ratios": [1.65, 1]},
    )
    ax, effect_ax = axes

    for zone in ["High", "Low"]:
        zone_data = data.loc[data["zone"].eq(zone)]
        for _, leaf in zone_data.groupby("run_code"):
            ax.plot(
                leaf["Q_level"],
                leaf["NPQ_at_Ci800"],
                color=COLORS[zone],
                marker="o",
                ms=3.8,
                lw=1.15,
                alpha=0.30,
            )
        medians = zone_data.groupby("Q_level", as_index=False)["NPQ_at_Ci800"].median()
        ax.plot(
            medians["Q_level"],
            medians["NPQ_at_Ci800"],
            color=COLORS[zone],
            marker="o",
            ms=7,
            lw=3.2,
            label=ZONE_LABELS[zone],
            zorder=5,
        )

    endpoint_labels(ax, data)
    ax.set(
        title="NPQ diverges as light increases",
        xlabel=r"Incident light, Q ($\mu$mol photons m$^{-2}$ s$^{-1}$)",
        ylabel="NPQ at Ci = 800",
        xlim=(-20, 2180),
        ylim=(0, 3.65),
    )
    ax.legend(loc="upper left")

    effect_ax.axhline(0, color="#555555", ls="--", lw=1.2)
    effect_ax.fill_between(
        stats["Q"],
        stats["bootstrap_2.5"],
        stats["bootstrap_97.5"],
        color="#6f4b8b",
        alpha=0.20,
    )
    effect_ax.plot(
        stats["Q"],
        stats["low_minus_high"],
        color="#6f4b8b",
        marker="o",
        ms=6.5,
        lw=2.7,
    )
    effect_ax.set(
        title="Low - high NPQ",
        xlabel=r"Incident light, Q ($\mu$mol photons m$^{-2}$ s$^{-1}$)",
        ylabel="Median NPQ difference",
        xlim=(-20, 2050),
        ylim=(-0.2, 1.5),
    )
    effect_ax.text(
        650,
        1.36,
        "Leaf-bootstrap interval\nclears zero from Q = 600",
        color="#5b3a74",
        fontsize=11,
        va="top",
    )

    fig.suptitle("Phragmites photoprotective heat dissipation")
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=2.2)
    fig.savefig(LIGHT_FIGURE, dpi=240, bbox_inches="tight")
    fig.savefig(LIGHT_PDF, bbox_inches="tight")
    plt.close(fig)


def interpolate_surface() -> pd.DataFrame:
    flashes = pd.read_csv(FLASH_PATH)
    active = flashes.loc[
        flashes["run_code"].isin(VALID_RUNS)
        & flashes["flash_kind"].eq("active")
        & flashes["standard_flash"]
        & ~flashes["overlapping_flash"]
    ].dropna(subset=["gas_Ci", "NPQ", "assigned_Q"])
    ci_grid = np.arange(300, 1201, 50, dtype=float)
    rows: list[dict[str, float | str]] = []
    for (run_code, q), group in active.groupby(["run_code", "assigned_Q"]):
        curve = (
            group.groupby("gas_Ci", as_index=False)["NPQ"]
            .median()
            .sort_values("gas_Ci")
        )
        if len(curve) < 3:
            continue
        ci = curve["gas_Ci"].to_numpy()
        npq = curve["NPQ"].to_numpy()
        zone = "High" if run_code in ZONE_RUNS["High"] else "Low"
        for target in ci_grid:
            if ci.min() <= target <= ci.max():
                rows.append(
                    {
                        "run_code": run_code,
                        "zone": zone,
                        "Q": float(q),
                        "Ci": target,
                        "NPQ": float(np.interp(target, ci, npq)),
                    }
                )
    return pd.DataFrame(rows)


def surface_matrix(data: pd.DataFrame, zone: str) -> np.ndarray:
    return (
        data.loc[data["zone"].eq(zone)]
        .groupby(["Q", "Ci"])["NPQ"]
        .median()
        .unstack("Ci")
        .reindex(index=Q_LEVELS)
        .to_numpy()
    )


def plot_surfaces(data: pd.DataFrame) -> None:
    ci_grid = np.sort(data["Ci"].unique())
    high = surface_matrix(data, "High")
    low = surface_matrix(data, "Low")
    difference = low - high
    x, y = np.meshgrid(ci_grid, Q_LEVELS)

    fig, axes = plt.subplots(1, 3, figsize=(17, 6.1), constrained_layout=True)
    levels = np.linspace(0.2, 3.6, 18)
    high_map = axes[0].contourf(x, y, high, levels=levels, cmap="viridis", extend="both")
    axes[1].contourf(x, y, low, levels=levels, cmap="viridis", extend="both")
    difference_levels = np.linspace(-0.2, 1.4, 17)
    difference_map = axes[2].contourf(
        x,
        y,
        difference,
        levels=difference_levels,
        cmap="PuOr_r",
        extend="both",
    )

    for ax, title in zip(
        axes,
        [
            "High / lower-salinity, shaded",
            "Low / saline, open-sun",
            "Low - high",
        ],
    ):
        ax.set(title=title, xlabel="Ci (umol mol-1)", ylabel=r"Q ($\mu$mol photons m$^{-2}$ s$^{-1}$)")
        ax.set_yticks(Q_LEVELS[::2])

    colorbar = fig.colorbar(high_map, ax=axes[:2], orientation="horizontal", pad=0.12, shrink=0.84)
    colorbar.set_label("Median NPQ")
    difference_bar = fig.colorbar(difference_map, ax=axes[2], orientation="horizontal", pad=0.12, shrink=0.90)
    difference_bar.set_label("Median NPQ difference")
    fig.suptitle("NPQ across the measured Q-Ci space")
    fig.savefig(SURFACE_FIGURE, dpi=240, bbox_inches="tight")
    fig.savefig(SURFACE_PDF, bbox_inches="tight")
    plt.close(fig)


def plot_energy_context(data: pd.DataFrame) -> None:
    metrics = [
        ("NPQ_at_Ci800", "NPQ", "Photoprotective heat dissipation"),
        ("PhiPS2_at_Ci800", "PhiPSII", "Operating PSII efficiency"),
        ("ETR_at_Ci800", "ETR", "Electron-transport proxy"),
        ("qL_at_Ci800", "qL", "Estimated open PSII centers"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.0), sharex=True)
    for ax, (column, ylabel, title) in zip(axes.flat, metrics):
        for zone in ["High", "Low"]:
            zone_data = data.loc[data["zone"].eq(zone)]
            for _, leaf in zone_data.groupby("run_code"):
                ax.plot(
                    leaf["Q_level"],
                    leaf[column],
                    color=COLORS[zone],
                    lw=1.0,
                    alpha=0.22,
                )
            medians = zone_data.groupby("Q_level", as_index=False)[column].median()
            ax.plot(
                medians["Q_level"],
                medians[column],
                color=COLORS[zone],
                marker="o",
                ms=5.5,
                lw=2.7,
                label=ZONE_LABELS[zone],
            )
        ax.set(title=title, ylabel=ylabel)
        ax.set_xlim(-20, 2050)
    for ax in axes[1]:
        ax.set_xlabel(r"Incident light, Q ($\mu$mol photons m$^{-2}$ s$^{-1}$)")
    axes[0, 0].legend(loc="upper left", fontsize=11)
    fig.suptitle("Energy use and photoprotection at common Ci = 800")
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.0, w_pad=1.6)
    fig.savefig(CONTEXT_FIGURE, dpi=240, bbox_inches="tight")
    fig.savefig(CONTEXT_PDF, bbox_inches="tight")
    plt.close(fig)


def plot_npq_by_ci_grid(data: pd.DataFrame) -> None:
    ci_levels = np.array([300, 350, 400, 500, 600, 800, 1000, 1200], dtype=float)
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(17.0, 8.8),
        sharex=True,
        sharey=True,
    )

    for ax, ci in zip(axes.flat, ci_levels):
        panel = data.loc[data["Ci"].eq(ci)]
        for zone in ["High", "Low"]:
            zone_data = panel.loc[panel["zone"].eq(zone)]
            for _, leaf in zone_data.groupby("run_code"):
                ax.plot(
                    leaf["Q"],
                    leaf["NPQ"],
                    color=COLORS[zone],
                    lw=1.0,
                    alpha=0.23,
                )
            medians = zone_data.groupby("Q", as_index=False)["NPQ"].median()
            ax.plot(
                medians["Q"],
                medians["NPQ"],
                color=COLORS[zone],
                marker="o",
                ms=4.8,
                lw=2.7,
                label=ZONE_LABELS[zone],
                zorder=5,
            )

        coverage_note = "*" if ci == 300 else ""
        ax.set_title(f"Ci = {ci:.0f}{coverage_note}", pad=7)
        ax.set_xlim(0, 2050)
        ax.set_ylim(0, 3.75)
        ax.set_xticks([0, 500, 1000, 1500, 2000])
        ax.grid(color="#d8d8d8", lw=0.65, alpha=0.65)

    for ax in axes[-1, :]:
        ax.set_xlabel(r"Q ($\mu$mol photons m$^{-2}$ s$^{-1}$)")
    for ax in axes[:, 0]:
        ax.set_ylabel("NPQ")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.008),
        fontsize=12,
    )
    fig.text(
        0.995,
        0.012,
        "* Ci = 300 has 61 of 63 leaf-by-Q curves; all other panels have complete coverage.",
        ha="right",
        va="bottom",
        fontsize=10,
        color="#555555",
    )
    fig.suptitle("NPQ light response across intercellular CO2")
    fig.tight_layout(rect=(0, 0.07, 1, 0.94), h_pad=1.35, w_pad=0.75)
    fig.savefig(CI_GRID_FIGURE, dpi=240, bbox_inches="tight")
    fig.savefig(CI_GRID_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    style()
    common = load_common()
    stats = bootstrap_by_q(common)
    surface = interpolate_surface()
    stats.to_csv(STATS_OUT, index=False)
    surface.to_csv(SURFACE_OUT, index=False)
    plot_light_response(common, stats)
    plot_surfaces(surface)
    plot_energy_context(common)
    plot_npq_by_ci_grid(surface)
    for path in [LIGHT_FIGURE, LIGHT_PDF, SURFACE_FIGURE, SURFACE_PDF, CONTEXT_FIGURE, CONTEXT_PDF, CI_GRID_FIGURE, CI_GRID_PDF, STATS_OUT, SURFACE_OUT]:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
