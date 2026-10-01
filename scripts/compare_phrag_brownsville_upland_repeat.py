from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import compare_phrag_brownsville_positions as comparison


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIRST_POINTS = OUTPUT_DIR / "phrag_brownsville_upland_points.csv"
REPEAT_POINTS = OUTPUT_DIR / "phrag_brownsville_upland_late_repeat_points.csv"

comparison.RUNS = {
    "First run": {"path": FIRST_POINTS, "color": "#276f9e"},
    "Late repeat": {"path": REPEAT_POINTS, "color": "#b45f35"},
}
comparison.FIGURE_OUT = OUTPUT_DIR / "phrag_brownsville_upland_first_vs_late_repeat.png"
comparison.SUMMARY_OUT = OUTPUT_DIR / "phrag_brownsville_upland_repeat_summary.csv"
comparison.FIGURE_TITLE = "Brownsville Upland Phragmites: First Run vs Late Repeat"
comparison.FIGURE_SUBTITLE = (
    "Same leaf and CO2 cartridge; the repeat began after the first complete low-to-high Q sequence. "
    "Q remains confounded with induction time."
)

DELIVERY_OUT = OUTPUT_DIR / "phrag_brownsville_upland_repeat_co2_delivery.png"
DELIVERY_CSV = OUTPUT_DIR / "phrag_brownsville_upland_repeat_co2_delivery.csv"
DELTA_OUT = OUTPUT_DIR / "phrag_brownsville_upland_repeat_deltas.png"
DELTA_CSV = OUTPUT_DIR / "phrag_brownsville_upland_repeat_deltas.csv"


def delivery_summary() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for label, path in [("First run", FIRST_POINTS), ("Late repeat", REPEAT_POINTS)]:
        points = pd.read_csv(path, low_memory=False)
        high = points.loc[points["CO2_r_sp"].ge(1900)].copy()
        for q_level, group in high.groupby("Q_level", sort=True):
            rows.append(
                {
                    "run": label,
                    "Q_level": float(q_level),
                    "n": len(group),
                    "CO2_command": float(group["CO2_r_sp"].median()),
                    "CO2_reference": float(group["CO2_r"].median()),
                    "CO2_sample": float(group["CO2_s"].median()),
                    "command_minus_reference": float((group["CO2_r_sp"] - group["CO2_r"]).median()),
                    "injector_percent": float(group["CO2_%"].median()),
                    "injector_percent_max": float(group["CO2_%"].max()),
                }
            )
    return pd.DataFrame(rows)


def plot_delivery(summary: pd.DataFrame) -> None:
    colors = {label: config["color"] for label, config in comparison.RUNS.items()}
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.78)
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4))

    for label, group in summary.groupby("run", sort=False):
        group = group.sort_values("Q_level")
        color = colors[label]
        axes[0].plot(group["Q_level"], group["command_minus_reference"], color=color, marker="o", lw=2.5, label=label)
        axes[1].plot(group["Q_level"], group["injector_percent"], color=color, marker="o", lw=2.5, label=label)

    axes[0].set_title("Reference CO2 Delivery Shortfall", loc="left", fontweight="bold")
    axes[0].set_xlabel("Q")
    axes[0].set_ylabel("Command - reference CO2 (ppm)")
    axes[0].set_ylim(0, max(30, float(summary["command_minus_reference"].max()) + 3))

    axes[1].set_title("Injector Demand Near Ramp Endpoint", loc="left", fontweight="bold")
    axes[1].set_xlabel("Q")
    axes[1].set_ylabel("CO2 injector (%)")
    axes[1].set_ylim(90, 100)
    axes[1].axhline(100, color="#555555", lw=0.9, ls=":")
    axes[1].legend(frameon=False, loc="lower right", fontsize=9)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, color="#d8ddd4", linewidth=0.8)

    fig.text(
        0.06,
        0.982,
        "Second Full A/Q/C Run on the Same CO2 Cartridge",
        ha="left",
        va="top",
        fontsize=20,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.925,
        "Statistics use rows commanded above 1900 ppm; stable delivery with little injector headroom.",
        ha="left",
        va="top",
        fontsize=10.5,
        color="#555555",
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.80, wspace=0.25)
    fig.savefig(DELIVERY_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def delta_a_matrix(bin_width: int = 100) -> pd.DataFrame:
    curves: list[pd.DataFrame] = []
    for label, path in [("first", FIRST_POINTS), ("repeat", REPEAT_POINTS)]:
        points = pd.read_csv(path, low_memory=False)
        data = points.loc[
            points["sane_for_curve"].astype(bool)
            & points["Q_level"].gt(0)
            & points["Ci"].between(0, 1800)
        ].copy()
        data["Ci_center"] = np.floor(data["Ci"] / bin_width) * bin_width + bin_width / 2
        curve = (
            data.groupby(["Q_level", "Ci_center"], observed=True)
            .agg(A=("A", "median"), n=("A", "size"))
            .reset_index()
        )
        curve = curve.loc[curve["n"].ge(3), ["Q_level", "Ci_center", "A"]].rename(columns={"A": f"A_{label}"})
        curves.append(curve)

    matched = curves[0].merge(curves[1], on=["Q_level", "Ci_center"], how="inner")
    matched["delta_A"] = matched["A_repeat"] - matched["A_first"]
    return matched.pivot(index="Q_level", columns="Ci_center", values="delta_A").sort_index()


def active_flash_summary(path: Path, prefix: str) -> pd.DataFrame:
    points = pd.read_csv(path, low_memory=False)
    active = points.loc[points["light_flash_logged"].astype(bool) & points["flash_flash_id"].notna()].copy()
    summary = (
        active.groupby("Q_level", sort=True)
        .agg(
            PhiPS2=("flash_flr_PhiPS2", "median"),
            ETR=("flash_flr_ETR", "median"),
            NPQ=("flash_flr_NPQ", "median"),
            qL=("flash_flr_qL", "median"),
        )
        .reset_index()
    )
    return summary.rename(columns={column: f"{column}_{prefix}" for column in ["PhiPS2", "ETR", "NPQ", "qL"]})


def repeat_delta_summary() -> pd.DataFrame:
    summary = pd.read_csv(comparison.SUMMARY_OUT)
    first = summary.loc[summary["position"].eq("First run")].set_index("Q_level")
    repeat = summary.loc[summary["position"].eq("Late repeat")].set_index("Q_level")
    metrics = ["A_p95", "A_max", "gsw_median", "Ci_max", "VPDleaf_median", "Tleaf_median"]
    rows: list[dict[str, float]] = []
    for q_level in first.index.intersection(repeat.index):
        row: dict[str, float] = {"Q_level": float(q_level)}
        for metric in metrics:
            first_value = float(first.loc[q_level, metric])
            repeat_value = float(repeat.loc[q_level, metric])
            row[f"{metric}_first"] = first_value
            row[f"{metric}_repeat"] = repeat_value
            row[f"{metric}_delta"] = repeat_value - first_value
            row[f"{metric}_percent_delta"] = (
                (repeat_value - first_value) / first_value * 100 if first_value != 0 else np.nan
            )
        rows.append(row)

    delta = pd.DataFrame(rows)
    first_flash = active_flash_summary(
        OUTPUT_DIR / "phrag_brownsville_upland_points_with_flash.csv", "first"
    )
    repeat_flash = active_flash_summary(
        OUTPUT_DIR / "phrag_brownsville_upland_late_repeat_points_with_flash.csv", "repeat"
    )
    delta = delta.merge(first_flash, on="Q_level", how="left").merge(repeat_flash, on="Q_level", how="left")
    for metric in ["PhiPS2", "ETR", "NPQ", "qL"]:
        delta[f"{metric}_delta"] = delta[f"{metric}_repeat"] - delta[f"{metric}_first"]
    return delta


def plot_deltas(matrix: pd.DataFrame, summary: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.76)
    fig = plt.figure(figsize=(15.4, 8.2))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.2, 1.2, 0.95], wspace=0.34, hspace=0.38)
    ax_heat = fig.add_subplot(grid[:, :2])
    ax_a = fig.add_subplot(grid[0, 2])
    ax_gsw = fig.add_subplot(grid[1, 2])

    limit = max(1.0, float(np.nanmax(np.abs(matrix.to_numpy()))))
    sns.heatmap(
        matrix,
        ax=ax_heat,
        cmap="vlag",
        center=0,
        vmin=-limit,
        vmax=limit,
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"label": "Delta A: repeat - first"},
    )
    ax_heat.set_title("Assimilation Difference Across the Measured Surface", loc="left", fontweight="bold")
    ax_heat.set_xlabel("Ci bin center (umol mol-1)")
    ax_heat.set_ylabel("Q")
    ax_heat.set_xticklabels([f"{float(label.get_text()):.0f}" for label in ax_heat.get_xticklabels()], rotation=45, ha="right")
    ax_heat.set_yticklabels([f"{float(label.get_text()):.0f}" for label in ax_heat.get_yticklabels()], rotation=0)

    ax_a.axhline(0, color="#555555", lw=0.9, ls=":")
    ax_a.plot(summary["Q_level"], summary["A_p95_delta"], color="#b45f35", marker="o", lw=2.6)
    for row in summary.itertuples():
        ax_a.annotate(
            f"{row.A_p95_delta:+.1f}\n({row.A_p95_percent_delta:+.0f}%)",
            (row.Q_level, row.A_p95_delta),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
            color="#7f4227",
        )
    ax_a.set_title("Change in Upper A", loc="left", fontweight="bold")
    ax_a.set_xlabel("Q")
    ax_a.set_ylabel("Delta 95th-percentile A")
    ax_a.set_ylim(-0.9, 4.2)

    ax_gsw.axhline(0, color="#555555", lw=0.9, ls=":")
    ax_gsw.plot(
        summary["Q_level"],
        summary["gsw_median_delta"] * 1000,
        color="#276f9e",
        marker="o",
        lw=2.6,
    )
    ax_gsw.set_title("Change in Median gsw", loc="left", fontweight="bold")
    ax_gsw.set_xlabel("Q")
    ax_gsw.set_ylabel("Delta gsw (mmol m-2 s-1)")

    for ax in [ax_a, ax_gsw]:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, color="#d8ddd4", linewidth=0.8)

    fig.text(
        0.06,
        0.982,
        "Brownsville Upland Phragmites: Late Repeat Minus First Run",
        ha="left",
        va="top",
        fontsize=21,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.937,
        "Positive values mean the late repeat was higher; matched 100-ppm Ci bins use only filtered DAT rows.",
        ha="left",
        va="top",
        fontsize=10.5,
        color="#555555",
    )
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.10, top=0.85)
    fig.savefig(DELTA_OUT, dpi=220, bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)


def main() -> None:
    comparison.main()
    delivery = delivery_summary()
    delivery.to_csv(DELIVERY_CSV, index=False)
    plot_delivery(delivery)
    delta = repeat_delta_summary()
    matrix = delta_a_matrix()
    delta.to_csv(DELTA_CSV, index=False)
    plot_deltas(matrix, delta)
    print(f"Wrote {DELIVERY_OUT}")
    print(f"Wrote {DELIVERY_CSV}")
    print(f"Wrote {DELTA_OUT}")
    print(f"Wrote {DELTA_CSV}")


if __name__ == "__main__":
    main()
