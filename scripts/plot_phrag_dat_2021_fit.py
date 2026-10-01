from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

SUMMARY_PATH = OUTPUT_DIR / "phrag_aq_dark_rerun_dat2021_fit_summary.csv"
INPUT_PATH = OUTPUT_DIR / "phrag_aq_dark_rerun_dat2021_fit_inputs.csv"
PREDICTION_PATH = OUTPUT_DIR / "phrag_aq_dark_rerun_dat2021_fit_predictions.csv"

FIT_OVERLAY_OUT = OUTPUT_DIR / "phrag_aq_dark_rerun_dat2021_fit_overlay.png"
LITERATURE_OUT = OUTPUT_DIR / "phrag_aq_dark_rerun_dat2021_literature_comparison.png"


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(SUMMARY_PATH)
    inputs = pd.read_csv(INPUT_PATH)
    predictions = pd.read_csv(PREDICTION_PATH)
    return summary, inputs, predictions


def plot_fit_overlay(summary: pd.DataFrame, inputs: pd.DataFrame, predictions: pd.DataFrame) -> None:
    primary_labels = [
        "Q400_binned25_median_CiLT500",
        "Q700_binned25_median_CiLT500",
        "Q1000_binned25_median_CiLT500",
        "Q1600_binned25_median_CiLT500",
    ]
    plot_inputs = inputs.loc[inputs["fit_label"].isin(primary_labels)].copy()
    plot_predictions = predictions.loc[predictions["fit_label"].isin(primary_labels)].copy()
    plot_summary = summary.loc[summary["fit_label"].isin(primary_labels)].copy()

    sns.set_theme(style="whitegrid", context="talk", font_scale=0.78)
    fig, ax = plt.subplots(figsize=(12.8, 7.4))
    cmap = plt.get_cmap("viridis")
    q_min = plot_summary["Q_level"].min()
    q_max = plot_summary["Q_level"].max()
    norm = plt.Normalize(q_min, q_max)

    for q_level in sorted(plot_summary["Q_level"].unique()):
        color = cmap(norm(q_level))
        label = f"Q {int(q_level)}"
        group_inputs = plot_inputs.loc[plot_inputs["Q_level"].eq(q_level)].sort_values("Ci")
        group_predictions = plot_predictions.loc[plot_predictions["Q_level"].eq(q_level)].sort_values("Ci")
        row = plot_summary.loc[plot_summary["Q_level"].eq(q_level)].iloc[0]
        alpha = 0.42 if q_level < 1000 else 0.74
        ax.scatter(group_inputs["Ci"], group_inputs["A"], s=34, color=color, alpha=alpha, edgecolor="none")
        ax.plot(group_predictions["Ci"], group_predictions["A_model"], color=color, lw=2.8, label=label)

    high_q = plot_summary.loc[plot_summary["Q_level"].isin([1000, 1600])].sort_values("Q_level")
    fit_note = "\n".join(
        f"Q {int(row.Q_level)}: Vcmax {row.Vcmax25:.0f}, Jmax {row.Jmax25:.0f}, Rd {row.Rd:.1f}"
        for row in high_q.itertuples()
    )
    ax.text(
        0.61,
        0.34,
        fit_note,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color="#303030",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d0d0d0", "alpha": 0.92},
    )

    ax.axhline(0, color="#555555", lw=1, ls=":")
    ax.set_xlim(30, 540)
    ax.set_xlabel(r"Ci used for fit ($\mu$mol mol$^{-1}$)")
    ax.set_ylabel(r"A / DAT Adyn ($\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$)")
    ax.set_title("DAT 2021-Style FvCB Fits", loc="left", fontweight="bold", pad=12)
    ax.text(
        0.02,
        0.98,
        "plantecophys::fitaci, default settings; 25-ppm Ci medians; Ci < 500 following Saathoff & Welles' fitting note.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color="#4c4c4c",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="Ramp light", frameon=False, loc="lower right", ncol=2, fontsize=9.5, title_fontsize=10)
    fig.tight_layout()
    fig.savefig(FIT_OVERLAY_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def literature_values() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"group": "M invasive\nEller & Brix", "type": "literature", "Vcmax25": 84, "Vcmax_se": 6, "Jmax25": 116, "Jmax_se": 10},
            {"group": "AI\nEller & Brix", "type": "literature", "Vcmax25": 99, "Vcmax_se": 11, "Jmax25": 137, "Jmax_se": 17},
            {"group": "I2\nEller & Brix", "type": "literature", "Vcmax25": 118, "Vcmax_se": 6, "Jmax25": 178, "Jmax_se": 10},
            {"group": "M1\nEller & Brix", "type": "literature", "Vcmax25": 123, "Vcmax_se": 6, "Jmax25": 172, "Jmax_se": 9},
        ]
    )


def plot_literature_comparison(summary: pd.DataFrame) -> None:
    primary = summary.loc[
        summary["fit_label"].isin(
            [
                "Q1000_binned25_median_CiLT500",
                "Q1600_binned25_median_CiLT500",
                "Q1600_raw_2hz_CiLT500",
            ]
        )
        & summary["status"].eq("ok")
    ].copy()
    label_map = {
        "Q1000_binned25_median_CiLT500": "This run\nQ1000 binned",
        "Q1600_binned25_median_CiLT500": "This run\nQ1600 binned",
        "Q1600_raw_2hz_CiLT500": "This run\nQ1600 raw",
    }
    primary["group"] = primary["fit_label"].map(label_map)
    primary["type"] = "this run"
    primary["Vcmax_se"] = primary["Vcmax25_se"]
    primary["Jmax_se"] = primary["Jmax25_se"]

    combined = pd.concat(
        [
            literature_values(),
            primary[["group", "type", "Vcmax25", "Vcmax_se", "Jmax25", "Jmax_se"]],
        ],
        ignore_index=True,
    )
    order = [
        "M invasive\nEller & Brix",
        "This run\nQ1000 binned",
        "This run\nQ1600 binned",
        "This run\nQ1600 raw",
        "AI\nEller & Brix",
        "I2\nEller & Brix",
        "M1\nEller & Brix",
    ]
    combined["group"] = pd.Categorical(combined["group"], categories=order, ordered=True)
    combined = combined.sort_values("group")

    colors = combined["type"].map({"literature": "#4f7c8a", "this run": "#8a5fbf"}).to_list()

    sns.set_theme(style="whitegrid", context="talk", font_scale=0.76)
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.7), sharex=True)
    x = np.arange(len(combined))
    for ax, value_col, err_col, title in [
        (axes[0], "Vcmax25", "Vcmax_se", "Vcmax"),
        (axes[1], "Jmax25", "Jmax_se", "Jmax"),
    ]:
        ax.bar(x, combined[value_col], color=colors, alpha=0.88)
        err = combined[err_col].to_numpy(dtype=float)
        mask = np.isfinite(err)
        ax.errorbar(x[mask], combined.loc[mask, value_col], yerr=err[mask], fmt="none", ecolor="#303030", capsize=3, lw=1)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(r"$\mu$mol m$^{-2}$ s$^{-1}$")
        ax.set_xticks(x)
        ax.set_xticklabels(combined["group"], rotation=35, ha="right")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, axis="y", color="#d9ded6")
    fig.text(0.06, 0.982, "Phragmites DAT Fits vs Published Haplotype Values", ha="left", va="top", fontsize=19, fontweight="bold")
    fig.text(
        0.06,
        0.93,
        "This-run estimates use Saathoff/Welles-style fitaci settings on Ci < 500; literature values are Eller & Brix 2013 Table 3.",
        ha="left",
        va="top",
        fontsize=10,
        color="#555555",
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.29, top=0.83, wspace=0.24)
    fig.savefig(LITERATURE_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    summary, inputs, predictions = load_tables()
    plot_fit_overlay(summary, inputs, predictions)
    plot_literature_comparison(summary)
    print("Wrote:")
    print(f"  {FIT_OVERLAY_OUT}")
    print(f"  {LITERATURE_OUT}")


if __name__ == "__main__":
    main()
