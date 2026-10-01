from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from photo26.curves import OUTPUT_DIR, read_licor_export


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "2026-07-06_phrag_bptest_dynamic_ramp"
    / "2026-07-06-1342_logdata_phrag_bptest"
)

POINTS_OUT = OUTPUT_DIR / "phrag_bptest_dynamic_points.csv"
PHASE_SUMMARY_OUT = OUTPUT_DIR / "phrag_bptest_dynamic_phase_summary.csv"
LOW_CO2_PARK_OUT = OUTPUT_DIR / "phrag_bptest_low_co2_park_summary.csv"

TIMELINE_OUT = OUTPUT_DIR / "phrag_bptest_dynamic_timeline.png"
RAMP_QUALITY_OUT = OUTPUT_DIR / "phrag_bptest_dynamic_ramp_quality.png"
ACI_DIAGNOSTIC_OUT = OUTPUT_DIR / "phrag_bptest_dynamic_aci_diagnostic.png"
LOW_CO2_PARK_OUT_PNG = OUTPUT_DIR / "phrag_bptest_low_co2_park_summary.png"

ALPHA_VC = "\u03b1Vc"
ALPHA_VH = "\u03b1Vh"
Q_LEVELS = np.array([0, 200, 400, 700, 1000, 1300, 1600], dtype=float)


def nearest_q(values: pd.Series) -> np.ndarray:
    q = values.to_numpy(dtype=float)
    return Q_LEVELS[np.argmin(np.abs(q[:, None] - Q_LEVELS[None, :]), axis=1)]


def load_points() -> pd.DataFrame:
    frame = read_licor_export(DATA_PATH).sort_values("elapsed").reset_index(drop=True).copy()
    frame["minute"] = frame["elapsed"] / 60
    frame["Q_level"] = nearest_q(frame["Q"])
    frame["ramp_region"] = frame["minute"].ge(25)
    post = frame.loc[frame["ramp_region"]].copy()
    phase_ids = post["Q_level"].ne(post["Q_level"].shift()).cumsum()
    frame["phase_id"] = np.nan
    frame.loc[post.index, "phase_id"] = phase_ids.to_numpy()
    frame["A_minus_Asty"] = frame["A"] - frame["Asty"]
    frame["co2_command_jump_back"] = frame["CO2_r_sp"].diff().lt(-20)
    frame["light_flash_logged"] = frame["LightAdaptedID"].astype(str).ne("-")
    frame["dark_flash_logged"] = frame["DarkAdaptedID"].astype(str).ne("-")
    frame["sensible_dat"] = (
        frame["Ci"].between(0, 2200)
        & frame["A"].between(-30, 80)
        & frame["CO2_r"].between(0, 2200)
        & frame["CO2_s"].between(0, 2200)
        & frame["Q"].between(-5, 1705)
    )
    return frame


def summarize_phases(points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    post = points.loc[points["phase_id"].notna()].copy()
    for phase_id, group in post.groupby("phase_id", sort=True):
        rows.append(
            {
                "phase_id": int(phase_id),
                "Q_level": float(group["Q_level"].mode().iloc[0]),
                "n": len(group),
                "start_min": float(group["minute"].min()),
                "end_min": float(group["minute"].max()),
                "duration_min": float(group["minute"].max() - group["minute"].min()),
                "Q_median": float(group["Q"].median()),
                "CO2_r_sp_min": float(group["CO2_r_sp"].min()),
                "CO2_r_sp_max": float(group["CO2_r_sp"].max()),
                "CO2_command_jump_backs": int(group["co2_command_jump_back"].sum()),
                "CO2_r_min": float(group["CO2_r"].min()),
                "CO2_r_max": float(group["CO2_r"].max()),
                "CO2_s_min": float(group["CO2_s"].min()),
                "CO2_s_max": float(group["CO2_s"].max()),
                "A_min": float(group["A"].min()),
                "A_median": float(group["A"].median()),
                "A_max": float(group["A"].max()),
                "Asty_min": float(group["Asty"].min()),
                "Asty_median": float(group["Asty"].median()),
                "Asty_max": float(group["Asty"].max()),
                "Ci_min": float(group["Ci"].min()),
                "Ci_median": float(group["Ci"].median()),
                "Ci_max": float(group["Ci"].max()),
                "gsw_median": float(group["gsw"].median()),
                "Flow_median": float(group["Flow"].median()),
                "Flow_min": float(group["Flow"].min()),
                "Flow_max": float(group["Flow"].max()),
                "alphaVc_median": float(group[ALPHA_VC].median()),
                "alphaVh_median": float(group[ALPHA_VH].median()),
                "light_flash_rows": int(group["light_flash_logged"].sum()),
                "dark_flash_rows": int(group["dark_flash_logged"].sum()),
                "sensible_dat_rows": int(group["sensible_dat"].sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_low_co2_park(points: pd.DataFrame) -> pd.DataFrame:
    stable_low = points.loc[
        points["phase_id"].notna()
        & points["Q_level"].isin([200, 400, 700, 1000, 1300, 1600])
        & points["CO2_r_sp"].between(69, 72)
        & points["CO2_r"].between(65, 80)
        & points["CO2_s"].between(65, 85)
        & points["dCsd/dt"].abs().lt(0.1)
        & points["sensible_dat"]
    ].copy()
    if stable_low.empty:
        return pd.DataFrame()
    return (
        stable_low.groupby("Q_level", sort=True)
        .agg(
            n=("A", "size"),
            start_min=("minute", "min"),
            end_min=("minute", "max"),
            A_median=("A", "median"),
            A_q10=("A", lambda s: s.quantile(0.10)),
            A_q90=("A", lambda s: s.quantile(0.90)),
            Asty_median=("Asty", "median"),
            Ci_median=("Ci", "median"),
            gsw_median=("gsw", "median"),
            CO2_r_median=("CO2_r", "median"),
            CO2_s_median=("CO2_s", "median"),
            Flow_median=("Flow", "median"),
        )
        .reset_index()
    )


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#d9ddd6", linewidth=0.8)


def phase_bounds(points: pd.DataFrame) -> pd.DataFrame:
    post = points.loc[points["phase_id"].notna()].copy()
    return (
        post.groupby(["phase_id", "Q_level"], sort=True)
        .agg(start=("minute", "min"), end=("minute", "max"))
        .reset_index()
    )


def add_phase_shading(ax: plt.Axes, bounds: pd.DataFrame, cmap, norm) -> None:
    for _, row in bounds.iterrows():
        color = cmap(norm(row["Q_level"]))
        ax.axvspan(row["start"], row["end"], color=color, alpha=0.055, lw=0)
        ax.axvline(row["start"], color="#b9b9b2", ls="--", lw=0.8, alpha=0.75)


def plot_timeline(points: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.72)
    fig, axes = plt.subplots(
        6,
        1,
        figsize=(14.5, 12.0),
        sharex=True,
        gridspec_kw={"height_ratios": [0.7, 1.1, 1.05, 1.05, 1.0, 0.85], "hspace": 0.14},
    )
    bounds = phase_bounds(points)
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0, vmax=1600)
    for ax in axes:
        add_phase_shading(ax, bounds, cmap, norm)
        style_axis(ax)

    axes[0].plot(points["minute"], points["Q"], color="#2d2d2d", lw=1.5)
    axes[0].set_ylabel("Q")
    axes[0].set_ylim(-80, 1720)
    for _, row in bounds.iterrows():
        if row["end"] - row["start"] < 1:
            continue
        axes[0].text(
            (row["start"] + row["end"]) / 2,
            1615,
            f"{int(row['Q_level'])}",
            ha="center",
            va="top",
            fontsize=8,
            color=cmap(norm(row["Q_level"])),
            fontweight="bold",
        )

    axes[1].plot(points["minute"], points["CO2_r_sp"], color="#333333", lw=1.2, label="setpoint")
    axes[1].plot(points["minute"], points["CO2_r"], color="#f28e2b", lw=1.0, label="reference")
    axes[1].plot(points["minute"], points["CO2_s"], color="#cc6677", lw=1.0, label="sample")
    axes[1].set_ylabel("CO2")
    axes[1].legend(frameon=False, ncol=3, fontsize=8, loc="upper left")

    plot_a = points["A"].clip(-25, 60)
    plot_asty = points["Asty"].clip(-25, 60)
    axes[2].plot(points["minute"], plot_a, color="#2f8f46", lw=1.3, label="A / Adyn")
    axes[2].plot(points["minute"], plot_asty, color="#7f7f7f", lw=1.0, alpha=0.75, label="Asty")
    axes[2].axhline(0, color="#555555", ls=":", lw=1)
    axes[2].set_ylabel("A\nclipped")
    axes[2].legend(frameon=False, ncol=2, fontsize=8, loc="upper left")
    outlier_n = int((points["A"].lt(-25) | points["A"].gt(60) | points["Asty"].lt(-25) | points["Asty"].gt(60)).sum())
    axes[2].text(0.99, 0.90, f"{outlier_n} clipped rows", transform=axes[2].transAxes, ha="right", va="top", fontsize=8)

    axes[3].plot(points["minute"], points["Ci"].clip(-200, 2200), color="#805bb7", lw=1.0, label="Ci")
    axes[3].set_ylabel("Ci\nclipped")

    axes[4].plot(points["minute"], points["gsw"], color="#2475a8", lw=1.2)
    axes[4].set_ylabel("gsw")

    axes[5].plot(points["minute"], points["Flow"], color="#333333", lw=1.0, label="Flow")
    axes5b = axes[5].twinx()
    axes5b.plot(points["minute"], points[ALPHA_VC], color="#2ca25f", lw=1.0, label="alphaVc")
    axes5b.plot(points["minute"], points[ALPHA_VH], color="#de2d26", lw=1.0, label="alphaVh")
    axes[5].set_ylabel("Flow")
    axes5b.set_ylabel("alpha V")
    axes5b.spines[["top"]].set_visible(False)
    axes5b.grid(False)
    lines1, labels1 = axes[5].get_legend_handles_labels()
    lines2, labels2 = axes5b.get_legend_handles_labels()
    axes[5].legend(lines1 + lines2, labels1 + labels2, frameon=False, ncol=3, fontsize=8, loc="upper left")
    axes[5].set_xlabel("Elapsed time after first logged point (min)")

    fig.text(0.06, 0.982, "Phragmites BP/DAT Run Diagnostic", ha="left", va="top", fontsize=21, fontweight="bold")
    fig.text(
        0.06,
        0.952,
        "Actual run stayed near 900 flow; CO2 commands and DAT outputs are shown as logged.",
        ha="left",
        va="top",
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(TIMELINE_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ramp_quality(points: pd.DataFrame) -> None:
    post = points.loc[points["phase_id"].notna() & points["Q_level"].isin(Q_LEVELS)].copy()
    # Keep the seven main phases, excluding the final safety/leftover Q0 phase.
    main = post.loc[post["phase_id"].isin(range(1, 8))].copy()
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.68)
    fig, axes = plt.subplots(4, 2, figsize=(14, 11), sharey=True)
    axes_flat = list(axes.flat)

    for ax, (phase_id, group) in zip(axes_flat, main.groupby("phase_id", sort=True)):
        group = group.sort_values("minute").copy()
        group["phase_min"] = group["minute"] - group["minute"].min()
        q_label = int(group["Q_level"].mode().iloc[0])
        ax.plot(group["phase_min"], group["CO2_r_sp"], color="#333333", lw=1.2, label="setpoint")
        ax.plot(group["phase_min"], group["CO2_r"], color="#f28e2b", lw=1.0, label="reference")
        ax.plot(group["phase_min"], group["CO2_s"], color="#cc6677", lw=1.0, label="sample")
        drops = int(group["co2_command_jump_back"].sum())
        ax.set_title(f"Q {q_label} / {drops} backward setpoint jumps", loc="left", fontweight="bold", fontsize=10)
        ax.set_xlabel("min in Q phase")
        style_axis(ax)

    for ax in axes[:, 0]:
        ax.set_ylabel("CO2 (ppm)")
    for ax in axes_flat[len(main["phase_id"].unique()) :]:
        ax.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.955), fontsize=9)
    fig.suptitle("CO2 Ramp Quality: The Program Did Not Produce Clean Ascending Ramps", y=0.99, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(RAMP_QUALITY_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_aci_diagnostic(points: pd.DataFrame) -> None:
    sensible = points.loc[
        points["phase_id"].notna()
        & points["Q_level"].isin([200, 400, 700, 1000, 1300, 1600])
        & points["sensible_dat"]
    ].copy()
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.78)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.4), sharex=True)
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0, vmax=1600)

    for q_level, group in sensible.groupby("Q_level", sort=True):
        color = cmap(norm(q_level))
        axes[0].scatter(group["Ci"], group["A"], s=12, color=color, alpha=0.55, edgecolor="none")
        axes[1].scatter(group["Ci"], group["Asty"], s=12, color=color, alpha=0.55, edgecolor="none")

        # Median bin trend is only a visual guide.
        bins = pd.cut(group["Ci"], np.arange(0, 2250, 100), include_lowest=True)
        trend = group.groupby(bins, observed=True).agg(Ci=("Ci", "median"), A=("A", "median"), Asty=("Asty", "median"))
        axes[0].plot(trend["Ci"], trend["A"], color=color, lw=2.0)
        axes[1].plot(trend["Ci"], trend["Asty"], color=color, lw=2.0, label=f"{int(q_level)}")

    axes[0].set_title("DAT A / Adyn (filtered view)", loc="left", fontweight="bold")
    axes[1].set_title("Asty, uncorrected steady-state equation", loc="left", fontweight="bold")
    for ax in axes:
        ax.axhline(0, color="#555555", ls=":", lw=1)
        ax.set_xlabel("Ci (ppm)")
        ax.set_xlim(0, 2100)
        style_axis(ax)
    axes[0].set_ylabel("A")
    axes[1].set_ylabel("Asty")
    axes[1].legend(title="Q", frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)
    dropped = int(points["phase_id"].notna().sum() - sensible.shape[0])
    fig.text(0.06, 0.982, "A/Ci Diagnostic Only: Not A Fit-Ready Surface", ha="left", va="top", fontsize=18, fontweight="bold")
    fig.text(
        0.06,
        0.942,
        f"Filtered to visually sane rows; {dropped} phase rows omitted due impossible A/Ci/CO2 values.",
        ha="left",
        va="top",
        fontsize=9.5,
        color="#555555",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    fig.savefig(ACI_DIAGNOSTIC_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_low_co2_park(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.82)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    x = summary["Q_level"]
    axes[0].plot(x, summary["A_median"], marker="o", color="#2f8f46", lw=2.2)
    axes[0].fill_between(x, summary["A_q10"], summary["A_q90"], color="#2f8f46", alpha=0.18, lw=0)
    axes[0].axhline(0, color="#555555", ls=":", lw=1)
    axes[0].set_ylabel("A at low CO2")
    axes[1].plot(x, summary["Ci_median"], marker="o", color="#805bb7", lw=2.2)
    axes[1].set_ylabel("Ci at low CO2")
    axes[2].plot(x, summary["gsw_median"], marker="o", color="#2475a8", lw=2.2)
    axes[2].set_ylabel("gsw at low CO2")
    for ax in axes:
        ax.set_xlabel("Q level")
        style_axis(ax)
    fig.suptitle("Salvageable Low-CO2 Parking Points", y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.savefig(LOW_CO2_PARK_OUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_outputs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    points = load_points()
    phase_summary = summarize_phases(points)
    low_co2_summary = summarize_low_co2_park(points)

    points.to_csv(POINTS_OUT, index=False)
    phase_summary.to_csv(PHASE_SUMMARY_OUT, index=False)
    low_co2_summary.to_csv(LOW_CO2_PARK_OUT, index=False)

    plot_timeline(points)
    plot_ramp_quality(points)
    plot_aci_diagnostic(points)
    plot_low_co2_park(low_co2_summary)

    print("Wrote:")
    for path in [
        POINTS_OUT,
        PHASE_SUMMARY_OUT,
        LOW_CO2_PARK_OUT,
        TIMELINE_OUT,
        RAMP_QUALITY_OUT,
        ACI_DIAGNOSTIC_OUT,
        LOW_CO2_PARK_OUT_PNG,
    ]:
        print(f"  {path}")
    print()
    print("Phase summary:")
    print(phase_summary.round(3).to_string(index=False))
    print()
    print("Low-CO2 parking summary:")
    print(low_co2_summary.round(3).to_string(index=False) if not low_co2_summary.empty else "none")


if __name__ == "__main__":
    write_outputs()
