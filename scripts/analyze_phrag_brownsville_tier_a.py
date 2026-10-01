from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

QC_PATH = OUTPUT_DIR / "phrag_brownsville_cohort_run_qc.csv"
CURVES_PATH = OUTPUT_DIR / "phrag_brownsville_cohort_binned_curves.csv"
Q_SUMMARY_PATH = OUTPUT_DIR / "phrag_brownsville_cohort_q_summary.csv"
FLUORESCENCE_PATH = OUTPUT_DIR / "phrag_brownsville_cohort_fluorescence_at_common_ci.csv"
JQ_PATH = OUTPUT_DIR / "phrag_brownsville_cohort_jq_parameters.csv"
POINTS_PATH = OUTPUT_DIR / "phrag_brownsville_cohort_points.csv"

LEAF_METRICS_OUT = OUTPUT_DIR / "phrag_brownsville_tier_a_leaf_metrics.csv"
CONTRASTS_OUT = OUTPUT_DIR / "phrag_brownsville_tier_a_zone_contrasts.csv"
SURFACE_OUT = OUTPUT_DIR / "phrag_brownsville_tier_a_standardized_surface.csv"
AMBIENT_OUT = OUTPUT_DIR / "phrag_brownsville_tier_a_ambient_exchange.csv"
FIGURE_OUT = OUTPUT_DIR / "phrag_brownsville_tier_a_empirical_summary.png"
REPORT_OUT = OUTPUT_DIR / "phrag_brownsville_tier_a_report.md"

Q_LEVELS = np.array([50, 100, 200, 350, 600, 1000, 1400, 1700, 2000], dtype=float)
CI_GRID = np.arange(400.0, 1200.1, 20.0)
ZONE_COLORS = {"High": "#2878a5", "Low": "#d97728"}


def load_inputs() -> tuple[pd.DataFrame, ...]:
    qc = pd.read_csv(QC_PATH)
    curves = pd.read_csv(CURVES_PATH)
    q_summary = pd.read_csv(Q_SUMMARY_PATH)
    fluorescence = pd.read_csv(FLUORESCENCE_PATH)
    jq = pd.read_csv(JQ_PATH)
    return qc, curves, q_summary, fluorescence, jq


def interpolate_surface(curves: pd.DataFrame, valid_codes: set[str]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for (run_code, q_level), group in curves.loc[curves["run_code"].isin(valid_codes)].groupby(
        ["run_code", "Q_level"]
    ):
        clean = (
            group[["Ci", "A"]]
            .dropna()
            .groupby("Ci", as_index=False)["A"]
            .median()
            .sort_values("Ci")
        )
        if len(clean) < 2:
            continue
        x = clean["Ci"].to_numpy(dtype=float)
        y = clean["A"].to_numpy(dtype=float)
        supported = CI_GRID[(CI_GRID >= x.min()) & (CI_GRID <= x.max())]
        for ci, assimilation in zip(supported, np.interp(supported, x, y), strict=True):
            rows.append(
                {
                    "run_code": run_code,
                    "Q_level": float(q_level),
                    "Ci": float(ci),
                    "A": float(assimilation),
                }
            )
    return pd.DataFrame(rows)


def surface_mean(surface: pd.DataFrame) -> pd.Series:
    domain = surface.loc[surface["Q_level"].between(200, 2000) & surface["Ci"].between(400, 1200)]
    per_q = (
        domain.groupby(["run_code", "Q_level"])
        .apply(lambda frame: np.trapezoid(frame["A"], frame["Ci"]) / 800.0, include_groups=False)
        .rename("mean_A_over_Ci")
        .reset_index()
    )
    return per_q.groupby("run_code").apply(
        lambda frame: np.trapezoid(frame["mean_A_over_Ci"], frame["Q_level"]) / 1800.0,
        include_groups=False,
    )


def ambient_exchange(valid_codes: set[str]) -> pd.DataFrame:
    columns = ["run_code", "Q_level", "clean_ramp", "Ca", "A", "E", "gsw"]
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(POINTS_PATH, usecols=columns, chunksize=150_000):
        selected = chunk.loc[
            chunk["run_code"].isin(valid_codes)
            & chunk["clean_ramp"].fillna(False)
            & chunk["Ca"].between(400, 440)
            & chunk["A"].notna()
            & chunk["E"].gt(0)
            & chunk["gsw"].gt(0)
        ]
        if not selected.empty:
            pieces.append(selected)
    data = pd.concat(pieces, ignore_index=True)
    summary = (
        data.groupby(["run_code", "Q_level"])
        .agg(n=("A", "size"), Ca=("Ca", "median"), A=("A", "median"), E=("E", "median"), gsw=("gsw", "median"))
        .reset_index()
    )
    # LI-6800 E is in mol H2O m-2 s-1; convert the denominator to mmol.
    summary["WUE_A_over_E"] = summary["A"] / (summary["E"] * 1000.0)
    summary["iWUE_A_over_gsw"] = summary["A"] / summary["gsw"]
    return summary


def value_at_q(frame: pd.DataFrame, column: str, q_level: float) -> pd.Series:
    return frame.loc[frame["Q_level"].eq(q_level)].set_index("run_code")[column]


def build_leaf_metrics(
    qc: pd.DataFrame,
    q_summary: pd.DataFrame,
    fluorescence: pd.DataFrame,
    jq: pd.DataFrame,
    surface: pd.DataFrame,
    ambient: pd.DataFrame,
) -> pd.DataFrame:
    valid_qc = qc.loc[~qc["suspect_run"]].copy()
    valid_qc["analysis_group"] = np.where(
        valid_qc["primary_comparison"], "sun_primary", "shade_pilot"
    )
    valid_qc.loc[valid_qc["run_code"].eq("P02"), "analysis_group"] = "sun_primary_caution"
    metrics = valid_qc[
        [
            "run_code",
            "zone",
            "illum",
            "analysis_group",
            "status_class",
            "start_datetime",
            "dark_FvFm",
            "VPDleaf_baseline_median",
            "Tleaf_baseline_median",
        ]
    ].set_index("run_code")

    peak_rows = q_summary.loc[q_summary.groupby("run_code")["A_curve_max"].idxmax()].set_index("run_code")
    metrics["Aobs_domain_max"] = peak_rows["A_curve_max"]
    metrics["Q_at_Aobs_domain_max"] = peak_rows["Q_level"]
    metrics["Ci_max_at_peak_Q"] = peak_rows["Ci_max"]
    metrics["surface_mean_A_Q200_2000_Ci400_1200"] = surface_mean(surface)

    for q_level in [1000.0, 1400.0, 2000.0]:
        suffix = int(q_level)
        for source, column, label in [
            (q_summary, "A_at_Ci800", "A_Ci800"),
            (q_summary, "A_at_Ci1200", "A_Ci1200"),
            (q_summary, "A_at_Ca420", "A_Ca420"),
            (q_summary, "gsw_low_Ca", "gsw_lowCa"),
            (q_summary, "gsw_at_Ca420", "gsw_Ca420"),
        ]:
            metrics[f"{label}_Q{suffix}"] = value_at_q(source, column, q_level)
        metrics[f"WUE_A_E_Ca420_Q{suffix}"] = value_at_q(ambient, "WUE_A_over_E", q_level)
        metrics[f"iWUE_A_gsw_Ca420_Q{suffix}"] = value_at_q(ambient, "iWUE_A_over_gsw", q_level)

    flr2000 = fluorescence.loc[fluorescence["Q_level"].eq(2000)].set_index("run_code")
    for column, output_name in [
        ("ETR_at_Ci800", "ETR_Ci800_Q2000"),
        ("PhiPS2_at_Ci800", "PhiPS2_Ci800_Q2000"),
        ("NPQ_at_Ci800", "NPQ_Ci800_Q2000"),
        ("qL_at_Ci800", "qL_Ci800_Q2000"),
    ]:
        metrics[output_name] = flr2000[column]

    metrics = metrics.join(
        jq.set_index("run_code")[["alpha_ETR", "Jmax_ETR", "theta_ETR", "rmse_ETR"]]
    )
    metrics["A_over_ETR_Ci800_Q2000"] = metrics["A_Ci800_Q2000"] / metrics["ETR_Ci800_Q2000"]
    return metrics.reset_index()


def bootstrap_contrasts(metrics: pd.DataFrame, draws: int = 40_000) -> pd.DataFrame:
    rng = np.random.default_rng(20260803)
    primary = metrics.loc[metrics["analysis_group"].str.startswith("sun_primary")].copy()
    metric_columns = [
        "Aobs_domain_max",
        "surface_mean_A_Q200_2000_Ci400_1200",
        "A_Ci800_Q2000",
        "A_Ci1200_Q2000",
        "A_Ca420_Q2000",
        "gsw_lowCa_Q2000",
        "gsw_Ca420_Q2000",
        "WUE_A_E_Ca420_Q2000",
        "iWUE_A_gsw_Ca420_Q2000",
        "dark_FvFm",
        "Jmax_ETR",
        "theta_ETR",
        "ETR_Ci800_Q2000",
        "NPQ_Ci800_Q2000",
        "A_over_ETR_Ci800_Q2000",
    ]
    rows: list[dict[str, float | int | str]] = []
    for sensitivity, frame in [
        ("all_primary", primary),
        ("exclude_P02", primary.loc[~primary["run_code"].eq("P02")]),
    ]:
        for metric in metric_columns:
            high = frame.loc[frame["zone"].eq("High"), metric].dropna().to_numpy(dtype=float)
            low = frame.loc[frame["zone"].eq("Low"), metric].dropna().to_numpy(dtype=float)
            if len(high) < 2 or len(low) < 2:
                continue
            boot_high = np.median(rng.choice(high, size=(draws, len(high)), replace=True), axis=1)
            boot_low = np.median(rng.choice(low, size=(draws, len(low)), replace=True), axis=1)
            delta = boot_high - boot_low
            rows.append(
                {
                    "sensitivity": sensitivity,
                    "metric": metric,
                    "n_high": len(high),
                    "n_low": len(low),
                    "high_median": np.median(high),
                    "low_median": np.median(low),
                    "high_minus_low": np.median(high) - np.median(low),
                    "bootstrap_delta_q025": np.quantile(delta, 0.025),
                    "bootstrap_delta_q975": np.quantile(delta, 0.975),
                    "probability_random_high_gt_low": np.mean(high[:, None] > low[None, :]),
                }
            )
    return pd.DataFrame(rows)


def add_zone_metadata(frame: pd.DataFrame, qc: pd.DataFrame) -> pd.DataFrame:
    return frame.merge(qc[["run_code", "zone", "primary_comparison", "suspect_run"]], on="run_code")


def plot_summary(
    metrics: pd.DataFrame,
    q_summary: pd.DataFrame,
    fluorescence: pd.DataFrame,
    surface: pd.DataFrame,
) -> None:
    primary = metrics.loc[metrics["analysis_group"].str.startswith("sun_primary")].copy()
    primary_codes = set(primary["run_code"])
    qplot = q_summary.loc[q_summary["run_code"].isin(primary_codes)].merge(
        primary[["run_code", "zone"]], on="run_code", suffixes=("", "_metric")
    )
    fplot = fluorescence.loc[
        fluorescence["run_code"].isin(primary_codes) & fluorescence["fluorescence_qc_ok"]
    ].merge(primary[["run_code", "zone"]], on="run_code", suffixes=("", "_metric"))

    sns.set_theme(style="whitegrid", context="talk", font_scale=0.78)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5))

    for zone, color in ZONE_COLORS.items():
        zone_data = qplot.loc[qplot["zone"].eq(zone)]
        for _, leaf in zone_data.groupby("run_code"):
            axes[0, 0].plot(leaf["Q_level"], leaf["A_at_Ci800"], color=color, alpha=0.28, lw=1.2)
            axes[1, 0].plot(leaf["Q_level"], leaf["gsw_low_Ca"], color=color, alpha=0.28, lw=1.2)
        med = zone_data.groupby("Q_level", as_index=False).median(numeric_only=True)
        axes[0, 0].plot(med["Q_level"], med["A_at_Ci800"], color=color, marker="o", lw=3, label=zone)
        axes[1, 0].plot(med["Q_level"], med["gsw_low_Ca"], color=color, marker="o", lw=3, label=zone)

        zone_f = fplot.loc[fplot["zone"].eq(zone)]
        for _, leaf in zone_f.groupby("run_code"):
            axes[1, 1].plot(leaf["Q_level"], leaf["ETR_at_Ci800"], color=color, alpha=0.28, lw=1.2)
        med_f = zone_f.groupby("Q_level", as_index=False).median(numeric_only=True)
        axes[1, 1].plot(med_f["Q_level"], med_f["ETR_at_Ci800"], color=color, marker="o", lw=3)

    axes[0, 0].set(title="A at common Ci = 800", xlabel="Q", ylabel="A")
    axes[0, 0].legend(title="Forest position", frameon=False)
    axes[1, 0].set(title="Conductance during low-CO2 park", xlabel="Q", ylabel="gsw")
    axes[1, 1].set(title="Fluorescence electron-transport proxy", xlabel="Q", ylabel="ETR at Ci = 800")

    for ax, metric, title, ylabel in [
        (axes[0, 1], "Aobs_domain_max", "Observed-domain maximum A", "Maximum observed A"),
        (axes[0, 2], "surface_mean_A_Q200_2000_Ci400_1200", "Mean response over common Q/Ci domain", "Mean A"),
        (axes[1, 2], "NPQ_Ci800_Q2000", "Photoprotective heat dissipation at high light", "NPQ at Q=2000, Ci=800"),
    ]:
        sns.stripplot(
            data=primary,
            x="zone",
            y=metric,
            order=["High", "Low"],
            palette=ZONE_COLORS,
            hue="zone",
            hue_order=["High", "Low"],
            jitter=0.10,
            size=8,
            edgecolor="white",
            linewidth=0.8,
            legend=False,
            ax=ax,
        )
        sns.boxplot(
            data=primary,
            x="zone",
            y=metric,
            order=["High", "Low"],
            color="white",
            width=0.48,
            showfliers=False,
            boxprops={"facecolor": "none", "linewidth": 1.2},
            whiskerprops={"linewidth": 1.2},
            medianprops={"color": "#222222", "linewidth": 1.6},
            ax=ax,
        )
        for _, row in primary.dropna(subset=[metric]).iterrows():
            ax.annotate(
                row["run_code"],
                (["High", "Low"].index(row["zone"]), row[metric]),
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=8,
                va="center",
                color="#444444",
            )
        ax.set(title=title, xlabel="", ylabel=ylabel)

    fig.suptitle("Brownsville Phragmites: Tier A Empirical Results", x=0.055, ha="left", fontsize=22, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.0, w_pad=1.7)
    fig.savefig(FIGURE_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(metrics: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    primary = metrics.loc[metrics["analysis_group"].str.startswith("sun_primary")]
    all_contrasts = contrasts.loc[contrasts["sensitivity"].eq("all_primary")].set_index("metric")

    def contrast_line(metric: str, digits: int = 2) -> str:
        row = all_contrasts.loc[metric]
        return (
            f"High median {row.high_median:.{digits}f}; low median {row.low_median:.{digits}f}; "
            f"high-low {row.high_minus_low:.{digits}f}; leaf-bootstrap 95% interval "
            f"[{row.bootstrap_delta_q025:.{digits}f}, {row.bootstrap_delta_q975:.{digits}f}]."
        )

    strongest = primary.loc[primary["Aobs_domain_max"].idxmax()]
    report = f"""# Tier A: empirical Brownsville Phragmites results

## Scope

- Primary comparison: {primary['zone'].eq('High').sum()} high-forest sun leaves and {primary['zone'].eq('Low').sum()} low-forest sun leaves.
- P02 is retained with caution; the contrast table also contains an `exclude_P02` sensitivity analysis.
- Shade leaves are retained as pilots but are not used for the high-versus-low sun comparison.
- P05, P09, and P15 remain quarantined.

## Main results

- Observed-domain maximum A: {contrast_line('Aobs_domain_max')}
- Mean A over Q=200-2000 and Ci=400-1200: {contrast_line('surface_mean_A_Q200_2000_Ci400_1200')}
- A at Q=2000 and Ci=800: {contrast_line('A_Ci800_Q2000')}
- Low-CO2-park gsw at Q=2000: {contrast_line('gsw_lowCa_Q2000', 3)}
- Dark-adapted Fv/Fm: {contrast_line('dark_FvFm', 3)}
- NPQ at Q=2000 and Ci=800: {contrast_line('NPQ_Ci800_Q2000', 3)}

The strongest valid sun leaf was {strongest.run_code} ({strongest.zone}), with an observed-domain maximum A of {strongest.Aobs_domain_max:.2f}.

## Interpretation boundary

These are leaf-level descriptive contrasts. The bootstrap intervals quantify instability under resampling of the measured leaves; they do not remove day, collection, or order confounding. The observed-domain maximum may occur at elevated Ci and is not synonymous with field Amax at ambient CO2.
"""
    REPORT_OUT.write_text(report, encoding="utf-8")


def main() -> None:
    qc, curves, q_summary, fluorescence, jq = load_inputs()
    valid_codes = set(qc.loc[~qc["suspect_run"], "run_code"])
    surface = interpolate_surface(curves, valid_codes)
    ambient = ambient_exchange(valid_codes)
    metrics = build_leaf_metrics(qc, q_summary, fluorescence, jq, surface, ambient)
    contrasts = bootstrap_contrasts(metrics)

    surface = add_zone_metadata(surface, qc)
    ambient = add_zone_metadata(ambient, qc)
    metrics.to_csv(LEAF_METRICS_OUT, index=False)
    contrasts.to_csv(CONTRASTS_OUT, index=False)
    surface.to_csv(SURFACE_OUT, index=False)
    ambient.to_csv(AMBIENT_OUT, index=False)
    plot_summary(metrics, q_summary, fluorescence, surface)
    write_report(metrics, contrasts)

    print(f"Wrote {LEAF_METRICS_OUT}")
    print(f"Wrote {CONTRASTS_OUT}")
    print(f"Wrote {SURFACE_OUT}")
    print(f"Wrote {AMBIENT_OUT}")
    print(f"Wrote {FIGURE_OUT}")
    print(f"Wrote {REPORT_OUT}")
    print("\nPrimary contrasts:")
    print(
        contrasts.loc[
            contrasts["sensitivity"].eq("all_primary"),
            [
                "metric",
                "n_high",
                "n_low",
                "high_median",
                "low_median",
                "high_minus_low",
                "bootstrap_delta_q025",
                "bootstrap_delta_q975",
                "probability_random_high_gt_low",
            ],
        ].round(4).to_string(index=False)
    )


if __name__ == "__main__":
    main()
