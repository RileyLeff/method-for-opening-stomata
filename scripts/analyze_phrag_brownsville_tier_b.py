from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import least_squares


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

QC_PATH = OUTPUT_DIR / "phrag_brownsville_cohort_run_qc.csv"
CURVES_PATH = OUTPUT_DIR / "phrag_brownsville_cohort_binned_curves.csv"
FLASHES_PATH = OUTPUT_DIR / "phrag_brownsville_cohort_flashes.csv"
JQ_PATH = OUTPUT_DIR / "phrag_brownsville_cohort_jq_parameters.csv"
POINTS_PATH = OUTPUT_DIR / "phrag_brownsville_cohort_points.csv"
PLANTECO_PATH = OUTPUT_DIR / "phrag_brownsville_tier_b_plantecophys_q2000.csv"

FITS_OUT = OUTPUT_DIR / "phrag_brownsville_tier_b_joint_fvcb_fits.csv"
PREDICTIONS_OUT = OUTPUT_DIR / "phrag_brownsville_tier_b_joint_fvcb_predictions.csv"
GM_OUT = OUTPUT_DIR / "phrag_brownsville_tier_b_variable_j_gm.csv"
CONTRASTS_OUT = OUTPUT_DIR / "phrag_brownsville_tier_b_parameter_contrasts.csv"
FIGURE_OUT = OUTPUT_DIR / "phrag_brownsville_tier_b_parameter_summary.png"
REPORT_OUT = OUTPUT_DIR / "phrag_brownsville_tier_b_report.md"

GAMMA_STAR = 42.75
KC = 404.9
KO = 278_400.0
OXYGEN = 210_000.0
KM = KC * (1.0 + OXYGEN / KO)

ZONE_COLORS = {"High": "#2878a5", "Low": "#d97728"}
PRIMARY_VARIANT = "joint_appCi_noTPU"


def electron_transport(q: np.ndarray, jmax: float, alpha: float, theta: float) -> np.ndarray:
    light_limited = alpha * q
    discriminant = np.maximum((light_limited + jmax) ** 2 - 4.0 * theta * light_limited * jmax, 0.0)
    return (light_limited + jmax - np.sqrt(discriminant)) / (2.0 * theta)


def fvcb_rates(
    ci: np.ndarray,
    q: np.ndarray,
    vcmax: float,
    jmax: float,
    rd: float,
    alpha: float,
    theta: float,
    tpu: float | None = None,
) -> tuple[np.ndarray, ...]:
    ac = vcmax * (ci - GAMMA_STAR) / (ci + KM)
    j = electron_transport(q, jmax, alpha, theta)
    aj = j * (ci - GAMMA_STAR) / (4.0 * ci + 8.0 * GAMMA_STAR)
    if tpu is None:
        ap = np.full_like(ac, np.inf)
    else:
        ap = np.full_like(ac, 3.0 * tpu)
    gross = np.minimum(np.minimum(ac, aj), ap)
    return gross - rd, ac, aj, ap, j


def parameter_spec(variant: str, jq_row: pd.Series | None) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    if variant == "joint_appCi_TPU":
        names = ["Vcmax25", "Jmax25", "Rd", "alpha_J", "theta_J", "TPU"]
        x0 = np.array([70.0, 115.0, 1.5, 0.25, 0.72, 8.0])
        lower = np.array([2.0, 5.0, 0.0, 0.03, 0.10, 0.3])
        upper = np.array([250.0, 400.0, 8.0, 0.60, 0.99, 35.0])
    elif variant == "joint_appCi_fluorescence_shape":
        if jq_row is None:
            raise ValueError("Fluorescence-shape fit requires a J(Q) row")
        names = ["Vcmax25", "Jmax25", "Rd"]
        x0 = np.array([70.0, float(jq_row["Jmax_ETR"]), 1.5])
        lower = np.array([2.0, 5.0, 0.0])
        upper = np.array([250.0, 400.0, 8.0])
    else:
        names = ["Vcmax25", "Jmax25", "Rd", "alpha_J", "theta_J"]
        x0 = np.array([70.0, 115.0, 1.5, 0.25, 0.72])
        lower = np.array([2.0, 5.0, 0.0, 0.03, 0.10])
        upper = np.array([250.0, 400.0, 8.0, 0.60, 0.99])
    return names, x0, lower, upper


def unpack_parameters(
    values: np.ndarray,
    variant: str,
    jq_row: pd.Series | None,
) -> tuple[float, float, float, float, float, float | None]:
    if variant == "joint_appCi_TPU":
        vcmax, jmax, rd, alpha, theta, tpu = values
    elif variant == "joint_appCi_fluorescence_shape":
        vcmax, jmax, rd = values
        assert jq_row is not None
        alpha = float(jq_row["alpha_ETR"])
        theta = float(jq_row["theta_ETR"])
        tpu = None
    else:
        vcmax, jmax, rd, alpha, theta = values
        tpu = None
    return float(vcmax), float(jmax), float(rd), float(alpha), float(theta), None if tpu is None else float(tpu)


def fit_one(
    run_code: str,
    data: pd.DataFrame,
    variant: str,
    jq_row: pd.Series | None,
    rng: np.random.Generator,
) -> tuple[dict[str, object], pd.DataFrame]:
    names, x0, lower, upper = parameter_spec(variant, jq_row)
    ci_observed = data["Ci"].to_numpy(dtype=float)
    assimilation = data["A"].to_numpy(dtype=float)
    q = data["Q_level"].to_numpy(dtype=float)
    if variant == "joint_fixed_gm027_noTPU":
        ci_model = ci_observed - assimilation / 0.27
    else:
        ci_model = ci_observed
    sane = np.isfinite(ci_model) & (ci_model > GAMMA_STAR + 2.0)
    ci_model = ci_model[sane]
    ci_observed = ci_observed[sane]
    assimilation = assimilation[sane]
    q = q[sane]

    def prediction(values: np.ndarray) -> np.ndarray:
        vcmax, jmax, rd, alpha, theta, tpu = unpack_parameters(values, variant, jq_row)
        return fvcb_rates(ci_model, q, vcmax, jmax, rd, alpha, theta, tpu)[0]

    starts = [x0]
    for _ in range(11):
        starts.append(lower + rng.uniform(size=len(x0)) * (upper - lower))

    results = []
    for start in starts:
        result = least_squares(
            lambda values: prediction(values) - assimilation,
            start,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=0.6,
            x_scale="jac",
            max_nfev=50_000,
        )
        results.append(result)
    result = min(results, key=lambda item: item.cost)
    fitted = prediction(result.x)
    residual = assimilation - fitted
    total = np.sum((assimilation - assimilation.mean()) ** 2)
    r_squared = 1.0 - np.sum(residual**2) / total if total > 0 else np.nan

    vcmax, jmax, rd, alpha, theta, tpu = unpack_parameters(result.x, variant, jq_row)
    _, ac, aj, ap, j = fvcb_rates(ci_model, q, vcmax, jmax, rd, alpha, theta, tpu)
    rate_stack = np.vstack([ac, aj, ap])
    limitation_index = np.argmin(rate_stack, axis=0)
    limitation = np.array(["Ac", "Aj", "Ap"])[limitation_index]

    degrees_freedom = max(len(assimilation) - len(result.x), 1)
    residual_variance = np.sum(residual**2) / degrees_freedom
    try:
        covariance = np.linalg.pinv(result.jac.T @ result.jac) * residual_variance
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        scale = np.maximum(np.abs(result.x), 0.1)
        scaled_condition = float(np.linalg.cond(result.jac * scale[None, :]))
        standard = np.sqrt(np.maximum(np.diag(covariance), 1e-20))
        correlation = covariance / standard[:, None] / standard[None, :]
        max_correlation = float(np.max(np.abs(correlation - np.eye(len(result.x)))))
    except np.linalg.LinAlgError:
        standard_errors = np.full(len(result.x), np.nan)
        scaled_condition = np.inf
        max_correlation = np.nan

    near_bound = any(
        (value - lo) / (hi - lo) < 0.01 or (hi - value) / (hi - lo) < 0.01
        for value, lo, hi in zip(result.x, lower, upper, strict=True)
    )
    summary: dict[str, object] = {
        "run_code": run_code,
        "variant": variant,
        "n_points": len(assimilation),
        "Vcmax25": vcmax,
        "Jmax25": jmax,
        "Rd": rd,
        "alpha_J": alpha,
        "theta_J": theta,
        "TPU": tpu,
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "R2": float(r_squared),
        "scaled_jacobian_condition": scaled_condition,
        "max_abs_parameter_correlation": max_correlation,
        "near_parameter_bound": near_bound,
        "converged": bool(result.success),
        "fraction_Ac_limited": float(np.mean(limitation == "Ac")),
        "fraction_Aj_limited": float(np.mean(limitation == "Aj")),
        "fraction_Ap_limited": float(np.mean(limitation == "Ap")),
    }
    for name, standard_error in zip(names, standard_errors, strict=True):
        summary[f"{name}_se_local"] = float(standard_error)

    predictions = pd.DataFrame(
        {
            "run_code": run_code,
            "variant": variant,
            "Q_level": q,
            "Ci_observed": ci_observed,
            "C_model": ci_model,
            "A_observed": assimilation,
            "A_model": fitted,
            "Ac_gross": ac,
            "Aj_gross": aj,
            "Ap_gross": ap,
            "J": j,
            "limitation": limitation,
        }
    )
    return summary, predictions


def dark_respiration(valid_codes: set[str]) -> pd.Series:
    columns = ["run_code", "minute", "Qin", "A", "mpf_or_recovery"]
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(POINTS_PATH, usecols=columns, chunksize=150_000):
        selected = chunk.loc[
            chunk["run_code"].isin(valid_codes)
            & chunk["Qin"].lt(1.0)
            & chunk["minute"].between(5.0, 7.5)
            & ~chunk["mpf_or_recovery"].fillna(False)
        ]
        if not selected.empty:
            pieces.append(selected)
    data = pd.concat(pieces, ignore_index=True)
    return -data.groupby("run_code")["A"].median()


def fit_all(
    curves: pd.DataFrame,
    qc: pd.DataFrame,
    jq: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20260803)
    valid_codes = qc.loc[~qc["suspect_run"], "run_code"].tolist()
    jq_index = jq.set_index("run_code")
    summaries: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    variants = ["joint_appCi_noTPU", "joint_fixed_gm027_noTPU", "joint_appCi_TPU"]

    for code in valid_codes:
        data = curves.loc[
            curves["run_code"].eq(code)
            & curves["Ci"].between(45, 1400)
            & curves["A"].between(-8, 40)
            & curves["Q_level"].isin([50, 100, 200, 350, 600, 1000, 1400, 1700, 2000])
        ].copy()
        jq_row = jq_index.loc[code] if code in jq_index.index else None
        run_variants = variants + (["joint_appCi_fluorescence_shape"] if jq_row is not None else [])
        for variant in run_variants:
            summary, prediction = fit_one(code, data, variant, jq_row, rng)
            summaries.append(summary)
            predictions.append(prediction)

    summary_frame = pd.DataFrame(summaries)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    rdark = dark_respiration(set(valid_codes))
    summary_frame["Rdark_observed"] = summary_frame["run_code"].map(rdark)
    metadata = qc.set_index("run_code")[["zone", "illum", "primary_comparison", "status_class"]]
    summary_frame = summary_frame.join(metadata, on="run_code")
    prediction_frame = prediction_frame.join(metadata[["zone", "illum"]], on="run_code")
    return summary_frame, prediction_frame


def variable_j_gm(
    flashes: pd.DataFrame,
    fits: pd.DataFrame,
    fluorescence_qc_codes: set[str],
) -> pd.DataFrame:
    fit_rd = fits.loc[fits["variant"].eq(PRIMARY_VARIANT)].set_index("run_code")["Rd"]
    selected = flashes.loc[
        flashes["standard_flash"].fillna(False)
        & flashes["run_code"].isin(fit_rd.index)
        & flashes["gas_Ci"].between(150, 1200)
        & flashes["assigned_Q"].ge(600)
        & flashes["ETR"].gt(10)
        & flashes["gas_A"].gt(0)
    ].copy()
    rows: list[dict[str, object]] = []
    for scale in [0.8, 1.0, 1.2]:
        data = selected.copy()
        data["Rd"] = data["run_code"].map(fit_rd)
        data["J_scaled"] = data["ETR"] * scale
        gross_a = data["gas_A"] + data["Rd"]
        denominator = data["J_scaled"] - 4.0 * gross_a
        data["Cc_variable_J"] = GAMMA_STAR * (data["J_scaled"] + 8.0 * gross_a) / denominator
        data["gm_variable_J"] = data["gas_A"] / (data["gas_Ci"] - data["Cc_variable_J"])
        data = data.loc[
            denominator.gt(2.0)
            & data["Cc_variable_J"].gt(GAMMA_STAR)
            & data["Cc_variable_J"].lt(data["gas_Ci"] - 2.0)
            & data["gm_variable_J"].between(0.01, 2.0)
        ]
        for code, group in data.groupby("run_code"):
            rows.append(
                {
                    "run_code": code,
                    "ETR_scale": scale,
                    "n_valid_flashes": len(group),
                    "gm_median": group["gm_variable_J"].median(),
                    "gm_q25": group["gm_variable_J"].quantile(0.25),
                    "gm_q75": group["gm_variable_J"].quantile(0.75),
                    "Cc_median": group["Cc_variable_J"].median(),
                    "fluorescence_qc_ok": code in fluorescence_qc_codes,
                }
            )
    result = pd.DataFrame(rows)
    metadata = fits.drop_duplicates("run_code").set_index("run_code")[["zone", "illum", "primary_comparison"]]
    return result.join(metadata, on="run_code")


def parameter_contrasts(fits: pd.DataFrame, jq: pd.DataFrame) -> pd.DataFrame:
    primary = fits.loc[
        fits["variant"].eq(PRIMARY_VARIANT) & fits["primary_comparison"]
    ].copy()
    combined = primary.merge(
        jq[["run_code", "Jmax_ETR", "alpha_ETR", "theta_ETR"]],
        on="run_code",
        how="left",
    )
    parameters = ["Vcmax25", "Jmax25", "Rd", "alpha_J", "theta_J", "Jmax_ETR", "alpha_ETR", "theta_ETR"]
    rows: list[dict[str, object]] = []
    for sensitivity, frame in [
        ("all_primary", combined),
        ("exclude_P02", combined.loc[~combined["run_code"].eq("P02")]),
    ]:
        for parameter in parameters:
            high = frame.loc[frame["zone"].eq("High"), parameter].dropna().to_numpy(dtype=float)
            low = frame.loc[frame["zone"].eq("Low"), parameter].dropna().to_numpy(dtype=float)
            if len(high) < 2 or len(low) < 2:
                continue
            rows.append(
                {
                    "sensitivity": sensitivity,
                    "parameter": parameter,
                    "n_high": len(high),
                    "n_low": len(low),
                    "high_median": np.median(high),
                    "low_median": np.median(low),
                    "high_minus_low": np.median(high) - np.median(low),
                    "probability_random_high_gt_low": np.mean(high[:, None] > low[None, :]),
                }
            )
    return pd.DataFrame(rows)


def plot_results(
    fits: pd.DataFrame,
    predictions: pd.DataFrame,
    gm: pd.DataFrame,
    planteco: pd.DataFrame,
) -> None:
    primary = fits.loc[fits["variant"].eq(PRIMARY_VARIANT) & fits["primary_comparison"]].copy()
    standard = planteco.merge(
        primary[["run_code", "zone", "primary_comparison", "Vcmax25"]], on="run_code", how="inner"
    )
    pred = predictions.loc[
        predictions["variant"].eq(PRIMARY_VARIANT) & predictions["run_code"].isin(primary["run_code"])
    ]
    gm_main = gm.loc[
        gm["ETR_scale"].eq(1.0)
        & gm["primary_comparison"]
        & gm["fluorescence_qc_ok"]
        & gm["n_valid_flashes"].ge(5)
    ].copy()

    sns.set_theme(style="whitegrid", context="talk", font_scale=0.76)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5))
    parameter_panels = [
        (axes[0, 0], "Vcmax25", "Joint-surface apparent Vcmax25", 84.0),
        (axes[0, 1], "Jmax25", "Joint-surface apparent Jmax25", 116.0),
        (axes[0, 2], "theta_J", "Gas-exchange J(Q) curvature", None),
    ]
    for ax, parameter, title, literature in parameter_panels:
        sns.stripplot(
            data=primary,
            x="zone",
            y=parameter,
            order=["High", "Low"],
            hue="zone",
            palette=ZONE_COLORS,
            jitter=0.10,
            size=8,
            edgecolor="white",
            linewidth=0.8,
            legend=False,
            ax=ax,
        )
        for _, row in primary.iterrows():
            ax.annotate(row["run_code"], (["High", "Low"].index(row["zone"]), row[parameter]), xytext=(6, 0), textcoords="offset points", fontsize=8, va="center")
        if literature is not None:
            ax.axhline(literature, color="#555555", ls="--", lw=1.2, label="Haplotype M literature")
            ax.legend(frameon=False, fontsize=8)
        ax.set(title=title, xlabel="", ylabel=parameter)

    axes[1, 0].scatter(
        standard["Vcmax25_app_Ci"],
        standard["Vcmax25"],
        c=standard["zone"].map(ZONE_COLORS),
        s=65,
        edgecolor="white",
    )
    finite = standard[["Vcmax25_app_Ci", "Vcmax25"]].to_numpy(dtype=float)
    max_value = np.nanmax(finite) * 1.08
    axes[1, 0].plot([0, max_value], [0, max_value], color="#555555", ls="--", lw=1.2)
    for _, row in standard.iterrows():
        axes[1, 0].annotate(row["run_code"], (row["Vcmax25_app_Ci"], row["Vcmax25"]), xytext=(5, 3), textcoords="offset points", fontsize=8)
    axes[1, 0].set(title="Conventional Q2000 versus joint-surface fit", xlabel="plantecophys Vcmax25", ylabel="Joint Vcmax25")

    if not gm_main.empty:
        sns.stripplot(
            data=gm_main,
            x="zone",
            y="gm_median",
            order=["High", "Low"],
            hue="zone",
            palette=ZONE_COLORS,
            jitter=0.10,
            size=8,
            edgecolor="white",
            linewidth=0.8,
            legend=False,
            ax=axes[1, 1],
        )
        for _, row in gm_main.iterrows():
            axes[1, 1].annotate(row["run_code"], (["High", "Low"].index(row["zone"]), row["gm_median"]), xytext=(6, 0), textcoords="offset points", fontsize=8, va="center")
    axes[1, 1].axhline(0.27, color="#555555", ls="--", lw=1.2)
    axes[1, 1].set(title="Variable-J gm diagnostic", xlabel="", ylabel="gm (mol m-2 s-1)")

    for zone, color in ZONE_COLORS.items():
        zone_pred = pred.loc[pred["zone"].eq(zone)]
        axes[1, 2].scatter(zone_pred["A_observed"], zone_pred["A_model"], color=color, alpha=0.25, s=12, label=zone)
    limit_min = min(pred["A_observed"].min(), pred["A_model"].min())
    limit_max = max(pred["A_observed"].max(), pred["A_model"].max())
    axes[1, 2].plot([limit_min, limit_max], [limit_min, limit_max], color="#333333", ls="--", lw=1.2)
    axes[1, 2].set(title="Joint FvCB target fit", xlabel="Observed A", ylabel="Modeled A")
    axes[1, 2].legend(frameon=False)

    fig.suptitle("Brownsville Phragmites: Tier B Physiological Fits", x=0.055, ha="left", fontsize=22, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.0, w_pad=1.7)
    fig.savefig(FIGURE_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(
    fits: pd.DataFrame,
    contrasts: pd.DataFrame,
    gm: pd.DataFrame,
    planteco: pd.DataFrame,
) -> None:
    primary = fits.loc[fits["variant"].eq(PRIMARY_VARIANT) & fits["primary_comparison"]]
    contrast = contrasts.loc[contrasts["sensitivity"].eq("all_primary")].set_index("parameter")
    standard_primary = planteco.loc[planteco["run_code"].isin(primary["run_code"])]
    default_count = standard_primary["fitmethod"].eq("default").sum()
    gm_main = gm.loc[
        gm["ETR_scale"].eq(1.0)
        & gm["primary_comparison"]
        & gm["fluorescence_qc_ok"]
        & gm["n_valid_flashes"].ge(5)
    ]
    theta_bound_count = int(primary["theta_J"].le(0.105).sum())
    tpu_fits = fits.loc[fits["variant"].eq("joint_appCi_TPU") & fits["primary_comparison"]]
    tpu_limited_count = int(tpu_fits["fraction_Ap_limited"].gt(0).sum())
    rd_comparable = primary.dropna(subset=["Rdark_observed"])
    rd_above_dark_count = int(rd_comparable["Rd"].gt(rd_comparable["Rdark_observed"]).sum())

    def line(parameter: str, digits: int = 2) -> str:
        row = contrast.loc[parameter]
        return (
            f"high median {row.high_median:.{digits}f}, low median {row.low_median:.{digits}f}, "
            f"high-low {row.high_minus_low:.{digits}f}"
        )

    report = f"""# Tier B: physiological-model results

## Conventional benchmark

`plantecophys::fitaci` fit all {len(standard_primary)} valid sun leaves at Q=2000, but only {default_count} used the default transition fit; the remainder required its bilinear fallback. Fits with a transition pinned near Ci=2000 should be treated as weakly partitioned rather than clean Vcmax/Jmax estimates.

## Joint multi-Q FvCB fit

- Apparent Vcmax25: {line('Vcmax25')} umol m-2 s-1.
- Apparent Jmax25: {line('Jmax25')} umol m-2 s-1.
- Gas-exchange theta: {line('theta_J', 3)}; {theta_bound_count}/{len(primary)} leaves reached the lower bound.
- Fluorescence ETRmax proxy: {line('Jmax_ETR')} umol electrons m-2 s-1.
- Fluorescence theta proxy: {line('theta_ETR', 3)}.

The TPU variant selected TPU limitation in {tpu_limited_count}/{len(tpu_fits)} sun leaves and did not improve the fits. TPU is not identified by this dataset. Fitted daytime Rd exceeded the directly observed dark CO2-efflux magnitude in {rd_above_dark_count}/{len(rd_comparable)} comparable leaves, another sign that the unconstrained gas-only model is assigning structural mismatch to Rd.

The joint fit uses storage-corrected A (`A == Adyn`) from all nine light levels and treats Ci as chloroplast CO2 only in the apparent/infinite-gm baseline. Estimates are therefore apparent capacities until gm is credibly constrained.

## Mesophyll-conductance diagnostic

At the LI-6800 ETR scaling, variable-J produced at least five algebraically valid flash estimates for only {len(gm_main)} fluorescence-QC-passing primary leaves. Their cross-leaf median was {gm_main['gm_median'].median():.3f} mol m-2 s-1, far below the approximately 0.27 Phragmites literature value. This is best interpreted as failure of the uncalibrated variable-J assumptions, not evidence that true gm is 0.06. Scaling ETR by 0.8 and 1.2 is retained because absorptance, PSII partitioning, Rd, and alternative electron sinks strongly affect the calculation.

## Interpretation boundary

Local optimizer standard errors do not include leaf-to-leaf variation, ramp-order confounding, or uncertainty in electron-transport calibration. The Bayesian tier will explicitly test which parameters move away from their priors and which remain prior-dominated.
"""
    REPORT_OUT.write_text(report, encoding="utf-8")


def main() -> None:
    qc = pd.read_csv(QC_PATH)
    curves = pd.read_csv(CURVES_PATH)
    flashes = pd.read_csv(FLASHES_PATH)
    jq = pd.read_csv(JQ_PATH)
    planteco = pd.read_csv(PLANTECO_PATH)

    fits, predictions = fit_all(curves, qc, jq)
    gm = variable_j_gm(flashes, fits, set(jq["run_code"]))
    contrasts = parameter_contrasts(fits, jq)

    fits.to_csv(FITS_OUT, index=False)
    predictions.to_csv(PREDICTIONS_OUT, index=False)
    gm.to_csv(GM_OUT, index=False)
    contrasts.to_csv(CONTRASTS_OUT, index=False)
    plot_results(fits, predictions, gm, planteco)
    write_report(fits, contrasts, gm, planteco)

    print(f"Wrote {FITS_OUT}")
    print(f"Wrote {PREDICTIONS_OUT}")
    print(f"Wrote {GM_OUT}")
    print(f"Wrote {CONTRASTS_OUT}")
    print(f"Wrote {FIGURE_OUT}")
    print(f"Wrote {REPORT_OUT}")
    print("\nPrimary joint fits:")
    print(
        fits.loc[fits["variant"].eq(PRIMARY_VARIANT) & fits["primary_comparison"], [
            "run_code", "zone", "Vcmax25", "Jmax25", "Rd", "alpha_J", "theta_J", "RMSE", "R2",
            "scaled_jacobian_condition", "near_parameter_bound", "Rdark_observed"
        ]].round(4).to_string(index=False)
    )
    print("\nVariable-J gm at native ETR scaling:")
    print(
        gm.loc[
            gm["ETR_scale"].eq(1.0)
            & gm["primary_comparison"]
            & gm["fluorescence_qc_ok"]
            & gm["n_valid_flashes"].ge(5)
        ].round(4).to_string(index=False)
    )


if __name__ == "__main__":
    main()
