from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

QC_PATH = OUTPUT_DIR / "phrag_brownsville_cohort_run_qc.csv"
CURVES_PATH = OUTPUT_DIR / "phrag_brownsville_cohort_binned_curves.csv"
FLUORESCENCE_PATH = OUTPUT_DIR / "phrag_brownsville_cohort_fluorescence_at_common_ci.csv"
TIER_B_FITS_PATH = OUTPUT_DIR / "phrag_brownsville_tier_b_joint_fvcb_fits.csv"

INPUT_OUT = OUTPUT_DIR / "phrag_brownsville_tier_c_model_input.csv"
TRACE_OUT = OUTPUT_DIR / "phrag_brownsville_tier_c_posterior.nc"
SUMMARY_OUT = OUTPUT_DIR / "phrag_brownsville_tier_c_posterior_summary.csv"
LEAF_OUT = OUTPUT_DIR / "phrag_brownsville_tier_c_leaf_parameters.csv"
CONTRAST_OUT = OUTPUT_DIR / "phrag_brownsville_tier_c_zone_contrasts.csv"
PREDICTION_OUT = OUTPUT_DIR / "phrag_brownsville_tier_c_predictions.csv"
FIGURE_OUT = OUTPUT_DIR / "phrag_brownsville_tier_c_bayesian_summary.png"
REPORT_OUT = OUTPUT_DIR / "phrag_brownsville_tier_c_report.md"

GAMMA_STAR = 42.75
KC = 404.9
KO = 278_400.0
OXYGEN = 210_000.0
KM = KC * (1.0 + OXYGEN / KO)

Q_LEVELS = np.array([50, 100, 200, 350, 600, 1000, 1400, 1700, 2000], dtype=float)
CI_TARGETS = np.array([100, 200, 300, 400, 600, 800, 1000, 1200, 1400], dtype=float)
ZONE_COLORS = {"High": "#2878a5", "Low": "#d97728"}


def empirical_j(q: np.ndarray, jmax: np.ndarray, alpha: np.ndarray, theta: np.ndarray) -> np.ndarray:
    light_limited = alpha * q
    discriminant = np.maximum((light_limited + jmax) ** 2 - 4.0 * theta * light_limited * jmax, 0.0)
    return (light_limited + jmax - np.sqrt(discriminant)) / (2.0 * theta)


def pytensor_j(q: pt.TensorVariable, jmax: pt.TensorVariable, alpha: pt.TensorVariable, theta: pt.TensorVariable) -> pt.TensorVariable:
    light_limited = alpha * q
    discriminant = pt.maximum((light_limited + jmax) ** 2 - 4.0 * theta * light_limited * jmax, 1e-8)
    return (light_limited + jmax - pt.sqrt(discriminant)) / (2.0 * theta)


def prepare_metadata(qc: pd.DataFrame) -> pd.DataFrame:
    all_runs = qc.copy()
    all_runs["start_datetime"] = pd.to_datetime(all_runs["start_datetime"])
    all_runs["date"] = all_runs["start_datetime"].dt.strftime("%Y-%m-%d")
    all_runs["day_order"] = (
        all_runs.groupby("date")["start_datetime"].rank(method="first").astype(int)
    )
    primary = all_runs.loc[(~all_runs["suspect_run"]) & all_runs["primary_comparison"]].copy()
    primary = primary.sort_values("start_datetime").reset_index(drop=True)
    primary["leaf_index"] = np.arange(len(primary))
    primary["zone_high"] = primary["zone"].eq("High").astype(float)
    order_sd = primary["day_order"].std(ddof=0)
    primary["order_z"] = (primary["day_order"] - primary["day_order"].mean()) / order_sd
    dates = {value: index for index, value in enumerate(sorted(primary["date"].unique()))}
    primary["day_index"] = primary["date"].map(dates)
    return primary


def interpolate_model_input(curves: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    metadata_index = metadata.set_index("run_code")
    for (run_code, q_level), group in curves.loc[curves["run_code"].isin(metadata["run_code"])].groupby(
        ["run_code", "Q_level"]
    ):
        clean = (
            group[["Ci", "A", "gsw"]]
            .dropna()
            .groupby("Ci", as_index=False)
            .median()
            .sort_values("Ci")
        )
        if len(clean) < 3:
            continue
        x = clean["Ci"].to_numpy(dtype=float)
        targets = CI_TARGETS[(CI_TARGETS >= x.min()) & (CI_TARGETS <= x.max())]
        meta = metadata_index.loc[run_code]
        q_progress = float(np.where(Q_LEVELS == q_level)[0][0] / (len(Q_LEVELS) - 1))
        for ci in targets:
            rows.append(
                {
                    "run_code": run_code,
                    "leaf_index": int(meta["leaf_index"]),
                    "zone": meta["zone"],
                    "zone_high": float(meta["zone_high"]),
                    "day_order": int(meta["day_order"]),
                    "order_z": float(meta["order_z"]),
                    "day_index": int(meta["day_index"]),
                    "Q_level": float(q_level),
                    "q_progress": q_progress,
                    "Ci": float(ci),
                    "A": float(np.interp(ci, x, clean["A"])),
                    "gsw": float(np.interp(ci, x, clean["gsw"])),
                }
            )
    return pd.DataFrame(rows)


def prepare_fluorescence(
    fluorescence: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    result = fluorescence.loc[
        fluorescence["run_code"].isin(metadata["run_code"])
        & fluorescence["fluorescence_qc_ok"]
        & fluorescence["ETR_at_Ci800"].notna()
    ].copy()
    return result.merge(metadata[["run_code", "leaf_index", "zone"]], on="run_code", how="inner")


def dark_data(tier_b_fits: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    result = (
        tier_b_fits.loc[tier_b_fits["variant"].eq("joint_appCi_noTPU"), ["run_code", "Rdark_observed"]]
        .drop_duplicates("run_code")
        .dropna(subset=["Rdark_observed"])
    )
    return result.merge(metadata[["run_code", "leaf_index"]], on="run_code", how="inner")


def build_model(
    gas: pd.DataFrame,
    flr: pd.DataFrame,
    dark: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pm.Model:
    n_leaf = len(metadata)
    coords = {
        "leaf": metadata["run_code"].tolist(),
        "gas_obs": np.arange(len(gas)),
        "flr_obs": np.arange(len(flr)),
        "dark_obs": np.arange(len(dark)),
    }

    leaf_zone_high = metadata["zone_high"].to_numpy(dtype=float)
    leaf_order = metadata["order_z"].to_numpy(dtype=float)
    gas_leaf = gas["leaf_index"].to_numpy(dtype=int)
    flr_leaf = flr["leaf_index"].to_numpy(dtype=int)
    dark_leaf = dark["leaf_index"].to_numpy(dtype=int)

    with pm.Model(coords=coords) as model:
        mu_log_vc = pm.Normal("mu_log_vc", mu=np.log(84.0), sigma=0.45)
        beta_vc_high = pm.Normal("beta_vc_high", mu=0.0, sigma=0.40)
        beta_vc_order = pm.Normal("beta_vc_order", mu=0.0, sigma=0.22)
        sigma_vc_leaf = pm.HalfNormal("sigma_vc_leaf", sigma=0.60)
        vc_raw = pm.Normal("vc_raw", 0.0, 1.0, dims="leaf")
        log_vc = (
            mu_log_vc
            + beta_vc_high * leaf_zone_high
            + beta_vc_order * leaf_order
            + sigma_vc_leaf * vc_raw
        )
        vcmax = pm.Deterministic("Vcmax25_leaf", pt.exp(log_vc), dims="leaf")

        mu_log_j = pm.Normal("mu_log_j", mu=np.log(116.0), sigma=0.38)
        beta_j_high = pm.Normal("beta_j_high", mu=0.0, sigma=0.40)
        beta_j_order = pm.Normal("beta_j_order", mu=0.0, sigma=0.22)
        sigma_j_leaf = pm.HalfNormal("sigma_j_leaf", sigma=0.55)
        j_raw = pm.Normal("j_raw", 0.0, 1.0, dims="leaf")
        log_j = (
            mu_log_j
            + beta_j_high * leaf_zone_high
            + beta_j_order * leaf_order
            + sigma_j_leaf * j_raw
        )
        jmax = pm.Deterministic("Jmax25_leaf", pt.exp(log_j), dims="leaf")

        mu_log_gm = pm.Normal("mu_log_gm", mu=np.log(0.27), sigma=0.40)
        beta_gm_high = pm.Normal("beta_gm_high", mu=0.0, sigma=0.35)
        sigma_gm_leaf = pm.HalfNormal("sigma_gm_leaf", sigma=0.40)
        gm_raw = pm.Normal("gm_raw", 0.0, 1.0, dims="leaf")
        gm = pm.Deterministic(
            "gm_leaf",
            pt.exp(mu_log_gm + beta_gm_high * leaf_zone_high + sigma_gm_leaf * gm_raw),
            dims="leaf",
        )

        mu_log_alpha = pm.Normal("mu_log_alpha", mu=np.log(0.30), sigma=0.25)
        beta_alpha_high = pm.Normal("beta_alpha_high", mu=0.0, sigma=0.25)
        alpha_raw = pm.Normal("alpha_raw", 0.0, 1.0, dims="leaf")
        alpha = pm.Deterministic(
            "alpha_J_leaf",
            pt.exp(mu_log_alpha + beta_alpha_high * leaf_zone_high + 0.08 * alpha_raw),
            dims="leaf",
        )

        theta_center = np.log(0.75 / 0.25)
        mu_logit_theta = pm.Normal("mu_logit_theta", mu=theta_center, sigma=0.55)
        beta_theta_high = pm.Normal("beta_theta_high", mu=0.0, sigma=0.45)
        theta_raw = pm.Normal("theta_raw", 0.0, 1.0, dims="leaf")
        theta = pm.Deterministic(
            "theta_J_leaf",
            pm.math.sigmoid(mu_logit_theta + beta_theta_high * leaf_zone_high + 0.55 * theta_raw),
            dims="leaf",
        )

        rdark = pm.LogNormal("Rdark_leaf", mu=np.log(1.5), sigma=0.48, dims="leaf")
        light_respiration_fraction = pm.Beta("light_respiration_fraction", alpha=6.0, beta=3.0)
        rd = pm.Deterministic("Rd_leaf", rdark * light_respiration_fraction, dims="leaf")
        if len(dark):
            pm.Normal(
                "Rdark_like",
                mu=rdark[dark_leaf],
                sigma=0.20,
                observed=dark["Rdark_observed"].to_numpy(dtype=float),
                dims="dark_obs",
            )

        fatigue = pm.HalfNormal("fatigue_leaf", sigma=1.5, dims="leaf")
        sigma_a = pm.HalfNormal("sigma_A_leaf", sigma=1.5, dims="leaf")

        gas_q = gas["Q_level"].to_numpy(dtype=float)
        gas_ci = gas["Ci"].to_numpy(dtype=float)
        gas_a = gas["A"].to_numpy(dtype=float)
        gas_gsw = np.maximum(gas["gsw"].to_numpy(dtype=float), 0.01)
        progress = gas["q_progress"].to_numpy(dtype=float)
        cc = pt.clip(gas_ci - gas_a / gm[gas_leaf], 1.0, 3000.0)
        ac = vcmax[gas_leaf] * (cc - GAMMA_STAR) / (cc + KM)
        j_gas = pytensor_j(gas_q, jmax[gas_leaf], alpha[gas_leaf], theta[gas_leaf])
        aj = j_gas * (cc - GAMMA_STAR) / (4.0 * cc + 8.0 * GAMMA_STAR)
        # A tight soft minimum preserves FvCB co-limitation while avoiding a
        # nondifferentiable ridge exactly where Ac and Aj exchange control.
        co_limitation_sharpness = 5.0
        gross_a = -pt.logaddexp(-co_limitation_sharpness * ac, -co_limitation_sharpness * aj) / co_limitation_sharpness
        a_mean = gross_a - rd[gas_leaf] - fatigue[gas_leaf] * progress
        noise_multiplier = np.sqrt(1.0 + (0.03 / gas_gsw) ** 2)
        pm.StudentT(
            "A_like",
            nu=4.0,
            mu=a_mean,
            sigma=sigma_a[gas_leaf] * noise_multiplier,
            observed=gas_a,
            dims="gas_obs",
        )

        sigma_etr = pm.HalfNormal("sigma_ETR", sigma=5.0)
        j_flr = pytensor_j(
            flr["Q_level"].to_numpy(dtype=float),
            jmax[flr_leaf],
            alpha[flr_leaf],
            theta[flr_leaf],
        )
        pm.StudentT(
            "ETR_like",
            nu=5.0,
            mu=j_flr,
            sigma=sigma_etr,
            observed=flr["ETR_at_Ci800"].to_numpy(dtype=float),
            dims="flr_obs",
        )

        pm.Deterministic("Vcmax_high_low_ratio", pt.exp(beta_vc_high))
        pm.Deterministic("Jmax_high_low_ratio", pt.exp(beta_j_high))
        pm.Deterministic("gm_high_low_ratio", pt.exp(beta_gm_high))
        pm.Deterministic("alpha_high_low_ratio", pt.exp(beta_alpha_high))
        theta_low = pm.math.sigmoid(mu_logit_theta)
        theta_high = pm.math.sigmoid(mu_logit_theta + beta_theta_high)
        pm.Deterministic("theta_high_minus_low", theta_high - theta_low)
        pm.Deterministic("Vcmax_order_ratio_per_sd", pt.exp(beta_vc_order))
        pm.Deterministic("Jmax_order_ratio_per_sd", pt.exp(beta_j_order))

    return model


def interval(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float).ravel()
    low, high = np.quantile(values, [0.025, 0.975])
    return float(np.median(values)), float(low), float(high)


def posterior_array(idata: az.InferenceData, variable: str) -> np.ndarray:
    return idata.posterior[variable].stack(sample=("chain", "draw")).values


def summarize_contrasts(idata: az.InferenceData) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specifications = [
        ("Vcmax_high_low_ratio", 1.0, "ratio", 0.40),
        ("Jmax_high_low_ratio", 1.0, "ratio", 0.40),
        ("gm_high_low_ratio", 1.0, "ratio", 0.35),
        ("alpha_high_low_ratio", 1.0, "ratio", 0.25),
        ("theta_high_minus_low", 0.0, "difference", 0.45),
        ("Vcmax_order_ratio_per_sd", 1.0, "ratio", 0.22),
        ("Jmax_order_ratio_per_sd", 1.0, "ratio", 0.22),
    ]
    beta_lookup = {
        "Vcmax_high_low_ratio": "beta_vc_high",
        "Jmax_high_low_ratio": "beta_j_high",
        "gm_high_low_ratio": "beta_gm_high",
        "alpha_high_low_ratio": "beta_alpha_high",
        "theta_high_minus_low": "beta_theta_high",
        "Vcmax_order_ratio_per_sd": "beta_vc_order",
        "Jmax_order_ratio_per_sd": "beta_j_order",
    }
    for variable, null, scale, prior_sd in specifications:
        values = posterior_array(idata, variable)
        median, low, high = interval(values)
        beta_values = posterior_array(idata, beta_lookup[variable])
        rows.append(
            {
                "contrast": variable,
                "scale": scale,
                "posterior_median": median,
                "hdi_2.5%": low,
                "hdi_97.5%": high,
                "probability_gt_null": float(np.mean(values > null)),
                "posterior_beta_sd": float(np.std(beta_values, ddof=1)),
                "prior_beta_sd": prior_sd,
                "posterior_to_prior_sd": float(np.std(beta_values, ddof=1) / prior_sd),
            }
        )
    return pd.DataFrame(rows)


def summarize_leaf_parameters(idata: az.InferenceData, metadata: pd.DataFrame) -> pd.DataFrame:
    parameter_names = [
        "Vcmax25_leaf",
        "Jmax25_leaf",
        "gm_leaf",
        "alpha_J_leaf",
        "theta_J_leaf",
        "Rdark_leaf",
        "Rd_leaf",
        "fatigue_leaf",
        "sigma_A_leaf",
    ]
    rows: list[dict[str, object]] = []
    for parameter in parameter_names:
        values = posterior_array(idata, parameter)
        for leaf_index, meta in metadata.iterrows():
            median, low, high = interval(values[leaf_index])
            rows.append(
                {
                    "run_code": meta["run_code"],
                    "zone": meta["zone"],
                    "day_order": meta["day_order"],
                    "parameter": parameter,
                    "posterior_median": median,
                    "hdi_2.5%": low,
                    "hdi_97.5%": high,
                }
            )
    return pd.DataFrame(rows)


def posterior_mean_predictions(
    idata: az.InferenceData,
    gas: pd.DataFrame,
) -> pd.DataFrame:
    def mean(variable: str) -> np.ndarray:
        return np.mean(posterior_array(idata, variable), axis=-1)

    vcmax = mean("Vcmax25_leaf")
    jmax = mean("Jmax25_leaf")
    gm = mean("gm_leaf")
    alpha = mean("alpha_J_leaf")
    theta = mean("theta_J_leaf")
    rd = mean("Rd_leaf")
    fatigue = mean("fatigue_leaf")
    leaf = gas["leaf_index"].to_numpy(dtype=int)
    ci = gas["Ci"].to_numpy(dtype=float)
    observed = gas["A"].to_numpy(dtype=float)
    q = gas["Q_level"].to_numpy(dtype=float)
    cc = np.clip(ci - observed / gm[leaf], 1.0, 3000.0)
    ac = vcmax[leaf] * (cc - GAMMA_STAR) / (cc + KM)
    j = empirical_j(q, jmax[leaf], alpha[leaf], theta[leaf])
    aj = j * (cc - GAMMA_STAR) / (4.0 * cc + 8.0 * GAMMA_STAR)
    co_limitation_sharpness = 5.0
    gross_a = -np.logaddexp(-co_limitation_sharpness * ac, -co_limitation_sharpness * aj) / co_limitation_sharpness
    predicted = gross_a - rd[leaf] - fatigue[leaf] * gas["q_progress"].to_numpy(dtype=float)
    result = gas.copy()
    result["Cc_posterior_mean"] = cc
    result["A_posterior_mean"] = predicted
    result["Ac_gross_posterior_mean"] = ac
    result["Aj_gross_posterior_mean"] = aj
    result["limitation_posterior_mean"] = np.where(ac <= aj, "Ac", "Aj")
    result["residual"] = observed - predicted
    return result


def plot_results(
    idata: az.InferenceData,
    metadata: pd.DataFrame,
    leaf_summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    predictions: pd.DataFrame,
    figure_out: Path = FIGURE_OUT,
) -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.75)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5))

    for ax, parameter, title, literature in [
        (axes[0, 0], "Vcmax25_leaf", "Posterior Vcmax25 by leaf", 84.0),
        (axes[0, 1], "Jmax25_leaf", "Posterior Jmax25 by leaf", 116.0),
    ]:
        data = leaf_summary.loc[leaf_summary["parameter"].eq(parameter)].copy()
        data = data.sort_values(["zone", "posterior_median"]).reset_index(drop=True)
        y = np.arange(len(data))
        colors = data["zone"].map(ZONE_COLORS)
        ax.hlines(y, data["hdi_2.5%"], data["hdi_97.5%"], color=colors, lw=2)
        ax.scatter(data["posterior_median"], y, c=colors, s=50, edgecolor="white", zorder=3)
        ax.axvline(literature, color="#555555", ls="--", lw=1.2)
        ax.set_yticks(y, data["run_code"])
        ax.set(title=title, xlabel=parameter, ylabel="")

    ratio_data = contrasts.loc[
        contrasts["contrast"].isin(
            ["Vcmax_high_low_ratio", "Jmax_high_low_ratio", "gm_high_low_ratio", "alpha_high_low_ratio"]
        )
    ].copy()
    labels = {
        "Vcmax_high_low_ratio": "Vcmax",
        "Jmax_high_low_ratio": "Jmax",
        "gm_high_low_ratio": "gm",
        "alpha_high_low_ratio": "alpha",
    }
    ratio_data["label"] = ratio_data["contrast"].map(labels)
    y = np.arange(len(ratio_data))
    axes[0, 2].hlines(y, ratio_data["hdi_2.5%"], ratio_data["hdi_97.5%"], color="#555555", lw=2)
    axes[0, 2].scatter(ratio_data["posterior_median"], y, color="#3a7d62", s=55, edgecolor="white")
    axes[0, 2].axvline(1.0, color="#333333", ls="--", lw=1.2)
    axes[0, 2].set_yticks(y, ratio_data["label"])
    axes[0, 2].set(title="High / low posterior ratios", xlabel="Ratio", ylabel="")

    theta_data = leaf_summary.loc[leaf_summary["parameter"].eq("theta_J_leaf")].copy()
    theta_data = theta_data.sort_values(["zone", "posterior_median"]).reset_index(drop=True)
    y = np.arange(len(theta_data))
    theta_colors = theta_data["zone"].map(ZONE_COLORS)
    axes[1, 0].hlines(y, theta_data["hdi_2.5%"], theta_data["hdi_97.5%"], color=theta_colors, lw=2)
    axes[1, 0].scatter(theta_data["posterior_median"], y, c=theta_colors, s=50, edgecolor="white")
    axes[1, 0].set_yticks(y, theta_data["run_code"])
    axes[1, 0].set(title="Fluorescence-informed theta", xlabel="theta", ylabel="")

    fatigue_data = leaf_summary.loc[leaf_summary["parameter"].eq("fatigue_leaf")].copy()
    fatigue_data = fatigue_data.sort_values(["zone", "posterior_median"]).reset_index(drop=True)
    y = np.arange(len(fatigue_data))
    fatigue_colors = fatigue_data["zone"].map(ZONE_COLORS)
    axes[1, 1].hlines(y, fatigue_data["hdi_2.5%"], fatigue_data["hdi_97.5%"], color=fatigue_colors, lw=2)
    axes[1, 1].scatter(fatigue_data["posterior_median"], y, c=fatigue_colors, s=50, edgecolor="white")
    axes[1, 1].set_yticks(y, fatigue_data["run_code"])
    axes[1, 1].set(title="Allowed A decline across run", xlabel="A decline from first to last Q", ylabel="")

    for zone, color in ZONE_COLORS.items():
        zone_data = predictions.loc[predictions["zone"].eq(zone)]
        axes[1, 2].scatter(zone_data["A"], zone_data["A_posterior_mean"], color=color, alpha=0.35, s=16, label=zone)
    lower = min(predictions["A"].min(), predictions["A_posterior_mean"].min())
    upper = max(predictions["A"].max(), predictions["A_posterior_mean"].max())
    axes[1, 2].plot([lower, upper], [lower, upper], color="#333333", ls="--", lw=1.2)
    axes[1, 2].set(title="Posterior-mean predictive fit", xlabel="Observed A", ylabel="Predicted A")
    axes[1, 2].legend(frameon=False)

    fig.suptitle("Brownsville Phragmites: Tier C Bayesian FvCB Model", x=0.055, ha="left", fontsize=22, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.0, w_pad=1.7)
    fig.savefig(figure_out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(
    idata: az.InferenceData,
    metadata: pd.DataFrame,
    leaf_summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    predictions: pd.DataFrame,
    report_out: Path = REPORT_OUT,
) -> None:
    contrast_index = contrasts.set_index("contrast")

    def contrast_line(name: str, digits: int = 2) -> str:
        row = contrast_index.loc[name]
        return (
            f"median {row.posterior_median:.{digits}f}, 95% interval "
            f"[{row['hdi_2.5%']:.{digits}f}, {row['hdi_97.5%']:.{digits}f}], "
            f"P(> null)={row.probability_gt_null:.2f}, posterior/prior SD={row.posterior_to_prior_sd:.2f}"
        )

    max_rhat = float(az.summary(idata)["r_hat"].max())
    divergences = int(idata.sample_stats["diverging"].sum())
    rmse = float(np.sqrt(np.mean(predictions["residual"] ** 2)))
    leaf_vc = leaf_summary.loc[leaf_summary["parameter"].eq("Vcmax25_leaf")]
    leaf_j = leaf_summary.loc[leaf_summary["parameter"].eq("Jmax25_leaf")]
    leaf_gm = leaf_summary.loc[leaf_summary["parameter"].eq("gm_leaf")]
    leaf_fatigue = leaf_summary.loc[leaf_summary["parameter"].eq("fatigue_leaf")]

    report = f"""# Tier C: hierarchical Bayesian FvCB results

## Model

The model jointly uses balanced A/Ci/Q observations, dark gas exchange, and fluorescence ETR. Leaf-level Vcmax, Jmax, gm, respiration, and monotonic run fatigue are partially pooled. Alpha and theta have zone means plus fixed leaf-heterogeneity scales calibrated to the clean standalone fluorescence fits; estimating those variance scales jointly with a free ETR calibration was not identifiable. High/low forest and within-day order enter the capacity hierarchy. Literature priors are centered at Vcmax25=84, Jmax25=116, and gm=0.27 but deliberately broad.

Collection date is not assigned a separate effect: two retained dates contain only one leaf, and zone/date/order are too entangled to distinguish date variation from leaf variation. The resulting zone contrast therefore remains vulnerable to date confounding despite the explicit order adjustment.

## Zone effects

- Vcmax high/low ratio: {contrast_line('Vcmax_high_low_ratio')}.
- Jmax high/low ratio: {contrast_line('Jmax_high_low_ratio')}.
- gm high/low ratio: {contrast_line('gm_high_low_ratio')}.
- alpha high/low ratio: {contrast_line('alpha_high_low_ratio')}.
- theta high-low difference: {contrast_line('theta_high_minus_low', 3)}.

An interval spanning 1 for a ratio, or 0 for theta difference, means the current data do not resolve the direction. A posterior/prior SD near 1 means the contrast remains largely prior-dominated.

## Order and state

- Vcmax order ratio per one SD later within day: {contrast_line('Vcmax_order_ratio_per_sd')}.
- Jmax order ratio per one SD later within day: {contrast_line('Jmax_order_ratio_per_sd')}.
- Median inferred full-run A decline ranged from {leaf_fatigue['posterior_median'].min():.2f} to {leaf_fatigue['posterior_median'].max():.2f} umol m-2 s-1 across leaves.

Because all Q sequences were ascending, fatigue and high-Q response remain structurally confounded. The shrinkage term is a sensitivity adjustment, not an independently identified physiological process.

## Absolute parameters

- Leaf Vcmax25 posterior medians span {leaf_vc['posterior_median'].min():.1f}-{leaf_vc['posterior_median'].max():.1f}.
- Leaf Jmax25 posterior medians span {leaf_j['posterior_median'].min():.1f}-{leaf_j['posterior_median'].max():.1f}.
- Leaf gm posterior medians span {leaf_gm['posterior_median'].min():.3f}-{leaf_gm['posterior_median'].max():.3f} mol m-2 s-1.
- Fluorescence ETR is used on the LI-6800 reported scale. Absolute J therefore inherits the instrument's absorptance and PSII-partition assumptions; alternative electron sinks remain model discrepancy rather than an independently fitted calibration factor.

## Diagnostics

- Posterior-mean A RMSE: {rmse:.2f} umol m-2 s-1.
- Divergences: {divergences}.
- Maximum reported R-hat: {max_rhat:.3f}.

These posteriors describe plausible parameter combinations under the model. Strong Vcmax/Jmax learning accompanied by weak gm or zone-effect contraction should be interpreted as biochemical capacity information with unresolved diffusion and treatment contrasts.
"""
    report_out.write_text(report, encoding="utf-8")


def with_suffix(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}") if suffix else path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the Tier-C Bayesian Brownsville Phragmites model.")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1500)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--exclude-run", action="append", default=[])
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()

    qc = pd.read_csv(QC_PATH)
    curves = pd.read_csv(CURVES_PATH)
    fluorescence = pd.read_csv(FLUORESCENCE_PATH)
    tier_b_fits = pd.read_csv(TIER_B_FITS_PATH)
    metadata = prepare_metadata(qc)
    if args.exclude_run:
        metadata = metadata.loc[~metadata["run_code"].isin(args.exclude_run)].copy().reset_index(drop=True)
        metadata["leaf_index"] = np.arange(len(metadata))
    gas = interpolate_model_input(curves, metadata)
    flr = prepare_fluorescence(fluorescence, metadata)
    dark = dark_data(tier_b_fits, metadata)
    input_out = with_suffix(INPUT_OUT, args.output_suffix)
    trace_out = with_suffix(TRACE_OUT, args.output_suffix)
    summary_out = with_suffix(SUMMARY_OUT, args.output_suffix)
    leaf_out = with_suffix(LEAF_OUT, args.output_suffix)
    contrast_out = with_suffix(CONTRAST_OUT, args.output_suffix)
    prediction_out = with_suffix(PREDICTION_OUT, args.output_suffix)
    figure_out = with_suffix(FIGURE_OUT, args.output_suffix)
    report_out = with_suffix(REPORT_OUT, args.output_suffix)
    gas.to_csv(input_out, index=False)

    model = build_model(gas, flr, dark, metadata)
    with model:
        pm.sample_prior_predictive(draws=500, random_seed=20260803)
        idata = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.cores,
            target_accept=0.93,
            random_seed=[20260803 + index for index in range(args.chains)],
            return_inferencedata=True,
        )
    idata.to_netcdf(trace_out)

    summary = az.summary(idata, round_to=None)
    summary.to_csv(summary_out)
    leaf_summary = summarize_leaf_parameters(idata, metadata)
    contrasts = summarize_contrasts(idata)
    predictions = posterior_mean_predictions(idata, gas)
    leaf_summary.to_csv(leaf_out, index=False)
    contrasts.to_csv(contrast_out, index=False)
    predictions.to_csv(prediction_out, index=False)
    plot_results(idata, metadata, leaf_summary, contrasts, predictions, figure_out)
    write_report(idata, metadata, leaf_summary, contrasts, predictions, report_out)

    for path in [trace_out, summary_out, leaf_out, contrast_out, prediction_out, figure_out, report_out]:
        print(f"Wrote {path}")
    print("\nZone and order contrasts:")
    print(contrasts.round(4).to_string(index=False))
    print("\nLeaf parameters:")
    print(
        leaf_summary.loc[
            leaf_summary["parameter"].isin(["Vcmax25_leaf", "Jmax25_leaf", "gm_leaf", "theta_J_leaf", "fatigue_leaf"])
        ].round(4).to_string(index=False)
    )


if __name__ == "__main__":
    main()
