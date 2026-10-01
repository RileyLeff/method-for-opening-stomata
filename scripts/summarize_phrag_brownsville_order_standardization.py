from __future__ import annotations

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
TRACE = OUTPUTS / "phrag_brownsville_tier_c_posterior.nc"
QC = OUTPUTS / "phrag_brownsville_cohort_run_qc.csv"
PROFILE_OUT = OUTPUTS / "phrag_brownsville_tier_c_order_profiles.csv"
STANDARDIZED_OUT = OUTPUTS / "phrag_brownsville_tier_c_order_standardized_contrasts.csv"
FIGURE_OUT = OUTPUTS / "phrag_brownsville_tier_c_order_standardization.png"

COLORS = {"High": "#2878a5", "Low": "#d97728"}


def samples(tree: xr.DataTree, variable: str) -> np.ndarray:
    return tree.posterior[variable].stack(sample=("chain", "draw")).values


def summarize(values: np.ndarray) -> tuple[float, float, float]:
    return tuple(np.quantile(values, [0.5, 0.025, 0.975]))


def retained_orders() -> np.ndarray:
    qc = pd.read_csv(QC)
    qc["start_datetime"] = pd.to_datetime(qc["start_datetime"])
    qc["date"] = qc["start_datetime"].dt.date
    qc["day_order"] = qc.groupby("date")["start_datetime"].rank(method="first").astype(int)
    keep = qc.loc[(~qc["suspect_run"]) & qc["primary_comparison"]]
    return keep["day_order"].to_numpy(dtype=float)


def main() -> None:
    tree = xr.open_datatree(TRACE)
    observed_orders = retained_orders()
    order_mean = observed_orders.mean()
    order_sd = observed_orders.std(ddof=0)
    order_grid = np.arange(1, 5, dtype=float)
    rng = np.random.default_rng(20260803)

    specifications = {
        "Vcmax25": ("mu_log_vc", "beta_vc_high", "beta_vc_order", "sigma_vc_leaf"),
        "Jmax25": ("mu_log_j", "beta_j_high", "beta_j_order", "sigma_j_leaf"),
    }
    profile_rows: list[dict[str, float | str]] = []
    standardized_rows: list[dict[str, float | str]] = []

    for parameter, (mu_name, zone_name, order_name, sigma_name) in specifications.items():
        mu = samples(tree, mu_name)
        beta_zone = samples(tree, zone_name)
        beta_order = samples(tree, order_name)
        sigma_leaf = samples(tree, sigma_name)
        standardized_by_zone: dict[str, np.ndarray] = {}

        for zone, zone_high in [("Low", 0.0), ("High", 1.0)]:
            order_values: list[np.ndarray] = []
            for order in order_grid:
                order_z = (order - order_mean) / order_sd
                center = np.exp(mu + beta_zone * zone_high + beta_order * order_z)
                predictive = np.exp(np.log(center) + sigma_leaf * rng.normal(size=center.shape))
                center_median, center_low, center_high = summarize(center)
                pred_median, pred_low, pred_high = summarize(predictive)
                profile_rows.append(
                    {
                        "parameter": parameter,
                        "zone": zone,
                        "day_order": order,
                        "population_center_median": center_median,
                        "population_center_2.5%": center_low,
                        "population_center_97.5%": center_high,
                        "new_leaf_predictive_median": pred_median,
                        "new_leaf_predictive_2.5%": pred_low,
                        "new_leaf_predictive_97.5%": pred_high,
                    }
                )
                order_values.append(center)

            observed_z = (observed_orders - order_mean) / order_sd
            standardized = np.mean(
                np.exp(mu[:, None] + beta_zone[:, None] * zone_high + beta_order[:, None] * observed_z[None, :]),
                axis=1,
            )
            standardized_by_zone[zone] = standardized
            median, low, high = summarize(standardized)
            standardized_rows.append(
                {
                    "parameter": parameter,
                    "estimand": f"{zone}_standardized_center",
                    "posterior_median": median,
                    "hdi_2.5%": low,
                    "hdi_97.5%": high,
                }
            )

        ratio = standardized_by_zone["High"] / standardized_by_zone["Low"]
        median, low, high = summarize(ratio)
        standardized_rows.append(
            {
                "parameter": parameter,
                "estimand": "high_low_ratio_same_order_mix",
                "posterior_median": median,
                "hdi_2.5%": low,
                "hdi_97.5%": high,
            }
        )

    profiles = pd.DataFrame(profile_rows)
    standardized = pd.DataFrame(standardized_rows)
    profiles.to_csv(PROFILE_OUT, index=False)
    standardized.to_csv(STANDARDIZED_OUT, index=False)

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    for ax, parameter, label in zip(axes, ["Vcmax25", "Jmax25"], ["Vcmax25", "Jmax25"]):
        for zone in ["High", "Low"]:
            data = profiles.loc[(profiles["parameter"] == parameter) & (profiles["zone"] == zone)]
            color = COLORS[zone]
            ax.fill_between(
                data["day_order"],
                data["new_leaf_predictive_2.5%"],
                data["new_leaf_predictive_97.5%"],
                color=color,
                alpha=0.12,
            )
            ax.fill_between(
                data["day_order"],
                data["population_center_2.5%"],
                data["population_center_97.5%"],
                color=color,
                alpha=0.28,
            )
            ax.plot(
                data["day_order"],
                data["population_center_median"],
                color=color,
                marker="o",
                lw=2.5,
                label=zone,
            )
        ax.set(title=f"Order-standardized {label}", xlabel="Within-day measurement order", ylabel=label)
        ax.set_yscale("log")
        ax.set_xticks(order_grid)
        ax.legend(frameon=False)

    fig.suptitle("High and low Phragmites evaluated across the same order range", fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "Inner band: uncertainty in the zone center. Outer band: predicted range for a new leaf.",
        ha="center",
        fontsize=11,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    fig.savefig(FIGURE_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)

    diagnostics = az.summary(tree, var_names=["beta_vc_order", "beta_j_order"])
    print(f"Wrote {PROFILE_OUT}")
    print(f"Wrote {STANDARDIZED_OUT}")
    print(f"Wrote {FIGURE_OUT}")
    print("\nStandardized contrasts:")
    print(standardized.round(4).to_string(index=False))
    print("\nOrder diagnostics:")
    print(diagnostics.to_string())


if __name__ == "__main__":
    main()
