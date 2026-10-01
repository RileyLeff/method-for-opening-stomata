from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
ADDITIVE_PATH = OUTPUTS / "phrag_brownsville_additive_yields_common_ci.csv"
NPQ_PATH = OUTPUTS / "phrag_brownsville_npq_interpolated_q_ci.csv"
TEST_OUT = OUTPUTS / "phrag_brownsville_npq_overall_significance_tests.csv"
LEAF_OUT = OUTPUTS / "phrag_brownsville_npq_overall_leaf_means.csv"

VALID_RUNS = ["P03", "P06", "P10", "P11", "P12", "P13", "P14"]
N_BOOTSTRAP = 500_000


def exact_permutation(
    values: np.ndarray, labels: np.ndarray, n_low: int
) -> tuple[float, float, float]:
    observed = values[labels == "Low"].mean() - values[labels == "High"].mean()
    differences: list[float] = []
    for low_indices in itertools.combinations(range(len(values)), n_low):
        low_mask = np.zeros(len(values), dtype=bool)
        low_mask[list(low_indices)] = True
        differences.append(values[low_mask].mean() - values[~low_mask].mean())
    null = np.asarray(differences)
    two_sided = float(np.mean(np.abs(null) >= abs(observed) - 1e-14))
    one_sided = float(np.mean(null >= observed - 1e-14))
    return float(observed), two_sided, one_sided


def metric_test(
    metric: str,
    leaf_data: pd.DataFrame,
    value_column: str,
    rng: np.random.Generator,
) -> dict[str, float | int | str]:
    high = leaf_data.loc[leaf_data["zone"].eq("High"), value_column].to_numpy()
    low = leaf_data.loc[leaf_data["zone"].eq("Low"), value_column].to_numpy()
    values = leaf_data[value_column].to_numpy()
    labels = leaf_data["zone"].to_numpy()

    difference, permutation_two_sided, permutation_one_sided = exact_permutation(
        values, labels, len(low)
    )
    welch = stats.ttest_ind(low, high, equal_var=False, alternative="two-sided")
    mann_whitney_two_sided = stats.mannwhitneyu(
        low, high, alternative="two-sided", method="exact"
    )
    mann_whitney_one_sided = stats.mannwhitneyu(
        low, high, alternative="greater", method="exact"
    )

    high_boot = high[
        rng.integers(0, len(high), size=(N_BOOTSTRAP, len(high)))
    ].mean(axis=1)
    low_boot = low[
        rng.integers(0, len(low), size=(N_BOOTSTRAP, len(low)))
    ].mean(axis=1)
    difference_boot = low_boot - high_boot
    relative_boot = difference_boot / high_boot * 100

    pooled_sd = np.sqrt(
        ((len(low) - 1) * low.var(ddof=1) + (len(high) - 1) * high.var(ddof=1))
        / (len(low) + len(high) - 2)
    )
    cohens_d = difference / pooled_sd
    small_sample_correction = 1 - 3 / (4 * (len(low) + len(high)) - 9)

    return {
        "metric": metric,
        "n_high_leaves": len(high),
        "n_low_leaves": len(low),
        "high_mean": float(high.mean()),
        "high_sd": float(high.std(ddof=1)),
        "low_mean": float(low.mean()),
        "low_sd": float(low.std(ddof=1)),
        "low_minus_high": difference,
        "relative_change_percent": float(difference / high.mean() * 100),
        "leaf_bootstrap_2.5_percent": float(np.quantile(relative_boot, 0.025)),
        "leaf_bootstrap_97.5_percent": float(np.quantile(relative_boot, 0.975)),
        "leaf_bootstrap_fraction_low_gt_high": float(np.mean(difference_boot > 0)),
        "exact_permutation_p_two_sided": permutation_two_sided,
        "exact_permutation_p_one_sided_low_gt_high": permutation_one_sided,
        "welch_t": float(welch.statistic),
        "welch_df": float(welch.df),
        "welch_p_two_sided": float(welch.pvalue),
        "mann_whitney_u": float(mann_whitney_two_sided.statistic),
        "mann_whitney_p_two_sided": float(mann_whitney_two_sided.pvalue),
        "mann_whitney_p_one_sided_low_gt_high": float(
            mann_whitney_one_sided.pvalue
        ),
        "hedges_g_low_minus_high": float(small_sample_correction * cohens_d),
    }


def main() -> None:
    additive = pd.read_csv(ADDITIVE_PATH)
    additive = additive.loc[
        additive["run_code"].isin(VALID_RUNS) & additive["Ci"].ge(350)
    ]
    additive_leaf = (
        additive.groupby(["run_code", "zone"], as_index=False)["YNPQ"].mean()
    )

    npq = pd.read_csv(NPQ_PATH)
    npq = npq.loc[npq["run_code"].isin(VALID_RUNS) & npq["Ci"].ge(350)]
    npq_leaf = npq.groupby(["run_code", "zone"], as_index=False)["NPQ"].mean()

    leaf_values = additive_leaf.merge(npq_leaf, on=["run_code", "zone"])
    rng = np.random.default_rng(20260804)
    tests = pd.DataFrame(
        [
            metric_test("Y(NPQ) additive yield", leaf_values, "YNPQ", rng),
            metric_test("NPQ index", leaf_values, "NPQ", rng),
        ]
    )
    leaf_values.to_csv(LEAF_OUT, index=False)
    tests.to_csv(TEST_OUT, index=False)
    print(tests.to_string(index=False))
    print(f"Wrote {LEAF_OUT}")
    print(f"Wrote {TEST_OUT}")


if __name__ == "__main__":
    main()
