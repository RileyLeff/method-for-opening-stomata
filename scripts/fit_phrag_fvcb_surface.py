from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import least_squares

from photo26.curves import OUTPUT_DIR, read_licor_export


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = PROJECT_ROOT / "data" / "2026-07-07_phrag_aq_dark_rerun" / "2026-07-07-1301_logdata_phrag_aq_dark"

FIT_SUMMARY_OUT = OUTPUT_DIR / "phrag_aq_dark_rerun_fvcb_fit_summary.csv"
LITERATURE_OUT = OUTPUT_DIR / "phrag_aq_dark_rerun_fvcb_literature_values.csv"
COMPARISON_OUT = OUTPUT_DIR / "phrag_aq_dark_rerun_fvcb_literature_comparison.png"
OVERLAY_OUT = OUTPUT_DIR / "phrag_aq_dark_rerun_fvcb_fit_overlay.png"

Q_LEVELS = np.array([0, 200, 400, 700, 1000, 1300, 1600], dtype=float)
ACTIVE_Q_LEVELS = [200, 400, 700, 1000, 1300, 1600]

GAMMA_STAR = 42.75
KC = 404.9
KO = 278_400.0
OXYGEN = 210_000.0
KM = KC * (1 + OXYGEN / KO)


def nearest_q(values: pd.Series) -> np.ndarray:
    q = values.to_numpy(dtype=float)
    return Q_LEVELS[np.argmin(np.abs(q[:, None] - Q_LEVELS[None, :]), axis=1)]


def load_filtered_points() -> pd.DataFrame:
    points = read_licor_export(LOG_PATH).sort_values("elapsed").reset_index(drop=True).copy()
    points["Q_level"] = nearest_q(points["Q"])
    points["minute"] = points["elapsed"] / 60
    phase_break = points["Q_level"].ne(points["Q_level"].shift()) | points["minute"].diff().gt(5)
    points["phase_id"] = phase_break.cumsum()
    points["phase_min"] = points["minute"] - points.groupby("phase_id")["minute"].transform("min")
    points["sane_for_curve"] = (
        points["Q_level"].isin(ACTIVE_Q_LEVELS)
        & points["Ci"].between(0, 2200)
        & points["A"].between(-20, 40)
        & points["gsw"].between(0.005, 1.0)
        & points["CO2_r"].between(40, 2050)
        & points["CO2_s"].between(40, 2050)
        & points["Flow"].between(490, 510)
        & points["VPDleaf"].between(0.75, 1.25)
    )
    return points.loc[points["sane_for_curve"]].copy()


def binned_response(points: pd.DataFrame, bin_width: int = 60, min_count: int = 3) -> pd.DataFrame:
    data = points.dropna(subset=["Ci", "A", "Qabs", "Q_level"]).copy()
    data["ci_bin"] = pd.cut(data["Ci"], np.arange(0, 2200 + bin_width, bin_width), include_lowest=True)
    binned = (
        data.groupby(["Q_level", "ci_bin"], observed=True)
        .agg(
            n=("A", "size"),
            Ci=("Ci", "median"),
            Q=("Q", "median"),
            Qabs=("Qabs", "median"),
            A_med=("A", "median"),
            A_q75=("A", lambda s: s.quantile(0.75)),
            A_q90=("A", lambda s: s.quantile(0.90)),
            A_max=("A", "max"),
            gsw=("gsw", "median"),
        )
        .reset_index()
    )
    return binned.loc[binned["n"].ge(min_count)].dropna().reset_index(drop=True)


def electron_transport(qabs: np.ndarray, jmax: float, alpha: float, theta: float) -> np.ndarray:
    absorbed = alpha * qabs
    disc = np.maximum((absorbed + jmax) ** 2 - 4 * theta * absorbed * jmax, 0)
    return ((absorbed + jmax) - np.sqrt(disc)) / (2 * theta)


def fvcb_prediction(params: np.ndarray, ci: np.ndarray, qabs: np.ndarray, *, free_alpha: bool) -> np.ndarray:
    if free_alpha:
        vcmax, jmax, rd, alpha = params
    else:
        vcmax, jmax, rd = params
        alpha = 0.24
    theta = 0.70
    ac = vcmax * (ci - GAMMA_STAR) / (ci + KM)
    j = electron_transport(qabs, jmax, alpha, theta)
    aj = j * (ci - GAMMA_STAR) / (4 * ci + 8 * GAMMA_STAR)
    return np.minimum(ac, aj) - rd


def fit_variant(data: pd.DataFrame, y_column: str, *, free_alpha: bool, label: str) -> dict[str, float | str | int]:
    ci = data["Ci"].to_numpy(dtype=float)
    qabs = data["Qabs"].to_numpy(dtype=float)
    observed = data[y_column].to_numpy(dtype=float)
    if free_alpha:
        x0 = np.array([70, 120, 2.0, 0.22])
        lower = np.array([5, 10, 0, 0.02])
        upper = np.array([300, 600, 8, 0.50])
    else:
        x0 = np.array([70, 120, 2.0])
        lower = np.array([5, 10, 0])
        upper = np.array([300, 600, 8])

    result = least_squares(
        lambda x: fvcb_prediction(x, ci, qabs, free_alpha=free_alpha) - observed,
        x0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=100_000,
    )
    fitted = fvcb_prediction(result.x, ci, qabs, free_alpha=free_alpha)
    total = np.sum((observed - observed.mean()) ** 2)
    r_squared = 1 - np.sum((observed - fitted) ** 2) / total

    if free_alpha:
        vcmax, jmax, rd, alpha = result.x
    else:
        vcmax, jmax, rd = result.x
        alpha = 0.24
    return {
        "fit": label,
        "target": y_column,
        "n_points": len(data),
        "Vcmax25_app_Ci": float(vcmax),
        "Jmax25_app_Ci": float(jmax),
        "Rd": float(rd),
        "alpha": float(alpha),
        "theta": 0.70,
        "r_squared": float(r_squared),
        "rmse": float(np.sqrt(np.mean((observed - fitted) ** 2))),
    }


def fit_all(binned: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for y_column in ["A_med", "A_q75", "A_q90", "A_max"]:
        rows.append(fit_variant(binned, y_column, free_alpha=False, label=f"{y_column}_fixed_alpha_theta"))
        rows.append(fit_variant(binned, y_column, free_alpha=True, label=f"{y_column}_free_alpha_fixed_theta"))
    return pd.DataFrame(rows)


def literature_values() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source": "Eller & Brix 2013 AoB Plants",
                "lineage": "I2 Land-type",
                "Vcmax25": 118,
                "Vcmax_se": 6,
                "Jmax25": 178,
                "Jmax_se": 10,
                "Amax": 34.3,
                "Amax_se": np.nan,
            },
            {
                "source": "Eller & Brix 2013 AoB Plants",
                "lineage": "M1 Delta-type",
                "Vcmax25": 123,
                "Vcmax_se": 6,
                "Jmax25": 172,
                "Jmax_se": 9,
                "Amax": 36.1,
                "Amax_se": np.nan,
            },
            {
                "source": "Eller & Brix 2013 AoB Plants",
                "lineage": "M Eurasian invasive",
                "Vcmax25": 84,
                "Vcmax_se": 6,
                "Jmax25": 116,
                "Jmax_se": 10,
                "Amax": 22.4,
                "Amax_se": 2.3,
            },
            {
                "source": "Eller & Brix 2013 AoB Plants",
                "lineage": "AI Greeny3-type",
                "Vcmax25": 99,
                "Vcmax_se": 11,
                "Jmax25": 137,
                "Jmax_se": 17,
                "Amax": 29.1,
                "Amax_se": 4.0,
            },
        ]
    )


def plot_comparison(fits: pd.DataFrame, literature: pd.DataFrame) -> None:
    fit_plot = fits.loc[
        fits["fit"].isin(
            [
                "A_med_fixed_alpha_theta",
                "A_med_free_alpha_fixed_theta",
                "A_q90_free_alpha_fixed_theta",
            ]
        )
    ].copy()
    fit_plot["lineage"] = fit_plot["fit"].map(
        {
            "A_med_fixed_alpha_theta": "This run: median, fixed alpha",
            "A_med_free_alpha_fixed_theta": "This run: median, free alpha",
            "A_q90_free_alpha_fixed_theta": "This run: upper envelope",
        }
    )
    fit_plot["source"] = "July 7 DAT surface"
    fit_plot["Vcmax25"] = fit_plot["Vcmax25_app_Ci"]
    fit_plot["Jmax25"] = fit_plot["Jmax25_app_Ci"]
    fit_plot["Vcmax_se"] = np.nan
    fit_plot["Jmax_se"] = np.nan
    combined = pd.concat(
        [
            literature[["source", "lineage", "Vcmax25", "Vcmax_se", "Jmax25", "Jmax_se"]],
            fit_plot[["source", "lineage", "Vcmax25", "Vcmax_se", "Jmax25", "Jmax_se"]],
        ],
        ignore_index=True,
    )
    order = [
        "M Eurasian invasive",
        "This run: median, fixed alpha",
        "This run: median, free alpha",
        "This run: upper envelope",
        "AI Greeny3-type",
        "I2 Land-type",
        "M1 Delta-type",
    ]
    combined["lineage"] = pd.Categorical(combined["lineage"], categories=order, ordered=True)
    combined = combined.sort_values("lineage")

    sns.set_theme(style="whitegrid", context="talk", font_scale=0.76)
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.4), sharex=True)
    colors = ["#805bb7" if "This run" in str(label) else "#4d7c8a" for label in combined["lineage"]]
    for ax, value_col, err_col, title in [
        (axes[0], "Vcmax25", "Vcmax_se", "Vcmax"),
        (axes[1], "Jmax25", "Jmax_se", "Jmax"),
    ]:
        x = np.arange(len(combined))
        ax.bar(x, combined[value_col], color=colors, alpha=0.85)
        errors = combined[err_col].to_numpy(dtype=float)
        mask = np.isfinite(errors)
        ax.errorbar(x[mask], combined.loc[mask, value_col], yerr=errors[mask], fmt="none", ecolor="#333333", capsize=3, lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(combined["lineage"], rotation=35, ha="right")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(r"$\mu$mol m$^{-2}$ s$^{-1}$")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, axis="y", color="#d8ddd4")
    fig.text(0.06, 0.982, "Phragmites FvCB Fit vs Literature", ha="left", va="top", fontsize=20, fontweight="bold")
    fig.text(
        0.06,
        0.93,
        "This-run values are apparent Ci-based fits from binned DAT rows; literature values are haplotype means at 25 C.",
        ha="left",
        va="top",
        fontsize=10,
        color="#555555",
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.28, top=0.84, wspace=0.22)
    fig.savefig(COMPARISON_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_overlay(binned: pd.DataFrame, fits: pd.DataFrame) -> None:
    row = fits.loc[fits["fit"].eq("A_med_free_alpha_fixed_theta")].iloc[0]
    params = np.array([row["Vcmax25_app_Ci"], row["Jmax25_app_Ci"], row["Rd"], row["alpha"]], dtype=float)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=min(ACTIVE_Q_LEVELS), vmax=max(ACTIVE_Q_LEVELS))

    sns.set_theme(style="whitegrid", context="talk", font_scale=0.76)
    fig, ax = plt.subplots(figsize=(11.8, 7.2))
    for q_level, group in binned.groupby("Q_level", sort=True):
        group = group.sort_values("Ci")
        color = cmap(norm(float(q_level)))
        ax.scatter(group["Ci"], group["A_med"], color=color, s=22, alpha=0.55)
        grid_ci = np.linspace(group["Ci"].min(), group["Ci"].max(), 160)
        grid_qabs = np.full_like(grid_ci, group["Qabs"].median())
        fitted = fvcb_prediction(params, grid_ci, grid_qabs, free_alpha=True)
        ax.plot(grid_ci, fitted, color=color, lw=2.2, label=f"Q {q_level:.0f}")
    ax.axhline(0, color="#555555", lw=1, ls=":")
    ax.set_xlabel(r"Ci ($\mu$mol mol$^{-1}$)")
    ax.set_ylabel(r"A ($\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$)")
    ax.set_title("FvCB Surface Fit Overlay", loc="left", fontweight="bold")
    ax.text(
        0.03,
        0.96,
        f"Vcmax={row['Vcmax25_app_Ci']:.1f}, Jmax={row['Jmax25_app_Ci']:.1f}\n"
        f"Rd={row['Rd']:.1f}, alpha={row['alpha']:.2f}, theta=0.70\n"
        f"R2={row['r_squared']:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#333333",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92},
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OVERLAY_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    filtered = load_filtered_points()
    binned = binned_response(filtered)
    fits = fit_all(binned)
    literature = literature_values()
    fits.to_csv(FIT_SUMMARY_OUT, index=False)
    literature.to_csv(LITERATURE_OUT, index=False)
    plot_comparison(fits, literature)
    plot_overlay(binned, fits)

    print("Wrote:")
    for path in [FIT_SUMMARY_OUT, LITERATURE_OUT, COMPARISON_OUT, OVERLAY_OUT]:
        print(f"  {path}")
    print("\nFit summary:")
    print(fits.to_string(index=False))


if __name__ == "__main__":
    main()
