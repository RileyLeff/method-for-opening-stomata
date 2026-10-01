from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import brentq, curve_fit

from photo26.curves import read_licor_export


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"

RUNS = {
    "July 9": {
        "key": "july9",
        "points": OUTPUT_DIR / "morella_aqc_surface_points.csv",
        "points_flash": OUTPUT_DIR / "morella_aqc_surface_points_with_flash.csv",
        "flashes": OUTPUT_DIR / "morella_aqc_surface_flash_summary.csv",
        "color": "#666666",
    },
    "July 10 outside": {
        "key": "current",
        "points": OUTPUT_DIR / "morella_outside_regular_points.csv",
        "points_flash": OUTPUT_DIR / "morella_outside_regular_points_with_flash.csv",
        "flashes": OUTPUT_DIR / "morella_outside_regular_flash_summary.csv",
        "color": "#168f75",
    },
}
JUNE_AQ = DATA_DIR / "2026-06-15-1020_logdata_morcer_test1"
JUNE_ACI = DATA_DIR / "2026-06-15-1053_logdata_morcertest2_Acicurve (1)"

AQ_OUT = OUTPUT_DIR / "morella_aq_curves_comparison.png"
ACI_OUT = OUTPUT_DIR / "morella_aci_curves_comparison_full.png"
CAPACITY_OUT = OUTPUT_DIR / "morella_photorch_capacity_comparison.png"
AQ_SUMMARY_OUT = OUTPUT_DIR / "morella_aq_fit_summary.csv"
ETR_SUMMARY_OUT = OUTPUT_DIR / "morella_fluorescence_j_fit_summary.csv"
CAPACITY_SUMMARY_OUT = OUTPUT_DIR / "morella_capacity_summary.csv"
VARIABLE_J_OUT = OUTPUT_DIR / "morella_variable_j_diagnostic.csv"

Q_LEVELS = np.array([200, 500, 800, 1100, 1300, 1700, 2000], dtype=float)


def style_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#d8ddd4", linewidth=0.8)


def binned_curve(points: pd.DataFrame, q_level: float | None = None) -> pd.DataFrame:
    data = points.loc[
        points["sane_for_curve"].astype(bool)
        & points["Q_level"].gt(0)
        & points["Ci"].between(0, 1900)
    ].copy()
    if q_level is not None:
        data = data.loc[data["Q_level"].eq(q_level)]
    data["ci_bin"] = pd.cut(data["Ci"], np.arange(0, 1920 + 60, 60), include_lowest=True)
    grouped = ["Q_level", "ci_bin"] if q_level is None else ["ci_bin"]
    curve = (
        data.groupby(grouped, observed=True)
        .agg(
            n=("A", "size"),
            Ci=("Ci", "median"),
            A=("A", "median"),
            A_q25=("A", lambda values: values.quantile(0.25)),
            A_q75=("A", lambda values: values.quantile(0.75)),
        )
        .reset_index()
    )
    return curve.loc[curve["n"].ge(3)].sort_values("Ci")


def dark_flash_values(path: Path) -> tuple[float, float]:
    flashes = pd.read_csv(path)
    dark = flashes.loc[flashes["flash_kind"].eq("dark")].iloc[0]
    return float(dark["flr_A_dark"]), float(dark["flr_Fv/Fm"])


def fixed_ci_slice(points: pd.DataFrame, target_ci: float, dark_a: float) -> pd.DataFrame:
    binned = binned_curve(points)
    rows = [{"Q": 0.0, "A": dark_a, "target_Ci": target_ci}]
    for q_level, group in binned.groupby("Q_level", sort=True):
        group = group.sort_values("Ci")
        if not (group["Ci"].min() <= target_ci <= group["Ci"].max()):
            continue
        rows.append(
            {
                "Q": float(q_level),
                "A": float(np.interp(target_ci, group["Ci"], group["A"])),
                "target_Ci": target_ci,
            }
        )
    return pd.DataFrame(rows)


def nreh_response(q: np.ndarray, amax: float, phi: float, theta: float, rd: float) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    light_term = phi * q + amax
    discriminant = np.maximum(light_term**2 - 4 * theta * phi * q * amax, 0)
    return (light_term - np.sqrt(discriminant)) / (2 * theta) - rd


def fit_nreh(data: pd.DataFrame, label: str, curve_type: str) -> tuple[dict[str, float | str], pd.DataFrame]:
    q = data["Q"].to_numpy(dtype=float)
    observed = data["A"].to_numpy(dtype=float)
    params, covariance = curve_fit(
        nreh_response,
        q,
        observed,
        p0=[max(observed.max() + 2, 1), 0.06, 0.7, max(-observed[q.argmin()], 0.1)],
        bounds=([0.1, 0.001, 0.05, 0], [60, 0.3, 0.999, 10]),
        maxfev=100_000,
    )
    fitted = nreh_response(q, *params)
    total = np.sum((observed - observed.mean()) ** 2)
    r_squared = 1 - np.sum((observed - fitted) ** 2) / total
    standard_errors = np.sqrt(np.diag(covariance))

    amax, phi, theta, rd = params
    q_comp = np.nan
    if nreh_response(np.array([0.0]), *params)[0] * nreh_response(np.array([100_000.0]), *params)[0] <= 0:
        q_comp = brentq(lambda value: nreh_response(np.array([value]), *params)[0], 0, 100_000)

    def gross_response(value: float) -> float:
        return float(nreh_response(np.array([value]), *params)[0] + rd)

    q95 = np.nan
    if gross_response(1_000_000) >= 0.95 * amax:
        q95 = brentq(lambda value: gross_response(value) - 0.95 * amax, 0, 1_000_000)

    grid_q = np.linspace(0, 2200, 300)
    grid = pd.DataFrame({"Q": grid_q, "A": nreh_response(grid_q, *params)})
    row: dict[str, float | str] = {
        "curve": label,
        "curve_type": curve_type,
        "n_points": len(data),
        "Amax_gross": float(amax),
        "Amax_net_asymptote": float(amax - rd),
        "phi": float(phi),
        "theta": float(theta),
        "Rd": float(rd),
        "Qcomp": float(q_comp),
        "Q95_gross": float(q95),
        "r_squared": float(r_squared),
        "Amax_se": float(standard_errors[0]),
        "phi_se": float(standard_errors[1]),
        "theta_se": float(standard_errors[2]),
        "Rd_se": float(standard_errors[3]),
        "theta_at_bound": bool(theta < 0.051 or theta > 0.998),
    }
    return row, grid


def electron_transport(q: np.ndarray, jmax: float, alpha: float, theta: float) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    light_term = alpha * q + jmax
    discriminant = np.maximum(light_term**2 - 4 * theta * alpha * q * jmax, 0)
    return (light_term - np.sqrt(discriminant)) / (2 * theta)


def fluorescence_etr_fit(label: str, path: Path) -> tuple[dict[str, float | str], pd.DataFrame, pd.DataFrame]:
    flashes = pd.read_csv(path)
    active = flashes.loc[flashes["flash_kind"].eq("active")].copy()
    q = active["pre_qin"].to_numpy(dtype=float)
    active["Q_level"] = Q_LEVELS[np.argmin(np.abs(q[:, None] - Q_LEVELS[None, :]), axis=1)]
    summary = (
        active.groupby("Q_level", as_index=False)
        .agg(
            ETR_median=("flr_ETR", "median"),
            ETR_p90=("flr_ETR", lambda values: values.quantile(0.90)),
            PhiPSII_median=("flr_PhiPS2", "median"),
            NPQ_median=("flr_NPQ", "median"),
        )
    )
    params, covariance = curve_fit(
        electron_transport,
        summary["Q_level"],
        summary["ETR_p90"],
        p0=[80, 0.35, 0.7],
        bounds=([1, 0.01, 0.05], [500, 1.0, 0.999]),
        maxfev=100_000,
    )
    fitted = electron_transport(summary["Q_level"].to_numpy(), *params)
    total = np.sum((summary["ETR_p90"] - summary["ETR_p90"].mean()) ** 2)
    r_squared = 1 - np.sum((summary["ETR_p90"] - fitted) ** 2) / total
    errors = np.sqrt(np.diag(covariance))
    grid_q = np.linspace(0, 2200, 300)
    grid = pd.DataFrame({"Q": grid_q, "ETR": electron_transport(grid_q, *params)})
    row: dict[str, float | str] = {
        "run": label,
        "Jmax_fluorescence_p90": float(params[0]),
        "alpha_fluorescence": float(params[1]),
        "theta_fluorescence": float(params[2]),
        "Jmax_se": float(errors[0]),
        "alpha_se": float(errors[1]),
        "theta_se": float(errors[2]),
        "r_squared": float(r_squared),
        "theta_at_bound": bool(params[2] < 0.051 or params[2] > 0.998),
    }
    return row, summary, grid


def plot_aq_comparison(points: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.70)
    fig, axes = plt.subplots(2, 2, figsize=(14.6, 10.0))
    aq_rows: list[dict[str, float | str]] = []

    for ax, target_ci in zip(axes[0], [500, 800]):
        for label, config in RUNS.items():
            dark_a, _ = dark_flash_values(config["flashes"])
            data = fixed_ci_slice(points[label], target_ci, dark_a)
            fit_row, grid = fit_nreh(data, f"{label}, Ci={target_ci}", "surface_fixed_Ci")
            aq_rows.append(fit_row)
            ax.scatter(data["Q"], data["A"], color=config["color"], s=42, zorder=3)
            ax.plot(grid["Q"], grid["A"], color=config["color"], lw=2.4, label=label)
        ax.axhline(0, color="#555555", lw=0.9, ls=":")
        ax.set_title(f"Surface-Derived A/Q at Ci = {target_ci}", loc="left", fontweight="bold")
        ax.set_xlabel(r"Q ($\mu$mol photons m$^{-2}$ s$^{-1}$)")
        ax.set_ylabel(r"A ($\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$)")
        ax.set_xlim(0, 2200)
        ax.set_ylim(-2, 22)
        style_axes(ax)
    axes[0, 0].legend(frameon=False, loc="lower right", fontsize=9)

    june = read_licor_export(JUNE_AQ)
    june_data = june[["Q", "A"]].dropna().sort_values("Q")
    june_row, june_grid = fit_nreh(june_data, "June 15 conventional A/Q", "conventional_Ca500")
    aq_rows.append(june_row)
    ax_june = axes[1, 0]
    ax_june.scatter(june_data["Q"], june_data["A"], color="#b65d32", s=48, label="Observed")
    ax_june.plot(june_grid["Q"], june_grid["A"], color="#7f3b20", lw=2.5, label="NREH fit")
    ax_june.axhline(0, color="#555555", lw=0.9, ls=":")
    ax_june.set_title("Original June 15 A/Q at Ca = 500", loc="left", fontweight="bold")
    ax_june.set_xlabel(r"Q ($\mu$mol photons m$^{-2}$ s$^{-1}$)")
    ax_june.set_ylabel(r"A ($\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$)")
    ax_june.set_xlim(0, 2200)
    ax_june.set_ylim(-2, 22)
    ax_june.legend(frameon=False, fontsize=9)
    style_axes(ax_june)

    etr_rows: list[dict[str, float | str]] = []
    ax_etr = axes[1, 1]
    for label, config in RUNS.items():
        fit_row, summary, grid = fluorescence_etr_fit(label, config["flashes"])
        etr_rows.append(fit_row)
        ax_etr.scatter(summary["Q_level"], summary["ETR_p90"], color=config["color"], s=42)
        ax_etr.plot(grid["Q"], grid["ETR"], color=config["color"], lw=2.4, label=label)
    ax_etr.set_title("Fluorescence ETR Upper Envelope", loc="left", fontweight="bold")
    ax_etr.set_xlabel(r"Q ($\mu$mol photons m$^{-2}$ s$^{-1}$)")
    ax_etr.set_ylabel(r"90th percentile ETR ($\mu$mol e$^-$ m$^{-2}$ s$^{-1}$)")
    ax_etr.set_xlim(0, 2200)
    ax_etr.set_ylim(0, 85)
    ax_etr.legend(frameon=False, loc="lower right", fontsize=9)
    style_axes(ax_etr)

    fig.text(0.06, 0.982, "Morella Light Responses Across Runs", ha="left", va="top", fontsize=21, fontweight="bold")
    fig.text(
        0.06,
        0.948,
        "July curves are slices through DAT A/Q/C surfaces; June is a conventional A/Q curve. ETR fits use each Q stage's 90th percentile flash value.",
        ha="left",
        va="top",
        fontsize=10.2,
        color="#555555",
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.87, wspace=0.22, hspace=0.34)
    fig.savefig(AQ_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)

    aq_summary = pd.DataFrame(aq_rows)
    etr_summary = pd.DataFrame(etr_rows)
    aq_summary.to_csv(AQ_SUMMARY_OUT, index=False)
    etr_summary.to_csv(ETR_SUMMARY_OUT, index=False)
    return aq_summary, etr_summary


def plot_aci_comparison(points: dict[str, pd.DataFrame]) -> None:
    june = read_licor_export(JUNE_ACI).sort_values("Ci")
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.70)
    fig, axes = plt.subplots(2, 2, figsize=(14.6, 10.0))
    curve_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]
    selected_q = [800, 1100, 2000]

    for ax, q_level in zip(curve_axes, selected_q):
        for label, config in RUNS.items():
            curve = binned_curve(points[label], q_level)
            ax.fill_between(curve["Ci"], curve["A_q25"], curve["A_q75"], color=config["color"], alpha=0.12, linewidth=0)
            ax.plot(curve["Ci"], curve["A"], color=config["color"], lw=2.6, label=label)
        if q_level == 1100:
            ax.scatter(june["Ci"], june["A"], color="#b65d32", s=34, marker="D", label="June 15, Q=1000")
            ax.plot(june["Ci"], june["A"], color="#b65d32", lw=1.5, alpha=0.8)
        ax.axhline(0, color="#555555", lw=0.9, ls=":")
        ax.set_title(f"Q = {q_level}", loc="left", fontweight="bold")
        ax.set_xlabel(r"Ci ($\mu$mol mol$^{-1}$)")
        ax.set_ylabel(r"A / DAT Adyn ($\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$)")
        ax.set_xlim(0, 1900)
        ax.set_ylim(0, 21)
        style_axes(ax)
    axes[0, 0].legend(frameon=False, loc="lower right", fontsize=9)
    axes[0, 1].legend(frameon=False, loc="lower right", fontsize=9)

    ax_peak = axes[1, 1]
    for label, config in RUNS.items():
        binned = binned_curve(points[label])
        peaks = binned.groupby("Q_level", as_index=False).agg(A_peak=("A", "max"))
        ax_peak.plot(peaks["Q_level"], peaks["A_peak"], color=config["color"], marker="o", lw=2.6, label=label)
    ax_peak.scatter([1000], [june["A"].max()], color="#b65d32", marker="D", s=46, label="June 15 A/Ci")
    ax_peak.set_title("Observed Peak A Across Q", loc="left", fontweight="bold")
    ax_peak.set_xlabel(r"Q ($\mu$mol photons m$^{-2}$ s$^{-1}$)")
    ax_peak.set_ylabel("Peak 60-ppm-bin median A")
    ax_peak.set_ylim(0, 21)
    ax_peak.legend(frameon=False, loc="lower right", fontsize=9)
    style_axes(ax_peak)

    fig.text(0.06, 0.982, "Morella A/Ci Curves Across Runs", ha="left", va="top", fontsize=21, fontweight="bold")
    fig.text(
        0.06,
        0.948,
        "July lines are 60-ppm Ci-bin medians from the same regular DAT program; the June curve was a conventional stepped A/Ci at Q = 1000.",
        ha="left",
        va="top",
        fontsize=10.2,
        color="#555555",
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.87, wspace=0.22, hspace=0.34)
    fig.savefig(ACI_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_capacity_fits() -> pd.DataFrame:
    rows = []
    for run in ["june", "july9", "current"]:
        rows.append(pd.read_csv(OUTPUT_DIR / f"morella_photorch_{run}_primary_summary.csv"))
        oxygen_path = OUTPUT_DIR / f"morella_photorch_{run}_oxygen_sensitive_tpu_summary.csv"
        if oxygen_path.exists():
            rows.append(pd.read_csv(oxygen_path))
    capacities = pd.concat(rows, ignore_index=True)
    capacities["display"] = capacities["run"].map(
        {"june": "June 15", "july9": "July 9", "current": "July 10 outside"}
    )
    return capacities


def plot_capacity_comparison(capacities: pd.DataFrame) -> None:
    primary = capacities.loc[capacities["variant"].eq("primary")].copy()
    primary["display"] = pd.Categorical(primary["display"], ["June 15", "July 9", "July 10 outside"], ordered=True)
    primary = primary.sort_values("display")
    oxygen = capacities.loc[capacities["variant"].eq("oxygen_sensitive_tpu")].copy()
    colors = ["#b65d32", "#666666", "#168f75"]
    x = np.arange(len(primary))

    sns.set_theme(style="whitegrid", context="talk", font_scale=0.72)
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.4))
    panels = [
        (axes[0, 0], "Vcmax25", r"Vcmax$_{25}$", (0, 65)),
        (axes[0, 1], "Jmax25", r"Jmax$_{25}$", (0, 85)),
        (axes[1, 0], "TPU25", r"TPU$_{25}$", (0, 4.2)),
    ]
    for ax, column, title, ylim in panels:
        values = primary[column].to_numpy(dtype=float)
        ax.bar(x, values, color=colors, alpha=0.88, width=0.62, label="Primary FvCB")
        for index, value in enumerate(values):
            ax.text(index, value + 0.03 * ylim[1], f"{value:.1f}", ha="center", va="bottom", fontsize=9)
        for row in oxygen.itertuples():
            index = list(primary["display"].astype(str)).index(row.display)
            value = getattr(row, column)
            ax.scatter(index, value, marker="D", s=48, facecolor="white", edgecolor=colors[index], linewidth=1.7, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels(primary["display"])
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(r"$\mu$mol m$^{-2}$ s$^{-1}$")
        ax.set_ylim(*ylim)
        style_axes(ax)

    ax_shape = axes[1, 1]
    ax_shape.plot(x, primary["alpha_incident"], color="#2d6f9f", marker="o", lw=2.2, label=r"$\alpha$ (incident Q)")
    ax_shape.plot(x, primary["theta"], color="#a65f2c", marker="s", lw=2.2, label=r"$\theta$")
    ax_shape.set_xticks(x)
    ax_shape.set_xticklabels(primary["display"])
    ax_shape.set_ylim(0, 1.0)
    ax_shape.set_title("Fitted J(Q) Shape", loc="left", fontweight="bold")
    ax_shape.set_ylabel("Parameter value")
    ax_shape.legend(frameon=False, fontsize=9, loc="lower right")
    style_axes(ax_shape)

    fig.text(0.06, 0.982, "Morella PhoTorch Capacity Comparison", ha="left", va="top", fontsize=21, fontweight="bold")
    fig.text(
        0.06,
        0.948,
        "Primary bars: standard FvCB with Rd fixed to 1% of Vcmax. Diamonds: oxygen-sensitive TPU diagnostic. Fixed activation energies normalize capacities to 25 C.",
        ha="left",
        va="top",
        fontsize=10.2,
        color="#555555",
    )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.09, top=0.87, wspace=0.23, hspace=0.34)
    fig.savefig(CAPACITY_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def variable_j_diagnostic(capacities: pd.DataFrame) -> pd.DataFrame:
    gas_constant = 0.0083144598
    reference_temperature = 298.15
    gamma_activation = 37.83
    rows = []
    for label, config in RUNS.items():
        points = pd.read_csv(config["points_flash"], low_memory=False)
        flashes = points.loc[points["light_flash_logged"].astype(bool) & points["Q_level"].gt(0)].copy()
        capacity = capacities.loc[
            capacities["run"].eq(config["key"]) & capacities["variant"].eq("primary")
        ].iloc[0]
        rd = float(capacity["Rd_mean_at_measurement_T"])
        temperature = flashes["Tleaf"].to_numpy(dtype=float) + 273.15
        gamma = 42.75 * np.exp(
            gamma_activation / (gas_constant * reference_temperature)
            - gamma_activation / (gas_constant * temperature)
        )
        ratio = 4 * (flashes["A"].to_numpy(dtype=float) + rd) / flashes["ETR"].to_numpy(dtype=float)
        cc = gamma * (1 + 2 * ratio) / (1 - ratio)
        gm = flashes["A"].to_numpy(dtype=float) / (flashes["Ci"].to_numpy(dtype=float) - cc)
        valid = (
            (ratio > 0)
            & (ratio < 1)
            & (cc > 0)
            & (cc < flashes["Ci"].to_numpy(dtype=float))
            & (gm > 0)
            & (gm < 5)
        )
        rows.append(
            {
                "run": label,
                "flash_points": len(flashes),
                "variable_J_valid_points": int(valid.sum()),
                "valid_fraction": float(valid.mean()),
                "median_4AplusRd_over_ETR": float(np.nanmedian(ratio)),
                "gm_median_valid_only": float(np.nanmedian(gm[valid])) if valid.any() else np.nan,
                "interpretation": "insufficient for gm" if valid.mean() < 0.5 else "screened estimate only",
            }
        )
    diagnostic = pd.DataFrame(rows)
    diagnostic.to_csv(VARIABLE_J_OUT, index=False)
    return diagnostic


def capacity_summary(
    points: dict[str, pd.DataFrame],
    capacities: pd.DataFrame,
    etr_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for label, config in RUNS.items():
        frame = points[label]
        sane = frame.loc[frame["sane_for_curve"].astype(bool) & frame["Q_level"].gt(0)].copy()
        q_gsw = sane.groupby("Q_level")["gsw"].median()
        _, fvfm = dark_flash_values(config["flashes"])
        primary = capacities.loc[
            capacities["run"].eq(config["key"]) & capacities["variant"].eq("primary")
        ].iloc[0]
        oxygen = capacities.loc[
            capacities["run"].eq(config["key"]) & capacities["variant"].eq("oxygen_sensitive_tpu")
        ]
        etr = etr_summary.loc[etr_summary["run"].eq(label)].iloc[0]
        row = {
            "run": label,
            "observed_A_max": float(sane["A"].max()),
            "observed_Ci_max": float(sane["Ci"].max()),
            "gsw_phase_median_min": float(q_gsw.min()),
            "gsw_phase_median_max": float(q_gsw.max()),
            "FvFm_dark": fvfm,
            "Vcmax25_primary": float(primary["Vcmax25"]),
            "Jmax25_primary": float(primary["Jmax25"]),
            "TPU25_primary": float(primary["TPU25"]),
            "alpha_primary": float(primary["alpha_incident"]),
            "theta_primary": float(primary["theta"]),
            "FvCB_R2_primary": float(primary["r_squared"]),
            "Jmax_fluorescence_p90": float(etr["Jmax_fluorescence_p90"]),
        }
        if not oxygen.empty:
            row.update(
                {
                    "Vcmax25_oxygen_sensitive": float(oxygen.iloc[0]["Vcmax25"]),
                    "Jmax25_oxygen_sensitive": float(oxygen.iloc[0]["Jmax25"]),
                    "TPU25_oxygen_sensitive": float(oxygen.iloc[0]["TPU25"]),
                    "alphaG": float(oxygen.iloc[0]["alphaG"]),
                    "FvCB_R2_oxygen_sensitive": float(oxygen.iloc[0]["r_squared"]),
                }
            )
        rows.append(row)

    june_primary = capacities.loc[
        capacities["run"].eq("june") & capacities["variant"].eq("primary")
    ].iloc[0]
    june_aq = read_licor_export(JUNE_AQ)
    june_aci = read_licor_export(JUNE_ACI)
    rows.append(
        {
            "run": "June 15 original curves",
            "observed_A_max": float(max(june_aq["A"].max(), june_aci["A"].max())),
            "observed_Ci_max": float(june_aci["Ci"].max()),
            "gsw_phase_median_min": float(min(june_aq["gsw"].median(), june_aci["gsw"].median())),
            "gsw_phase_median_max": float(max(june_aq["gsw"].median(), june_aci["gsw"].median())),
            "FvFm_dark": np.nan,
            "Vcmax25_primary": float(june_primary["Vcmax25"]),
            "Jmax25_primary": float(june_primary["Jmax25"]),
            "TPU25_primary": float(june_primary["TPU25"]),
            "alpha_primary": float(june_primary["alpha_incident"]),
            "theta_primary": float(june_primary["theta"]),
            "FvCB_R2_primary": float(june_primary["r_squared"]),
            "Jmax_fluorescence_p90": np.nan,
        }
    )
    summary = pd.DataFrame(rows)
    summary.to_csv(CAPACITY_SUMMARY_OUT, index=False)
    return summary


def main() -> None:
    points = {label: pd.read_csv(config["points"], low_memory=False) for label, config in RUNS.items()}
    capacities = load_capacity_fits()
    aq_summary, etr_summary = plot_aq_comparison(points)
    plot_aci_comparison(points)
    plot_capacity_comparison(capacities)
    variable_j = variable_j_diagnostic(capacities)
    summary = capacity_summary(points, capacities, etr_summary)

    print("Wrote:")
    for path in [
        AQ_OUT,
        ACI_OUT,
        CAPACITY_OUT,
        AQ_SUMMARY_OUT,
        ETR_SUMMARY_OUT,
        CAPACITY_SUMMARY_OUT,
        VARIABLE_J_OUT,
    ]:
        print(f"  {path}")
    print("\nCapacity summary:")
    print(summary.round(3).to_string(index=False))
    print("\nA/Q fits:")
    print(aq_summary.round(3).to_string(index=False))
    print("\nVariable-J screen:")
    print(variable_j.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
