from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

import analyze_phrag_brownsville_biochemical_limits as core


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
METADATA_PATH = PROJECT_ROOT / "data" / "brownsville_phragmites_sample_metadata_2026.csv"
Q_LEVELS = core.Q_LEVELS
ZONE_COLORS = {"High": "#2475a8", "Low": "#d97728"}
STATUS_COLORS = {
    "include": "#2f8f46",
    "caution": "#c58a17",
    "exclude": "#c44e52",
    "partial": "#8b6bb1",
}

RUNS = [
    core.Run(
        code="P01",
        run_name="P01",
        time_window="Jul 24 11:04-13:56",
        zone="High",
        color=ZONE_COLORS["High"],
        dataset_dir=PROJECT_ROOT / "data" / "2026-07-24_phrag_brownsville_p01_biochemical_limits",
        log_name="2026-07-24-1101_logdata_phrag_brwns_P017242026",
    ),
    core.Run(
        code="P04",
        run_name="P04",
        time_window="Jul 24 14:16-17:09",
        zone="High",
        color=ZONE_COLORS["High"],
        dataset_dir=PROJECT_ROOT / "data" / "2026-07-24_phrag_brownsville_p04_biochemical_limits",
        log_name="2026-07-24-1415_logdata_Phra_brwns_P047242026",
    ),
    core.Run(
        code="P02",
        run_name="P02",
        time_window="Jul 26 10:49-13:41",
        zone="Low",
        color=ZONE_COLORS["Low"],
        dataset_dir=PROJECT_ROOT / "data" / "2026-07-26_phrag_brownsville_p02_biochemical_limits",
        log_name="2026-07-26-1046_logdata_phrag_brws_P027262026",
    ),
    core.Run(
        code="P03",
        run_name="P03",
        time_window="Jul 26 13:46-16:39",
        zone="Low",
        color=ZONE_COLORS["Low"],
        dataset_dir=PROJECT_ROOT / "data" / "2026-07-26_phrag_brownsville_p03_biochemical_limits",
        log_name="2026-07-26-1344_logdata_phrag_brwns_P037262026",
    ),
    core.Run(
        code="P05",
        run_name="P05",
        time_window="Jul 27 09:40-12:32",
        zone="Low",
        color=ZONE_COLORS["Low"],
        dataset_dir=PROJECT_ROOT / "data" / "2026-07-27_phrag_brownsville_p05_biochemical_limits",
        log_name="2026-07-27-0937_logdata_phrag_brwns_P057272026",
    ),
    core.Run(
        code="P06",
        run_name="P06",
        time_window="Jul 27 12:42-15:34",
        zone="Low",
        color=ZONE_COLORS["Low"],
        dataset_dir=PROJECT_ROOT / "data" / "2026-07-27_phrag_brownsville_p06_biochemical_limits",
        log_name="2026-07-27-1239_logdata_phrag_brws_P067272026",
    ),
    core.Run(
        code="P07",
        run_name="P07",
        time_window="Jul 27 15:44-18:36",
        zone="High",
        color=ZONE_COLORS["High"],
        dataset_dir=PROJECT_ROOT / "data" / "2026-07-27_phrag_brownsville_p07_biochemical_limits",
        log_name="2026-07-27-1541_logdata_phrag_brwns_P077272026",
    ),
    core.Run(
        code="P08",
        run_name="P08",
        time_window="Jul 28 14:37-17:30",
        zone="High",
        color=ZONE_COLORS["High"],
        dataset_dir=PROJECT_ROOT / "data" / "2026-07-28_phrag_brownsville_p08_biochemical_limits",
        log_name="2026-07-28-1435_logdata_phrag_brwns_P087282026",
    ),
    core.Run(
        code="P09",
        run_name="P09",
        time_window="Jul 28 17:36-20:29",
        zone="High",
        color=ZONE_COLORS["High"],
        dataset_dir=PROJECT_ROOT / "data" / "2026-07-28_phrag_brownsville_p09_biochemical_limits",
        log_name="2026-07-28-1734_logdata_phrag_brwns_P097282026",
    ),
    core.Run(
        code="P10",
        run_name="P10",
        time_window="Jul 30 07:32-11:06",
        zone="Low",
        color=ZONE_COLORS["Low"],
        dataset_dir=PROJECT_ROOT / "data" / "2026-07-30_phrag_brownsville_p10_biochemical_limits",
        log_name="2026-07-30-0731_logdata_phrag_brwns_P107302026",
    ),
    core.Run(
        code="P11",
        run_name="P11",
        time_window="Jul 30 11:08-14:02",
        zone="Low",
        color=ZONE_COLORS["Low"],
        dataset_dir=PROJECT_ROOT / "data" / "2026-07-30_phrag_brownsville_p11_biochemical_limits",
        log_name="2026-07-30-1108_logdata_pharg_brwns_P117302026",
    ),
    core.Run(
        code="P12",
        run_name="P12",
        time_window="Jul 30 14:14-17:07",
        zone="High",
        color=ZONE_COLORS["High"],
        dataset_dir=PROJECT_ROOT / "data" / "2026-07-30_phrag_brownsville_p12_biochemical_limits",
        log_name="2026-07-30-1413_logdata_phrag_brwns_P127302026",
    ),
    core.Run(
        code="P13",
        run_name="P13",
        time_window="Jul 30 17:16-20:09",
        zone="High",
        color=ZONE_COLORS["High"],
        dataset_dir=PROJECT_ROOT / "data" / "2026-07-30_phrag_brownsville_p13_biochemical_limits",
        log_name="2026-07-30-1715_logdata_phrag_brwns_P137302026",
    ),
    core.Run(
        code="P14",
        run_name="P14",
        time_window="Jul 31 10:18-13:11",
        zone="High",
        color=ZONE_COLORS["High"],
        dataset_dir=PROJECT_ROOT / "data" / "2026-07-31_phrag_brownsville_p14_biochemical_limits",
        log_name="2026-07-31-1016_logdata_phrag_brwns_P147312026",
    ),
    core.Run(
        code="P15",
        run_name="P15",
        time_window="Jul 31 13:15-16:07",
        zone="High",
        color=ZONE_COLORS["High"],
        dataset_dir=PROJECT_ROOT / "data" / "2026-07-31_phrag_brownsville_p15_biochemical_limits",
        log_name="2026-07-31-1313_logdata_phrag_brwns_P0157312026",
    ),
]

RUN_BY_CODE = {run.code: run for run in RUNS}

POINTS_OUT = OUTPUT_DIR / "phrag_brownsville_cohort_points.csv"
FLASHES_OUT = OUTPUT_DIR / "phrag_brownsville_cohort_flashes.csv"
CURVES_OUT = OUTPUT_DIR / "phrag_brownsville_cohort_binned_curves.csv"
Q_SUMMARY_OUT = OUTPUT_DIR / "phrag_brownsville_cohort_q_summary.csv"
FLUORESCENCE_OUT = OUTPUT_DIR / "phrag_brownsville_cohort_fluorescence_at_common_ci.csv"
JQ_OUT = OUTPUT_DIR / "phrag_brownsville_cohort_jq_parameters.csv"
QC_OUT = OUTPUT_DIR / "phrag_brownsville_cohort_run_qc.csv"
COUPLING_OUT = OUTPUT_DIR / "phrag_brownsville_sun_leaf_carbon_fluorescence_coupling.csv"

ACI_FIGURE_OUT = OUTPUT_DIR / "phrag_brownsville_cohort_aci_small_multiples.png"
ZONE_ACI_GRID_OUT = OUTPUT_DIR / "phrag_brownsville_h2or21_high_low_individual_aci_grid.png"
ZONE_ORDER_ACI_GRID_OUT = OUTPUT_DIR / "phrag_brownsville_zone_by_within_day_order_aci_grid.png"
COMPARISON_FIGURE_OUT = OUTPUT_DIR / "phrag_brownsville_cohort_zone_comparison.png"
QC_FIGURE_OUT = OUTPUT_DIR / "phrag_brownsville_cohort_qc_diagnostics.png"
CAPACITY_FIGURE_OUT = OUTPUT_DIR / "phrag_brownsville_cohort_capacity_summary.png"
ORDER_FIGURE_OUT = OUTPUT_DIR / "phrag_brownsville_cohort_measurement_order.png"
WITHIN_DAY_ORDER_FIGURE_OUT = OUTPUT_DIR / "phrag_brownsville_within_day_order_effect.png"
COUPLING_FIGURE_OUT = OUTPUT_DIR / "phrag_brownsville_sun_leaf_coupling_diagnostic.png"


def metadata_by_code() -> pd.DataFrame:
    metadata = pd.read_csv(METADATA_PATH).fillna("")
    metadata["run_code"] = metadata["plant_id"].str.slice(0, 3)
    metadata["zone"] = metadata["zone"].str.title()
    return metadata.set_index("run_code", drop=False)


def measured_phase_windows(points: pd.DataFrame) -> pd.DataFrame:
    return (
        points.loc[points["is_measured_ramp"]]
        .groupby("phase_id", sort=True)
        .agg(
            Q_level=("Q_level", "median"),
            start=("datetime", "min"),
            end=("datetime", "max"),
        )
        .reset_index()
    )


def best_regular_six(events: pd.DataFrame) -> list[int]:
    if len(events) <= 6:
        return events.index.tolist()
    best_indices: list[int] = []
    best_score = np.inf
    seconds = events["trigger_datetime"].astype("int64").to_numpy() / 1e9
    for selected in combinations(range(len(events)), 6):
        spacing = np.diff(seconds[list(selected)])
        score = float(np.abs(spacing - 90).sum() + 3 * np.maximum(60 - spacing, 0).sum())
        if score < best_score:
            best_score = score
            best_indices = events.iloc[list(selected)].index.tolist()
    return best_indices


def assign_standard_flashes(flashes: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    flashes = flashes.copy()
    flashes["seconds_since_previous"] = flashes["trigger_datetime"].diff().dt.total_seconds()
    flashes["assigned_Q"] = np.nan
    flashes["standard_flash"] = False
    phases = measured_phase_windows(points)

    for phase in phases.itertuples():
        events = flashes.loc[
            flashes["flash_kind"].eq("active")
            & flashes["trigger_datetime"].between(phase.start, phase.end)
        ].sort_values("trigger_datetime")
        if events.empty:
            continue
        flashes.loc[events.index, "assigned_Q"] = phase.Q_level
        nominal = events.loc[events["pre_qin"].sub(phase.Q_level).abs().lt(75)]
        selected = best_regular_six(nominal if len(nominal) >= 6 else events)
        flashes.loc[selected, "standard_flash"] = True
    flashes["Q_level_raw"] = flashes["Q_level"]
    flashes["Q_level"] = flashes["assigned_Q"].fillna(flashes["Q_level"])
    flashes["overlapping_flash"] = (
        flashes["flash_kind"].eq("active")
        & flashes["seconds_since_previous"].lt(60)
    )
    return flashes


def load_run(run: core.Run, metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    points = core.load_points(run)
    points["CO2_%"] = pd.to_numeric(points["CO2_%"], errors="coerce")
    points["co2_source_saturated"] = points["CO2_%"].ge(99.5)
    points["co2_delivery_ratio"] = points["CO2_r"] / points["CO2_r_sp"]
    points["co2_delivery_ok"] = (
        points["CO2_r_sp"].le(200)
        | points["co2_delivery_ratio"].ge(0.75)
    )
    points["valid_for_cohort"] = (
        points["clean_ramp"]
        & points["co2_delivery_ok"]
        & ~points["co2_source_saturated"]
    )
    row = metadata.loc[run.code]
    for column in ["illum", "canopy_depth", "canopy_height", "health", "notes"]:
        points[column] = row[column]

    flashes = assign_standard_flashes(core.load_flashes(run, points), points)
    for column in ["illum", "canopy_depth", "canopy_height", "health", "notes"]:
        flashes[column] = row[column]
    return points, flashes


def binned_curves(points: pd.DataFrame) -> pd.DataFrame:
    valid = points.loc[points["valid_for_cohort"]].copy()
    valid["ci_bin"] = pd.cut(valid["Ci"], np.arange(0, 2101, 50), include_lowest=True)
    binned = (
        valid.groupby(["run_code", "position", "Q_level", "ci_bin"], observed=True)
        .agg(
            n=("A", "size"),
            Ci=("Ci", "median"),
            Ca=("Ca", "median"),
            A=("A", "median"),
            A_q25=("A", lambda values: values.quantile(0.25)),
            A_q75=("A", lambda values: values.quantile(0.75)),
            gsw=("gsw", "median"),
            VPDleaf=("VPDleaf", "median"),
        )
        .reset_index()
    )
    return binned.loc[binned["n"].ge(3)].sort_values(["run_code", "Q_level", "Ci"])


def interpolate_value(group: pd.DataFrame, x_column: str, y_column: str, target: float) -> float:
    finite = group[[x_column, y_column]].replace([np.inf, -np.inf], np.nan).dropna()
    finite = finite.sort_values(x_column).drop_duplicates(x_column)
    if len(finite) < 2 or not finite[x_column].min() <= target <= finite[x_column].max():
        return np.nan
    return float(np.interp(target, finite[x_column], finite[y_column]))


def q_summary(points: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (code, q_level), group in points.loc[points["valid_for_cohort"]].groupby(["run_code", "Q_level"]):
        curve = curves.loc[curves["run_code"].eq(code) & curves["Q_level"].eq(q_level)]
        low_ca = group.loc[group["Ca"].between(150, 300)]
        ambient = group.loc[group["Ca"].between(380, 460)]
        rows.append(
            {
                "run_code": code,
                "zone": RUN_BY_CODE[code].zone,
                "Q_level": q_level,
                "n_clean": len(group),
                "Ci_max": group["Ci"].max(),
                "Ca_max": group["Ca"].max(),
                "A_curve_max": curve["A"].max(),
                "A_at_Ci400": interpolate_value(curve, "Ci", "A", 400),
                "A_at_Ci800": interpolate_value(curve, "Ci", "A", 800),
                "A_at_Ci1200": interpolate_value(curve, "Ci", "A", 1200),
                "A_at_Ca420": ambient["A"].median(),
                "Ci_at_Ca420": ambient["Ci"].median(),
                "gsw_low_Ca": low_ca["gsw"].median(),
                "gsw_at_Ca420": ambient["gsw"].median(),
                "VPDleaf_median": group["VPDleaf"].median(),
                "Tleaf_median": group["Tleaf"].median(),
            }
        )
    return pd.DataFrame(rows)


def fluorescence_at_common_ci(flashes: pd.DataFrame, target_ci: float = 800) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selected = flashes.loc[
        flashes["flash_kind"].eq("active")
        & flashes["standard_flash"]
        & flashes["gas_Ci"].notna()
        & flashes["PhiPS2"].between(0, 0.90)
        & flashes["ETR"].ge(0)
        & flashes["ETR"].le(0.46 * flashes["Q_level"])
    ].copy()
    for (code, q_level), group in selected.groupby(["run_code", "Q_level"]):
        fluorescence_qc_ok = code not in {"P02", "P05"}
        etr = interpolate_value(group, "gas_Ci", "ETR", target_ci)
        phi = interpolate_value(group, "gas_Ci", "PhiPS2", target_ci)
        npq = interpolate_value(group, "gas_Ci", "NPQ", target_ci)
        ql = interpolate_value(group, "gas_Ci", "qL", target_ci)
        rows.append(
            {
                "run_code": code,
                "zone": RUN_BY_CODE[code].zone,
                "Q_level": q_level,
                "target_Ci": target_ci,
                "n_flashes": len(group),
                "fluorescence_qc_ok": fluorescence_qc_ok,
                "ETR_at_Ci800_raw": etr,
                "PhiPS2_at_Ci800_raw": phi,
                "NPQ_at_Ci800_raw": npq,
                "qL_at_Ci800_raw": ql,
                "ETR_at_Ci800": etr if fluorescence_qc_ok else np.nan,
                "PhiPS2_at_Ci800": phi if fluorescence_qc_ok else np.nan,
                "NPQ_at_Ci800": npq if fluorescence_qc_ok else np.nan,
                "qL_at_Ci800": ql if fluorescence_qc_ok else np.nan,
            }
        )
    return pd.DataFrame(rows)


def nonrectangular_hyperbola(q: np.ndarray, alpha: float, jmax: float, theta: float) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    total = alpha * q + jmax
    discriminant = np.maximum(total**2 - 4 * theta * alpha * q * jmax, 0)
    return (total - np.sqrt(discriminant)) / (2 * theta)


def fit_jq_parameters(fluorescence: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for code, group in fluorescence.groupby("run_code"):
        data = group.dropna(subset=["ETR_at_Ci800"]).sort_values("Q_level")
        if len(data) < 7:
            continue
        try:
            params, covariance = curve_fit(
                nonrectangular_hyperbola,
                data["Q_level"],
                data["ETR_at_Ci800"],
                p0=[0.31, max(90, data["ETR_at_Ci800"].max() * 1.1), 0.75],
                bounds=([0.01, 20, 0.01], [1.0, 400, 0.999]),
                maxfev=30000,
            )
            predicted = nonrectangular_hyperbola(data["Q_level"].to_numpy(), *params)
            standard_error = np.sqrt(np.diag(covariance))
            rows.append(
                {
                    "run_code": code,
                    "zone": RUN_BY_CODE[code].zone,
                    "n_Q": len(data),
                    "alpha_ETR": params[0],
                    "Jmax_ETR": params[1],
                    "theta_ETR": params[2],
                    "alpha_se": standard_error[0],
                    "Jmax_se": standard_error[1],
                    "theta_se": standard_error[2],
                    "rmse_ETR": np.sqrt(np.mean((data["ETR_at_Ci800"] - predicted) ** 2)),
                }
            )
        except (RuntimeError, ValueError):
            continue
    return pd.DataFrame(rows)


def dark_fvfm(points: pd.DataFrame, flashes: pd.DataFrame) -> tuple[float, str]:
    dark = flashes.loc[flashes["flash_kind"].eq("dark") & flashes["FvFm"].gt(0), "FvFm"]
    if not dark.empty:
        return float(dark.iloc[0]), "event file"
    logged = points.loc[
        points["DarkAdaptedID"].astype(str).ne("-")
        & pd.to_numeric(points["Fv/Fm"], errors="coerce").gt(0),
        "Fv/Fm",
    ]
    if not logged.empty:
        return float(pd.to_numeric(logged, errors="coerce").iloc[0]), "log summary"
    return np.nan, "missing"


def qc_summary(
    all_points: dict[str, pd.DataFrame],
    all_flashes: dict[str, pd.DataFrame],
    q_summaries: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run in RUNS:
        points = all_points[run.code]
        flashes = all_flashes[run.code]
        active = flashes.loc[flashes["flash_kind"].eq("active")]
        selected = active.loc[active["standard_flash"]]
        fvfm, fvfm_source = dark_fvfm(points, flashes)
        q_data = q_summaries.loc[q_summaries["run_code"].eq(run.code)]
        complete_q = int(q_data.groupby("Q_level")["Ci_max"].max().ge(1400).sum())
        baseline = points.loc[~points["mpf_or_recovery"]]
        pulse = points.loc[points["mpf_or_recovery"]]

        status_class = "include"
        status = "include"
        primary = points["illum"].iloc[0].strip().lower() == "sun"
        if not primary:
            status_class = "partial"
            status = "pilot only: shade leaf excluded from sun comparison"
        if run.code == "P02":
            status_class = "caution"
            status = "include with caution: extra/overlapping MPFs"
        elif run.code == "P05":
            status_class = "exclude"
            status = "exclude primary comparison: severe MPF overlap"
            primary = False
        elif run.code == "P09":
            status_class = "partial"
            status = "partial only: CO2 source exhausted after Q200"
            primary = False
        elif run.code == "P15":
            status_class = "exclude"
            status = "quarantine: progressive within-run A and gsw loss"
            primary = False

        suspect_run = run.code in {"P05", "P09", "P15"}
        suspect_reason = ""
        if run.code == "P05":
            suspect_reason = "severe overlapping MPF perturbations"
        elif run.code == "P09":
            suspect_reason = "CO2 source exhausted after Q200"
        elif run.code == "P15":
            suspect_reason = (
                "progressive within-run loss of A and gsw across increasing Q; "
                "instrument controls and CO2 delivery remained stable"
            )

        rows.append(
            {
                "run_code": run.code,
                "plant_id": points["run_code"].iloc[0],
                "zone": run.zone,
                "illum": points["illum"].iloc[0],
                "canopy_height": points["canopy_height"].iloc[0],
                "time_window": run.time_window,
                "start_datetime": points["datetime"].min(),
                "start_hour": (
                    points["datetime"].min().hour
                    + points["datetime"].min().minute / 60
                    + points["datetime"].min().second / 3600
                ),
                "status_class": status_class,
                "status": status,
                "primary_comparison": primary,
                "suspect_run": suspect_run,
                "suspect_reason": suspect_reason,
                "duration_min": points["minute"].max() - points["minute"].min(),
                "complete_Q_levels_Cimax_ge_1400": complete_q,
                "active_flash_files": len(active),
                "selected_standard_flashes": len(selected),
                "extra_or_unassigned_active_flashes": len(active) - len(selected),
                "overlapping_active_flashes": int(active["overlapping_flash"].sum()),
                "dark_FvFm": fvfm,
                "dark_FvFm_source": fvfm_source,
                "VPDleaf_baseline_median": baseline["VPDleaf"].median(),
                "VPDleaf_baseline_p95": baseline["VPDleaf"].quantile(0.95),
                "VPDleaf_pulse_p95": pulse["VPDleaf"].quantile(0.95),
                "Tleaf_baseline_median": baseline["Tleaf"].median(),
                "Tleaf_pulse_p95": pulse["Tleaf"].quantile(0.95),
                "H2O_r_median": points["H2O_r"].median(),
                "Flow_median": points["Flow"].median(),
                "leaf_area_setting_cm2": points["S"].median(),
                "CO2_percent_max": points["CO2_%"].max(),
            }
        )
    return pd.DataFrame(rows)


def carbon_fluorescence_coupling(
    q_data: pd.DataFrame,
    fluorescence: pd.DataFrame,
    qc: pd.DataFrame,
) -> pd.DataFrame:
    gas = q_data[
        ["run_code", "zone", "Q_level", "A_at_Ci800", "gsw_low_Ca", "VPDleaf_median"]
    ]
    flr = fluorescence[
        ["run_code", "Q_level", "ETR_at_Ci800", "NPQ_at_Ci800", "PhiPS2_at_Ci800"]
    ]
    status = qc[
        ["run_code", "primary_comparison", "leaf_area_setting_cm2", "dark_FvFm"]
    ]
    result = gas.merge(flr, on=["run_code", "Q_level"], how="left").merge(
        status, on="run_code", how="left"
    )
    result = result.loc[result["primary_comparison"]].copy()

    gamma_star = 42.75
    rd_assumed = 1.0
    ci_target = 800.0
    carboxylation_fraction = (ci_target - gamma_star) / (ci_target + 2 * gamma_star)
    result["J_linked_A_expectation"] = (
        result["ETR_at_Ci800"] / 4 * carboxylation_fraction - rd_assumed
    )
    result["apparent_A_J_coupling"] = (
        (result["A_at_Ci800"] + rd_assumed)
        / (result["ETR_at_Ci800"] / 4 * carboxylation_fraction)
    )
    return result.sort_values(["zone", "run_code"])


def plot_aci_small_multiples(curves: pd.DataFrame, qc: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.64)
    columns = 3
    rows = int(np.ceil(len(RUNS) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(17, 4.05 * rows), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(rows, columns)
    q_norm = Normalize(vmin=Q_LEVELS.min(), vmax=Q_LEVELS.max())
    cmap = plt.get_cmap("viridis")
    order = [run.code for run in RUNS]

    for ax, code in zip(axes.flat, order):
        run = RUN_BY_CODE[code]
        run_curves = curves.loc[curves["run_code"].eq(code)]
        status = qc.loc[qc["run_code"].eq(code)].iloc[0]
        for q_level, group in run_curves.groupby("Q_level", sort=True):
            ax.plot(group["Ci"], group["A"], color=cmap(q_norm(q_level)), lw=1.8, alpha=0.95)
        ax.axhline(0, color="#555555", lw=0.8, ls=":")
        ax.set_title(
            f"{code} | {run.zone} | {status['illum']}",
            loc="left",
            color=ZONE_COLORS[run.zone],
            fontweight="bold",
        )
        ax.text(
            0.02,
            0.96,
            status["status"],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.4,
            color=STATUS_COLORS[status["status_class"]],
        )
        ax.set_xlim(0, 2000)
        ax.set_ylim(-2, 34)
        core.style_axes(ax)

    for ax in axes.flat[len(order):]:
        ax.set_visible(False)

    for ax in axes[-1, :]:
        ax.set_xlabel("Ci (umol mol-1)")
    for ax in axes[:, 0]:
        ax.set_ylabel("A / DAT Adyn")
    colorbar = fig.colorbar(
        ScalarMappable(norm=q_norm, cmap=cmap),
        ax=axes,
        pad=0.015,
        fraction=0.022,
    )
    colorbar.set_label("Q (umol photons m-2 s-1)")
    fig.text(
        0.055, 0.985,
        "Brownsville Phragmites A/Ci/Q Cohort",
        ha="left", va="top", fontsize=22, fontweight="bold",
    )
    fig.text(
        0.055, 0.956,
        "Median 50-ppm Ci bins after MPF recovery and CO2-delivery QC; sun and pilot shade leaves remain visible.",
        ha="left", va="top", fontsize=10.2, color="#555555",
    )
    fig.subplots_adjust(left=0.07, right=0.91, bottom=0.065, top=0.91, wspace=0.18, hspace=0.25)
    fig.savefig(ACI_FIGURE_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_zone_grouped_aci_grid(curves: pd.DataFrame, qc: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.64)
    high_codes = [run.code for run in RUNS if run.zone == "High"]
    low_codes = [run.code for run in RUNS if run.zone == "Low"]
    rows = max(int(np.ceil(len(high_codes) / 2)), int(np.ceil(len(low_codes) / 2)))
    fig, axes = plt.subplots(
        rows,
        4,
        figsize=(18, 3.55 * rows + 0.3),
        sharex=True,
        sharey=True,
    )
    axes = np.asarray(axes).reshape(rows, 4)
    q_norm = Normalize(vmin=Q_LEVELS.min(), vmax=Q_LEVELS.max())
    cmap = plt.get_cmap("viridis")
    status_by_code = qc.set_index("run_code")

    placements: dict[str, tuple[int, int]] = {}
    for index, code in enumerate(high_codes):
        placements[code] = (index // 2, index % 2)
    for index, code in enumerate(low_codes):
        placements[code] = (index // 2, 2 + index % 2)

    for ax in axes.flat:
        ax.set_visible(False)

    for code, (row, column) in placements.items():
        ax = axes[row, column]
        ax.set_visible(True)
        run = RUN_BY_CODE[code]
        status = status_by_code.loc[code]
        run_curves = curves.loc[curves["run_code"].eq(code)]
        for q_level, group in run_curves.groupby("Q_level", sort=True):
            ax.plot(
                group["Ci"],
                group["A"],
                color=cmap(q_norm(q_level)),
                lw=1.9,
                alpha=0.96,
            )
        ax.axhline(0, color="#555555", lw=0.8, ls=":")
        ax.set_title(
            f"{code} | {status['illum']}",
            loc="left",
            color=ZONE_COLORS[run.zone],
            fontweight="bold",
        )
        if status["status_class"] in {"caution", "exclude", "partial"}:
            ax.text(
                0.02,
                0.96,
                status["status"],
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.2,
                color=STATUS_COLORS[status["status_class"]],
            )
        ax.set_xlim(0, 2000)
        ax.set_ylim(-2, 34)
        core.style_axes(ax)

    for row in range(rows):
        for column in [0, 2]:
            if axes[row, column].get_visible():
                axes[row, column].set_ylabel("A / DAT Adyn")
    for column in range(4):
        visible = [axes[row, column] for row in range(rows) if axes[row, column].get_visible()]
        if visible:
            visible[-1].set_xlabel("Ci (umol mol-1)")

    colorbar = fig.colorbar(
        ScalarMappable(norm=q_norm, cmap=cmap),
        ax=axes,
        pad=0.012,
        fraction=0.018,
    )
    colorbar.set_label("Q (umol photons m-2 s-1)")
    fig.text(
        0.055,
        0.988,
        "Brownsville Phragmites Individual A/Ci/Q Curves",
        ha="left",
        va="top",
        fontsize=22,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.956,
        "All individuals measured under the fixed H2O_r = 21 mmol mol-1 regime; curves are median 50-ppm Ci bins after MPF-recovery and CO2-delivery QC.",
        ha="left",
        va="top",
        fontsize=9.8,
        color="#555555",
    )
    fig.text(
        0.265,
        0.925,
        "HIGH ZONE",
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=ZONE_COLORS["High"],
    )
    fig.text(
        0.685,
        0.925,
        "LOW ZONE",
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=ZONE_COLORS["Low"],
    )
    fig.subplots_adjust(left=0.07, right=0.91, bottom=0.07, top=0.88, wspace=0.18, hspace=0.27)
    fig.savefig(ZONE_ACI_GRID_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_zone_by_order_aci_grid(curves: pd.DataFrame, qc: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.60)
    fig = plt.figure(figsize=(22, 15.5))
    grid = fig.add_gridspec(
        4,
        10,
        left=0.075,
        right=0.925,
        bottom=0.155,
        top=0.84,
        width_ratios=[1, 0.22, 1, 1, 0.44, 1, 1, 1, 0.22, 1],
        wspace=0.16,
        hspace=0.33,
    )
    q_norm = Normalize(vmin=Q_LEVELS.min(), vmax=Q_LEVELS.max())
    cmap = plt.get_cmap("viridis")
    status = qc.copy()
    status["start_datetime"] = pd.to_datetime(status["start_datetime"])
    status["date_label"] = status["start_datetime"].dt.strftime("%m/%d")
    status["day_order"] = (
        status.groupby(status["start_datetime"].dt.date)["start_datetime"]
        .rank(method="first")
        .astype(int)
    )
    status = status.sort_values("start_datetime")
    status = status.loc[~status["suspect_run"]].copy()
    status["illum_normalized"] = status["illum"].str.strip().str.lower()
    status_by_code = status.set_index("run_code")
    first_axis: plt.Axes | None = None
    plotted_axes: list[plt.Axes] = []
    subcolumns = [
        ("High", "sun", (0,)),
        ("High", "shade", (2, 3)),
        ("Low", "sun", (5, 6, 7)),
        ("Low", "shade", (9,)),
    ]

    for row, day_order in enumerate([1, 2, 3, 4]):
        for zone, illumination, data_columns in subcolumns:
            codes = status.loc[
                status["day_order"].eq(day_order)
                & status["zone"].eq(zone)
                & status["illum_normalized"].eq(illumination),
                "run_code",
            ].tolist()
            if not codes:
                ax = fig.add_subplot(grid[row, data_columns[0]:data_columns[-1] + 1])
                ax.set_axis_off()
                ax.text(
                    0.5,
                    0.5,
                    "No sample",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="#999999",
                )
                continue

            if len(codes) > len(data_columns):
                raise ValueError(
                    f"Not enough fixed slots for {zone} {illumination} order {day_order}: {codes}"
                )

            for index, code in enumerate(codes):
                ax = fig.add_subplot(
                    grid[row, data_columns[index]],
                    sharex=first_axis,
                    sharey=first_axis,
                )
                if first_axis is None:
                    first_axis = ax
                plotted_axes.append(ax)
                run_curves = curves.loc[curves["run_code"].eq(code)]
                run_status = status_by_code.loc[code]
                for q_level, group in run_curves.groupby("Q_level", sort=True):
                    ax.plot(
                        group["Ci"],
                        group["A"],
                        color=cmap(q_norm(q_level)),
                        lw=1.65,
                        alpha=0.96,
                    )
                ax.axhline(0, color="#555555", lw=0.7, ls=":")
                ax.set_title(
                    f"{code} | {run_status['date_label']}",
                    loc="left",
                    color=ZONE_COLORS[zone],
                    fontweight="bold",
                    fontsize=9.2,
                )
                if run_status["status_class"] in {"caution", "exclude", "partial"}:
                    ax.text(
                        0.02,
                        0.96,
                        run_status["status"],
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=6.0,
                        color=STATUS_COLORS[run_status["status_class"]],
                    )
                ax.set_xlim(0, 2000)
                ax.set_ylim(-2, 34)
                ax.set_yticks(np.arange(0, 31, 5))
                core.style_axes(ax)
                if index > 0:
                    ax.tick_params(axis="y", labelleft=False)
                x_tick_labels = ax.get_xticklabels()
                if index > 0 and x_tick_labels:
                    x_tick_labels[0].set_visible(False)
                if index < len(codes) - 1 and x_tick_labels:
                    x_tick_labels[-1].set_visible(False)

            for unused_column in data_columns[len(codes):]:
                ax = fig.add_subplot(grid[row, unused_column])
                ax.set_axis_off()

    colorbar_axis = fig.add_axes([0.31, 0.048, 0.38, 0.014])
    colorbar = fig.colorbar(
        ScalarMappable(norm=q_norm, cmap=cmap),
        cax=colorbar_axis,
        orientation="horizontal",
        ticks=[50, 500, 1000, 1500, 2000],
    )
    colorbar.set_label("Incident light, Q (umol photons m-2 s-1)", labelpad=7)
    colorbar.outline.set_linewidth(0.7)
    fig.text(
        0.055,
        0.987,
        "Brownsville Phragmites A/Ci/Q by Zone and Measurement Order",
        ha="left",
        va="top",
        fontsize=22,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.955,
        "Rows show ordinal measurement position within each date. Within each zone, sun is left and shade is right; suspect runs P05, P09, and P15 are excluded.",
        ha="left",
        va="top",
        fontsize=9.8,
        color="#555555",
    )
    def column_span(start: int, end: int) -> tuple[float, float]:
        return (
            grid[0, start].get_position(fig).x0,
            grid[0, end].get_position(fig).x1,
        )

    high_left, high_right = column_span(0, 3)
    low_left, low_right = column_span(5, 9)
    zone_headers = [
        (high_left, high_right, "HIGH ZONE", ZONE_COLORS["High"]),
        (low_left, low_right, "LOW ZONE", ZONE_COLORS["Low"]),
    ]
    for left, right, label, color in zone_headers:
        fig.text(
            (left + right) / 2,
            0.915,
            label,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color=color,
        )
        fig.add_artist(
            Line2D(
                [left, right],
                [0.897, 0.897],
                transform=fig.transFigure,
                color=color,
                lw=2.2,
                alpha=0.9,
                solid_capstyle="round",
            )
        )

    subheaders = [
        (*column_span(0, 0), "SUN"),
        (*column_span(2, 3), "SHADE"),
        (*column_span(5, 7), "SUN"),
        (*column_span(9, 9), "SHADE"),
    ]
    for left, right, label in subheaders:
        fig.text(
            (left + right) / 2,
            0.872,
            label,
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color="#4f5559",
        )

    separator_specs = [
        (1, "#c7cdd1", 0.9, 0.895),
        (4, "#7d878d", 1.6, 0.928),
        (8, "#c7cdd1", 0.9, 0.895),
    ]
    for spacer_column, color, width, y_top in separator_specs:
        spacer = grid[0, spacer_column].get_position(fig)
        separator_x = (spacer.x0 + spacer.x1) / 2
        fig.add_artist(
            Line2D(
                [separator_x, separator_x],
                [0.14, y_top],
                transform=fig.transFigure,
                color=color,
                lw=width,
                alpha=0.9,
                zorder=0,
            )
        )

    row_centers = [0.744, 0.553, 0.362, 0.171]
    for order, y_position in enumerate(row_centers, start=1):
        fig.text(
            0.035,
            y_position,
            f"ORDER {order}",
            ha="center",
            va="center",
            rotation=90,
            fontsize=11,
            fontweight="bold",
            color="#555555",
        )
    fig.text(0.50, 0.105, "Ci (umol mol-1)", ha="center", va="center", fontsize=12)
    fig.text(0.015, 0.48, "A / DAT Adyn", ha="center", va="center", rotation=90, fontsize=12)
    fig.savefig(ZONE_ORDER_ACI_GRID_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def add_zone_medians(ax: plt.Axes, data: pd.DataFrame, metric: str, qc: pd.DataFrame) -> None:
    primary_codes = set(qc.loc[qc["primary_comparison"], "run_code"])
    primary = data.loc[data["run_code"].isin(primary_codes)]
    for zone, group in primary.groupby("zone"):
        median = group.groupby("Q_level")[metric].median().dropna()
        ax.plot(
            median.index,
            median.values,
            color=ZONE_COLORS[zone],
            lw=4.0,
            alpha=0.95,
            label=f"{zone} median (QC set)",
        )


def plot_zone_comparison(
    q_data: pd.DataFrame,
    fluorescence: pd.DataFrame,
    qc: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.72)
    fig, axes = plt.subplots(2, 2, figsize=(15.8, 10.2))
    status_by_code = qc.set_index("run_code")
    comparison_codes = set(qc.loc[~qc["suspect_run"], "run_code"])

    specifications = [
        (axes[0, 0], q_data, "A_at_Ci800", "A at common Ci = 800", "A"),
        (axes[0, 1], fluorescence, "ETR_at_Ci800", "ETR at common Ci = 800", "ETR"),
        (axes[1, 0], q_data, "gsw_low_Ca", "gsw during low-CO2 portions", "gsw"),
        (axes[1, 1], fluorescence, "NPQ_at_Ci800", "NPQ at common Ci = 800", "NPQ"),
    ]
    for ax, data, metric, title, ylabel in specifications:
        for code, group in data.loc[data["run_code"].isin(comparison_codes)].groupby("run_code"):
            status = status_by_code.loc[code]
            linestyle = "-" if status["primary_comparison"] else ":"
            ax.plot(
                group["Q_level"],
                group[metric],
                color=ZONE_COLORS[status["zone"]],
                lw=1.25,
                marker="o",
                ms=3.5,
                alpha=0.48 if status["primary_comparison"] else 0.65,
                ls=linestyle,
            )
            last = group.dropna(subset=[metric]).sort_values("Q_level")
            if not last.empty:
                endpoint = last.iloc[-1]
                ax.annotate(
                    code,
                    (endpoint["Q_level"], endpoint[metric]),
                    xytext=(5, 0),
                    textcoords="offset points",
                    fontsize=7.5,
                    color=ZONE_COLORS[status["zone"]],
                    va="center",
                )
        add_zone_medians(ax, data, metric, qc)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Q (umol photons m-2 s-1)")
        ax.set_ylabel(ylabel)
        core.style_axes(ax)

    axes[0, 0].axhline(0, color="#555555", lw=0.8, ls=":")
    axes[1, 0].set_ylim(0, 0.18)
    handles = [
        Line2D([0], [0], color=ZONE_COLORS["High"], lw=3, label="High zone"),
        Line2D([0], [0], color=ZONE_COLORS["Low"], lw=3, label="Low zone"),
        Line2D([0], [0], color="#555555", lw=1.5, ls=":", label="QC-excluded/partial"),
    ]
    axes[0, 0].legend(handles=handles, frameon=False, fontsize=8.5, loc="upper left")
    fig.text(
        0.055, 0.985,
        "Brownsville Phragmites: Sun-Leaf Zone Comparison at Matched Ci",
        ha="left", va="top", fontsize=22, fontweight="bold",
    )
    fig.text(
        0.055, 0.95,
        "Thick lines are sun-leaf medians (high n=2, low n=5); thin dotted lines are pilot shade or QC-excluded leaves.",
        ha="left", va="top", fontsize=10.2, color="#555555",
    )
    fig.subplots_adjust(left=0.075, right=0.95, bottom=0.08, top=0.89, hspace=0.34, wspace=0.23)
    fig.savefig(COMPARISON_FIGURE_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_qc_diagnostics(
    all_points: dict[str, pd.DataFrame],
    qc: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.70)
    fig, axes = plt.subplots(2, 2, figsize=(15.6, 10))
    order = [run.code for run in RUNS]
    x = np.arange(len(order))
    qc_ordered = qc.set_index("run_code").loc[order]

    ax = axes[0, 0]
    ax.scatter(x, qc_ordered["VPDleaf_baseline_median"], color="#7b4a3b", s=48, label="baseline median")
    ax.scatter(x, qc_ordered["VPDleaf_pulse_p95"], color="#d95f70", s=48, marker="s", label="MPF-window p95")
    ax.axhline(1, color="#333333", lw=1, ls="--")
    ax.set_xticks(x, order)
    ax.set_ylabel("VPDleaf (kPa)")
    ax.set_title("Moisture Control", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5)
    core.style_axes(ax)

    ax = axes[0, 1]
    ax.bar(x - 0.18, qc_ordered["selected_standard_flashes"], width=0.36, color="#2f8f46", label="regular selected")
    ax.bar(x + 0.18, qc_ordered["extra_or_unassigned_active_flashes"], width=0.36, color="#c44e52", label="extra/unassigned")
    ax.axhline(54, color="#333333", lw=1, ls="--", label="expected active MPFs")
    ax.set_xticks(x, order)
    ax.set_ylabel("active flash files")
    ax.set_title("Fluorometry Event Audit", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5)
    core.style_axes(ax)

    p09 = all_points["P09"]
    ax = axes[1, 0]
    ax.plot(p09["minute"], p09["CO2_r_sp"], color="#333333", lw=1.0, label="CO2_r setpoint")
    ax.plot(p09["minute"], p09["CO2_r"], color="#f28e2b", lw=1.1, label="delivered CO2_r")
    ax2 = ax.twinx()
    ax2.plot(p09["minute"], p09["CO2_%"], color="#c44e52", lw=1.0, label="CO2 actuator")
    ax2.axhline(100, color="#c44e52", lw=0.8, ls=":")
    ax.set_title("P09: CO2 Source Exhaustion", loc="left", fontweight="bold")
    ax.set_xlabel("elapsed time (min)")
    ax.set_ylabel("CO2 (ppm)")
    ax2.set_ylabel("CO2 actuator (%)")
    core.style_axes(ax)
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, frameon=False, fontsize=8, loc="upper left")

    p05 = all_points["P05"]
    ax = axes[1, 1]
    p05_ramp = p05.loc[p05["is_measured_ramp"] & p05["A"].between(-10, 40)]
    ax.plot(p05_ramp["minute"], p05_ramp["A"], color="#2f8f46", lw=0.8, alpha=0.75)
    pulse = p05_ramp.loc[p05_ramp["mpf_pulse"]]
    ax.scatter(pulse["minute"], pulse["A"], color="#c44e52", s=12, alpha=0.8, label="logged MPF pulse")
    ax.axhline(0, color="#555555", lw=0.8, ls=":")
    ax.set_title("P05: Dense MPF Perturbations", loc="left", fontweight="bold")
    ax.set_xlabel("elapsed time (min)")
    ax.set_ylabel("A / DAT Adyn")
    ax.legend(frameon=False, fontsize=8.5)
    core.style_axes(ax)

    fig.text(
        0.055, 0.985,
        "Brownsville Phragmites Run Quality",
        ha="left", va="top", fontsize=22, fontweight="bold",
    )
    fig.subplots_adjust(left=0.075, right=0.92, bottom=0.08, top=0.91, hspace=0.32, wspace=0.25)
    fig.savefig(QC_FIGURE_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_capacity_summary(
    q_data: pd.DataFrame,
    jq: pd.DataFrame,
    qc: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.72)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    status = qc.set_index("run_code")
    q2000 = q_data.loc[q_data["Q_level"].eq(2000)].set_index("run_code")
    jq_indexed = jq.set_index("run_code")
    metrics = [
        ("A_at_Ci800", q2000, "A at Q2000, Ci800"),
        ("Jmax_ETR", jq_indexed, "Fluorescence Jmax at Ci800"),
        ("theta_ETR", jq_indexed, "Fluorescence theta at Ci800"),
    ]
    rng = np.random.default_rng(20260729)
    for ax, (metric, table, title) in zip(axes, metrics):
        for zone_index, zone in enumerate(["High", "Low"]):
            codes = [code for code in table.index if status.loc[code, "zone"] == zone]
            for code in codes:
                if bool(status.loc[code, "suspect_run"]):
                    continue
                value = table.loc[code, metric]
                if pd.isna(value):
                    continue
                primary = bool(status.loc[code, "primary_comparison"])
                ax.scatter(
                    zone_index + rng.uniform(-0.08, 0.08),
                    value,
                    color=ZONE_COLORS[zone],
                    s=62,
                    marker="o" if primary else "X",
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=3,
                )
                ax.annotate(code, (zone_index, value), xytext=(8, 0), textcoords="offset points", fontsize=7.5)
        ax.set_xticks([0, 1], ["High", "Low"])
        ax.set_title(title, loc="left", fontweight="bold")
        core.style_axes(ax)
    axes[2].set_ylim(0, 1)
    fig.text(
        0.055, 0.985,
        "Leaf-Level Capacity Variation",
        ha="left", va="top", fontsize=21, fontweight="bold",
    )
    fig.text(
        0.055, 0.94,
        "X markers are QC-excluded runs; leaf-to-leaf variation dominates this first small cohort.",
        ha="left", va="top", fontsize=10.0, color="#555555",
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.12, top=0.82, wspace=0.28)
    fig.savefig(CAPACITY_FIGURE_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_carbon_fluorescence_coupling(coupling: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.76)
    fig, axes = plt.subplots(1, 3, figsize=(18.2, 6.1))
    q2000 = coupling.loc[coupling["Q_level"].eq(2000)]

    ax = axes[0]
    valid = q2000.dropna(subset=["ETR_at_Ci800", "A_at_Ci800"])
    etr_range = np.linspace(60, max(155, valid["ETR_at_Ci800"].max() + 5), 150)
    gamma_star = 42.75
    ci_target = 800.0
    fraction = (ci_target - gamma_star) / (ci_target + 2 * gamma_star)
    ax.plot(
        etr_range,
        etr_range / 4 * fraction - 1,
        color="#555555",
        lw=1.4,
        ls="--",
        label="J-linked expectation",
    )
    for row in valid.itertuples():
        ax.scatter(
            row.ETR_at_Ci800,
            row.A_at_Ci800,
            color=ZONE_COLORS[row.zone],
            s=78,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.annotate(
            f"{row.run_code}  ({row.apparent_A_J_coupling:.2f})",
            (row.ETR_at_Ci800, row.A_at_Ci800),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
            color=ZONE_COLORS[row.zone],
        )
    ax.set_xlabel("ETR at Q2000, Ci800")
    ax.set_ylabel("A at Q2000, Ci800")
    ax.set_title("Carbon Gain per Unit Electron Transport", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    core.style_axes(ax)

    ax = axes[1]
    for code, group in coupling.dropna(subset=["apparent_A_J_coupling"]).groupby("run_code"):
        row = group.iloc[-1]
        ax.plot(
            group["Q_level"],
            group["apparent_A_J_coupling"],
            color=ZONE_COLORS[row["zone"]],
            lw=1.5,
            marker="o",
            ms=3.5,
            alpha=0.8,
        )
        ax.annotate(
            code,
            (row["Q_level"], row["apparent_A_J_coupling"]),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=8,
            color=ZONE_COLORS[row["zone"]],
            va="center",
        )
    ax.axhline(1, color="#555555", lw=1.1, ls="--")
    ax.set_xlabel("Q (umol photons m-2 s-1)")
    ax.set_ylabel("apparent A/J coupling")
    ax.set_title("Coupling Across Light", loc="left", fontweight="bold")
    core.style_axes(ax)

    ax = axes[2]
    for row in q2000.itertuples():
        ax.scatter(
            row.gsw_low_Ca,
            row.A_at_Ci800,
            color=ZONE_COLORS[row.zone],
            s=78,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.annotate(
            row.run_code,
            (row.gsw_low_Ca, row.A_at_Ci800),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
            color=ZONE_COLORS[row.zone],
        )
    ax.set_xlabel("gsw during low-CO2 ramp")
    ax.set_ylabel("A at Q2000, Ci800")
    ax.set_title("A Tracks Stomatal Opening Across Leaves", loc="left", fontweight="bold")
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=ZONE_COLORS[zone], label=f"{zone} zone")
        for zone in ["High", "Low"]
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="upper left")
    core.style_axes(ax)

    fig.text(
        0.055,
        0.985,
        "Sun-Leaf Carbon–Fluorescence Coupling Diagnostic",
        ha="left",
        va="top",
        fontsize=21,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.94,
        "Parentheses show apparent A/J coupling. Leaves filled the 6 cm² chamber, so low values indicate physiological or gas-exchange/fluorescence decoupling.",
        ha="left",
        va="top",
        fontsize=9.7,
        color="#555555",
    )
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.12, top=0.82, wspace=0.25)
    fig.savefig(COUPLING_FIGURE_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_measurement_order(
    q_data: pd.DataFrame,
    jq: pd.DataFrame,
    qc: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.72)
    order = qc.sort_values("start_datetime").reset_index(drop=True).copy()
    order["sequence"] = np.arange(len(order))
    q2000 = q_data.loc[q_data["Q_level"].eq(2000)].set_index("run_code")
    jq_indexed = jq.set_index("run_code")
    metrics = [
        ("A_at_Ci800", q2000, "A at Q2000, Ci800"),
        ("gsw_low_Ca", q2000, "gsw during low-CO2 ramp"),
        ("Jmax_ETR", jq_indexed, "Fluorescence Jmax at Ci800"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), sharex=True)
    for ax, (metric, table, title) in zip(axes, metrics):
        plotted: list[tuple[int, float]] = []
        for row in order.itertuples():
            if row.run_code not in table.index or pd.isna(table.loc[row.run_code, metric]):
                continue
            value = float(table.loc[row.run_code, metric])
            ax.scatter(
                row.sequence,
                value,
                color=ZONE_COLORS[row.zone],
                s=70,
                marker="o" if row.primary_comparison else "X",
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            plotted.append((row.sequence, value))
        if plotted:
            ax.plot(
                [item[0] for item in plotted],
                [item[1] for item in plotted],
                color="#8c8c8c",
                linewidth=1.0,
                alpha=0.45,
                zorder=1,
            )
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xticks(
            order["sequence"],
            [
                f"{row.run_code}\n{pd.Timestamp(row.start_datetime):%m/%d %H:%M}"
                for row in order.itertuples()
            ],
            rotation=45,
            ha="right",
        )
        ax.tick_params(axis="x", labelsize=7.5)
        core.style_axes(ax)

    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=ZONE_COLORS[zone], label=f"{zone} zone")
        for zone in ["High", "Low"]
    ]
    handles.append(
        Line2D([0], [0], marker="X", linestyle="", color="#666666", label="QC excluded/partial")
    )
    axes[2].legend(handles=handles, frameon=False, fontsize=8, loc="best")
    fig.text(
        0.055,
        0.985,
        "Capacity Across Measurement Order",
        ha="left",
        va="top",
        fontsize=21,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.94,
        "July 30 alternates high- and low-capacity leaves, weakening a simple clock-time explanation for the variation.",
        ha="left",
        va="top",
        fontsize=10.0,
        color="#555555",
    )
    fig.subplots_adjust(left=0.065, right=0.98, bottom=0.24, top=0.82, wspace=0.28)
    fig.savefig(ORDER_FIGURE_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_within_day_order(
    q_data: pd.DataFrame,
    jq: pd.DataFrame,
    qc: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.74)
    status = qc.copy()
    status["start_datetime"] = pd.to_datetime(status["start_datetime"])
    status["date_label"] = status["start_datetime"].dt.strftime("%m/%d")
    status["day_order"] = (
        status.groupby(status["start_datetime"].dt.date)["start_datetime"]
        .rank(method="first")
        .astype(int)
    )
    status = status.loc[~status["suspect_run"]]
    q2000 = q_data.loc[q_data["Q_level"].eq(2000)].merge(
        status[["run_code", "zone", "date_label", "day_order"]],
        on=["run_code", "zone"],
    )
    jq_ordered = jq.merge(
        status[["run_code", "zone", "date_label", "day_order"]],
        on=["run_code", "zone"],
    )

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.9), sharex=True)
    specifications = [
        (axes[0], q2000, "A_at_Ci800", "A at Q2000, Ci800"),
        (axes[1], q2000, "gsw_low_Ca", "gsw during low-CO2 ramp"),
        (axes[2], jq_ordered, "Jmax_ETR", "Fluorescence Jmax at Ci800"),
    ]
    for ax, data, metric, title in specifications:
        for date_label, group in data.groupby("date_label", sort=False):
            group = group.sort_values("day_order")
            ax.plot(
                group["day_order"],
                group[metric],
                color="#a5a5a5",
                lw=1.2,
                alpha=0.7,
                zorder=1,
            )
            for row in group.itertuples():
                ax.scatter(
                    row.day_order,
                    getattr(row, metric),
                    color=ZONE_COLORS[row.zone],
                    s=72,
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=3,
                )
                ax.annotate(
                    f"{row.run_code}\n{date_label}",
                    (row.day_order, getattr(row, metric)),
                    xytext=(5, 3),
                    textcoords="offset points",
                    fontsize=7.3,
                    color=ZONE_COLORS[row.zone],
                )
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel("measurement order within day")
        ax.set_title(title, loc="left", fontweight="bold")
        core.style_axes(ax)

    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=ZONE_COLORS[zone], label=f"{zone} zone")
        for zone in ["High", "Low"]
    ]
    axes[2].legend(handles=handles, frameon=False, fontsize=8.5, loc="best")
    fig.text(
        0.055,
        0.985,
        "Phragmites Capacity by Within-Day Measurement Order",
        ha="left",
        va="top",
        fontsize=21,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.94,
        "Lines connect clean measurements from the same date. P12's rebound rules out simple instrument drift, but most matched pairs decline sharply.",
        ha="left",
        va="top",
        fontsize=9.8,
        color="#555555",
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.13, top=0.81, wspace=0.27)
    fig.savefig(WITHIN_DAY_ORDER_FIGURE_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    metadata = metadata_by_code()
    all_points: dict[str, pd.DataFrame] = {}
    all_flashes: dict[str, pd.DataFrame] = {}
    curve_frames: list[pd.DataFrame] = []

    for run in RUNS:
        points, flashes = load_run(run, metadata)
        all_points[run.code] = points
        all_flashes[run.code] = flashes
        curve_frames.append(binned_curves(points))

    points = pd.concat(all_points.values(), ignore_index=True)
    flashes = pd.concat(all_flashes.values(), ignore_index=True)
    curves = pd.concat(curve_frames, ignore_index=True)
    q_data = q_summary(points, curves)
    fluorescence = fluorescence_at_common_ci(flashes)
    jq = fit_jq_parameters(fluorescence)
    qc = qc_summary(all_points, all_flashes, q_data)
    coupling = carbon_fluorescence_coupling(q_data, fluorescence, qc)

    points.to_csv(POINTS_OUT, index=False)
    flashes.to_csv(FLASHES_OUT, index=False)
    curves.to_csv(CURVES_OUT, index=False)
    q_data.to_csv(Q_SUMMARY_OUT, index=False)
    fluorescence.to_csv(FLUORESCENCE_OUT, index=False)
    jq.to_csv(JQ_OUT, index=False)
    qc.to_csv(QC_OUT, index=False)
    coupling.to_csv(COUPLING_OUT, index=False)

    plot_aci_small_multiples(curves, qc)
    plot_zone_grouped_aci_grid(curves, qc)
    plot_zone_by_order_aci_grid(curves, qc)
    plot_zone_comparison(q_data, fluorescence, qc)
    plot_qc_diagnostics(all_points, qc)
    plot_capacity_summary(q_data, jq, qc)
    plot_carbon_fluorescence_coupling(coupling)
    plot_measurement_order(q_data, jq, qc)
    plot_within_day_order(q_data, jq, qc)

    print("Wrote cohort outputs:")
    for path in [
        POINTS_OUT,
        FLASHES_OUT,
        CURVES_OUT,
        Q_SUMMARY_OUT,
        FLUORESCENCE_OUT,
        JQ_OUT,
        QC_OUT,
        COUPLING_OUT,
        ACI_FIGURE_OUT,
        ZONE_ACI_GRID_OUT,
        ZONE_ORDER_ACI_GRID_OUT,
        COMPARISON_FIGURE_OUT,
        QC_FIGURE_OUT,
        CAPACITY_FIGURE_OUT,
        COUPLING_FIGURE_OUT,
        ORDER_FIGURE_OUT,
        WITHIN_DAY_ORDER_FIGURE_OUT,
    ]:
        print(f"  {path}")
    print("\nRun QC:")
    qc_display = qc.copy()
    numeric_columns = qc_display.select_dtypes(include=np.number).columns
    qc_display[numeric_columns] = qc_display[numeric_columns].round(3)
    print(qc_display.to_string(index=False))
    print("\nJ(Q) parameters:")
    print(jq.round(4).to_string(index=False))
    print("\nQ2000 common-Ci summary:")
    print(
        q_data.loc[q_data["Q_level"].eq(2000), ["run_code", "zone", "A_at_Ci800", "A_at_Ci1200", "gsw_low_Ca", "Ci_max"]]
        .round(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
