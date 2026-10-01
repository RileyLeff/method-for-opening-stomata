from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.interpolate import griddata

from photo26.curves import OUTPUT_DIR, read_licor_export


PROJECT_ROOT = Path(__file__).resolve().parents[1]
Q_LEVELS = np.array([50, 100, 200, 350, 600, 1000, 1400, 1700, 2000], dtype=float)
ALL_Q_LEVELS = np.insert(Q_LEVELS, 0, 0)
Q_NORM = Normalize(vmin=float(Q_LEVELS.min()), vmax=float(Q_LEVELS.max()))
Q_CMAP = plt.get_cmap("viridis")
FLASH_NAME = re.compile(
    r"MPF-\d+-(?P<date>\d{8})-(?P<hour>\d{2})_(?P<minute>\d{2})_(?P<second>\d{2})"
)


@dataclass(frozen=True)
class Run:
    code: str
    run_name: str
    time_window: str
    zone: str
    color: str
    dataset_dir: Path
    log_name: str

    @property
    def label(self) -> str:
        return f"{self.run_name} ({self.time_window})"

    @property
    def prefix(self) -> str:
        return f"phrag_brownsville_{self.code.lower()}_biochemical_limits"


RUNS = [
    Run(
        code="P01",
        run_name="Run 1",
        time_window="11:04-13:56",
        zone="High",
        color="#2475a8",
        dataset_dir=PROJECT_ROOT / "data" / "2026-07-24_phrag_brownsville_p01_biochemical_limits",
        log_name="2026-07-24-1101_logdata_phrag_brwns_P017242026",
    ),
    Run(
        code="P04",
        run_name="Run 2",
        time_window="14:16-17:09",
        zone="High",
        color="#d97728",
        dataset_dir=PROJECT_ROOT / "data" / "2026-07-24_phrag_brownsville_p04_biochemical_limits",
        log_name="2026-07-24-1415_logdata_Phra_brwns_P047242026",
    ),
]

NUMERIC_COLUMNS = [
    "elapsed",
    "A",
    "Asty",
    "Adyn",
    "Ca",
    "Ci",
    "gsw",
    "Q",
    "CO2_r",
    "CO2_s",
    "CO2_r_sp",
    "H2O_r",
    "H2O_s",
    "H2O_r_sp",
    "VPDleaf",
    "Tleaf",
    "TleafCnd",
    "Txchg_sp",
    "Flow",
    "Humidifier_%",
    "Desiccant_%",
]


def as_float(value: object) -> float:
    if value in (None, "", "-"):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def q_color(q_value: float) -> tuple[float, float, float, float]:
    return Q_CMAP(Q_NORM(float(q_value)))


def nearest_q(values: pd.Series, levels: np.ndarray = ALL_Q_LEVELS) -> np.ndarray:
    array = values.to_numpy(dtype=float)
    return levels[np.argmin(np.abs(array[:, None] - levels[None, :]), axis=1)]


def style_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#d8ddd4", linewidth=0.75)
    ax.tick_params(colors="#2d2d2d")


def flash_time_from_name(path: Path) -> pd.Timestamp:
    match = FLASH_NAME.fullmatch(path.stem)
    if match is None:
        raise ValueError(f"Unrecognized MPF filename: {path.name}")
    values = match.groupdict()
    return pd.to_datetime(
        f"{values['date']} {values['hour']}:{values['minute']}:{values['second']}",
        format="%Y%m%d %H:%M:%S",
    )


def load_points(run: Run) -> pd.DataFrame:
    points = read_licor_export(run.dataset_dir / run.log_name).copy()
    points = points.loc[pd.to_numeric(points["elapsed"], errors="coerce").notna()].copy()
    for column in NUMERIC_COLUMNS:
        points[column] = pd.to_numeric(points[column], errors="coerce")
    points = points.sort_values("elapsed").reset_index(drop=True)
    acquisition_date = pd.to_datetime(run.log_name[:10], format="%Y-%m-%d")
    clock_time = pd.to_timedelta(points["hhmmss"].astype(str), errors="coerce")
    day_rollover = clock_time.diff().dt.total_seconds().lt(-12 * 60 * 60).cumsum()
    points["datetime"] = (
        acquisition_date + clock_time + pd.to_timedelta(day_rollover, unit="D")
    )
    points["minute"] = points["elapsed"] / 60
    points["run_code"] = run.code
    points["run_name"] = run.run_name
    points["position"] = run.zone

    # MPFs briefly drive Qin above 10,000. Carry the chamber-light value
    # through those pulses so the records remain assigned to their ramp.
    points["Q_nominal"] = points["Q"].where(points["Q"].le(2500)).ffill().bfill()
    points["Q_level"] = nearest_q(points["Q_nominal"])
    points["phase_id"] = (
        points["Q_level"].ne(points["Q_level"].shift())
        | points["minute"].diff().gt(5)
    ).cumsum()
    points["phase_minute"] = (
        points["minute"] - points.groupby("phase_id")["minute"].transform("min")
    )
    phase_co2_max = points.groupby("phase_id")["CO2_r_sp"].transform("max")
    points["is_measured_ramp"] = points["Q_level"].isin(Q_LEVELS) & phase_co2_max.gt(1900)

    pulse = points["Q"].gt(3000)
    elapsed = points["elapsed"].to_numpy()
    most_recent_pulse = np.maximum.accumulate(np.where(pulse.to_numpy(), elapsed, -np.inf))
    points["seconds_after_mpf"] = elapsed - most_recent_pulse
    points["mpf_pulse"] = pulse
    points["mpf_or_recovery"] = points["seconds_after_mpf"].between(0, 25, inclusive="both")
    points["Ci_Ca"] = points["Ci"] / points["Ca"]

    points["clean_ramp"] = (
        points["is_measured_ramp"]
        & points["Q"].sub(points["Q_level"]).abs().lt(75)
        & ~points["mpf_or_recovery"]
        & points["Ci"].between(0, 2100)
        & points["A"].between(-20, 40)
        & points["gsw"].between(0.005, 1.0)
        & points["CO2_r"].between(40, 2050)
        & points["CO2_s"].between(40, 2050)
        & points["Flow"].between(580, 620)
        & points["VPDleaf"].between(0.70, 1.30)
    )
    return points


def load_flashes(run: Run, points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted((run.dataset_dir / "flashes").glob("*.json")):
        payload = json.loads(path.read_text())
        flr = payload.get("FLR", {})
        light_id = str(flr.get("LightAdaptedID", "-"))
        row = {
            "run_code": run.code,
            "run_name": run.run_name,
            "position": run.zone,
            "flash_id": path.stem,
            "event_id": payload.get("EVENT_ID"),
            "trigger_datetime": flash_time_from_name(path),
            "payload_datetime": pd.to_datetime(
                f"{payload.get('DATE')} {payload.get('TIME')}",
                format="%Y%m%d %H:%M:%S",
                errors="coerce",
            ),
            "pre_qin": as_float(payload.get("Pre_Qin")),
            "pre_qabs": as_float(payload.get("Pre_Qabs")),
            "flash_kind": "active" if light_id not in {"", "-", "None"} else "dark",
            "dark_id": flr.get("DarkAdaptedID"),
            "light_id": flr.get("LightAdaptedID"),
            "FvFm": as_float(flr.get("Fv/Fm")),
            "FvFm_prime": as_float(flr.get("Fv'/Fm'")),
            "PhiPS2": as_float(flr.get("PhiPS2")),
            "ETR": as_float(flr.get("ETR")),
            "NPQ": as_float(flr.get("NPQ")),
            "qL": as_float(flr.get("qL")),
            "qP": as_float(flr.get("qP")),
            "PhiCO2": as_float(flr.get("PhiCO2")),
            "p2_r2": as_float(payload.get("P2_R2")),
            "p1_delta_f": as_float(payload.get("P1_DELTAF")),
            "p3_delta_f": as_float(payload.get("P3_DELTAF")),
        }
        rows.append(row)
    flashes = pd.DataFrame(rows).sort_values("trigger_datetime").reset_index(drop=True)
    flashes["Q_level"] = nearest_q(flashes["pre_qin"], Q_LEVELS)

    active = flashes["flash_kind"].eq("active")
    pulse_times = points.loc[points["mpf_pulse"], "datetime"].drop_duplicates().sort_values()
    offsets: list[float] = []
    for trigger in flashes.loc[active, "trigger_datetime"]:
        delta = (pulse_times - trigger).dt.total_seconds()
        candidates = delta.loc[delta.between(15, 70)]
        if not candidates.empty:
            offsets.append(float(candidates.iloc[np.argmin(np.abs(candidates - 40))]))
    clock_offset = float(np.median(offsets))
    flashes["trigger_to_logged_pulse_s"] = clock_offset
    flashes["gas_event_datetime"] = flashes["trigger_datetime"]
    flashes.loc[active, "gas_event_datetime"] = (
        flashes.loc[active, "trigger_datetime"]
        + pd.to_timedelta(clock_offset, unit="s")
    )

    # Associate fluorescence with the stable gas-exchange state immediately
    # before the logged saturating pulse, never with the pulse row itself.
    gas_columns = ["A", "Asty", "Ca", "Ci", "gsw", "VPDleaf", "Tleaf", "CO2_r_sp", "elapsed"]
    for column in gas_columns:
        flashes[f"gas_{column}"] = np.nan
    for row_index in flashes.index[active]:
        event_time = flashes.at[row_index, "gas_event_datetime"]
        window = points.loc[
            points["datetime"].between(
                event_time - pd.Timedelta(seconds=6),
                event_time - pd.Timedelta(seconds=0.25),
            )
            & ~points["mpf_pulse"]
        ]
        if window.empty:
            continue
        for column in gas_columns:
            flashes.at[row_index, f"gas_{column}"] = float(window[column].median())
    return flashes


def binned_curves(points: pd.DataFrame) -> pd.DataFrame:
    clean = points.loc[points["clean_ramp"]].copy()
    clean["ci_bin"] = pd.cut(clean["Ci"], np.arange(0, 2101, 50), include_lowest=True)
    binned = (
        clean.groupby(["run_code", "run_name", "position", "Q_level", "ci_bin"], observed=True)
        .agg(
            n=("A", "size"),
            Ci=("Ci", "median"),
            Ca=("Ca", "median"),
            Q=("Q_nominal", "median"),
            A=("A", "median"),
            A_q25=("A", lambda values: values.quantile(0.25)),
            A_q75=("A", lambda values: values.quantile(0.75)),
            gsw=("gsw", "median"),
            VPDleaf=("VPDleaf", "median"),
        )
        .reset_index()
    )
    return binned.loc[binned["n"].ge(3)].sort_values(["Q_level", "Ci"]).reset_index(drop=True)


def summarize_run(run: Run, points: pd.DataFrame, flashes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for q_level, group in points.loc[points["clean_ramp"]].groupby("Q_level", sort=True):
        ambient = group.loc[group["Ca"].between(380, 460)]
        low_ca = group.loc[group["Ca"].between(150, 300)]
        high_ci = group.loc[group["Ci"].between(1300, 1800)]
        high_ci_slope = (
            float(np.polyfit(high_ci["Ci"], high_ci["A"], 1)[0] * 100)
            if len(high_ci) >= 20
            else np.nan
        )
        rows.append(
            {
                "run_code": run.code,
                "run_name": run.run_name,
                "position": run.zone,
                "Q_level": q_level,
                "n_clean": len(group),
                "A_p95": group["A"].quantile(0.95),
                "A_max": group["A"].max(),
                "A_at_Ca420": ambient["A"].median(),
                "Ci_at_Ca420": ambient["Ci"].median(),
                "gsw_low_Ca": low_ca["gsw"].median(),
                "gsw_at_Ca420": ambient["gsw"].median(),
                "gsw_median": group["gsw"].median(),
                "gsw_max": group["gsw"].max(),
                "Ci_max": group["Ci"].max(),
                "Ca_max": group["Ca"].max(),
                "high_Ci_A_slope_per_100_Ci": high_ci_slope,
                "VPDleaf_median": group["VPDleaf"].median(),
                "Tleaf_median": group["Tleaf"].median(),
            }
        )
    q_summary = pd.DataFrame(rows)

    active = flashes.loc[flashes["flash_kind"].eq("active")]
    fluorescence = (
        active.groupby(["run_code", "run_name", "position", "Q_level"], sort=True)
        .agg(
            n_flashes=("flash_id", "size"),
            PhiPS2=("PhiPS2", "median"),
            ETR=("ETR", "median"),
            NPQ=("NPQ", "median"),
            qL=("qL", "median"),
            qP=("qP", "median"),
            FvFm_prime=("FvFm_prime", "median"),
            gas_Ci=("gas_Ci", "median"),
            gas_A=("gas_A", "median"),
        )
        .reset_index()
    )
    return q_summary, fluorescence


def measured_phases(points: pd.DataFrame) -> pd.DataFrame:
    phase = (
        points.loc[points["is_measured_ramp"]]
        .groupby("phase_id", sort=True)
        .agg(
            Q_level=("Q_level", "median"),
            start_min=("minute", "min"),
            end_min=("minute", "max"),
        )
        .reset_index()
    )
    phase["mid_min"] = (phase["start_min"] + phase["end_min"]) / 2
    return phase


def add_q_bands(ax: plt.Axes, points: pd.DataFrame, labels: bool = False) -> None:
    for row in measured_phases(points).itertuples():
        color = q_color(row.Q_level)
        ax.axvspan(row.start_min, row.end_min, color=color, alpha=0.09, lw=0)
        if labels:
            ax.text(
                row.mid_min,
                0.91,
                f"{int(row.Q_level)}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                color=color,
                fontsize=7.5,
                fontweight="bold",
            )


def plot_run_overview(run: Run, points: pd.DataFrame, curves: pd.DataFrame) -> Path:
    output = OUTPUT_DIR / f"{run.prefix}_overview.png"
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.72)
    fig = plt.figure(figsize=(16.8, 9.6))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.12, 1.0], hspace=0.32, wspace=0.12)
    ax_aci = fig.add_subplot(grid[0, 0])
    ax_gsw = fig.add_subplot(grid[1, 0])
    ax_surface = fig.add_subplot(grid[:, 1], projection="3d")

    raw = points.loc[
        points["is_measured_ramp"]
        & points["Ci"].between(0, 2100)
        & points["A"].between(-5, 30)
    ]
    for q_level, group in raw.groupby("Q_level", sort=True):
        ax_aci.plot(group["Ci"], group["A"], color=q_color(q_level), lw=0.7, alpha=0.23)
    for q_level, group in curves.groupby("Q_level", sort=True):
        color = q_color(q_level)
        ax_aci.fill_between(
            group["Ci"], group["A_q25"], group["A_q75"],
            color=color, alpha=0.10, linewidth=0,
        )
        ax_aci.plot(group["Ci"], group["A"], color=color, lw=2.25)
        endpoint = group.iloc[group["Ci"].argmax()]
        ax_aci.annotate(
            f"{int(q_level)}",
            (endpoint["Ci"], endpoint["A"]),
            xytext=(8, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            color=color,
            fontsize=8,
            fontweight="bold",
            clip_on=False,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )
    ax_aci.axhline(0, color="#555555", lw=0.9, ls=":")
    ax_aci.set_title("A/Ci Across Light", loc="left", fontweight="bold")
    ax_aci.set_xlabel("Ci (umol mol-1)")
    ax_aci.set_ylabel("A / DAT Adyn (umol CO2 m-2 s-1)")
    ax_aci.set_xlim(0, 2025)
    ax_aci.set_ylim(0, max(15, float(raw["A"].max()) + 1))
    style_axes(ax_aci)
    colorbar = fig.colorbar(
        ScalarMappable(norm=Q_NORM, cmap=Q_CMAP),
        ax=ax_aci,
        pad=0.015,
        fraction=0.045,
    )
    colorbar.set_label("Q (umol photons m-2 s-1)")

    add_q_bands(ax_gsw, points, labels=True)
    ramp_points = points.loc[points["is_measured_ramp"]]
    ax_gsw.plot(ramp_points["minute"], ramp_points["gsw"], color="#2475a8", lw=0.9, alpha=0.82)
    clean = points.loc[points["clean_ramp"]]
    ax_gsw.scatter(clean["minute"], clean["gsw"], color="#2475a8", s=2, alpha=0.18, linewidth=0)
    ax_gsw.set_title("Stomatal Conductance Through the Ramps", loc="left", fontweight="bold")
    ax_gsw.set_xlabel("Elapsed time (min)")
    ax_gsw.set_ylabel("gsw (mol H2O m-2 s-1)")
    ax_gsw.set_ylim(0, min(0.6, max(0.18, float(ramp_points["gsw"].quantile(0.995)) * 1.08)))
    style_axes(ax_gsw)

    if len(curves) >= 20:
        ci_grid = np.linspace(curves["Ci"].quantile(0.01), curves["Ci"].quantile(0.99), 80)
        q_grid = np.linspace(Q_LEVELS.min(), Q_LEVELS.max(), 70)
        xx, yy = np.meshgrid(ci_grid, q_grid)
        zz = griddata(curves[["Ci", "Q"]].to_numpy(), curves["A"].to_numpy(), (xx, yy), method="linear")
        ax_surface.plot_surface(xx, yy, zz, cmap=Q_CMAP, alpha=0.83, linewidth=0, antialiased=True)
        ax_surface.scatter(
            curves["Ci"], curves["Q"], curves["A"],
            c=curves["Q"], cmap=Q_CMAP, norm=Q_NORM, s=9, alpha=0.36,
        )
    ax_surface.set_title("A/Q/Ci Surface", pad=18, fontweight="bold")
    ax_surface.set_xlabel("Ci (umol mol-1)", labelpad=9)
    ax_surface.set_ylabel("Q (umol photons m-2 s-1)", labelpad=9)
    ax_surface.set_zlabel("A", labelpad=7)
    ax_surface.view_init(elev=25, azim=-128)
    ax_surface.set_zlim(0, max(15, float(curves["A"].max()) + 1))

    fig.text(
        0.055, 0.98,
        f"Brownsville {run.zone} Phragmites: {run.label}",
        ha="left", va="top", fontsize=22, fontweight="bold",
    )
    fig.text(
        0.055, 0.945,
        "Thin traces retain every plausible raw ramp row; thick curves and the surface omit MPF pulse/recovery windows.",
        ha="left", va="top", fontsize=10.2, color="#555555",
    )
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.085, top=0.88)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_controls(all_points: dict[str, pd.DataFrame]) -> Path:
    output = OUTPUT_DIR / "phrag_brownsville_biochemical_limits_controls.png"
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.65)
    fig, axes = plt.subplots(5, 2, figsize=(16, 12.6), sharex="col")
    for column, run in enumerate(RUNS):
        points = all_points[run.code]
        for row in range(5):
            add_q_bands(axes[row, column], points, labels=row == 0)

        axes[0, column].plot(points["minute"], points["Q_nominal"], color="#333333", lw=1.1)
        axes[0, column].set_title(run.label, loc="left", fontweight="bold")
        axes[0, column].set_ylabel("Q")

        axes[1, column].plot(points["minute"], points["VPDleaf"], color="#7b4a3b", lw=1.0)
        axes[1, column].axhline(1.0, color="#333333", lw=0.9, ls="--")
        axes[1, column].set_ylabel("VPDleaf\n(kPa)")
        axes[1, column].set_ylim(0.82, 1.22)

        axes[2, column].plot(points["minute"], points["Tleaf"], color="#d95f70", lw=1.0, label="Tleaf")
        axes[2, column].axhline(25, color="#333333", lw=0.9, ls="--", label="25 C")
        axes[2, column].set_ylabel("Tleaf\n(C)")
        axes[2, column].set_ylim(24.75, 26.2)
        axes[2, column].legend(frameon=False, ncol=2, fontsize=8, loc="upper left")

        axes[3, column].plot(points["minute"], points["H2O_r_sp"], color="#333333", lw=1.0, label="H2O_r setpoint")
        axes[3, column].plot(points["minute"], points["H2O_r"], color="#2a9d8f", lw=0.9, label="H2O_r")
        axes[3, column].plot(points["minute"], points["H2O_s"], color="#4c78a8", lw=0.9, label="H2O_s")
        axes[3, column].set_ylabel("H2O\n(mmol mol-1)")
        axes[3, column].legend(frameon=False, ncol=3, fontsize=7.5, loc="upper left")

        axes[4, column].plot(points["minute"], points["Humidifier_%"], color="#2a9d8f", lw=0.9, label="Humidifier")
        axes[4, column].plot(points["minute"], points["Desiccant_%"], color="#e76f51", lw=0.9, label="Desiccant")
        axes[4, column].set_ylabel("Actuator\n(%)")
        axes[4, column].set_xlabel("Elapsed time (min)")
        axes[4, column].legend(frameon=False, ncol=2, fontsize=8, loc="upper left")

        for ax in axes[:, column]:
            style_axes(ax)

    fig.text(
        0.055, 0.986,
        "Brownsville New-Program Environmental Control",
        ha="left", va="top", fontsize=21, fontweight="bold",
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.06, top=0.93, hspace=0.17, wspace=0.18)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_comparison(
    all_points: dict[str, pd.DataFrame],
    all_curves: dict[str, pd.DataFrame],
    q_summary: pd.DataFrame,
    fluorescence: pd.DataFrame,
) -> Path:
    output = OUTPUT_DIR / "phrag_brownsville_high_runs_biochemical_comparison.png"
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.72)
    fig, axes = plt.subplots(2, 2, figsize=(15.8, 10.2))
    selected_q = [350, 600, 1000, 2000]

    ax = axes[0, 0]
    for run in RUNS:
        curves = all_curves[run.code]
        for q_level in selected_q:
            group = curves.loc[curves["Q_level"].eq(q_level)]
            ax.plot(
                group["Ci"], group["A"],
                color=q_color(q_level),
                lw=2.3 if run.code == "P01" else 1.8,
                ls="-" if run.code == "P01" else "--",
                alpha=0.96,
            )
    ax.axhline(0, color="#555555", lw=0.9, ls=":")
    ax.set_title("A/Ci at Selected Q", loc="left", fontweight="bold")
    ax.set_xlabel("Ci (umol mol-1)")
    ax.set_ylabel("A / DAT Adyn")
    ax.set_xlim(0, 1950)
    ax.set_ylim(0, 23)
    style_axes(ax)
    q_handles = [Line2D([0], [0], color=q_color(q), lw=2.3, label=str(q)) for q in selected_q]
    run_handles = [
        Line2D([0], [0], color="#333333", lw=2.2, ls="-", label=RUNS[0].label),
        Line2D([0], [0], color="#333333", lw=2.0, ls="--", label=RUNS[1].label),
    ]
    legend_q = ax.legend(handles=q_handles, title="Q", frameon=False, ncol=2, fontsize=8.5, loc="upper left")
    ax.add_artist(legend_q)
    ax.legend(handles=run_handles, frameon=False, fontsize=8.5, loc="lower right")

    ax = axes[0, 1]
    for run in RUNS:
        group = q_summary.loc[q_summary["run_code"].eq(run.code)].sort_values("Q_level")
        ax.plot(group["Q_level"], group["A_p95"], color=run.color, marker="o", lw=2.5, label=f"{run.label}: upper A")
        ax.plot(group["Q_level"], group["A_at_Ca420"], color=run.color, marker="s", lw=1.8, ls="--", label=f"{run.label}: Ca ~420")
    ax.set_title("Light Response of Upper and Ambient-CO2 A", loc="left", fontweight="bold")
    ax.set_xlabel("Q (umol photons m-2 s-1)")
    ax.set_ylabel("A")
    ax.set_ylim(0, 23)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style_axes(ax)

    ax = axes[1, 0]
    for run in RUNS:
        group = q_summary.loc[q_summary["run_code"].eq(run.code)].sort_values("Q_level")
        ax.plot(group["Q_level"], group["gsw_low_Ca"], color=run.color, marker="o", lw=2.4, label=f"{run.label}: Ca 150-300")
        ax.plot(group["Q_level"], group["gsw_at_Ca420"], color=run.color, marker="s", lw=1.8, ls="--", label=f"{run.label}: Ca ~420")
    ax.set_title("Stomatal Conductance During Ramps", loc="left", fontweight="bold")
    ax.set_xlabel("Q (umol photons m-2 s-1)")
    ax.set_ylabel("gsw (mol H2O m-2 s-1)")
    ax.set_ylim(0, 0.16)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style_axes(ax)

    ax = axes[1, 1]
    ax2 = ax.twinx()
    for run in RUNS:
        group = fluorescence.loc[fluorescence["run_code"].eq(run.code)].sort_values("Q_level")
        ax.plot(group["Q_level"], group["PhiPS2"], color=run.color, marker="o", lw=2.4, label=f"{run.label}: PhiPSII")
        ax2.plot(group["Q_level"], group["NPQ"], color=run.color, marker="s", lw=1.8, ls="--", label=f"{run.label}: NPQ")
    ax.set_title("Photochemical Yield and Energy Dissipation", loc="left", fontweight="bold")
    ax.set_xlabel("Q (umol photons m-2 s-1)")
    ax.set_ylabel("PhiPSII")
    ax2.set_ylabel("NPQ")
    ax.set_ylim(0, 0.8)
    ax2.set_ylim(0, 2.5)
    style_axes(ax)
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, frameon=False, fontsize=8, loc="center right")

    fig.text(
        0.055, 0.982,
        "Brownsville High Phragmites: Run 1 vs Run 2",
        ha="left", va="top", fontsize=22, fontweight="bold",
    )
    fig.text(
        0.055, 0.948,
        "Sequential high-position leaves measured with the same new-program controls and processing.",
        ha="left", va="top", fontsize=10.2, color="#555555",
    )
    fig.subplots_adjust(left=0.075, right=0.93, bottom=0.08, top=0.89, hspace=0.34, wspace=0.24)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_points: dict[str, pd.DataFrame] = {}
    all_flashes: dict[str, pd.DataFrame] = {}
    all_curves: dict[str, pd.DataFrame] = {}
    q_summaries: list[pd.DataFrame] = []
    fluorescence_summaries: list[pd.DataFrame] = []
    outputs: list[Path] = []

    for run in RUNS:
        points = load_points(run)
        flashes = load_flashes(run, points)
        curves = binned_curves(points)
        q_summary, fluorescence = summarize_run(run, points, flashes)
        all_points[run.code] = points
        all_flashes[run.code] = flashes
        all_curves[run.code] = curves
        q_summaries.append(q_summary)
        fluorescence_summaries.append(fluorescence)

        points_out = OUTPUT_DIR / f"{run.prefix}_points.csv"
        flashes_out = OUTPUT_DIR / f"{run.prefix}_flashes.csv"
        curves_out = OUTPUT_DIR / f"{run.prefix}_binned_curves.csv"
        summary_out = OUTPUT_DIR / f"{run.prefix}_q_summary.csv"
        fluorescence_out = OUTPUT_DIR / f"{run.prefix}_fluorescence_summary.csv"
        points.to_csv(points_out, index=False)
        flashes.to_csv(flashes_out, index=False)
        curves.to_csv(curves_out, index=False)
        q_summary.to_csv(summary_out, index=False)
        fluorescence.to_csv(fluorescence_out, index=False)
        outputs.extend([points_out, flashes_out, curves_out, summary_out, fluorescence_out])
        outputs.append(plot_run_overview(run, points, curves))

    q_summary_all = pd.concat(q_summaries, ignore_index=True)
    fluorescence_all = pd.concat(fluorescence_summaries, ignore_index=True)
    combined_summary_out = OUTPUT_DIR / "phrag_brownsville_biochemical_limits_q_summary.csv"
    combined_fluorescence_out = OUTPUT_DIR / "phrag_brownsville_biochemical_limits_fluorescence_summary.csv"
    q_summary_all.to_csv(combined_summary_out, index=False)
    fluorescence_all.to_csv(combined_fluorescence_out, index=False)
    outputs.extend([combined_summary_out, combined_fluorescence_out])
    outputs.append(plot_controls(all_points))
    outputs.append(plot_comparison(all_points, all_curves, q_summary_all, fluorescence_all))

    print("Wrote:")
    for path in outputs:
        print(f"  {path}")
    print("\nGas-exchange summary:")
    print(q_summary_all.round(3).to_string(index=False))
    print("\nFluorescence summary:")
    print(fluorescence_all.round(3).to_string(index=False))
    for run in RUNS:
        points = all_points[run.code]
        flashes = all_flashes[run.code]
        dark = flashes.loc[flashes["flash_kind"].eq("dark"), "FvFm"]
        print(
            f"\n{run.label}: duration={points['minute'].max() - points['minute'].min():.1f} min, "
            f"VPDleaf median={points['VPDleaf'].median():.3f}, "
            f"Tleaf median={points['Tleaf'].median():.3f}, "
            f"H2O_r median={points['H2O_r'].median():.3f}, "
            f"active MPFs={(flashes['flash_kind'] == 'active').sum()}, "
            f"dark Fv/Fm={dark.iloc[0]:.3f}, "
            f"flash-clock offset={flashes['trigger_to_logged_pulse_s'].iloc[0]:.1f} s"
        )


if __name__ == "__main__":
    main()
