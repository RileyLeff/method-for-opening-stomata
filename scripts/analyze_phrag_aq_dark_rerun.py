from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.interpolate import griddata

from photo26.curves import OUTPUT_DIR, read_licor_export


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "2026-07-07_phrag_aq_dark_rerun"
LOG_PATH = DATASET_DIR / "2026-07-07-1301_logdata_phrag_aq_dark"
FLASH_DIR = DATASET_DIR / "flashes"

POINTS_OUT = OUTPUT_DIR / "phrag_aq_dark_rerun_points.csv"
FLASH_OUT = OUTPUT_DIR / "phrag_aq_dark_rerun_flash_summary.csv"
JOINED_OUT = OUTPUT_DIR / "phrag_aq_dark_rerun_points_with_flash.csv"
PHASE_OUT = OUTPUT_DIR / "phrag_aq_dark_rerun_phase_summary.csv"
FLASH_JOIN_OUT = OUTPUT_DIR / "phrag_aq_dark_rerun_flash_join_report.csv"

TIMELINE_OUT = OUTPUT_DIR / "phrag_aq_dark_rerun_timeline.png"
STOMATAL_TS_OUT = OUTPUT_DIR / "phrag_aq_dark_rerun_stomatal_time_series.png"
RAMP_QUALITY_OUT = OUTPUT_DIR / "phrag_aq_dark_rerun_ramp_quality.png"
ACI_OUT = OUTPUT_DIR / "phrag_aq_dark_rerun_aci_by_q.png"
SURFACE_OUT = OUTPUT_DIR / "phrag_aq_dark_rerun_a_surface_3d.png"
AQC_SURFACE_OUT = OUTPUT_DIR / "phrag_aq_dark_rerun_aqc_surface_3d.png"
FLUORO_OUT = OUTPUT_DIR / "phrag_aq_dark_rerun_fluorescence.png"

ALPHA_VC = "\u03b1Vc"
ALPHA_VH = "\u03b1Vh"
Q_LEVELS = np.array([0, 200, 400, 700, 1000, 1300, 1600], dtype=float)
ACTIVE_Q_LEVELS = [200, 400, 700, 1000, 1300, 1600]

FLR_COLUMNS = [
    "DarkAdaptedID",
    "Qmax_d",
    "Fo",
    "Fm",
    "Fv/Fm",
    "A_dark",
    "LightAdaptedID",
    "Qmax",
    "Fs",
    "Fm'",
    "PhiPS2",
    "PS2/1",
    "Qabs_fs",
    "A_fs",
    "ETR",
    "PhiCO2",
    "NPQ",
    "alt._Fo'",
    "DarkPulseID",
    "Fmin",
    "Fo'",
    "Fv'/Fm'",
    "qP",
    "qN",
    "qP_Fo",
    "qN_Fo",
    "qL",
    "1-qL",
]


def as_float(value: Any) -> float:
    if value in (None, "", "-"):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def nearest_q(values: pd.Series) -> np.ndarray:
    q = values.to_numpy(dtype=float)
    return Q_LEVELS[np.argmin(np.abs(q[:, None] - Q_LEVELS[None, :]), axis=1)]


def style_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#d8ddd4", linewidth=0.8)
    ax.tick_params(colors="#2d2d2d")


def q_color(q_value: float) -> tuple[float, float, float, float] | str:
    if q_value == 0:
        return "#9a8fab"
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=min(ACTIVE_Q_LEVELS), vmax=max(ACTIVE_Q_LEVELS))
    return cmap(norm(float(q_value)))


def load_points() -> pd.DataFrame:
    points = read_licor_export(LOG_PATH).sort_values("elapsed").reset_index(drop=True).copy()
    points["minute"] = points["elapsed"] / 60
    points["Q_level"] = nearest_q(points["Q"])
    phase_break = points["Q_level"].ne(points["Q_level"].shift()) | points["minute"].diff().gt(5)
    points["phase_id"] = phase_break.cumsum()
    points["phase_min"] = points["minute"] - points.groupby("phase_id")["minute"].transform("min")
    points["co2_command_jump_back"] = points.groupby("phase_id")["CO2_r_sp"].diff().lt(-20)
    points["light_flash_logged"] = points["LightAdaptedID"].astype(str).ne("-")
    points["dark_flash_logged"] = points["DarkAdaptedID"].astype(str).ne("-")
    points["A_minus_Asty"] = points["A"] - points["Asty"]
    points["Ci_Ca"] = points["Ci"] / points["Ca"]
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
    return points


def load_flash_jsons() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(FLASH_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        flr = data.get("FLR", {})
        row: dict[str, Any] = {
            "flash_id": path.stem,
            "flash_file": path.name,
            "event_id": data.get("EVENT_ID"),
            "flash_date": data.get("DATE"),
            "flash_time": data.get("TIME"),
            "flash_timestamp": as_float(data.get("TIMESTAMP")),
            "pre_qin": as_float(data.get("Pre_Qin")),
            "pre_qabs": as_float(data.get("Pre_Qabs")),
            "pre_q_red": as_float(data.get("Pre_Q_red")),
            "pre_q_blue": as_float(data.get("Pre_Q_blue")),
            "pre_favg": as_float(data.get("Pre_Favg")),
            "p1_maxf": as_float(data.get("P1_MAXF")),
            "p2_slp": as_float(data.get("P2_SLP")),
            "p2_r2": as_float(data.get("P2_R2")),
            "p3_maxf": as_float(data.get("P3_MAXF")),
        }
        for column in FLR_COLUMNS:
            value = flr.get(column)
            if column in {"DarkAdaptedID", "LightAdaptedID", "DarkPulseID"}:
                row[f"flr_{column}"] = value
            else:
                row[f"flr_{column}"] = as_float(value)
        light_id = row.get("flr_LightAdaptedID")
        row["flash_kind"] = "active" if isinstance(light_id, str) and light_id not in {"", "-"} else "dark"
        rows.append(row)

    flashes = pd.DataFrame(rows).sort_values("flash_timestamp").reset_index(drop=True)
    flashes["seconds_since_previous"] = flashes["flash_timestamp"].diff()
    return flashes


def join_flash_data(points: pd.DataFrame, flashes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    active_cols = [
        "flash_id",
        "event_id",
        "flash_file",
        "flash_timestamp",
        "seconds_since_previous",
        "pre_qin",
        "pre_qabs",
        "pre_q_red",
        "pre_q_blue",
        "pre_favg",
        "p1_maxf",
        "p2_slp",
        "p2_r2",
        "p3_maxf",
        "flr_LightAdaptedID",
        "flr_DarkAdaptedID",
        "flr_Fs",
        "flr_Fm'",
        "flr_PhiPS2",
        "flr_ETR",
        "flr_NPQ",
        "flr_qL",
        "flr_qP",
        "flr_Fv'/Fm'",
        "flr_PhiCO2",
    ]
    active = (
        flashes.loc[flashes["flash_kind"].eq("active"), active_cols]
        .rename(columns={"flr_LightAdaptedID": "LightAdaptedID"})
        .add_prefix("flash_")
        .rename(columns={"flash_LightAdaptedID": "LightAdaptedID"})
    )
    dark = (
        flashes.loc[flashes["flash_kind"].eq("dark"), ["flash_id", "event_id", "flash_file", "flr_Fo", "flr_Fm", "flr_Fv/Fm"]]
        .add_prefix("dark_flash_")
        .rename(columns={"dark_flash_flash_id": "DarkAdaptedID"})
    )
    joined = points.merge(active, on="LightAdaptedID", how="left")
    joined = joined.merge(dark, on="DarkAdaptedID", how="left")

    light_expected = points.loc[points["LightAdaptedID"].astype(str).ne("-"), "LightAdaptedID"].nunique()
    dark_expected = points.loc[points["DarkAdaptedID"].astype(str).ne("-"), "DarkAdaptedID"].nunique()
    active_matched = joined["flash_flash_id"].nunique() if "flash_flash_id" in joined else 0
    dark_matched = joined["dark_flash_event_id"].nunique() if "dark_flash_event_id" in joined else 0
    unmatched_light = sorted(
        set(points.loc[points["LightAdaptedID"].astype(str).ne("-"), "LightAdaptedID"].dropna().astype(str))
        - set(flashes.loc[flashes["flash_kind"].eq("active"), "flr_LightAdaptedID"].dropna().astype(str))
    )
    unmatched_dark = sorted(
        set(points.loc[points["DarkAdaptedID"].astype(str).ne("-"), "DarkAdaptedID"].dropna().astype(str))
        - set(flashes.loc[flashes["flash_kind"].eq("dark"), "flash_id"].dropna().astype(str))
    )
    report = pd.DataFrame(
        [
            {"metric": "log_rows", "value": len(points), "detail": ""},
            {"metric": "unique_light_ids_in_log", "value": int(light_expected), "detail": ""},
            {"metric": "unique_light_ids_matched", "value": int(active_matched), "detail": ""},
            {"metric": "unique_dark_ids_in_log", "value": int(dark_expected), "detail": ""},
            {"metric": "unique_dark_ids_matched", "value": int(dark_matched), "detail": ""},
            {"metric": "flash_json_total", "value": len(flashes), "detail": ""},
            {"metric": "flash_json_active", "value": int(flashes["flash_kind"].eq("active").sum()), "detail": ""},
            {"metric": "flash_json_dark", "value": int(flashes["flash_kind"].eq("dark").sum()), "detail": ""},
            {"metric": "minimum_flash_spacing_sec", "value": float(flashes["seconds_since_previous"].dropna().min()), "detail": ""},
            {"metric": "unmatched_light_ids", "value": len(unmatched_light), "detail": ", ".join(unmatched_light)},
            {"metric": "unmatched_dark_ids", "value": len(unmatched_dark), "detail": ", ".join(unmatched_dark)},
        ]
    )
    return joined, report


def summarize_phases(points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for phase_id, group in points.groupby("phase_id", sort=True):
        rows.append(
            {
                "phase_id": int(phase_id),
                "Q_level": float(group["Q_level"].mode().iloc[0]),
                "n": len(group),
                "start_min": float(group["minute"].min()),
                "end_min": float(group["minute"].max()),
                "duration_min": float(group["minute"].max() - group["minute"].min()),
                "CO2_r_sp_min": float(group["CO2_r_sp"].min()),
                "CO2_r_sp_max": float(group["CO2_r_sp"].max()),
                "CO2_command_jump_backs": int(group["co2_command_jump_back"].sum()),
                "Flow_median": float(group["Flow"].median()),
                "Flow_min": float(group["Flow"].min()),
                "Flow_max": float(group["Flow"].max()),
                "VPDleaf_median": float(group["VPDleaf"].median()),
                "VPDleaf_min": float(group["VPDleaf"].min()),
                "VPDleaf_max": float(group["VPDleaf"].max()),
                "A_median": float(group["A"].median()),
                "A_min": float(group["A"].min()),
                "A_max": float(group["A"].max()),
                "Ci_min": float(group["Ci"].min()),
                "Ci_max": float(group["Ci"].max()),
                "gsw_median": float(group["gsw"].median()),
                "light_flash_rows": int(group["light_flash_logged"].sum()),
                "dark_flash_rows": int(group["dark_flash_logged"].sum()),
                "sane_curve_rows": int(group["sane_for_curve"].sum()),
            }
        )
    return pd.DataFrame(rows)


def binned_curve_points(points: pd.DataFrame, bin_width: int = 60, min_count: int = 3) -> pd.DataFrame:
    data = points.loc[points["sane_for_curve"]].dropna(subset=["Ci", "A", "Q_level"]).copy()
    bins = np.arange(0, 2200 + bin_width, bin_width)
    data["ci_bin"] = pd.cut(data["Ci"], bins=bins, include_lowest=True)
    binned = (
        data.groupby(["Q_level", "ci_bin"], observed=True)
        .agg(
            n=("A", "size"),
            Ci=("Ci", "median"),
            Q=("Q", "median"),
            A=("A", "median"),
            A_q25=("A", lambda s: s.quantile(0.25)),
            A_q75=("A", lambda s: s.quantile(0.75)),
            gsw=("gsw", "median"),
            VPDleaf=("VPDleaf", "median"),
        )
        .reset_index()
    )
    return binned.loc[binned["n"].ge(min_count)].sort_values(["Q_level", "Ci"]).reset_index(drop=True)


def plot_timeline(points: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.66)
    fig, axes = plt.subplots(
        7,
        1,
        figsize=(15.8, 12.4),
        sharex=True,
        gridspec_kw={"height_ratios": [0.8, 1.2, 1.2, 1.0, 1.0, 1.0, 1.0]},
    )

    q_colors = {
        0: "#7b5aa6",
        200: "#4c63a3",
        400: "#2e7f95",
        700: "#2ca25f",
        1000: "#78c143",
        1300: "#b5bd35",
        1600: "#e0b72d",
    }
    for phase_id, group in points.groupby("phase_id", sort=True):
        q = int(group["Q_level"].mode().iloc[0])
        color = q_colors.get(q, "#dddddd")
        for ax in axes:
            ax.axvspan(group["minute"].min(), group["minute"].max(), color=color, alpha=0.075, lw=0)
        axes[0].text(
            group["minute"].mean(),
            0.86,
            str(q),
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="top",
            color=color,
            fontsize=8,
            fontweight="bold",
        )

    axes[0].plot(points["minute"], points["Q"], color="#333333", lw=1.4)
    axes[0].set_ylabel("Q")
    axes[1].plot(points["minute"], points["CO2_r_sp"], color="#333333", lw=1.0, label="setpoint")
    axes[1].plot(points["minute"], points["CO2_r"], color="#f28e2b", lw=1.0, label="reference")
    axes[1].plot(points["minute"], points["CO2_s"], color="#d95f70", lw=1.0, label="sample")
    axes[1].set_ylabel("CO2")
    axes[1].legend(frameon=False, ncol=3, loc="upper left", fontsize=8)

    axes[2].plot(points["minute"], points["A"].clip(-12, 28), color="#2f8f46", lw=1.1, label="A / Adyn")
    axes[2].plot(points["minute"], points["Asty"].clip(-12, 28), color="#999999", lw=0.85, alpha=0.8, label="Asty")
    axes[2].axhline(0, color="#555555", lw=0.9, ls=":")
    axes[2].set_ylabel("A\nclipped")
    axes[2].legend(frameon=False, ncol=2, loc="upper left", fontsize=8)

    axes[3].plot(points["minute"], points["Ci"].clip(-100, 2200), color="#805bb7", lw=0.95)
    axes[3].set_ylabel("Ci\nclipped")

    axes[4].plot(points["minute"], points["gsw"].clip(-0.02, 0.45), color="#2475a8", lw=1.0)
    axes[4].axhline(0, color="#555555", lw=0.8, ls=":")
    axes[4].set_ylabel("gsw\nclipped")

    axes[5].plot(points["minute"], points["VPDleaf"], color="#7b4a3b", lw=1.0, label="VPDleaf")
    axes[5].axhline(1.0, color="#555555", lw=0.9, ls=":")
    axes[5].set_ylabel("VPDleaf")

    axes[6].plot(points["minute"], points["Flow"], color="#333333", lw=0.95, label="Flow")
    axes6b = axes[6].twinx()
    axes6b.plot(points["minute"], points[ALPHA_VC], color="#2ca25f", lw=0.9, label="alphaVc")
    axes6b.plot(points["minute"], points[ALPHA_VH], color="#de2d26", lw=0.9, label="alphaVh")
    axes[6].set_ylabel("Flow")
    axes6b.set_ylabel("alpha V")
    axes6b.spines[["top"]].set_visible(False)
    axes6b.grid(False)
    lines1, labels1 = axes[6].get_legend_handles_labels()
    lines2, labels2 = axes6b.get_legend_handles_labels()
    axes[6].legend(lines1 + lines2, labels1 + labels2, frameon=False, ncol=3, loc="upper left", fontsize=8)
    axes[6].set_xlabel("Elapsed time after first logged point (min)")

    for ax in axes:
        style_axes(ax)
    fig.text(0.06, 0.982, "Phragmites DAT Ramp Rerun Diagnostic", ha="left", va="top", fontsize=21, fontweight="bold")
    fig.text(
        0.06,
        0.952,
        "CO2 ramps, VPD, flow, and dynamic A as logged; extreme Ci/A artifacts are clipped in the timeline only.",
        ha="left",
        va="top",
        fontsize=10,
        color="#555555",
    )
    fig.subplots_adjust(left=0.08, right=0.93, bottom=0.06, top=0.91, hspace=0.17)
    fig.savefig(TIMELINE_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_stomatal_time_series(points: pd.DataFrame) -> None:
    plot = points.sort_values("minute").copy()
    plot["Ci_plot"] = plot["Ci"].clip(0, 2200)
    plot["Ca_plot"] = plot["Ca"].clip(0, 2200)
    plot["Ci_Ca_plot"] = plot["Ci_Ca"].replace([np.inf, -np.inf], np.nan).clip(-0.2, 2.0)
    plot["gsw_plot"] = plot["gsw"].clip(-0.02, 0.45)

    sns.set_theme(style="whitegrid", context="talk", font_scale=0.76)
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(15.8, 9.8),
        sharex=True,
        gridspec_kw={"height_ratios": [0.75, 1.0, 1.25, 1.0]},
    )

    for phase_id, group in plot.groupby("phase_id", sort=True):
        q_level = float(group["Q_level"].mode().iloc[0])
        color = q_color(q_level)
        for ax in axes:
            ax.axvspan(group["minute"].min(), group["minute"].max(), color=color, alpha=0.10, lw=0)
        axes[0].text(
            group["minute"].mean(),
            0.82,
            f"{int(q_level)}",
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="top",
            color=color,
            fontsize=8.5,
            fontweight="bold",
        )

    axes[0].plot(plot["minute"], plot["Q"], color="#333333", lw=1.4)
    axes[0].set_ylabel("Q")

    axes[1].plot(plot["minute"], plot["gsw_plot"], color="#2475a8", lw=1.25)
    axes[1].axhline(0, color="#555555", lw=0.9, ls=":")
    axes[1].set_ylabel("gsw\nclipped")

    axes[2].plot(plot["minute"], plot["Ca_plot"], color="#f28e2b", lw=1.35, label="Ca")
    axes[2].plot(plot["minute"], plot["Ci_plot"], color="#805bb7", lw=1.1, label="Ci")
    flash_rows = plot.loc[plot["light_flash_logged"]]
    axes[2].scatter(
        flash_rows["minute"],
        flash_rows["Ci"].clip(0, 2200),
        s=26,
        facecolor="white",
        edgecolor=[q_color(q) for q in flash_rows["Q_level"]],
        linewidth=1.0,
        zorder=3,
        label="MPF flash rows",
    )
    axes[2].set_ylabel("CO2\nclipped")
    axes[2].legend(frameon=False, ncol=3, loc="upper left", fontsize=8.5)

    axes[3].plot(plot["minute"], plot["Ci_Ca_plot"], color="#2f8f46", lw=1.2)
    axes[3].axhline(0.7, color="#333333", lw=1.1, ls="--", label="Ci/Ca = 0.7")
    axes[3].set_ylim(-0.05, 1.45)
    axes[3].set_ylabel("Ci/Ca\nclipped")
    axes[3].set_xlabel("Elapsed time after first logged point (min)")
    axes[3].legend(frameon=False, loc="upper left", fontsize=8.5)

    for ax in axes:
        style_axes(ax)

    fig.text(0.06, 0.982, "Phragmites DAT Rerun: Stomatal Time Series", ha="left", va="top", fontsize=21, fontweight="bold")
    fig.text(
        0.06,
        0.944,
        "Q phase colors match the A/Ci curves; CO2 and ratio panels are clipped only to keep artifact rows visible without wrecking the scale.",
        ha="left",
        va="top",
        fontsize=10.5,
        color="#555555",
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.88, hspace=0.20)
    fig.savefig(STOMATAL_TS_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ramp_quality(points: pd.DataFrame, phase_summary: pd.DataFrame) -> None:
    main = points.loc[points["Q_level"].isin(Q_LEVELS)].copy()
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.68)
    fig, axes = plt.subplots(4, 2, figsize=(14, 11), sharey=True)
    axes_flat = list(axes.flat)

    for ax, (phase_id, group) in zip(axes_flat, main.groupby("phase_id", sort=True)):
        group = group.sort_values("minute")
        q_label = int(group["Q_level"].mode().iloc[0])
        back_jumps = int(group["co2_command_jump_back"].sum())
        ax.plot(group["phase_min"], group["CO2_r_sp"], color="#333333", lw=1.2, label="setpoint")
        ax.plot(group["phase_min"], group["CO2_r"], color="#f28e2b", lw=1.05, label="reference")
        ax.plot(group["phase_min"], group["CO2_s"], color="#d95f70", lw=1.05, label="sample")
        flash_rows = group.loc[group["light_flash_logged"]]
        ax.scatter(flash_rows["phase_min"], flash_rows["CO2_r_sp"], s=18, color="#2f8f46", zorder=3, label="flash")
        ax.set_title(f"Q {q_label} / {back_jumps} backward setpoint jumps", loc="left", fontsize=10, fontweight="bold")
        ax.set_xlabel("min in Q phase")
        if ax in axes_flat[::2]:
            ax.set_ylabel("CO2 (ppm)")
        style_axes(ax)
    for ax in axes_flat[len(main["phase_id"].unique()) :]:
        ax.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.94))
    fig.text(0.5, 0.985, "CO2 Ramp Quality: Clean Setpoint Ramps After the Fix", ha="center", va="top", fontsize=18, fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.06, top=0.89, wspace=0.18, hspace=0.42)
    fig.savefig(RAMP_QUALITY_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_aci(points: pd.DataFrame) -> None:
    raw = points.loc[points["Q_level"].isin(ACTIVE_Q_LEVELS)].copy()
    sane = raw.loc[raw["sane_for_curve"]].copy()
    flashes = sane.loc[sane["light_flash_logged"]].copy()
    binned = binned_curve_points(points)
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=min(ACTIVE_Q_LEVELS), vmax=max(ACTIVE_Q_LEVELS))

    sns.set_theme(style="whitegrid", context="talk", font_scale=0.82)
    fig, ax = plt.subplots(figsize=(12.4, 7.6))
    omitted = len(raw) - len(sane)
    ax.scatter(
        sane["Ci"],
        sane["A"],
        c=sane["Q_level"],
        cmap=cmap,
        norm=norm,
        s=8,
        alpha=0.12,
        linewidth=0,
        label="filtered DAT rows",
    )
    for q_level, group in binned.groupby("Q_level", sort=True):
        group = group.sort_values("Ci")
        color = cmap(norm(float(q_level)))
        ax.fill_between(group["Ci"], group["A_q25"], group["A_q75"], color=color, alpha=0.12, linewidth=0)
        ax.plot(group["Ci"], group["A"], color=color, lw=2.6, alpha=0.96)
        end = group.iloc[group["Ci"].argmax()]
        ax.text(end["Ci"] + 18, end["A"], f"{int(q_level)}", color=color, fontsize=9, fontweight="bold", va="center")
    ax.scatter(
        flashes["Ci"],
        flashes["A"],
        s=54,
        facecolor="white",
        edgecolor=[cmap(norm(float(q))) for q in flashes["Q_level"]],
        linewidth=1.4,
        zorder=4,
        label="MPF flash log rows",
    )
    ax.axhline(0, color="#555555", lw=1.0, ls=":")
    ax.set_title("Robust A/Ci Curves Across Q", loc="left", fontsize=16, fontweight="bold")
    ax.text(
        0.01,
        0.98,
        f"{omitted} raw rows omitted; lines are median A in 60-ppm Ci bins",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#555555",
    )
    ax.set_xlabel(r"Ci ($\mu$mol mol$^{-1}$)")
    ax.set_ylabel(r"A / DAT Adyn ($\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$)")
    ax.set_xlim(0, min(2050, float(sane["Ci"].max()) + 110))
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    style_axes(ax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(sm, ax=ax, pad=0.02)
    colorbar.set_label(r"Q ($\mu$mol photons m$^{-2}$ s$^{-1}$)")
    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.12, top=0.90)
    fig.savefig(ACI_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_surface(points: pd.DataFrame) -> None:
    plot_points = binned_curve_points(points)
    if len(plot_points) < 20:
        return
    ci_grid = np.linspace(max(0, plot_points["Ci"].quantile(0.01)), min(2000, plot_points["Ci"].quantile(0.99)), 80)
    q_grid = np.linspace(plot_points["Q"].min(), plot_points["Q"].max(), 70)
    xx, yy = np.meshgrid(ci_grid, q_grid)
    zz = griddata(plot_points[["Ci", "Q"]].to_numpy(), plot_points["A"].to_numpy(), (xx, yy), method="linear")

    sns.set_theme(style="white", context="talk", font_scale=0.72)
    fig = plt.figure(figsize=(11.8, 8.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xx, yy, zz, cmap="viridis", alpha=0.82, linewidth=0, antialiased=True)
    ax.scatter(plot_points["Ci"], plot_points["Q"], plot_points["A"], c=plot_points["Q"], cmap="viridis", s=10, alpha=0.42)
    ax.set_title("A/Q/C Surface From Filtered DAT Rows", pad=18, fontweight="bold")
    ax.set_xlabel(r"Ci ($\mu$mol mol$^{-1}$)")
    ax.set_ylabel(r"Q ($\mu$mol photons m$^{-2}$ s$^{-1}$)")
    ax.set_zlabel("A", labelpad=8)
    ax.view_init(elev=26, azim=-126)
    fig.savefig(SURFACE_OUT, dpi=220, bbox_inches="tight")
    fig.savefig(AQC_SURFACE_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_fluorescence(joined: pd.DataFrame, flashes: pd.DataFrame) -> None:
    active = joined.loc[joined["light_flash_logged"] & joined["flash_flash_id"].notna()].copy()
    if active.empty:
        return
    rename = {
        "flash_flr_PhiPS2": "fl_PhiPS2",
        "flash_flr_ETR": "fl_ETR",
        "flash_flr_NPQ": "fl_NPQ",
        "flash_flr_qL": "fl_qL",
        "flash_flr_qP": "fl_qP",
        "flash_flr_Fv'/Fm'": "fl_FvFm_prime",
        "flash_flr_PhiCO2": "fl_PhiCO2",
    }
    active = active.rename(columns=rename)
    dark = flashes.loc[flashes["flash_kind"].eq("dark")].copy().reset_index(drop=True)
    dark["order"] = np.arange(1, len(dark) + 1)
    q_summary = active.groupby("Q_level", sort=True).median(numeric_only=True).reset_index()
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=min(ACTIVE_Q_LEVELS), vmax=max(ACTIVE_Q_LEVELS))

    sns.set_theme(style="whitegrid", context="talk", font_scale=0.72)
    fig = plt.figure(figsize=(15.6, 9.6))
    grid = fig.add_gridspec(2, 3, wspace=0.32, hspace=0.38)
    ax_dark = fig.add_subplot(grid[0, 0])
    ax_decomp = fig.add_subplot(grid[0, 1])
    ax_energy = fig.add_subplot(grid[0, 2])
    ax_phi_ci = fig.add_subplot(grid[1, 0])
    ax_npq_ci = fig.add_subplot(grid[1, 1])
    ax_coupling = fig.add_subplot(grid[1, 2])

    ax_dark.axhspan(0.79, 0.83, color="#e6f2e5", alpha=0.75, lw=0)
    ax_dark.plot(dark["order"], dark["flr_Fv/Fm"], color="#2f8f46", marker="o", lw=2.0, ms=5.0)
    ax_dark.set_title("Dark Flash Baseline", loc="left", fontweight="bold")
    ax_dark.set_xlabel("dark flash sequence")
    ax_dark.set_ylabel(r"$F_v/F_m$")
    ax_dark.text(
        0.05,
        0.08,
        f"{dark['flr_Fv/Fm'].iloc[0]:.3f} -> {dark['flr_Fv/Fm'].iloc[-1]:.3f}",
        transform=ax_dark.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#2f8f46",
        fontweight="bold",
    )
    style_axes(ax_dark)

    for column, label, color in [
        ("fl_PhiPS2", r"$\Phi$PSII", "#2f8f46"),
        ("fl_FvFm_prime", r"$F_v'/F_m'$", "#2475a8"),
        ("fl_qP", "qP", "#805bb7"),
        ("fl_qL", "qL", "#d55e00"),
    ]:
        if column in q_summary:
            ax_decomp.plot(q_summary["Q_level"], q_summary[column], marker="o", lw=2.0, label=label, color=color)
    ax_decomp.set_title("Active Flash Yield By Q", loc="left", fontweight="bold")
    ax_decomp.set_xlabel("Q")
    ax_decomp.set_ylabel("yield / open centers")
    ax_decomp.set_ylim(0, 0.82)
    ax_decomp.legend(frameon=False, fontsize=8.5)
    style_axes(ax_decomp)

    ax_energy.plot(q_summary["Q_level"], q_summary["fl_ETR"], marker="o", lw=2.2, color="#227c9d")
    ax_energy.set_title("ETR and NPQ", loc="left", fontweight="bold")
    ax_energy.set_xlabel("Q")
    ax_energy.set_ylabel("ETR", color="#227c9d")
    ax_energy.tick_params(axis="y", labelcolor="#227c9d")
    ax_energy2 = ax_energy.twinx()
    ax_energy2.plot(q_summary["Q_level"], q_summary["fl_NPQ"], marker="s", lw=2.0, color="#b35c1e")
    ax_energy2.set_ylabel("NPQ", color="#b35c1e")
    ax_energy2.tick_params(axis="y", labelcolor="#b35c1e")
    ax_energy2.spines[["top"]].set_visible(False)
    ax_energy2.grid(False)
    style_axes(ax_energy)

    for q_level, group in active.groupby("Q_level", sort=True):
        color = cmap(norm(float(q_level)))
        group = group.sort_values("Ci")
        ax_phi_ci.plot(group["Ci"], group["fl_PhiPS2"], marker="o", lw=1.6, color=color, ms=4.2)
        ax_npq_ci.plot(group["Ci"], group["fl_NPQ"], marker="o", lw=1.6, color=color, ms=4.2)
    ax_phi_ci.set_title(r"$\Phi$PSII Along Ramps", loc="left", fontweight="bold")
    ax_phi_ci.set_xlabel("Ci")
    ax_phi_ci.set_ylabel(r"$\Phi$PSII")
    style_axes(ax_phi_ci)
    ax_npq_ci.set_title("NPQ Along Ramps", loc="left", fontweight="bold")
    ax_npq_ci.set_xlabel("Ci")
    ax_npq_ci.set_ylabel("NPQ")
    style_axes(ax_npq_ci)

    scatter = ax_coupling.scatter(
        active["fl_ETR"],
        active["A"],
        c=active["Ci"],
        s=32 + 0.03 * active["Q"],
        cmap="viridis",
        edgecolor="white",
        linewidth=0.45,
        alpha=0.92,
    )
    ax_coupling.axhline(0, color="#555555", lw=1.0, ls=":")
    ax_coupling.set_title("Electrons vs Carbon Gain", loc="left", fontweight="bold")
    ax_coupling.set_xlabel("ETR")
    ax_coupling.set_ylabel("A")
    style_axes(ax_coupling)
    colorbar = fig.colorbar(scatter, ax=ax_coupling, pad=0.02, shrink=0.95)
    colorbar.set_label("Ci")

    fig.text(0.06, 0.982, "Phragmites MPF Fluorescence During DAT Ramps", ha="left", va="top", fontsize=21, fontweight="bold")
    fig.text(
        0.06,
        0.948,
        "The flash bundle matched every logged active and dark flash; active flashes occurred about once per minute.",
        ha="left",
        va="top",
        fontsize=10.5,
        color="#555555",
    )
    fig.subplots_adjust(left=0.07, right=0.96, bottom=0.08, top=0.88, wspace=0.32, hspace=0.38)
    fig.savefig(FLUORO_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    points = load_points()
    flashes = load_flash_jsons()
    joined, join_report = join_flash_data(points, flashes)
    phase_summary = summarize_phases(points)

    points.to_csv(POINTS_OUT, index=False)
    flashes.to_csv(FLASH_OUT, index=False)
    joined.to_csv(JOINED_OUT, index=False)
    phase_summary.to_csv(PHASE_OUT, index=False)
    join_report.to_csv(FLASH_JOIN_OUT, index=False)

    plot_timeline(points)
    plot_stomatal_time_series(points)
    plot_ramp_quality(points, phase_summary)
    plot_aci(points)
    plot_surface(points)
    plot_fluorescence(joined, flashes)

    print("Wrote:")
    for path in [
        POINTS_OUT,
        FLASH_OUT,
        JOINED_OUT,
        PHASE_OUT,
        FLASH_JOIN_OUT,
        TIMELINE_OUT,
        STOMATAL_TS_OUT,
        RAMP_QUALITY_OUT,
        ACI_OUT,
        SURFACE_OUT,
        AQC_SURFACE_OUT,
        FLUORO_OUT,
    ]:
        print(f"  {path}")
    print("\nPhase summary:")
    print(phase_summary.to_string(index=False))
    print("\nFlash join report:")
    print(join_report.to_string(index=False))
    active = joined.loc[joined["light_flash_logged"] & joined["flash_flash_id"].notna()].rename(
        columns={
            "flash_flr_PhiPS2": "fl_PhiPS2",
            "flash_flr_ETR": "fl_ETR",
            "flash_flr_NPQ": "fl_NPQ",
            "flash_flr_qL": "fl_qL",
        }
    )
    if not active.empty:
        print("\nActive flash summary by Q:")
        print(
            active.groupby("Q_level", sort=True)
            .agg(
                n=("A", "size"),
                A_median=("A", "median"),
                A_max=("A", "max"),
                Ci_min=("Ci", "min"),
                Ci_max=("Ci", "max"),
                gsw_median=("gsw", "median"),
                PhiPS2_median=("fl_PhiPS2", "median"),
                ETR_median=("fl_ETR", "median"),
                NPQ_median=("fl_NPQ", "median"),
                qL_median=("fl_qL", "median"),
            )
            .reset_index()
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
