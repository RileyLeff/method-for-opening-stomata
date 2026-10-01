from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.interpolate import griddata

from photo26.curves import OUTPUT_DIR, read_licor_export


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "2026-06-27_phrag_qci_surface"
REGULAR_LOG_DIR = DATASET_DIR / "regular_logs"
FLASH_JSON_DIR = DATASET_DIR / "flashes_json"
DARK_TXT_DIR = DATASET_DIR / "dark_adapted_txt"

POINTS_OUT = OUTPUT_DIR / "phrag_qci_surface_points.csv"
FLASH_OUT = OUTPUT_DIR / "phrag_qci_flash_summary.csv"
DARK_OUT = OUTPUT_DIR / "phrag_qci_dark_summary.csv"
JOINED_OUT = OUTPUT_DIR / "phrag_qci_surface_points_with_flash.csv"
CURVE_SUMMARY_OUT = OUTPUT_DIR / "phrag_qci_curve_summary.csv"
JOIN_REPORT_OUT = OUTPUT_DIR / "phrag_qci_flash_join_report.csv"

ACI_PLOT_OUT = OUTPUT_DIR / "phrag_qci_aci_by_q.png"
SURFACE_3D_OUT = OUTPUT_DIR / "phrag_qci_a_surface_3d.png"
CONTOUR_OUT = OUTPUT_DIR / "phrag_qci_a_surface_contour.png"
FLUORO_OUT = OUTPUT_DIR / "phrag_qci_fluorescence_maps.png"
FLUORO_INTERPRETATION_OUT = OUTPUT_DIR / "phrag_qci_fluorescence_interpretation.png"
FLUORO_Q_SUMMARY_OUT = OUTPUT_DIR / "phrag_qci_fluorescence_by_q_summary.csv"
GSW_TIMELINE_OUT = OUTPUT_DIR / "phrag_qci_gsw_timeline.png"
GSW_PANEL_OUT = OUTPUT_DIR / "phrag_qci_gsw_by_curve_panels.png"
GSW_SUMMARY_OUT = OUTPUT_DIR / "phrag_qci_gsw_curve_summary.csv"


GAS_COLUMNS = [
    "obs",
    "elapsed",
    "time",
    "date",
    "hhmmss",
    "E",
    "Emm",
    "A",
    "Ca",
    "Ci",
    "Pci",
    "Pca",
    "gsw",
    "gbw",
    "gtw",
    "gtc",
    "Tleaf",
    "Tleaf2",
    "TleafCnd",
    "VPDleaf",
    "CO2_r_sp",
    "H2O_r_sp",
    "CO2_r",
    "CO2_s",
    "H2O_r",
    "H2O_s",
    "Q",
    "Qabs",
    "Q_red",
    "Q_blue",
    "Q_farred",
    "f_red",
    "f_blue",
    "f_farred",
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
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def q_label(q_value: float) -> str:
    return f"Q{int(round(q_value))}"


def load_surface_points() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    skipped: list[dict[str, str]] = []

    for path in sorted(REGULAR_LOG_DIR.iterdir()):
        if path.is_dir():
            continue
        try:
            frame = read_licor_export(path)
        except ValueError as exc:
            skipped.append({"source_file": path.name, "reason": str(exc)})
            continue

        if not {"A", "Ci", "Q"}.issubset(frame.columns):
            skipped.append({"source_file": path.name, "reason": "missing A, Ci, or Q"})
            continue

        keep = [column for column in GAS_COLUMNS if column in frame.columns]
        frame = frame[keep].copy()
        frame.insert(0, "source_file", path.name)
        frame["source_row"] = np.arange(1, len(frame) + 1)
        frame["elapsed_min"] = frame["elapsed"] / 60 if "elapsed" in frame.columns else np.nan
        if "date" in frame.columns:
            frame["obs_datetime"] = pd.to_datetime(frame["date"], errors="coerce")
        curve_q = float(frame["Q"].median())
        frame["curve_q"] = curve_q
        frame["curve_q_label"] = q_label(curve_q)
        frame["curve_q_order"] = int(round(curve_q))
        if {"Ci", "Ca"}.issubset(frame.columns):
            frame["Ci_Ca"] = frame["Ci"] / frame["Ca"]
        frames.append(frame)

    if not frames:
        raise RuntimeError(f"No parsable LI-6800 logs found in {REGULAR_LOG_DIR}")

    points = pd.concat(frames, ignore_index=True)
    points = points.sort_values(["curve_q", "source_file", "source_row"]).reset_index(drop=True)
    skipped_frame = pd.DataFrame(skipped)
    return points, skipped_frame


def load_flash_jsons() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(FLASH_JSON_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        flr = data.get("FLR", {})
        row: dict[str, Any] = {
            "flash_id": path.stem,
            "flash_file": path.name,
            "event_id": data.get("EVENT_ID"),
            "flash_date": data.get("DATE"),
            "flash_time": data.get("TIME"),
            "flash_timestamp": data.get("TIMESTAMP"),
            "flash_type": data.get("TYPE"),
            "pre_qin": as_float(data.get("Pre_Qin")),
            "pre_qabs": as_float(data.get("Pre_Qabs")),
            "pre_q_red": as_float(data.get("Pre_Q_red")),
            "pre_q_blue": as_float(data.get("Pre_Q_blue")),
            "pre_favg": as_float(data.get("Pre_Favg")),
            "p1_maxf": as_float(data.get("P1_MAXF")),
            "p2_slp": as_float(data.get("P2_SLP")),
            "p2_r2": as_float(data.get("P2_R2")),
            "p3_maxf": as_float(data.get("P3_MAXF")),
            "top_level_fo": as_float(data.get("Fo")),
            "top_level_fs": as_float(data.get("Fs")),
            "top_level_fmax": as_float(data.get("FMAX")),
        }
        for column in FLR_COLUMNS:
            value = flr.get(column)
            row[f"flr_{column}"] = as_float(value) if column not in {"DarkAdaptedID", "LightAdaptedID", "DarkPulseID"} else value
        light_id = row.get("flr_LightAdaptedID")
        row["flash_kind"] = "active" if isinstance(light_id, str) and light_id not in {"", "-"} else "dark"
        rows.append(row)

    return pd.DataFrame(rows)


def load_dark_txt() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(DARK_TXT_DIR.glob("*.txt")):
        data = json.loads(path.read_text())
        rows.append(
            {
                "dark_file": path.name,
                "DarkAdaptedID": data.get("DarkAdaptedID"),
                "Adark": as_float(data.get("Adark")),
                "Qmax_d": as_float(data.get("Qmax_d")),
                "Fo": as_float(data.get("Fo")),
                "Fm": as_float(data.get("Fm")),
                "Fv/Fm": (as_float(data.get("Fm")) - as_float(data.get("Fo"))) / as_float(data.get("Fm")),
                "PS2/1": as_float(data.get("PS2/1")),
            }
        )
    return pd.DataFrame(rows)


def join_flash_data(points: pd.DataFrame, flashes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    active_cols = [
        "flash_id",
        "event_id",
        "flash_file",
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

    light_expected = points["LightAdaptedID"].astype(str).ne("-").sum() if "LightAdaptedID" in points else 0
    dark_expected = points["DarkAdaptedID"].astype(str).ne("-").sum() if "DarkAdaptedID" in points else 0
    active_matched = joined["flash_flash_id"].notna().sum() if "flash_flash_id" in joined else 0
    dark_matched = joined["dark_flash_event_id"].notna().sum() if "dark_flash_event_id" in joined else 0

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
            {"metric": "surface_points", "value": len(points), "detail": ""},
            {"metric": "light_adapted_point_ids", "value": int(light_expected), "detail": ""},
            {"metric": "light_adapted_json_matches", "value": int(active_matched), "detail": ""},
            {"metric": "dark_adapted_point_ids", "value": int(dark_expected), "detail": ""},
            {"metric": "dark_adapted_json_matches", "value": int(dark_matched), "detail": ""},
            {"metric": "flash_json_total", "value": len(flashes), "detail": ""},
            {"metric": "flash_json_active", "value": int(flashes["flash_kind"].eq("active").sum()), "detail": ""},
            {"metric": "flash_json_dark", "value": int(flashes["flash_kind"].eq("dark").sum()), "detail": ""},
            {"metric": "unmatched_light_ids", "value": len(unmatched_light), "detail": ", ".join(unmatched_light)},
            {"metric": "unmatched_dark_ids", "value": len(unmatched_dark), "detail": ", ".join(unmatched_dark)},
        ]
    )
    return joined, report


def summarize_curves(points: pd.DataFrame) -> pd.DataFrame:
    agg: dict[str, tuple[str, str] | tuple[str, Any]] = {
        "n": ("A", "size"),
        "Q_median": ("Q", "median"),
        "Qabs_median": ("Qabs", "median"),
        "Ci_min": ("Ci", "min"),
        "Ci_max": ("Ci", "max"),
        "Ca_min": ("Ca", "min"),
        "Ca_max": ("Ca", "max"),
        "A_min": ("A", "min"),
        "A_max": ("A", "max"),
        "A_at_420ish": ("A", lambda values: values.iloc[0]),
        "gsw_mean": ("gsw", "mean"),
        "gsw_max": ("gsw", "max"),
        "PhiPS2_median": ("PhiPS2", "median"),
        "ETR_max": ("ETR", "max"),
        "NPQ_median": ("NPQ", "median"),
        "qL_median": ("qL", "median"),
    }
    present = {
        name: spec
        for name, spec in agg.items()
        if spec[0] in points.columns
    }
    return points.groupby("curve_q_label", sort=False).agg(**present).reset_index()


def summarize_gsw_curves(points: pd.DataFrame) -> pd.DataFrame:
    if "gsw" not in points.columns:
        return pd.DataFrame()
    ordered = points.sort_values(["curve_q", "source_row"]).copy()
    rows: list[dict[str, float | int | str]] = []
    for q_label_value, group in ordered.groupby("curve_q_label", sort=False):
        group = group.sort_values("source_row")
        rows.append(
            {
                "curve_q_label": q_label_value,
                "n": len(group),
                "Q_median": float(group["Q"].median()),
                "gsw_first": float(group["gsw"].iloc[0]),
                "gsw_last": float(group["gsw"].iloc[-1]),
                "gsw_min": float(group["gsw"].min()),
                "gsw_max": float(group["gsw"].max()),
                "gsw_mean": float(group["gsw"].mean()),
                "Ci_first": float(group["Ci"].iloc[0]),
                "Ci_max": float(group["Ci"].max()),
                "A_max": float(group["A"].max()),
            }
        )
    return pd.DataFrame(rows)


def summarize_fluorescence_by_q(points: pd.DataFrame) -> pd.DataFrame:
    active = points.loc[points["LightAdaptedID"].astype(str).ne("-")].copy()
    metric_columns = [
        "PhiPS2",
        "ETR",
        "NPQ",
        "qL",
        "qP",
        "Fv'/Fm'",
        "PhiCO2",
        "A",
        "Ci",
        "Q",
        "Qabs",
    ]
    present = [column for column in metric_columns if column in active.columns]
    if active.empty or not present:
        return pd.DataFrame()
    summary = (
        active.groupby("curve_q_label", sort=False)[present]
        .agg(["mean", "median", "min", "max"])
        .reset_index()
    )
    summary.columns = [
        column if not isinstance(column, tuple) else "_".join(part for part in column if part)
        for column in summary.columns
    ]
    return summary


def style_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#d8ddd4", linewidth=0.8)
    ax.tick_params(colors="#2d2d2d")


def plot_aci_by_q(points: pd.DataFrame) -> None:
    plot_points = points.dropna(subset=["A", "Ci", "Q"]).copy()
    plot_points = plot_points.sort_values(["curve_q", "source_row"])

    sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)
    fig, ax = plt.subplots(figsize=(12.2, 7.6))
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0, vmax=float(plot_points["curve_q"].max()))

    for q_value, group in plot_points.groupby("curve_q", sort=True):
        group = group.sort_values("source_row")
        color = cmap(norm(q_value))
        initial_branch = group.loc[group["source_row"].le(6)]
        high_branch = group.loc[group["source_row"].gt(6)]
        for branch, linestyle, alpha, linewidth in [
            (initial_branch, "--", 0.72, 1.9),
            (high_branch, "-", 1.0, 2.4),
        ]:
            ax.plot(
                branch["Ci"],
                branch["A"],
                color=color,
                lw=linewidth,
                ls=linestyle,
                alpha=alpha,
                marker="o",
                ms=5.5,
                mec="white",
                mew=0.7,
                label=f"{q_value:.0f}" if linestyle == "-" else None,
            )
        endpoint = group.loc[group["Ci"].idxmax()]
        ax.text(
            float(endpoint["Ci"]) + 18,
            float(endpoint["A"]),
            f"{q_value:.0f}",
            color=color,
            fontsize=9,
            fontweight="bold",
            va="center",
            clip_on=False,
        )

    ax.axhline(0, color="#4c4c4c", lw=1.1, ls=":")
    ax.axvline(0, color="#4c4c4c", lw=1.1, ls=":")
    ax.set_xlabel(r"Intercellular CO$_2$, Ci ($\mu$mol mol$^{-1}$)")
    ax.set_ylabel(r"Net assimilation, A ($\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$)")
    fig.text(
        0.08,
        0.97,
        "Phragmites A/Ci Slices Across Light Levels",
        ha="left",
        va="top",
        fontsize=20,
        fontweight="bold",
        color="#262626",
    )
    fig.text(
        0.08,
        0.93,
        "Dashed: initial CO2 drawdown. Solid: post-low-CO2/high-Ci branch. Labels are incident Q.",
        ha="left",
        va="top",
        fontsize=10,
        color="#555555",
    )
    ax.set_xlim(max(-25, plot_points["Ci"].min() - 40), plot_points["Ci"].max() + 105)
    style_axes(ax)
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
    colorbar.set_label(r"Incident Q ($\mu$mol photons m$^{-2}$ s$^{-1}$)")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(ACI_PLOT_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def add_global_minutes(points: pd.DataFrame) -> pd.DataFrame:
    plot_points = points.copy()
    if "obs_datetime" in plot_points.columns:
        obs_time = pd.to_datetime(plot_points["obs_datetime"], errors="coerce")
    elif "date" in plot_points.columns:
        obs_time = pd.to_datetime(plot_points["date"], errors="coerce")
    else:
        obs_time = pd.Series(pd.NaT, index=plot_points.index)

    if obs_time.notna().any():
        plot_points["global_min"] = (obs_time - obs_time.min()).dt.total_seconds() / 60
        return plot_points

    # Fallback for odd exports: reconstruct run order from the per-file elapsed clock.
    offset = 0.0
    starts: dict[str, float] = {}
    for _, group in plot_points.groupby("source_file", sort=False):
        source = str(group["source_file"].iloc[0])
        starts[source] = offset
        offset += float(group["elapsed_min"].max()) + 4.0
    plot_points["global_min"] = plot_points["elapsed_min"] + plot_points["source_file"].map(starts)
    return plot_points


def plot_gsw_timeline(points: pd.DataFrame) -> None:
    plot_points = add_global_minutes(points.dropna(subset=["gsw", "Q"]).copy())
    plot_points = plot_points.sort_values(["global_min", "source_row"])
    curve_bounds = (
        plot_points.groupby(["curve_q", "curve_q_label"], sort=True)
        .agg(start=("global_min", "min"), end=("global_min", "max"), gsw_mean=("gsw", "mean"), ci_max=("Ci", "max"))
        .reset_index()
    )

    sns.set_theme(style="whitegrid", context="talk", font_scale=0.82)
    fig, (ax_gsw, ax_ci) = plt.subplots(
        2,
        1,
        figsize=(13.8, 8.4),
        sharex=True,
        gridspec_kw={"height_ratios": [1.25, 0.85], "hspace": 0.10},
    )
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0, vmax=float(plot_points["curve_q"].max()))

    for ax in [ax_gsw, ax_ci]:
        for _, curve in curve_bounds.iterrows():
            color = cmap(norm(float(curve["curve_q"])))
            ax.axvspan(curve["start"], curve["end"], color=color, alpha=0.08, lw=0)
            ax.axvline(curve["start"], color="#b5b5ad", lw=0.9, ls="--", alpha=0.85)
        style_axes(ax)

    for q_value, group in plot_points.groupby("curve_q", sort=True):
        group = group.sort_values("global_min")
        color = cmap(norm(float(q_value)))
        ax_gsw.plot(group["global_min"], group["gsw"], color=color, marker="o", ms=5.2, lw=2.0, mec="white", mew=0.6)
        ax_ci.plot(group["global_min"], group["Ci"], color=color, marker="o", ms=4.8, lw=1.8, mec="white", mew=0.5)

    y_top = float(plot_points["gsw"].max()) * 1.05
    for _, curve in curve_bounds.iterrows():
        mid = (float(curve["start"]) + float(curve["end"])) / 2
        ax_gsw.text(
            mid,
            y_top,
            str(curve["curve_q_label"]),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=cmap(norm(float(curve["curve_q"]))),
        )

    ax_gsw.set_ylim(bottom=0, top=y_top * 1.11)
    ax_gsw.set_ylabel(r"gsw (mol H$_2$O m$^{-2}$ s$^{-1}$)")
    ax_ci.set_ylabel(r"Ci ($\mu$mol mol$^{-1}$)")
    ax_ci.set_xlabel("Elapsed time from first logged point (min)")
    ax_ci.axhline(420, color="#555555", lw=1.0, ls=":", alpha=0.75)
    ax_ci.text(
        0.995,
        420,
        "420",
        transform=ax_ci.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )

    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=[ax_gsw, ax_ci], pad=0.018, shrink=0.88)
    colorbar.set_label(r"Incident Q ($\mu$mol photons m$^{-2}$ s$^{-1}$)")
    fig.text(0.08, 0.975, "Phragmites gsw Through Sequential A/Ci Curves", ha="left", va="top", fontsize=20, fontweight="bold")
    fig.text(
        0.08,
        0.938,
        "Each shaded block is one A/Ci curve in run order; lower panel keeps Ci visible for the high-Q ceiling pattern.",
        ha="left",
        va="top",
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout(rect=[0, 0, 0.94, 0.90])
    fig.savefig(GSW_TIMELINE_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_gsw_by_curve_panels(points: pd.DataFrame) -> None:
    plot_points = points.dropna(subset=["gsw", "Q"]).copy()
    q_values = sorted(plot_points["curve_q"].unique())
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.72)
    fig, axes = plt.subplots(2, 4, figsize=(14.8, 7.8), sharey=True)
    axes_flat = list(axes.flat)
    cmap = plt.get_cmap("viridis")
    norm_q = Normalize(vmin=0, vmax=float(max(q_values)))
    norm_co2 = Normalize(vmin=0, vmax=1200)

    scatter = None
    for ax, q_value in zip(axes_flat, q_values):
        group = plot_points.loc[plot_points["curve_q"].eq(q_value)].sort_values("source_row")
        color = cmap(norm_q(float(q_value)))
        ax.plot(group["elapsed_min"], group["gsw"], color=color, lw=1.8, alpha=0.9)
        scatter = ax.scatter(
            group["elapsed_min"],
            group["gsw"],
            c=group["CO2_r_sp"],
            cmap="crest",
            norm=norm_co2,
            s=44,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        ax.set_title(f"Q {q_value:.0f}", loc="left", color=color, fontweight="bold")
        ax.text(
            0.98,
            0.92,
            f"{group['gsw'].iloc[0]:.3f} -> {group['gsw'].iloc[-1]:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            color="#555555",
        )
        style_axes(ax)
        ax.set_xlim(left=-0.4, right=max(18, float(group["elapsed_min"].max()) + 0.8))

    for ax in axes_flat[len(q_values) :]:
        ax.axis("off")

    for ax in axes[-1, :]:
        ax.set_xlabel("Minutes within curve")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"gsw (mol H$_2$O m$^{-2}$ s$^{-1}$)")

    if scatter is not None and len(q_values) < len(axes_flat):
        legend_host = axes_flat[-1]
        colorbar_axis = legend_host.inset_axes([0.32, 0.12, 0.18, 0.76])
        fig.colorbar(scatter, cax=colorbar_axis, label=r"CO$_2$ setpoint ($\mu$mol mol$^{-1}$)")
    fig.suptitle("gsw Within Each A/Ci Curve", y=0.99, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(GSW_PANEL_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def endpoint_label(ax: plt.Axes, x: float, y: float, label: str, color: tuple[float, float, float, float]) -> None:
    ax.text(
        x,
        y,
        label,
        color=color,
        fontsize=8.5,
        fontweight="bold",
        va="center",
        ha="left",
        clip_on=False,
    )


def plot_fluorescence_interpretation(points: pd.DataFrame, dark: pd.DataFrame) -> None:
    active = points.loc[points["LightAdaptedID"].astype(str).ne("-")].dropna(subset=["Q", "Ci"]).copy()
    if active.empty:
        return

    q_summary = active.groupby("curve_q", sort=True).median(numeric_only=True).reset_index()
    q_max = float(active["curve_q"].max())
    cmap = plt.get_cmap("viridis")
    norm_q = Normalize(vmin=0, vmax=q_max)

    sns.set_theme(style="whitegrid", context="talk", font_scale=0.74)
    fig = plt.figure(figsize=(15.8, 9.8))
    grid = fig.add_gridspec(2, 3, wspace=0.30, hspace=0.38)
    ax_dark = fig.add_subplot(grid[0, 0])
    ax_decomp = fig.add_subplot(grid[0, 1])
    ax_energy = fig.add_subplot(grid[0, 2])
    ax_ql_ci = fig.add_subplot(grid[1, 0])
    ax_npq_ci = fig.add_subplot(grid[1, 1])
    ax_coupling = fig.add_subplot(grid[1, 2])

    dark_plot = dark.copy().reset_index(drop=True)
    if not dark_plot.empty and "Fv/Fm" in dark_plot.columns:
        dark_plot["flash_order"] = np.arange(1, len(dark_plot) + 1)
        ax_dark.axhspan(0.79, 0.83, color="#e6f2e5", alpha=0.75, lw=0)
        ax_dark.plot(
            dark_plot["flash_order"],
            dark_plot["Fv/Fm"],
            color="#2f8f46",
            marker="o",
            lw=2.2,
            ms=5.2,
            mec="white",
            mew=0.6,
        )
        ax_dark.set_title("Dark-Adapted Baseline", loc="left", fontweight="bold")
        ax_dark.set_xlabel("Dark flash sequence")
        ax_dark.set_ylabel(r"$F_v/F_m$")
        ax_dark.set_ylim(min(0.775, float(dark_plot["Fv/Fm"].min()) - 0.006), max(0.835, float(dark_plot["Fv/Fm"].max()) + 0.006))
        ax_dark.text(
            0.05,
            0.08,
            f"{dark_plot['Fv/Fm'].iloc[0]:.3f} -> {dark_plot['Fv/Fm'].iloc[-1]:.3f}",
            transform=ax_dark.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            color="#2f8f46",
            fontweight="bold",
        )
        ax_dark.text(
            0.98,
            0.93,
            "high initial baseline,\nslight drift down",
            transform=ax_dark.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            color="#555555",
        )
    else:
        ax_dark.text(0.5, 0.5, "No dark flash summary", ha="center", va="center", transform=ax_dark.transAxes)
    style_axes(ax_dark)

    decomp_cols = [
        ("PhiPS2", r"$\Phi$PSII", "#2f8f46"),
        ("Fv'/Fm'", r"$F_v'/F_m'$", "#2475a8"),
        ("qP", "qP", "#805bb7"),
        ("qL", "qL", "#d55e00"),
    ]
    for column, label, color in decomp_cols:
        if column not in q_summary.columns:
            continue
        ax_decomp.plot(q_summary["curve_q"], q_summary[column], marker="o", lw=2.0, ms=5.0, label=label, color=color)
    ax_decomp.set_title(r"Why $\Phi$PSII Falls", loc="left", fontweight="bold")
    ax_decomp.set_xlabel(r"Incident Q ($\mu$mol photons m$^{-2}$ s$^{-1}$)")
    ax_decomp.set_ylabel("fluorescence yield / open centers")
    ax_decomp.set_ylim(0, 0.86)
    ax_decomp.legend(frameon=False, fontsize=8.5, loc="upper right")
    style_axes(ax_decomp)

    if {"ETR", "NPQ"}.issubset(q_summary.columns):
        etr_color = "#227c9d"
        npq_color = "#b35c1e"
        ax_energy.plot(q_summary["curve_q"], q_summary["ETR"], marker="o", lw=2.2, color=etr_color, label="ETR")
        ax_energy.set_ylabel(r"ETR ($\mu$mol e$^-$ m$^{-2}$ s$^{-1}$)", color=etr_color)
        ax_energy.tick_params(axis="y", labelcolor=etr_color)
        ax_energy.set_xlabel(r"Incident Q ($\mu$mol photons m$^{-2}$ s$^{-1}$)")
        ax_energy.set_title("ETR Saturates, NPQ Rises", loc="left", fontweight="bold")
        ax_energy_2 = ax_energy.twinx()
        ax_energy_2.plot(q_summary["curve_q"], q_summary["NPQ"], marker="s", lw=2.0, color=npq_color, label="NPQ")
        ax_energy_2.set_ylabel("NPQ", color=npq_color)
        ax_energy_2.tick_params(axis="y", labelcolor=npq_color)
        ax_energy_2.spines[["top"]].set_visible(False)
        ax_energy_2.grid(False)
        style_axes(ax_energy)

    for q_value, group in active.groupby("curve_q", sort=True):
        group = group.sort_values("Ci")
        color = cmap(norm_q(float(q_value)))
        ax_ql_ci.plot(group["Ci"], group["qL"], color=color, marker="o", lw=1.8, ms=4.4, alpha=0.95)
        endpoint_label(ax_ql_ci, float(group["Ci"].max()) + 14, float(group["qL"].iloc[-1]), f"{q_value:.0f}", color)
    ax_ql_ci.set_title("More Ci Reopens PSII", loc="left", fontweight="bold")
    ax_ql_ci.set_xlabel(r"Ci ($\mu$mol mol$^{-1}$)")
    ax_ql_ci.set_ylabel("qL")
    ax_ql_ci.set_xlim(left=max(0, float(active["Ci"].min()) - 20), right=float(active["Ci"].max()) + 95)
    style_axes(ax_ql_ci)

    for q_value, group in active.groupby("curve_q", sort=True):
        group = group.sort_values("Ci")
        color = cmap(norm_q(float(q_value)))
        ax_npq_ci.plot(group["Ci"], group["NPQ"], color=color, marker="o", lw=1.8, ms=4.4, alpha=0.95)
        endpoint_label(ax_npq_ci, float(group["Ci"].max()) + 14, float(group["NPQ"].iloc[-1]), f"{q_value:.0f}", color)
    ax_npq_ci.set_title("More Ci Relaxes NPQ", loc="left", fontweight="bold")
    ax_npq_ci.set_xlabel(r"Ci ($\mu$mol mol$^{-1}$)")
    ax_npq_ci.set_ylabel("NPQ")
    ax_npq_ci.set_xlim(left=max(0, float(active["Ci"].min()) - 20), right=float(active["Ci"].max()) + 95)
    style_axes(ax_npq_ci)

    coupling = active.dropna(subset=["ETR", "A", "Ci", "Q"])
    scatter = ax_coupling.scatter(
        coupling["ETR"],
        coupling["A"],
        c=coupling["Ci"],
        s=30 + 0.03 * coupling["Q"],
        cmap="viridis",
        edgecolor="white",
        linewidth=0.45,
        alpha=0.92,
    )
    ax_coupling.axhline(0, color="#555555", lw=1.0, ls=":")
    ax_coupling.set_title("Electrons vs Carbon Gain", loc="left", fontweight="bold")
    ax_coupling.set_xlabel(r"ETR ($\mu$mol e$^-$ m$^{-2}$ s$^{-1}$)")
    ax_coupling.set_ylabel(r"A ($\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$)")
    ax_coupling.text(
        0.04,
        0.18,
        "low-Ci points can keep\nETR high while A stays low",
        transform=ax_coupling.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )
    style_axes(ax_coupling)
    colorbar = fig.colorbar(scatter, ax=ax_coupling, pad=0.02, shrink=0.94)
    colorbar.set_label(r"Ci ($\mu$mol mol$^{-1}$)")

    fig.text(0.06, 0.982, "Phragmites Fluorescence Interpretation", ha="left", va="top", fontsize=21, fontweight="bold")
    fig.text(
        0.06,
        0.946,
        "Dark flash gives baseline PSII capacity; active flashes show high light shifting energy toward NPQ unless Ci/A provide an electron sink.",
        ha="left",
        va="top",
        fontsize=10.5,
        color="#555555",
    )
    fig.subplots_adjust(left=0.07, right=0.96, bottom=0.08, top=0.88, wspace=0.30, hspace=0.38)
    fig.savefig(FLUORO_INTERPRETATION_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_surface_3d(points: pd.DataFrame) -> None:
    plot_points = points.dropna(subset=["A", "Ci", "Q"]).copy()
    sns.set_theme(style="white", context="talk", font_scale=0.85)

    fig = plt.figure(figsize=(11.8, 8.4))
    ax = fig.add_subplot(111, projection="3d")
    norm = Normalize(vmin=float(plot_points["A"].min()), vmax=float(plot_points["A"].max()))
    trisurf = ax.plot_trisurf(
        plot_points["Ci"],
        plot_points["Q"],
        plot_points["A"],
        cmap="viridis",
        norm=norm,
        alpha=0.78,
        linewidth=0.25,
        edgecolor="#263238",
        antialiased=True,
    )
    ax.scatter(
        plot_points["Ci"],
        plot_points["Q"],
        plot_points["A"],
        c=plot_points["A"],
        cmap="viridis",
        norm=norm,
        s=28,
        edgecolor="white",
        linewidth=0.45,
        depthshade=False,
    )

    ax.set_xlabel(r"Ci ($\mu$mol mol$^{-1}$)", labelpad=10)
    ax.set_ylabel(r"Q ($\mu$mol photons m$^{-2}$ s$^{-1}$)", labelpad=12)
    ax.set_zlabel(r"A ($\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$)", labelpad=10)
    ax.set_title("Phragmites A = f(Ci, Q)", pad=16, fontweight="bold")
    ax.view_init(elev=28, azim=-138)
    ax.xaxis.pane.set_facecolor((0.96, 0.97, 0.94, 1))
    ax.yaxis.pane.set_facecolor((0.96, 0.97, 0.94, 1))
    ax.zaxis.pane.set_facecolor((1, 1, 1, 0))
    fig.colorbar(trisurf, ax=ax, shrink=0.72, pad=0.08, label=r"A ($\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$)")
    fig.tight_layout()
    fig.savefig(SURFACE_3D_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_surface_contour(points: pd.DataFrame) -> None:
    plot_points = points.dropna(subset=["A", "Ci", "Q"]).copy()
    xi = np.linspace(float(plot_points["Ci"].min()), float(plot_points["Ci"].max()), 180)
    yi = np.linspace(float(plot_points["Q"].min()), float(plot_points["Q"].max()), 180)
    grid_x, grid_y = np.meshgrid(xi, yi)
    grid_z = griddata((plot_points["Ci"], plot_points["Q"]), plot_points["A"], (grid_x, grid_y), method="linear")

    sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)
    fig, ax = plt.subplots(figsize=(11.4, 7.4))
    contour = ax.contourf(grid_x, grid_y, grid_z, levels=18, cmap="viridis")
    lines = ax.contour(grid_x, grid_y, grid_z, levels=8, colors="white", linewidths=0.65, alpha=0.75)
    ax.clabel(lines, inline=True, fontsize=8, fmt="%.1f")
    ax.scatter(plot_points["Ci"], plot_points["Q"], c="black", s=18, alpha=0.82)
    ax.set_xlabel(r"Ci ($\mu$mol mol$^{-1}$)")
    ax.set_ylabel(r"Q ($\mu$mol photons m$^{-2}$ s$^{-1}$)")
    ax.set_xlim(float(plot_points["Ci"].min()) - 25, float(plot_points["Ci"].max()) + 30)
    ax.set_ylim(float(plot_points["Q"].min()) - 65, float(plot_points["Q"].max()) + 85)
    fig.text(
        0.08,
        0.97,
        "Interpolated Phragmites A Surface",
        ha="left",
        va="top",
        fontsize=20,
        fontweight="bold",
        color="#262626",
    )
    fig.text(
        0.08,
        0.93,
        "Black points are measured observations; filled colors are linear interpolation between sampled points.",
        ha="left",
        va="top",
        fontsize=10,
        color="#555555",
    )
    style_axes(ax)
    fig.colorbar(contour, ax=ax, pad=0.02, label=r"A ($\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$)")
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(CONTOUR_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_fluorescence_maps(points: pd.DataFrame) -> None:
    plot_points = points.loc[points["LightAdaptedID"].astype(str).ne("-")].dropna(subset=["Ci", "Q"]).copy()
    metrics = [
        ("PhiPS2", r"$\Phi$PSII"),
        ("ETR", "ETR"),
        ("NPQ", "NPQ"),
        ("qL", "qL"),
    ]
    metrics = [(column, title) for column, title in metrics if column in plot_points.columns]
    if not metrics:
        return

    sns.set_theme(style="whitegrid", context="talk", font_scale=0.78)
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.0), sharex=True, sharey=True)
    for ax, (column, title) in zip(axes.flat, metrics):
        valid = plot_points.dropna(subset=[column])
        scatter = ax.scatter(
            valid["Ci"],
            valid["Q"],
            c=valid[column],
            cmap="mako" if column in {"NPQ"} else "viridis",
            s=48,
            edgecolor="white",
            linewidth=0.45,
        )
        ax.set_title(title, loc="left", fontweight="bold")
        style_axes(ax)
        fig.colorbar(scatter, ax=ax, pad=0.02)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"Ci ($\mu$mol mol$^{-1}$)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Q ($\mu$mol photons m$^{-2}$ s$^{-1}$)")
    fig.suptitle("Organized Fluorescence at A/Ci Surface Points", y=0.99, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FLUORO_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_outputs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    points, skipped = load_surface_points()
    flashes = load_flash_jsons()
    dark = load_dark_txt()
    joined, join_report = join_flash_data(points, flashes)
    curve_summary = summarize_curves(points)
    gsw_summary = summarize_gsw_curves(points)
    fluorescence_summary = summarize_fluorescence_by_q(points)

    points.to_csv(POINTS_OUT, index=False)
    flashes.to_csv(FLASH_OUT, index=False)
    dark.to_csv(DARK_OUT, index=False)
    joined.to_csv(JOINED_OUT, index=False)
    curve_summary.to_csv(CURVE_SUMMARY_OUT, index=False)
    gsw_summary.to_csv(GSW_SUMMARY_OUT, index=False)
    fluorescence_summary.to_csv(FLUORO_Q_SUMMARY_OUT, index=False)
    join_report.to_csv(JOIN_REPORT_OUT, index=False)
    if not skipped.empty:
        skipped.to_csv(OUTPUT_DIR / "phrag_qci_skipped_regular_logs.csv", index=False)

    plot_aci_by_q(points)
    plot_surface_3d(points)
    plot_surface_contour(points)
    plot_fluorescence_maps(joined)
    plot_fluorescence_interpretation(joined, dark)
    plot_gsw_timeline(points)
    plot_gsw_by_curve_panels(points)

    print("Wrote:")
    for path in [
        POINTS_OUT,
        FLASH_OUT,
        DARK_OUT,
        JOINED_OUT,
        CURVE_SUMMARY_OUT,
        GSW_SUMMARY_OUT,
        FLUORO_Q_SUMMARY_OUT,
        JOIN_REPORT_OUT,
        ACI_PLOT_OUT,
        SURFACE_3D_OUT,
        CONTOUR_OUT,
        FLUORO_OUT,
        FLUORO_INTERPRETATION_OUT,
        GSW_TIMELINE_OUT,
        GSW_PANEL_OUT,
    ]:
        print(f"  {path}")
    print()
    print(curve_summary.to_string(index=False))
    print()
    print(gsw_summary.to_string(index=False))
    if not fluorescence_summary.empty:
        print()
        print(fluorescence_summary.to_string(index=False))
    print()
    print(join_report.to_string(index=False))


if __name__ == "__main__":
    write_outputs()
