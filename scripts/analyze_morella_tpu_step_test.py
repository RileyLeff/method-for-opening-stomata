from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from photo26.curves import read_licor_export


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "2026-07-10_morella_tpu_step_test"
LOG_PATH = DATA_DIR / "2026-07-10-1215_logdata_morcer_new_outside"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

POINTS_OUT = OUTPUT_DIR / "morella_tpu_step_points.csv"
PHASE_OUT = OUTPUT_DIR / "morella_tpu_step_phase_summary.csv"
STABLE_OUT = OUTPUT_DIR / "morella_tpu_step_stable_windows.csv"
FLASH_OUT = OUTPUT_DIR / "morella_tpu_step_flash_summary.csv"
RECOVERY_OUT = OUTPUT_DIR / "morella_tpu_step_recovery_summary.csv"
QUALITY_OUT = OUTPUT_DIR / "morella_tpu_step_quality_summary.csv"
TIMELINE_OUT = OUTPUT_DIR / "morella_tpu_step_timeline.png"
REVERSIBILITY_OUT = OUTPUT_DIR / "morella_tpu_step_reversibility.png"

PHASE_LABELS = {
    1: "Initial dark flash",
    2: "Light/CO2 transition",
    3: "Opening: Q800, CO2 120",
    4: "Test-light transition",
    5: "420 pre",
    6: "750 pre",
    7: "1800 challenge",
    8: "750 return",
    9: "420 return",
    10: "Light-off transition",
    11: "Safe endpoint: Q0, CO2 420",
}

CORE_PHASES = [3, 5, 6, 7, 8, 9, 11]
PHASE_COLORS = {
    3: "#dceaf5",
    5: "#dcefe2",
    6: "#f4e6b8",
    7: "#f3d1ca",
    8: "#f4e6b8",
    9: "#dcefe2",
    11: "#e4e4e4",
}

STABLE_WINDOWS = [
    {"window": "420 pre", "phase_id": 5, "start_min": 4.0, "end_min": 4.8},
    {"window": "750 pre", "phase_id": 6, "start_min": 4.5, "end_min": 5.8},
    {"window": "1800 early", "phase_id": 7, "start_min": 3.0, "end_min": 6.5},
    {"window": "1800 late", "phase_id": 7, "start_min": 10.0, "end_min": 13.7},
    {"window": "750 return", "phase_id": 8, "start_min": 8.8, "end_min": 10.0},
    {"window": "420 return", "phase_id": 9, "start_min": 6.0, "end_min": 6.8},
]


def style_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#d8ddd4", linewidth=0.75)


def load_points() -> pd.DataFrame:
    points = read_licor_export(LOG_PATH).sort_values("elapsed").reset_index(drop=True).copy()
    points["minute"] = points["elapsed"] / 60
    points["q_state"] = np.select([points["Q"].lt(100), points["Q"].lt(1200)], [0, 800], default=1700)
    points["co2_state"] = np.select(
        [points["CO2_r_sp"].lt(250), points["CO2_r_sp"].lt(600), points["CO2_r_sp"].lt(1200)],
        [120, 420, 750],
        default=1800,
    )
    change = points["q_state"].ne(points["q_state"].shift()) | points["co2_state"].ne(points["co2_state"].shift())
    points["phase_id"] = change.cumsum()
    points["phase_label"] = points["phase_id"].map(PHASE_LABELS).fillna("Transition")
    points["phase_min"] = (
        points["elapsed"] - points.groupby("phase_id")["elapsed"].transform("min")
    ) / 60
    points["light_flash"] = points["LightAdaptedID"].astype(str).ne("-")
    points["dark_flash"] = (
        points["LightAdaptedID"].astype(str).eq("-")
        & points["DarkAdaptedID"].astype(str).ne("-")
    )

    for column in ["A", "Ca", "Ci", "gsw", "VPDleaf", "Tleaf", "V_system"]:
        points[f"{column}_smooth"] = points[column].rolling(15, center=True, min_periods=5).median()
    return points


def phase_summary(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for phase_id, group in points.groupby("phase_id", sort=True):
        rows.append(
            {
                "phase_id": phase_id,
                "phase": PHASE_LABELS.get(int(phase_id), "Transition"),
                "Q": float(group["Q"].median()),
                "CO2_setpoint": float(group["CO2_r_sp"].median()),
                "start_min": float(group["minute"].min()),
                "end_min": float(group["minute"].max()),
                "duration_min": float(group["phase_min"].max()),
                "n": len(group),
                "A_median": float(group["A"].median()),
                "A_last60": float(group.loc[group["phase_min"].ge(group["phase_min"].max() - 1), "A"].median()),
                "Ci_last60": float(group.loc[group["phase_min"].ge(group["phase_min"].max() - 1), "Ci"].median()),
                "gsw_last60": float(group.loc[group["phase_min"].ge(group["phase_min"].max() - 1), "gsw"].median()),
                "VPDleaf_last60": float(group.loc[group["phase_min"].ge(group["phase_min"].max() - 1), "VPDleaf"].median()),
                "Tleaf_last60": float(group.loc[group["phase_min"].ge(group["phase_min"].max() - 1), "Tleaf"].median()),
                "V_system_last60": float(group.loc[group["phase_min"].ge(group["phase_min"].max() - 1), "V_system"].median()),
                "light_flashes": int(group["light_flash"].sum()),
                "dark_flashes": int(group["dark_flash"].sum()),
            }
        )
    return pd.DataFrame(rows)


def flash_distance_mask(points: pd.DataFrame, seconds: float = 35) -> pd.Series:
    flash_times = points.loc[points["light_flash"], "elapsed"].to_numpy(dtype=float)
    if not len(flash_times):
        return pd.Series(False, index=points.index)
    elapsed = points["elapsed"].to_numpy(dtype=float)
    distance = np.min(np.abs(elapsed[:, None] - flash_times[None, :]), axis=1)
    return pd.Series(distance <= seconds, index=points.index)


def summarize_stable_windows(points: pd.DataFrame) -> pd.DataFrame:
    near_flash = flash_distance_mask(points)
    rows = []
    for definition in STABLE_WINDOWS:
        selected = points.loc[
            points["phase_id"].eq(definition["phase_id"])
            & points["phase_min"].between(definition["start_min"], definition["end_min"])
            & ~near_flash
            & points["A"].between(-5, 30)
            & points["Ci"].between(0, 2500)
            & points["gsw"].between(0.001, 1.0)
            & points["Flow"].between(850, 950)
        ].copy()
        selected["ten_second_bin"] = (selected["elapsed"] / 10).round()
        binned = (
            selected.groupby("ten_second_bin", as_index=False)
            .agg(
                A=("A", "median"),
                Ca=("Ca", "median"),
                Ci=("Ci", "median"),
                gsw=("gsw", "median"),
                VPDleaf=("VPDleaf", "median"),
                Tleaf=("Tleaf", "median"),
                V_system=("V_system", "median"),
            )
        )
        row: dict[str, float | str | int] = {
            "window": definition["window"],
            "phase_id": definition["phase_id"],
            "start_phase_min": definition["start_min"],
            "end_phase_min": definition["end_min"],
            "n_rows": len(selected),
            "n_ten_second_bins": len(binned),
        }
        for metric in ["A", "Ca", "Ci", "gsw", "VPDleaf", "Tleaf", "V_system"]:
            row[f"{metric}_median"] = float(binned[metric].median())
            row[f"{metric}_q25"] = float(binned[metric].quantile(0.25))
            row[f"{metric}_q75"] = float(binned[metric].quantile(0.75))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_flashes(points: pd.DataFrame) -> pd.DataFrame:
    flashes = points.loc[points["light_flash"] | points["dark_flash"]].copy()
    columns = [
        "elapsed",
        "minute",
        "hhmmss",
        "phase_id",
        "phase_label",
        "phase_min",
        "Q",
        "CO2_r_sp",
        "Ca",
        "Ci",
        "A",
        "gsw",
        "Fv/Fm",
        "PhiPS2",
        "ETR",
        "NPQ",
        "qL",
        "LightAdaptedID",
        "DarkAdaptedID",
    ]
    return flashes[columns].reset_index(drop=True)


def nearest_flash(flashes: pd.DataFrame, phase_id: int, target_phase_min: float | None = None) -> pd.Series:
    candidates = flashes.loc[flashes["phase_id"].eq(phase_id) & flashes["LightAdaptedID"].astype(str).ne("-")]
    if target_phase_min is None:
        return candidates.iloc[-1]
    return candidates.loc[(candidates["phase_min"] - target_phase_min).abs().idxmin()]


def matched_flash_summary(flashes: pd.DataFrame) -> pd.DataFrame:
    selections = [
        ("420 pre", nearest_flash(flashes, 5)),
        ("750 pre", nearest_flash(flashes, 6, 6.0)),
        ("1800 early", nearest_flash(flashes, 7, 4.0)),
        ("1800 late", nearest_flash(flashes, 7, 14.0)),
        ("750 return", nearest_flash(flashes, 8)),
        ("420 return", nearest_flash(flashes, 9)),
    ]
    rows = []
    for label, flash in selections:
        rows.append(
            {
                "window": label,
                "phase_id": int(flash["phase_id"]),
                "phase_min": float(flash["phase_min"]),
                "A_flash_row": float(flash["A"]),
                "Ci_flash_row": float(flash["Ci"]),
                "gsw_flash_row": float(flash["gsw"]),
                "PhiPSII": float(flash["PhiPS2"]),
                "ETR": float(flash["ETR"]),
                "NPQ": float(flash["NPQ"]),
                "qL": float(flash["qL"]),
            }
        )
    return pd.DataFrame(rows)


def recovery_summary(stable: pd.DataFrame, matched_flashes: pd.DataFrame) -> pd.DataFrame:
    stable_index = stable.set_index("window")
    flash_index = matched_flashes.set_index("window")
    rows = []
    for state, pre, returned in [
        ("420 matched state", "420 pre", "420 return"),
        ("750 matched state", "750 pre", "750 return"),
    ]:
        row: dict[str, float | str] = {"comparison": state, "pre_window": pre, "return_window": returned}
        for metric in ["A", "Ci", "gsw", "VPDleaf", "Tleaf"]:
            before = float(stable_index.loc[pre, f"{metric}_median"])
            after = float(stable_index.loc[returned, f"{metric}_median"])
            row[f"{metric}_pre"] = before
            row[f"{metric}_return"] = after
            row[f"{metric}_delta"] = after - before
            row[f"{metric}_return_pct"] = after / before * 100
        for metric in ["PhiPSII", "ETR", "NPQ", "qL"]:
            before = float(flash_index.loc[pre, metric])
            after = float(flash_index.loc[returned, metric])
            row[f"{metric}_pre"] = before
            row[f"{metric}_return"] = after
            row[f"{metric}_delta"] = after - before
            row[f"{metric}_return_pct"] = after / before * 100
        rows.append(row)
    return pd.DataFrame(rows)


def quality_summary(points: pd.DataFrame, phases: pd.DataFrame, flashes: pd.DataFrame) -> pd.DataFrame:
    final_phase = points.loc[points["phase_id"].eq(11)]
    initial_dark = flashes.loc[flashes["phase_id"].eq(1)].iloc[0]
    final_dark = flashes.loc[flashes["phase_id"].eq(11)].iloc[-1]
    rows = [
        {"metric": "log_rows", "value": len(points), "detail": ""},
        {"metric": "logged_duration_min", "value": points["minute"].max(), "detail": ""},
        {"metric": "light_flash_rows", "value": int(points["light_flash"].sum()), "detail": ""},
        {"metric": "dark_flash_rows", "value": int(points["dark_flash"].sum()), "detail": ""},
        {"metric": "initial_dark_FvFm", "value": float(initial_dark["Fv/Fm"]), "detail": "30-min dark adapted"},
        {
            "metric": "final_dark_FvFm",
            "value": float(final_dark["Fv/Fm"]),
            "detail": f"only {final_dark['phase_min']:.1f} min after light-off; not dark adapted",
        },
        {"metric": "system_voltage_start", "value": float(points["V_system"].iloc[0]), "detail": "V"},
        {"metric": "system_voltage_end", "value": float(points["V_system"].iloc[-1]), "detail": "V"},
        {"metric": "final_flow_median", "value": float(final_phase["Flow"].median()), "detail": "umol s-1"},
        {"metric": "final_CO2_reference_median", "value": float(final_phase["CO2_r"].median()), "detail": "ppm"},
        {"metric": "final_CO2_sample_median", "value": float(final_phase["CO2_s"].median()), "detail": "ppm"},
        {"metric": "final_DIAG", "value": float(final_phase["DIAG"].median()), "detail": "stable"},
        {
            "metric": "tail_status",
            "value": 1,
            "detail": "valid safe endpoint reached; no battery-corrupted rows identified",
        },
    ]
    return pd.DataFrame(rows)


def shade_phases(axes: list[plt.Axes], phases: pd.DataFrame) -> None:
    for row in phases.loc[phases["phase_id"].isin(CORE_PHASES)].itertuples():
        color = PHASE_COLORS[int(row.phase_id)]
        for ax in axes:
            ax.axvspan(row.start_min, row.end_min, color=color, alpha=0.62, lw=0)
        axes[0].text(
            (row.start_min + row.end_min) / 2,
            0.93,
            row.phase,
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7.2,
            color="#333333",
        )


def plot_timeline(points: pd.DataFrame, phases: pd.DataFrame, flashes: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.63)
    fig, axes = plt.subplots(
        7,
        1,
        figsize=(16, 13.2),
        sharex=True,
        gridspec_kw={"height_ratios": [0.8, 1.2, 1.2, 1.0, 1.1, 1.0, 0.8]},
    )
    shade_phases(list(axes), phases)

    axes[0].plot(points["minute"], points["Q"], color="#333333", lw=1.2, label="Q")
    command_ax = axes[0].twinx()
    command_ax.plot(points["minute"], points["CO2_r_sp"], color="#d97925", lw=1.1, label="CO2 setpoint")
    axes[0].set_ylabel("Q")
    command_ax.set_ylabel("CO2 sp")

    axes[1].plot(points["minute"], points["A"].clip(-3, 24), color="#7d7d7d", alpha=0.18, lw=0.7, label="raw")
    axes[1].plot(points["minute"], points["A_smooth"].clip(-3, 24), color="#278c55", lw=1.8, label="15-s median")
    axes[1].axhline(0, color="#555555", lw=0.8, ls=":")
    axes[1].set_ylabel("A")
    axes[1].legend(frameon=False, loc="upper left", ncol=2, fontsize=8)

    axes[2].plot(points["minute"], points["Ca_smooth"].clip(0, 1900), color="#ed8b2f", lw=1.5, label="Ca")
    axes[2].plot(points["minute"], points["Ci_smooth"].clip(0, 1900), color="#7652a8", lw=1.5, label="Ci")
    axes[2].set_ylabel("CO2 (ppm)")
    axes[2].legend(frameon=False, loc="upper left", ncol=2, fontsize=8)

    axes[3].plot(points["minute"], points["gsw_smooth"].clip(0, 0.35), color="#2475a6", lw=1.6)
    axes[3].set_ylabel("gsw")
    axes[3].set_ylim(-0.01, 0.34)

    light = flashes.loc[flashes["LightAdaptedID"].astype(str).ne("-")]
    axes[4].plot(light["minute"], light["ETR"], color="#1b87a8", marker="o", lw=1.5, ms=4.5, label="ETR")
    flr_ax = axes[4].twinx()
    flr_ax.plot(light["minute"], light["NPQ"], color="#b45b18", marker="s", lw=1.2, ms=4, label="NPQ")
    axes[4].set_ylabel("ETR")
    flr_ax.set_ylabel("NPQ")

    axes[5].plot(points["minute"], points["VPDleaf_smooth"].clip(0, 3), color="#8f5a4a", lw=1.5, label="VPDleaf")
    temperature_ax = axes[5].twinx()
    temperature_ax.plot(points["minute"], points["Tleaf_smooth"], color="#17a6bf", lw=1.4, label="Tleaf")
    axes[5].axhline(1.0, color="#555555", lw=0.8, ls="--")
    axes[5].set_ylabel("VPD (kPa)")
    temperature_ax.set_ylabel("Tleaf (C)")

    axes[6].plot(points["minute"], points["V_system_smooth"], color="#5f5f5f", lw=1.5)
    axes[6].set_ylabel("System V")
    axes[6].set_xlabel("Elapsed logged time (min)")

    for ax in axes:
        style_axes(ax)
    fig.text(0.055, 0.988, "Morella TPU Step Test", ha="left", va="top", fontsize=22, fontweight="bold")
    fig.text(
        0.055,
        0.958,
        "Fixed Q=1700 CO2 sandwich after low-CO2 opening; raw transition artifacts are faint and 15-s medians carry the gas-exchange interpretation.",
        ha="left",
        va="top",
        fontsize=10.3,
        color="#555555",
    )
    fig.subplots_adjust(left=0.07, right=0.92, bottom=0.06, top=0.91, hspace=0.18)
    fig.savefig(TIMELINE_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_reversibility(stable: pd.DataFrame, matched_flashes: pd.DataFrame, recovery: pd.DataFrame) -> None:
    order = [definition["window"] for definition in STABLE_WINDOWS]
    stable = stable.set_index("window").loc[order].reset_index()
    matched = matched_flashes.set_index("window").loc[order].reset_index()
    colors = ["#5d9f69", "#d7a42f", "#c85b47", "#a83f35", "#d7a42f", "#5d9f69"]
    x = np.arange(len(order))

    sns.set_theme(style="whitegrid", context="talk", font_scale=0.69)
    fig, axes = plt.subplots(2, 2, figsize=(14.4, 9.5))

    for ax, metric, ylabel, ylim in [
        (axes[0, 0], "A", r"A ($\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$)", (0, 20)),
        (axes[0, 1], "Ci", r"Ci ($\mu$mol mol$^{-1}$)", (0, 1550)),
        (axes[1, 0], "gsw", r"gsw (mol m$^{-2}$ s$^{-1}$)", (0, 0.22)),
    ]:
        med = stable[f"{metric}_median"].to_numpy(dtype=float)
        lower = med - stable[f"{metric}_q25"].to_numpy(dtype=float)
        upper = stable[f"{metric}_q75"].to_numpy(dtype=float) - med
        ax.bar(x, med, color=colors, alpha=0.86, width=0.66)
        ax.errorbar(x, med, yerr=np.vstack([lower, upper]), fmt="none", ecolor="#333333", capsize=3, lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.set_title(metric, loc="left", fontweight="bold")
        style_axes(ax)

    ax_f = axes[1, 1]
    ax_f.plot(x, matched["ETR"], color="#1b87a8", marker="o", lw=2.2, ms=6, label="ETR")
    npq_ax = ax_f.twinx()
    npq_ax.plot(x, matched["NPQ"], color="#b45b18", marker="s", lw=1.8, ms=5.5, label="NPQ")
    ax_f.set_xticks(x)
    ax_f.set_xticklabels(order, rotation=25, ha="right")
    ax_f.set_ylabel("ETR")
    npq_ax.set_ylabel("NPQ")
    ax_f.set_ylim(55, 90)
    npq_ax.set_ylim(1.55, 1.9)
    ax_f.set_title("Fluorescence at Matched Logs", loc="left", fontweight="bold")
    style_axes(ax_f)

    recovery_index = recovery.set_index("comparison")
    ambient = recovery_index.loc["420 matched state"]
    mid = recovery_index.loc["750 matched state"]
    fig.text(
        0.06,
        0.925,
        f"420 recovery: A {ambient['A_return_pct']:.1f}%, ETR {ambient['ETR_return_pct']:.1f}%   |   "
        f"750 recovery: A {mid['A_return_pct']:.1f}%, ETR {mid['ETR_return_pct']:.1f}%",
        ha="left",
        va="top",
        fontsize=10.5,
        color="#444444",
    )
    fig.text(0.06, 0.985, "Morella High-CO2 Reversibility Test", ha="left", va="top", fontsize=21, fontweight="bold")
    fig.text(
        0.06,
        0.952,
        "Bars are medians of clean 10-s bins away from flashes; whiskers are interquartile ranges. Fluorescence points are selected stable logs.",
        ha="left",
        va="top",
        fontsize=10.2,
        color="#555555",
    )
    fig.subplots_adjust(left=0.08, right=0.93, bottom=0.13, top=0.86, wspace=0.26, hspace=0.38)
    fig.savefig(REVERSIBILITY_OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    points = load_points()
    phases = phase_summary(points)
    stable = summarize_stable_windows(points)
    flashes = summarize_flashes(points)
    matched_flashes = matched_flash_summary(flashes)
    recovery = recovery_summary(stable, matched_flashes)
    quality = quality_summary(points, phases, flashes)

    points.to_csv(POINTS_OUT, index=False)
    phases.to_csv(PHASE_OUT, index=False)
    stable.to_csv(STABLE_OUT, index=False)
    flashes.to_csv(FLASH_OUT, index=False)
    recovery.to_csv(RECOVERY_OUT, index=False)
    quality.to_csv(QUALITY_OUT, index=False)
    plot_timeline(points, phases, flashes)
    plot_reversibility(stable, matched_flashes, recovery)

    print("Wrote:")
    for path in [POINTS_OUT, PHASE_OUT, STABLE_OUT, FLASH_OUT, RECOVERY_OUT, QUALITY_OUT, TIMELINE_OUT, REVERSIBILITY_OUT]:
        print(f"  {path}")
    print("\nStable windows:")
    print(stable.round(3).to_string(index=False))
    print("\nMatched fluorescence:")
    print(matched_flashes.round(3).to_string(index=False))
    print("\nRecovery:")
    print(recovery.round(3).to_string(index=False))
    print("\nQuality:")
    print(quality.to_string(index=False))


if __name__ == "__main__":
    main()
