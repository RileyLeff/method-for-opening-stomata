from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from photo26.control_plots import co2_stages, load_control


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTROL_PATH = PROJECT_ROOT / "data" / "2026-06-21-0833_logdata_morcer_sun_control"
CURVES_PATH = PROJECT_ROOT / "outputs" / "phrag_brownsville_cohort_binned_curves.csv"
FITS_PATH = PROJECT_ROOT / "outputs" / "phrag_brownsville_tier_b_joint_fvcb_fits.csv"


def finite(value: object, digits: int = 6) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits)


def export_low_co2(output_dir: Path) -> None:
    frame = load_control(CONTROL_PATH).sort_values("minute").reset_index(drop=True)
    stage_frame = co2_stages(frame)
    stages: list[dict[str, float | int | None]] = []
    stage_vpd_medians: list[float] = []

    for stage in stage_frame.itertuples(index=False):
        part = frame.loc[frame["minute"].between(stage.start, stage.end, inclusive="left")]
        stage_vpd_medians.append(float(part["VPDleaf"].median()))
        stages.append(
            {
                "start": finite(stage.start),
                "end": finite(stage.end),
                "co2": int(stage.co2),
                "tailGsw": finite(part["gsw"].tail(5).mean()),
                "tailA": finite(part["A"].tail(5).mean()),
            }
        )

    records = []
    for row in frame.itertuples(index=False):
        records.append(
            {
                "minute": finite(row.minute),
                "a": finite(row.A),
                "gsw": finite(row.gsw),
                "ca": finite(row.Ca),
                "ci": finite(row.Ci),
                "co2Reference": finite(row.CO2_r),
                "co2Sample": finite(row.CO2_s),
                "q": finite(row.Q),
                "vpdLeaf": finite(row.VPDleaf),
                "tLeaf": finite(row.Tleaf),
            }
        )

    opening = 100 * (stages[2]["tailGsw"] / stages[0]["tailGsw"] - 1)
    payload = {
        "species": "Morella cerifera",
        "treatment": "Low-CO2 stomatal-opening test",
        "records": records,
        "stages": stages,
        "summary": {
            "gswIncreasePercent": finite(opening, 1),
            "qMedian": finite(frame["Q"].median(), 1),
            "vpdLeafStageMedianMin": finite(min(stage_vpd_medians), 2),
            "vpdLeafStageMedianMax": finite(max(stage_vpd_medians), 2),
        },
        "units": {
            "minute": "min",
            "a": "umol CO2 m-2 s-1",
            "gsw": "mol H2O m-2 s-1",
            "ca": "umol mol-1",
            "ci": "umol mol-1",
            "q": "umol photons m-2 s-1",
            "vpdLeaf": "kPa",
            "tLeaf": "deg C",
        },
    }
    (output_dir / "low-co2-opening.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def export_surface(output_dir: Path, run_code: str = "P10") -> None:
    curves = pd.read_csv(CURVES_PATH)
    run = curves.loc[curves["run_code"].eq(run_code)].copy()
    q_levels = np.sort(run["Q_level"].dropna().unique().astype(float))

    ci_support = run.groupby("Q_level")["Ci"].agg(["min", "max"])
    common_ci_min = float(np.ceil(ci_support["min"].max() / 10) * 10)
    common_ci_max = float(np.floor(ci_support["max"].min() / 10) * 10)
    ci_grid = np.linspace(common_ci_min, common_ci_max, 100)
    assimilation = np.full((len(q_levels), len(ci_grid)), np.nan)

    for row_index, q_level in enumerate(q_levels):
        group = (
            run.loc[run["Q_level"].eq(q_level), ["Ci", "A"]]
            .dropna()
            .sort_values("Ci")
            .drop_duplicates("Ci")
        )
        valid = (ci_grid >= group["Ci"].min()) & (ci_grid <= group["Ci"].max())
        assimilation[row_index, valid] = np.interp(
            ci_grid[valid],
            group["Ci"].to_numpy(),
            group["A"].to_numpy(),
        )

    fit_rows = pd.read_csv(FITS_PATH)
    fit = fit_rows.loc[
        fit_rows["run_code"].eq(run_code)
        & fit_rows["variant"].eq("joint_appCi_fluorescence_shape")
    ].iloc[0]

    points = []
    for row in run.sort_values(["Q_level", "Ci"]).itertuples(index=False):
        points.append(
            {
                "ci": finite(row.Ci),
                "q": finite(row.Q_level),
                "a": finite(row.A),
                "gsw": finite(row.gsw),
            }
        )

    payload = {
        "species": "Phragmites australis",
        "runCode": run_code,
        "site": "Brownsville Preserve, low elevation",
        "leafContext": "sun leaf",
        "ci": [finite(value) for value in ci_grid],
        "q": [finite(value) for value in q_levels],
        "assimilation": [
            [finite(value) for value in row]
            for row in assimilation
        ],
        "points": points,
        "fit": {
            "variant": str(fit["variant"]),
            "vcmax25": finite(fit["Vcmax25"], 2),
            "jmax25": finite(fit["Jmax25"], 2),
            "rd": finite(fit["Rd"], 2),
            "alphaJ": finite(fit["alpha_J"], 3),
            "thetaJ": finite(fit["theta_J"], 3),
            "rmse": finite(fit["RMSE"], 2),
            "r2": finite(fit["R2"], 3),
        },
        "units": {
            "ci": "umol mol-1",
            "q": "umol photons m-2 s-1",
            "a": "umol CO2 m-2 s-1",
            "gsw": "mol H2O m-2 s-1",
            "vcmax25": "umol CO2 m-2 s-1",
            "jmax25": "umol electrons m-2 s-1",
        },
    }
    (output_dir / "phrag-p10-surface.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_low_co2(args.output_dir)
    export_surface(args.output_dir)


if __name__ == "__main__":
    main()
