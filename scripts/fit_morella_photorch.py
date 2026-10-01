from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from photorch import fvcb

from photo26.curves import read_licor_export


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"

SURFACE_RUNS = {
    "july9": OUTPUT_DIR / "morella_aqc_surface_points.csv",
    "current": OUTPUT_DIR / "morella_outside_regular_points.csv",
}
JUNE_ACI = DATA_DIR / "2026-06-15-1053_logdata_morcertest2_Acicurve (1)"
JUNE_AQ = DATA_DIR / "2026-06-15-1020_logdata_morcer_test1"


def surface_fit_frame(path: Path) -> pd.DataFrame:
    points = pd.read_csv(path, low_memory=False)
    data = points.loc[
        points["sane_for_curve"].astype(bool)
        & points["Q_level"].gt(0)
        & points["Ci"].between(60, 1900)
    ].copy()
    data["ci_bin"] = (data["Ci"] / 60).round() * 60
    binned = (
        data.groupby(["Q_level", "ci_bin"], as_index=False)
        .agg(
            A=("A", "median"),
            Ci=("Ci", "median"),
            Qin=("Q", "median"),
            Tleaf=("Tleaf", "median"),
            n=("A", "size"),
        )
    )
    binned = binned.loc[binned["n"].ge(3)].copy()
    binned["CurveID"] = binned["Q_level"].astype(int)
    binned["FittingGroup"] = "Morella"
    return binned


def june_fit_frame() -> pd.DataFrame:
    aci = read_licor_export(JUNE_ACI)
    aq = read_licor_export(JUNE_AQ)

    def format_curve(frame: pd.DataFrame, curve_id: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "CurveID": curve_id,
                "FittingGroup": "Morella",
                "Ci": frame["Ci"],
                "A": frame["A"],
                "Qin": frame["Q"],
                "Tleaf": frame["Tleaf"],
            }
        )

    return pd.concat([format_curve(aci, 1), format_curve(aq, 2)], ignore_index=True)


def tensor_scalar(value: torch.Tensor | float) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return float(np.asarray(value).ravel()[0])


def coefficient_of_determination(observed: torch.Tensor, fitted: torch.Tensor) -> float:
    residual = torch.sum((observed - fitted) ** 2)
    total = torch.sum((observed - observed.mean()) ** 2)
    return 1 - float(residual) / float(total)


def run_fit(run: str, oxygen_sensitive: bool) -> tuple[Path, Path]:
    if run == "june":
        frame = june_fit_frame()
        light_response_ids = [2]
        preprocess = True
    else:
        frame = surface_fit_frame(SURFACE_RUNS[run])
        light_response_ids = None
        preprocess = False

    lcd = fvcb.initLicordata(
        frame,
        preprocess=preprocess,
        lightresp_id=light_response_ids,
        smoothingwindow=5,
        printout=False,
    )
    torch.manual_seed(0)
    model = fvcb.model(
        lcd,
        LightResp_type=2,
        TempResp_type=1,
        onefit=True,
        fitag=oxygen_sensitive,
        fitRd=False,
        printout=False,
    )

    # Retain PhoTorch's published activation energies while estimating capacities at 25 C.
    for parameter in ["dHa_Vcmax", "dHa_Jmax", "dHa_TPU"]:
        getattr(model.TempResponse, parameter).requires_grad_(False)

    result = fvcb.fit(
        model,
        learn_rate=0.02,
        maxiteration=15_000 if run != "june" else 18_000,
        minloss=0.01,
        fitcorr=False,
        ApCithreshold=600,
        printout=False,
    )
    fitted_model = result.model
    with torch.no_grad():
        prediction, ac, aj, ap = fitted_model()

    limitation = torch.argmin(torch.stack((ac, aj, ap)), dim=0).cpu().numpy()
    variant = "oxygen_sensitive_tpu" if oxygen_sensitive else "primary"
    summary_path = OUTPUT_DIR / f"morella_photorch_{run}_{variant}_summary.csv"
    prediction_path = OUTPUT_DIR / f"morella_photorch_{run}_{variant}_predictions.csv"

    summary = pd.DataFrame(
        [
            {
                "run": run,
                "variant": variant,
                "n_points": len(lcd.A),
                "Vcmax25": tensor_scalar(fitted_model.Vcmax25),
                "Jmax25": tensor_scalar(fitted_model.Jmax25),
                "TPU25": tensor_scalar(fitted_model.TPU25),
                "Rd_fraction_of_Vcmax": tensor_scalar(fitted_model.Rdratio),
                "Rd_mean_at_measurement_T": float(fitted_model.Rd.mean()),
                "alpha_incident": tensor_scalar(fitted_model.LightResponse.alpha),
                "theta": tensor_scalar(fitted_model.LightResponse.theta),
                "alphaG": tensor_scalar(fitted_model.alphaG),
                "rmse": float(torch.sqrt(torch.mean((prediction - lcd.A) ** 2))),
                "r_squared": coefficient_of_determination(lcd.A, prediction),
                "rubisco_limited_n": int(np.sum(limitation == 0)),
                "electron_limited_n": int(np.sum(limitation == 1)),
                "tpu_limited_n": int(np.sum(limitation == 2)),
                "Tleaf_min": float(lcd.Tleaf.min() - 273.15),
                "Tleaf_median": float(lcd.Tleaf.median() - 273.15),
                "Tleaf_max": float(lcd.Tleaf.max() - 273.15),
            }
        ]
    )
    summary.to_csv(summary_path, index=False)

    prediction_frames: list[pd.DataFrame] = []
    for curve_id in lcd.IDs:
        indices = lcd.getIndicesbyID(curve_id)
        prediction_frames.append(
            pd.DataFrame(
                {
                    "CurveID": curve_id,
                    "Ci": lcd.Ci[indices].cpu().numpy(),
                    "Q": lcd.Q[indices].cpu().numpy(),
                    "Tleaf": lcd.Tleaf[indices].cpu().numpy() - 273.15,
                    "A_observed": lcd.A[indices].cpu().numpy(),
                    "A_model": prediction[indices].cpu().numpy(),
                    "Ac": ac[indices].cpu().numpy(),
                    "Aj": aj[indices].cpu().numpy(),
                    "Ap": ap[indices].cpu().numpy(),
                    "limitation": np.array(["Ac", "Aj", "Ap"])[limitation[indices]],
                }
            )
        )
    pd.concat(prediction_frames, ignore_index=True).to_csv(prediction_path, index=False)

    print(summary.round(4).to_string(index=False))
    print(f"Wrote {summary_path}")
    print(f"Wrote {prediction_path}")
    return summary_path, prediction_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=["july9", "current", "june"], required=True)
    parser.add_argument("--oxygen-sensitive", action="store_true")
    args = parser.parse_args()
    if args.run == "june" and args.oxygen_sensitive:
        parser.error("The oxygen-sensitive diagnostic is only used for the July surfaces.")
    run_fit(args.run, args.oxygen_sensitive)


if __name__ == "__main__":
    main()
