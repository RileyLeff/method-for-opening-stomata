from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from photorch import fvcb

from .curves import ACI_FILE_HINT, AQ_FILE_HINT, OUTPUT_DIR, curve_frame, find_data_file


def r_squared(observed: np.ndarray, fitted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    fitted = np.asarray(fitted, dtype=float)
    total = np.sum((observed - observed.mean()) ** 2)
    if total == 0:
        return float("nan")
    return float(1 - np.sum((observed - fitted) ** 2) / total)


def photorch_frame(frame: pd.DataFrame, curve_id: int) -> pd.DataFrame:
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


def scalar_or_list(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if np.isscalar(value):
        return float(value)
    values = np.asarray(value).tolist()
    if isinstance(values, list) and len(values) == 1:
        return float(values[0])
    return values


def model_values(model: torch.nn.Module) -> tuple[dict[str, Any], np.ndarray]:
    with torch.no_grad():
        prediction, *_ = model()

    values: dict[str, Any] = {
        "Vcmax25": scalar_or_list(model.Vcmax25),
        "Jmax25": scalar_or_list(model.Jmax25),
        "TPU25": scalar_or_list(model.TPU25),
        "Vcmax_mean": float(model.Vcmax.detach().cpu().mean()),
        "Jmax_mean": float(model.Jmax.detach().cpu().mean()),
        "TPU_mean": float(model.TPU.detach().cpu().mean()),
        "Rd_mean": float(model.Rd.detach().cpu().mean()),
    }
    if hasattr(model, "Rd25"):
        values["Rd25"] = scalar_or_list(model.Rd25)
    if getattr(model, "Rdratio", None) is not None:
        values["Rdratio"] = scalar_or_list(model.Rdratio)
    if hasattr(model.LightResponse, "alpha"):
        values["alpha"] = scalar_or_list(model.LightResponse.alpha)
    if hasattr(model.LightResponse, "theta"):
        values["theta"] = scalar_or_list(model.LightResponse.theta)

    return values, prediction.detach().cpu().numpy()


def run_variant(
    name: str,
    frame: pd.DataFrame,
    *,
    lightresp_id: list[int] | None,
    light_response_type: int,
    onefit: bool,
    fit_rd: bool,
    learning_rate: float,
    iterations: int,
) -> dict[str, Any]:
    lcd = fvcb.initLicordata(
        frame,
        preprocess=True,
        lightresp_id=lightresp_id,
        smoothingwindow=5,
        printout=False,
    )
    model = fvcb.model(
        lcd,
        LightResp_type=light_response_type,
        TempResp_type=0,
        onefit=onefit,
        fitRd=fit_rd,
        printout=False,
    )
    result = fvcb.fit(
        model,
        learn_rate=learning_rate,
        maxiteration=iterations,
        minloss=0.05,
        printout=False,
    )
    parameters, prediction = model_values(result.model)

    row: dict[str, Any] = {
        "variant": name,
        "n_points": int(len(lcd.A)),
        "loss": float(result.losses[-1].detach().cpu()),
        "r2_all": r_squared(lcd.A.cpu().numpy(), prediction),
        **parameters,
    }
    for curve_id in lcd.IDs:
        indices = lcd.getIndicesbyID(curve_id)
        prefix = f"curve_{int(curve_id)}"
        observed = lcd.A[indices].cpu().numpy()
        fitted = prediction[indices]
        row[f"{prefix}_n"] = int(len(indices))
        row[f"{prefix}_r2"] = r_squared(observed, fitted)
        row[f"{prefix}_observed_max_A"] = float(np.max(observed))
        row[f"{prefix}_fitted_max_A"] = float(np.max(fitted))

    return row


def build_photorch_summary() -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)

    aci = curve_frame(find_data_file(ACI_FILE_HINT), ["A", "Ci", "Ca", "Q", "Qabs", "Tleaf"])
    aq = curve_frame(find_data_file(AQ_FILE_HINT), ["A", "Q", "Qabs", "Ci", "Ca", "Tleaf"])

    aci_photorch = photorch_frame(aci, curve_id=1)
    aq_photorch = photorch_frame(aq, curve_id=2)
    combined = pd.concat([aci_photorch, aq_photorch], ignore_index=True)

    rows = [
        run_variant(
            "aci_only_free_rd",
            aci_photorch,
            lightresp_id=None,
            light_response_type=0,
            onefit=False,
            fit_rd=True,
            learning_rate=0.05,
            iterations=12_000,
        ),
        run_variant(
            "aci_only_rd_1pct_vcmax",
            aci_photorch,
            lightresp_id=None,
            light_response_type=0,
            onefit=False,
            fit_rd=False,
            learning_rate=0.05,
            iterations=12_000,
        ),
        run_variant(
            "combined_aci_aq_rd_1pct_vcmax",
            combined,
            lightresp_id=[2],
            light_response_type=2,
            onefit=True,
            fit_rd=False,
            learning_rate=0.03,
            iterations=18_000,
        ),
    ]

    output = OUTPUT_DIR / "photorch_fit_summary.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def main() -> None:
    output = build_photorch_summary()
    print(f"Wrote {output}")
