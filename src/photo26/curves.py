from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import brentq, curve_fit


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

ACI_FILE_HINT = "Acicurve"
AQ_FILE_HINT = "test1"


@dataclass(frozen=True)
class FvcbFixedParams:
    """Approximate 25 C constants for a compact C3 FvCB curve fit."""

    gamma_star: float = 42.75
    kc: float = 404.9
    ko: float = 278_400.0
    oxygen: float = 210_000.0
    alpha: float = 0.24
    theta_j: float = 0.7


@dataclass(frozen=True)
class FitResult:
    params: dict[str, float]
    r_squared: float


def _split_tsv(line: str) -> list[str]:
    parts = line.rstrip("\n").split("\t")
    while parts and parts[-1] == "":
        parts.pop()
    return parts


def _unique_column_names(names: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    unique: list[str] = []

    for index, name in enumerate(names):
        base = name.strip() or f"unnamed_{index + 1}"
        count = counts.get(base, 0)
        counts[base] = count + 1
        unique.append(base if count == 0 else f"{base}_{count + 1}")

    return unique


def read_licor_export(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        data_start = lines.index("[Data]")
    except ValueError as exc:
        raise ValueError(f"{path} does not contain a [Data] section") from exc

    names = _unique_column_names(_split_tsv(lines[data_start + 2]))
    rows = []
    for line in lines[data_start + 4 :]:
        if not line.strip():
            continue
        row = _split_tsv(line)
        if len(row) < len(names):
            row.extend([""] * (len(names) - len(row)))
        rows.append(row[: len(names)])

    frame = pd.DataFrame(rows, columns=names)
    for column in frame.columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().any():
            frame[column] = numeric

    return frame


def find_data_file(name_hint: str) -> Path:
    matches = sorted(DATA_DIR.glob(f"*{name_hint}*"))
    if not matches:
        raise FileNotFoundError(f"No data file in {DATA_DIR} matches {name_hint!r}")
    return matches[0]


def curve_frame(path: Path, required: list[str]) -> pd.DataFrame:
    frame = read_licor_export(path)
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    return frame[required].dropna(subset=["A"]).reset_index(drop=True)


def electron_transport_rate(q_abs: float, jmax: float, fixed: FvcbFixedParams) -> float:
    absorbed = fixed.alpha * q_abs
    discriminant = max((absorbed + jmax) ** 2 - 4 * fixed.theta_j * absorbed * jmax, 0)
    return ((absorbed + jmax) - np.sqrt(discriminant)) / (2 * fixed.theta_j)


def fvcb_assimilation(
    ci: np.ndarray,
    vcmax: float,
    jmax: float,
    rd: float,
    tpu: float,
    q_abs: float,
    fixed: FvcbFixedParams,
) -> np.ndarray:
    ci = np.asarray(ci, dtype=float)
    j = electron_transport_rate(q_abs, jmax, fixed)
    gamma = fixed.gamma_star
    km = fixed.kc * (1 + fixed.oxygen / fixed.ko)

    wc = vcmax * (ci - gamma) / (ci + km)
    wj = j * (ci - gamma) / (4 * ci + 8 * gamma)
    wp = np.full_like(ci, 3 * tpu)
    return np.minimum.reduce([wc, wj, wp]) - rd


def fit_aci(data: pd.DataFrame, fixed: FvcbFixedParams) -> tuple[FitResult, pd.DataFrame]:
    q_abs = float(data["Qabs"].median())

    def model(ci: np.ndarray, vcmax: float, jmax: float, rd: float, tpu: float) -> np.ndarray:
        return fvcb_assimilation(ci, vcmax, jmax, rd, tpu, q_abs, fixed)

    ci = data["Ci"].to_numpy(dtype=float)
    a = data["A"].to_numpy(dtype=float)
    params, _ = curve_fit(
        model,
        ci,
        a,
        p0=[35, 70, 1.0, 3.0],
        bounds=([1, 1, 0, 0.1], [300, 500, 10, 50]),
        maxfev=50_000,
    )

    fitted = model(ci, *params)
    r_squared = coefficient_of_determination(a, fitted)
    grid_ci = np.linspace(max(1, ci.min() * 0.7), ci.max() * 1.05, 400)
    grid = pd.DataFrame({"Ci": grid_ci, "A": model(grid_ci, *params)})

    result = FitResult(
        params={
            "Vcmax": float(params[0]),
            "Jmax": float(params[1]),
            "Rd": float(params[2]),
            "TPU": float(params[3]),
            "Qabs_median": q_abs,
        },
        r_squared=r_squared,
    )
    return result, grid


def aq_response(q: np.ndarray, amax: float, phi: float, theta: float, rd: float) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    light_term = phi * q + amax
    discriminant = np.maximum(light_term**2 - 4 * theta * phi * q * amax, 0)
    gross = (light_term - np.sqrt(discriminant)) / (2 * theta)
    return gross - rd


def fit_aq(data: pd.DataFrame) -> tuple[FitResult, pd.DataFrame, float | None]:
    q = data["Q"].to_numpy(dtype=float)
    a = data["A"].to_numpy(dtype=float)

    params, _ = curve_fit(
        aq_response,
        q,
        a,
        p0=[max(a.max() + 0.5, 1), 0.05, 0.7, max(-a.min(), 0.1)],
        bounds=([0.1, 0.001, 0.05, 0], [50, 0.3, 0.999, 10]),
        maxfev=50_000,
    )

    fitted = aq_response(q, *params)
    r_squared = coefficient_of_determination(a, fitted)
    grid_q = np.linspace(0, q.max() * 1.05, 400)
    grid = pd.DataFrame({"Q": grid_q, "A": aq_response(grid_q, *params)})

    q_comp = light_compensation_point(params, q.max() * 1.05)
    result = FitResult(
        params={
            "Amax": float(params[0]),
            "phi": float(params[1]),
            "theta": float(params[2]),
            "Rd": float(params[3]),
        },
        r_squared=r_squared,
    )
    return result, grid, q_comp


def coefficient_of_determination(observed: np.ndarray, fitted: np.ndarray) -> float:
    residual_sum = float(np.sum((observed - fitted) ** 2))
    total_sum = float(np.sum((observed - observed.mean()) ** 2))
    return 1 - residual_sum / total_sum if total_sum else float("nan")


def light_compensation_point(params: np.ndarray, q_max: float) -> float | None:
    def response(q_value: float) -> float:
        return float(aq_response(np.array([q_value]), *params)[0])

    if response(0) == 0:
        return 0.0
    if response(0) * response(q_max) > 0:
        return None
    return float(brentq(response, 0, q_max))


def add_parameter_box(ax: plt.Axes, lines: list[str], *, loc: str = "lower right") -> None:
    positions = {
        "lower right": (0.98, 0.04, "right", "bottom"),
        "upper left": (0.04, 0.96, "left", "top"),
    }
    x, y, horizontal, vertical = positions[loc]
    ax.text(
        x,
        y,
        "\n".join(lines),
        transform=ax.transAxes,
        ha=horizontal,
        va=vertical,
        fontsize=9,
        linespacing=1.25,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "0.55",
            "alpha": 0.92,
        },
    )


def plot_aci(data: pd.DataFrame, fit: FitResult, grid: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.6))
    sns.scatterplot(data=data, x="Ci", y="A", ax=ax, s=80, color="#1f77b4", label="Observed")
    sns.lineplot(data=grid, x="Ci", y="A", ax=ax, linewidth=2.4, color="#d62728", label="FvCB fit")

    ax.axhline(0, color="0.35", linewidth=1, linestyle=":")
    ax.set_title("A/Ci Response")
    ax.set_xlabel("Intercellular CO$_2$, Ci (umol mol$^{-1}$)")
    ax.set_ylabel("Net assimilation, A (umol CO$_2$ m$^{-2}$ s$^{-1}$)")
    ax.legend(frameon=False, loc="upper left")

    p = fit.params
    add_parameter_box(
        ax,
        [
            "Fitted FvCB parameters",
            f"Vcmax = {p['Vcmax']:.2f}",
            f"Jmax = {p['Jmax']:.2f}",
            f"TPU = {p['TPU']:.2f}",
            f"Rd = {p['Rd']:.2f}",
            f"R2 = {fit.r_squared:.3f}",
            "",
            "Fixed constants",
            "Gamma* = 42.75",
            "Kc = 404.9, Ko = 278400",
            f"Qabs median = {p['Qabs_median']:.1f}",
        ],
        loc="lower right",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_aq(
    data: pd.DataFrame,
    fit: FitResult,
    grid: pd.DataFrame,
    q_comp: float | None,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.6))
    sns.scatterplot(data=data, x="Q", y="A", ax=ax, s=80, color="#2ca02c", label="Observed")
    sns.lineplot(data=grid, x="Q", y="A", ax=ax, linewidth=2.4, color="#9467bd", label="Hyperbola fit")

    ax.axhline(0, color="0.35", linewidth=1, linestyle=":")
    if q_comp is not None:
        ax.axvline(q_comp, color="0.45", linewidth=1, linestyle="--")
        ax.annotate(
            f"Qcomp = {q_comp:.1f}",
            xy=(q_comp, 0),
            xytext=(8, 12),
            textcoords="offset points",
            fontsize=9,
            color="0.25",
        )

    ax.set_title("A/Q Response")
    ax.set_xlabel("Incident light, Q (umol photons m$^{-2}$ s$^{-1}$)")
    ax.set_ylabel("Net assimilation, A (umol CO$_2$ m$^{-2}$ s$^{-1}$)")
    ax.legend(frameon=False, loc="lower right", fontsize=10)

    p = fit.params
    lines = [
        "Fitted light-response parameters",
        f"Amax = {p['Amax']:.2f}",
        f"phi = {p['phi']:.4f}",
        f"theta = {p['theta']:.3f}",
        f"Rd = {p['Rd']:.2f}",
        f"R2 = {fit.r_squared:.3f}",
    ]
    if q_comp is not None:
        lines.append(f"Qcomp = {q_comp:.1f}")
    add_parameter_box(ax, lines, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def build_curves() -> tuple[Path, Path]:
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.15)
    OUTPUT_DIR.mkdir(exist_ok=True)

    aci_path = find_data_file(ACI_FILE_HINT)
    aq_path = find_data_file(AQ_FILE_HINT)

    aci_data = curve_frame(aci_path, ["A", "Ci", "Ca", "Q", "Qabs", "Tleaf"])
    aq_data = curve_frame(aq_path, ["A", "Q", "Qabs", "Ci", "Ca", "Tleaf"])

    fixed = FvcbFixedParams()
    aci_fit, aci_grid = fit_aci(aci_data, fixed)
    aq_fit, aq_grid, q_comp = fit_aq(aq_data)

    aci_output = OUTPUT_DIR / "aci_curve.png"
    aq_output = OUTPUT_DIR / "aq_curve.png"
    plot_aci(aci_data, aci_fit, aci_grid, aci_output)
    plot_aq(aq_data, aq_fit, aq_grid, q_comp, aq_output)
    return aci_output, aq_output


def main() -> None:
    aci_output, aq_output = build_curves()
    print(f"Wrote {aci_output}")
    print(f"Wrote {aq_output}")
