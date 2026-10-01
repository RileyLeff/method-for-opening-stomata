from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CURVES_PATH = OUTPUT_DIR / "phrag_brownsville_cohort_binned_curves.csv"
Q_SUMMARY_PATH = OUTPUT_DIR / "phrag_brownsville_cohort_q_summary.csv"
QC_PATH = OUTPUT_DIR / "phrag_brownsville_cohort_run_qc.csv"

RUN_CODE = "P10"
Q_LEVELS = np.array([50, 100, 200, 350, 600, 1000, 1400, 1700, 2000], dtype=float)
Q_COLORS = [
    "#440154",
    "#482878",
    "#3E4989",
    "#31688E",
    "#26828E",
    "#1F9E89",
    "#35B779",
    "#90D743",
    "#FDE725",
]

ACI_HTML = OUTPUT_DIR / "xy_p10_aciq_lines.html"
ACI_PNG = OUTPUT_DIR / "xy_p10_aciq_lines.png"
SURFACE_HTML = OUTPUT_DIR / "xy_p10_a_q_ci_contour.html"
SURFACE_PNG = OUTPUT_DIR / "xy_p10_a_q_ci_contour.png"
SURFACE_ART_PNG = OUTPUT_DIR / "xy_p10_photosynthesis_surface_art.png"
SURFACE_ART_SVG = OUTPUT_DIR / "xy_p10_photosynthesis_surface_art.svg"
COHORT_HTML = OUTPUT_DIR / "xy_phrag_a_gsw_by_zone.html"
COHORT_PNG = OUTPUT_DIR / "xy_phrag_a_gsw_by_zone.png"


def photo_theme() -> xy.Theme:
    return xy.theme(
        background="#FFFFFF",
        plot_background="#FAFCFA",
        grid_color="#DCE4DF",
        axis_color="#65716A",
        text_color="#17231C",
        crosshair_color="#2F7D5A",
        selection_color="#2F7D5A",
        selection_fill="rgba(47, 125, 90, 0.14)",
    )


def export_chart(chart: xy.Chart | xy.FacetChart, html_path: Path, png_path: Path) -> None:
    chart.to_html(html_path)
    chart.to_png(png_path, scale=2.0)


def build_aci_chart(curves: pd.DataFrame) -> xy.Chart:
    run = curves.loc[curves["run_code"].eq(RUN_CODE)].copy()
    line_marks: list[xy.Mark] = []
    for q_level, color in zip(Q_LEVELS, Q_COLORS, strict=True):
        group = run.loc[run["Q_level"].eq(q_level)].sort_values("Ci")
        line_marks.append(
            xy.line(
                x="Ci",
                y="A",
                data=group,
                name=f"Q {q_level:g}",
                color=color,
                width=2.4,
            )
        )

    return xy.line_chart(
        *line_marks,
        xy.hline(0, color="#6B746E", width=1.0, opacity=0.8),
        xy.x_axis(
            label="Intercellular CO2, Ci (umol mol-1)",
            domain=(0, 2000),
            bounds=(0, 2000),
            tick_values=[0, 500, 1000, 1500, 2000],
        ),
        xy.y_axis(
            label="Assimilation, A (umol CO2 m-2 s-1)",
            domain=(-2, 28),
            bounds=(-2, 28),
            tick_values=[0, 5, 10, 15, 20, 25],
        ),
        xy.legend(loc="upper left", ncols=3, title="Incident Q"),
        xy.tooltip(
            fields=["Ci", "A", "Q_level", "gsw"],
            labels={"Q_level": "Q", "gsw": "gsw"},
            format={"Ci": ".0f", "A": ".2f", "Q_level": ".0f", "gsw": ".3f"},
        ),
        xy.interaction_config(crosshair=True, navigation=True, select=True),
        xy.modebar(),
        photo_theme(),
        title="P10 Brownsville Phragmites A/Ci response across light",
        width=1200,
        height=700,
    )


def interpolated_surface(curves: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    run = curves.loc[curves["run_code"].eq(RUN_CODE)].copy()
    ci_support = run.groupby("Q_level")["Ci"].agg(["min", "max"])
    common_ci_min = float(np.ceil(ci_support["min"].max() / 10) * 10)
    common_ci_max = float(np.floor(ci_support["max"].min() / 10) * 10)
    ci_grid = np.linspace(common_ci_min, common_ci_max, 100)
    surface = np.full((len(Q_LEVELS), len(ci_grid)), np.nan)
    for row, q_level in enumerate(Q_LEVELS):
        group = (
            run.loc[run["Q_level"].eq(q_level), ["Ci", "A"]]
            .dropna()
            .sort_values("Ci")
            .drop_duplicates("Ci")
        )
        valid = (ci_grid >= group["Ci"].min()) & (ci_grid <= group["Ci"].max())
        surface[row, valid] = np.interp(
            ci_grid[valid],
            group["Ci"].to_numpy(),
            group["A"].to_numpy(),
        )
    return ci_grid, Q_LEVELS, surface


def build_surface_chart(curves: pd.DataFrame) -> xy.Chart:
    ci_grid, q_grid, assimilation = interpolated_surface(curves)
    levels = np.arange(0, 27, 2, dtype=float)
    return xy.contour_chart(
        xy.contour(
            z=assimilation,
            x=ci_grid,
            y=q_grid,
            levels=levels,
            filled=True,
            colormap="viridis",
            name="A",
            opacity=0.98,
            extend="both",
        ),
        xy.contour(
            z=assimilation,
            x=ci_grid,
            y=q_grid,
            levels=levels,
            filled=False,
            color="rgba(255, 255, 255, 0.72)",
            width=0.8,
            opacity=0.9,
        ),
        xy.legend(show=False),
        xy.x_axis(
            label="Intercellular CO2, Ci (umol mol-1)",
            domain=(float(ci_grid[0]), float(ci_grid[-1])),
            bounds=(float(ci_grid[0]), float(ci_grid[-1])),
            tick_values=[130, 500, 1000, 1400],
        ),
        xy.y_axis(
            label="Incident light, Q (umol photons m-2 s-1)",
            domain=(50, 2000),
            bounds=(50, 2000),
            tick_values=[50, 500, 1000, 1500, 2000],
        ),
        xy.colorbar(
            title="Assimilation, A (umol CO2 m-2 s-1)",
            orientation="horizontal",
            ticks=[0, 5, 10, 15, 20, 25],
        ),
        xy.interaction_config(crosshair=True, navigation=True),
        xy.modebar(),
        photo_theme(),
        title="P10 photosynthetic response surface",
        width=1200,
        height=700,
    )


def build_surface_art(curves: pd.DataFrame) -> xy.Chart:
    ci_grid, q_grid, assimilation = interpolated_surface(curves)
    return xy.contour_chart(
        xy.contour(
            z=assimilation,
            x=ci_grid,
            y=q_grid,
            levels=np.arange(0, 27, 2, dtype=float),
            filled=True,
            colormap="viridis",
            opacity=1.0,
            extend="both",
        ),
        xy.x_axis(
            show=False,
            domain=(float(ci_grid[0]), float(ci_grid[-1])),
            bounds=(float(ci_grid[0]), float(ci_grid[-1])),
        ),
        xy.y_axis(show=False, domain=(50, 2000), bounds=(50, 2000)),
        xy.legend(show=False),
        xy.colorbar(show=False),
        xy.theme(
            background="#440154",
            plot_background="#440154",
            grid_color="#440154",
            axis_color="#440154",
            text_color="#440154",
        ),
        width=1800,
        height=1100,
        padding=0,
    )


def build_cohort_chart(q_summary: pd.DataFrame, qc: pd.DataFrame) -> xy.FacetChart:
    accepted = set(qc.loc[~qc["suspect_run"], "run_code"])
    data = q_summary.loc[q_summary["run_code"].isin(accepted)].copy()
    data = data.dropna(subset=["gsw_at_Ca420", "A_at_Ca420", "Q_level", "zone"])
    data["zone"] = pd.Categorical(data["zone"], categories=["High", "Low"], ordered=True)

    return xy.facet_chart(
        xy.scatter(
            x="gsw_at_Ca420",
            y="A_at_Ca420",
            color="Q_level",
            size=7.5,
            colormap="viridis",
            color_domain=(50, 2000),
            opacity=0.86,
            stroke="#FFFFFF",
            stroke_width=0.8,
        ),
        xy.legend(show=False),
        xy.hline(0, color="#6B746E", width=1.0, opacity=0.8),
        xy.x_axis(
            label="Stomatal conductance at Ca 420, gsw (mol H2O m-2 s-1)",
            domain=(0, 0.30),
            bounds=(0, 0.30),
            tick_values=[0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
            format=".2f",
        ),
        xy.y_axis(
            label="Assimilation at Ca 420, A (umol CO2 m-2 s-1)",
            domain=(-2, 15),
            bounds=(-2, 15),
            tick_values=[0, 5, 10, 15],
        ),
        xy.colorbar(
            title="Incident Q",
            orientation="horizontal",
            ticks=[50, 500, 1000, 1500, 2000],
        ),
        xy.tooltip(
            fields=["run_code", "zone", "Q_level", "gsw_at_Ca420", "A_at_Ca420"],
            labels={
                "run_code": "Run",
                "Q_level": "Q",
                "gsw_at_Ca420": "gsw",
                "A_at_Ca420": "A",
            },
            format={"Q_level": ".0f", "gsw_at_Ca420": ".3f", "A_at_Ca420": ".2f"},
        ),
        xy.interaction_config(crosshair=True, navigation=True, select=True),
        xy.modebar(),
        photo_theme(),
        by="zone",
        data=data,
        cols=2,
        share_x=True,
        share_y=True,
        link="both",
        width=1200,
        height=500,
        gap=18,
        title="Brownsville Phragmites: carbon gain versus stomatal conductance",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    curves = pd.read_csv(CURVES_PATH)
    q_summary = pd.read_csv(Q_SUMMARY_PATH)
    qc = pd.read_csv(QC_PATH)

    products = [
        (build_aci_chart(curves), ACI_HTML, ACI_PNG),
        (build_surface_chart(curves), SURFACE_HTML, SURFACE_PNG),
        (build_cohort_chart(q_summary, qc), COHORT_HTML, COHORT_PNG),
    ]
    for chart, html_path, png_path in products:
        export_chart(chart, html_path, png_path)
        print(f"Wrote {html_path}")
        print(f"Wrote {png_path}")

    surface_art = build_surface_art(curves)
    surface_art.to_png(SURFACE_ART_PNG, scale=2.0)
    surface_art.to_svg(SURFACE_ART_SVG)
    print(f"Wrote {SURFACE_ART_PNG}")
    print(f"Wrote {SURFACE_ART_SVG}")


if __name__ == "__main__":
    main()
