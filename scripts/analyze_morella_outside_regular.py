from pathlib import Path

import analyze_phrag_bucket_highq as analysis


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "2026-07-10_morella_outside_regular_aqc"
PREFIX = "morella_outside_regular"

analysis.DATASET_DIR = DATASET_DIR
analysis.LOG_PATH = DATASET_DIR / "2026-07-10-0924_logdata_morcer_aq_outside"
analysis.FLASH_DIR = DATASET_DIR / "flashes"
analysis.RUN_LABEL = "Morella cerifera, Outside Regular A/Q/C"
analysis.VPD_SANE_RANGE = (0.55, 2.25)

analysis.POINTS_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_points.csv"
analysis.FLASH_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_flash_summary.csv"
analysis.JOINED_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_points_with_flash.csv"
analysis.PHASE_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_phase_summary.csv"
analysis.FLASH_JOIN_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_flash_join_report.csv"

analysis.TIMELINE_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_timeline.png"
analysis.STOMATAL_TS_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_stomatal_time_series.png"
analysis.RAMP_QUALITY_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_ramp_quality.png"
analysis.ACI_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_aci_by_q.png"
analysis.SURFACE_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_a_surface_3d.png"
analysis.AQC_SURFACE_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_aqc_surface_3d.png"
analysis.FLUORO_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_fluorescence.png"
ENV_STOMATAL_OUT = analysis.OUTPUT_DIR / f"{PREFIX}_vpd_q_ci_ca_gsw_time_series.png"


if __name__ == "__main__":
    analysis.main()
    analysis.plot_vpd_q_ci_ca_gsw_time_series(analysis.load_points(), ENV_STOMATAL_OUT)
    print(f"  {ENV_STOMATAL_OUT}")
