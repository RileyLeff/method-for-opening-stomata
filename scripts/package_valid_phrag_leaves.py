from __future__ import annotations

import argparse
import io
from datetime import date
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
QC_PATH = OUTPUT_DIR / "phrag_brownsville_cohort_run_qc.csv"
METADATA_PATH = DATA_DIR / "brownsville_phragmites_sample_metadata_2026.csv"

PROCESSED_TABLES = [
    "phrag_brownsville_cohort_binned_curves.csv",
    "phrag_brownsville_cohort_q_summary.csv",
    "phrag_brownsville_cohort_flashes.csv",
    "phrag_brownsville_cohort_fluorescence_at_common_ci.csv",
    "phrag_brownsville_cohort_jq_parameters.csv",
    "phrag_brownsville_sun_leaf_carbon_fluorescence_coupling.csv",
]


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def find_run_directory(run_code: str) -> Path:
    matches = sorted(DATA_DIR.glob(f"*_phrag_brownsville_{run_code.lower()}_biochemical_limits"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one source directory for {run_code}, found {matches}")
    return matches[0]


def source_inventory(run_code: str) -> dict[str, object]:
    source_dir = find_run_directory(run_code)
    top_files = sorted(path for path in source_dir.iterdir() if path.is_file())
    logs = [path for path in top_files if path.suffix.lower() != ".zip"]
    flash_archives = [path for path in top_files if path.suffix.lower() == ".zip"]
    flash_dir = source_dir / "flashes"
    flash_files = sorted(path for path in flash_dir.rglob("*") if path.is_file())

    if len(logs) != 1:
        raise RuntimeError(f"Expected one LI-6800 log for {run_code}, found {logs}")
    if not flash_archives:
        raise RuntimeError(f"No original flash archive found for {run_code}")
    if not flash_files:
        raise RuntimeError(f"No extracted flash files found for {run_code}")

    return {
        "source_dir": source_dir,
        "log": logs[0],
        "flash_archives": flash_archives,
        "flash_files": flash_files,
        "flash_json_files": sum(path.suffix.lower() == ".json" for path in flash_files),
        "flash_xlsx_files": sum(path.suffix.lower() == ".xlsx" for path in flash_files),
        "flash_txt_files": sum(path.suffix.lower() == ".txt" for path in flash_files),
    }


def make_readme(valid_qc: pd.DataFrame, excluded_qc: pd.DataFrame) -> str:
    valid_codes = ", ".join(valid_qc["run_code"])
    excluded_lines = "\n".join(
        f"- {row.run_code}: {row.suspect_reason}"
        for row in excluded_qc.itertuples(index=False)
    )
    return f"""Brownsville Phragmites valid leaf data and fluorometry flashes
================================================================

Created: {date.today().isoformat()}
Valid leaves: {len(valid_qc)} ({valid_codes})

Selection rule
--------------
This package includes every cohort run for which suspect_run is False in
phrag_brownsville_cohort_run_qc.csv. It retains the shade pilot leaves and P02,
which is marked caution because it contains extra/overlapping multiphase flashes.

Excluded suspect runs
---------------------
{excluded_lines}

Contents
--------
- raw/Pxx/: untouched LI-6800 log, original MPF ZIP, and extracted flash files.
- manifest.csv: one-row-per-leaf inventory and key metadata/QC fields.
- metadata_valid.csv: sample metadata for included leaves.
- qc_valid.csv: complete cohort QC fields for included leaves.
- qc_excluded.csv: excluded leaves and reasons, for provenance.
- processed/: analysis-ready gas-exchange and fluorometry summary tables filtered
  to the included leaves.

The original MPF ZIP and extracted flash files are both retained intentionally:
the ZIP is the archival source, while the extracted JSON/XLSX/TXT files are easy
to inspect and analyze.

The 182 MB phrag_brownsville_cohort_points.csv derivative is not included because
it reproduces observations already preserved in the untouched LI-6800 logs. It
can be regenerated with scripts/analyze_phrag_brownsville_cohort.py.
"""


def build_archive(destination: Path) -> tuple[list[str], pd.DataFrame]:
    qc = pd.read_csv(QC_PATH)
    qc["run_code"] = qc["run_code"].astype(str)
    valid_qc = qc.loc[~qc["suspect_run"].astype(bool)].copy()
    excluded_qc = qc.loc[qc["suspect_run"].astype(bool)].copy()
    valid_codes = valid_qc["run_code"].tolist()

    metadata = pd.read_csv(METADATA_PATH, dtype={"plant_id": str})
    metadata.insert(0, "run_code", metadata["plant_id"].str[:3])
    valid_metadata = metadata.loc[metadata["run_code"].isin(valid_codes)].copy()

    inventories = {code: source_inventory(code) for code in valid_codes}
    manifest = valid_qc.copy()
    manifest = manifest.merge(valid_metadata, on="run_code", how="left", suffixes=("_qc", "_metadata"))
    manifest["source_directory"] = manifest["run_code"].map(
        lambda code: inventories[code]["source_dir"].name
    )
    manifest["source_log"] = manifest["run_code"].map(
        lambda code: inventories[code]["log"].name
    )
    manifest["original_flash_archives"] = manifest["run_code"].map(
        lambda code: ";".join(path.name for path in inventories[code]["flash_archives"])
    )
    for column in ("flash_json_files", "flash_xlsx_files", "flash_txt_files"):
        manifest[column] = manifest["run_code"].map(lambda code: inventories[code][column])

    root_name = destination.stem
    archived_names: list[str] = []

    def archive_name(relative: str) -> str:
        return f"{root_name}/{relative}"

    def write_bytes(zf: ZipFile, relative: str, payload: bytes) -> None:
        name = archive_name(relative)
        zf.writestr(name, payload)
        archived_names.append(name)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=6) as zf:
        write_bytes(zf, "README.txt", make_readme(valid_qc, excluded_qc).encode("utf-8"))
        write_bytes(zf, "manifest.csv", csv_bytes(manifest))
        write_bytes(zf, "metadata_valid.csv", csv_bytes(valid_metadata))
        write_bytes(zf, "qc_valid.csv", csv_bytes(valid_qc))
        write_bytes(zf, "qc_excluded.csv", csv_bytes(excluded_qc))

        for table_name in PROCESSED_TABLES:
            table_path = OUTPUT_DIR / table_name
            frame = pd.read_csv(table_path)
            if "run_code" in frame.columns:
                frame = frame.loc[frame["run_code"].isin(valid_codes)].copy()
            write_bytes(zf, f"processed/{table_name}", csv_bytes(frame))

        for code in valid_codes:
            inventory = inventories[code]
            source_dir = inventory["source_dir"]
            source_files = [inventory["log"], *inventory["flash_archives"], *inventory["flash_files"]]
            for path in source_files:
                relative_source = path.relative_to(source_dir)
                name = archive_name(f"raw/{code}/{relative_source.as_posix()}")
                zf.write(path, name)
                archived_names.append(name)

    return archived_names, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Package non-suspect Brownsville Phragmites runs.")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / f"brownsville_phragmites_valid_leaf_data_and_flashes_{date.today().isoformat()}.zip",
    )
    args = parser.parse_args()

    archived_names, manifest = build_archive(args.output.resolve())
    print(f"Created {args.output.resolve()}")
    print(f"Included leaves: {', '.join(manifest['run_code'])}")
    print(f"Archive members: {len(archived_names):,}")
    print(f"Archive size: {args.output.resolve().stat().st_size / 1024 / 1024:.1f} MiB")


if __name__ == "__main__":
    main()
