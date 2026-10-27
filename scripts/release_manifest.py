"""Write or verify release checksums and an instrument-log index (standard library only)."""

import argparse
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def files(area):
    for path in sorted((ROOT / area).rglob("*")):
        if not path.is_file() or path.name in {"README.md", "manifest.csv", "run_index.csv", ".DS_Store"}:
            continue
        if {"__pycache__", "harlan_2026", "student_presentation_slide6"}.intersection(path.parts):
            continue
        if area == "outputs" and path.parent == ROOT / area and path.suffix == ".zip":
            continue
        yield path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_table(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_index():
    with (ROOT / "outputs/phrag_brownsville_cohort_run_qc.csv").open() as stream:
        qc = {r["run_code"]: r for r in csv.DictReader(stream)}
    with (ROOT / "docs/release-audit/cohort-validation.csv").open() as stream:
        cohort = {r["source_log"]: r["run_code"] for r in csv.DictReader(stream)}
    rows = []
    for path in files("data"):
        if "logdata" not in path.name:
            continue
        header = path.read_text(encoding="utf-8").split("[Data]", 1)[0]
        fields = dict(line.split("\t", 1) for line in header.splitlines() if "\t" in line)
        rel = path.relative_to(ROOT).as_posix()
        code = cohort.get(rel, "")
        quality = qc.get(code, {})
        status = "method_development"
        if code:
            status = "quarantined" if quality["suspect_run"] == "True" else (
                "primary_sun" if quality["primary_comparison"] == "True" else "pilot_shade")
        species = "Morella cerifera" if any(s in rel.lower() for s in ("morcer", "morella")) else (
            "Phragmites australis" if "phrag" in rel.lower() or "pharg" in rel.lower() or "phra_" in rel.lower() else "unknown")
        rows.append({
            "source_log": rel, "file_opened": fields.get("File opened", ""),
            "species_from_filename": species, "run_code": code,
            "plant_id": quality.get("plant_id", ""), "elevation_zone": quality.get("zone", ""),
            "canopy_light": quality.get("illum", ""), "status": status,
            "qc_reason": quality.get("suspect_reason", ""),
            "primary_fluorescence": bool(code and status == "primary_sun" and code != "P02"),
            "dark_reference_source": quality.get("dark_FvFm_source", ""),
            "console_firmware": fields.get("Console ver", ""),
            "instrument_scripts": fields.get("Scripts ver", ""),
            "head_version": fields.get("Head ver", ""),
            "fluorometer_version": fields.get("Flr. Version", ""),
            "chamber": fields.get("Chamber type", ""),
            "program_association": "biochemical_limits_flow600 (inferred; no instrument source hash)" if code else "not established",
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    parser.add_argument("--raw-only", action="store_true", help="Check raw data without requiring the LFS derivative")
    args = parser.parse_args()
    errors = []
    for area in (["data"] if args.raw_only else ["data", "outputs"]):
        actual = [{"path": p.relative_to(ROOT).as_posix(), "bytes": str(p.stat().st_size), "sha256": digest(p)} for p in files(area)]
        manifest = ROOT / area / "manifest.csv"
        if args.write:
            write_table(manifest, actual)
        else:
            with manifest.open() as stream:
                expected = {r["path"]: r for r in csv.DictReader(stream)}
            observed = {r["path"]: r for r in actual}
            for name in sorted(expected.keys() | observed.keys()):
                if expected.get(name) != observed.get(name):
                    errors.append(name)
        print(f"{area}: {len(actual)} files inspected")
    if args.write:
        write_table(ROOT / "data/run_index.csv", run_index())
    else:
        with (ROOT / "data/run_index.csv").open() as stream:
            existing = list(csv.DictReader(stream))
        calculated = [{k: str(v) for k, v in row.items()} for row in run_index()]
        if existing != calculated:
            errors.append("data/run_index.csv (metadata drift)")
    if errors:
        raise SystemExit("Manifest mismatch (LFS pointer or regenerated output may be the cause):\n" + "\n".join(errors))
    print("Manifests written" if args.write else "All requested checksums and run-index fields match")


if __name__ == "__main__":
    main()
