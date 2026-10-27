"""Inventory local release candidates without modifying scientific inputs/outputs."""

from __future__ import annotations

import ast
import argparse
import csv
import hashlib
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "docs" / "release-audit"
TREES = ("bp", "data", "scripts", "src", "outputs", "output", "references", "tmp", "planning")


def destination(path: Path) -> str:
    top = path.parts[0]
    if top in {"references", "tmp"}:
        return "local-only; third-party rights or temporary assets"
    if top == "data":
        return "Zenodo raw data; metadata also in GitHub"
    if top in {"outputs", "output"}:
        return "review derivative; selected results in Zenodo"
    if top == "bp":
        return "GitHub; distinguish final program from historical variants"
    if path.name.startswith(("build_", "poster_", "redraw_", "try_xy")):
        return "optional presentation tooling; portability review"
    return "GitHub candidate"


def write_csv(name: str, rows: list[dict]) -> None:
    with (DEST / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-cohort", action="store_true")
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    tracked = set(subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines())
    paths = [p for tree in TREES for p in (ROOT / tree).rglob("*") if p.is_file()]
    paths += [ROOT / name for name in ("README.md", "pyproject.toml", "uv.lock", ".gitignore", ".python-version")]
    rows = []
    sources = []
    for path in sorted(paths):
        relative = path.relative_to(ROOT)
        if "__pycache__" in relative.parts or path.name == ".DS_Store":
            continue
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        rows.append({"path": relative.as_posix(), "bytes": path.stat().st_size,
                     "sha256": digest, "tracked": relative.as_posix() in tracked,
                     "release_destination": destination(relative)})
        if path.suffix == ".py":
            imports = set()
            try:
                tree = ast.parse(path.read_text())
                status = "syntax OK; not executed"
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imports.update(alias.name for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        imports.add(node.module)
            except (SyntaxError, UnicodeError) as exc:
                status = str(exc)
            sources.append({"path": relative.as_posix(), "status": status,
                            "imports": ";".join(sorted(imports)),
                            "release_destination": destination(relative)})
    write_csv("file-inventory.csv", rows)
    write_csv("python-imports.csv", sources)
    print(f"Inventoried {len(rows)} files and {len(sources)} Python sources in {DEST}")
    if args.validate_cohort:
        import analyze_phrag_brownsville_cohort as cohort
        import analyze_phrag_brownsville_biochemical_limits as core

        validation = []
        for run in cohort.RUNS:
            points = core.load_points(run)
            flashes = core.load_flashes(run, points)
            validation.append({
                "run_code": run.code, "source_log": str((run.dataset_dir / run.log_name).relative_to(ROOT)),
                "raw_gas_rows": len(points), "flash_json_records": len(flashes),
                "active_flash_records": int(flashes.flash_kind.eq("active").sum()),
                "dark_flash_json_records": int(flashes.flash_kind.eq("dark").sum()),
                "active_flashes_with_gas_A": int(flashes.gas_A.notna().sum()),
                "core_clean_rows_before_cohort_filter": int(points.clean_ramp.sum()),
            })
        write_csv("cohort-validation.csv", validation)
        print(f"Parsed all {len(validation)} cohort runs; scientific output files unchanged")


if __name__ == "__main__":
    main()
