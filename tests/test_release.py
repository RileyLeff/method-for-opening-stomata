"""Small regression checks against the archived data, without rewriting results."""

import ast
import csv
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from photo26.curves import read_licor_export
import analyze_phrag_brownsville_cohort as cohort
import analyze_phrag_brownsville_biochemical_limits as core
from analyze_phrag_brownsville_tier_b import electron_transport, fvcb_rates


class ParserTests(unittest.TestCase):
    def test_duplicate_names_and_missing_fields(self):
        text = "[Header]\n[Data]\ngroups\nA\tA\tlabel\nunits\n1\t2\tleaf\n3\t\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log"
            path.write_text(text)
            frame = read_licor_export(path)
        self.assertEqual(list(frame), ["A", "A_2", "label"])
        self.assertEqual(frame.A.tolist(), [1, 3])
        self.assertTrue(pd.isna(frame.A_2.iloc[1]))

    def test_missing_data_section(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log"
            path.write_text("Not a gas exchange export")
            with self.assertRaisesRegex(ValueError, "does not contain"):
                read_licor_export(path)


class ResearchRegressionTests(unittest.TestCase):
    def test_source_syntax(self):
        for directory in ["bp", "scripts", "src", "tests"]:
            for path in (ROOT / directory).rglob("*.py"):
                with self.subTest(path=path.name):
                    ast.parse(path.read_text())

    def test_run_index(self):
        with (ROOT / "data/run_index.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        names = [r["source_log"] for r in rows]
        actual = {p.relative_to(ROOT).as_posix() for p in (ROOT / "data").rglob("*logdata*") if p.is_file()}
        self.assertEqual(set(names), actual)
        self.assertEqual(len(names), len(set(names)))
        for row in rows:
            if row["status"] == "quarantined":
                self.assertTrue(row["qc_reason"])

    def test_all_cohort_counts(self):
        with (ROOT / "docs/release-audit/cohort-validation.csv").open() as stream:
            expected = {r["run_code"]: r for r in csv.DictReader(stream)}
        for run in cohort.RUNS:
            with self.subTest(run=run.code):
                points = core.load_points(run)
                flashes = core.load_flashes(run, points)
                self.assertEqual(len(points), int(expected[run.code]["raw_gas_rows"]))
                self.assertEqual(len(flashes), int(expected[run.code]["flash_json_records"]))

    def test_cohort_membership(self):
        qc = pd.read_csv(ROOT / "outputs/phrag_brownsville_cohort_run_qc.csv")
        self.assertEqual(set(qc.loc[qc.suspect_run, "run_code"]), {"P05", "P09", "P15"})
        retained = qc.loc[~qc.suspect_run]
        self.assertEqual(len(retained), 12)
        primary = retained.loc[retained.primary_comparison]
        self.assertEqual(primary.groupby("zone").size().to_dict(), {"High": 3, "Low": 5})
        self.assertEqual(set(cohort.RUN_BY_CODE), set(qc.run_code))

    def test_real_log_and_flash_join(self):
        run = cohort.RUN_BY_CODE["P12"]
        points = core.load_points(run)
        flashes = core.load_flashes(run, points)
        self.assertEqual(len(points), 5156)
        self.assertEqual(len(flashes), 55)
        self.assertEqual(int(flashes.gas_A.notna().sum()), 54)
        self.assertEqual(int(flashes.flash_kind.eq("dark").sum()), 1)
        self.assertTrue(points.loc[points.mpf_pulse, "mpf_or_recovery"].all())
        self.assertFalse(points.loc[points.mpf_or_recovery, "clean_ramp"].any())

    def test_saved_flash_counts(self):
        qc = pd.read_csv(ROOT / "outputs/phrag_brownsville_cohort_run_qc.csv")
        flashes = pd.read_csv(ROOT / "outputs/phrag_brownsville_cohort_flashes.csv")
        retained = flashes.loc[flashes.run_code.isin(qc.loc[~qc.suspect_run, "run_code"])]
        self.assertEqual(len(retained), 665)
        self.assertEqual(int(retained.flash_kind.eq("active").sum()), 656)

    def test_j_limits(self):
        q = np.array([0.0, 100.0, 1000.0, 1e6])
        j = electron_transport(q, 120.0, 0.3, 0.7)
        self.assertEqual(j[0], 0)
        self.assertTrue(np.all(np.diff(j) > 0))
        self.assertTrue(np.all(j <= 120))
        self.assertAlmostEqual(j[-1], 120, delta=0.1)

    def test_fvcb_dark_respiration(self):
        a, *_ = fvcb_rates(np.array([400.0]), np.array([0.0]), 70, 120, 1.2, 0.3, 0.7)
        self.assertAlmostEqual(a[0], -1.2)


if __name__ == "__main__":
    unittest.main()
