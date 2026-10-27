"""Print or execute the primary analysis in dependency order; no execution by default."""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    "cohort": [("python", "analyze_phrag_brownsville_cohort.py")],
    "tier-a": [("python", "analyze_phrag_brownsville_tier_a.py")],
    "tier-b": [("Rscript", "fit_phrag_brownsville_tier_b_plantecophys.R"),
               ("python", "analyze_phrag_brownsville_tier_b.py")],
    "bayes": [("python", "analyze_phrag_brownsville_tier_c_bayesian.py")],
    "sensitivity": [("python", "analyze_phrag_brownsville_tier_c_bayesian.py", "--exclude-run", "P02", "--output-suffix", "sensitivity_no_p02")],
    "posterior-figures": [("python", "summarize_phrag_brownsville_order_standardization.py"),
                          ("python", "plot_phrag_brownsville_capacity_envelopes_literature.py")],
    "fluorescence": [("python", "plot_phrag_brownsville_npq.py"),
                     ("python", "analyze_phrag_brownsville_additive_yields.py"),
                     ("python", "test_phrag_brownsville_npq_zone_difference.py")],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", *STAGES], default="all")
    parser.add_argument("--execute", action="store_true", help="Explicitly permit overwriting generated outputs")
    parser.add_argument("--skip-bayes", action="store_true", help="Use committed posteriors for figures, do not refit")
    args = parser.parse_args()
    for name, commands in STAGES.items():
        if args.stage not in {"all", name} or (args.skip_bayes and name in {"bayes", "sensitivity"}):
            continue
        for executable, script, *extra in commands:
            cmd = [sys.executable if executable == "python" else executable, str(ROOT / "scripts" / script), *extra]
            print(f"[{name}] {shlex.join(cmd)}", flush=True)
            if args.execute:
                (ROOT / "outputs").mkdir(exist_ok=True)
                subprocess.run(cmd, cwd=ROOT, check=True)
    if not args.execute:
        print("Dry run only. Add --execute to overwrite outputs. Select --stage for one step; prerequisites must already exist.")


if __name__ == "__main__":
    main()
