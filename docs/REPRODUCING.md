# Reproducing the analysis

## Environment

Install Git LFS and uv; clone and run `git lfs pull`. The sole LFS dependency is
the generated cohort points CSV; the cohort stage can regenerate it from raw logs.
Use Python 3.13.7 and the committed uv lock. No API keys or network services are
needed after dependencies and data have been installed.

```sh
uv sync --frozen --extra bayes
uv run --frozen --extra bayes python -m unittest discover -s tests -v
uv run --frozen --extra bayes python scripts/release_manifest.py --check
```

Tier B also requires R and `plantecophys`. The original local environment reports
R 4.2.2 and plantecophys 1.4.6. Python's uv lock does not lock R packages. Record
`sessionInfo()` for a rerun, and use 1.4.6 for a version-matched comparison.

```r
install.packages("plantecophys", repos="https://cloud.r-project.org")
packageVersion("plantecophys")  # a current installation may differ from 1.4.6
```

For the exact older package when necessary, install the CRAN archive source for
plantecophys 1.4.6 and its dependencies. Do not assume version equality simply
because installation succeeded.

## Execution order

The runner prints commands by default. A single selected stage does not rerun
prerequisites automatically. All scripts find data relative to their source tree.

```sh
uv run --frozen --extra bayes python scripts/reproduce.py
uv run --frozen --extra bayes python scripts/reproduce.py --stage cohort --execute
uv run --frozen --extra bayes python scripts/reproduce.py --stage tier-a --execute
uv run --frozen --extra bayes python scripts/reproduce.py --stage tier-b --execute
uv run --frozen --extra bayes python scripts/reproduce.py --stage fluorescence --execute
uv run --frozen --extra bayes python scripts/reproduce.py --stage posterior-figures --execute
```

The last command uses the saved posterior files. Refit them only intentionally:

```sh
uv run --frozen --extra bayes python scripts/reproduce.py --stage bayes --execute
uv run --frozen --extra bayes python scripts/reproduce.py --stage sensitivity --execute
```

`--stage all --skip-bayes --execute` runs the non-sampling sequence and uses the
committed posteriors for summaries. `--stage all --execute` includes both expensive
sampling runs. Outputs are overwritten: use a separate clone for reproduction.
Keep saved reference draws for comparisons rather than overwriting the only copy.

The Bayesian default is four chains, 1,500 tuning iterations, 1,000 draws per chain,
target_accept 0.93, and seeds 20260803 + chain index. Lower draw counts are only smoke
tests, not replacement scientific results. Floating-point, sampler, and library
differences can change results even with matching seeds.

## Interpretation and verification

1. Confirm raw manifest hashes and run mapping first.
2. Confirm cohort counts: 15 total, 12 retained, eight primary sun gas leaves,
   seven primary fluorescence leaves. Preserve all raw exclusions.
3. Compare regenerated numeric tables to committed tables, not merely plot appearance.
4. Inspect fit failures, parameter bounds, divergences, R-hat, ESS, and posterior
   predictive errors. A command exiting successfully does not establish a sound fit.
5. Record the exact commit, installed package versions, and R session.

The existing `photo26` and `photo26-photorch` console commands are historical
single-curve demos, not this full cohort pipeline. PhoTorch needs `--extra photorch`.
XY experiments need `--extra xy`. Presentation builders have external asset
dependencies and are excluded from the supported reproduction path.

## Publishing safely

The private full-state snapshot is not a release branch. Do not push `--all`,
`--mirror`, or `--tags`. The release worktree is based on the earlier code-only
history and excludes downloaded manuals, commercial fonts, and render scratch.

For a fresh clone, enable the supplied guard with:

```sh
git config core.hooksPath .githooks
```

The guard rejects the known private snapshot ancestry and oversized ordinary Git
blobs, then invokes Git LFS's pre-push handler. It does not replace a rights or
privacy review. Before public release, verify LFS upload/billing and test a fresh
remote clone. If using GitHub Download ZIP, enable LFS archive inclusion and check
that the large table contains data rather than a pointer. These remote operations
have not been performed by the local setup.
