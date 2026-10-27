# GitHub and research-release audit

Historical pre-organization audit. For the current layout and resolved items,
see ../ORGANIZATION.md and ../VALIDATION.md. The CSV inventory here includes
private working assets that are deliberately absent from the release.

Audit date: October 27, 2026. Deadline: October 30, 2026.
Scope: this local photo26 repository, its configured GitHub repository, and local
research artifacts. No files were published, staged, or committed by this audit.
This is a release inventory, not a full scientific code review or a new analysis.

## Bottom line

The scientific materials exist. The repository is not yet a usable public release.
The work needed is curation, documentation, explicit licensing, and a clean-clone
reproduction check, not new experiments or new models.

GitHub: https://github.com/RileyLeff/photo26 is PRIVATE, with size 0 and no remote
refs returned by `git ls-remote upstream`. The local main branch has 10 commits,
ending at `0f788d4`. The remote is named `upstream`, not `origin`.
GitHub has no license, description, or populated default branch content.

Before this audit, tracked files were the Python package, 44 analysis/presentation
scripts (42 Python, two R), pyproject.toml, uv.lock, README, and minimal gitignore.
All of bp/, data/, outputs/, output/, references/, and tmp/ were untracked.
The README documents the original two-curve prototype, not the final experiment.

## Inventory

Counts exclude .DS_Store and __pycache__; MiB is logical file size, not disk usage.
These are pre-audit counts; the audit adds its own script and documentation.

| Area | Files | MiB | Contents | Release destination |
| --- | ---: | ---: | --- | --- |
| bp | 11 | 0.07 | Eight instrument programs, three setup guides | GitHub, with final/historical status |
| src | 4 | 0.03 | LI-6800 parser, original fits, plotting, PhoTorch adapter | GitHub |
| scripts | 44 | 0.66 | Ingestion, QC, fitting, statistical tests, figures, packaging | GitHub, classified by purpose |
| data | 3,757 | 344.09 | Original logs, flash ZIPs and extracted events, sample metadata | Zenodo; metadata/manifests also GitHub |
| outputs | 419 | 733.44 | Tables, figures, posterior draws, archives, presentation assets | Selected derivatives in Zenodo |
| output | 14 | 276.25 | Submission/poster PDFs and print images | Optional communication archive |
| references | 25 | 134.14 | Downloaded LI-COR documentation and a paper | Link to originals; do not blanket-license |
| tmp | 580 | 112.22 | Fonts, render intermediates, downloaded assets | Local only |

The detailed file inventory contains SHA-256, byte size, tracked status, and a
provisional destination for every inventoried file. Destination is a triage label,
not a redistribution-rights determination. It intentionally omits .git, .venv,
caches, and the inventory's own output directory.

## Instrument programs

**Primary release candidate:**
`bp/aciq_dat_biochemical_limits_fixed_h2or_flow600.py`, with its matching SETUP.md.
This is the nine-Q ladder used by the final Brownsville cohort.

- External dark adaptation >=30 minutes; five-minute in-chamber equilibration.
- One minute dark gas exchange and a dark flash.
- Opening/induction at Q=600, CO2_r=120 for 18 minutes.
- Measured Q: 50, 100, 200, 350, 600, 1000, 1400, 1700, 2000.
- CO2_r ramps: 120 to 2000 ppm at 200 ppm/min.
- Flow 600, matching DAT calibration required; Tleaf target 25 C.
- H2O_r fixed at 21 mmol/mol. This is NOT a guarantee of VPDleaf=1 kPa.
- First 60 ramp seconds flash-free; asynchronous MPFs every 90 seconds.
- Checks that logging is active and observation count increments; closes log on normal completion.
- Nominal runtime about 2 h 53 min; actual run times belong in metadata.

**Historical stable predecessor:** `aciq_dat_fixed_h2or_async_mpf_flow600.py`
and SETUP.md. Different Q ladder, dark schedule, and 60-second flash interval;
do not offer both as indistinguishable current programs.

**Useful diagnostic:** `h2o_flash_actuator_diagnostic.py` and SETUP.md.

**Experimental/history:** `test1.py`, `qac_dat_test2.py`,
`aciq_dat_bucket_highq_test3.py`, `aciq_dat_bucket_highq_vpd1_flow600.py`,
`morella_tpu_step_test.py`. Preserve with context, not as recommended defaults.

Release checks still needed: identify tested LI-6800 firmware and chamber setup;
document asynchronous action-topic dependence; validate operator inputs; document
abort/recovery behavior. Syntax checks do not establish instrument compatibility.
The instrument-supplied bpdefs module is not a desktop Python dependency.

## Data and cohort accounting

There are 30 dated data directories plus eight early standalone logs and one
sample metadata CSV. The final cohort is P01-P15, July 24-31.

- All 15: 78,576 gas-exchange rows and 869 flash JSON records.
- Retained 12: 62,326 gas-exchange rows and 665 flash JSON records.
- Primary sun-leaf gas comparison: eight leaves, not twelve.
- High elevation / lower salinity: P12, P13, P14 (three sun leaves).
- Low elevation / higher salinity: P02, P03, P06, P10, P11 (five sun leaves).
- Retained shade pilots: P01, P04, P07, P08, all high elevation.
- Primary fluorescence comparison drops P02: three versus four leaves.
- P05 excluded: overlapping MPFs. P09 excluded: CO2 exhaustion after Q200.
- P15 quarantined: progressive loss of A and gsw with increasing Q despite stable
  controls. Preserve that judgment and raw data; do not erase an inconvenient run.

Counts are raw gas rows and distinct JSON records, not independent replicates,
not filtered/binned curve points, and not JSON+XLSX counted twice. The retained
665 include 656 active records and nine dark records. Extra MPFs remain in raw data.

The stored `zone` values High and Low describe ELEVATION, not salinity. Make that
mapping explicit everywhere. Habitat light availability also differs. The CSV
does not establish genotype, measured per-leaf salinity, harvest times, or
genetically independent plants; do not infer those fields from sample IDs.

P03/P06/P07/P09/P15 have no standalone dark-flash JSON in their run directories;
the existing QC uses log-summary Fv/Fm instead. Some have DarkAdapted TXT files.
Document dark-reference provenance rather than describing every run as having
an archived dark flash waveform. Verify per-leaf dark IDs before final release.

Existing 49.59 MiB valid-leaf ZIP has 1,373 members and passes CRC verification.
It includes raw logs, original flash archives, extracted events, metadata, QC,
and selected processed tables. It excludes raw P05/P09/P15, retaining only their
QC reasons. Keep this convenience subset, but archive excluded raw runs separately
for a complete release. Its `raw/Pxx/` layout differs from the paths expected by
the analysis scripts; supply a matching-layout archive or an explicit import step.

Early Morella source material is present: June 15 A/Q and A/Ci logs, June 21
low-CO2 opening run, June 22 blue-light and indoor runs, and the overnight run.
These belong in the method-development archive. The filename "control" does
not establish a simultaneous, randomized no-intervention control.

## Analysis map

The import inventory lists every Python source and its imports. Main execution
order, inferred from code (not yet a fresh end-to-end reproduction):

1. `analyze_phrag_brownsville_cohort.py`: raw logs + flash JSON + metadata ->
   points, flash alignment, QC, binned curves, fluorescence interpolation, J(Q).
   Imports `analyze_phrag_brownsville_biochemical_limits.py` and photo26.curves.
2. `analyze_phrag_brownsville_tier_a.py`: empirical response metrics/contrasts.
3. `fit_phrag_brownsville_tier_b_plantecophys.R`: Q2000 reference fits.
4. `analyze_phrag_brownsville_tier_b.py`: multi-Q apparent-Ci FvCB variants,
   TPU comparison, fluorescence-shape comparison, exploratory variable-J gm.
   Requires the R result from step 3.
5. `analyze_phrag_brownsville_tier_c_bayesian.py`: joint hierarchical model;
   default four chains, 1,500 tune and 1,000 draws. Repeat with
   `--exclude-run P02 --output-suffix sensitivity_no_p02` for the sensitivity run.
6. `summarize_phrag_brownsville_order_standardization.py` and
   `plot_phrag_brownsville_capacity_envelopes_literature.py`: posterior summaries.
7. `plot_phrag_brownsville_npq.py` and
   `analyze_phrag_brownsville_additive_yields.py`: fluorescence/Q/Ci figures and yields.
8. `test_phrag_brownsville_npq_zone_difference.py`: leaf-level statistical tests;
   this is an analysis script, NOT a software test suite. Requires both step-7 outputs.
9. `package_valid_phrag_leaves.py`: filtered archive (currently incomplete for full provenance).

Other scripts document single-run diagnostics, earlier Morella/Phrag fits,
PhoTorch comparisons, and publication figures. Retain as historical analyses,
clearly separated from the reproducible primary path.

`make_morella_poster_figures.py` and `make_morella_slide6_figure.py` create the
stomatal-opening story figures. `export_blog_interactives.py` exports JSON for
the blog from local inputs. PDF/poster builders use external blog assets and
commercial/local fonts, so are not currently portable release entry points.

Existing tier reports, primary/sensitivity posterior NetCDFs, parameter tables,
NPQ tables, and figure bundles are present. Preserve them with provenance before
regenerating. No new Bayesian fits were run during this audit.

## Scientific documentation that must accompany the code

- Preserve raw pulses/sawtooth; explain the separate fitting mask, including
  25-second MPF recovery and bounds on gsw, VPD, flow, CO2, Ci, and A.
- Explain flash timestamp alignment (estimated 15-70 second offset search and
  pre-pulse gas window) and selection of six regular flashes per Q.
- Ci is not Cc; apparent-Ci fits and gm-prior-dependent fits are different estimates.
- ETR uses instrument absorptance/PSII partition assumptions; gm is not directly
  measured or robustly identified by these data.
- Q rises with time, so run-state changes and Q response are confounded.
- Report leaf-level replication, order imbalance, cutting/bucket handling, and
  environmental confounding. A null contrast is not evidence of equivalence.
- Verify literature values/units and temperature assumptions against cited sources.

## Release blockers and proposed fixes

- [ ] Replace prototype README with method overview, scope, data download instructions,
  environment setup, primary pipeline, and explicit limitations.
- [ ] Add MIT license for owned code; choose CC BY 4.0 or CC0 for data/documentation.
  Record authorship and required collaborator approvals; do not license third-party
  manuals, fonts, logos, and papers as our own.
- [ ] Track current bp program and setup guide; label older variants historical.
- [ ] Add environment/local-output exclusions. Existing gitignore only excludes
  __pycache__ and .DS_Store. Do not use blanket `git add .`.
- [ ] Create a small parser/flash/QC regression fixture and actual tests/CI.
- [ ] Document Python >=3.13.7 and uv lock; pin/document R and plantecophys.
  Local R is 4.2.2, plantecophys 1.4.6. No R lockfile exists.
- [ ] Separate core and optional presentation dependencies. reportlab, pypdf,
  qrcode, fontTools, skia, etc. are imported but not declared directly in pyproject.
  Several scientific imports are currently transitive dependencies.
- [ ] Make the data archive layout compatible with scripts; include all cohort
  raw data and excluded-run QC, plus method-development evidence.
- [ ] Add data dictionary, units, missing-value conventions, code-to-run mapping,
  dark-reference provenance, and acquisition/handling metadata where known.
- [ ] Fix stale hard-coded cohort figure counts: cohort script currently labels
  sun medians "high n=2, low n=5" although primary high sun n=3.
- [ ] Reconcile report summaries: combined report RMSE=1.81 versus Tier-C report
  RMSE=1.77. Determine whether these describe different versions/aggregations.
- [ ] Run primary pipeline in a clean environment using released data only;
  record commands, versions, seeds, diagnostics, expected outputs, and runtime.
- [ ] Create tagged release, Zenodo DOI, CITATION.cff, and cross-links to protocol
  and report; then publish GitHub and send Astera public URLs.

## What was verified here

- Read Git status/history/remotes and queried GitHub visibility and repository state.
- Parsed all 54 pre-existing Python sources for syntax (including eight BPs).
- Successfully loaded gas and flash data for every P01-P15 run using current code,
  without rewriting scientific outputs. Per-run counts are in cohort-validation.csv.
- Compared raw gas counts with the existing cohort points table.
- Checked CRC integrity of the four top-level output ZIP bundles.
- Searched current scripts/src/bp/configs for common credential assignment/key
  patterns: no matches. This is not a full secret/history/privacy audit.
- Inventoried local files with checksums. No GitHub push or visibility change.

Not yet verified: fresh uv installation, full R/Bayesian rerun, LI-6800 execution,
all archive-to-analysis path mappings, all provenance/redistribution rights,
or whether anything has separately been published to protocols.io/Zenodo.

## Suggested release boundary

GitHub should carry owned code, final and clearly labeled historical BPs, setup
guides, reproducibility instructions, small metadata/manifests, tests, licenses,
and citation information. Keep this repository's identity rather than inventing
a new project or reorganizing every script just before the deadline.

Zenodo should carry original gas logs/flash archives, excluded-run provenance,
processed tables, posterior draws, technical report, and selected figures.
Reference those by DOI from GitHub; avoid checking a 181.8 MiB generated CSV,
118 MiB vendor manual, or large poster images into Git history.

To refresh the inventory and read-only cohort validation:

```sh
uv run --frozen python scripts/audit_release_inventory.py --validate-cohort
```
