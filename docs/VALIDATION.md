# Release validation, October 27, 2026

## Completed

- Full original workspace committed privately as `715bac5`; release branch starts
  from `0f788d4` and does not contain that snapshot in its ancestry.
- All 3,757 original data files match the pre-organization SHA-256 inventory.
- All 15 cohort logs and flash collections load; retained gas/flash counts agree.
- Ten regression tests pass: parser edge cases, source syntax, complete run-index
  coverage, all-run counts, primary cohort membership, pulse masking/flash join,
  saved flash counts, and basic FvCB/J invariants.
- Fresh release Python environment installed from the lock with the bayes extra.
  Splitting optional dependencies did not change any locked third-party versions.
- A separate local clone (without shared Git object shortcuts) installed the core
  environment, fetched and hydrated the LFS table, passed all ten tests, and
  verified every raw/result checksum. This is a local-origin check, not a remote
  GitHub quota or download test. Slide-template caches were subsequently excluded
  from the release; raw data and scientific results were not changed.
- Full non-sampling pipeline ran successfully in a disposable workspace with raw
  data linked read-only by convention and existing posterior files copied in.
  PYTHONPATH was pointed at the disposable source tree so the shared parser's
  output paths did not refer back to the original workspace.
- Recomputed cohort, Tier A, R benchmark, Python Tier B, posterior summaries, NPQ,
  additive yields, and leaf-level significance tests. No Bayesian sampling rerun.
- 23 of 32 regenerated CSVs match references at rtol=1e-7, atol=1e-9; nine differ.
  See the comparison and per-column difference tables in release-audit/.
- Confirmed the sole LFS object is the 190,592,913-byte cohort points CSV. Largest
  ordinary staged Git blob is approximately 15.39 MiB. No vendor/manual/temp/font
  paths are staged. Tested that the push hook rejects the private snapshot.
- Corrected one figure's outdated sample count and shade-pilot legend; inspected
  its regenerated image. Corrected the combined report RMSE to 1.77, matching
  the saved prediction table's 1.7740712952. Original snapshot remains unchanged.

## Numerical reproducibility caveat

Most differences are small optimizer/rounding differences. Some sensitivity-fit
parameters are not stable even though the modeled response is essentially unchanged:

| Run / fit | Saved Jmax25 | Rerun Jmax25 |
| --- | ---: | ---: |
| P13, fixed gm=0.27, no TPU | 74.34 | 155.12 |
| P07, apparent-Ci TPU variant | 264.14 | 258.69 |

P07's TPU parameter also moves from 21.07 to 25.60. Across all Tier-B predictions,
the largest change in modeled A is only about 0.00061 umol m^-2 s^-1. This is
consistent with weak identifiability of inactive limiting branches; the exact
runtime/optimizer cause was not isolated. These are not newly established
physiological differences. Do not claim byte-for-byte reproducibility or treat
every fitted parameter as identified because the fit converged.

The saved reference numeric results were deliberately retained, rather than
replaced with the rerun. The comparison uses the same row/column order and checks
all numeric and nonnumeric fields, with differences recorded rather than hidden.
Neither a passing software test nor close A predictions proves the biological
model is correct. The capacity caveats in LIMITATIONS.md still apply.

## Not completed by local setup

- Actual execution of the release program on a LI-6800 or compatibility testing
  on other firmware; recorded versions are provenance, not a certification.
- Fresh Bayesian sampling or reruns of every historical single-experiment script.
- Remote GitHub push, public visibility change, LFS billing/archive configuration,
  remote download verification, Zenodo DOI, or protocols.io publication.
- Independent literature-value revalidation or resolution of missing collection
  histories, genotype, and dark-waveform provenance.

The historical release-audit inventory predates organization and is not the
current file list. Use data/manifest.csv and outputs/manifest.csv for this release.
