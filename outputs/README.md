# Saved scientific outputs

These are reference results from the working analyses, not a claim that every
historical figure represents the final comparison. Each family remains in place
to preserve script compatibility. Raw evidence is in `../data/`.

| Family | Producer / interpretation |
| --- | --- |
| phrag_brownsville_cohort_* | Cohort ingestion, QC, points, curves, flash joins, common-Ci summaries |
| phrag_brownsville_tier_a_* | Empirical leaf-level responses and contrasts |
| phrag_brownsville_tier_b_plantecophys_* | R conventional Q2000 benchmark |
| phrag_brownsville_tier_b_joint_* | Python multi-Q FvCB variants |
| phrag_brownsville_tier_b_variable_j_* | Exploratory gm calculation, not a validated gm measurement |
| phrag_brownsville_tier_c_* | Joint Bayesian fit, diagnostics, posterior draws, sensitivity, order standardization |
| phrag_brownsville_vcmax_jmax_* | Capacity envelopes and literature comparison |
| phrag_brownsville_npq_* | NPQ interpolation/plots and leaf-level tests |
| phrag_brownsville_additive_* / phrag_brownsville_yno_* | Fluorescence energy-partition summaries |
| poster_morella_* | Morella opening/blue-light story figures |
| Other species/run-specific names | Earlier experiment-specific diagnostics and fits; see script index |

The main starting reports are [combined analysis](phrag_brownsville_three_tier_analysis.md),
[Tier A](phrag_brownsville_tier_a_report.md), [Tier B](phrag_brownsville_tier_b_report.md),
[Tier C](phrag_brownsville_tier_c_report.md), and the
[fluorometry walkthrough](phrag_brownsville_fluorometry_walkthrough.md).
The combined report's RMSE was corrected during release preparation to 1.77,
matching the saved 588-row prediction table (1.7740713 before rounding).

The large `phrag_brownsville_cohort_points.csv` is the sole LFS file. Fetch it with
`git lfs pull`, or regenerate with the cohort script. The other tables, figures,
and primary/sensitivity `.nc` posterior files are ordinary Git files.

`manifest.csv` captures the release-reference checksums. A reproduction run will
overwrite matching files and may legitimately change checksums, especially plots
or stochastic fits. Inspect differences before updating the manifest; do not
silently bless regenerated results. ZIP convenience bundles are omitted because
their component files are available here or in data/. Poster branding assets and
render scratch files are intentionally not distributed.
