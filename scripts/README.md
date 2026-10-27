# Script index

Run scripts from the repository root with `uv run --frozen python scripts/NAME.py`.
For Bayesian code/posterior summaries add `--extra bayes` to uv. Scripts write to
outputs/ and may overwrite existing results. The directory stays flat because
research scripts import one another by name.

## Supported cohort pipeline

| Script | Role |
| --- | --- |
| reproduce.py | Dry-run/execute orchestration in dependency order |
| analyze_phrag_brownsville_biochemical_limits.py | Shared final-protocol loaders, pulse masks, flash alignment; historical two-run entry point |
| analyze_phrag_brownsville_cohort.py | All 15 runs, metadata/QC, derived tables and diagnostic figures |
| analyze_phrag_brownsville_tier_a.py | Empirical comparisons and order summaries |
| fit_phrag_brownsville_tier_b_plantecophys.R | Conventional R benchmark; run before Python Tier B |
| analyze_phrag_brownsville_tier_b.py | Joint FvCB variants, exploratory variable-J gm |
| analyze_phrag_brownsville_tier_c_bayesian.py | Joint hierarchical model and sensitivity fit |
| summarize_phrag_brownsville_order_standardization.py | Pooled-order posterior comparison |
| plot_phrag_brownsville_capacity_envelopes_literature.py | Capacity envelopes and external comparison |
| plot_phrag_brownsville_npq.py | NPQ across light and Ci |
| analyze_phrag_brownsville_additive_yields.py | Additive fluorescence yields and figures |
| test_phrag_brownsville_npq_zone_difference.py | Statistical analysis, NOT a software test |

## Method development and earlier runs

These entry points preserve the investigation and are not interchangeable with
the final cohort analysis. Source filenames/constants identify their specific inputs.

- analyze_bptest_dynamic_ramp.py
- analyze_h2o_flash_diagnostic.py
- analyze_morella_aqc_surface.py
- analyze_morella_capacities.py
- analyze_morella_outside_regular.py
- analyze_morella_tpu_step_test.py
- analyze_phrag_aq_dark_rerun.py
- analyze_phrag_brownsville_salinity_edge.py
- analyze_phrag_brownsville_upland.py
- analyze_phrag_brownsville_upland_late_repeat.py
- analyze_phrag_bucket_highq.py
- analyze_phrag_crc_fixed_h2or_async.py
- analyze_phrag_crc_indoor_controller_test.py
- analyze_phrag_crc_indoor_vpd_flow600.py
- analyze_phrag_indoor_tpu_controller_test.py
- compare_morella_regular_runs.py
- compare_phrag_brownsville_positions.py
- compare_phrag_brownsville_upland_repeat.py
- fit_morella_photorch.py (requires photorch extra)
- fit_phrag_dat_2021_method.R (requires plantecophys)
- fit_phrag_fvcb_surface.py
- plot_phrag_dat_2021_fit.py
- plot_phrag_qci_surface.py

## Exports and presentation

- make_morella_poster_figures.py: early method figures and data exports.
- make_morella_slide6_figure.py: simplified talk figure.
- export_blog_interactives.py: machine-readable plot data for the blog.
- try_xy_photosynthesis.py: plotting experiment; requires xy extra.
- build_hidden_science_pdf.py, build_harlan_2026.py,
  build_harlan_2026_leaf_poster.py, poster_warp.py, redraw_stoma_background.py:
  historical presentation tooling, external fonts/assets and optional packages;
  not portable scientific reproduction targets. See THIRD_PARTY.md.

## Release tools

- release_manifest.py: standard-library checksum/index writing and verification.
- audit_release_inventory.py: whole local workspace inventory; includes local-only
  material when present. The October 27 audit is a historical snapshot, not the
  public release manifest.
- package_valid_phrag_leaves.py: historical valid-only convenience ZIP. It excludes
  suspect raw files and changes paths; the complete data/ tree is preferred for
  reproduction. This packager is not the whole release.

Software regression tests live in ../tests/, not in scripts/.
