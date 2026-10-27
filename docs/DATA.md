# Data guide

All original research data are included in this repository, including excluded
runs. Zenodo will provide a permanent archival copy, not a prerequisite for use.

## Reading the files

`data/` preserves original filenames and dated acquisition folders. Early Morella
and Phragmites logs are directly in that directory. LI-6800 logs often have **no
extension**. They contain header/configuration information followed by a `[Data]`
section with grouped column headings, names, and units. Do not open them as simple
header-on-first-line CSVs. `photo26.curves.read_licor_export` reads that format.

Each later run normally includes an original MPF ZIP and its extracted `flashes/`
directory. JSON and XLSX files with the same MPF identifier represent the same
event: do not count them as two observations. DarkAdapted TXT summaries are also
retained where available. Raw files are never filtered in place.

`data/brownsville_phragmites_sample_metadata_2026.csv` is the preserved sample
metadata export. `outputs/phrag_brownsville_cohort_run_qc.csv` is the analysis QC
table. The source-of-truth run-to-log mapping is `RUNS` in
`scripts/analyze_phrag_brownsville_cohort.py`.

## Groups and replication

**High and Low mean elevation, not salinity.** High elevation is the lower-salinity
forest end; low elevation is the higher-salinity forest-marsh edge. These are
habitat categories, not measured salt concentrations for each leaf. Habitat light
availability also differs, so this is not an isolated experimental salinity effect.

| Role | IDs | Number |
| --- | --- | ---: |
| Primary high-elevation sun leaves | P12, P13, P14 | 3 |
| Primary low-elevation sun leaves | P02, P03, P06, P10, P11 | 5 |
| Retained high-elevation shade pilots | P01, P04, P07, P08 | 4 |
| Quarantined runs | P05, P09, P15 | 3 |

P02 remains in the gas comparison with a sensitivity analysis excluding it;
its fluorescence sequence is excluded from the primary fluorescence comparison.
That comparison therefore has seven leaves (three high, four low).

All 15 logs contain 78,576 gas rows and 869 flash JSON records. The retained twelve
contain 62,326 gas rows and 665 flash JSON records (656 active, nine dark).
Those are instrument observations, **not independent biological replicates**.
Sample IDs do not establish genetically independent plants in this clonal species.

## QC, selection, and missingness

- P05: severe overlapping MPF perturbations.
- P09: CO2 source exhausted after Q200.
- P15: progressive loss of A and gsw during the increasing-Q sequence despite
  stable instrument controls. Treat this as a flagged physiological/run-quality
  observation, not proof that the recorded data are intrinsically invalid.
- P03, P06, P07, P09, and P15 lack standalone dark-flash JSON in their folders.
  Existing analysis obtains dark Fv/Fm from log summaries. Some have separate
  DarkAdapted TXT files; waveform provenance is not equivalent across leaves.

Gas fitting masks exclude saturating pulses and the following 25 seconds and
apply bounds on environmental/gas variables. Cohort filters additionally screen
CO2 delivery and actuator saturation. See `load_points` in the biochemical-limits
script and `load_run` in the cohort script for the exact predicates. Diagnostics
and raw data retain these observations.

Flashes are aligned using the observed pulse timing, with an estimated offset
searched between 15 and 70 seconds; gas values come from the pre-pulse window.
Six regularly spaced active flashes per Q are selected where available. These
are analysis assumptions, not guaranteed universal instrument timing.

Binned curves use 50 ppm Ci bins with at least three valid gas rows per bin.
Interpolated summaries are only returned within observed Ci support. Empty/NaN
processed values mean missing, screened, unsupported, or unavailable estimates;
consult associated QC rather than substituting zero. Raw exports also use `-`.

## Common fields

Original log unit rows are authoritative, especially for firmware-dependent fields.

| Field | Meaning | Units |
| --- | --- | --- |
| A | Reported net carbon assimilation | umol CO2 m^-2 s^-1 |
| Asty / Adyn | Steady/dynamic assimilation outputs; inspect instrument configuration | umol CO2 m^-2 s^-1 |
| gsw | Stomatal conductance to water vapor | mol H2O m^-2 s^-1 |
| Ci | Intercellular CO2, not chloroplastic Cc | umol mol^-1 |
| Ca | Chamber CO2 concentration | umol mol^-1 |
| CO2_r / CO2_s | Reference/sample CO2 | umol mol^-1 |
| Q | Incident photon flux; includes physical flash excursions | umol photons m^-2 s^-1 |
| Q_level | Assigned nominal light level | umol photons m^-2 s^-1 |
| H2O_r / H2O_s | Reference/sample water vapor mole fraction | mmol mol^-1 |
| VPDleaf | Vapor pressure deficit at the leaf | kPa |
| Tleaf | Leaf temperature | degrees C |
| Flow | Chamber flow | umol s^-1 |
| elapsed / minute | Elapsed time | seconds / minutes |
| Fv/Fm | Dark-adapted maximum PSII efficiency | dimensionless |
| PhiPS2 | Operating PSII quantum yield | dimensionless |
| ETR | Instrument-derived electron transport proxy | umol electrons m^-2 s^-1 |
| NPQ | Nonphotochemical quenching index; not an additive energy fraction | dimensionless |
| qL | Estimate of open PSII centers under lake-model assumptions | dimensionless |
| YII / YNPQ / YNO | Photochemical, regulated, and unregulated yields | dimensionless |

ETR and additive yields inherit optical/model assumptions. Y(NO) is not a direct
measurement of leaf injury. Fluorescence does not independently identify gm here.

## Provenance limits

All 15 final-cohort headers record console Bluestem v.2.1.13, scripts 2024.01
2.1.13 (Apr 2024), head/fluorometer 1.4.23, and chamber 6800-01. These are observed
versions, not a claim about the latest available firmware.

The supplied metadata do not fully record collection-to-measurement intervals,
bucket duration, per-leaf salinity, or haplotype/genotype confirmation. Do not fill
these gaps with assumptions. Cutting/bucket handling and measurement order can
affect physiology. The method-development logs include indoor/cut-plant experiments
and should not be pooled indiscriminately with the final field-site cohort.

The June 21 Morella filename contains "control", but the sequence itself is not a
randomized concurrent control experiment. Preserve its filename and describe the
actual treatment sequence.
