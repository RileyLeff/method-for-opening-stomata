# Tier C: hierarchical Bayesian FvCB results

## Model

The model jointly uses balanced A/Ci/Q observations, dark gas exchange, and fluorescence ETR. Leaf-level Vcmax, Jmax, gm, respiration, and monotonic run fatigue are partially pooled. Alpha and theta have zone means plus fixed leaf-heterogeneity scales calibrated to the clean standalone fluorescence fits; estimating those variance scales jointly with a free ETR calibration was not identifiable. High/low forest and within-day order enter the capacity hierarchy. Literature priors are centered at Vcmax25=84, Jmax25=116, and gm=0.27 but deliberately broad.

Collection date is not assigned a separate effect: two retained dates contain only one leaf, and zone/date/order are too entangled to distinguish date variation from leaf variation. The resulting zone contrast therefore remains vulnerable to date confounding despite the explicit order adjustment.

## Zone effects

- Vcmax high/low ratio: median 0.84, 95% interval [0.44, 1.61], P(> null)=0.30, posterior/prior SD=0.84.
- Jmax high/low ratio: median 0.99, 95% interval [0.64, 1.49], P(> null)=0.48, posterior/prior SD=0.53.
- gm high/low ratio: median 0.92, 95% interval [0.46, 1.82], P(> null)=0.41, posterior/prior SD=1.02.
- alpha high/low ratio: median 1.11, 95% interval [0.98, 1.26], P(> null)=0.95, posterior/prior SD=0.26.
- theta high-low difference: median 0.016, 95% interval [-0.106, 0.132], P(> null)=0.61, posterior/prior SD=0.71.

An interval spanning 1 for a ratio, or 0 for theta difference, means the current data do not resolve the direction. A posterior/prior SD near 1 means the contrast remains largely prior-dominated.

## Order and state

- Vcmax order ratio per one SD later within day: median 0.93, 95% interval [0.64, 1.34], P(> null)=0.34, posterior/prior SD=0.86.
- Jmax order ratio per one SD later within day: median 0.88, 95% interval [0.71, 1.12], P(> null)=0.12, posterior/prior SD=0.51.
- Median inferred full-run A decline ranged from 0.05 to 3.80 umol m-2 s-1 across leaves.

Because all Q sequences were ascending, fatigue and high-Q response remain structurally confounded. The shrinkage term is a sensitivity adjustment, not an independently identified physiological process.

## Absolute parameters

- Leaf Vcmax25 posterior medians span 6.3-73.7.
- Leaf Jmax25 posterior medians span 75.0-166.4.
- Leaf gm posterior medians span 0.173-0.394 mol m-2 s-1.
- Fluorescence ETR is used on the LI-6800 reported scale. Absolute J therefore inherits the instrument's absorptance and PSII-partition assumptions; alternative electron sinks remain model discrepancy rather than an independently fitted calibration factor.

## Diagnostics

- Posterior-mean A RMSE: 1.77 umol m-2 s-1.
- Divergences: 0.
- Maximum reported R-hat: 1.000.

These posteriors describe plausible parameter combinations under the model. Strong Vcmax/Jmax learning accompanied by weak gm or zone-effect contraction should be interpreted as biochemical capacity information with unresolved diffusion and treatment contrasts.
