# Tier C: hierarchical Bayesian FvCB results

## Model

The model jointly uses balanced A/Ci/Q observations, dark gas exchange, and fluorescence ETR. Leaf-level Vcmax, Jmax, gm, respiration, and monotonic run fatigue are partially pooled. Alpha and theta have zone means plus fixed leaf-heterogeneity scales calibrated to the clean standalone fluorescence fits; estimating those variance scales jointly with a free ETR calibration was not identifiable. High/low forest and within-day order enter the capacity hierarchy. Literature priors are centered at Vcmax25=84, Jmax25=116, and gm=0.27 but deliberately broad.

Collection date is not assigned a separate effect: two retained dates contain only one leaf, and zone/date/order are too entangled to distinguish date variation from leaf variation. The resulting zone contrast therefore remains vulnerable to date confounding despite the explicit order adjustment.

## Zone effects

- Vcmax high/low ratio: median 0.84, 95% interval [0.45, 1.57], P(> null)=0.29, posterior/prior SD=0.81.
- Jmax high/low ratio: median 1.02, 95% interval [0.66, 1.53], P(> null)=0.54, posterior/prior SD=0.53.
- gm high/low ratio: median 0.93, 95% interval [0.45, 1.90], P(> null)=0.42, posterior/prior SD=1.03.
- alpha high/low ratio: median 1.05, 95% interval [0.93, 1.20], P(> null)=0.79, posterior/prior SD=0.27.
- theta high-low difference: median 0.008, 95% interval [-0.111, 0.124], P(> null)=0.56, posterior/prior SD=0.71.

An interval spanning 1 for a ratio, or 0 for theta difference, means the current data do not resolve the direction. A posterior/prior SD near 1 means the contrast remains largely prior-dominated.

## Order and state

- Vcmax order ratio per one SD later within day: median 0.93, 95% interval [0.65, 1.34], P(> null)=0.35, posterior/prior SD=0.84.
- Jmax order ratio per one SD later within day: median 0.91, 95% interval [0.74, 1.18], P(> null)=0.21, posterior/prior SD=0.54.
- Median inferred full-run A decline ranged from 0.05 to 3.91 umol m-2 s-1 across leaves.

Because all Q sequences were ascending, fatigue and high-Q response remain structurally confounded. The shrinkage term is a sensitivity adjustment, not an independently identified physiological process.

## Absolute parameters

- Leaf Vcmax25 posterior medians span 6.3-67.6.
- Leaf Jmax25 posterior medians span 75.0-149.0.
- Leaf gm posterior medians span 0.302-0.422 mol m-2 s-1.
- Fluorescence ETR is used on the LI-6800 reported scale. Absolute J therefore inherits the instrument's absorptance and PSII-partition assumptions; alternative electron sinks remain model discrepancy rather than an independently fitted calibration factor.

## Diagnostics

- Posterior-mean A RMSE: 1.64 umol m-2 s-1.
- Divergences: 0.
- Maximum reported R-hat: 1.010.

These posteriors describe plausible parameter combinations under the model. Strong Vcmax/Jmax learning accompanied by weak gm or zone-effect contraction should be interpreted as biochemical capacity information with unresolved diffusion and treatment contrasts.
