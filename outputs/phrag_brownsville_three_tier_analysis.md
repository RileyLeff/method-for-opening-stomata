# Brownsville Phragmites three-tier analysis

## Cohort and QC

The primary comparison contains five low-forest sun leaves (P02, P03, P06, P10, P11) and three high-forest sun leaves (P12, P13, P14). P02 gas exchange is retained with caution, but its overlapping fluorescence sequence is excluded. P05, P09, and P15 remain quarantined. Four high-forest shade leaves are pilots and do not enter the sun-leaf high/low comparison.

## Tier A: empirical responses

- Observed-domain maximum A: high median 11.27, low median 13.20 umol m-2 s-1; bootstrap high-low interval [-21.36, 12.69].
- Mean A over Q=200-2000 and Ci=400-1200: high 5.15, low 5.33; interval [-13.52, 14.47].
- A at Q=2000 and Ci=800: high 5.29, low 6.27; interval [-17.38, 17.22].
- Low-CO2-park gsw at Q=2000: high 0.059, low 0.078 mol m-2 s-1; interval [-0.054, 0.216].
- Dark Fv/Fm: high 0.830, low 0.822.
- NPQ at Q=2000 and Ci=800: high 2.237, low 3.224; bootstrap high-low interval [-1.302, -0.218].

The carbon-exchange summaries do not resolve a high/low difference. High leaves showed lower high-light NPQ in this sample, but the small leaf count and order imbalance make that a lead rather than a population conclusion.

## Tier B: conventional physiological fits

The conventional Q=2000 `plantecophys::fitaci` benchmark required a bilinear fallback for six of eight sun leaves, indicating weak Ac/Aj transition identification. Gas-only multi-Q FvCB fits also pushed theta to its lower bound for seven of eight leaves. TPU was selected in zero of eight leaves and did not improve fit.

- Gas-only apparent Vcmax25 medians: high 16.90, low 19.15.
- Gas-only apparent Jmax25 medians: high 55.12, low 74.32.
- Standalone fluorescence ETR asymptote proxy: high 106.02, low 106.81.
- Standalone fluorescence theta proxy: high 0.817, low 0.713.

Variable-J produced at least five algebraically valid gm estimates for only P10 and P12. Their medians, 0.044 and 0.072 mol m-2 s-1, are best read as failure of uncalibrated ETR/electron-allocation assumptions rather than credible Phragmites gm estimates.

## Tier C: Bayesian joint model

The model jointly fits 588 balanced A/Ci/Q observations, 63 clean ETR observations, and seven dark-respiration observations. Vcmax, Jmax, gm, respiration, residual error, and monotonic run-state decline vary by leaf. Alpha and theta have zone means and fixed leaf-heterogeneity scales calibrated from the clean standalone fluorescence fits. A shared within-day order slope is estimated from the full primary pool. Date is not modeled.

The final four-chain fit had zero divergences, zero maximum-tree-depth events, maximum R-hat 1.00, and minimum bulk ESS 1453.

| Quantity | Posterior median | 95% interval | Interpretation |
|---|---:|---:|---|
| Vcmax high/low | 0.84 | 0.44-1.61 | unresolved |
| Jmax high/low | 0.99 | 0.64-1.49 | unresolved |
| gm high/low | 0.92 | 0.46-1.82 | unresolved and prior-dominated |
| alpha high/low | 1.11 | 0.98-1.26 | suggestive, not resolved |
| theta high-low | 0.016 | -0.106-0.132 | unresolved |
| Vcmax ratio per 1 SD later order | 0.93 | 0.64-1.34 | unresolved |
| Jmax ratio per 1 SD later order | 0.88 | 0.71-1.12 | negative tendency |

Leaf Vcmax25 medians ranged from 6.3 to 73.7 and Jmax25 from 75.0 to 166.4 umol m-2 s-1. The model's posterior-mean A RMSE was 1.77 umol m-2 s-1 (reconciled against the saved 588-row prediction table during release preparation). gm remains broad, especially for P12, and should not be treated as directly measured.

## Common-order comparison

Both zones were evaluated over the same pooled order distribution after fitting the shared order slope.

- Order-standardized Vcmax25 center: high 30.31 [14.20, 75.33], low 36.12 [20.45, 75.20].
- Order-standardized Jmax25 center: high 112.98 [77.98, 162.51], low 113.76 [87.30, 149.99].

This is the requested pooled-order strategy. Because the current model has a shared linear order effect and no zone-by-order interaction, standardization changes the zone-specific levels but not the high/low ratio. A zone-specific order response is not identifiable from this order overlap.

## Sensitivity and bottom line

Removing P02 changes little on the gas side: Vcmax high/low becomes 0.84 [0.45, 1.57], Jmax 1.02 [0.66, 1.53], and gm 0.93 [0.45, 1.90]. The alpha tendency weakens to 1.05 [0.93, 1.20], confirming that it should not be presented as a firm treatment effect.

The defensible result is substantial leaf-to-leaf physiological variation with no resolved high/low difference in A, Vcmax, Jmax, gm, or theta in this small cohort. The most interesting leads are lower high-light NPQ in the high leaves and a pooled tendency for Jmax to decline with later measurement order. Both deserve targeted replication with balanced or randomized order.

