# Tier B: physiological-model results

## Conventional benchmark

`plantecophys::fitaci` fit all 8 valid sun leaves at Q=2000, but only 2 used the default transition fit; the remainder required its bilinear fallback. Fits with a transition pinned near Ci=2000 should be treated as weakly partitioned rather than clean Vcmax/Jmax estimates.

## Joint multi-Q FvCB fit

- Apparent Vcmax25: high median 16.90, low median 19.15, high-low -2.25 umol m-2 s-1.
- Apparent Jmax25: high median 55.12, low median 74.32, high-low -19.20 umol m-2 s-1.
- Gas-exchange theta: high median 0.100, low median 0.100, high-low -0.000; 7/8 leaves reached the lower bound.
- Fluorescence ETRmax proxy: high median 106.02, low median 106.81, high-low -0.79 umol electrons m-2 s-1.
- Fluorescence theta proxy: high median 0.817, low median 0.713, high-low 0.104.

The TPU variant selected TPU limitation in 0/8 sun leaves and did not improve the fits. TPU is not identified by this dataset. Fitted daytime Rd exceeded the directly observed dark CO2-efflux magnitude in 5/7 comparable leaves, another sign that the unconstrained gas-only model is assigning structural mismatch to Rd.

The joint fit uses storage-corrected A (`A == Adyn`) from all nine light levels and treats Ci as chloroplast CO2 only in the apparent/infinite-gm baseline. Estimates are therefore apparent capacities until gm is credibly constrained.

## Mesophyll-conductance diagnostic

At the LI-6800 ETR scaling, variable-J produced at least five algebraically valid flash estimates for only 2 fluorescence-QC-passing primary leaves. Their cross-leaf median was 0.058 mol m-2 s-1, far below the approximately 0.27 Phragmites literature value. This is best interpreted as failure of the uncalibrated variable-J assumptions, not evidence that true gm is 0.06. Scaling ETR by 0.8 and 1.2 is retained because absorptance, PSII partitioning, Rd, and alternative electron sinks strongly affect the calculation.

## Interpretation boundary

Local optimizer standard errors do not include leaf-to-leaf variation, ramp-order confounding, or uncertainty in electron-transport calibration. The Bayesian tier will explicitly test which parameters move away from their priors and which remain prior-dominated.
