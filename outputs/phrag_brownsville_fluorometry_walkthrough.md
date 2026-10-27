# Brownsville Phragmites fluorometry walkthrough

## What a fluorescence flash measures

Chlorophyll fluorescence is the small fraction of absorbed excitation energy re-emitted as light. Its yield changes with the state of PSII reaction centers and with regulated heat dissipation. A saturating multiphase flash briefly drives PSII centers closed, allowing the LI-6800 to estimate maximum fluorescence under the leaf's current state.

### Dark-adapted flash

- `Fo`: minimum fluorescence when PSII centers are open and regulated non-photochemical quenching has relaxed.
- `Fm`: maximum fluorescence during the saturating flash, when PSII centers are driven closed.
- `Fv = Fm - Fo`.
- `Fv/Fm = (Fm - Fo) / Fm`: maximum dark-adapted PSII photochemical efficiency.

The dark flash also seeds the dark `Fm` reference used by later quenching calculations. This makes `NPQ` stateful: if the baseline dark state was incomplete or the reference ceases to represent the leaf, every later NPQ value inherits that problem.

### Light-adapted flash

- `Fs`: steady fluorescence immediately before the flash.
- `Fm'`: maximum fluorescence during the flash while light-induced quenching remains engaged.
- `Fo'`: minimum light-adapted fluorescence, measured or estimated after a dark/far-red event.
- `PhiPSII = (Fm' - Fs) / Fm'`: operating quantum yield of PSII. It is the fraction of absorbed photons delivered to PSII that are being used photochemically under the current state.
- `Fv'/Fm' = (Fm' - Fo') / Fm'`: potential PSII efficiency if all centers were open while the current non-photochemical quenching remained engaged.
- `qP = (Fm' - Fs) / (Fm' - Fo')`: photochemical quenching coefficient under a puddle-style formulation.
- `qL = qP * Fo' / Fs`: lake-model estimate of the fraction of open PSII reaction centers; `1-qL` estimates the closed fraction.
- `NPQ = (Fm - Fm') / Fm'`: Stern-Volmer non-photochemical quenching, a dimensionless index of regulated excitation-energy dissipation as heat.

`PhiPSII`, `qL`, and `NPQ` describe different pieces of the same energy-allocation state. A leaf can have low `PhiPSII` because many centers are closed, because heat dissipation is high, or both.

## From PhiPSII to ETR

The normal calculation is approximately:

`ETR = PhiPSII * absorbed photon flux * fraction of absorbed photons assigned to PSII`

On the LI-6800 this uses its `Qabs` and `PS2/1` assumptions/settings. ETR is therefore not a direct electron count. It is a fluorescence-derived proxy for linear electron transport whose absolute scale depends on leaf absorptance, spectral composition, PSI/PSII partitioning, cyclic flow, photorespiration, nitrate reduction, the Mehler reaction, and other alternative sinks.

This explains the high-light pattern in the data. At Ci=800 and Q=2000, median `PhiPSII` is only about 0.12, but multiplying that declining efficiency by much larger Q still gives ETR near 100 umol electrons m-2 s-1. ETR approaches a plateau while NPQ rises and qL falls.

## What this dataset contains

- Eight valid primary gas-exchange leaves: five low and three high.
- Seven clean primary fluorescence series: P03, P06, P10, P11, P12, P13, and P14.
- P02 fluorescence is excluded because flash sequences overlapped, although its gas exchange remains in the primary analysis.
- Each clean leaf contributes nine Q levels. Multiple flashes along each CO2 ramp are summarized at a common Ci=800, yielding 63 ETR observations for the joint model.
- Seven usable dark gas-exchange observations help anchor dark respiration. The dark fluorescence flash provides Fv/Fm but does not independently identify daytime Rd or gm.

The active flashes are treated as discrete probes of the leaf state at their ramp conditions. They are not treated as ordinary high-Q points in a continuous light record. Gas points affected by post-flash transients are handled separately in the gas-exchange processing.

## Direct fluorescence results

At Q=2000 and Ci=800:

| Metric | High median | Low median |
|---|---:|---:|
| ETR | 101.25 | 102.59 |
| PhiPSII | 0.120 | 0.122 |
| NPQ | 2.237 | 3.224 |
| qL | 0.094 | 0.119 |

Dark Fv/Fm medians were 0.830 high and 0.822 low. Most values were in a narrow 0.819-0.831 range; P02 was lower at 0.784 and deserves caution. The high-light ETR and PhiPSII medians are nearly identical between zones. The clearest raw fluorescence separation is higher NPQ in the low leaves, paired with somewhat higher qL. With only three versus four clean leaves, this is a physiological lead, not a stable population estimate.

Standalone non-rectangular-hyperbola fits to each ETR-Q series gave median ETR asymptotes of 106.0 high and 106.8 low. Their leafwise alpha and theta estimates varied enough that zone medians alone were not persuasive.

## How fluorescence enters J(Q)

The model uses:

`J(Q) = (alpha*Q + Jmax - sqrt((alpha*Q + Jmax)^2 - 4*theta*alpha*Q*Jmax)) / (2*theta)`

- `alpha`: initial effective slope of electron transport versus incident Q. Here it absorbs true quantum efficiency plus absorptance and PSII-partition effects.
- `theta`: curvature. Values nearer 1 make a sharper transition into saturation; lower values make a more gradual bend.
- `Jmax`: asymptotic electron-transport capacity on the assumed ETR scale.

Gas exchange then uses J in the electron-transport-limited FvCB rate:

`Aj = J * (Cc - Gamma*) / (4*Cc + 8*Gamma*)`

The joint model asks for one parameter set that can explain both ETR-Q shape and A-Ci-Q carbon exchange. It uses ETR in the likelihood, but it does not force NPQ or qL into FvCB because neither maps stoichiometrically to net carbon assimilation without a much richer state and energy-partition model.

The final model allows leaf variation in alpha and theta at fixed heterogeneity scales estimated from the standalone clean curves. This avoids two opposite errors: pretending every flash point is an independent leaf, or fitting a free leaf-level curvature variance plus an arbitrary ETR scale that the data cannot identify.

## What alpha and theta say here

- Effective alpha high/low ratio: 1.11 with 95% interval 0.98-1.26. There is a tendency toward a steeper effective low-light J response in high leaves, but it is not resolved.
- Theta high-low difference: 0.016 with interval -0.106 to 0.132. There is no evidence of a zone curvature difference.
- Jmax high/low ratio: 0.99 with interval 0.64-1.49. The fluorescence-constrained electron-transport capacity does not separate by zone.

Removing P02 weakens alpha to 1.05 [0.93, 1.20], even though P02 contributes no fluorescence likelihood. That happens because the joint carbon and electron-transport model reallocates capacity among correlated parameters. It is a useful reminder that this alpha is an effective joint-model parameter, not a direct optical measurement.

## Why variable-J gm did not work

The variable-J method uses fluorescence-derived J and gas exchange to infer chloroplast CO2:

`Cc = Gamma* * (J + 8*(A + Rd)) / (J - 4*(A + Rd))`

and then:

`gm = A / (Ci - Cc)`

This calculation becomes unstable when `J - 4*(A + Rd)` is small, when Cc falls outside a physical range, or when fluorescence ETR is not the electron flux assumed by the carbon model. At native LI-6800 ETR scaling, only P10 and P12 yielded at least five algebraically usable points, with median gm about 0.044 and 0.072 mol m-2 s-1. Those values are much lower than the roughly 0.27 Phragmites literature expectation and are highly sensitive to scaling ETR by 0.8 or 1.2.

The appropriate conclusion is that the uncalibrated variable-J assumptions fail for these data, not that Phragmites gm is truly near 0.06. A dark flash alone cannot repair this because it constrains maximum PSII efficiency and the NPQ reference, not leaf absorptance, PSII partitioning, alternative sinks, or daytime respiration.

## Order, state, and the requested standardization

There are two distinct order problems:

1. `Within-day leaf order`: which leaf run was first, second, third, or fourth that day. Tier C estimates one shared order slope from the full primary pool.
2. `Within-run Q progression`: every run increases Q in the same order. A leaf-level monotonic fatigue term allows A to decline over the run, but fatigue and true high-Q behavior cannot be fully separated without randomized or reversed Q sequences.

For within-day order, both zones were projected across orders 1-4 and then averaged over the same pooled observed order distribution. The fitted capacity ratios per one-SD later order were 0.93 [0.64, 1.34] for Vcmax and 0.88 [0.71, 1.12] for Jmax. This gives a negative Jmax order tendency but does not resolve it.

The common-order standardized centers are:

- Vcmax25: high 30.3 [14.2, 75.3], low 36.1 [20.5, 75.2].
- Jmax25: high 113.0 [78.0, 162.5], low 113.8 [87.3, 150.0].

Because the model assumes a shared linear order effect, the high/low ratio is the same at each order. A zone-by-order interaction would require better overlap and more leaves; fitting one now would mainly describe priors.

## Best next measurements

- Randomize or alternate Q order on replicate leaves to separate high-Q physiology from run fatigue.
- Balance high and low leaves across within-day order.
- Measure absorptance spectrally and, if possible, constrain PSI/PSII partitioning before using variable-J gm quantitatively.
- Include an independent Rd/light-respiration constraint such as a Kok or Laisk block if gm is a priority.
- Continue active flashes, because they materially identify J(Q) shape, but retain NPQ and qL as state diagnostics until a dynamic energy-allocation model is explicitly fitted.

