# photo26

Python workflow for plotting LI-6800 response curves from the files in `data/`.

```bash
uv run photo26
```

The command writes:

- `outputs/aci_curve.png`
- `outputs/aq_curve.png`

The A/Ci plot uses an explicit C3 FvCB-style curve fit with fitted `Vcmax`,
`Jmax`, `TPU`, and `Rd` labels. The A/Q plot uses a non-rectangular hyperbola
fit with `Amax`, `phi`, `theta`, `Rd`, and light compensation point labels.

To compare against PhoTorch:

```bash
uv run photo26-photorch
```

That command writes `outputs/photorch_fit_summary.csv`.
