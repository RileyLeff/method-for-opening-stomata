# A/Ci/Q DAT With Fixed H2O_r and Asynchronous MPFs

Program: `aciq_dat_fixed_h2or_async_mpf_flow600.py`

Default runtime is approximately 2 hours 50 minutes. The defaults reproduce the
current Q ladder, 120-2000 ppm CO2 ramps at 200 ppm/min, and flow 600.

## Console setup

1. Stop all other Background Programs, Programs, Auto Control, and system tests.
2. Open a new log file with the full gas-exchange, dynamic, fluorometer,
   controller, and diagnostic fields.
3. Enable Dynamic Assimilation and load the DAT calibration made at flow 600.
4. Suppress automatic CO2 and H2O matching during the measurement.
5. Configure the chamber as Full fluorometer.
6. Optimize and select the MultiPhase flash configuration.
7. Set Fluorometer Action at Log to `0: Nothing`. This is important: the BP
   triggers MPFs directly and ordinary logs must not launch a synchronous flash.
8. Confirm the intended spectrum, normally R90B40.
9. Fill and seat the humidifier reservoir and inspect the desiccant.
10. Clamp the leaf, open the log, launch the BP, and review its dialog defaults.

The BP now checks the logger before touching the experimental controls. It will
stop immediately unless `LOG:IsFileOpen` is true. It then writes one no-flash
test observation and confirms that `LOG:ObsCount` increased. A successful check
appears in the BP run log as `ACIQ|LOGGER_READY` with the active filename and
observation count.

If either check fails, do not proceed with a long run. Close the file, create a
new one through **Log Files > Open a Log File > New File**, confirm the
upper-right button reads `Log (0)`, and verify that one manual log changes it to
`Log (1)`.

## Moisture control

The BP sets H2O_r to 21.0 mmol mol-1 once at startup and never changes the
humidity-control target. This same incoming moisture condition applies during
dark adaptation, low-CO2 parks, and every CO2 ramp. Actual H2O_s and VPDleaf
remain measured outcomes and can vary with transpiration.

At each illuminated Q level the BP parks at CO2_r = 120 ppm, acclimates the
leaf, runs the ascending CO2 ramp with no-flash gas-exchange logging, and
triggers an asynchronous MPF every 60 seconds after a pre-flash gas log.

## Analysis

- Join MPF files to the gas log using event timestamps.
- Initially flag the first 25 seconds after each MPF. The diagnostic showed
  that flash heating peaks after about 6 seconds and normally settles within
  16-26 seconds.
- Inspect the actual CO2_r_sp trace. It should no longer contain the plateau
  and catch-up jump produced by synchronous `LOG(flr="Default")`.
- Inspect VPDleaf, H2O_r, H2O_s, Humidifier_%, and Desiccant_% by Q level.
- The BP run log records the selected fixed value under `ACIQ|FIXED_H2O_R`.
- At normal completion the BP records the final observation count, adds a closing
  remark to the measurement file, and explicitly closes the file. Refresh the
  file browser before viewing or exporting it.
