# A/Ci/Q Biochemical-Limits Program

Program: `aciq_dat_biochemical_limits_fixed_h2or_flow600.py`

The default run measures A/Ci ramps at Q = 50, 100, 200, 350, 600, 1000,
1400, 1700, and 2000 umol m-2 s-1. It uses fixed H2O_r = 21 mmol mol-1,
Tleaf = 25 C, flow 600, CO2_r ramps from 120 to 2000 ppm at 200 ppm min-1,
and asynchronous MultiPhase flashes.

Expected default runtime is approximately 2 hours 53 minutes after launch.

## Before Clamping

1. Dark-adapt the target leaf outside the chamber for at least 30 minutes.
2. Prepare a new measurement log before moving the leaf.
3. Transfer and clamp the leaf with minimal exposure to room or sunlight.

## Console Setup

1. Stop all other Background Programs, Programs, Auto Control, and system tests.
2. Open a new log file with full gas-exchange, dynamic, fluorometer, controller,
   and diagnostic fields.
3. Enable Dynamic Assimilation using the calibration made at flow 600.
4. Suppress automatic CO2 and H2O matching during the measurement.
5. Configure the chamber as Full fluorometer.
6. Optimize and select the MultiPhase flash configuration.
7. Set Fluorometer Action at Log to `0: Nothing`.
8. Confirm the recommended `r90B40` spectrum.
9. Fill and seat the humidifier reservoir and inspect the desiccant.
10. Launch the BP and review its parameter dialog.

The BP refuses to start without an active measurement file. It writes a
no-flash test observation and verifies that the observation count increases
before changing the experimental controls.

## Default Sequence

1. Equilibrate the externally dark-adapted leaf for 5 minutes at Q = 0,
   CO2_r = 420 ppm, Tleaf = 25 C, and H2O_r = 21 mmol mol-1.
2. Log dark gas exchange every 10 seconds for 60 seconds.
3. Run one dark-adapted FoFm flash.
4. Open stomata and induce photosynthesis for 18 minutes at Q = 600 and
   CO2_r = 120 ppm while logging no-flash gas exchange every 10 seconds.
5. Drop to Q = 50 and park at low CO2 for 10 minutes.
6. Run the Q = 50 A/Ci ramp, leaving its first 60 seconds free of active MPFs.
7. Park for 5 minutes at each later Q level, then run the corresponding ramp.
8. Trigger asynchronous MPFs at 60, 150, 240, 330, 420, and 510 seconds of
   each default 564-second ramp.
9. Restore CO2_r = 420 and Q = 0, then close the measurement file.

## Why The Flash Schedule Changed

The previous one-minute flash schedule disturbed about 40 percent of each
ramp after applying a 25-second post-flash recovery mask. It also launched the
first MPF at the beginning of the ramp, directly perturbing the low-Ci branch
needed to estimate apparent Vcmax.

The new schedule preserves the first 60 seconds and spaces MPFs 90 seconds
apart. It still produces approximately 54 active flashes across the nine Q
levels while leaving substantially more undisturbed gas-exchange data.

## Analysis

- Join MPF files to gas exchange using event timestamps.
- Preserve all raw observations in diagnostic plots.
- Flag each physical MPF and its post-flash recovery period for static FvCB
  fitting.
- Treat induction logs separately from the measured A/Ci ramps.
- Use the dark baseline as an estimate of dark respiration, not as a fixed
  value for respiration in the light.
- Use active flashes jointly with gas exchange for electron-transport and
  variable-J screens.
- Verify actual Q rather than assuming the setpoint was attained.
