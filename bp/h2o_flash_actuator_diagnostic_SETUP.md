# LI-6800 H2O / Flash Diagnostic

Expected runtime: approximately 16-17 minutes. This is a controller test,
not a physiological curve; dark adaptation and stomatal induction are not
required.

## Before starting

1. Clamp a usable leaf and configure the normal R90B40 spectrum.
2. Stop every other Background Program and Program.
3. Turn Auto Control off. Verify that no system test is running.
4. Fill and seat the humidifier reservoir and confirm the desiccant is usable.
5. Configure the chamber as Full fluorometer.
6. In Log Setup > Fluorometer Options, set **Flr Action at Log** to
   `1: FoFm (dark) or FsFm' (light)`, set **Flash type** to `MultiPhase`,
   and set the minimum flash interval to 60 seconds. Action 2 is a different
   setting that adds an Fo' dark pulse; it does not mean MultiPhase.
7. Suppress automatic CO2 and H2O matching for this test.
8. Enable Dynamic Assimilation and select the DAT calibration made at flow 600.
9. Open a new log file. Include the full diagnostic/system output, especially:
   VPDleaf, Tleaf, Txchg, Hr/H2O_r, Hs/H2O_s, H2O_r_sp, Humidifier_%,
   Desiccant_%, CO2_r, CO2_r_sp, Qin, Flow, Fan_speed, and fluorometer IDs.
10. Start `h2o_flash_actuator_diagnostic.py` and leave its defaults unchanged.

The program sets flow to 600 umol s-1, fan speed to 10000 rpm, Tleaf to 25 C,
VPD_leaf to 1.0 kPa, Q to 800, and initial CO2_r to 120 ppm.

## Test sequence

- `EQUIL`: 4 minutes, constant conditions, no flash.
- `A_NOFLASH`: 3-minute CO2 ramp from 120 to 720 ppm, no flash.
- `RECOVER_1`: 2 minutes at 120 ppm; VPD control is reasserted.
- `B_DEFAULT`: identical ramp using `LOG(flr="Default")` and 60-second MPFs.
- `RECOVER_2`: 2 minutes at 120 ppm; VPD control is reasserted.
- `C_ONE_MPF`: one explicit MPF followed by 2 minutes of no-flash logging.

Do not manually adjust humidity, temperature, light, CO2, or the actuators
after pressing Continue. If the actuator display jumps, let the test finish.

## Bring back

1. The complete measurement log.
2. The flash-file ZIP.
3. The completed BP Monitor run log, especially every line beginning
   `H2ODIAG`. A screenshot is adequate if the console cannot export it.
4. A note about any warning or `H2O SP Override` displayed during the test.

## Interpretation

| Result | Primary implication |
| --- | --- |
| A fails before any MPF | CO2 ramp, parallel control, or general H2O controller problem |
| A holds but B fails at an MPF | `LOG(Default)` / fluorometer interaction strongly implicated |
| A and B hold but C fails | Single explicit MPF action implicated |
| H2O controller `Type`, `Target`, or `Active` changes | Internal controller mode is being replaced |
| H2O override list gains an entry | The entry identifies the process taking control |
| Controller remains Fully-Auto on VPD_leaf but commands full drying | LI-6800 controller/firmware fault or stale actuator-state bug |
