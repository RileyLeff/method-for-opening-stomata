# LI-6800 background programs

## Start here

Use [aciq_dat_biochemical_limits_fixed_h2or_flow600.py](aciq_dat_biochemical_limits_fixed_h2or_flow600.py)
with its [setup guide](aciq_dat_biochemical_limits_fixed_h2or_flow600_SETUP.md).
This is the release candidate matching the final nine-light-level Brownsville
cohort. These programs run on the instrument, not desktop Python: `bpdefs` is
provided by LI-COR.

The final cohort logs record Bluestem console v.2.1.13, head/fluorometer 1.4.23,
and 6800-01 chamber. Compatibility with other firmware has not been verified.
Perform a short supervised logging/control test before a long unattended run.

## Default experiment

External dark adaptation >=30 min; five-minute dark equilibration at CO2_r=420;
dark gas exchange and dark flash; 18-minute opening/induction at Q600 and CO2_r=120;
then ascending ramps at Q=50,100,200,350,600,1000,1400,1700,2000. CO2_r rises from
120 to 2000 ppm at 200 ppm/min. Active flashes start 60 seconds into each ramp,
then repeat every 90 seconds. Expected total is about 2 h 53 min.

Flow is 600 umol/s, leaf-temperature target is 25 C, and reference water vapor
is 21 mmol/mol. **Fixed H2O_r does not pin leaf VPD.** Monitor measured VPDleaf,
temperature, flow, and actuator behavior throughout. DAT calibration must match
the measurement configuration/flow. The intended spectrum is r90B40.

The program verifies an open log and incrementing observation counter, triggers
flashes asynchronously, and closes the log on normal completion. A killed program,
power loss, or instrument fault may bypass normal cleanup. After interruption,
inspect controls, restore appropriate CO2/light conditions manually, close/export
the log, and verify saved records and flash files. Logging checks are not a
guarantee of storage health.

## Other files

| Program | Status / purpose |
| --- | --- |
| aciq_dat_fixed_h2or_async_mpf_flow600.py | Earlier fixed-H2O_r protocol, different Q/dark/flash schedule; setup guide included |
| h2o_flash_actuator_diagnostic.py | Short diagnostic of humidity/flash/control interaction; setup guide included |
| morella_tpu_step_test.py | Experimental TPU diagnostic, not the main acquisition protocol |
| aciq_dat_bucket_highq_vpd1_flow600.py | Historical VPD-control development |
| aciq_dat_bucket_highq_test3.py | Historical development |
| qac_dat_test2.py | Historical development |
| test1.py | Historical prototype |

Historical programs are preserved for provenance, not recommended as equivalent
alternatives. No experimental behavior was changed during release organization.
Operator parameter validation is incomplete: use positive ramp rates, end CO2
above start CO2, sensible timing, and the documented Q ladder.

For official programming documentation and methods papers, see
[references](../docs/REFERENCES.md). The source files here are research software,
not LI-COR-endorsed instrument software.
