# =====================================================================
#  h2o_flash_actuator_diagnostic.py -- LI-6800 Background Program
#
#  Short A/B diagnostic for the H2O actuator failure observed during
#  continuous DAT A/Ci ramps. At one light level, the program runs:
#
#    EQUIL       4 min at Q=800, CO2_r=120, no fluorometer action
#    A_NOFLASH   3 min CO2 ramp 120 -> 720, LOG(flr="0: Nothing")
#    RECOVER_1   2 min at CO2_r=120, VPD_leaf reasserted
#    B_DEFAULT   3 min identical ramp, LOG(flr="Default")
#    RECOVER_2   2 min at CO2_r=120, VPD_leaf reasserted
#    C_ONE_MPF   one explicit light-adapted MPF, then 2 min monitoring
#
#  Expected duration: about 16-17 min, including the flash operations.
#
#  The measurement log records the normal gas-exchange and actuator
#  variables. The BP run log also gets a snapshot every 5 s of the H2O
#  controller constants and override list. These snapshots expose Active,
#  Type, Target, SetPoint, PercentH, PercentD, Status, and any override.
#
#  Console setup before launch:
#    - Stop every other Background Program, Program, and Auto Control.
#    - Open a NEW log file with full diagnostic/system variables enabled.
#    - Fluorometer functions: Full fluorometer.
#    - Log Options > Flr Action at Log:
#        1: FoFm (dark) or FsFm' (light).
#    - Log Options > Flash type: MultiPhase.
#    - Log Options > Minimum flash interval: 60 s.
#    - Suppress automatic CO2 and H2O matching during this short test.
#    - Enable Dynamic Assimilation and use the flow-600 DAT calibration.
#    - Confirm the humidifier reservoir and desiccant are serviceable.
# =====================================================================

from bpdefs import (ASSIGN, DIALOG, EXEC, IF, LOOP, SETCONTROL, LOG,
                    RETURN, EditBox)


H2O_CTL = "licor/li6850/scripts/h2o control/constants"
H2O_OVERRIDE = "licor/li6850/scripts/h2o control/override"


def ctl_snapshot(phase):
    """Build an EXEC step that writes controller state to the BP run log."""
    source = (
        "BP.log('H2ODIAG|{phase}|t=%.1f|ctl=%s|override=%s' % "
        "(t, str(BP.li6850.Latest('{ctl}', default={{}})), "
        "str(BP.li6850.Latest('{override}', default={{}}))))"
    ).format(phase=phase, ctl=H2O_CTL, override=H2O_OVERRIDE)
    return EXEC(0, source=source)


def telemetry(phase):
    """Log controller state at roughly five-second intervals."""
    return IF("t >= ctlNext", steps=(
        ctl_snapshot(phase),
        EXEC(0, source="ctlNext = ctlNext + 5"),
    ))


steps = [

    # ---- operator parameters ----------------------------------------
    ASSIGN("Qtest", exp="800",
           dlg=EditBox("'Diagnostic light'", units="'umol m-2 s-1'",
                       desc="'Use 800 to reproduce the failed middle-Q ramps'")),
    ASSIGN("cStart", exp="120",
           dlg=EditBox("'Ramp start CO2'", units="'ppm'",
                       desc="'Low-CO2 park and ramp start'")),
    ASSIGN("cEnd", exp="720",
           dlg=EditBox("'Ramp end CO2'", units="'ppm'",
                       desc="'Three minutes at the default 200 ppm/min'")),
    ASSIGN("rate", exp="200",
           dlg=EditBox("'Ramp rate'", units="'ppm min-1'",
                       desc="'Match the production DAT ramp rate'")),
    ASSIGN("leafTemp", exp="25",
           dlg=EditBox("'Leaf temperature'", units="'C'",
                       desc="'Tleaf control target'")),
    ASSIGN("vpdLeaf", exp="1.0",
           dlg=EditBox("'Leaf VPD'", units="'kPa'",
                       desc="'VPD_leaf target throughout the diagnostic'")),

    DIALOG(title="'H2O / flash actuator diagnostic'",
           sub="'About 16 minutes: no-flash ramp vs Default MPF ramp'",
           text="'Stop all other BPs and Auto Control. Open a new log. Set action 1, flash type MultiPhase, and a 60-s minimum interval.'",
           items="Qtest, cStart, cEnd, rate, leafTemp, vpdLeaf",
           buttons="'Cancel','Continue'",
           var="button"),
    IF("button == 'Cancel'", steps=(
        RETURN(),
    )),

    # ---- derived values and controller telemetry subscriptions ------
    ASSIGN("rampSec", exp="(cEnd - cStart) / rate * 60.0"),
    EXEC(0, source="BP.subscribe('licor/li6850/scripts/h2o control/constants', {})"),
    EXEC(0, source="BP.subscribe('licor/li6850/scripts/h2o control/override', {})"),
    EXEC(0, source="BP.setLogBuffering(100)"),

    # ---- chamber state ----------------------------------------------
    SETCONTROL("Flow On/Off", "On", ""),
    SETCONTROL("Flow", "600", "float"),
    SETCONTROL("Fan On/Off", "On", ""),
    SETCONTROL("Fan_rpm", "10000", "float"),
    SETCONTROL("CO2 On/Off", "On", ""),
    SETCONTROL("CO2_r", "cStart", "float"),
    SETCONTROL("Temp On/Off", "On", ""),
    SETCONTROL("Tleaf", "leafTemp", "float"),
    SETCONTROL("H2O On/Off", "On", ""),
    SETCONTROL("VPD_leaf", "vpdLeaf", "float"),
    SETCONTROL("Qin", "Qtest", "float"),
    SETCONTROL("Meas:Modulation", "On", ""),

    # ---- EQUIL: controller-only baseline ----------------------------
    EXEC(0, source="BP.log('H2ODIAG|BEGIN|EQUIL')"),
    ASSIGN("ctlNext", exp="0"),
    LOOP(dur="240", units="Seconds", var="t", mininc="0", steps=(
        LOG(flr="0: Nothing"),
        telemetry("EQUIL"),
    )),

    # ---- A: same CO2 ramp, fluorometer explicitly disabled ----------
    EXEC(0, source="BP.log('H2ODIAG|BEGIN|A_NOFLASH')"),
    ASSIGN("ctlNext", exp="0"),
    LOOP(dur="rampSec", units="Seconds", var="t", mininc="0", steps=(
        SETCONTROL("CO2_r", "cStart + rate / 60.0 * t", "float"),
        LOG(flr="0: Nothing"),
        telemetry("A_NOFLASH"),
    )),

    # ---- recover and re-establish VPD control -----------------------
    SETCONTROL("CO2_r", "cStart", "float"),
    SETCONTROL("H2O On/Off", "On", ""),
    SETCONTROL("VPD_leaf", "vpdLeaf", "float"),
    EXEC(0, source="BP.log('H2ODIAG|BEGIN|RECOVER_1')"),
    ASSIGN("ctlNext", exp="0"),
    LOOP(dur="120", units="Seconds", var="t", mininc="0", steps=(
        LOG(flr="0: Nothing"),
        telemetry("RECOVER_1"),
    )),

    # ---- B: identical ramp using the production Default action ------
    EXEC(0, source="BP.log('H2ODIAG|BEGIN|B_DEFAULT')"),
    ASSIGN("ctlNext", exp="0"),
    LOOP(dur="rampSec", units="Seconds", var="t", mininc="0", steps=(
        SETCONTROL("CO2_r", "cStart + rate / 60.0 * t", "float"),
        LOG(flr="Default"),
        telemetry("B_DEFAULT"),
    )),

    # ---- recover before a single explicit flash --------------------
    SETCONTROL("CO2_r", "cStart", "float"),
    SETCONTROL("H2O On/Off", "On", ""),
    SETCONTROL("VPD_leaf", "vpdLeaf", "float"),
    EXEC(0, source="BP.log('H2ODIAG|BEGIN|RECOVER_2')"),
    ASSIGN("ctlNext", exp="0"),
    LOOP(dur="120", units="Seconds", var="t", mininc="0", steps=(
        LOG(flr="0: Nothing"),
        telemetry("RECOVER_2"),
    )),

    # ---- C: one MPF, followed by no-flash monitoring ----------------
    EXEC(0, source="BP.log('H2ODIAG|BEGIN|C_ONE_MPF')"),
    LOG(flr="1: FoFm (dark) or FsFm' (light)"),
    ASSIGN("ctlNext", exp="0"),
    LOOP(dur="120", units="Seconds", var="t", mininc="0", steps=(
        LOG(flr="0: Nothing"),
        telemetry("C_ONE_MPF"),
    )),

    # ---- safe endpoint ----------------------------------------------
    EXEC(0, source="BP.log('H2ODIAG|END')"),
    SETCONTROL("CO2_r", "420", "float"),
    SETCONTROL("Qin", "0", "float"),
]
