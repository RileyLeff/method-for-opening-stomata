# =====================================================================
#  aciq_dat_fixed_h2or_async_mpf_flow600.py -- LI-6800 Background Program
#
#  A/Ci ramps across Q with one fixed incoming-moisture condition. H2O_r
#  is set once at startup and remains fixed through dark adaptation, the
#  low-CO2 parks, and every CO2 ramp. The default 21.0 mmol mol-1 is based
#  on the flow-600 diagnostic at Tleaf=25 C and near-sea-level pressure.
#
#  Gas-exchange LOG steps explicitly request no fluorometer action, so
#  they cannot block the CO2 control loop. MPFs are triggered asynchronously
#  every 60 s through the documented fluorometer MQTT action topic.
#
#  The BP refuses to start unless the LI-6800 reports an active log file.
#  It immediately writes one no-flash test observation and verifies that the
#  observation count increased. On normal completion, it closes the file so
#  the data are finalized before viewing or export.
#
#  This gives every leaf the same incoming water-vapor boundary condition,
#  prevents the controller from chasing MPF heat pulses, and preserves the
#  commanded CO2 ramp trajectory. Actual H2O_s and VPDleaf remain measured
#  outcomes and may vary with leaf transpiration.
#
#  Default sequence:
#    1) Establish CO2_r=420, Q=0, Tleaf=25 C, H2O_r=21.0 mmol mol-1.
#    2) Dark adapt for 30 min and log one dark-adapted flash.
#    3) Run a Q=0 diagnostic CO2 ramp with no light-adapted flashes.
#    4) Run ramps at Q = 200, 500, 800, 1100, 1300, 1700, and 2000.
#       Q=200 receives an 18-min low-CO2 park; later Q levels get 5 min.
#    5) Restore CO2_r=420 and Q=0; fixed H2O_r remains active.
#
#  Expected runtime with defaults: approximately 2 hours 50 minutes.
#
#  Console setup before launch:
#    - Stop every other BP, Program, Auto Control, and system test.
#    - Open a NEW log file with the full diagnostic/system fields.
#    - Enable Dynamic Assimilation and use the flow-600 DAT calibration.
#    - Suppress automatic CO2 and H2O matching during the ramps.
#    - Chamber setup > Fluorometer functions as: Full fluorometer.
#    - Configure and optimize the MultiPhase flash before launch.
#    - Log Setup > Fluorometer action at log: 0: Nothing. The BP triggers
#      MPFs directly; the console's minimum-flash interval is not used.
#    - Confirm the intended light spectrum, normally R90B40.
#    - Fill and seat the humidifier reservoir; check the desiccant.
#
#  Analysis note:
#    The physical MPF still heats the leaf. Join flash files by timestamp
#    and flag post-flash gas-exchange rows (initially use 25 s). The ramp
#    itself should remain continuous because the BP never waits for MPFs.
# =====================================================================

from bpdefs import (ASSIGN, DIALOG, EXEC, IF, LOOP, SETCONTROL, WAIT, LOG,
                    RETURN, EditBox)


def dat_ramp(label, with_flashes):
    """Build a continuous DAT ramp with optional nonblocking MPFs."""
    ramp_steps = [
        SETCONTROL("CO2_r", "cStart + rate / 60.0 * t", "float"),
        LOG(flr="0: Nothing"),
    ]
    if with_flashes:
        ramp_steps.append(
            IF("t >= nextFlash", steps=(
                EXEC(
                    0,
                    source=(
                        "BP.publish('licor/li6850/scripts/flr/action', "
                        "{'cmd':'mpf'})"
                    ),
                ),
                EXEC(0, source="nextFlash = t + flashSec"),
            ))
        )

    return (
        EXEC(0, source="BP.log('ACIQ|RAMP_BEGIN|{0}')".format(label)),
        ASSIGN("nextFlash", exp="flashStartSec"),
        LOOP(
            dur="rampSec",
            units="Seconds",
            var="t",
            mininc="0",
            steps=tuple(ramp_steps),
        ),
        EXEC(0, source="BP.log('ACIQ|RAMP_END|{0}')".format(label)),
    )


steps = [

    # ---- operator parameters ----------------------------------------
    ASSIGN("Qstr", exp="'200,500,800,1100,1300,1700,2000'",
           dlg=EditBox("'Light ladder'", units="'umol m-2 s-1'",
                       desc="'Q=0 is measured separately as a dark diagnostic'")),
    ASSIGN("cStart", exp="120",
           dlg=EditBox("'Ramp start / park CO2'", units="'ppm'",
                       desc="'Low CO2 used during stomatal-opening parks'")),
    ASSIGN("cEnd", exp="2000",
           dlg=EditBox("'Ramp end CO2'", units="'ppm'",
                       desc="'Final reference CO2 setpoint'")),
    ASSIGN("rate", exp="200",
           dlg=EditBox("'Ramp rate'", units="'ppm min-1'",
                       desc="'DAT CO2 ramp rate'")),
    ASSIGN("leafTemp", exp="25",
           dlg=EditBox("'Leaf temperature'", units="'C'",
                       desc="'Tleaf control target'")),
    ASSIGN("h2oR", exp="21.0",
           dlg=EditBox("'Fixed reference H2O'", units="'mmol mol-1'",
                       desc="'One incoming-moisture target for the entire run'")),
    ASSIGN("startupMin", exp="5",
           dlg=EditBox("'Initial controller wait'", units="'min'",
                       desc="'Controller equilibration before dark adaptation'")),
    ASSIGN("darkMin", exp="30",
           dlg=EditBox("'Dark adaptation'", units="'min'",
                       desc="'Dark period before FoFm'")),
    ASSIGN("parkFirstMin", exp="18",
           dlg=EditBox("'Q=200 low-CO2 park'", units="'min'",
                       desc="'Long first-light opening and induction park'")),
    ASSIGN("parkMin", exp="5",
           dlg=EditBox("'Later low-CO2 parks'", units="'min'",
                       desc="'Acclimation at later Q levels'")),
    ASSIGN("setSec", exp="90",
           dlg=EditBox("'Pre-ramp settle'", units="'s'",
                       desc="'Additional settle after each Q/CO2 park'")),
    ASSIGN("flashStartSec", exp="0",
           dlg=EditBox("'First MPF time'", units="'s into ramp'",
                       desc="'A no-flash gas log is made before this MPF is triggered'")),
    ASSIGN("flashSec", exp="60",
           dlg=EditBox("'Async MPF interval'", units="'s'",
                       desc="'Direct MPF trigger interval; does not block the ramp'")),

    DIALOG(title="'A/Ci/Q DAT: fixed H2O_r + async MPF'",
           sub="'One fixed H2O_r for the entire flow-600 run'",
           text="'Stop other BPs and Auto Control. Open a new log. Enable DAT calibrated at flow 600. Set fluorometer action at log to Nothing.'",
           items=(
               "Qstr, cStart, cEnd, rate, leafTemp, h2oR, startupMin, "
               "darkMin, parkFirstMin, parkMin, setSec, flashStartSec, flashSec"
           ),
           buttons="'Cancel','Continue'",
           var="button"),
    IF("button == 'Cancel'", steps=(
        RETURN(),
    )),

    # ---- fail-fast datalogger check ---------------------------------
    ASSIGN("logOpen", exp="bool(BP.getSD('LOG:IsFileOpen'))"),
    IF("not logOpen", steps=(
        DIALOG(
            title="'NO ACTIVE LOG FILE'",
            sub="'The A/Ci/Q run has not started'",
            text=(
                "'Open a measurement file using Log Files > Open a Log File > "
                "New File, confirm the upper-right button reads Log (0), then "
                "launch this BP again.'"
            ),
            buttons="['Stop']",
            var="logGuardAck",
        ),
        RETURN(),
    )),
    ASSIGN("logFile", exp="str(BP.getSD('LOG:FileName'))"),
    ASSIGN("obsBefore", exp="int(BP.getSD('LOG:ObsCount'))"),
    LOG(flr="0: Nothing"),
    WAIT(dur="2", units="Seconds"),
    ASSIGN("obsAfter", exp="int(BP.getSD('LOG:ObsCount'))"),
    IF("obsAfter <= obsBefore", steps=(
        DIALOG(
            title="'LOG WRITE TEST FAILED'",
            sub="'The A/Ci/Q run has not started'",
            text=(
                "'A file appears open, but a no-flash test observation did not "
                "increase its observation count. Close the file, verify a manual "
                "log increments Log (#), and inspect free storage before retrying.'"
            ),
            buttons="['Stop']",
            var="logTestAck",
        ),
        RETURN(),
    )),
    EXEC(
        0,
        source=(
            "BP.log('ACIQ|LOGGER_READY|file=%s|obs=%d' % "
            "(logFile, obsAfter))"
        ),
    ),
    EXEC(
        0,
        source=(
            "BP.publish('licor/li6850/scripts/datalogger/constants/update', "
            "{'log_remark':'ACIQ fixed-H2O_r DAT run started'})"
        ),
    ),

    # ---- derived values ---------------------------------------------
    ASSIGN("Qlad", exp="[float(s) for s in Qstr.replace(' ','').split(',') if s]"),
    ASSIGN("rampSec", exp="(cEnd - cStart) / rate * 60.0"),

    # ---- fixed chamber configuration --------------------------------
    SETCONTROL("Flow On/Off", "On", ""),
    SETCONTROL("Flow", "600", "float"),
    SETCONTROL("Fan On/Off", "On", ""),
    SETCONTROL("Fan_rpm", "10000", "float"),
    SETCONTROL("CO2 On/Off", "On", ""),
    SETCONTROL("Temp On/Off", "On", ""),
    SETCONTROL("Tleaf", "leafTemp", "float"),
    SETCONTROL("H2O On/Off", "On", ""),
    SETCONTROL("H2O_r", "h2oR", "float"),
    SETCONTROL("Meas:Modulation", "On", ""),
    EXEC(0, source="BP.log('ACIQ|FIXED_H2O_R|%.3f' % h2oR)"),

    # ---- 1) dark adaptation at ambient CO2 --------------------------
    SETCONTROL("CO2_r", "420", "float"),
    SETCONTROL("Qin", "0", "float"),
    WAIT(dur="startupMin", units="Minutes"),
    WAIT(dur="darkMin", units="Minutes"),
    LOG(flr="1: FoFm (dark) or FsFm' (light)"),

    # ---- 2) Q=0 diagnostic ramp -------------------------------------
    SETCONTROL("CO2_r", "cStart", "float"),
    WAIT(dur="setSec", units="Seconds"),
    *dat_ramp("Q0", with_flashes=False),

    # ---- 3) first light level: long Q=200 opening park --------------
    SETCONTROL("CO2_r", "cStart", "float"),
    SETCONTROL("Qin", "Qlad[0]", "float"),
    WAIT(dur="parkFirstMin", units="Minutes"),
    WAIT(dur="setSec", units="Seconds"),
    *dat_ramp("Q200", with_flashes=True),

    # ---- 4) remaining light ladder ----------------------------------
    LOOP(list="Qlad[1:]", var="Q", steps=(
        SETCONTROL("CO2_r", "cStart", "float"),
        SETCONTROL("Qin", "Q", "float"),
        WAIT(dur="parkMin", units="Minutes"),
        WAIT(dur="setSec", units="Seconds"),

        EXEC(0, source="BP.log('ACIQ|Q_LEVEL|%.0f|H2O_r=%.3f' % (Q, h2oR))"),
        *dat_ramp("Q_DYNAMIC", with_flashes=True),
    )),

    # ---- 5) safe endpoint -------------------------------------------
    SETCONTROL("CO2_r", "420", "float"),
    SETCONTROL("Qin", "0", "float"),
    ASSIGN("obsFinal", exp="int(BP.getSD('LOG:ObsCount'))"),
    EXEC(
        0,
        source=(
            "BP.log('ACIQ|END|file=%s|obs=%d' % (logFile, obsFinal))"
        ),
    ),
    EXEC(
        0,
        source=(
            "BP.publish('licor/li6850/scripts/datalogger/constants/update', "
            "{'log_remark':'ACIQ run complete; closing measurement file'})"
        ),
    ),
    WAIT(dur="1", units="Seconds"),
    EXEC(
        0,
        source=(
            "BP.publish('licor/li6850/scripts/datalogger/constants/update', "
            "{'file_close':' '})"
        ),
    ),
    WAIT(dur="2", units="Seconds"),
    ASSIGN("logStillOpen", exp="bool(BP.getSD('LOG:IsFileOpen'))"),
    IF("logStillOpen", steps=(
        DIALOG(
            title="'LOG FILE DID NOT CLOSE'",
            sub="'Measurements are complete'",
            text=(
                "'The BP requested a clean close, but the logger still reports "
                "an active file. Close it manually before viewing or export.'"
            ),
            buttons="['OK']",
            var="logCloseAck",
        ),
    )),
]
