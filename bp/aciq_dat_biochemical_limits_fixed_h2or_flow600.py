# =====================================================================
#  aciq_dat_biochemical_limits_fixed_h2or_flow600.py
#  LI-6800 Background Program
#
#  A/Ci ramps across a Q ladder designed to identify the low-light slope,
#  J(Q) curvature, Jmax, apparent Vcmax, and high-CO2 behavior. The leaf is
#  dark-adapted outside the chamber before launch. The BP then provides a
#  short dark equilibration, records dark gas exchange, runs one FoFm flash,
#  opens/induces the leaf at low CO2 and Q=600, and measures nine A/Ci ramps.
#
#  H2O_r is fixed at 21 mmol mol-1 for the entire run. At flow 600 and
#  Tleaf=25 C, this held VPDleaf close to 1 kPa in the July 21 diagnostic.
#  H2O_s and VPDleaf remain measured outcomes.
#
#  Default measurement grid:
#    Q = 50, 100, 200, 350, 600, 1000, 1400, 1700, 2000
#    CO2_r = 120 to 2000 ppm at 200 ppm min-1
#
#  Sequence:
#    1) CO2_r=420, Q=0, Tleaf=25 C, H2O_r=21; equilibrate for 5 min.
#    2) Log dark gas exchange for 60 s and run one dark-adapted MPF.
#    3) Set CO2_r=120 and Q=600 for an 18-min opening/induction period.
#    4) Set Q=50 and park at low CO2 for 10 min before the first ramp.
#    5) At later Q levels, park at low CO2 for 5 min before each ramp.
#    6) During each ramp, preserve the first 60 s as no-flash low-Ci data,
#       then trigger asynchronous MPFs every 90 s.
#    7) Restore CO2_r=420 and Q=0, then close the measurement file.
#
#  Expected runtime with defaults: approximately 2 hours 53 minutes.
#
#  Console setup before launch:
#    - Dark-adapt the leaf outside the chamber for at least 30 min.
#    - Transfer and clamp the leaf with minimal light exposure.
#    - Stop every other BP, Program, Auto Control, and system test.
#    - Open a NEW log file with full diagnostic/system fields.
#    - Enable Dynamic Assimilation with the flow-600 DAT calibration.
#    - Suppress automatic CO2 and H2O matching during ramps.
#    - Chamber setup > Fluorometer functions as: Full fluorometer.
#    - Optimize and select the MultiPhase flash configuration.
#    - Log Setup > Fluorometer action at log: 0: Nothing.
#    - Confirm the recommended r90B40 light spectrum.
#    - Fill the humidifier reservoir and inspect the desiccant.
#
#  Analysis note:
#    MPFs are nonblocking for the programmed CO2 ramp but still heat and
#    perturb the leaf. Join flash files by timestamp and flag post-flash
#    gas-exchange rows. The first 60 s of every ramp are intentionally free
#    of active flashes to protect the low-Ci Vcmax information.
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
        EXEC(0, source="BP.log('ACIQBIO|RAMP_BEGIN|{0}')".format(label)),
        ASSIGN("nextFlash", exp="flashStartSec"),
        LOOP(
            dur="rampSec",
            units="Seconds",
            var="t",
            mininc="0",
            steps=tuple(ramp_steps),
        ),
        EXEC(0, source="BP.log('ACIQBIO|RAMP_END|{0}')".format(label)),
    )


steps = [

    # ---- operator parameters ----------------------------------------
    ASSIGN("Qstr", exp="'50,100,200,350,600,1000,1400,1700,2000'",
           dlg=EditBox("'Measured Q ladder'", units="'umol m-2 s-1'",
                       desc="'Nine levels spanning initial slope through continuous-light maximum'")),
    ASSIGN("qOpen", exp="600",
           dlg=EditBox("'Opening / induction Q'", units="'umol m-2 s-1'",
                       desc="'Applied at low CO2 after the dark flash'")),
    ASSIGN("cStart", exp="120",
           dlg=EditBox("'Ramp start / park CO2'", units="'ppm'",
                       desc="'Low CO2 used during opening and pre-ramp parks'")),
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
    ASSIGN("darkEqMin", exp="5",
           dlg=EditBox("'In-chamber dark equilibration'", units="'min'",
                       desc="'Leaf must already be dark-adapted before clamping'")),
    ASSIGN("openMin", exp="18",
           dlg=EditBox("'Low-CO2 Q600 induction'", units="'min'",
                       desc="'Opens stomata and completes light induction'")),
    ASSIGN("firstParkMin", exp="10",
           dlg=EditBox("'First Q low-CO2 park'", units="'min'",
                       desc="'Allows relaxation after Q600 induction'")),
    ASSIGN("parkMin", exp="5",
           dlg=EditBox("'Later low-CO2 parks'", units="'min'",
                       desc="'Acclimation at later measured Q levels'")),
    ASSIGN("setSec", exp="90",
           dlg=EditBox("'Pre-ramp settle'", units="'s'",
                       desc="'Additional settle after each low-CO2 park'")),
    ASSIGN("flashStartSec", exp="60",
           dlg=EditBox("'First active MPF time'", units="'s into ramp'",
                       desc="'Protects the first 60 s of low-Ci gas exchange'")),
    ASSIGN("flashSec", exp="90",
           dlg=EditBox("'Async MPF interval'", units="'s'",
                       desc="'Six active flashes per default ramp'")),

    DIALOG(title="'A/Ci/Q biochemical limits: fixed H2O_r'",
           sub="'External dark adaptation; Q600 induction; nine measured Q levels'",
           text="'Open a new log. Enable flow-600 DAT. Set fluorometer action at log to Nothing. Confirm r90B40.'",
           items=(
               "Qstr, qOpen, cStart, cEnd, rate, leafTemp, h2oR, "
               "darkEqMin, openMin, firstParkMin, parkMin, setSec, "
               "flashStartSec, flashSec"
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
            sub="'The biochemical-limits run has not started'",
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
            sub="'The biochemical-limits run has not started'",
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
            "BP.log('ACIQBIO|LOGGER_READY|file=%s|obs=%d' % "
            "(logFile, obsAfter))"
        ),
    ),
    EXEC(
        0,
        source=(
            "BP.publish('licor/li6850/scripts/datalogger/constants/update', "
            "{'log_remark':'A/Ci/Q biochemical-limits run started'})"
        ),
    ),

    # ---- derived values ---------------------------------------------
    ASSIGN("Qlad", exp="[float(s) for s in Qstr.replace(' ','').split(',') if s]"),
    ASSIGN("rampSec", exp="(cEnd - cStart) / rate * 60.0"),
    ASSIGN("openSec", exp="openMin * 60.0"),

    # ---- fixed chamber configuration -------------------------------
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
    EXEC(0, source="BP.log('ACIQBIO|FIXED_H2O_R|%.3f' % h2oR)"),

    # ---- 1) externally dark-adapted leaf: chamber equilibration -----
    SETCONTROL("CO2_r", "420", "float"),
    SETCONTROL("Qin", "0", "float"),
    WAIT(dur="darkEqMin", units="Minutes"),
    EXEC(0, source="BP.log('ACIQBIO|DARK_BASELINE_BEGIN')"),
    LOOP(dur="60", units="Seconds", var="dark_t", mininc="10", steps=(
        LOG(flr="0: Nothing"),
    )),
    EXEC(0, source="BP.log('ACIQBIO|DARK_FLASH')"),
    LOG(flr="1: FoFm (dark) or FsFm' (light)"),

    # ---- 2) low-CO2 stomatal opening and light induction ------------
    SETCONTROL("CO2_r", "cStart", "float"),
    SETCONTROL("Qin", "qOpen", "float"),
    EXEC(
        0,
        source=(
            "BP.log('ACIQBIO|OPEN_BEGIN|Q=%.0f|CO2=%.0f' % "
            "(qOpen, cStart))"
        ),
    ),
    LOOP(dur="openSec", units="Seconds", var="open_t", mininc="10", steps=(
        LOG(flr="0: Nothing"),
    )),
    EXEC(0, source="BP.log('ACIQBIO|OPEN_END')"),

    # ---- 3) first measured Q: longer low-CO2 relaxation park --------
    SETCONTROL("CO2_r", "cStart", "float"),
    SETCONTROL("Qin", "Qlad[0]", "float"),
    WAIT(dur="firstParkMin", units="Minutes"),
    WAIT(dur="setSec", units="Seconds"),
    EXEC(
        0,
        source=(
            "BP.log('ACIQBIO|Q_LEVEL|%.0f|H2O_r=%.3f' % "
            "(Qlad[0], h2oR))"
        ),
    ),
    *dat_ramp("Q_FIRST", with_flashes=True),

    # ---- 4) remaining measured Q levels -----------------------------
    LOOP(list="Qlad[1:]", var="Q", steps=(
        SETCONTROL("CO2_r", "cStart", "float"),
        SETCONTROL("Qin", "Q", "float"),
        WAIT(dur="parkMin", units="Minutes"),
        WAIT(dur="setSec", units="Seconds"),
        EXEC(
            0,
            source=(
                "BP.log('ACIQBIO|Q_LEVEL|%.0f|H2O_r=%.3f' % "
                "(Q, h2oR))"
            ),
        ),
        *dat_ramp("Q_DYNAMIC", with_flashes=True),
    )),

    # ---- 5) safe endpoint and clean file close ----------------------
    SETCONTROL("CO2_r", "420", "float"),
    SETCONTROL("Qin", "0", "float"),
    ASSIGN("obsFinal", exp="int(BP.getSD('LOG:ObsCount'))"),
    EXEC(
        0,
        source=(
            "BP.log('ACIQBIO|END|file=%s|obs=%d' % "
            "(logFile, obsFinal))"
        ),
    ),
    EXEC(
        0,
        source=(
            "BP.publish('licor/li6850/scripts/datalogger/constants/update', "
            "{'log_remark':'A/Ci/Q biochemical-limits run complete; closing file'})"
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
