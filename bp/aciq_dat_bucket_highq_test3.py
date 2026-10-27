# =====================================================================
#  aciq_dat_bucket_highq.py -- LI-6800 Background Program
#
#  Rainy-day cut-in-bucket A/Ci/Q DAT run.
#
#  Strategy:
#    1) Clean dark-adapted fluorescence baseline (FoFm).
#    2) Dark CO2-response slice at Q=0 (diagnostic; not an FvCB curve).
#    3) Light ladder: at each Q, park at low CO2, then one ascending DAT
#       ramp. The first light level gets a longer low-CO2 park because
#       induction and stomatal opening are least complete immediately
#       after dark adaptation.
#
#  Console setup before launch:
#    - Start Up > Chamber setup > Fluorometer functions as: Full fluorometer.
#    - Enable Dynamic Assimilation in gas-exchange computations.
#    - Dynamic tuning + range match done AT FLOW 900.
#    - Log Setup > Fluorometer Options:
#        Flr Action at Log = 2; MultiPhase; Minimum flash interval = 60 s.
#    - Humidifier reservoir full/seated; confirm Tleaf holds 25 C and
#      VPDleaf holds ~1.0 kPa live.
#    - Open a log file before starting.
#
#  Notes:
#    - This program uses VPD_leaf control directly. If your firmware/Builder
#      rejects that target, set VPDleaf = 1.0 kPa manually before launch and
#      comment out the SETCONTROL("VPD_leaf", ...) line below.
#    - Flow is fixed at 900 because DAT coefficients are flow-specific.
#    - Q = 0 is a diagnostic dark CO2-response slice, not a photosynthetic
#      FvCB capacity curve.
#    - Ramps are low-to-high only: no hysteresis/down-ramp check.
# =====================================================================

from bpdefs import (ASSIGN, DIALOG, LOOP, SETCONTROL, WAIT, LOG,
                    EditBox)

steps = [

    # ---- operator parameters ----------------------------------------
    ASSIGN("Qstr",     exp="'200,500,800,1100,1300,1700,2000'", dlg=EditBox(label="Light ladder, ascending (umol)")),
    ASSIGN("cStart",   exp="120",  dlg=EditBox(label="Ramp start / park CO2 (ppm)")),
    ASSIGN("cEnd",     exp="2000", dlg=EditBox(label="Ramp end CO2 (ppm)")),
    ASSIGN("rate",     exp="200",  dlg=EditBox(label="Ramp rate (ppm/min)")),
    ASSIGN("leafTemp", exp="25",   dlg=EditBox(label="Tleaf setpoint (C)")),
    ASSIGN("vpdLeaf",  exp="1.0",  dlg=EditBox(label="VPD_leaf setpoint (kPa)")),
    ASSIGN("darkMin",  exp="30",   dlg=EditBox(label="Dark-adapt (min)")),
    ASSIGN("parkFirstMin", exp="18", dlg=EditBox(label="Long park at FIRST Q (min)")),
    ASSIGN("parkMin",  exp="5",    dlg=EditBox(label="Park at later Q levels (min)")),
    ASSIGN("setSec",   exp="90",   dlg=EditBox(label="Settle at ramp start (s)")),

    DIALOG(title="A/Ci/Q bucket DAT high-Q",
           subtitle="Dark baseline, dark slice, long first-Q park, ascending ladder. Flow fixed 900.",
           grid=["Qstr", "cStart", "cEnd", "rate", "leafTemp", "vpdLeaf",
                 "darkMin", "parkFirstMin", "parkMin", "setSec"],
           buttons=["Continue", "Cancel"],
           result="btn"),

    # ---- derived values ---------------------------------------------
    ASSIGN("Qlad",    exp="[float(s) for s in Qstr.replace(' ','').split(',') if s]"),
    ASSIGN("rampSec", exp="(cEnd - cStart) / rate * 60.0"),

    # ---- fixed chamber state ----------------------------------------
    # Setpoints and controller power are separate LI-6800 controls.
    # Initialize both so the BP never inherits an H2O-off console state.
    SETCONTROL("Flow On/Off", "On", ""),
    SETCONTROL("Flow", "900", "float"),
    SETCONTROL("CO2 On/Off", "On", ""),
    SETCONTROL("Temp On/Off", "On", ""),
    SETCONTROL("Tleaf", "leafTemp", "float"),
    SETCONTROL("H2O On/Off", "On", ""),
    SETCONTROL("VPD_leaf", "vpdLeaf", "float"),
    SETCONTROL("Meas:Modulation", "On", ""),

    # ---- 1) dark-adapted baseline at normal CO2 ---------------------
    SETCONTROL("CO2_r", "420", "float"),
    SETCONTROL("Qin", "0", "float"),
    WAIT(dur="darkMin", units="Minutes"),
    LOG(flr="1: FoFm (dark) or FsFm' (light)"),

    # ---- 2) dark CO2-response slice (Q=0) ---------------------------
    SETCONTROL("CO2_r", "cStart", "float"),
    WAIT(dur="setSec", units="Seconds"),
    LOOP(dur="rampSec", units="Seconds", var="t", mininc="0", steps=(
        SETCONTROL("CO2_r", "cStart + rate / 60.0 * t", "float"),
        LOG(flr="0: Nothing"),
    )),
    SETCONTROL("CO2_r", "cStart", "float"),

    # ---- 3) FIRST light level: long park + ascending ramp -----------
    SETCONTROL("Qin", "Qlad[0]", "float"),
    SETCONTROL("CO2_r", "cStart", "float"),
    WAIT(dur="parkFirstMin", units="Minutes"),
    WAIT(dur="setSec", units="Seconds"),
    LOOP(dur="rampSec", units="Seconds", var="t", mininc="0", steps=(
        SETCONTROL("CO2_r", "cStart + rate / 60.0 * t", "float"),
        LOG(flr="Default"),
    )),
    SETCONTROL("CO2_r", "cStart", "float"),

    # ---- 4) remaining ladder: park + ascending ramp -----------------
    LOOP(list="Qlad[1:]", var="Q", steps=(
        SETCONTROL("Qin", "Q", "float"),
        SETCONTROL("CO2_r", "cStart", "float"),
        WAIT(dur="parkMin", units="Minutes"),
        WAIT(dur="setSec", units="Seconds"),

        LOOP(dur="rampSec", units="Seconds", var="t", mininc="0", steps=(
            SETCONTROL("CO2_r", "cStart + rate / 60.0 * t", "float"),
            LOG(flr="Default"),
        )),

        SETCONTROL("CO2_r", "cStart", "float"),
    )),

    # ---- 5) park safely ---------------------------------------------
    SETCONTROL("CO2_r", "420", "float"),
    SETCONTROL("Qin", "0", "float"),
]
