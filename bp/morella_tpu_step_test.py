# =====================================================================
#  morella_tpu_step_test.py -- LI-6800 Background Program
#
#  Fixed-light CO2 step test for the reversible high-CO2 decline seen in
#  Morella cerifera. This is a state/reversibility diagnostic, not an
#  A/Ci ramp and not a conventional steady-state FvCB parameterization.
#
#  Sequence (default settings):
#    1) Dark adapt at Q=0 and CO2_r=420; collect FoFm.
#    2) Open stomata for 18 min at Q=800 and CO2_r=120.
#    3) At Q=1700, step CO2_r through:
#         420 (6 min) -> 750 (8 min) -> 1800 (15 min)
#         -> 750 (10 min) -> 420 (8 min).
#    4) Return to Q=0 and CO2_r=420.
#
#  Gas exchange is logged every second. Active fluorescence flashes are
#  explicit and several minutes apart so they do not become periodic
#  structure in CO2 space. The high-CO2 hold has three flashes to follow
#  PhiPSII, ETR, NPQ, and qL while assimilation changes.
#
#  Console setup before launch:
#    - Open a log file.
#    - Chamber setup > Fluorometer functions as: Full fluorometer.
#    - Configure the light spectrum as R90B40. Qin uses the spectrum that
#      is already configured on the console.
#    - Log Setup > Fluorometer Options:
#        Flr Action at Log = 2: MultiPhase
#        Minimum flash interval = 60 s or less
#    - Enable Dynamic Assimilation; use tuning/range match made at flow 900.
#    - Use the 70-2000 ppm CO2 range match and confirm the CO2 canister can
#      hold the requested high setpoint.
#    - Set system averaging to None/0 for the one-second transition record.
#    - Fill/seat the humidifier reservoir and watch VPDleaf during startup.
#
#  The dialog and control names follow the LI-6800 BP Details syntax. The
#  fluorescence action strings must match the choices on this instrument.
# =====================================================================

from bpdefs import (ASSIGN, DIALOG, IF, LOOP, SETCONTROL, WAIT, LOG,
                    RETURN, EditBox)


steps = [

    # ---- operator parameters ----------------------------------------
    ASSIGN("qOpen", exp="800",
           dlg=EditBox("'Opening light'", units="'umol m-2 s-1'",
                       desc="'Light during the low-CO2 stomatal-opening hold'")),
    ASSIGN("qTest", exp="1700",
           dlg=EditBox("'Test light'", units="'umol m-2 s-1'",
                       desc="'Fixed light during the CO2 step sequence'")),
    ASSIGN("cOpen", exp="120",
           dlg=EditBox("'Opening CO2_r'", units="'ppm'",
                       desc="'Low CO2 used to promote stomatal opening'")),
    ASSIGN("cMid", exp="750",
           dlg=EditBox("'Mid CO2_r'", units="'ppm'",
                       desc="'Reference CO2 near the prior assimilation peak'")),
    ASSIGN("cHigh", exp="1800",
           dlg=EditBox("'High CO2_r'", units="'ppm'",
                       desc="'High-CO2 challenge setpoint'")),
    ASSIGN("leafTemp", exp="25",
           dlg=EditBox("'Leaf temperature'", units="'C'",
                       desc="'Tleaf control setpoint'")),
    ASSIGN("vpdLeaf", exp="1.0",
           dlg=EditBox("'Leaf VPD'", units="'kPa'",
                       desc="'VPD_leaf control setpoint'")),
    ASSIGN("darkMin", exp="30",
           dlg=EditBox("'Dark adaptation'", units="'min'",
                       desc="'Dark-adaptation duration before FoFm'")),
    ASSIGN("openMin", exp="18",
           dlg=EditBox("'Opening hold'", units="'min'",
                       desc="'Duration at opening light and low CO2'")),

    DIALOG(title="'Morella TPU step test'",
           sub="'Dark FoFm, low-CO2 opening, then a fixed-Q CO2 sandwich'",
           text="'Flow is fixed at 900. Confirm R90B40 and open a log file.'",
           items="qOpen, qTest, cOpen, cMid, cHigh, leafTemp, vpdLeaf, darkMin, openMin",
           buttons="'Cancel','Continue'",
           var="button"),
    IF("button == 'Cancel'", steps=(
        RETURN(),
    )),

    ASSIGN("openSec", exp="openMin * 60.0"),

    # ---- fixed chamber state ----------------------------------------
    # Setpoints and controller power are separate LI-6800 controls.
    # Initialize both so the BP never inherits H2O- or temperature-off.
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

    # ---- 2) induce and open stomata without severe high-light stress -
    SETCONTROL("Qin", "qOpen", "float"),
    SETCONTROL("CO2_r", "cOpen", "float"),
    LOOP(dur="openSec", units="Seconds", var="t", mininc="1", steps=(
        LOG(flr="0: Nothing"),
    )),

    # ---- 3a) ambient baseline at test light: 6 min ------------------
    SETCONTROL("Qin", "qTest", "float"),
    SETCONTROL("CO2_r", "420", "float"),
    LOOP(dur="300", units="Seconds", var="t", mininc="1", steps=(
        LOG(flr="0: Nothing"),
    )),
    LOG(flr="Default"),
    LOOP(dur="60", units="Seconds", var="t", mininc="1", steps=(
        LOG(flr="0: Nothing"),
    )),

    # ---- 3b) first mid-CO2 reference: 8 min -------------------------
    SETCONTROL("CO2_r", "cMid", "float"),
    LOOP(dur="360", units="Seconds", var="t", mininc="1", steps=(
        LOG(flr="0: Nothing"),
    )),
    LOG(flr="Default"),
    LOOP(dur="120", units="Seconds", var="t", mininc="1", steps=(
        LOG(flr="0: Nothing"),
    )),

    # ---- 3c) high-CO2 challenge: 15 min, three fluorescence probes --
    SETCONTROL("CO2_r", "cHigh", "float"),
    LOOP(dur="120", units="Seconds", var="t", mininc="1", steps=(
        LOG(flr="0: Nothing"),
    )),
    LOG(flr="Default"),
    LOOP(dur="300", units="Seconds", var="t", mininc="1", steps=(
        LOG(flr="0: Nothing"),
    )),
    LOG(flr="Default"),
    LOOP(dur="420", units="Seconds", var="t", mininc="1", steps=(
        LOG(flr="0: Nothing"),
    )),
    LOG(flr="Default"),
    LOOP(dur="60", units="Seconds", var="t", mininc="1", steps=(
        LOG(flr="0: Nothing"),
    )),

    # ---- 3d) return to identical mid CO2: 10 min --------------------
    SETCONTROL("CO2_r", "cMid", "float"),
    LOOP(dur="120", units="Seconds", var="t", mininc="1", steps=(
        LOG(flr="0: Nothing"),
    )),
    LOG(flr="Default"),
    LOOP(dur="360", units="Seconds", var="t", mininc="1", steps=(
        LOG(flr="0: Nothing"),
    )),
    LOG(flr="Default"),
    LOOP(dur="120", units="Seconds", var="t", mininc="1", steps=(
        LOG(flr="0: Nothing"),
    )),

    # ---- 3e) ambient recovery: 8 min --------------------------------
    SETCONTROL("CO2_r", "420", "float"),
    LOOP(dur="420", units="Seconds", var="t", mininc="1", steps=(
        LOG(flr="0: Nothing"),
    )),
    LOG(flr="Default"),
    LOOP(dur="60", units="Seconds", var="t", mininc="1", steps=(
        LOG(flr="0: Nothing"),
    )),

    # ---- 4) safe endpoint -------------------------------------------
    SETCONTROL("CO2_r", "420", "float"),
    SETCONTROL("Qin", "0", "float"),
]
