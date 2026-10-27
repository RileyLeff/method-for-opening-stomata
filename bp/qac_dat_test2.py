# =====================================================================
#  AciQ_DAT.py   —  LI-6800 Background Program  (DYNAMIC ASSIMILATION)
#
#  Hold stomata open by parking at low CO2, then measure each A/Ci as a
#  fast ASCENDING CO2 ramp with DAT (storage-corrected A) at every Q
#  level.  Parking low between levels keeps stomata from reclosing, so
#  the per-level overhead collapses to Q-acclimation + the ramp itself.
#
#  An opening DIALOG sets Q levels, the ramp, and timings at launch.
#
#  ---- CONSOLE SETUP (do these on the console, NOT in the BP) ----------
#   Start Up > Chamber setup > Fluorometer functions as: Full fluorometer
#   ENABLE Dynamic Assimilation in the gas-exchange computations, so A is
#     storage-corrected during the ramp.  (This BP only drives CO2_r and
#     logs fast; DAT does the non-steady-state math.)
#   Log Setup > Fluorometer Options: Flr Action at Log = 2; MultiPhase;
#     Minimum flash interval = 60 s  (a few clean flashes per ramp).
#   Run HIGH FLOW for a sharp gas-phase response; range-match 70-2000;
#   open a log file before starting.
#
#   ---- VERIFY-IN-BUILDER (firmware-specific) --------------------------
#   * EditBox(...) and DIALOG(...) keyword names (build one in the Builder).
#   * LOG flr= strings must match your Flr Action menu text.
#   * Confirm var=t in LOOP[dur] is ELAPSED SECONDS on your firmware --
#     the ramp expression assumes seconds.
# =====================================================================

from bpdefs import (ASSIGN, DIALOG, LOOP, SETCONTROL, WAIT, LOG,
                    Nothing, EditBox)

steps = [

    # ---- operator parameters (opening dialog) -----------------------
    ASSIGN("Qstr",    exp="'0,200,400,700,1000,1300,1600'", dlg=EditBox(label="Q levels, ascending (umol)")),
    ASSIGN("cStart",  exp="70",   dlg=EditBox(label="Ramp start / park CO2 (ppm)")),
    ASSIGN("cEnd",    exp="2000", dlg=EditBox(label="Ramp end CO2 (ppm)")),
    ASSIGN("rate",    exp="300",  dlg=EditBox(label="Ramp rate (ppm/min)")),
    ASSIGN("darkMin", exp="30",   dlg=EditBox(label="Dark-adapt (min)")),
    ASSIGN("indMin",  exp="20",   dlg=EditBox(label="Induction at 420 (min)")),
    ASSIGN("qSetMin", exp="5",    dlg=EditBox(label="Acclimation per Q level (min)")),
    ASSIGN("setSec",  exp="120",  dlg=EditBox(label="Settle at ramp start (s)")),

    DIALOG(title="A/Ci/Q  (DAT ramp)",
           subtitle="Set levels and ramp, then Continue",
           grid=["Qstr", "cStart", "cEnd", "rate", "darkMin",
                 "indMin", "qSetMin", "setSec"],
           buttons=["Continue", "Cancel"],
           result="btn"),

    # ---- derived values ---------------------------------------------
    ASSIGN("Qlist",   exp="[float(s) for s in Qstr.replace(' ','').split(',') if s]"),
    ASSIGN("rampSec", exp="(cEnd - cStart) / rate * 60.0"),   # seconds for the full sweep

    # ---- 1) dark-adapted baseline -----------------------------------
    SETCONTROL("Meas:Modulation", "On", ""),
    SETCONTROL("Qin", "0", "float"),
    WAIT(dur="darkMin", units="Minutes"),
    LOG(flr="1: FoFm (dark) or FsFm' (light)"),   # Fv/Fm; seeds the DarkAdapted file

    # ---- 2) induction at working light + ambient CO2 ----------------
    SETCONTROL("CO2_r", "420", "float"),
    SETCONTROL("Qin", "Qlist[1]", "float"),
    WAIT(dur="indMin", units="Minutes"),

    # ---- 3) park low so stomata open, then ramp per Q level ---------
    SETCONTROL("CO2_r", "cStart", "float"),
    LOOP(list="Qlist", var="Q", steps=(
        SETCONTROL("Qin", "Q", "float"),
        WAIT(dur="qSetMin", units="Minutes"),     # induction at this Q; stomata open (low CO2)
        SETCONTROL("CO2_r", "cStart", "float"),
        WAIT(dur="setSec", units="Seconds"),      # settle the reference at ramp start

        # ---- DAT ascending CO2 ramp; log every new data set (2 Hz) ---
        LOOP(dur="rampSec", units="Seconds", var="t", mininc="0", steps=(
            SETCONTROL("CO2_r", "cStart + rate / 60.0 * t", "float"),
            LOG(flr="Default"),                   # DAT computes A; 60-s throttle flashes a subset
        )),

        # ---- OPTIONAL hysteresis check: descending ramp at this Q ----
        # If ascending and descending A/Ci overlap, the ramp rate is honest.
        # LOOP(dur="rampSec", units="Seconds", var="t", mininc="0", steps=(
        #     SETCONTROL("CO2_r", "cEnd - rate / 60.0 * t", "float"),
        #     LOG(flr="0: Nothing"),
        # )),

        SETCONTROL("CO2_r", "cStart", "float"),   # park low into the next Q level
    )),

    # ---- 4) park safely ---------------------------------------------
    SETCONTROL("CO2_r", "420", "float"),
    SETCONTROL("Qin", "0", "float"),
]
