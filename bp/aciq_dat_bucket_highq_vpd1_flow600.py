# =====================================================================
#  aciq_dat_bucket_highq_vpd1_flow600.py -- LI-6800 Background Program
#
#  Dynamic A/Ci ramps across Q, with low-CO2 parking to encourage stomatal
#  opening. This revision keeps measured leaf VPD near 1.0 kPa from startup
#  onward and uses the DAT calibration made at flow 600 umol s-1.
#
#  Default sequence:
#    1) Establish Tleaf=25 C and VPDleaf=1.0 kPa at Q=0, CO2_r=420.
#    2) Dark adapt for 30 min and collect FoFm.
#    3) Run a diagnostic dark CO2 ramp at Q=0.
#    4) At Q = 200, 500, 800, 1100, 1300, 1700, and 2000, park at
#       CO2_r=120 and then run one ascending DAT ramp to 2000 ppm.
#
#  Console setup before launch:
#    - Open a log file.
#    - Chamber setup > Fluorometer functions as: Full fluorometer.
#    - Enable Dynamic Assimilation in gas-exchange computations.
#    - Use the dynamic tuning/range match made at FLOW 600.
#    - Log Setup > Fluorometer Options:
#        Flr Action at Log = 1: FoFm (dark) or FsFm' (light)
#        Flash type = MultiPhase
#        Minimum flash interval = 60 s
#    - Confirm the configured light spectrum (normally R90B40).
#    - Fill and seat the humidifier reservoir.
#    - Stop Auto Control and any other BP that could write H2O or Temp.
#
#  VPD behavior:
#    - VPD_leaf=1.0 is issued before dark adaptation and reissued after
#      every Q transition.
#    - Before each ramp, measured VPDleaf must remain continuously within
#      vpdTol for vpdHoldSec before the program exits the gate.
#    - If the target is not reached within vpdMaxMin, the program continues.
#      VPDleaf remains controlled at 1.0 kPa and is recorded with the ramp,
#      so off-target starts can be identified during analysis.
#
#  Q=0 is a dark diagnostic slice, not a conventional FvCB capacity curve.
#  Ramps are ascending only; no hysteresis/down-ramp check is included.
# =====================================================================

from bpdefs import (ASSIGN, BREAK, DIALOG, EXEC, IF, LOOP, SETCONTROL, WAIT, LOG,
                    RETURN, DataDict, EditBox)


steps = [

    # ---- operator parameters ----------------------------------------
    ASSIGN("Qstr", exp="'200,500,800,1100,1300,1700,2000'",
           dlg=EditBox("'Light ladder'", units="'umol m-2 s-1'",
                       desc="'Ascending Q values; Q=0 is measured separately'")),
    ASSIGN("cStart", exp="120",
           dlg=EditBox("'Ramp start / park CO2'", units="'ppm'",
                       desc="'Low CO2 used during stomatal-opening parks'")),
    ASSIGN("cEnd", exp="2000",
           dlg=EditBox("'Ramp end CO2'", units="'ppm'",
                       desc="'Final reference CO2 setpoint in each ramp'")),
    ASSIGN("rate", exp="200",
           dlg=EditBox("'Ramp rate'", units="'ppm min-1'",
                       desc="'DAT CO2 ramp rate'")),
    ASSIGN("leafTemp", exp="25",
           dlg=EditBox("'Leaf temperature'", units="'C'",
                       desc="'Tleaf control setpoint'")),
    ASSIGN("vpdLeaf", exp="1.0",
           dlg=EditBox("'Leaf VPD'", units="'kPa'",
                       desc="'VPD_leaf control setpoint throughout the run'")),
    ASSIGN("vpdTol", exp="0.08",
           dlg=EditBox("'VPD tolerance'", units="'kPa'",
                       desc="'Measured VPDleaf must be this close to target'")),
    ASSIGN("startupMin", exp="5",
           dlg=EditBox("'Initial controller wait'", units="'min'",
                       desc="'Minimum wait before dark adaptation begins'")),
    ASSIGN("vpdHoldSec", exp="60",
           dlg=EditBox("'VPD confirmation hold'", units="'s'",
                       desc="'Minimum VPD equilibration time at each checkpoint'")),
    ASSIGN("vpdMaxMin", exp="10",
           dlg=EditBox("'Maximum VPD wait'", units="'min'",
                       desc="'Continue after this time even if VPD remains off target'")),
    ASSIGN("darkMin", exp="30",
           dlg=EditBox("'Dark adaptation'", units="'min'",
                       desc="'Dark-adaptation duration before FoFm'")),
    ASSIGN("parkFirstMin", exp="18",
           dlg=EditBox("'First-Q low-CO2 park'", units="'min'",
                       desc="'Long opening/induction park at Q=200'")),
    ASSIGN("parkMin", exp="5",
           dlg=EditBox("'Later low-CO2 parks'", units="'min'",
                       desc="'Opening/acclimation park at later Q values'")),
    ASSIGN("setSec", exp="90",
           dlg=EditBox("'Ramp-start settle'", units="'s'",
                       desc="'Gas-system settle after returning to low CO2'")),

    DIALOG(title="'A/Ci/Q DAT: VPD 1.0, flow 600'",
           sub="'Dark slice plus ascending A/Ci ramps across the Q ladder'",
           text="'Flow is fixed at 600 and fan at 10000 rpm. Confirm DAT was calibrated at flow 600.'",
           items="Qstr, cStart, cEnd, rate, leafTemp, vpdLeaf, vpdTol, startupMin, vpdHoldSec, vpdMaxMin, darkMin, parkFirstMin, parkMin, setSec",
           buttons="'Cancel','Continue'",
           var="button"),
    IF("button == 'Cancel'", steps=(
        RETURN(),
    )),

    # ---- derived and live values ------------------------------------
    ASSIGN("Qlad", exp="[float(s) for s in Qstr.replace(' ','').split(',') if s]"),
    ASSIGN("rampSec", exp="(cEnd - cStart) / rate * 60.0"),
    ASSIGN("vpdNow", dd=DataDict('VPDleaf', 'GasEx'), track=True),

    # ---- fixed chamber state ----------------------------------------
    SETCONTROL("Flow On/Off", "On", ""),
    SETCONTROL("Flow", "600", "float"),
    SETCONTROL("Fan On/Off", "On", ""),
    SETCONTROL("Fan_rpm", "10000", "float"),
    SETCONTROL("CO2 On/Off", "On", ""),
    SETCONTROL("Temp On/Off", "On", ""),
    SETCONTROL("Tleaf", "leafTemp", "float"),
    SETCONTROL("H2O On/Off", "On", ""),
    SETCONTROL("VPD_leaf", "vpdLeaf", "float"),
    SETCONTROL("Meas:Modulation", "On", ""),

    # ---- 1) establish the requested environment before dark adapt ---
    SETCONTROL("CO2_r", "420", "float"),
    SETCONTROL("Qin", "0", "float"),
    WAIT(dur="startupMin", units="Minutes"),
    ASSIGN("vpdReadySince", exp="-1"),
    LOOP(dur="vpdMaxMin", units="Minutes", var="vpdWait", mininc="1", steps=(
        IF("abs(vpdNow - vpdLeaf) > vpdTol", steps=(
            EXEC(0, source="vpdReadySince = -1"),
        )),
        IF("abs(vpdNow - vpdLeaf) <= vpdTol and vpdReadySince < 0", steps=(
            EXEC(0, source="vpdReadySince = vpdWait"),
        )),
        IF("vpdReadySince >= 0 and vpdWait - vpdReadySince >= vpdHoldSec", steps=(
            BREAK(),
        )),
    )),

    # ---- 2) dark-adapted baseline at normal CO2 ---------------------
    WAIT(dur="darkMin", units="Minutes"),
    SETCONTROL("H2O On/Off", "On", ""),
    SETCONTROL("VPD_leaf", "vpdLeaf", "float"),
    ASSIGN("vpdReadySince", exp="-1"),
    LOOP(dur="vpdMaxMin", units="Minutes", var="vpdWait", mininc="1", steps=(
        IF("abs(vpdNow - vpdLeaf) > vpdTol", steps=(
            EXEC(0, source="vpdReadySince = -1"),
        )),
        IF("abs(vpdNow - vpdLeaf) <= vpdTol and vpdReadySince < 0", steps=(
            EXEC(0, source="vpdReadySince = vpdWait"),
        )),
        IF("vpdReadySince >= 0 and vpdWait - vpdReadySince >= vpdHoldSec", steps=(
            BREAK(),
        )),
    )),
    LOG(flr="1: FoFm (dark) or FsFm' (light)"),

    # ---- 3) diagnostic dark CO2-response slice at Q=0 ---------------
    SETCONTROL("CO2_r", "cStart", "float"),
    SETCONTROL("H2O On/Off", "On", ""),
    SETCONTROL("VPD_leaf", "vpdLeaf", "float"),
    WAIT(dur="setSec", units="Seconds"),
    ASSIGN("vpdReadySince", exp="-1"),
    LOOP(dur="vpdMaxMin", units="Minutes", var="vpdWait", mininc="1", steps=(
        IF("abs(vpdNow - vpdLeaf) > vpdTol", steps=(
            EXEC(0, source="vpdReadySince = -1"),
        )),
        IF("abs(vpdNow - vpdLeaf) <= vpdTol and vpdReadySince < 0", steps=(
            EXEC(0, source="vpdReadySince = vpdWait"),
        )),
        IF("vpdReadySince >= 0 and vpdWait - vpdReadySince >= vpdHoldSec", steps=(
            BREAK(),
        )),
    )),
    LOOP(dur="rampSec", units="Seconds", var="t", mininc="0", steps=(
        SETCONTROL("CO2_r", "cStart + rate / 60.0 * t", "float"),
        LOG(flr="0: Nothing"),
    )),
    SETCONTROL("CO2_r", "cStart", "float"),

    # ---- 4) first light level: long low-CO2 park --------------------
    SETCONTROL("Qin", "Qlad[0]", "float"),
    SETCONTROL("CO2_r", "cStart", "float"),
    SETCONTROL("H2O On/Off", "On", ""),
    SETCONTROL("VPD_leaf", "vpdLeaf", "float"),
    WAIT(dur="parkFirstMin", units="Minutes"),
    SETCONTROL("H2O On/Off", "On", ""),
    SETCONTROL("VPD_leaf", "vpdLeaf", "float"),
    WAIT(dur="setSec", units="Seconds"),
    ASSIGN("vpdReadySince", exp="-1"),
    LOOP(dur="vpdMaxMin", units="Minutes", var="vpdWait", mininc="1", steps=(
        IF("abs(vpdNow - vpdLeaf) > vpdTol", steps=(
            EXEC(0, source="vpdReadySince = -1"),
        )),
        IF("abs(vpdNow - vpdLeaf) <= vpdTol and vpdReadySince < 0", steps=(
            EXEC(0, source="vpdReadySince = vpdWait"),
        )),
        IF("vpdReadySince >= 0 and vpdWait - vpdReadySince >= vpdHoldSec", steps=(
            BREAK(),
        )),
    )),
    LOOP(dur="rampSec", units="Seconds", var="t", mininc="0", steps=(
        SETCONTROL("CO2_r", "cStart + rate / 60.0 * t", "float"),
        LOG(flr="Default"),
    )),
    SETCONTROL("CO2_r", "cStart", "float"),

    # ---- 5) remaining Q ladder --------------------------------------
    LOOP(list="Qlad[1:]", var="Q", steps=(
        SETCONTROL("Qin", "Q", "float"),
        SETCONTROL("CO2_r", "cStart", "float"),
        SETCONTROL("H2O On/Off", "On", ""),
        SETCONTROL("VPD_leaf", "vpdLeaf", "float"),
        WAIT(dur="parkMin", units="Minutes"),

        # Reassert the same 1.0-kPa target after the Q/thermal transition.
        SETCONTROL("H2O On/Off", "On", ""),
        SETCONTROL("VPD_leaf", "vpdLeaf", "float"),
        WAIT(dur="setSec", units="Seconds"),
        ASSIGN("vpdReadySince", exp="-1"),
        LOOP(dur="vpdMaxMin", units="Minutes", var="vpdWait", mininc="1", steps=(
            IF("abs(vpdNow - vpdLeaf) > vpdTol", steps=(
                EXEC(0, source="vpdReadySince = -1"),
            )),
            IF("abs(vpdNow - vpdLeaf) <= vpdTol and vpdReadySince < 0", steps=(
                EXEC(0, source="vpdReadySince = vpdWait"),
            )),
            IF("vpdReadySince >= 0 and vpdWait - vpdReadySince >= vpdHoldSec", steps=(
                BREAK(),
            )),
        )),

        LOOP(dur="rampSec", units="Seconds", var="t", mininc="0", steps=(
            SETCONTROL("CO2_r", "cStart + rate / 60.0 * t", "float"),
            LOG(flr="Default"),
        )),

        SETCONTROL("CO2_r", "cStart", "float"),
    )),

    # ---- 6) safe endpoint -------------------------------------------
    SETCONTROL("CO2_r", "420", "float"),
    SETCONTROL("Qin", "0", "float"),
]
