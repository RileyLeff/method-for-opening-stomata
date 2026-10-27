# =====================================================================
#  AciQ_prime_spike.py   —  LI-6800 Background Program
#  A/Ci/Q surface via low-CO2 priming: hold stomata open on a CO2
#  drawdown, then capture biochemically-limited A on the way back up
#  before the stomata reclose.  (Phragmites / Morella)
#
#  CONSOLE SETUP THIS FILE ASSUMES (set these first — NOT in the BP):
#   - Start Up > Chamber setup > Fluorometer functions as: Full fluorometer
#   - Log Setup > Fluorometer Options:
#         Flr Action at Log      = 2   (FoFm dark / FsFm'Fo' light)
#         Flash type             = MultiPhase  (optimize red target+duration first)
#         Minimum flash interval = ~15 s  (so only the capture logs flash)
#   - Log Setup > Stability: target the DELTAS (dCO2, dH2O), NOT the
#         controlled concentration; add a reference-slope gate; set a
#         TIGHT max wait so the up-branch bails before stomata move.
#   - Range-match across ~70-2000 ppm before starting.  Open a log file.
#
#  CONFIRM IN BUILDER: LOG() arg list, and the WAIT[Stability] signature
#  if you swap it in for the fixed dwells below.
# =====================================================================

from bpdefs import SETCONTROL, WAIT, LOG

# ---- protocol parameters (edit freely) ------------------------------
Q_LEVELS = [0, 200, 400, 700, 1000, 1300, 1600]    # incident light, ascending
CO2_DOWN = [400, 300, 200, 150, 100, 70]           # drawdown branch + priming
CO2_UP   = [150, 300, 500, 700, 1000, 1500, 2000]  # biochemical (high-Ci) branch

DARK_MIN   = 20     # dark-adaptation before the FoFm baseline
INDUCT_MIN = 20     # induction at working light + 420 before the first curve
QSTEP_SEC  = 240    # acclimation after each light-level change
DOWN_SEC   = 45     # dwell per drawdown point (steady-ish; could be WAIT[Stability])
PRIME_SEC  = 150    # extra hold at the lowest CO2 to max-out gsw
UP_SEC     = 25     # dwell per up-branch point - SHORT, racing stomatal reclosure
# ---------------------------------------------------------------------

def capture(co2, dwell):
    """One measured point: set CO2, settle, log (flash fires per Log Setup)."""
    return [
        SETCONTROL("CO2_r", str(co2), "float"),
        WAIT(dur=str(dwell), units="Seconds"),   # <- swap for WAIT[Stability] once tuned
        LOG(),
    ]

steps = []

# 1) Dark-adapted baseline -------------------------------------------
steps += [
    SETCONTROL("Meas:Modulation", "On", ""),     # measuring beam on
    SETCONTROL("Qin", "0", "float"),             # actinic off
    WAIT(dur=str(DARK_MIN), units="Minutes"),
    LOG(),                                        # FoFm -> Fv/Fm, seeds DarkAdapted file
]

# 2) Induction at working light + ambient CO2 ------------------------
steps += [
    SETCONTROL("CO2_r", "420", "float"),
    SETCONTROL("Qin", str(Q_LEVELS[1]), "float"),  # use 200, not the 0 slice
    WAIT(dur=str(INDUCT_MIN), units="Minutes"),
]

# 3) The A/Ci/Q surface ----------------------------------------------
for Q in Q_LEVELS:
    steps += [
        SETCONTROL("Qin", str(Q), "float"),
        WAIT(dur=str(QSTEP_SEC), units="Seconds"),
    ]
    # 3a) drawdown: low-Ci points AND progressively open the stomata
    #     (at Q=0 this is just the respiration CO2-response; priming is
    #      mostly inert in the dark — fine to keep for the bottom slice)
    for c in CO2_DOWN:
        steps += capture(c, DOWN_SEC)
    # 3b) hold at the prime so gsw is fully up before we sprint
    steps += [WAIT(dur=str(PRIME_SEC), units="Seconds")]
    # 3c) up-branch: biochemical ceiling while stomata are still open
    for c in CO2_UP:
        steps += capture(c, UP_SEC)

# 4) Park safely ------------------------------------------------------
steps += [
    SETCONTROL("CO2_r", "420", "float"),
    SETCONTROL("Qin", "0", "float"),
]
