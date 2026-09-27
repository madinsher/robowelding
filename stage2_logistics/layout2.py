"""Stage-2 layout: the logistics zone on the loading side (-X) of the stage-1 welding cell.  Single source of truth.

World frame of stage 1 is kept: metres, Z up, positioner at the origin, welding-robot track along Y at x = 2.0,
cell fence x in [-2.4, 4.2], y in [-3.4, 3.4] with the loading opening (light curtain) on the -X side, |y| < 1.2.

Stage 2 adds a fenced logistics zone x in [LOG_FENCE_X0, -2.4], y in [-3.4, 3.4] (it shares the cell's -X fence):

                 +Y fence (kit pallet and pipe buffer are loaded from outside through light-curtain openings)
    +------------------------------------------------------------------------------+ y = 3.4
    |  STORAGE   KIT PALLET        PIPE BUFFER      ASSEMBLY STATION   TACK ROBOT  |
    |  (stepped  flanges+elbows    (V-cradles)      (fixture+clamps)   (small arm) |  WELDING CELL
    |   rack)                                                                      |  (stage 1)
    |  ====================== HANDLER TRACK (y = 0) =========================== ->  | opening  (0,0)
    |                                                                              |  positioner
    |            OUTPUT CONVEYOR (carrier pallets) --- QC ARCH (marking + scan)     |
    +------------------------------------------------------------------------------+ y = -3.4
   x = -11.4                                                                    x = -2.4

Spool frames ("standing" pose): every assembly position holds the spool as on the positioner at tilt 0 — flange face
down (spool-local z up, flange back face on local z = 0), elbow up, pipe leg horizontal along local +X.  A spool frame
is written as (origin, x_dir): origin = flange back-face centre, x_dir = world direction of the pipe leg (local +X).
Wherever the handler serves a spool, the pipe leg points toward the handler track so the grasp is always top-down.

Part frames: flange, elbow and pipe are separate objects whose root frame IS the spool frame of the assembled spool
(no per-part offset).  So a part placed "in its assembled position at spool frame F" has world pose F.
"""
import math

import tools  # noqa: F401
from cell import layout as L

DEG = math.pi / 180.0
FPS = L.FPS

# ================================================================== zones
CELL_FENCE_X0 = L.FENCE_X[0]                   # -2.4: shared fence line with the loading opening
LOAD_OPENING_Y = L.LOAD_OPENING["width"] / 2   # 1.2
LOG_FENCE_X0 = -11.4                            # -X end of the logistics fence
LOG_FENCE_Y = (L.FENCE_Y[0], L.FENCE_Y[1])      # (-3.4, 3.4): same as the cell
LOG_FENCE_H = L.FENCE_H
# openings in the logistics fence (light curtains, see environment2): kit pallet and pipe buffer loading on +Y
LOG_FENCE_OPENINGS = [
    dict(side='+Y', x0=-7.85, x1=-6.35, curtain=True, name="kit_pallet_bay"),
    dict(side='+Y', x0=-6.15, x1=-5.05, curtain=True, name="pipe_buffer_bay"),
    dict(side='-X', y0=-2.1, y1=2.1, curtain=True, name="storage_back"),      # rack back side: unloading by AGV/forklift
]
PERSONNEL_DOOR = dict(side='-Y', center_x=-9.3, width=1.0)

# ================================================================== handler robot (loader) on a floor track along X
HANDLER_SCALE = 1.25                  # IRB 4600 x1.25 ~ IRB 6700-150/3.20 class (reach 3.2 m, 150 kg)
HANDLER_TRACK_Y = 0.0
HANDLER_TRACK_X = (-8.30, -3.10)      # carriage travel (robot J1 axis x); -3.10 = loading station at the opening
HANDLER_TRACK_BED = (-8.95, -2.62)    # rail bed incl. end stops (the cell fence hazard stripe starts at x = -2.46)
HANDLER_TRACK_TOP_Z = 0.55            # robot base plate height on the carriage
HANDLER_YAW = 0.0                     # J1 = 0 points +X (toward the positioner)
HANDLER_HOME_X = -6.40
HANDLER_Q_HOME = (0.0, -0.35, 0.55, 0.0, 1.15, 0.0)   # arm folded, gripper down, clear of every station
# zone rule: the handler may cross x = SAFE_X_HANDLER (any link / gripper / carried part) only while the welding robot
# is in its home or transit pose and the positioner is at rest.  Checked by tests/t_collision2.py.
SAFE_X_HANDLER = CELL_FENCE_X0 - 0.05

# ================================================================== tack-welding robot (small arm, fixed pedestal)
TACK_SCALE = 0.57                     # IRB 4600 x0.57 ~ IRB 1600-1.45 class
TACK_BASE = (-3.20, 2.45)             # pedestal centre (x, y); beside the assembly station on its +X side
TACK_BASE_Z = 0.40                    # pedestal top
TACK_YAW = math.pi                    # J1 = 0 faces -X (toward the station)
TACK_Q_HOME = (0.09, -0.46, -0.52, 0.0, 1.77, 0.0)   # torch vertical, tip ~(-3.6, 2.4, 1.45): above the pedestal

# ================================================================== spool geometry recap (spool-local, from stage 1)
PIPE_R = L.PIPE_OD / 2                              # 0.13655
SEAM_A_Z = L.SEAM_A_Z                               # 0.10  (flange-elbow joint, horizontal circle about local Z)
SEAM_B_X = L.SEAM_B_CENTER[0]                       # 0.381 (elbow-pipe joint, circle about local X)
PIPE_AXIS_Z = L.SEAM_B_CENTER[2]                    # 0.481 (pipe-leg axis height above the flange face)
PIPE_END_X = SEAM_B_X + L.PIPE_LEN                  # 0.981
SPOOL_REACH_X = PIPE_END_X                          # pipe end along local X
SPOOL_BACK_X = -L.FLANGE_OD / 2                     # -0.2025 (flange rim behind the flange centre)
SPOOL_TOP_Z = PIPE_AXIS_Z + PIPE_R                  # 0.618

# ================================================================== gripper (tong gripper with V-prism jaws)
GRIP_TCP_Z = 0.42          # grip centre (the gripped cylinder's axis) along tool0 +Z
GRIP_JAW_OPEN = 1.0        # set_open(1) fully open;  0 = closed on a DN250 pipe (r = PIPE_R)
GRIP_PAD_WIDTH = 0.09      # pad extent along the pipe axis (tool X)
# grasp frames in PART / SPOOL-LOCAL coordinates: origin = grip centre on the axis of the gripped cylinder,
# z = approach direction (down, local -Z), x = cylinder axis.  GRASP[name] = (origin, x_axis, z_axis, jaw_open)
GRASP = {
    # flange lying face down: the jaws pinch the weld-neck hub (r = PIPE_R on its straight part) from above
    "flange": ((0.0, 0.0, 0.085), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0), 0.05),
    # elbow standing on its lower end: gripped near its upper (horizontal) end, pad normal to the tube axis
    "elbow": ((0.310, 0.0, SEAM_A_Z + L.ELBOW_R * math.sin(math.acos(1 - 0.310 / L.ELBOW_R))),
              (math.sin(math.acos(1 - 0.310 / L.ELBOW_R)), 0.0, math.cos(math.acos(1 - 0.310 / L.ELBOW_R))),
              (math.cos(math.acos(1 - 0.310 / L.ELBOW_R)), 0.0, -math.sin(math.acos(1 - 0.310 / L.ELBOW_R))), 0.0),
    # loose pipe: gripped at its middle
    "pipe": ((SEAM_B_X + L.PIPE_LEN / 2, 0.0, PIPE_AXIS_Z), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0), 0.0),
    # assembled (tacked / welded) spool: gripped on the pipe leg close to the elbow (nearer the centre of mass)
    "spool": ((0.550, 0.0, PIPE_AXIS_Z), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0), 0.0),
}

# ================================================================== positioner hand-over (stage-1 positioner, tilt 0)
POS_LOAD_ROT = 180.0        # faceplate angle while the handler loads / unloads: pipe leg along -X, toward the handler
POS_UNLOAD_ROT = 540.0      # after welding the faceplate is at 360 (stage 1 ROT_A1); +180 -> pipe leg toward -X again
HANDLER_X_AT_POSITIONER = HANDLER_TRACK_X[1]
POS_APPROACH_DZ = 0.12      # the spool is lowered onto the faceplate from this height (wrist stays under the hood)
# clamp studs: the six stage-1 flange bolts (pos_bolt{k}, pos_washer{k}, k even) act as automatic clamp studs:
# retracted STUD_DROP below their seat (hidden inside the fixture ring) while the faceplate is empty
STUD_DROP = 0.05

# ================================================================== assembly & tack station
STATION_ORIGIN = (-4.40, 2.50, 0.80)     # spool frame origin: flange back face centre on the fixture plate
STATION_XDIR = (0.0, -1.0)               # pipe leg points -Y, toward the handler track
STATION_TABLE = dict(center=(-4.40, 2.25), size=(1.10, 1.70), top_z=0.78)   # welded table; plate 20 mm on top
# supports and clamps in SPOOL-LOCAL coordinates of the station frame
STATION_SUPPORTS = {
    "flange_seat": dict(center=(0.0, 0.0, -0.02), radius=0.215, thick=0.02),        # centring plate with 3 pins
    "elbow_vblock": dict(center=(0.300, 0.0, None)),    # V-block under the elbow end (z from the elbow surface)
    "pipe_vblock": dict(center=(0.860, 0.0, None)),     # V-block under the pipe near its end
    "pipe_stop": dict(center=(PIPE_END_X + 0.015, 0.0, PIPE_AXIS_Z)),              # end stop setting the root gap
}
STATION_CLAMPS = {             # pneumatic swing / toggle clamps: name -> (pivot local xyz, closes onto local xyz)
    "flange_l": ((-0.05, 0.30, 0.05), (-0.05, 0.19, 0.03)),
    "flange_r": ((-0.05, -0.30, 0.05), (-0.05, -0.19, 0.03)),
    "elbow": ((0.300, 0.30, PIPE_AXIS_Z - 0.05), (0.300, 0.10, PIPE_AXIS_Z + 0.07)),
    "pipe": ((0.860, 0.30, PIPE_AXIS_Z + 0.02), (0.860, 0.06, PIPE_AXIS_Z + PIPE_R)),
}
GAP_SENSOR = dict(local=(SEAM_B_X, -0.28, PIPE_AXIS_Z + 0.18))   # laser gap sensor on a bracket looking at seam B
# tack welds: 3 per joint on the tack-robot side (station local +Y = world +X).  Angles in degrees:
#   joint A: point = (R cos a, R sin a, SEAM_A_Z) ;  joint B: point = (SEAM_B_X, R cos b, PIPE_AXIS_Z + R sin b)
TACKS_A = (150.0, 90.0, 30.0)         # nominal; the planner may shift each by up to +-20 deg to stay reachable
TACKS_B = (90.0, 30.0, -30.0)
TACK_LEN = 0.018           # tack bead length along the seam (m)

# ================================================================== kit pallet (flanges + elbows) and pipe buffer
KIT_PALLET = dict(center=(-7.10, 2.40), size=(1.20, 1.60), top_z=0.144)   # euro-style pallet under a steel cassette
KIT_CASSETTE_Z = 0.20                     # top of the cassette base plate (nests stand on it)
# part frames in the cassette (part root = spool frame, see module docstring): (origin xyz, x_dir)
KIT_FLANGES = [((-7.40, 2.85, 0.23), (0.0, -1.0)), ((-6.80, 2.85, 0.23), (0.0, -1.0))]
KIT_ELBOWS = [((-7.40, 2.20, 0.20), (0.0, -1.0)), ((-6.80, 2.20, 0.20), (0.0, -1.0))]
PIPE_BUFFER = dict(center=(-5.60, 2.55), size=(1.10, 0.80), axis_z=0.90)  # V-cradle rack, pipes along Y
PIPE_BUFFER_X = (-5.33, -5.87)            # pipe centre x of the 2 slots (slot 0 is picked first); pitch >= gripper
                                          # half-width open (0.383) + PIPE_R + air
# spool-frame of a pipe lying in the buffer: pipe centre at (x, 2.55, 0.90), pipe axis along -Y (toward the track)
def pipe_buffer_frame(i):
    x = PIPE_BUFFER_X[i]
    cx, cy = x, PIPE_BUFFER["center"][1]
    mid = SEAM_B_X + L.PIPE_LEN / 2
    return ((cx, cy + mid, PIPE_BUFFER["axis_z"] - PIPE_AXIS_Z), (0.0, -1.0))

# ================================================================== output conveyor with carrier pallets
CONVEYOR = dict(x0=-8.05, x1=-3.55, y=-2.35, top_z=0.72, width=1.50, roller_pitch=0.25, roller_r=0.045)
CARRIER = dict(size=(0.70, 1.45), thick=0.06, seat_z=0.80)   # pallet footprint (x, y); flange seat height (world z)
# carrier-pallet stations (flange-centre x); spool frame on a carrier: origin (x, CONV_SPOOL_Y, seat_z), x_dir +Y
CONV_SPOOL_Y = -2.85
CONV_LOAD_X = -4.20
CONV_QC_X = -6.00
CONV_END_X = -7.50
CONV_SPEED = 0.45          # m/s cruise
# ================================================================== marking + QC arch (portal over the conveyor)
QC_ARCH = dict(x=CONV_QC_X, y0=-3.25, y1=-1.25, beam_z=2.10)      # posts at y0 / y1, carriage runs along Y
QC_MARK_LOCAL = (0.48, 0.0, PIPE_AXIS_Z + PIPE_R)   # laser-marked ID on the pipe top, spool-local
QC_SCAN_LOCAL = (SEAM_B_X, 0.0, PIPE_AXIS_Z + PIPE_R)   # scan line crosses seam B on its top
QC_LAMP = dict(local=(0.0, 0.0, 0.0))                # signal tower on the arch post (see marking_qc)

# ================================================================== storage: 2-tier stepped rack at the -X end
# tier 0 (front, low) and tier 1 (back, high); tier-1 bays are staggered half a pitch so the V-posts under the tier-1
# pipe legs stand in the gaps between the tier-0 spools
STORAGE = dict(x=(-10.10, -10.90), seat_z=(0.35, 1.00),
               bay_y=((-1.80, -1.08, -0.36, 0.36, 1.08, 1.80), (-1.44, -0.72, 0.00, 0.72, 1.44)))
# pitch 0.72: flange rim (r 0.2025) + half a V-head of the neighbouring tier (0.12) + ~35 mm air on each side
# bay spool frames: tier t, bay i -> origin (x[t], bay_y[t][i], seat_z[t]), pipe leg +X toward the track
STORAGE_TARGET = (0, 2)                   # (tier, bay) the demo spool goes into
STORAGE_FILLED = [(0, 0), (0, 1), (0, 4), (1, 0), (1, 1), (1, 2), (1, 4)]   # pre-filled with finished spools
AGV = dict(park=(-12.60, 2.20), lane_x=-12.60, path=[(-12.60, -6.0), (-12.60, 2.20)])   # outside the fence, optional

def storage_frame(tier, bay):
    return ((STORAGE["x"][tier], STORAGE["bay_y"][tier][bay], STORAGE["seat_z"][tier]), (1.0, 0.0))

# ================================================================== extra lights for the logistics zone
LOG_KEY_LIGHT = dict(pos=(-6.2, 0.0, 6.0), target=(-8.5, 0.0, 0.8), energy=900.0, size=3.0)   # aimed away from the cell opening

# ================================================================== timeline
# The stage-1 welding choreography (1104 frames) is embedded in the stage-2 scene timeline at WELD_OFFSET:
# stage-2 frame = WELD_OFFSET + stage-1 frame.  Scene frames before WELD_OFFSET + 97 are the new "pre" part
# (kitting, assembly, tacking, loading), frames after WELD_OFFSET + 930 are the new "post" part (unloading,
# marking, QC, storage).  The weld itself is NOT re-rendered: the edit (edl.py) splices in the stage-1 video.
WELD_OFFSET = None          # set by the choreography planner (plan2.py) once the pre-part length is known
