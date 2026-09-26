"""Cell layout constants. World frame: metres, Z up, X toward the robot, Y along the track.

Everything (environment, VFX, animation) must agree with these numbers.
"""
import math

FPS = 24
DEG = math.pi / 180.0

# ---------------------------------------------------------------- positioner
# 2-axis tilt/rotate positioner with U-cradle, pedestal centred at the origin.
# The TILT AXIS IS PARALLEL TO THE TRACK (world X): at tilt = +90 the faceplate faces +Y (sideways), so the pipe leg
# sweeps a disc in the plane y ~ 0.76 that the robot arm (reaching along -X) never crosses.
POS_PEDESTAL_FOOTPRINT = (0.9, 1.0, 0.75)   # x, y, z (height of the pedestal box)
POS_TILT_AXIS_Z = 1.35            # tilt axis = world X through (0, 0, 1.35) — high enough for the pipe to swing clear
POS_TILT_HALF_WIDTH = 0.62        # trunnion bearings at x = +/- 0.62
POS_FACEPLATE_OFFSET = 0.25       # faceplate centre is 0.25 above the tilt axis at tilt = 0
POS_FACEPLATE_RADIUS = 0.32
POS_FIXTURE_THICK = 0.03          # adapter ring between faceplate and the spool flange
# tilt = 0   -> faceplate faces +Z (loading position)
# tilt = +90 -> faceplate faces +Y, centre at (0, 0.25, 1.35);  tilt = -90 -> faces -Y  (180° index = tilt +90 -> -90)

# ---------------------------------------------------------------- spool (DN250 pipe + LR elbow + WN flange)
PIPE_OD = 0.2731
PIPE_WALL = 0.0093
ELBOW_R = 0.381                   # long-radius 1.5D
PIPE_LEN = 0.60
FLANGE_OD = 0.405
FLANGE_THK = 0.028
FLANGE_HUB_LEN = 0.10             # weld-neck hub: seam A is at spool-local z = FLANGE_HUB_LEN
FLANGE_HOLES = 12
FLANGE_PCD = 0.355
FLANGE_HOLE_D = 0.026
# Spool local frame: flange back face on z = 0 plane, flange axis = +Z, elbow bends toward +X.
SEAM_A_Z = FLANGE_HUB_LEN                     # circle, axis local +Z, centre (0,0,SEAM_A_Z)
SEAM_B_CENTER = (ELBOW_R, 0.0, FLANGE_HUB_LEN + ELBOW_R)   # circle, axis local +X
SEAM_RADIUS = PIPE_OD / 2
BEAD_WIDTH = 0.020
BEAD_HEIGHT = 0.0045

# ---------------------------------------------------------------- robot & track
TRACK_X = 2.0                     # track centre line x
TRACK_Y_MIN, TRACK_Y_MAX = -2.0, 2.0
TRACK_TOP_Z = 0.45                # robot base plate height (carriage top)
ROBOT_YAW = math.pi               # base rotated so the arm faces the positioner (-X world)
ROBOT_HOME_Y = -1.9
# home joint angles (rad): j1..j6 — upper arm back, forearm up, torch vertical over the track (park pose)
ROBOT_Q_HOME = (0.0, -0.55, 0.95, 0.0, 1.15, 0.0)

# ---------------------------------------------------------------- torch (tool frame on tool0)
TORCH_BRACKET_LEN = 0.10
TORCH_NECK_LEN = 0.20
TORCH_BEND_ANGLE = 45 * DEG
TORCH_HEAD_LEN = 0.17             # after the bend, to the contact tip
TORCH_STICKOUT = 0.015
TORCH_NOZZLE_R = 0.011

# ---------------------------------------------------------------- cell fence & equipment (footprints)
FENCE_X = (-2.4, 4.2)
FENCE_Y = (-3.4, 3.4)
FENCE_H = 2.1
FENCE_DOOR = dict(side='-Y', center_x=-1.0, width=1.2)        # sliding access door
LOAD_OPENING = dict(side='-X', width=2.4, light_curtain=True)  # loading side with light curtain
CONTROLLER_CABINET = dict(pos=(3.6, 2.4, 0.0), size=(0.7, 0.55, 1.4))   # IRC5-style cabinet, inside fence corner
WELD_POWER_SOURCE = dict(pos=(3.4, 1.2, 0.0), size=(0.45, 0.7, 0.95))
WIRE_DRUM = dict(pos=(3.4, 0.35, 0.0), radius=0.26, height=0.8)
TORCH_CLEANER = dict(pos=(2.0, -2.55, 0.0), height=0.95)      # torch cleaning station at the -Y track end
FUME_HOOD = dict(pos=(0.3, 0.0, 3.05), size=(1.6, 1.6, 0.35))  # above the positioner (clear of the pipe sweep), duct to ceiling
PARTS_PALLET = dict(pos=(-1.5, 2.2, 0.0))                       # pallet with spool parts
FINISHED_RACK = dict(pos=(-1.5, -2.2, 0.0))

# ---------------------------------------------------------------- hall
HALL_SIZE = (36.0, 30.0, 9.0)     # x, y, height; centred on the cell
COLUMN_SPACING = 6.0

# ---------------------------------------------------------------- render
RES_FINAL = (1920, 1080)
RES_PREVIEW = (960, 540)
