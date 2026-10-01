#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the pipe-spool welding cell (ABB IRB 4600 on a linear track + 2-axis positioner) in RoboDK.

Run from RoboDK (File > Open the .py, or Program > Add Python program) or from a terminal while
RoboDK is running:  python build_station.py

The script creates a NEW station with:
  * a linear track (7th axis, Y direction) with the ABB IRB 4600-40/2.55 mounted on it (yaw 180 deg),
  * a MIG torch tool (geometry generated as STL, TCP from the bracket/neck/45-deg-head chain),
  * a 2-axis tilt/rotate positioner (two chained 1-axis mechanisms: tilt about world X through (0,0,1350),
    faceplate rotation about its normal; tilt +90 -> faceplate faces +Y, -90 -> faces -Y),
  * the DN250 spool (spool.stl exported from Blender, mm, spool-local frame) mounted 30 mm above the faceplate,
  * seam A / seam B as curve objects (points + outward normals) plus "curve follow" machining projects,
  * explicit Cartesian target lists and MoveL programs for both seams (torch normal to the surface,
    10 deg push angle, optional weaving),
  * the storyboard choreography as a main program "DemoCycle" (track / positioner / robot sub-programs,
    arc on/off macros that deposit a weld-bead trace, view-switch macros for the 12 camera shots),
  * cell dressing (fence, cabinets, hood, hall columns, crane beam) as coloured boxes,
  * camera reference frames for every storyboard shot (optionally opened as 2D simulated cameras).

Units inside this file: mm and degrees (RoboDK convention). The layout numbers mirror
demo_video/cell/layout.py (metres) and the choreography numbers mirror demo_video/cell/animation.py -
tests/t_robodk_math.py asserts they stay in sync.

Choreography (as in cell/animation.py and post/storyboard.json, 46 s / 1104 frames):
  1. tilt 0 -> +90 (faceplate faces +Y), track home (-1900) -> +200, robot home -> transit -> above seam A
  2. laser seam search: +/-60 mm along the seam axis (world Y) 35 mm above the top of seam A
  3. seam A: torch static at the top of the seam (0, 380, 1487), the faceplate rotates -20 -> +360 deg (380 deg)
     -> the pipe leg ends up along +X (toward the robot)
  4. lift, transit pose (torch down, TCP (1350, y, 2050) over the track), track -> +761
  5. seam B sectors 1-2 at tilt +90 (seam centre (381, 761, 1350), axis X): from 12 o'clock toward -Y, then
     toward +Y, each -4..+94 deg, radial approach/retract 120 mm, return via 1:30 o'clock retracted 160 mm
  6. transit pose, positioner flips tilt +90 -> -90 over the top (pipe stays along +X), track -> -761
  7. sectors 3-4 at tilt -90 (seam centre (381, -761, 1350)), same pattern, retract 150 mm to transit
  8. robot home, track home (-1900), tilt -90 -> 0

The geometry/choreography maths (seam points, TCP pose, target frames) is pure python (robomath only) and
lives in the first half of the file so it can be unit-tested without RoboDK (see tests/t_robodk_math.py).
"""
import glob
import math
import os
import struct

from robodk import robolink as rl
from robodk import robomath as rm
from robodk.robomath import Mat, transl, rotx, roty, rotz, invH, eye

# =====================================================================================================
# 1. Layout constants (mm / deg) - mirror of cell/layout.py
# =====================================================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
GEN_DIR = os.path.join(HERE, "generated")          # STL of the torch, generated macros
SPOOL_STL = os.path.join(HERE, "spool.stl")

# positioner (pedestal centred at the origin). The TILT AXIS IS PARALLEL TO THE TRACK (world X):
# tilt = 0 -> faceplate faces +Z (loading), tilt = +90 -> faces +Y (centre (0, 250, 1350)), tilt = -90 -> faces -Y.
POS_PEDESTAL_FOOTPRINT = (900.0, 1000.0, 750.0)   # x, y, height of the pedestal box
POS_TILT_AXIS_Z = 1350.0          # tilt axis = world X through (0, 0, 1350)
POS_TILT_HALF_WIDTH = 620.0       # trunnion bearings at x = +/-620
POS_TOWER_X = POS_TILT_HALF_WIDTH + 60.0   # bearing towers at x = +/-680
POS_FACEPLATE_OFFSET = 250.0      # faceplate centre above the tilt axis (tilt = 0)
POS_FACEPLATE_RADIUS = 320.0
POS_FIXTURE_THICK = 30.0          # adapter ring between faceplate and spool flange (spool 30 mm above the faceplate)
FLANGE_BOLTS = True               # six bolts through the spool flange; the stage-2 station sets False (swing clamps)
TILT_LIMITS = (-120.0, 120.0)     # the 180 deg index is tilt +90 -> -90 over the top
ROT_LIMITS = (-720.0, 720.0)

# spool (DN250 pipe + LR elbow + WN flange); spool-local frame: flange back face on z=0, axis +Z,
# elbow bends toward +X
PIPE_OD = 273.1
ELBOW_R = 381.0
PIPE_LEN = 600.0
FLANGE_HUB_LEN = 100.0
SEAM_A_Z = FLANGE_HUB_LEN                                # circle, axis +Z, centre (0,0,SEAM_A_Z)
SEAM_B_CENTER = (ELBOW_R, 0.0, FLANGE_HUB_LEN + ELBOW_R)  # circle, axis +X
SEAM_RADIUS = PIPE_OD / 2

# robot & track
TRACK_X = 2000.0
TRACK_Y_MIN, TRACK_Y_MAX = -2000.0, 2000.0
TRACK_TOP_Z = 450.0               # robot base plate height (carriage top)
ROBOT_YAW_DEG = 180.0             # base rotated so the arm faces the positioner (-X world)
ROBOT_HOME_Y = -1900.0
# home joints (deg) = cell/layout.py ROBOT_Q_HOME (rad) converted: upper arm back, forearm up, torch vertical
ROBOT_Q_HOME = [0.0, -31.5, 54.4, 0.0, 65.9, 0.0]
ROBOT_LIBRARY_GLOBS = ["*IRB*4600*40*2.55*.robot", "*IRB*4600*2.55*.robot", "*IRB*4600*.robot"]

# torch (tool frame on the flange): bracket + neck along +Z, 45 deg bend toward +X, head, stickout
TORCH_BRACKET_LEN = 100.0
TORCH_NECK_LEN = 200.0
TORCH_BEND_DEG = 45.0
TORCH_HEAD_LEN = 170.0
TORCH_STICKOUT = 15.0
TORCH_NOZZLE_R = 11.0

# cell fence & equipment (world, mm)
FENCE_X = (-2400.0, 4200.0)
FENCE_Y = (-3400.0, 3400.0)
FENCE_H = 2100.0
FENCE_DOOR = dict(side="-Y", center_x=-1000.0, width=1200.0)
LOAD_OPENING = dict(side="-X", width=2400.0)
CONTROLLER_CABINET = dict(pos=(3600.0, 2400.0, 0.0), size=(700.0, 550.0, 1400.0))
WELD_POWER_SOURCE = dict(pos=(3400.0, 1200.0, 0.0), size=(450.0, 700.0, 950.0))
WIRE_DRUM = dict(pos=(3400.0, 350.0, 0.0), radius=260.0, height=800.0)
TORCH_CLEANER = dict(pos=(2000.0, -2550.0, 0.0), height=950.0)
FUME_HOOD = dict(pos=(300.0, 0.0, 3050.0), size=(1600.0, 1600.0, 350.0))   # above the pipe sweep
PARTS_PALLET = dict(pos=(-1500.0, 2200.0, 0.0))
FINISHED_RACK = dict(pos=(-1500.0, -2200.0, 0.0))
HALL_SIZE = (36000.0, 30000.0, 9000.0)
COLUMN_SPACING = 6000.0

# choreography (same as cell/animation.py; metres there)
ROT_A0 = -20.0                    # faceplate angle at the start of seam A
ROT_A1 = ROT_A0 + 380.0           # = 360 -> pipe leg along +X (toward the robot) for the seam-B sectors
ROT_A_MID = ROT_A0 + 106.0        # camera switch C3b -> C4 during seam A (frame 301 of 241..456)
TILT_B1, TILT_B2 = 90.0, -90.0    # seam B sectors 1-2 / 3-4 (180 deg index = tilt flip over the top)
SEAM_B_Y = POS_FACEPLATE_OFFSET + POS_FIXTURE_THICK + SEAM_B_CENTER[2]   # |y| of the seam-B centre at tilt +/-90 (761)
TRACK_A = 200.0                   # track station for seam A (arm slightly ahead of the seam plane)
TRACK_B1, TRACK_B2 = SEAM_B_Y, -SEAM_B_Y                                 # track stations for the sectors
TRANSIT_TCP = (1350.0, 2050.0)    # (x, z) of the parked torch tip, torch down, over the track (y = track station)
SECTOR_A0, SECTOR_A1 = -4.0, 94.0 # each seam-B sector from 12 o'clock: -4 .. +94 deg (overlaps at 12, 3, 9 o'clock)
SECTOR_DEG = SECTOR_A1 - SECTOR_A0
PUSH_ANGLE_DEG = 10.0
LIFT_APPROACH = 120.0             # radial approach / retract at the sector starts/ends
LIFT_PARK = 150.0                 # lift after seam A and after sectors 2/4 (before the transit pose)
VIA_RETRACT = 160.0               # return move between sectors: via 1:30 o'clock retracted this much
VIA_ANGLE = 45.0
SCAN_HEIGHT = 35.0                # laser seam search height above the seam
SCAN_HALF = 60.0                  # laser sweep +/- across the joint (along the seam axis = world Y at tilt 90)

# process speeds
WELD_SPEED_MMS = 10.0             # welding travel speed (8..12 mm/s)
AIR_SPEED_MMS = 500.0             # linear air moves
AIR_JOINT_DEGS = 60.0             # joint moves
TRACK_SPEED_MMS = 500.0
POS_INDEX_DEGS = 20.0             # positioner tilt / indexing speed
POS_WELD_DEGS = 10.0              # faceplate speed during seam A (10 deg/s = 24 mm/s at R136; use 4.2 for 10 mm/s)
WEAVE_AMPL_MM = 0.0               # optional weaving amplitude along the seam axis (0 = off)
WEAVE_FREQ_HZ = 2.5
SECTOR_POINTS = 24                # MoveL targets per 98 deg sector (without weaving)
CURVE_POINTS = 90                 # points per seam curve (AddCurve)

# camera shots = cell/animation.py SHOTS start poses: (name, first frame, last frame, camera (mm), aim (mm), lens mm)
SHOTS = [
    ("C1_wide", 1, 96, (-4600, -7200, 4000), (900, 0, 1100), 28),
    ("C2_setup", 97, 192, (-200, 3100, 2500), (600, 200, 1350), 30),
    ("C3a_laser", 193, 240, (750, 1550, 1950), (0, 400, 1470), 85),
    ("C3b_closeA", 241, 300, (900, 1350, 2000), (0, 400, 1470), 60),
    ("C4_midA", 301, 456, (900, 2000, 2900), (50, 420, 1450), 50),
    ("C5_reposition", 457, 540, (4400, -3200, 3400), (1400, 300, 1000), 30),
    ("C6a_closeB1", 541, 648, (1300, -550, 2050), (420, 760, 1420), 42),
    ("C6b_closeB1", 649, 708, (1300, 2050, 2050), (420, 760, 1420), 42),
    ("C7_index", 709, 810, (-3400, -1800, 2400), (400, 0, 1400), 32),
    ("C8a_closeB2", 811, 888, (1300, -2050, 2050), (420, -760, 1420), 42),
    ("C8b_closeB2", 889, 930, (1300, 550, 2050), (420, -760, 1420), 42),
    ("C9_final", 931, 1104, (5400, -6200, 4000), (800, -200, 1100), 30),
]
OPEN_2D_CAMERAS = False           # True: also open a simulated 2D camera window for every shot
VIEW_OPENGL = True                # 3D view convention for setViewPose (camera looks along -Z, Y up); see README
SAVE_RDK = True                   # save <station>.rdk next to this script when done
STATION_NAME = "PipeSpool_WeldCell"
# BuildMechanism geometry convention: False = the shape objects are taken as they sit in the station (world mm,
# mechanism at joints_build) and `base` only places the joint axis; True = the shapes are expected w.r.t. the
# mechanism base frame (they are pre-transformed by invH(base)). Flip it if the built positioner/track looks rotated.
MECH_GEOMETRY_IN_BASE_FRAME = False

# optional library mechanisms instead of the built ones (glob patterns inside PATH_LIBRARY, "" = build)
TRACK_LIBRARY_GLOB = ""
POSITIONER_LIBRARY_GLOB = ""

D2R = math.pi / 180.0


# =====================================================================================================
# 2. Pure maths (no RoboDK connection needed)
# =====================================================================================================
def v3(p):
    return [float(p[0]), float(p[1]), float(p[2])]


def xform_point(T, p):
    """World point of a local point p under pose T."""
    x, y, z = p
    return rm.add3(rm.add3(T.Pos(), rm.mult3(T.VX(), x)), rm.add3(rm.mult3(T.VY(), y), rm.mult3(T.VZ(), z)))


def xform_vec(T, v):
    x, y, z = v
    return rm.add3(rm.mult3(T.VX(), x), rm.add3(rm.mult3(T.VY(), y), rm.mult3(T.VZ(), z)))


def pose_from_axes(x, y, z, p):
    return Mat([[x[0], y[0], z[0], p[0]],
                [x[1], y[1], z[1], p[1]],
                [x[2], y[2], z[2], p[2]],
                [0.0, 0.0, 0.0, 1.0]])


def spool_pose(tilt_deg, rot_deg):
    """Pose of the spool-local frame in the world for given positioner joints (mm).
    Tilt rotates about world X through (0,0,POS_TILT_AXIS_Z) with the Blender sign convention
    (cell/positioner.set_tilt: +90 -> faceplate faces +Y), i.e. rotx(-tilt)."""
    return (transl(0, 0, POS_TILT_AXIS_Z) * rotx(-tilt_deg * D2R) * transl(0, 0, POS_FACEPLATE_OFFSET)
            * rotz(rot_deg * D2R) * transl(0, 0, POS_FIXTURE_THICK))


def tilt_base_pose():
    """Base pose of the tilt mechanism (1R about its Z): Z = world -X through (0,0,POS_TILT_AXIS_Z), so that a
    positive joint value is a rotation of -joint about world +X (= spool_pose / positioner.set_tilt)."""
    return transl(0, 0, POS_TILT_AXIS_Z) * roty(-90 * D2R)      # mechanism Z -> world -X, mechanism X -> world +Z


def rot_base_pose_world():
    """Base pose (world, tilt = 0) of the faceplate rotation mechanism."""
    return transl(0, 0, POS_TILT_AXIS_Z + POS_FACEPLATE_OFFSET)


def track_base_pose():
    """Base pose of the linear track: translation along world +Y at x = TRACK_X."""
    return transl(TRACK_X, 0, 0) * rotx(-90 * D2R)              # mechanism Z -> world +Y


def robot_base_pose_world(track_y):
    return transl(TRACK_X, track_y, TRACK_TOP_Z) * rotz(ROBOT_YAW_DEG * D2R)


def tcp_pose():
    """MIG torch TCP with respect to the robot flange (tool0)."""
    return (transl(0, 0, TORCH_BRACKET_LEN + TORCH_NECK_LEN) * roty(TORCH_BEND_DEG * D2R)
            * transl(0, 0, TORCH_HEAD_LEN + TORCH_STICKOUT))


def seam_local(which):
    """(centre, axis, u, v) of a seam circle in spool-local coordinates: point(a) = c + R (cos a u + sin a v)."""
    if which == "A":
        return [0.0, 0.0, SEAM_A_Z], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    return list(SEAM_B_CENTER), [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]


def seam_points_local(which, n=CURVE_POINTS, closed=True):
    """Seam circle as rows [x, y, z, i, j, k] (mm, outward normals) in the spool frame, for AddCurve."""
    c, ax, u, v = seam_local(which)
    rows = []
    count = n + 1 if closed else n
    for k in range(count):
        a = 2 * math.pi * (k % n) / n
        nrm = rm.add3(rm.mult3(u, math.cos(a)), rm.mult3(v, math.sin(a)))
        p = rm.add3(c, rm.mult3(nrm, SEAM_RADIUS))
        rows.append(p + nrm)
    return rows


def seam_frame_world(which, T_spool):
    c, ax, u, v = seam_local(which)
    return xform_point(T_spool, c), xform_vec(T_spool, ax), xform_vec(T_spool, u), xform_vec(T_spool, v)


def seam_point(sf, a_deg):
    """(point, outward normal, tangent) at angle a on a world seam frame."""
    c, ax, u, v = sf
    n = rm.add3(rm.mult3(u, math.cos(a_deg * D2R)), rm.mult3(v, math.sin(a_deg * D2R)))
    p = rm.add3(c, rm.mult3(n, SEAM_RADIUS))
    return p, n, rm.cross(ax, n)


def top_angle(sf):
    """Seam angle (deg) of the highest point of the circle (12 o'clock)."""
    c, ax, u, v = sf
    return math.atan2(v[2], u[2]) / D2R


def robot_lean(p, track_y):
    """Direction the torch body leans to: toward the robot base, tilted up."""
    d = rm.normalize3([TRACK_X - p[0], track_y - p[1], 0.0])
    return rm.add3(d, [0.0, 0.0, 0.9])


def torch_target(p, n, t, lean, push_deg=PUSH_ANGLE_DEG):
    """TCP pose: Z into the surface (-n) tilted by the push angle toward the travel direction t,
    X (torch body) as close as possible to `lean`."""
    n = rm.normalize3(n)
    t = rm.normalize3(t)
    z = rm.add3(rm.mult3(n, -math.cos(push_deg * D2R)), rm.mult3(t, math.sin(push_deg * D2R)))
    x = rm.normalize3(rm.subs3(lean, rm.mult3(z, rm.dot(lean, z))))
    y = rm.cross(z, x)
    return pose_from_axes(x, y, z, p)


def lifted(T, dz):
    """Same orientation, position raised by dz (world Z)."""
    return transl(0, 0, dz) * T


def radial(T, n, d):
    """Same orientation, position moved by d along the (outward) surface normal n."""
    n = rm.normalize3(n)
    return transl(n[0] * d, n[1] * d, n[2] * d) * T


def transit_target(track_y):
    """Parked torch: tip at TRANSIT_TCP over the track (y = track station), pointing straight down, body leaning +X.
    Outside the sweep of the part for every track / positioner motion."""
    p = [TRANSIT_TCP[0], float(track_y), TRANSIT_TCP[1]]
    return torch_target(p, [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.9], 0.0)


def seam_a_static_target(track_y=TRACK_A):
    """Torch pose for seam A: top (12 o'clock) of seam A at tilt TILT_B1 / rot ROT_A0; the part rotates under
    the torch about the faceplate axis (world +Y), the torch travels the opposite way relative to the part."""
    sf = seam_frame_world("A", spool_pose(TILT_B1, ROT_A0))
    p, n, t = seam_point(sf, top_angle(sf))
    # the surface moves with omega x r = axis x r = t; the torch travels the opposite way relative to the part -> -t
    return torch_target(p, n, rm.mult3(t, -1.0), robot_lean(p, track_y))


def seam_a_movel_targets(n_pts=4 * SECTOR_POINTS, track_y=TRACK_A):
    """Alternative: the robot drives the torch around the full seam A circle while the part is static.
    Returns (poses, normals)."""
    sf = seam_frame_world("A", spool_pose(TILT_B1, ROT_A0))
    a0 = top_angle(sf)
    poses, normals = [], []
    for k in range(n_pts + 1):
        p, n, t = seam_point(sf, a0 + 360.0 * k / n_pts)
        poses.append(torch_target(p, n, t, robot_lean(p, track_y)))
        normals.append(n)
    return poses, normals


def laser_scan_targets(n_pts=7, track_y=TRACK_A):
    """Laser seam search: sweep +/-SCAN_HALF along the seam-A axis (world Y at tilt 90) SCAN_HEIGHT above the
    12 o'clock point, torch orientation as for the weld."""
    sf = seam_frame_world("A", spool_pose(TILT_B1, ROT_A0))
    ax = sf[1]
    TA = seam_a_static_target(track_y)
    out = []
    for k in range(n_pts):
        s = k / (n_pts - 1.0)
        x = -SCAN_HALF + 2 * SCAN_HALF * s
        out.append(transl(ax[0] * x, ax[1] * x, ax[2] * x + SCAN_HEIGHT) * TA)
    return out


def sector_targets(tilt_deg, track_y, toward_plus_y, n_pts=None, weave=WEAVE_AMPL_MM):
    """Seam-B sector at tilt +/-90 with the pipe along +X (rot = ROT_A1): targets from 12 o'clock (SECTOR_A0)
    over SECTOR_DEG toward -Y or +Y. Returns (poses, outward normals, via pose): the via pose is the 1:30 o'clock
    point retracted VIA_RETRACT along its normal (push angle 0) used for the return move between sectors."""
    sf = seam_frame_world("B", spool_pose(tilt_deg, ROT_A1))
    a_top = top_angle(sf)
    p5, _, _ = seam_point(sf, a_top + 5.0)
    sgn = 1.0 if ((p5[1] > sf[0][1]) == toward_plus_y) else -1.0
    arc_len = SEAM_RADIUS * SECTOR_DEG * D2R
    if n_pts is None:
        n_pts = SECTOR_POINTS
        if weave > 0:        # at least 4 targets per weave period
            n_pts = max(n_pts, int(arc_len / (WELD_SPEED_MMS / WEAVE_FREQ_HZ / 4.0)) + 1)
    poses, normals = [], []
    for k in range(n_pts + 1):
        s = k / float(n_pts)
        p, n, t = seam_point(sf, a_top + sgn * (SECTOR_A0 + SECTOR_DEG * s))
        if weave > 0:
            w = weave * math.sin(2 * math.pi * WEAVE_FREQ_HZ * (s * arc_len / WELD_SPEED_MMS))
            p = rm.add3(p, rm.mult3(sf[1], w))
        poses.append(torch_target(p, n, rm.mult3(t, sgn), robot_lean(p, track_y)))
        normals.append(n)
    pv, nv, tv = seam_point(sf, a_top + sgn * VIA_ANGLE)
    via = radial(torch_target(pv, nv, rm.mult3(tv, sgn), robot_lean(pv, track_y), 0.0), nv, VIA_RETRACT)
    return poses, normals, via


def choreography():
    """All Cartesian target poses of the demo (world/cell frame, mm) grouped per program."""
    TA = seam_a_static_target()
    movel, movel_n = seam_a_movel_targets()
    ch = dict(
        weldA=TA,
        approachA=lifted(TA, LIFT_PARK),
        scan=laser_scan_targets(),
        weldA_movel=movel,
        weldA_movel_n=movel_n,
        transit=transit_target(TRACK_A),
        transitB1=transit_target(TRACK_B1),
        transitB2=transit_target(TRACK_B2),
    )
    for key, tilt, ty, plus_y in (("sector1", TILT_B1, TRACK_B1, False), ("sector2", TILT_B1, TRACK_B1, True),
                                  ("sector3", TILT_B2, TRACK_B2, False), ("sector4", TILT_B2, TRACK_B2, True)):
        poses, normals, via = sector_targets(tilt, ty, plus_y)
        ch[key] = poses
        ch[key + "_n"] = normals
        ch[key + "_via"] = via
    return ch


def camera_frame_pose(cam, aim, up=(0.0, 0.0, 1.0)):
    """Pose of a camera reference frame (RoboDK 2D camera convention: +Z = viewing direction, +X right, +Y down)."""
    f = rm.normalize3(rm.subs3(v3(aim), v3(cam)))
    x = rm.normalize3(rm.cross(f, v3(up)))
    y = rm.cross(f, x)
    return pose_from_axes(x, y, f, v3(cam))


def view_pose(cam, aim, up=(0.0, 0.0, 1.0), opengl=VIEW_OPENGL):
    """Argument for Robolink.setViewPose: pose of the world with respect to the view (camera).
    OpenGL convention: the view looks along its -Z, +Y is up on screen. Set opengl=False for +Z forward."""
    f = rm.normalize3(rm.subs3(v3(aim), v3(cam)))
    x = rm.normalize3(rm.cross(f, v3(up)))
    if opengl:
        z = rm.mult3(f, -1.0)
        y = rm.cross(z, x)
    else:
        z = f
        y = rm.cross(z, x)
    return invH(pose_from_axes(x, y, z, v3(cam)))


def lens_to_fov_deg(lens_mm, sensor_h_mm=20.25):
    """Vertical field of view of a 36 mm-wide 16:9 sensor for a given focal length."""
    return 2 * math.atan(sensor_h_mm / 2 / lens_mm) / D2R


# ---------------------------------------------------------------- triangle mesh helpers (mm)
def box_tris(center, size, rot=None):
    """12 triangles of an axis-aligned box (optionally rotated by pose `rot` about its centre)."""
    cx, cy, cz = center
    hx, hy, hz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
    corners = [[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    if rot is not None:
        corners = [xform_vec(rot, c) for c in corners]
    corners = [[c[0] + cx, c[1] + cy, c[2] + cz] for c in corners]
    # index: sx*4 + sy*2 + sz with (-1 -> 0, 1 -> 1)
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = []
    for a, b, c, d in faces:
        tris.append([corners[a], corners[b], corners[c]])
        tris.append([corners[a], corners[c], corners[d]])
    return tris


def cylinder_tris(p0, p1, r, n=24, r1=None):
    """Closed cylinder / cone frustum from p0 to p1 (radius r at p0, r1 at p1)."""
    r1 = r if r1 is None else r1
    p0, p1 = v3(p0), v3(p1)
    ax = rm.normalize3(rm.subs3(p1, p0))
    ref = [0.0, 0.0, 1.0] if abs(ax[2]) < 0.9 else [1.0, 0.0, 0.0]
    u = rm.normalize3(rm.cross(ax, ref))
    v = rm.cross(ax, u)
    ring0, ring1 = [], []
    for k in range(n):
        a = 2 * math.pi * k / n
        d = rm.add3(rm.mult3(u, math.cos(a)), rm.mult3(v, math.sin(a)))
        ring0.append(rm.add3(p0, rm.mult3(d, r)))
        ring1.append(rm.add3(p1, rm.mult3(d, r1)))
    tris = []
    for k in range(n):
        j = (k + 1) % n
        tris.append([ring0[k], ring1[k], ring1[j]])
        tris.append([ring0[k], ring1[j], ring0[j]])
        tris.append([p0, ring0[j], ring0[k]])
        tris.append([p1, ring1[k], ring1[j]])
    return tris


def tris_transform(tris, T):
    """Triangles moved by pose T."""
    return [[xform_point(T, p) for p in tri] for tri in tris]


def torch_mesh_triangles():
    """MIG torch geometry in the flange (tool0) frame, mm: bracket, neck, 45-deg bend, head, gas nozzle, tip."""
    tris = []
    tris += box_tris((0, 0, TORCH_BRACKET_LEN * 0.35), (110, 110, TORCH_BRACKET_LEN * 0.7))        # bracket
    tris += cylinder_tris((0, 0, TORCH_BRACKET_LEN * 0.7), (0, 0, TORCH_BRACKET_LEN), 40)          # clamp
    z_bend = TORCH_BRACKET_LEN + TORCH_NECK_LEN
    tris += cylinder_tris((0, 0, TORCH_BRACKET_LEN), (0, 0, z_bend), 22)                            # neck
    head_dir = [math.sin(TORCH_BEND_DEG * D2R), 0.0, math.cos(TORCH_BEND_DEG * D2R)]
    bend = [0.0, 0.0, z_bend]
    tris += cylinder_tris(bend, rm.add3(bend, rm.mult3(head_dir, 20.0)), 26, 16)                   # bend knuckle
    head_end = rm.add3(bend, rm.mult3(head_dir, TORCH_HEAD_LEN - 45.0))
    tris += cylinder_tris(bend, head_end, 16)                                                       # swan neck
    nozzle_end = rm.add3(bend, rm.mult3(head_dir, TORCH_HEAD_LEN))
    tris += cylinder_tris(head_end, nozzle_end, 15.0, 20, TORCH_NOZZLE_R)                           # gas nozzle
    tip_end = rm.add3(bend, rm.mult3(head_dir, TORCH_HEAD_LEN + TORCH_STICKOUT))
    tris += cylinder_tris(nozzle_end, tip_end, 1.0, 8)                                              # wire stickout
    return tris


def write_stl(path, tris, name="mesh"):
    """Write a binary STL (mm)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(name.encode("ascii")[:80].ljust(80, b"\0"))
        fh.write(struct.pack("<I", len(tris)))
        for a, b, c in tris:
            nrm = rm.cross(rm.subs3(b, a), rm.subs3(c, a))
            ln = rm.norm(nrm)
            nrm = rm.mult3(nrm, 1.0 / ln) if ln > 1e-9 else [0.0, 0.0, 0.0]
            fh.write(struct.pack("<12fH", *(nrm + list(a) + list(b) + list(c)), 0))


def tris_to_mat(tris):
    """Triangle list -> 3xN Mat as expected by Robolink.AddShape."""
    return Mat([list(p) for tri in tris for p in tri]).tr()


# =====================================================================================================
# 3. Generated macro sources (added to the station as Python programs)
# =====================================================================================================
ARC_ON_SRC = '''# ArcOn: start the weld-bead trace (simulated spray gun projected on the spool). Generated by build_station.py
from robodk.robolink import *
from robodk.robomath import *
RDK = Robolink()
tool = RDK.Item("MIG torch", ITEM_TYPE_TOOL)
part = RDK.Item("Spool DN250", ITEM_TYPE_OBJECT)
if RDK.getParam("ARC_GUN") is None:
    # bead volume around the TCP (mm): centre / vertex A / vertex B / RGBA for the near and far planes
    near = [0, 0, -3, 3, 0, -3, 0, 3, -3, 0.85, 0.55, 0.25, 1.0]
    far = [0, 0, 8, 6, 0, 8, 0, 6, 8, 0.85, 0.55, 0.25, 1.0]
    RDK.Spray_Add(tool, part, "PROJECT PARTICLE=SPHERE(3.5,8,1,1,0.45) STEP=1x1 RAND=0", Mat([near, far]).tr())
    RDK.setParam("ARC_GUN", "1")
RDK.Spray_SetState(SPRAY_ON)
'''

ARC_OFF_SRC = '''# ArcOff: stop the weld-bead trace. Generated by build_station.py
from robodk.robolink import *
RDK = Robolink()
RDK.Spray_SetState(SPRAY_OFF)
'''

VIEW_SRC = '''# {name}: move the 3D view to storyboard shot {shot} (frames {f0}-{f1}; camera {cam} mm -> aim {aim} mm, lens {lens} mm).
# Generated by build_station.py
from robodk.robolink import *
from robodk.robomath import *
RDK = Robolink()
RDK.setViewPose(Mat({rows}))
'''


def write_macros():
    """Write ArcOn/ArcOff and one View_* macro per storyboard shot into GEN_DIR (stale View_* files are removed);
    return {name: path}."""
    os.makedirs(GEN_DIR, exist_ok=True)
    files = {}
    for name, src in (("ArcOn", ARC_ON_SRC), ("ArcOff", ARC_OFF_SRC)):
        path = os.path.join(GEN_DIR, name + ".py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(src)
        files[name] = path
    for shot, f0, f1, cam, aim, lens in SHOTS:
        name = "View_" + shot
        rows = [[round(v, 6) for v in row] for row in view_pose(cam, aim).rows]
        path = os.path.join(GEN_DIR, name + ".py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(VIEW_SRC.format(name=name, shot=shot, f0=f0, f1=f1, cam=list(cam), aim=list(aim), lens=lens,
                                     rows=rows))
        files[name] = path
    for old in glob.glob(os.path.join(GEN_DIR, "View_*.py")):
        if os.path.basename(old)[:-3] not in files:
            os.remove(old)
    return files


# =====================================================================================================
# 4. RoboDK builders
# =====================================================================================================
class Cell:
    """Items created in the station (filled by build())."""

    def __init__(self):
        self.station = None
        self.frame = None          # "Cell" reference frame at the station origin
        self.track = None
        self.robot = None
        self.torch = None
        self.tilt = None
        self.rot = None
        self.spool = None
        self.curves = {}
        self.programs = {}
        self.targets = {}
        self.cameras = {}
        self.macros = {}


def add_shape_object(RDK, name, shapes, parent=0):
    """One object made of several coloured shapes: shapes = [(tris, [r,g,b,a]), ...]."""
    payload = []
    for tris, color in shapes:
        payload.append(tris_to_mat(tris))
        payload.append(list(color))
    item = RDK.AddShape(payload, parent)
    item.setName(name)
    return item


def mech_shapes(shapes, base):
    """Shapes of a mechanism link, built in world mm; see MECH_GEOMETRY_IN_BASE_FRAME."""
    if not MECH_GEOMETRY_IN_BASE_FRAME:
        return shapes
    Ti = invH(base)
    return [(tris_transform(tris, Ti), color) for tris, color in shapes]


def base_frame_of(RDK, mech):
    """Reference frame that carries a robot/mechanism (its parent frame); created if it has none."""
    par = mech.Parent()
    if par.Valid() and par.Type() == rl.ITEM_TYPE_FRAME:
        return par
    fr = RDK.AddFrame(mech.Name() + " Base")
    mech.setParent(fr)
    return fr


def flange_at_zero(mech, ndof):
    """Pose of a mechanism flange w.r.t. its base at joints = 0 (identity for the mechanisms built here)."""
    return mech.SolveFK([0.0] * ndof)


def mount_on_flange(RDK, item, mech, ndof, pose_in_mech_base):
    """Attach item (object, or a robot/mechanism via its base frame) to the flange of `mech`, so that with
    the mechanism at joints = 0 the item sits at pose_in_mech_base (w.r.t. the mechanism base)."""
    rel = invH(flange_at_zero(mech, ndof)) * pose_in_mech_base
    holder = item if item.Type() in (rl.ITEM_TYPE_OBJECT, rl.ITEM_TYPE_FRAME) else base_frame_of(RDK, item)
    holder.setParent(mech)
    holder.setPose(rel)
    return holder


def find_library_file(RDK, patterns):
    """First file in PATH_LIBRARY (recursively) matching one of the glob patterns, else None."""
    lib = RDK.getParam("PATH_LIBRARY")
    if not lib:
        return None, lib
    for pat in patterns:
        for root in (lib, os.path.join(lib, "**")):
            hits = sorted(glob.glob(os.path.join(root, pat), recursive=True))
            hits += sorted(glob.glob(os.path.join(root, pat.lower()), recursive=True))
            if hits:
                return hits[0], lib
    return None, lib


def set_state(cell, track_y, tilt_deg, rot_deg):
    """Put the external axes in a given state (used while solving targets)."""
    cell.track.setJoints([track_y])
    cell.tilt.setJoints([tilt_deg])
    cell.rot.setJoints([rot_deg])


# ---------------------------------------------------------------- track + robot
def add_track(RDK, cell):
    """Linear track (7th axis) along world Y at x = TRACK_X: 1-axis translation mechanism."""
    if TRACK_LIBRARY_GLOB:
        path, _ = find_library_file(RDK, [TRACK_LIBRARY_GLOB])
        if path:
            track = RDK.AddFile(path)
            base_frame_of(RDK, track).setPose(track_base_pose())
            cell.track = track
            return track
    base = track_base_pose()
    length = TRACK_Y_MAX - TRACK_Y_MIN + 1200.0
    bed = add_shape_object(RDK, "Track bed", mech_shapes([
        (box_tris((TRACK_X, 0, 110), (900, length, 220)), [0.17, 0.18, 0.21, 1]),
        (box_tris((TRACK_X - 300, 0, 245), (60, length - 100, 50)), [0.75, 0.75, 0.78, 1]),
        (box_tris((TRACK_X + 300, 0, 245), (60, length - 100, 50)), [0.75, 0.75, 0.78, 1]),
        (box_tris((TRACK_X - 420, 0, 240), (40, length - 200, 40)), [0.1, 0.1, 0.1, 1]),
        (box_tris((TRACK_X + 620, 0, 80), (220, length - 300, 160)), [0.2, 0.2, 0.2, 1]),   # cable chain trough
    ], base))
    carriage = add_shape_object(RDK, "Track carriage", mech_shapes([                           # at joint 0 (y = 0)
        (box_tris((TRACK_X, 0, 270 + (TRACK_TOP_Z - 270) / 2), (860, 1050, TRACK_TOP_Z - 270)), [0.22, 0.24, 0.28, 1]),
        (box_tris((TRACK_X, 0, TRACK_TOP_Z - 15), (780, 780, 30)), [0.8, 0.8, 0.82, 1]),
    ], base))
    track = RDK.BuildMechanism(rl.MAKE_ROBOT_1T, [bed, carriage], [], [0.0], [ROBOT_HOME_Y], [1],
                               [TRACK_Y_MIN], [TRACK_Y_MAX], base, eye(4), "Linear track Y")
    if not track.Valid():
        raise RuntimeError("BuildMechanism failed for the linear track")
    track.setJoints([ROBOT_HOME_Y])
    cell.track = track
    return track


def add_robot(RDK, cell):
    """ABB IRB 4600-40/2.55 from the local library, mounted on the track carriage with yaw 180 deg."""
    path, lib = find_library_file(RDK, ROBOT_LIBRARY_GLOBS)
    if path is None:
        msg = ("ABB IRB 4600-40/2.55 not found in the RoboDK library folder (%s). "
               "Download it from the online library (File > Open online library) and re-run, "
               "or pick the .robot file now." % lib)
        print(msg)
        RDK.ShowMessage(msg, False)
        from robodk import robodialogs
        path = robodialogs.getOpenFileName(lib or "", "", "Select the ABB IRB 4600 .robot file", ".robot",
                                           [("RoboDK robot", ".robot")])
        if not path:
            raise RuntimeError("No robot file selected")
    robot = RDK.AddFile(path)
    if not robot.Valid():
        raise RuntimeError("Could not load robot file: " + path)
    robot.setName("ABB IRB 4600-40/2.55")
    # robot base w.r.t. the track base at joint 0 (the carriage is at world y = 0 there)
    rel = invH(track_base_pose()) * robot_base_pose_world(0.0)
    mount_on_flange(RDK, robot, cell.track, 1, rel)
    robot.setJointsHome(ROBOT_Q_HOME)
    robot.setJoints(ROBOT_Q_HOME)
    cell.robot = robot
    return robot


def add_torch(RDK, cell):
    """MIG torch: STL generated in the flange frame, loaded as a tool of the robot, TCP from tcp_pose()."""
    os.makedirs(GEN_DIR, exist_ok=True)
    stl = os.path.join(GEN_DIR, "torch.stl")
    write_stl(stl, torch_mesh_triangles(), "MIG torch")
    torch = RDK.AddFile(stl, cell.robot)          # an object attached to a robot becomes a tool
    torch.setName("MIG torch")
    torch.setPoseTool(tcp_pose())
    torch.setColor([0.15, 0.15, 0.17, 1.0])
    cell.robot.setPoseTool(torch)
    cell.torch = torch
    return torch


# ---------------------------------------------------------------- positioner + spool
def positioner_static_shapes():
    """Pedestal, bed, bearing towers (x = +/-POS_TOWER_X), caps and the tilt drive (world mm), mirroring
    cell/positioner.py."""
    zt = POS_TILT_AXIS_Z
    hw = POS_TILT_HALF_WIDTH
    tx = POS_TOWER_X
    fx, fy, fz = POS_PEDESTAL_FOOTPRINT
    tw = 260.0                        # tower thickness (x)
    z_bed = 330.0                     # top of the welded bed
    bed_x = 2 * tx + tw
    blue = [0.15, 0.33, 0.56, 1]
    dark = [0.15, 0.15, 0.16, 1]
    grey = [0.2, 0.21, 0.23, 1]
    yellow = [0.95, 0.75, 0.05, 1]
    gx = tx + 160 + 70                # tilt gearbox centre x (right side)
    zm = zt - 170 - 160               # tilt motor centre z
    shapes = [
        (box_tris((0, 0, 15), (2 * hw + 500, fy + 250, 30)), dark),                        # base plate
        (box_tris((0, 0, (z_bed + 30) / 2), (bed_x, fy, z_bed - 30)), blue),               # welded bed
        (box_tris((0, 0, z_bed + fz / 2), (fx, fy, fz)), blue),                             # pedestal
        (box_tris((0, 0, 90), (bed_x + 20, fy + 20, 50)), yellow),                          # hazard stripe
    ]
    for s in (-1, 1):
        shapes.append((box_tris((s * tx, 0, (zt + z_bed) / 2), (tw, 420, zt - z_bed)), blue))            # tower
        shapes.append((cylinder_tris((s * tx - 133, 0, zt), (s * tx + 133, 0, zt), 210, 48), blue))     # tower head
        shapes.append((cylinder_tris((s * (tx + 60) - 100, 0, zt), (s * (tx + 60) + 100, 0, zt), 150), blue))  # cap
    xf = -(tx + 160)
    shapes.append((cylinder_tris((xf - 20, 0, zt), (xf, 0, zt), 100, 32), dark))                       # end cover (left)
    shapes.append((cylinder_tris((tx + 160, 0, zt), (tx + 160 + 30, 0, zt), 115), dark))               # drive adapter
    shapes.append((box_tris((gx, 0, zt), (140, 300, 340)), blue))                                      # tilt gearbox
    shapes.append((cylinder_tris((gx, 0, zt - 170 - 20), (gx, 0, zt - 170), 100), dark))              # motor flange
    shapes.append((cylinder_tris((gx, 0, zm - 160), (gx, 0, zm + 160), 85), grey))                      # tilt motor
    shapes.append((cylinder_tris((gx, 0, zm - 160 - 40), (gx, 0, zm - 160), 92), dark))               # fan cover
    shapes.append((box_tris((gx, 115, zm + 40), (90, 60, 80)), grey))                                   # terminal box
    return shapes


def positioner_cradle_shapes():
    """U-cradle (tilt link at tilt = 0, world mm): trunnion hubs, arms, cross beam, rotary drive."""
    zt = POS_TILT_AXIS_Z
    hw = POS_TILT_HALF_WIDTH
    zf = zt + POS_FACEPLATE_OFFSET
    blue = [0.15, 0.33, 0.56, 1]
    dark = [0.15, 0.15, 0.16, 1]
    grey = [0.2, 0.21, 0.23, 1]
    steel = [0.7, 0.7, 0.72, 1]
    shapes = []
    for s in (-1, 1):
        hx = s * (hw - 80)
        shapes.append((cylinder_tris((hx - 100, 0, zt), (hx + 100, 0, zt), 140), steel))                       # hub
        shapes.append((box_tris((s * (hw - 140), 0, zt + (POS_FACEPLATE_OFFSET - 60) / 2),
                                (120, 220, POS_FACEPLATE_OFFSET + 120)), blue))                                  # arm
    shapes.append((box_tris((0, 0, zf - 140), (2 * hw - 200, 260, 120)), blue))                                 # cross beam
    shapes.append((cylinder_tris((0, 0, zf - 140), (0, 0, zf - 60), 200), grey))                                # rotary drive
    shapes.append((cylinder_tris((280, -360, zf - 140), (280, -80, zf - 140), 75), grey))                       # rotary motor
    shapes.append((cylinder_tris((280, -390, zf - 140), (280, -360, zf - 140), 80), dark))                      # motor fan
    return shapes


def add_positioner(RDK, cell):
    """2-axis tilt/rotate positioner as two chained 1-axis rotation mechanisms: tilt about world X through
    (0,0,POS_TILT_AXIS_Z) (+90 -> faceplate faces +Y), faceplate POS_FACEPLATE_OFFSET above the tilt axis rotating
    about its normal."""
    if POSITIONER_LIBRARY_GLOB:
        path, _ = find_library_file(RDK, [POSITIONER_LIBRARY_GLOB])
        if path:
            pos = RDK.AddFile(path)
            pos.setName("Positioner (library)")
            cell.tilt = cell.rot = pos
            return pos
    zt = POS_TILT_AXIS_Z
    dark = [0.15, 0.15, 0.16, 1]
    steel = [0.7, 0.7, 0.72, 1]
    cast = [0.45, 0.46, 0.48, 1]
    yellow = [0.95, 0.75, 0.05, 1]
    tbase = tilt_base_pose()
    pedestal = add_shape_object(RDK, "Positioner pedestal", mech_shapes(positioner_static_shapes(), tbase))
    cradle = add_shape_object(RDK, "Positioner cradle", mech_shapes(positioner_cradle_shapes(), tbase))
    tilt = RDK.BuildMechanism(rl.MAKE_ROBOT_1R, [pedestal, cradle], [], [0.0], [0.0], [1],
                              [TILT_LIMITS[0]], [TILT_LIMITS[1]], tbase, eye(4), "Positioner tilt")
    if not tilt.Valid():
        raise RuntimeError("BuildMechanism failed for the positioner tilt axis")
    zf = zt + POS_FACEPLATE_OFFSET
    R = POS_FACEPLATE_RADIUS
    rbase = rot_base_pose_world()
    housing = add_shape_object(RDK, "Faceplate drive", mech_shapes([
        (cylinder_tris((0, 0, zf - 62), (0, 0, zf - 58), 210), dark),
    ], rbase))
    plate_shapes = [
        (cylinder_tris((0, 0, zf - 60), (0, 0, zf), R, 48), cast),
        (cylinder_tris((0, 0, zf), (0, 0, zf + POS_FIXTURE_THICK), 215, 48), yellow),      # fixture adapter ring
    ]
    for k in range(4):                                                                       # radial T-slots
        plate_shapes.append((box_tris((0, 0, zf - 1), (2 * R - 60, 16, 4), rotz(k * math.pi / 4)), dark))
    for k in range(6 if FLANGE_BOLTS else 0):                                                # flange bolts
        a = 2 * math.pi * (2 * k + 0.5) / 12
        x, y = 177.5 * math.cos(a), 177.5 * math.sin(a)
        plate_shapes.append((cylinder_tris((x, y, zf + POS_FIXTURE_THICK + 28), (x, y, zf + POS_FIXTURE_THICK + 44), 8, 6), steel))
    faceplate = add_shape_object(RDK, "Faceplate", mech_shapes(plate_shapes, rbase))
    rot = RDK.BuildMechanism(rl.MAKE_ROBOT_1R, [housing, faceplate], [], [0.0], [ROT_A0], [1],
                             [ROT_LIMITS[0]], [ROT_LIMITS[1]], rbase, eye(4), "Positioner faceplate")
    if not rot.Valid():
        raise RuntimeError("BuildMechanism failed for the positioner faceplate axis")
    # chain: faceplate mechanism rides on the tilt flange
    mount_on_flange(RDK, rot, tilt, 1, invH(tbase) * rbase)
    tilt.setJoints([0.0])
    rot.setJoints([ROT_A0])
    cell.tilt, cell.rot = tilt, rot
    return tilt, rot


def add_spool(RDK, cell):
    """Spool STL (spool-local frame, mm) mounted POS_FIXTURE_THICK above the faceplate, plus seam curve objects."""
    if not os.path.exists(SPOOL_STL):
        raise RuntimeError("spool.stl not found next to build_station.py (export it from Blender, see README)")
    spool = RDK.AddFile(SPOOL_STL)
    spool.setName("Spool DN250")
    spool.setColor([0.45, 0.42, 0.40, 1.0])
    on_plate = transl(0, 0, POS_FIXTURE_THICK)
    mount_on_flange(RDK, spool, cell.rot, 1, on_plate)
    cell.spool = spool
    for which, color in (("A", [1.0, 0.2, 0.1, 1.0]), ("B", [0.1, 0.5, 1.0, 1.0])):
        curve = RDK.AddCurve(seam_points_local(which), 0, False, rl.PROJECTION_NONE)
        curve.setName("Seam %s curve" % which)
        curve.setColorCurve(color)
        mount_on_flange(RDK, curve, cell.rot, 1, on_plate)
        cell.curves[which] = curve
    return spool


def add_curve_follow_projects(RDK, cell):
    """One 'Curve Follow Project' per seam (Utilities > Curve follow project). The generated program is not part
    of DemoCycle; open the project to tune approach/retract and tool orientation (path to tool = push angle)."""
    for which, curve in cell.curves.items():
        mp = RDK.AddMachiningProject("Seam %s curve follow" % which, cell.robot)
        mp.setPoseFrame(cell.frame)
        mp.setPoseTool(cell.torch)
        mp.setPose(roty(-PUSH_ANGLE_DEG * D2R))      # path-to-tool orientation: push angle
        try:
            prog, status = mp.setMachiningParameters(part=curve, params="ReorderAuto=0")
            cell.programs["CurveFollow_" + which] = prog
        except Exception as exc:               # solving can fail on unreachable sections; keep the project
            print("Curve follow project %s: %s" % (which, exc))


# ---------------------------------------------------------------- targets & programs
class Prog:
    """Thin helper that adds instructions to a RoboDK program item."""

    def __init__(self, RDK, cell, name, mech, is_robot=True):
        self.RDK, self.cell = RDK, cell
        self.item = RDK.AddProgram(name, mech)
        self.mech = mech
        self.name = name
        if is_robot:
            self.item.setPoseFrame(cell.frame)
            self.item.setPoseTool(cell.torch)
            self.item.setRounding(5.0)
        cell.programs[name] = self.item

    def call(self, name):
        self.item.RunInstruction(name, rl.INSTRUCTION_CALL_PROGRAM)

    def comment(self, text):
        self.item.RunInstruction(text, rl.INSTRUCTION_COMMENT)

    def air(self):
        self.item.setSpeed(AIR_SPEED_MMS, AIR_JOINT_DEGS)
        self.item.setRounding(10.0)

    def weld(self):
        self.item.setSpeed(WELD_SPEED_MMS, 20.0)
        self.item.setRounding(1.0)

    def arc(self, on):
        self.item.setDO("WeldArc", 1 if on else 0)
        self.call("ArcOn" if on else "ArcOff")

    def joint_target(self, name, joints):
        t = self.RDK.AddTarget(name, self.cell.frame, self.mech)
        t.setAsJointTarget()
        t.setJoints(list(joints))
        self.cell.targets[name] = t
        return t

    def cart_target(self, name, pose, seed=None):
        """Cartesian target; if the robot can solve it now, the joint configuration is stored with the target."""
        t = self.RDK.AddTarget(name, self.cell.frame, self.mech)
        t.setAsCartesianTarget()
        q = solve_robot(self.cell, pose, seed)
        if q is not None:
            t.setJoints(q)
        t.setPose(pose)
        self.cell.targets[name] = t
        return t

    def movej(self, target):
        self.item.MoveJ(target)

    def movel(self, target):
        self.item.MoveL(target)


def solve_robot(cell, pose_cell, seed=None):
    """Joints reaching TCP pose (w.r.t. the Cell frame) with the track where it currently is; None if unreachable."""
    robot = cell.robot
    ref = invH(robot.PoseAbs())                  # Cell frame (= station origin) w.r.t. the robot base
    q = robot.SolveIK(pose_cell, seed if seed is not None else ROBOT_Q_HOME, tcp_pose(), ref)
    ql = list(q.tolist()) if isinstance(q, Mat) else list(q)
    if len(ql) < 6:
        return None
    ql = [float(v) for v in ql[:6]]
    robot.setJoints(ql)
    return ql


def add_mechanism_programs(RDK, cell):
    """Track and positioner sub-programs (joint targets) used by DemoCycle."""
    for name, y in (("Track_Home", ROBOT_HOME_Y), ("Track_ToA", TRACK_A), ("Track_ToB1", TRACK_B1), ("Track_ToB2", TRACK_B2)):
        p = Prog(RDK, cell, name, cell.track, is_robot=False)
        p.item.setSpeedJoints(TRACK_SPEED_MMS)
        p.movej(p.joint_target(name + "_T", [y]))
    # tilt: 0 -> +90 (faceplate faces +Y), 180 deg index = +90 -> -90 over the top, back to 0 at the end
    for name, deg in (("Pos_TiltUp", TILT_B1), ("Pos_Index180", TILT_B2), ("Pos_TiltDown", 0.0)):
        p = Prog(RDK, cell, name, cell.tilt, is_robot=False)
        p.item.setSpeedJoints(POS_INDEX_DEGS)
        p.movej(p.joint_target(name + "_T", [deg]))
    for name, deg, speed in (("Pos_RotStart", ROT_A0, POS_INDEX_DEGS),
                             ("Pos_RotSeamA_1", ROT_A_MID, POS_WELD_DEGS),
                             ("Pos_RotSeamA_2", ROT_A1, POS_WELD_DEGS)):
        p = Prog(RDK, cell, name, cell.rot, is_robot=False)
        p.item.setSpeedJoints(speed)
        p.movej(p.joint_target(name + "_T", [deg]))


def add_robot_programs(RDK, cell):
    """Robot sub-programs for the choreography plus the explicit MoveL seam programs."""
    ch = choreography()
    robot = cell.robot

    # ---- home
    p = Prog(RDK, cell, "Robot_Home", robot)
    p.air()
    home = p.joint_target("Home", ROBOT_Q_HOME)
    p.movej(home)

    # ---- transit (park) pose: torch down over the track, outside the part sweep. Solved once at the seam-A
    # station and stored as a JOINT target, so it is the same arm configuration at every track station.
    set_state(cell, TRACK_A, TILT_B1, ROT_A0)
    robot.setJoints(ROBOT_Q_HOME)
    q_transit = solve_robot(cell, ch["transit"], ROBOT_Q_HOME)
    p = Prog(RDK, cell, "Robot_Transit", robot)
    p.comment("Transit pose: torch down, TCP at x=%.0f z=%.0f over the track" % TRANSIT_TCP)
    p.air()
    if q_transit is not None:
        transit = p.joint_target("Transit", q_transit)
    else:
        print("Transit pose unreachable at track y=%.0f - stored as a Cartesian target" % TRACK_A)
        transit = p.cart_target("Transit", ch["transit"])
    p.movej(transit)

    # ---- seam A: approach (via transit), laser scan, static torch while the positioner rotates
    set_state(cell, TRACK_A, TILT_B1, ROT_A0)
    p = Prog(RDK, cell, "Robot_ApproachA", robot)
    p.air()
    p.movej(transit)
    appA = p.cart_target("ApproachA", ch["approachA"], q_transit)
    p.movej(appA)

    p = Prog(RDK, cell, "Robot_ScanA", robot)
    p.comment("Laser seam search: +/-%.0f mm along the seam axis, %.0f mm above seam A" % (SCAN_HALF, SCAN_HEIGHT))
    p.air()
    scan = [p.cart_target("ScanA_%02d" % k, T) for k, T in enumerate(ch["scan"])]
    p.movel(scan[0])
    p.item.setDO("LaserScan", 1)
    p.item.setSpeed(60.0)
    for t in scan[1:]:
        p.movel(t)
    p.item.setDO("LaserScan", 0)

    p = Prog(RDK, cell, "Robot_WeldA", robot)
    p.comment("Seam A (flange - elbow): torch static at 12 o'clock, the positioner rotates the spool %.0f deg" % (ROT_A1 - ROT_A0))
    p.air()
    weldA = p.cart_target("WeldA", ch["weldA"])
    p.movel(weldA)
    p.arc(True)
    p.call("Pos_RotSeamA_1")
    p.call("View_C4_midA")
    p.call("Pos_RotSeamA_2")
    p.arc(False)
    p.air()
    liftA = p.cart_target("LiftA", lifted(ch["weldA"], LIFT_PARK))
    p.movel(liftA)
    p.movej(transit)

    # explicit MoveL alternative for seam A (part static, robot goes around); not used by DemoCycle
    p = Prog(RDK, cell, "Robot_WeldA_MoveL", robot)
    p.comment("Alternative: robot drives the torch around seam A (positioner static at tilt %.0f / rot %.0f)" % (TILT_B1, ROT_A0))
    add_movel_seam(p, "A_", ch["weldA_movel"], ch["weldA_movel_n"], LIFT_PARK, LIFT_PARK)

    # ---- seam B sectors 1-2 (tilt +90, pipe along +X, track at +761)
    set_state(cell, TRACK_B1, TILT_B1, ROT_A1)
    robot.setJoints(q_transit or ROBOT_Q_HOME)
    p = Prog(RDK, cell, "Robot_ApproachB1", robot)
    p.air()
    p.movej(p.cart_target("ApproachB1", radial(ch["sector1"][0], ch["sector1_n"][0], LIFT_APPROACH), q_transit))
    p = Prog(RDK, cell, "Robot_WeldB_S1", robot)
    p.comment("Seam B sector 1: 12 o'clock -> -Y side (%.0f..%.0f deg), then via 1:30 o'clock retracted" % (SECTOR_A0, SECTOR_A1))
    add_movel_seam(p, "S1_", ch["sector1"], ch["sector1_n"], None, LIFT_APPROACH, via=ch["sector1_via"])
    p = Prog(RDK, cell, "Robot_WeldB_S2", robot)
    p.comment("Seam B sector 2: 12 o'clock -> +Y side, then retract to the transit pose")
    add_movel_seam(p, "S2_", ch["sector2"], ch["sector2_n"], LIFT_APPROACH, LIFT_PARK, end=transit)

    # ---- seam B sectors 3-4 (after the 180 deg index: tilt -90, track at -761)
    set_state(cell, TRACK_B2, TILT_B2, ROT_A1)
    robot.setJoints(q_transit or ROBOT_Q_HOME)
    p = Prog(RDK, cell, "Robot_ApproachB2", robot)
    p.air()
    p.movej(p.cart_target("ApproachB2", radial(ch["sector3"][0], ch["sector3_n"][0], LIFT_APPROACH), q_transit))
    p = Prog(RDK, cell, "Robot_WeldB_S3", robot)
    p.comment("Seam B sector 3 (after the 180 deg index): 12 o'clock -> -Y side, then via 1:30 o'clock retracted")
    add_movel_seam(p, "S3_", ch["sector3"], ch["sector3_n"], None, LIFT_APPROACH, via=ch["sector3_via"])
    p = Prog(RDK, cell, "Robot_WeldB_S4", robot)
    p.comment("Seam B sector 4: 12 o'clock -> +Y side, then retract to the transit pose")
    add_movel_seam(p, "S4_", ch["sector4"], ch["sector4_n"], LIFT_APPROACH, LIFT_PARK, end=transit)

    # restore the start state
    set_state(cell, ROBOT_HOME_Y, 0.0, ROT_A0)
    robot.setJoints(ROBOT_Q_HOME)


def add_movel_seam(p, prefix, poses, normals, lift_in, lift_out, via=None, end=None):
    """MoveL weld along `poses`: optional radial approach from lift_in (along the surface normal) before the first
    point, arc on/off, radial retract lift_out after the last point, optional MoveJ to a via target and/or a final
    target (e.g. the transit joint target)."""
    p.air()
    if lift_in is not None:
        p.movej(p.cart_target(prefix + "app", radial(poses[0], normals[0], lift_in)))
    start = p.cart_target(prefix + "start", poses[0])
    p.movel(start)
    p.weld()
    p.arc(True)
    for k, T in enumerate(poses[1:], 1):
        p.movel(p.cart_target(prefix + "%03d" % k, T))
    p.arc(False)
    p.air()
    p.movel(p.cart_target(prefix + "lift", radial(poses[-1], normals[-1], lift_out)))
    if via is not None:
        p.movej(p.cart_target(prefix + "via", via))
    if end is not None:
        p.movej(end)


def add_demo_cycle(RDK, cell):
    """Main program: the storyboard sequence with view switches (View_* macros) between shots."""
    p = Prog(RDK, cell, "DemoCycle", cell.robot)
    p.comment("Robotic pipe-spool welding cell demo: seam A (rotating part) + seam B (4 sectors, 180 deg tilt index)")
    p.call("View_C1_wide")
    p.call("Cell_Home")
    p.item.Pause(1500)
    p.call("View_C2_setup")
    p.call("Pos_TiltUp")
    p.call("Track_ToA")
    p.call("Robot_ApproachA")
    p.call("View_C3a_laser")
    p.call("Robot_ScanA")
    p.call("View_C3b_closeA")
    p.call("Robot_WeldA")                 # switches to View_C4_midA half-way; ends in the transit pose
    p.call("View_C5_reposition")
    p.call("Track_ToB1")
    p.call("Robot_ApproachB1")
    p.call("View_C6a_closeB1")
    p.call("Robot_WeldB_S1")
    p.call("View_C6b_closeB1")
    p.call("Robot_WeldB_S2")              # ends in the transit pose
    p.call("View_C7_index")
    p.call("Pos_Index180")                # tilt +90 -> -90 over the top (pipe stays along +X)
    p.call("Track_ToB2")
    p.call("Robot_ApproachB2")
    p.call("View_C8a_closeB2")
    p.call("Robot_WeldB_S3")
    p.call("View_C8b_closeB2")
    p.call("Robot_WeldB_S4")              # ends in the transit pose
    p.call("View_C9_final")
    p.call("Robot_Home")
    p.call("Track_Home")
    p.call("Pos_TiltDown")
    p.item.Pause(2000)

    h = Prog(RDK, cell, "Cell_Home", cell.robot)
    h.call("Robot_Home")
    h.call("Track_Home")
    h.call("Pos_TiltDown")
    h.call("Pos_RotStart")
    h.item.setDO("WeldArc", 0)
    h.item.setDO("LaserScan", 0)


def add_macros(RDK, cell):
    """ArcOn / ArcOff and View_* Python programs (files generated next to this script)."""
    for name, path in write_macros().items():
        item = RDK.AddFile(path)
        item.setName(name)
        cell.macros[name] = item


# ---------------------------------------------------------------- cameras & dressing
def add_cameras(RDK, cell):
    """One reference frame per storyboard shot (+Z = viewing direction); optional 2D camera windows."""
    folder = RDK.AddFrame("Cameras")
    folder.setVisible(False)
    for name, f0, f1, cam, aim, lens in SHOTS:
        fr = RDK.AddFrame("Cam_" + name, folder)
        fr.setPose(camera_frame_pose(cam, aim))
        fr.setVisible(False)
        cell.cameras[name] = fr
        if OPEN_2D_CAMERAS:
            fov = lens_to_fov_deg(lens)
            RDK.Cam2D_Add(fr, "FOCAL_LENGTH=%.1f FOV=%.1f FAR_LENGTH=25000 SIZE=960x540 NO_TASKBAR" % (lens, fov))


def fence_segments(a, b, gaps):
    """Split the interval [a, b] by [(g0, g1), ...] gaps -> list of (start, end)."""
    segs = [(a, b)]
    for g0, g1 in gaps:
        out = []
        for s0, s1 in segs:
            if g1 <= s0 or g0 >= s1:
                out.append((s0, s1))
            else:
                if g0 > s0:
                    out.append((s0, g0))
                if g1 < s1:
                    out.append((g1, s1))
        segs = out
    return segs


def environment_shapes():
    """Cell dressing as coloured boxes/cylinders (world mm): fence, equipment, hall columns, crane beam."""
    post_c = [0.95, 0.75, 0.05, 1]
    panel_c = [0.45, 0.5, 0.55, 0.35]
    red = [0.55, 0.1, 0.1, 1]
    grey = [0.6, 0.62, 0.64, 1]
    dark = [0.2, 0.2, 0.22, 1]
    wood = [0.6, 0.45, 0.25, 1]
    shapes = []
    x0, x1 = FENCE_X
    y0, y1 = FENCE_Y
    door = FENCE_DOOR
    dgap = (door["center_x"] - door["width"] / 2, door["center_x"] + door["width"] / 2)
    lgap = (-LOAD_OPENING["width"] / 2, LOAD_OPENING["width"] / 2)
    sides = [  # (fixed axis, fixed value, moving axis range, gaps)
        ("y", y0, (x0, x1), [dgap]),                     # -Y side with the sliding door
        ("y", y1, (x0, x1), []),
        ("x", x0, (y0, y1), [lgap]),                     # -X loading side with light curtain opening
        ("x", x1, (y0, y1), []),
    ]
    for axis, val, rng, gaps in sides:
        for s0, s1 in fence_segments(rng[0], rng[1], gaps):
            n = max(1, int(round((s1 - s0) / 2000.0)))
            for k in range(n + 1):
                u = s0 + (s1 - s0) * k / n
                c = (u, val, FENCE_H / 2) if axis == "y" else (val, u, FENCE_H / 2)
                shapes.append((box_tris(c, (60, 60, FENCE_H)), post_c))
            mid = ((s0 + s1) / 2, val, FENCE_H / 2 + 50) if axis == "y" else (val, (s0 + s1) / 2, FENCE_H / 2 + 50)
            size = (s1 - s0, 20, FENCE_H - 250) if axis == "y" else (20, s1 - s0, FENCE_H - 250)
            shapes.append((box_tris(mid, size), panel_c))
    # light curtain posts at the loading opening
    for y in lgap:
        shapes.append((box_tris((x0, y, 700), (80, 80, 1400)), [0.9, 0.8, 0.1, 1]))
    for eq, color in ((CONTROLLER_CABINET, [0.85, 0.85, 0.87, 1]), (WELD_POWER_SOURCE, [0.75, 0.12, 0.1, 1])):
        px, py, pz = eq["pos"]
        sx, sy, sz = eq["size"]
        shapes.append((box_tris((px, py, pz + sz / 2), (sx, sy, sz)), color))
    wd = WIRE_DRUM
    shapes.append((cylinder_tris((wd["pos"][0], wd["pos"][1], 0), (wd["pos"][0], wd["pos"][1], wd["height"]), wd["radius"]), dark))
    tc = TORCH_CLEANER
    shapes.append((box_tris((tc["pos"][0], tc["pos"][1], tc["height"] / 2), (400, 400, tc["height"])), grey))
    fh = FUME_HOOD
    shapes.append((box_tris((fh["pos"][0], fh["pos"][1], fh["pos"][2] + fh["size"][2] / 2), fh["size"]), grey))
    shapes.append((cylinder_tris((fh["pos"][0], fh["pos"][1], fh["pos"][2] + fh["size"][2]),
                                 (fh["pos"][0], fh["pos"][1], HALL_SIZE[2] - 500), 160), grey))
    for pal in (PARTS_PALLET, FINISHED_RACK):
        px, py, _ = pal["pos"]
        shapes.append((box_tris((px, py, 72), (1200, 1000, 144)), wood))
        shapes.append((cylinder_tris((px - 500, py, 144 + PIPE_OD / 2), (px + 500, py, 144 + PIPE_OD / 2), PIPE_OD / 2, 20), [0.4, 0.37, 0.35, 1]))
    # hall: red steel columns on a COLUMN_SPACING grid (only the closest ring) and the yellow crane bridge
    for cx in (-COLUMN_SPACING, 2 * COLUMN_SPACING):
        for cy in (-COLUMN_SPACING, COLUMN_SPACING):
            shapes.append((box_tris((cx, cy, HALL_SIZE[2] / 2), (300, 300, HALL_SIZE[2])), red))
    shapes.append((box_tris((COLUMN_SPACING / 2, 4000, HALL_SIZE[2] - 1400), (3 * COLUMN_SPACING + 600, 500, 700)), [0.95, 0.72, 0.05, 1]))
    shapes.append((box_tris((0, 0, -25), (HALL_SIZE[0] * 0.6, HALL_SIZE[1] * 0.6, 50)), [0.52, 0.5, 0.48, 1]))   # concrete slab
    return shapes


def add_environment(RDK, cell):
    add_shape_object(RDK, "Cell dressing", environment_shapes())


# ---------------------------------------------------------------- station
def build(RDK):
    """Build the whole station in the connected RoboDK instance; returns the Cell with all items."""
    cell = Cell()
    RDK.Render(False)
    cell.station = RDK.AddStation(STATION_NAME)
    cell.frame = RDK.AddFrame("Cell")
    cell.frame.setPose(eye(4))
    add_environment(RDK, cell)
    add_track(RDK, cell)
    add_robot(RDK, cell)
    add_torch(RDK, cell)
    add_positioner(RDK, cell)
    add_spool(RDK, cell)
    add_macros(RDK, cell)
    add_cameras(RDK, cell)
    add_mechanism_programs(RDK, cell)
    add_robot_programs(RDK, cell)
    add_demo_cycle(RDK, cell)
    add_curve_follow_projects(RDK, cell)
    for t in cell.targets.values():               # hide the target markers one by one: Program.ShowTargets(False) takes the
        try:                                      # targets into that program in RoboDK 5.9 and the other programs lose them
            t.setVisible(False, False)
        except TypeError:
            t.setVisible(False)
    for name in ("Robot_Transit", "Robot_ApproachA", "Robot_ScanA", "Robot_WeldA", "Robot_ApproachB1", "Robot_WeldB_S1",
                 "Robot_WeldB_S2", "Robot_ApproachB2", "Robot_WeldB_S3", "Robot_WeldB_S4", "Robot_WeldA_MoveL"):
        prog = cell.programs[name]
        try:
            ok, t_s, dist, frac, problems = prog.Update(rl.COLLISION_OFF)
            if frac < 1.0:
                print("Program %s: %.0f%% of the path is valid - %s" % (name, frac * 100, problems))
        except Exception as exc:
            print("Program %s: Update failed (%s)" % (name, exc))
    RDK.setSimulationSpeed(1.0)
    RDK.setViewPose(view_pose(SHOTS[0][3], SHOTS[0][4]))
    RDK.Render(True)
    if SAVE_RDK:
        RDK.Save(os.path.join(HERE, STATION_NAME + ".rdk"), cell.station)
    return cell


def main():
    RDK = rl.Robolink()
    cell = build(RDK)
    print("Station '%s' built: %d programs, %d targets. Run 'DemoCycle' (Tools > Record to capture the video)."
          % (STATION_NAME, len(cell.programs), len(cell.targets)))


if __name__ == "__main__":
    main()
