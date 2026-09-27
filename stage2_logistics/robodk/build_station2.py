#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the FULL-CYCLE pipe-spool station in RoboDK: the stage-1 welding cell + the stage-2 logistics zone.

Run from RoboDK (File > Open this .py, right click > Run Python script) or from a terminal while RoboDK is running:
    python build_station2.py                    build the station "PipeSpool_FullCycle" (saved next to this script)
    python3 build_station2.py --export-plan     refresh plan2_robodk.json from stage2_logistics/plan2.py (needs the
                                                stage-2 python: numpy + bpy); add --fresh to re-solve the plan

What is built (mm / deg, RoboDK convention; the layout comes from stage2_logistics/layout2.py in metres):
  * the stage-1 cell, built by the stage-1 functions of demo_video/robodk/build_station.py (imported, not modified):
    welding robot ABB IRB 4600 on its Y track, MIG torch, 2-axis positioner, cell dressing, cameras, the stage-1
    programs incl. "DemoCycle" (the welding choreography).  Instead of the one-piece stage-1 spool the positioner gets
    a "Faceplate spool frame" (+ seam curves, clamp studs): the spool arrives there as three separate parts;
  * handler robot ABB IRB 6700-150/3.20 (RoboDK library) on a 1-axis floor track along X ("Handler track X",
    BuildMechanism 1T, joint value = world x of the J1 axis), tong gripper tool (STL generated here, TCP Tz(420));
  * tack robot ABB IRB 1600-6/1.45 (or IRB 1520ID) on a pedestal with the stage-1 MIG torch;
  * static equipment (AddShape): kit pallet with nests, pipe buffer, assembly station table + fixture, tack pedestal,
    QC arch, stepped storage rack (pre-filled spools), logistics fence outline, AGV, floor extension;
  * output conveyor with the carrier pallet as a 1-axis mechanism along X ("Carrier pallet X", joint value = x of the
    flange centre on the pallet) + a "Carrier spool frame" riding on it;
  * spool parts "Part flange" / "Part elbow" / "Part pipe" (+ "Part flange 2" of the next kit) as separate objects
    whose frame is the spool frame of the assembled spool (layout2 convention);
  * macros (Python programs written to generated/): GripperClose / GripperOpen (attach / release parts, the released
    parts go to the holder under the TCP: faceplate, carrier or the Logistics frame), StudsUp / StudsDown,
    TackArcOn / TackArcOff, ArcOn / ArcOff (stage-1 names, bead traced on the elbow), ResetParts, QC_MarkScan,
    View_* (stage-1 shots) and View2_* (stage-2 shots from cameras2.py);
  * programs: HTrack_* (handler carriage), Conv_* (carrier), Pos_RotLoad / Pos_RotUnload (faceplate), Handler_Home,
    KitFlange, KitElbow, KitPipe, TackWeld, Tack_Home, PickTackedSpool, LoadPositioner, Handler_Park,
    UnloadPositioner, ToConveyor, Conveyor_QC, KitFlange2, StoreSpool, Cycle2_Home and the main program DemoCycle2:
    kitting -> tacking -> loading -> DemoCycle (stage-1 welding) -> unloading -> conveyor + QC -> storage.

Targets: every Cartesian target uses the frames of stage2_logistics/plan2.py (station_frame, carrier_frame,
grasp_frame, spool_on_positioner(0, POS_LOAD_ROT), layout2.storage_frame, kit frames) converted to mm - they are
re-computed here from layout2 with robomath (tests/t_robodk2.py asserts they equal plan2's numpy functions).  The plan
data that needs the full stage-2 environment (the handler / tack joints of the planned animation, the tack torch poses)
is read from the snapshot plan2_robodk.json (written by --export-plan).  Those joints belong to the SCALED IRB 4600
used for the Blender visual (x1.25 handler, x0.57 tack robot), so here they are only IK seeds: RoboDK solves the
library robot's IK near them, which keeps the arm configuration of the video (front/back, elbow up, wrist).
"""
import ast
import glob
import json
import math
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))           # stage2_logistics/robodk
STAGE2 = os.path.dirname(HERE)                              # stage2_logistics
REPO = os.path.dirname(STAGE2)
DEMO = os.path.join(REPO, "demo_video")
STAGE1_RDK = os.path.join(DEMO, "robodk")
for _p in (STAGE1_RDK, STAGE2):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from robodk import robolink as rl                           # noqa: E402
from robodk import robomath as rm                           # noqa: E402
from robodk.robomath import Mat, transl, rotx, roty, rotz, invH, eye   # noqa: E402

import build_station as B                                   # noqa: E402  stage-1 builder, used as a library


def _import_layout2():
    """layout2 imports `tools`, which pulls the bpy modules of stage 1.  Inside RoboDK's python there is no bpy, but
    layout2 itself only needs cell.layout (pure python): fall back to a stub `tools` that just puts demo_video/ on the
    path."""
    try:
        import layout2
        return layout2
    except Exception:
        for name in ("tools", "layout2"):
            sys.modules.pop(name, None)
        stub = types.ModuleType("tools")
        stub.DEMO = DEMO
        if DEMO not in sys.path:
            sys.path.insert(0, DEMO)
        sys.modules["tools"] = stub
        import layout2
        return layout2


L2 = _import_layout2()
L1 = L2.L                                                    # stage-1 cell/layout.py (metres)

# =====================================================================================================================
# 1. Settings and constants (mm / deg)
# =====================================================================================================================
M = 1000.0
D2R = math.pi / 180.0
STATION_NAME = "PipeSpool_FullCycle"
SAVE_RDK = True                   # save <STATION_NAME>.rdk next to this script when done
REPLACE_OPEN_STATION = True       # close an already open station with the same name first (re-running = rebuild)
REFRESH_PLAN = False              # True: re-export plan2_robodk.json from plan2 before building (stage-2 python only)
ADD_CURVE_FOLLOW = True           # stage-1 "curve follow" machining projects for the seams
USE_VIEWS = True                  # View2_* macros (stage-2 camera shots) called from the programs
GEN_DIR = os.path.join(HERE, "generated")
PLAN_JSON = os.path.join(HERE, "plan2_robodk.json")
CAMERAS2_PY = os.path.join(STAGE2, "cameras2.py")

# item names
LOGISTICS_FRAME = "Logistics"
FACEPLATE_FRAME = "Faceplate spool frame"
CARRIER_FRAME = "Carrier spool frame"
HANDLER_NAME = "ABB IRB 6700-150/3.20"
TACK_NAME = "ABB IRB 1600-6/1.45"
HTRACK_NAME = "Handler track X"
CARRIER_NAME = "Carrier pallet X"
GRIPPER_NAME = "Tong gripper"
TACK_TORCH_NAME = "Tack torch"
STUDS_NAME = "Clamp studs"
PART_NAMES = dict(flange="Part flange", elbow="Part elbow", pipe="Part pipe", flange2="Part flange 2")
HANDLER_LIBRARY_GLOBS = ["*IRB*6700*150*3.20*.robot", "*IRB*6700*150*3*20*.robot", "*IRB*6700*150*.robot",
                         "*IRB*6700*.robot"]
TACK_LIBRARY_GLOBS = ["*IRB*1600*6*1.45*.robot", "*IRB*1600*1.45*.robot", "*IRB*1600*1*45*.robot",
                      "*IRB*1520*ID*.robot", "*IRB*1520*.robot", "*IRB*1600*.robot"]

# nominal data-sheet values used by the reach checks: J2 (shoulder) height above the base plate, flange -> wrist centre
# distance, reach = max distance of the wrist centre from the J1 axis at shoulder height
ROBOT_SPECS = {
    "handler": dict(model=HANDLER_NAME, shoulder_z=780.0, d6=200.0, reach=3200.0),
    "tack": dict(model=TACK_NAME, shoulder_z=486.5, d6=65.0, reach=1450.0),
}

# handler (layout2, metres -> mm)
HANDLER_TRACK_X = (L2.HANDLER_TRACK_X[0] * M, L2.HANDLER_TRACK_X[1] * M)     # carriage travel = J1 axis x
HANDLER_TRACK_Y = L2.HANDLER_TRACK_Y * M
HANDLER_TOP_Z = L2.HANDLER_TRACK_TOP_Z * M
HANDLER_YAW_DEG = math.degrees(L2.HANDLER_YAW)
HANDLER_HOME_X = L2.HANDLER_HOME_X * M
HANDLER_Q_HOME = [math.degrees(q) for q in L2.HANDLER_Q_HOME]
X_AT_POSITIONER = L2.HANDLER_X_AT_POSITIONER * M
GRIP_TCP_Z = L2.GRIP_TCP_Z * M
# tack robot
TACK_BASE = (L2.TACK_BASE[0] * M, L2.TACK_BASE[1] * M, L2.TACK_BASE_Z * M)
TACK_YAW_DEG = math.degrees(L2.TACK_YAW)
TACK_Q_HOME = [math.degrees(q) for q in L2.TACK_Q_HOME]
TACK_BACK = 70.0                  # approach / retract along the torch axis (plan2.solve_tacks)
TACK_ARC_MS = 400.0               # a tack is a short static arc
# choreography literals of plan2._script (mirrored; tests/t_robodk2.py checks them against the plan snapshot)
TRAVEL_Z = 1550.0                 # plan2.TRAVEL_Z: TCP height for transfers along the track
SWING_X = -4800.0                 # plan2.SWING_X: carriage x of the pre-entry pose (outside the cell fence)
PRE_ENTRY_TCP_X = SWING_X + 1400.0    # TCP x of the pre-entry pose (fence line at -2400)
LIFT_HIGH_Z, LIFT_HIGH_DY = 2300.0, -550.0     # the tacked spool is lifted to z = 2.30 m, 0.55 m toward the track
WAIT_DX, WAIT_Y = -900.0, 1100.0  # handler waiting pose next to the station during tacking (carriage x_st - 0.9)
IDLE_X, IDLE_TCP = -4400.0, (-3200.0, 0.0, 2000.0)   # parked pose while welding
LIFTS = dict(flange=250.0, elbow=300.0, pipe=250.0, spool=350.0, carrier=250.0, conv_end=300.0, store=300.0)

# speeds
HANDLER_SPEEDS = dict(air=(1000.0, 80.0, 50.0), fine=(250.0, 30.0, 0.0))    # (mm/s, deg/s, rounding mm)
TACK_SPEEDS = dict(air=(600.0, 90.0, 20.0), fine=(100.0, 30.0, 0.0))
HANDLER_TRACK_MMS = 1000.0
CONV_SPEED_MMS = L2.CONV_SPEED * M

# colours
STEEL = [0.70, 0.70, 0.72, 1.0]
DARK = [0.17, 0.18, 0.21, 1.0]
BLUE = [0.12, 0.31, 0.55, 1.0]
BLUE2 = [0.15, 0.33, 0.56, 1.0]
YELLOW = [0.95, 0.75, 0.05, 1.0]
WOOD = [0.60, 0.45, 0.25, 1.0]
GREY = [0.60, 0.62, 0.64, 1.0]
WHITE = [0.90, 0.90, 0.92, 1.0]
POLY = [0.85, 0.82, 0.70, 1.0]
PANEL = [0.45, 0.50, 0.55, 0.35]
PIPE_C = [0.45, 0.42, 0.40, 1.0]
FLANGE_C = [0.55, 0.55, 0.57, 1.0]
SOURCE_C = [0.75, 0.12, 0.10, 1.0]
CONCRETE = [0.52, 0.50, 0.48, 1.0]
LAMP = dict(red=[0.9, 0.1, 0.1, 1.0], amber=[1.0, 0.6, 0.05, 1.0], green=[0.1, 0.8, 0.2, 1.0])

# spool geometry (mm, spool-local: flange back face z = 0, axis +Z, elbow toward +X, pipe along +X)
R_O = L1.PIPE_OD / 2 * M
R_I = R_O - L1.PIPE_WALL * M
ELBOW_R = L1.ELBOW_R * M
SEAM_A_Z = L2.SEAM_A_Z * M
SEAM_B_X = L2.SEAM_B_X * M
PIPE_AXIS_Z = L2.PIPE_AXIS_Z * M
PIPE_LEN = L1.PIPE_LEN * M
FLANGE_OD = L1.FLANGE_OD * M
FLANGE_THK = L1.FLANGE_THK * M

# handler events of plan2 whose joints / TCP poses are stored in the snapshot
HANDLER_EVENTS = ("start", "kit_flange_grip", "stn_flange_contact", "kit_elbow_grip", "stn_elbow_contact",
                  "buf_pipe_grip", "stn_pipe_contact", "tack_start", "stn_spool_grip", "pre_entry", "entry_done",
                  "load_contact", "handler_out", "handler_idle", "unload_grip", "handler_out2", "conv_load_contact",
                  "kit_flange2_grip", "stn_flange2_contact", "conv_end_grip", "store_contact", "handler_home_end")


# =====================================================================================================================
# 2. Pure maths: unit conversions, frames of plan2 / layout2 (no RoboDK connection needed)
# =====================================================================================================================
def mat_mm(T_m):
    """4x4 pose in metres (nested lists or numpy) -> robomath Mat in mm."""
    rows = [[float(T_m[i][j]) for j in range(4)] for i in range(4)]
    for i in range(3):
        rows[i][3] *= M
    rows[3] = [0.0, 0.0, 0.0, 1.0]
    return Mat(rows)


def mat_m(T):
    """robomath Mat in mm -> 4x4 nested list in metres (inverse of mat_mm)."""
    rows = [[float(T[i, j]) for j in range(4)] for i in range(4)]
    for i in range(3):
        rows[i][3] /= M
    return rows


def deg(q_rad):
    return [math.degrees(float(v)) for v in q_rad]


def rad(q_deg):
    return [math.radians(float(v)) for v in q_deg]


def frame_xz(x_axis, z_axis, origin_mm):
    """Right-handed frame from an x direction and a z direction (x orthogonalised against z) = kin.frame (mm)."""
    z = rm.normalize3([float(v) for v in z_axis])
    x = [float(v) for v in x_axis]
    x = rm.normalize3(rm.subs3(x, rm.mult3(z, rm.dot(x, z))))
    return B.pose_from_axes(x, rm.cross(z, x), z, [float(v) for v in origin_mm])


def planar_frame(origin_m, x_dir_xy):
    """Spool 'standing' frame (Z up, X along x_dir_xy) = kin.planar_frame, origin in metres -> Mat in mm."""
    return frame_xz([x_dir_xy[0], x_dir_xy[1], 0.0], [0.0, 0.0, 1.0], [v * M for v in origin_m])


def grasp_frame(name):
    """layout2.GRASP[name] as a pose in the part / spool frame (mm) = plan2.grasp_frame."""
    o, x, z, _ = L2.GRASP[name]
    return frame_xz(x, z, [v * M for v in o])


def station_frame():
    return planar_frame(L2.STATION_ORIGIN, L2.STATION_XDIR)


def carrier_frame(x_m):
    """Spool frame on a carrier pallet whose flange centre is at world x = x_m (metres) = plan2.carrier_frame."""
    return planar_frame((x_m, L2.CONV_SPOOL_Y, L2.CARRIER["seat_z"]), (0.0, 1.0))


def positioner_frame(rot_deg, tilt_deg=0.0):
    """Spool frame on the positioner = plan2.spool_on_positioner (= stage-1 animation.spool_world)."""
    return B.spool_pose(tilt_deg, rot_deg)


def kit_flange_frame(i):
    return planar_frame(*L2.KIT_FLANGES[i])


def kit_elbow_frame(i):
    return planar_frame(*L2.KIT_ELBOWS[i])


def buffer_pipe_frame(i):
    return planar_frame(*L2.pipe_buffer_frame(i))


def storage_frame(tier, bay):
    return planar_frame(*L2.storage_frame(tier, bay))


def handler_base_world(x_mm):
    """Base of the handler robot with the carriage (J1 axis) at world x = x_mm."""
    return transl(x_mm, HANDLER_TRACK_Y, HANDLER_TOP_Z) * rotz(HANDLER_YAW_DEG * D2R)


def handler_track_base():
    """Base of the handler track mechanism: its Z = world +X, so the joint value is the world x of the carriage."""
    return transl(0.0, HANDLER_TRACK_Y, 0.0) * roty(90 * D2R)


def carrier_track_base():
    """Base of the carrier mechanism: its Z = world +X, joint value = world x of the flange centre on the pallet."""
    return roty(90 * D2R)


def tack_base_world():
    return transl(*TACK_BASE) * rotz(TACK_YAW_DEG * D2R)


def gripper_tcp():
    return transl(0.0, 0.0, GRIP_TCP_Z)


def above(T, dz):
    """Same orientation, raised by dz (world Z)."""
    return transl(0.0, 0.0, dz) * T


def oriented_like(G, ref):
    """G or G rotated 180 deg about its Z - the tongs are symmetric - whichever is closer to the orientation of ref
    (the planned TCP pose); plan2.h_ik makes the same choice."""
    if ref is None:
        return G
    G2 = G * rotz(math.pi)
    return G2 if rm.pose_angle_between(G2, ref) < rm.pose_angle_between(G, ref) else G


def wrist_centre(T_tcp, tool, d6):
    """Wrist centre (J5 axis point) of a robot whose tool TCP is at T_tcp: flange = T_tcp * tool^-1, minus d6 along
    the flange Z."""
    return (T_tcp * invH(tool) * transl(0.0, 0.0, -d6)).Pos()


def reach_of(who, pose, x_mm=None):
    """(distance of the wrist centre from the J1 axis point at shoulder height, nominal reach) for a TCP pose."""
    spec = ROBOT_SPECS[who]
    if who == "handler":
        base, tool = handler_base_world(x_mm), gripper_tcp()
    else:
        base, tool = tack_base_world(), B.tcp_pose()
    shoulder = B.xform_point(base, [0.0, 0.0, spec["shoulder_z"]])
    return rm.norm(rm.subs3(wrist_centre(pose, tool, spec["d6"]), shoulder)), spec["reach"]


# =====================================================================================================================
# 3. Plan snapshot (plan2 -> plan2_robodk.json)
# =====================================================================================================================
def export_plan(path=PLAN_JSON, use_cache=True):
    """Write the plan data RoboDK needs (joints + TCP poses of the handler at the choreography events, the 6 tack
    torch poses / joints) using plan2 itself.  Needs the stage-2 python environment (numpy + bpy)."""
    import numpy as np  # noqa: F401
    import plan2
    P = plan2.solve(use_cache=use_cache, verbose=False)
    H = plan2.K.scaled_arm(L2.HANDLER_SCALE)
    TK = plan2.K.scaled_arm(L2.TACK_SCALE)
    n = int(P["n_frames"])
    events = {k: int(v) for k, v in P["events"].items()}
    handler = {}
    for ev in HANDLER_EVENTS:
        if ev not in events:
            continue
        i = min(max(events[ev], 1), n) - 1
        x = float(P["handler_x"][i])
        q = [float(v) for v in P["handler_q"][i]]
        T = H.fk(np.array(q), plan2.handler_base(x), plan2.GRIP_T)
        handler[ev] = dict(frame=events[ev], x=x, q=q, tcp=[[float(v) for v in row] for row in T])
    tacks = []
    for (a0, _a1), tp in zip(P["intervals"]["tack_arc"], P["tack_points"]):
        q = [float(v) for v in P["tack_q"][a0 - 1]]
        T = TK.fk(np.array(q), plan2.tack_base(), plan2.TORCH_T)
        tacks.append(dict(key=tp["key"], angle=float(tp["angle"]), frame=int(a0), q=q,
                          tcp=[[float(v) for v in row] for row in T]))
    data = dict(
        source="stage2_logistics/plan2.py: plan2.solve() + FK of the scaled IRB 4600 arms (kin.scaled_arm); "
               "written by build_station2.py --export-plan",
        plan_hash=plan2._hash(), n_frames=n, weld_offset=int(P["weld_offset"]), events=events,
        units=dict(length="m", angle="rad"), handler_scale=L2.HANDLER_SCALE, tack_scale=L2.TACK_SCALE,
        travel_z=float(plan2.TRAVEL_Z), swing_x=float(plan2.SWING_X),
        handler=handler, tack=tacks, handler_q_home=list(L2.HANDLER_Q_HOME), tack_q_home=list(L2.TACK_Q_HOME),
        ik_fail=[list(map(str, f)) for f in P.get("ik_fail", [])])
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    return path


def load_plan(path=PLAN_JSON, refresh=None):
    """The plan snapshot (dict); {} if it is missing (nominal targets, home joints as seeds)."""
    refresh = REFRESH_PLAN if refresh is None else refresh
    if refresh:
        try:
            export_plan(path)
        except Exception as exc:
            print("plan2 refresh failed (%s) - using the snapshot %s" % (exc, path))
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    print("WARNING: %s is missing - nominal targets with the home joints as IK seeds "
          "(run: python3 build_station2.py --export-plan)" % path)
    return {}


# =====================================================================================================================
# 4. Choreography as data: waypoints + program steps (pure; realised in RoboDK by realize_program)
# =====================================================================================================================
# steps: ("comment", text) | ("speed", "air"|"fine") | ("track", x_mm, label) -> call HTrack_<label> |
#        ("movej"|"movel", waypoint) | ("home",) joint move to the home joints | ("call", program or macro) |
#        ("view", shot) -> call View2_<shot> if it exists | ("do", io, value) | ("pause", ms)
def handler_plan(plan):
    """Handler waypoints and programs in DemoCycle2 order.
    Returns (waypoints, programs): waypoints[name] = dict(pose=Mat (gripper TCP, world mm), x=carriage x (mm),
    seed=IK seed (deg, from the plan joints at `event`), event=plan2 event, exact=True if the pose is the planned TCP
    pose at that event); programs = [(name, steps), ...]."""
    evs = (plan or {}).get("handler", {})
    wps = {}

    def seed_of(event):
        e = evs.get(event)
        return deg(e["q"]) if e else list(HANDLER_Q_HOME)

    def ref_of(event):
        e = evs.get(event)
        return mat_mm(e["tcp"]) if e else None

    def wp(name, pose, x, event, exact=False):
        wps[name] = dict(pose=pose, x=float(x), seed=seed_of(event), event=event, exact=exact)
        return name

    def pick_place(prefix, T_part, grasp, x, lift, event, label, grip_close, extra=()):
        """Pick (grip_close=True) or place template: (carriage to x) -> above the grasp at travel height ->
        approach lift mm above -> linear down -> gripper macro -> linear up -> back to travel height."""
        G = oriented_like(T_part * grasp_frame(grasp), ref_of(event))
        A = above(G, lift)
        steps = [("speed", "air"), ("track", x, label)]
        travel = A.Pos()[2] < TRAVEL_Z - 50.0
        if travel:
            steps += [("movej", wp(prefix + "_Travel", above(A, TRAVEL_Z - A.Pos()[2]), x, event)),
                      ("movel", wp(prefix + "_Approach", A, x, event))]
        else:
            steps += [("movej", wp(prefix + "_Approach", A, x, event))]
        steps += [("speed", "fine"), ("movel", wp(prefix, G, x, event, exact=True)),
                  ("do", "GripperClosed", 1 if grip_close else 0),
                  ("call", "GripperClose" if grip_close else "GripperOpen")]
        steps += list(extra)
        steps += [("movel", prefix + "_Approach"), ("speed", "air")]
        if travel:
            steps += [("movel", prefix + "_Travel")]
        return steps

    Fs = station_frame()
    x_st = L2.STATION_ORIGIN[0] * M
    kit_x = [f[0][0] * M for f in L2.KIT_FLANGES]
    kit_ex = [f[0][0] * M for f in L2.KIT_ELBOWS]
    progs = []

    progs.append(("Handler_Home", [("comment", "Handler: arm folded (home joints), carriage to the home position"),
                                   ("speed", "air"), ("home",), ("track", HANDLER_HOME_X, "Home")]))
    # ---- kitting: flange, elbow, pipe -> assembly station (plan2 events kit_flange_grip .. stn_pipe_contact)
    progs.append(("KitFlange", [("comment", "Kitting 1/3: flange from the kit pallet onto the assembly station")]
                  + pick_place("KitFlange_Pick", kit_flange_frame(0), "flange", kit_x[0], LIFTS["flange"],
                               "kit_flange_grip", "KitA", True)
                  + [("view", "S2_02_flange")]
                  + pick_place("KitFlange_Place", Fs, "flange", x_st, LIFTS["flange"], "stn_flange_contact",
                               "Station", False)
                  + [("do", "StationClampFlange", 1)]))
    progs.append(("KitElbow", [("comment", "Kitting 2/3: elbow from the kit pallet onto the flange")]
                  + pick_place("KitElbow_Pick", kit_elbow_frame(0), "elbow", kit_ex[0], LIFTS["elbow"],
                               "kit_elbow_grip", "KitA", True)
                  + [("view", "S2_03_elbow")]
                  + pick_place("KitElbow_Place", Fs, "elbow", x_st, LIFTS["elbow"], "stn_elbow_contact",
                               "Station", False)
                  + [("do", "StationClampElbow", 1)]))
    T_wait = oriented_like(transl(x_st + WAIT_DX, WAIT_Y, TRAVEL_Z) * rotx(math.pi), ref_of("tack_start"))
    progs.append(("KitPipe", [("comment", "Kitting 3/3: pipe from the buffer against the end stop (root gap)")]
                  + pick_place("KitPipe_Pick", buffer_pipe_frame(0), "pipe", L2.PIPE_BUFFER_X[0] * M, LIFTS["pipe"],
                               "buf_pipe_grip", "Buffer", True)
                  + [("view", "S2_04_pipe")]
                  + pick_place("KitPipe_Place", Fs, "pipe", x_st, LIFTS["pipe"], "stn_pipe_contact", "Station", False)
                  + [("do", "StationClampPipe", 1), ("do", "GapOK", 1),
                     ("comment", "Handler waits next to the station (clear of the tack robot)"),
                     ("track", x_st + WAIT_DX, "Wait"),
                     ("movej", wp("Wait_Station", T_wait, x_st + WAIT_DX, "tack_start", exact=True))]))
    # ---- tacked spool: pick at the station, lift high toward the track
    G_st = oriented_like(Fs * grasp_frame("spool"), ref_of("stn_spool_grip"))
    T_high = transl(0.0, LIFT_HIGH_DY, LIFT_HIGH_Z - G_st.Pos()[2]) * G_st
    progs.append(("PickTackedSpool", [("comment", "Unclamp, pick the tacked spool on its pipe leg, lift it high"),
                                      ("do", "StationClamps", 0), ("do", "GapOK", 0), ("view", "S2_07_pick")]
                  + pick_place("Spool_Pick", Fs, "spool", x_st, LIFTS["spool"], "stn_spool_grip", "Station", True)
                  + [("movej", wp("Spool_LiftHigh", T_high, x_st, "stn_spool_grip"))]))
    # ---- loading: pre-entry outside the fence, the carriage runs in (the TCP moves straight along +X), linear entry,
    # lower onto the faceplate (pipe leg toward -X at POS_LOAD_ROT), studs up, open, out the same way
    G_pos = oriented_like(positioner_frame(L2.POS_LOAD_ROT) * grasp_frame("spool"), ref_of("load_contact"))
    G_app = above(G_pos, L2.POS_APPROACH_DZ * M)
    G_pre = transl(PRE_ENTRY_TCP_X - G_app.Pos()[0], 0.0, 0.0) * G_app
    G_entry = transl(X_AT_POSITIONER - SWING_X, 0.0, 0.0) * G_pre        # same joints as G_pre, carriage moved in
    wp("Load_PreEntry", G_pre, SWING_X, "pre_entry", exact=True)
    wp("Load_Entry", G_entry, X_AT_POSITIONER, "pre_entry")
    wp("Load_Approach", G_app, X_AT_POSITIONER, "entry_done", exact=True)
    wp("Load_Place", G_pos, X_AT_POSITIONER, "load_contact", exact=True)
    progs.append(("LoadPositioner", [
        ("comment", "Load the tacked spool onto the positioner (tilt 0, faceplate %.0f deg): pre-entry at carriage "
                    "x=%.0f outside the fence, muting, carriage in, linear entry, lower, studs up, open, exit"
         % (L2.POS_LOAD_ROT, SWING_X)),
        ("speed", "air"), ("track", SWING_X, "Swing"), ("movej", "Load_PreEntry"), ("view", "S2_08_load"),
        ("do", "LightCurtainMuting", 1), ("track", X_AT_POSITIONER, "Positioner"), ("movel", "Load_Entry"),
        ("movel", "Load_Approach"), ("speed", "fine"), ("movel", "Load_Place"), ("call", "StudsUp"),
        ("do", "GripperClosed", 0), ("call", "GripperOpen"), ("movel", "Load_Approach"), ("speed", "air"),
        ("movel", "Load_Entry"), ("track", SWING_X, "Swing"), ("do", "LightCurtainMuting", 0)]))
    T_idle = oriented_like(transl(*IDLE_TCP) * rotx(math.pi) * rotz(math.pi), ref_of("handler_idle"))
    progs.append(("Handler_Park", [("comment", "Handler parked in front of the opening while the cell welds"),
                                   ("view", "S2_09_ready"), ("speed", "air"), ("track", IDLE_X, "Idle"),
                                   ("movej", wp("Handler_Idle", T_idle, IDLE_X, "handler_idle", exact=True))]))
    progs.append(("UnloadPositioner", [
        ("comment", "Unload the welded spool (faceplate turned to %.0f deg: pipe leg toward -X again)"
         % L2.POS_UNLOAD_ROT),
        ("speed", "air"), ("track", SWING_X, "Swing"), ("movej", "Load_PreEntry"), ("view", "S2_11_unload"),
        ("do", "LightCurtainMuting", 1), ("track", X_AT_POSITIONER, "Positioner"), ("movel", "Load_Entry"),
        ("movel", "Load_Approach"), ("speed", "fine"), ("movel", "Load_Place"), ("do", "GripperClosed", 1),
        ("call", "GripperClose"), ("call", "StudsDown"), ("movel", "Load_Approach"), ("speed", "air"),
        ("movel", "Load_Entry"), ("track", SWING_X, "Swing"), ("do", "LightCurtainMuting", 0)]))
    progs.append(("ToConveyor", [("comment", "Welded spool onto the carrier pallet at the conveyor load station"),
                                 ("view", "S2_12_carrier")]
                  + pick_place("Conv_Place", carrier_frame(L2.CONV_LOAD_X), "spool", L2.CONV_LOAD_X * M,
                               LIFTS["carrier"], "conv_load_contact", "ConvLoad", False)))
    progs.append(("KitFlange2", [("comment", "Next kit: flange 2 onto the assembly station (the cycle continues)")]
                  + pick_place("KitFlange2_Pick", kit_flange_frame(1), "flange", kit_x[1], LIFTS["flange"],
                               "kit_flange2_grip", "KitB", True)
                  + pick_place("KitFlange2_Place", Fs, "flange", x_st, LIFTS["flange"], "stn_flange2_contact",
                               "Station", False)
                  + [("do", "StationClampFlange", 1)]))
    tier, bay = L2.STORAGE_TARGET
    progs.append(("StoreSpool", [("comment", "Finished spool from the conveyor end into storage bay (%d, %d)"
                                  % (tier, bay)), ("view", "S2_16_store")]
                  + pick_place("ConvEnd_Pick", carrier_frame(L2.CONV_END_X), "spool", L2.CONV_END_X * M,
                               LIFTS["conv_end"], "conv_end_grip", "ConvEnd", True)
                  + pick_place("Store_Place", storage_frame(tier, bay), "spool", HANDLER_TRACK_X[0], LIFTS["store"],
                               "store_contact", "Storage", False)))
    return wps, progs


TACK_REACH_MARGIN = 0.97         # nominal tack poses: wrist centre within 97 % of the reach (tack and back-off pose)


def tack_pose(joint, ang_deg, up):
    """Torch pose of a tack (plan2.solve_tacks geometry): point on seam A / B at ang_deg, torch along the surface
    normal tilted by `up` toward the open side (A: up, B: along the pipe), no push angle, body leaning to the robot."""
    Fs = station_frame()
    a = ang_deg * D2R
    if joint == "A":
        n_l, p_l = [math.cos(a), math.sin(a), 0.0], [R_O * math.cos(a), R_O * math.sin(a), SEAM_A_Z]
        t_l, bias = [-math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]
    else:
        n_l = [0.0, math.cos(a), math.sin(a)]
        p_l = [SEAM_B_X, R_O * math.cos(a), PIPE_AXIS_Z + R_O * math.sin(a)]
        t_l, bias = [0.0, -math.sin(a), math.cos(a)], [1.0, 0.0, 0.0]
    p = B.xform_point(Fs, p_l)
    n = rm.normalize3(B.xform_vec(Fs, rm.add3(n_l, rm.mult3(bias, up))))
    lean = rm.add3(rm.normalize3([TACK_BASE[0] - p[0], TACK_BASE[1] - p[1], 0.0]), [0.0, 0.0, 0.9])
    return B.torch_target(p, n, B.xform_vec(Fs, t_l), lean, 0.0)


def tack_nominal():
    """Tack torch poses without the plan: for each layout2 tack the first variant in plan2's search order (angle
    offset 0, +10, -10, +20, -20 deg; torch tilts A: 1.0 / 0.5 / 2.0, B: 0.35 / 0.7 / 0.0) whose tack and back-off
    poses keep the wrist centre inside TACK_REACH_MARGIN of the IRB 1600 reach (the best one if none does)."""
    out = []
    specs = [("A", i, a) for i, a in enumerate(L2.TACKS_A)] + [("B", i, b) for i, b in enumerate(L2.TACKS_B)]
    limit = TACK_REACH_MARGIN * ROBOT_SPECS["tack"]["reach"]
    for joint, i, nominal in specs:
        best = None
        for dang in (0.0, 10.0, -10.0, 20.0, -20.0):
            for up in ((1.0, 0.5, 2.0) if joint == "A" else (0.35, 0.7, 0.0)):
                T = tack_pose(joint, nominal + dang, up)
                back = transl(*rm.mult3(T.VZ(), -TACK_BACK)) * T
                d = max(reach_of("tack", T)[0], reach_of("tack", back)[0])
                if best is None or d < best[0]:
                    best = (d, T, nominal + dang)
                if d <= limit:
                    break
            if best[0] <= limit:
                break
        out.append(dict(key="%s%d" % (joint, i), angle=best[2], pose=best[1], seed=list(TACK_Q_HOME)))
    return out


def tack_plan(plan):
    """Tack robot waypoints and programs: 6 tacks (MoveJ 70 mm back along the torch axis, MoveL in, short arc, MoveL
    back).  Torch poses / joint seeds from the snapshot (FK of the planned tack joints), else tack_nominal()."""
    planned = {t["key"]: dict(key=t["key"], angle=t["angle"], pose=mat_mm(t["tcp"]), seed=deg(t["q"]))
               for t in (plan or {}).get("tack", [])}
    # every tack of layout2 is welded; a tack the planner could not reach with the scaled arm keeps its nominal pose
    tacks = [planned.get(t["key"], t) for t in tack_nominal()]
    wps = {}
    steps = [("comment", "Tack welding: %d tacks, 3 per joint on the tack-robot side (seam A: flange-elbow, seam B: "
                         "elbow-pipe)" % len(tacks)), ("view", "S2_05_tackA"), ("speed", "air"), ("home",)]
    for t in tacks:
        T = t["pose"]
        back = transl(*rm.mult3(T.VZ(), -TACK_BACK)) * T
        nb, nt = "Tack_%s_back" % t["key"], "Tack_%s" % t["key"]
        src = "plan2" if t["key"] in planned else "nominal pose, not in the plan snapshot"
        wps[nb] = dict(pose=back, x=None, seed=t["seed"], event="tack %s" % t["key"], exact=False)
        wps[nt] = dict(pose=T, x=None, seed=t["seed"], event="tack %s" % t["key"], exact=t["key"] in planned)
        steps += [("comment", "Tack %s (%.0f deg, %s)" % (t["key"], t["angle"], src)), ("speed", "air"), ("movej", nb),
                  ("speed", "fine"), ("movel", nt), ("do", "TackArc", 1), ("call", "TackArcOn"),
                  ("pause", TACK_ARC_MS), ("call", "TackArcOff"), ("do", "TackArc", 0), ("movel", nb)]
    steps += [("speed", "air"), ("home",)]
    return wps, [("TackWeld", steps), ("Tack_Home", [("speed", "air"), ("home",)])]


DEMO_CYCLE2 = [   # (kind, name): the main program; "view" entries are skipped when the shot macro does not exist
    ("view", "S2_01_wide"), ("call", "Cycle2_Home"), ("pause", 1000),
    ("call", "KitFlange"), ("call", "KitElbow"), ("call", "KitPipe"), ("call", "TackWeld"),
    ("call", "PickTackedSpool"), ("call", "LoadPositioner"), ("call", "Handler_Park"),
    ("call", "DemoCycle"),                                  # stage-1 welding choreography (existing program)
    ("view", "S2_10_done"), ("call", "Pos_RotUnload"), ("call", "UnloadPositioner"), ("call", "ToConveyor"),
    ("call", "Conveyor_QC"), ("call", "KitFlange2"), ("call", "StoreSpool"), ("view", "S2_17_final"),
    ("call", "Handler_Home"), ("pause", 2000)]

CYCLE2_HOME = ["ResetParts", "Handler_Home", "Tack_Home", "Conv_ToLoad", "Robot_Home", "Track_Home", "Pos_TiltDown",
               "Pos_RotLoad", "StudsDown"]


# =====================================================================================================================
# 5. Meshes (mm): parts, gripper, equipment
# =====================================================================================================================
def revolve_tris(profile, n=36, T=None):
    """Closed surface of revolution about the local Z axis of a closed (r, z) polygon."""
    rings = [[[r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n), z] for k in range(n)]
             for r, z in profile]
    tris = []
    m = len(profile)
    for i in range(m):
        A, Bq = rings[i], rings[(i + 1) % m]
        for k in range(n):
            k1 = (k + 1) % n
            tris.append([A[k], Bq[k], Bq[k1]])
            tris.append([A[k], Bq[k1], A[k1]])
    return B.tris_transform(tris, T) if T is not None else tris


def bar_tris(p0, p1, w, t):
    """Box from p0 to p1 (long axis), width w along the axis closest to world X, thickness t."""
    d = rm.subs3(list(p1), list(p0))
    z = rm.normalize3(d)
    x = [1.0, 0.0, 0.0] if abs(z[0]) < 0.9 else [0.0, 1.0, 0.0]
    x = rm.normalize3(rm.subs3(x, rm.mult3(z, rm.dot(x, z))))
    rot = B.pose_from_axes(x, rm.cross(z, x), z, [0.0, 0.0, 0.0])
    return B.box_tris(rm.mult3(rm.add3(list(p0), list(p1)), 0.5), (w, t, rm.norm(d)), rot)


def ring_tris(r0, r1, z0, z1, n=32):
    """Annular ring (seat ring, cup) about local Z."""
    return revolve_tris([(r0, z0), (r1, z0), (r1, z1), (r0, z1)], n)


def flange_tris(n=36):
    """Weld-neck flange (stage-1 spool profile, simplified): back face z = 0, hub up to seam A."""
    hub = R_O + 28.0
    prof = [(R_I, 0.0), (FLANGE_OD / 2, 0.0), (FLANGE_OD / 2, FLANGE_THK), (hub + 12.0, FLANGE_THK),
            (hub, FLANGE_THK + 12.0), (R_O, SEAM_A_Z - 35.0), (R_O, SEAM_A_Z), (R_I, SEAM_A_Z)]
    return revolve_tris(prof, n)


def elbow_tris(n_bend=16, n_tube=24):
    """90 deg long-radius elbow: from (0, 0, SEAM_A_Z) heading +Z, bending toward +X, to (SEAM_B_X, 0, PIPE_AXIS_Z)."""
    def ring(th, r):
        c = [ELBOW_R - ELBOW_R * math.cos(th), 0.0, SEAM_A_Z + ELBOW_R * math.sin(th)]
        u = [math.cos(th), 0.0, -math.sin(th)]
        return [rm.add3(c, [u[0] * r * math.cos(a), r * math.sin(a), u[2] * r * math.cos(a)])
                for a in (2 * math.pi * k / n_tube for k in range(n_tube))]
    ths = [0.5 * math.pi * i / n_bend for i in range(n_bend + 1)]
    outer = [ring(th, R_O) for th in ths]
    inner = [ring(th, R_I) for th in ths]
    tris = []
    for rings in (outer, inner):
        for i in range(n_bend):
            A, Bq = rings[i], rings[i + 1]
            for k in range(n_tube):
                k1 = (k + 1) % n_tube
                tris += [[A[k], Bq[k], Bq[k1]], [A[k], Bq[k1], A[k1]]]
    for i in (0, n_bend):
        for k in range(n_tube):
            k1 = (k + 1) % n_tube
            tris += [[outer[i][k], inner[i][k], inner[i][k1]], [outer[i][k], inner[i][k1], outer[i][k1]]]
    return tris


def pipe_tris(n=24):
    """Straight pipe along +X from seam B to the pipe end (axis at z = PIPE_AXIS_Z)."""
    return revolve_tris([(R_I, 0.0), (R_O, 0.0), (R_O, PIPE_LEN), (R_I, PIPE_LEN)], n,
                        transl(SEAM_B_X, 0.0, PIPE_AXIS_Z) * roty(90 * D2R))


def spool_part_tris():
    return dict(flange=flange_tris(), elbow=elbow_tris(), pipe=pipe_tris())


def gripper_mesh_triangles():
    """Tong gripper in the flange (tool0) frame, jaws closed on a DN250 pipe whose axis runs along tool X through the
    TCP (0, 0, GRIP_TCP_Z); simplified from stage2_logistics/gripper.py (adapter, rotary module, housing, hinged
    jaws with two V-prism pads touching the pipe 38 deg above / 10 deg below its equator, pneumatic cylinders)."""
    zc, R = GRIP_TCP_Z, R_O
    tris = []
    tris += B.cylinder_tris((0, 0, 0), (0, 0, 18), 100, 32)             # adapter plate
    tris += B.cylinder_tris((0, 0, 18), (0, 0, 78), 85, 32)             # rotary module
    tris += B.box_tris((0, 0, 114), (150, 410, 72))                     # housing (z 78..150)
    tris += B.box_tris((-100, 0, 112), (50, 140, 60))                   # valve block
    for s in (-1.0, 1.0):
        tris += B.cylinder_tris((-70, s * 250, 130), (70, s * 250, 130), 30, 16)   # hinge boss
        tris += bar_tris((0, s * 250, 130), (0, s * 205, 450), 110, 44)          # jaw arm (cheek plates)
        tris += B.box_tris((0, s * 182.5, 390), (90, 65, 140))                   # V-prism web (|y| 150..215)
        for phi, zs in ((38.0, -1.0), (10.0, 1.0)):                              # pads tangent to the pipe
            ny, nz = s * math.cos(phi * D2R), zs * math.sin(phi * D2R)
            c = (0.0, R * ny + 15 * ny, zc + R * nz + 15 * nz)
            z = [0.0, ny, nz]
            rot = B.pose_from_axes([1.0, 0.0, 0.0], rm.cross(z, [1.0, 0.0, 0.0]), z, [0.0, 0.0, 0.0])
            tris += B.box_tris(c, (90, 60, 30), rot)
        tris += B.cylinder_tris((0, s * 70, 168), (0, s * 200, 250), 22, 12)     # pneumatic cylinder
    return tris


def in_frame(F, shapes):
    return [(B.tris_transform(t, F), c) for t, c in shapes]


def seat_ring_local(z0=-30.0, pins=True):
    """Machined seat ring under the flange back face (local z0..0) + 3 centring pins outside the flange rim."""
    shapes = [(ring_tris(193.0, 222.0, z0, 0.0), STEEL)]
    if pins:
        for a in (90.0, 210.0, 330.0):
            x, y = 212.5 * math.cos(a * D2R), 212.5 * math.sin(a * D2R)
            shapes.append((B.cylinder_tris((x, y, z0), (x, y, 19.0), 8.0, 8), DARK))
    return shapes


def kit_pallet_shapes():
    cx, cy = [v * M for v in L2.KIT_PALLET["center"]]
    sx, sy = [v * M for v in L2.KIT_PALLET["size"]]
    top, cz = L2.KIT_PALLET["top_z"] * M, L2.KIT_CASSETTE_Z * M
    shapes = [(B.box_tris((cx, cy, top / 2), (sx, sy, top)), WOOD),
              (B.box_tris((cx, cy, (top + cz - 16) / 2), (sx - 40, sy - 40, cz - 16 - top)), DARK),
              (B.box_tris((cx, cy, cz - 8), (sx - 40, sy - 40, 16)), BLUE)]
    for ux in (-1, 1):
        for uy in (-1, 1):
            shapes.append((B.box_tris((cx + ux * (sx / 2 - 50), cy + uy * (sy / 2 - 50), cz + 180), (60, 60, 360)), DARK))
    for i in range(len(L2.KIT_FLANGES)):
        F = kit_flange_frame(i)
        shapes += in_frame(F, seat_ring_local(cz - F.Pos()[2]))
    for i in range(len(L2.KIT_ELBOWS)):
        F = kit_elbow_frame(i)            # elbow lower end (local z = SEAM_A_Z) in a cup, V-rest under the upper end
        z0 = cz - F.Pos()[2]
        cup = [(revolve_tris([(0.0, z0), (190.0, z0), (190.0, z0 + 10), (168.0, z0 + 10), (168.0, 140.0),
                              (150.0, 140.0), (150.0, SEAM_A_Z), (0.0, SEAM_A_Z)], 32), POLY),
               (B.box_tris((368.0, 0.0, (z0 + 330.0) / 2), (22.0, 200.0, 330.0 - z0)), BLUE2)]
        shapes += in_frame(F, cup)
    return shapes


def pipe_buffer_shapes():
    cx, cy = [v * M for v in L2.PIPE_BUFFER["center"]]
    sx = L2.PIPE_BUFFER["size"][0] * M
    x0, x1 = cx - sx / 2 + 30, cx + sx / 2 - 30
    pipe_bottom = L2.PIPE_BUFFER["axis_z"] * M - R_O
    shapes = []
    for s in (-1, 1):
        y = cy + s * 220
        shapes.append((B.box_tris(((x0 + x1) / 2, y, 640), (x1 - x0, 80, 100)), BLUE))           # cradle beam
        for xl in (x0 + 45, x1 - 45):
            shapes.append((B.box_tris((xl, y, 295), (80, 80, 590)), DARK))                       # legs
        for xp in L2.PIPE_BUFFER_X:                                                             # V-blocks
            shapes.append((B.box_tris((xp * M, y, (690 + pipe_bottom - 15) / 2), (260, 60, pipe_bottom - 15 - 690)), POLY))
    y_end = cy + PIPE_LEN / 2
    shapes.append((B.box_tris(((x0 + x1) / 2, y_end + 11.5, 855), (x1 - x0, 20, 270)), BLUE))   # end stop
    shapes.append((B.box_tris(((x0 + x1) / 2, y_end + 1.0, 855), (x1 - x0, 2, 240)), [0.1, 0.1, 0.1, 1.0]))
    for xl in (x0 + 45, x1 - 45):
        shapes.append((B.box_tris((xl, y_end + 50, 495), (60, 60, 990)), DARK))
    return shapes


def station_shapes():
    tcx, tcy = [v * M for v in L2.STATION_TABLE["center"]]
    tsx, tsy = [v * M for v in L2.STATION_TABLE["size"]]
    top = L2.STATION_TABLE["top_z"] * M
    shapes = [(B.box_tris((tcx, tcy, top - 50), (tsx, tsy, 60)), BLUE),                         # welded frame
              (B.box_tris((tcx, tcy, top - 10), (tsx - 20, tsy - 20, 20)), STEEL)]               # table plate
    for ux in (-1, 1):
        for uy in (-1, 1):
            shapes.append((B.box_tris((tcx + ux * (tsx / 2 - 60), tcy + uy * (tsy / 2 - 60), (top - 80) / 2),
                                      (80, 80, top - 80)), DARK))
    Fs = station_frame()
    z0 = top - Fs.Pos()[2]                                  # plate top in the station (spool) frame: -20
    local = seat_ring_local(z0)
    for lx, zt in ((300.0, 315.0), (860.0, PIPE_AXIS_Z - R_O - 15.0)):                         # V-blocks
        local.append((B.box_tris((lx, 0.0, (z0 + zt) / 2), (60.0, 200.0, zt - z0)), BLUE2))
        local.append((B.box_tris((lx, 0.0, zt + 5), (70.0, 220.0, 10.0)), POLY))
    xs = L2.STATION_SUPPORTS["pipe_stop"]["center"][0] * M
    local.append((B.box_tris((xs + 10, 0.0, (z0 + 600) / 2), (20.0, 220.0, 600 - z0)), YELLOW))  # pipe end stop
    for pivot, _close in L2.STATION_CLAMPS.values():                                            # pneumatic clamps
        px, py, pz = [v * M for v in pivot]
        local.append((B.box_tris((px, py, pz), (70.0, 70.0, 110.0)), DARK))
        if pz - 55 > z0 + 5:
            local.append((B.box_tris((px, py, (z0 + pz - 55) / 2), (50.0, 50.0, pz - 55 - z0)), STEEL))
    gx, gy, gz = [v * M for v in L2.GAP_SENSOR["local"]]                                         # gap sensor + lamp
    local.append((B.box_tris((gx, gy, (z0 + gz - 30) / 2), (40.0, 40.0, gz - 30 - z0)), DARK))
    local.append((B.box_tris((gx, gy, gz), (70.0, 90.0, 60.0)), DARK))
    local.append((B.cylinder_tris((gx, gy, gz + 30), (gx, gy, gz + 70), 20.0, 12), LAMP["green"]))
    return shapes + in_frame(Fs, local)


def tack_pedestal_shapes():
    x, y, z = TACK_BASE
    return [(B.box_tris((x, y, 10), (700, 700, 20)), DARK),
            (B.box_tris((x, y, (20 + z - 30) / 2), (500, 500, z - 50)), BLUE),
            (B.box_tris((x, y, z - 15), (560, 560, 30)), STEEL),
            (B.box_tris((-2700.0, 3050.0, 450.0), (450.0, 600.0, 900.0)), SOURCE_C),              # tack power source
            (B.cylinder_tris((-2700.0, 3050.0, 900.0), (-2700.0, 3050.0, 1100.0), 180.0, 16), DARK)]  # wire feeder


def conveyor_shapes():
    C = L2.CONVEYOR
    x0, x1, yc, w = C["x0"] * M, C["x1"] * M, C["y"] * M, C["width"] * M
    top, rr, pitch = C["top_z"] * M, C["roller_r"] * M, C["roller_pitch"] * M
    za = top - rr
    shapes = []
    for s in (-1, 1):
        shapes.append((B.box_tris(((x0 + x1) / 2, yc + s * (w / 2 + 30), 650), (x1 - x0, 60, 180)), BLUE))
        for xl in (x0 + 100, x0 + 1550, x0 + 3000, x1 - 100):
            shapes.append((B.box_tris((xl, yc + s * (w / 2 + 30), 280), (80, 80, 560)), DARK))
    for xl in (x0 + 100, x0 + 1550, x0 + 3000, x1 - 100):
        shapes.append((B.box_tris((xl, yc, 150), (80, w + 60, 60)), DARK))
    for k in range(int(round((x1 - x0) / pitch))):
        x = x0 + pitch * (k + 0.5)
        shapes.append((B.cylinder_tris((x, yc - w / 2 + 15, za), (x, yc + w / 2 - 15, za), rr, 12), STEEL))
    shapes.append((B.box_tris((x1 - 250, yc + w / 2 + 170, 560), (260, 200, 200)), GREY))       # drive motor
    for xe in (x0 - 20, x1 + 20):
        shapes.append((B.box_tris((xe, yc, top + 40), (40, w, 120)), YELLOW))                     # end stops
    return shapes


def carrier_shapes(x_mm):
    """Carrier pallet with its flange centre at world x = x_mm: plate on the rollers, seat ring + pins, V-post."""
    sx, sy = [v * M for v in L2.CARRIER["size"]]
    th, top = L2.CARRIER["thick"] * M, L2.CONVEYOR["top_z"] * M
    F = carrier_frame(x_mm / M)
    z0 = top + th - F.Pos()[2]
    shapes = [(B.box_tris((x_mm, L2.CONVEYOR["y"] * M, top + th / 2), (sx, sy, th)), BLUE2)]
    zt = PIPE_AXIS_Z - R_O - 10.0
    local = seat_ring_local(z0) + [(B.box_tris((860.0, 0.0, (z0 + zt - 35) / 2), (70.0, 70.0, zt - 35 - z0)), DARK),
                                   (B.box_tris((860.0, 0.0, zt - 17.5), (60.0, 240.0, 35.0)), POLY)]
    return shapes + in_frame(F, local)


def qc_arch_shapes():
    Q = L2.QC_ARCH
    x, y0, y1, bz = Q["x"] * M, Q["y0"] * M, Q["y1"] * M, Q["beam_z"] * M
    yc = y0 + 250
    return [(B.box_tris((x, y0, (bz + 120) / 2), (140, 140, bz + 120)), BLUE),
            (B.box_tris((x, y1, (bz + 120) / 2), (140, 140, bz + 120)), BLUE),
            (B.box_tris((x, (y0 + y1) / 2, bz + 60), (180, y1 - y0 + 140, 120)), BLUE),
            (B.box_tris((x, yc, bz - 60), (260, 260, 120)), GREY),                                  # carriage
            (B.cylinder_tris((x, yc, bz - 120), (x, yc, bz - 320), 60, 16), DARK),                  # laser marker
            (B.box_tris((x + 160, yc, bz - 200), (80, 120, 160)), DARK),                            # profile scanner
            (B.cylinder_tris((x, y1, bz + 120), (x, y1, bz + 200), 40, 12), LAMP["green"]),          # signal tower
            (B.cylinder_tris((x, y1, bz + 200), (x, y1, bz + 280), 40, 12), LAMP["amber"]),
            (B.cylinder_tris((x, y1, bz + 280), (x, y1, bz + 360), 40, 12), LAMP["red"]),
            (B.box_tris((x + 200, y1 + 150, 1500), (60, 380, 300)), DARK)]                          # HMI


def storage_rack_shapes():
    S = L2.STORAGE
    tx, sz = [v * M for v in S["x"]], [v * M for v in S["seat_z"]]
    gtop = [z - 40 for z in sz]                          # girder top (seat ring 30 + base 10 below the seat)
    gh = (120.0, 140.0)
    ry = max(abs(y) for bays in S["bay_y"] for y in bays) * M + 300.0     # girders reach past the outer seats
    shapes = []
    for t in (0, 1):
        shapes.append((B.box_tris((tx[t], 0, gtop[t] - gh[t] / 2), (300, 2 * ry, gh[t])), BLUE))
    legs0 = [-ry + 50, ry - 50] + [y * M for y in S["bay_y"][1]]           # ends + under the tier-1 V-posts
    for y in legs0:                                                                                # tier-0 legs
        shapes.append((B.box_tris((tx[0], y, (gtop[0] - gh[0]) / 2), (80, 80, gtop[0] - gh[0])), DARK))
    for k in range(5):                                                                             # tier-1 columns
        y = -ry + 50 + (2 * ry - 100) * k / 4.0
        shapes.append((B.box_tris((tx[1], y, (gtop[1] - gh[1]) / 2), (100, 100, gtop[1] - gh[1])), DARK))
    shapes.append((B.box_tris((tx[0] + 860, 0, 40), (100, 2 * ry, 80)), DARK))                   # tier-0 V-post rail
    for t in (0, 1):
        z_pipe = sz[t] + PIPE_AXIS_Z - R_O
        z_foot = 80.0 if t == 0 else gtop[0]
        for b in range(len(S["bay_y"][t])):
            F = storage_frame(t, b)
            shapes += in_frame(F, [(ring_tris(193.0, 222.0, -30.0, 0.0), STEEL),
                                   (ring_tris(0.0, 245.0, -40.0, -30.0), STEEL)])
            x, y = tx[t] + 860, S["bay_y"][t][b] * M
            shapes.append((B.box_tris((x, y, (z_foot + z_pipe - 40) / 2), (70, 70, z_pipe - 40 - z_foot)), YELLOW))
            shapes.append((B.box_tris((x, y, z_pipe - 20), (60, 240, 40)), POLY))
    back = L2.LOG_FENCE_X0 * M + 120.0
    for y in (-ry + 50, ry - 50):
        shapes.append((B.box_tris((back, y, L2.LOG_FENCE_H * M / 2), (120, 120, L2.LOG_FENCE_H * M)), BLUE))
    shapes.append((B.box_tris((back, 0, L2.LOG_FENCE_H * M - 100), (120, 2 * ry, 200)), BLUE))  # header with the sign
    shapes.append((B.box_tris((back + 80, 0, 100), (60, 2 * ry - 100, 200)), YELLOW))            # hazard bumper
    return shapes


def logistics_fence_shapes():
    """Fence of the logistics zone in the stage-1 style (posts every ~2 m, translucent panels) on the +Y, -Y and -X
    sides (the +X side is the stage-1 cell fence), openings with light-curtain posts, muting lamps at the cell
    opening."""
    x0, x1 = L2.LOG_FENCE_X0 * M, L2.CELL_FENCE_X0 * M
    y0, y1 = L2.LOG_FENCE_Y[0] * M, L2.LOG_FENCE_Y[1] * M
    H = L2.LOG_FENCE_H * M
    gaps = {"+Y": [], "-Y": [], "-X": []}
    curtains = []
    for o in L2.LOG_FENCE_OPENINGS:
        if o["side"] in ("+Y", "-Y"):
            g = (o["x0"] * M, o["x1"] * M)
            yy = y1 if o["side"] == "+Y" else y0
            curtains += [(g[0], yy), (g[1], yy)] if o.get("curtain") else []
        else:
            g = (o["y0"] * M, o["y1"] * M)
            curtains += [(x0, g[0]), (x0, g[1])] if o.get("curtain") else []
        gaps[o["side"]].append(g)
    door = L2.PERSONNEL_DOOR
    gaps[door["side"]].append(((door["center_x"] - door["width"] / 2) * M, (door["center_x"] + door["width"] / 2) * M))
    sides = [("y", y1, (x0, x1), gaps["+Y"]), ("y", y0, (x0, x1), gaps["-Y"]), ("x", x0, (y0, y1), gaps["-X"])]
    shapes = []
    for axis, val, rng, gp in sides:
        for s0, s1 in B.fence_segments(rng[0], rng[1], gp):
            n = max(1, int(round((s1 - s0) / 2000.0)))
            for k in range(n + 1):
                u = s0 + (s1 - s0) * k / n
                c = (u, val, H / 2) if axis == "y" else (val, u, H / 2)
                shapes.append((B.box_tris(c, (60, 60, H)), YELLOW))
            mid = ((s0 + s1) / 2, val, H / 2 + 50) if axis == "y" else (val, (s0 + s1) / 2, H / 2 + 50)
            size = (s1 - s0, 20, H - 250) if axis == "y" else (20, s1 - s0, H - 250)
            shapes.append((B.box_tris(mid, size), PANEL))
    for cx, cy in curtains:
        shapes.append((B.box_tris((cx, cy, 700), (80, 80, 1400)), [0.9, 0.8, 0.1, 1.0]))
    for s in (-1, 1):                                                                           # muting lamps
        shapes.append((B.box_tris((-2600.0, s * 1350.0, 750.0), (80, 80, 1500)), DARK))
        shapes.append((B.cylinder_tris((-2600.0, s * 1350.0, 1500.0), (-2600.0, s * 1350.0, 1620.0), 45, 12), LAMP["amber"]))
    return shapes


def agv_shapes():
    px, py = [v * M for v in L2.AGV["park"]]
    return [(B.box_tris((px, py, 150), (840, 1300, 300)), WHITE),
            (B.box_tris((px, py, 310), (700, 1100, 20)), DARK),
            (B.cylinder_tris((px + 300, py + 550, 300), (px + 300, py + 550, 380), 30, 12), LAMP["amber"])]


def floor_shapes():
    return [(B.box_tris((-12400.0, 0.0, -25.0), (3200.0, 18000.0, 50.0)), CONCRETE)]   # beyond the stage-1 slab


def handler_bed_shapes():
    x0, x1 = L2.HANDLER_TRACK_BED[0] * M + 60, L2.HANDLER_TRACK_BED[1] * M - 60
    y, mid, ln = HANDLER_TRACK_Y, (x0 + x1) / 2, x1 - x0
    shapes = [(B.box_tris((mid, y, 120), (ln, 940, 240)), DARK)]
    for s in (-1, 1):
        shapes.append((B.box_tris((mid, y + s * 320, 265), (ln - 40, 60, 50)), STEEL))          # linear rails
    shapes.append((B.box_tris((mid, y - 420, 260), (ln - 80, 30, 40)), DARK))                    # gear rack
    shapes.append((B.box_tris((mid, y + 640, 90), (ln, 220, 180)), GREY))                        # cable trough
    for xe in (x0 - 30, x1 + 30):
        shapes.append((B.box_tris((xe, y, 200), (60, 900, 400)), YELLOW))                        # end stops
    return shapes


def handler_carriage_shapes(x):
    y, top = HANDLER_TRACK_Y, HANDLER_TOP_Z
    return [(B.box_tris((x - 85, y, (290 + top - 30) / 2), (890, 1100, top - 30 - 290)), BLUE2),
            (B.box_tris((x, y, top - 15), (900, 900, 30)), STEEL),
            (B.box_tris((x - 100, y - 620, 400), (250, 180, 250)), GREY)]                         # drive


def studs_local():
    """The six clamp studs / nuts over the flange (spool-local), shown while clamped (StudsUp)."""
    out = []
    for k in range(6):
        a = 2 * math.pi * (2 * k + 0.5) / 12
        x, y = 177.5 * math.cos(a), 177.5 * math.sin(a)
        out += B.cylinder_tris((x, y, FLANGE_THK), (x, y, FLANGE_THK + 20), 17.0, 6)
    return out


def logistics_static_objects():
    """Static equipment objects: name -> [(tris, colour)] (world mm)."""
    return {"Kit pallet": kit_pallet_shapes(), "Pipe buffer": pipe_buffer_shapes(),
            "Assembly station": station_shapes(), "Tack pedestal": tack_pedestal_shapes(),
            "QC arch": qc_arch_shapes(), "Storage rack": storage_rack_shapes(),
            "Logistics fence": logistics_fence_shapes(), "AGV": agv_shapes(), "Logistics floor": floor_shapes()}


def part_placements():
    """Movable parts (GripperClose candidates) and decor parts with their start frames (world mm)."""
    movable = {PART_NAMES["flange"]: ("flange", kit_flange_frame(0)), PART_NAMES["elbow"]: ("elbow", kit_elbow_frame(0)),
               PART_NAMES["pipe"]: ("pipe", buffer_pipe_frame(0)), PART_NAMES["flange2"]: ("flange", kit_flange_frame(1))}
    decor = {"Kit elbow 2": (("elbow",), kit_elbow_frame(1))}
    for i in range(1, len(L2.PIPE_BUFFER_X)):
        decor["Buffer pipe %d" % (i + 1)] = (("pipe",), buffer_pipe_frame(i))
    for t, b in L2.STORAGE_FILLED:
        decor["Stored spool T%dB%d" % (t, b)] = (("flange", "elbow", "pipe"), storage_frame(t, b))
    return movable, decor


def grasp_points():
    """GripperClose candidates: part name -> grasp points (part frame, mm).  The pipe carries both the loose-pipe
    grasp and the assembled-spool grasp (the tongs close on the pipe leg of a tacked / welded spool)."""
    g = {k: [v * M for v in L2.GRASP[k][0]] for k in L2.GRASP}
    return {PART_NAMES["flange"]: [g["flange"]], PART_NAMES["elbow"]: [g["elbow"]],
            PART_NAMES["pipe"]: [g["pipe"], g["spool"]], PART_NAMES["flange2"]: [g["flange"]]}


# =====================================================================================================================
# 6. Macro sources (Python programs added to the station)
# =====================================================================================================================
ARC_ON2_SRC = '''# ArcOn: start the weld-bead trace (simulated spray gun projected on the elbow, which carries both seams).
# Generated by build_station2.py (replaces the stage-1 macro of the same name: the stage-2 spool is 3 parts)
from robodk.robolink import *
from robodk.robomath import *
RDK = Robolink()
tool = RDK.Item("MIG torch", ITEM_TYPE_TOOL)
part = RDK.Item({part!r}, ITEM_TYPE_OBJECT)
gid = RDK.getParam("ARC_GUN_ID")
if not gid:
    near = [0, 0, -3, 3, 0, -3, 0, 3, -3, 0.85, 0.55, 0.25, 1.0]
    far = [0, 0, 8, 6, 0, 8, 0, 6, 8, 0.85, 0.55, 0.25, 1.0]
    gid = RDK.Spray_Add(tool, part, "PROJECT PARTICLE=SPHERE(3.5,8,1,1,0.45) STEP=1x1 RAND=0", Mat([near, far]).tr())
    RDK.setParam("ARC_GUN_ID", str(gid))
RDK.Spray_SetState(SPRAY_ON, int(gid))
'''

ARC_OFF2_SRC = '''# ArcOff: stop the weld-bead trace. Generated by build_station2.py
from robodk.robolink import *
RDK = Robolink()
gid = RDK.getParam("ARC_GUN_ID")
RDK.Spray_SetState(SPRAY_OFF, int(gid) if gid else -1)
'''

TACK_ARC_ON_SRC = '''# TackArcOn: tack-welding arc on (short spray trace of the tack torch on the elbow). Generated by build_station2.py
from robodk.robolink import *
from robodk.robomath import *
RDK = Robolink()
gid = RDK.getParam("TACK_GUN_ID")
if not gid:
    tool = RDK.Item({torch!r}, ITEM_TYPE_TOOL)
    part = RDK.Item({part!r}, ITEM_TYPE_OBJECT)
    near = [0, 0, -3, 4, 0, -3, 0, 4, -3, 0.9, 0.6, 0.3, 1.0]
    far = [0, 0, 8, 7, 0, 8, 0, 7, 8, 0.9, 0.6, 0.3, 1.0]
    gid = RDK.Spray_Add(tool, part, "PROJECT PARTICLE=SPHERE(4,8,1,1,0.5) STEP=1x1 RAND=0", Mat([near, far]).tr())
    RDK.setParam("TACK_GUN_ID", str(gid))
RDK.Spray_SetState(SPRAY_ON, int(gid))
'''

TACK_ARC_OFF_SRC = '''# TackArcOff: tack-welding arc off. Generated by build_station2.py
from robodk.robolink import *
RDK = Robolink()
gid = RDK.getParam("TACK_GUN_ID")
if gid:
    RDK.Spray_SetState(SPRAY_OFF, int(gid))
'''

GRIPPER_CLOSE_SRC = '''# GripperClose: close the tong gripper and take the part whose grasp point is at the TCP, plus every other part with
# the same frame (the parts of an assembled spool share the spool frame). Generated by build_station2.py
from robodk.robolink import *
from robodk.robomath import *
RDK = Robolink()
robot = RDK.Item({robot!r}, ITEM_TYPE_ROBOT)
tool = RDK.Item({tool!r}, ITEM_TYPE_TOOL)
GRASP_POINTS = {grasp!r}     # part -> grasp points in the part frame (mm)
TOL = {tol!r}                # mm


def dist(a, b):
    return sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5


tcp = (robot.PoseAbs() * robot.SolveFK(robot.Joints()) * tool.PoseTool()).Pos()
best = None
for name, pts in GRASP_POINTS.items():
    part = RDK.Item(name, ITEM_TYPE_OBJECT)
    if not part.Valid() or part.Parent() == tool:
        continue
    P = part.PoseAbs()
    d = min(dist(P * p, tcp) for p in pts)
    if best is None or d < best[0]:
        best = (d, part)
if best is None or best[0] > TOL:
    RDK.ShowMessage("GripperClose: no part within %.0f mm of the gripper TCP" % TOL, False)
else:
    P0 = best[1].PoseAbs()
    for name in GRASP_POINTS:
        part = RDK.Item(name, ITEM_TYPE_OBJECT)
        if not part.Valid() or part.Parent() == tool:
            continue
        Pi = part.PoseAbs()
        if part == best[1] or (dist(Pi.Pos(), P0.Pos()) < 2.0 and pose_angle_between(Pi, P0) < 0.02):
            part.setParentStatic(tool)
RDK.setParam("GRIPPER", "CLOSED")
'''

GRIPPER_OPEN_SRC = '''# GripperOpen: open the tong gripper and release the held parts to the holder under them: the positioner faceplate,
# the carrier pallet or the Logistics frame; a part within a few mm of a nominal seat is snapped onto it (removes the
# numeric residual so the next grasp matches exactly). Generated by build_station2.py
from robodk.robolink import *
from robodk.robomath import *
RDK = Robolink()
tool = RDK.Item({tool!r}, ITEM_TYPE_TOOL)
HOLDERS = {holders!r}     # (frame name, zone [xmin, xmax, ymin, ymax] of the part origin or None, nominal poses)
held = [c for c in tool.Childs() if c.Type() == ITEM_TYPE_OBJECT]
if held:
    p = held[0].PoseAbs().Pos()
    name, zone, seats = HOLDERS[-1]
    for h in HOLDERS:
        z = h[1]
        if z is not None and z[0] <= p[0] <= z[1] and z[2] <= p[1] <= z[3]:
            name, zone, seats = h
            break
    holder = RDK.Item(name, ITEM_TYPE_FRAME)
    for part in held:
        part.setParentStatic(holder)
        rel = part.Pose()
        for s in seats:
            S = Mat(s)
            if norm(subs3(rel.Pos(), S.Pos())) < {snap_mm!r} and pose_angle_between(rel, S) < {snap_rad!r}:
                part.setPose(S)
                break
RDK.setParam("GRIPPER", "OPEN")
'''

STUDS_SRC = '''# {name}: positioner clamp studs {state} (the flange of the spool is {verb}). Generated by build_station2.py
from robodk.robolink import *
RDK = Robolink()
studs = RDK.Item({studs!r}, ITEM_TYPE_OBJECT)
if studs.Valid():
    studs.setVisible({visible})
RDK.setParam("POS_STUDS", {param!r})
'''

RESET_PARTS_SRC = '''# ResetParts: put every movable part back to its start pose (kit pallet / pipe buffer), studs down, clear the bead
# traces - lets DemoCycle2 run again. Generated by build_station2.py
from robodk.robolink import *
from robodk.robomath import *
RDK = Robolink()
frame = RDK.Item({frame!r}, ITEM_TYPE_FRAME)
START = {start!r}
for name, rows in START.items():
    part = RDK.Item(name, ITEM_TYPE_OBJECT)
    if part.Valid():
        part.setParent(frame)
        part.setPose(Mat(rows))
studs = RDK.Item({studs!r}, ITEM_TYPE_OBJECT)
if studs.Valid():
    studs.setVisible(False)
RDK.Spray_Clear()
RDK.setParam("GRIPPER", "OPEN")
RDK.setParam("POS_STUDS", "DOWN")
'''

QC_SRC = '''# QC_MarkScan: laser marking of the spool ID + weld-profile scan of seam B at the QC arch. Generated by build_station2.py
from robodk.robolink import *
RDK = Robolink()
RDK.ShowMessage("QC arch: SPL-02 marked, seam B profile scanned - QC OK", False)
RDK.setParam("QC", "OK")
'''

VIEW2_SRC = '''# {name}: move the 3D view to stage-2 shot {shot} (scene frames {f0}-{f1}; camera {cam} mm -> aim {aim} mm,
# lens {lens} mm).  Generated by build_station2.py from stage2_logistics/cameras2.py
from robodk.robolink import *
from robodk.robomath import *
RDK = Robolink()
RDK.setViewPose(Mat({rows}))
'''


def stage2_shots(events=None):
    """Camera shots of stage2_logistics/cameras2.py (parsed with ast: that module imports bpy)."""
    try:
        with open(CAMERAS2_PY, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
    except (OSError, SyntaxError):
        return []
    raw = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("PRE_SHOTS", "POST_SHOTS")
                                                for t in node.targets):
            try:
                raw += list(ast.literal_eval(node.value))
            except ValueError:
                pass
    events = events or {}
    out = []
    for s in raw:
        name, a, b, c0, _c1, a0, _a1, lens = s[:8]
        f0 = events[a[0]] + a[1] if a[0] in events else None
        f1 = events[b[0]] + b[1] if b[0] in events else None
        out.append(dict(name=name, cam=[v * M for v in c0], aim=[v * M for v in a0], lens=lens, f0=f0, f1=f1))
    return out


def holder_table():
    """(frame, zone, nominal seats) for GripperOpen; the last entry (zone None) is the default."""
    C = L2.CONVEYOR
    seats = [station_frame(), storage_frame(*L2.STORAGE_TARGET)]
    seats += [kit_flange_frame(i) for i in range(len(L2.KIT_FLANGES))]
    seats += [kit_elbow_frame(i) for i in range(len(L2.KIT_ELBOWS))]
    seats += [buffer_pipe_frame(i) for i in range(len(L2.PIPE_BUFFER_X))]
    seats += [storage_frame(t, b) for t in (0, 1) for b in range(len(L2.STORAGE["bay_y"][t]))]
    rows = lambda T: [[round(float(T[i, j]), 9) for j in range(4)] for i in range(4)]   # noqa: E731
    return [(FACEPLATE_FRAME, [L2.CELL_FENCE_X0 * M, 1e5, -1e5, 1e5], [rows(eye(4))]),
            (CARRIER_FRAME, [C["x0"] * M, C["x1"] * M, (C["y"] - C["width"] / 2) * M - 300, (C["y"] + C["width"] / 2) * M],
             [rows(eye(4))]),
            (LOGISTICS_FRAME, None, [rows(T) for T in seats])]


def write_macros2(cell):
    """Stage-2 macros into GEN_DIR (stale View2_*.py removed); returns {name: path}."""
    os.makedirs(GEN_DIR, exist_ok=True)
    rows = lambda T: [[round(float(T[i, j]), 6) for j in range(4)] for i in range(4)]   # noqa: E731
    elbow = PART_NAMES["elbow"]
    src = {
        "GripperClose": GRIPPER_CLOSE_SRC.format(robot=HANDLER_NAME, tool=GRIPPER_NAME, grasp=grasp_points(), tol=40.0),
        "GripperOpen": GRIPPER_OPEN_SRC.format(tool=GRIPPER_NAME, holders=holder_table(), snap_mm=15.0, snap_rad=0.05),
        "StudsUp": STUDS_SRC.format(name="StudsUp", state="up", verb="clamped", studs=STUDS_NAME, visible=True, param="UP"),
        "StudsDown": STUDS_SRC.format(name="StudsDown", state="down", verb="free", studs=STUDS_NAME, visible=False,
                                      param="DOWN"),
        "TackArcOn": TACK_ARC_ON_SRC.format(torch=TACK_TORCH_NAME, part=elbow),
        "TackArcOff": TACK_ARC_OFF_SRC,
        "ResetParts": RESET_PARTS_SRC.format(frame=LOGISTICS_FRAME, studs=STUDS_NAME,
                                             start={n: rows(T) for n, T in cell.part_start.items()}),
        "QC_MarkScan": QC_SRC,
    }
    if USE_VIEWS:
        for s in stage2_shots((cell.plan or {}).get("events")):
            name = "View2_" + s["name"]
            src[name] = VIEW2_SRC.format(name=name, shot=s["name"], f0=s["f0"], f1=s["f1"], cam=s["cam"], aim=s["aim"],
                                         lens=s["lens"], rows=rows(B.view_pose(s["cam"], s["aim"])))
    files = {}
    for name, text in src.items():
        path = os.path.join(GEN_DIR, name + ".py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        files[name] = path
    for old in glob.glob(os.path.join(GEN_DIR, "View2_*.py")):
        if os.path.basename(old)[:-3] not in files:
            os.remove(old)
    return files


# =====================================================================================================================
# 7. RoboDK builders
# =====================================================================================================================
class Cell2(B.Cell):
    """Items of the full-cycle station (stage-1 items in the B.Cell fields)."""

    def __init__(self):
        B.Cell.__init__(self)
        self.plan = {}
        self.logistics = None
        self.faceplate_frame = None
        self.carrier_frame = None
        self.htrack = None
        self.handler = None
        self.gripper = None
        self.tack = None
        self.tack_torch = None
        self.carrier = None
        self.studs = None
        self.objects = {}
        self.parts = {}
        self.part_start = {}
        self.target_info = {}         # stage-2 Cartesian targets: name -> dict(robot, pose, x, seed, joints, event)
        self.track_programs = {}      # HTrack_* -> carriage x (mm)
        self.steps = {}               # stage-2 program -> choreography steps
        self.problems = []


def add_library_robot(RDK, globs, model, what):
    """Robot from the RoboDK library folder (first glob match); a file dialog if it is not there."""
    path, lib = B.find_library_file(RDK, globs)
    if path is None:
        msg = ("%s (%s) not found in the RoboDK library folder (%s). Download it from the online library "
               "(File > Open online library) and re-run, or pick the .robot file now." % (model, what, lib))
        print(msg)
        RDK.ShowMessage(msg, False)
        from robodk import robodialogs
        path = robodialogs.getOpenFileName(lib or "", "", "Select the %s .robot file" % model, ".robot",
                                           [("RoboDK robot", ".robot")])
        if not path:
            raise RuntimeError("No robot file selected for " + model)
    robot = RDK.AddFile(path)
    if not robot.Valid():
        raise RuntimeError("Could not load robot file: " + path)
    robot.setName(model)
    return robot


def close_open_station(RDK):
    """Idempotent re-runs: close a station with the same name that is still open (unsaved changes are lost)."""
    try:
        st = RDK.Item(STATION_NAME, rl.ITEM_TYPE_STATION)
        if st is not None and st.Valid() and st.Name() == STATION_NAME:
            RDK.setActiveStation(st)
            RDK.CloseStation()
    except Exception as exc:                  # older RoboDK builds: just add a new station
        print("Could not close the previous station (%s)" % exc)


def add_faceplate_frame(RDK, cell):
    """Replaces stage-1 add_spool: a reference frame on the faceplate at the spool frame (30 mm above the plate), the
    seam curves on it (for the curve-follow projects) and the clamp studs (hidden = retracted)."""
    fr = RDK.AddFrame(FACEPLATE_FRAME)
    B.mount_on_flange(RDK, fr, cell.rot, 1, transl(0, 0, B.POS_FIXTURE_THICK))
    cell.faceplate_frame = fr
    for which, color in (("A", [1.0, 0.2, 0.1, 1.0]), ("B", [0.1, 0.5, 1.0, 1.0])):
        curve = RDK.AddCurve(B.seam_points_local(which), 0, False, rl.PROJECTION_NONE)
        curve.setName("Seam %s curve" % which)
        curve.setColorCurve(color)
        curve.setParent(fr)
        curve.setPose(eye(4))
        cell.curves[which] = curve
    studs = B.add_shape_object(RDK, STUDS_NAME, [(studs_local(), STEEL)])
    studs.setParent(fr)
    studs.setPose(eye(4))
    studs.setVisible(False)
    cell.studs = studs
    cell.objects[STUDS_NAME] = studs


def add_static(RDK, cell):
    for name, shapes in logistics_static_objects().items():
        cell.objects[name] = B.add_shape_object(RDK, name, shapes)


def add_handler_track(RDK, cell):
    """Handler floor track: 1-axis translation mechanism along world X, joint value = carriage x (mm)."""
    base = handler_track_base()
    bed = B.add_shape_object(RDK, "Handler track bed", B.mech_shapes(handler_bed_shapes(), base))
    carriage = B.add_shape_object(RDK, "Handler track carriage", B.mech_shapes(handler_carriage_shapes(HANDLER_HOME_X), base))
    track = RDK.BuildMechanism(rl.MAKE_ROBOT_1T, [bed, carriage], [], [HANDLER_HOME_X], [HANDLER_HOME_X], [1],
                               [HANDLER_TRACK_X[0]], [HANDLER_TRACK_X[1]], base, eye(4), HTRACK_NAME)
    if not track.Valid():
        raise RuntimeError("BuildMechanism failed for the handler track")
    track.setJoints([HANDLER_HOME_X])
    cell.htrack = track
    return track


def add_handler(RDK, cell):
    robot = add_library_robot(RDK, HANDLER_LIBRARY_GLOBS, HANDLER_NAME, "handler robot")
    B.mount_on_flange(RDK, robot, cell.htrack, 1, invH(handler_track_base()) * handler_base_world(0.0))
    robot.setJointsHome(HANDLER_Q_HOME)
    robot.setJoints(HANDLER_Q_HOME)
    cell.handler = robot
    stl = os.path.join(GEN_DIR, "gripper.stl")
    B.write_stl(stl, gripper_mesh_triangles(), GRIPPER_NAME)
    tool = RDK.AddFile(stl, robot)
    tool.setName(GRIPPER_NAME)
    tool.setPoseTool(gripper_tcp())
    tool.setColor([0.15, 0.33, 0.56, 1.0])
    robot.setPoseTool(tool)
    cell.gripper = tool
    return robot


def add_tack_robot(RDK, cell):
    robot = add_library_robot(RDK, TACK_LIBRARY_GLOBS, TACK_NAME, "tack-welding robot")
    B.base_frame_of(RDK, robot).setPose(tack_base_world())
    robot.setJointsHome(TACK_Q_HOME)
    robot.setJoints(TACK_Q_HOME)
    cell.tack = robot
    stl = os.path.join(GEN_DIR, "torch.stl")
    if not os.path.exists(stl):
        B.write_stl(stl, B.torch_mesh_triangles(), "MIG torch")
    torch = RDK.AddFile(stl, robot)
    torch.setName(TACK_TORCH_NAME)
    torch.setPoseTool(B.tcp_pose())
    torch.setColor([0.15, 0.15, 0.17, 1.0])
    robot.setPoseTool(torch)
    cell.tack_torch = torch
    return robot


def add_conveyor(RDK, cell):
    """Output conveyor (static link) + carrier pallet (moving link): 1T mechanism along X, joint = flange-centre x."""
    base = carrier_track_base()
    x_load = L2.CONV_LOAD_X * M
    conv = B.add_shape_object(RDK, "Output conveyor", B.mech_shapes(conveyor_shapes(), base))
    pallet = B.add_shape_object(RDK, "Carrier pallet", B.mech_shapes(carrier_shapes(x_load), base))
    mech = RDK.BuildMechanism(rl.MAKE_ROBOT_1T, [conv, pallet], [], [x_load], [x_load], [1],
                              [L2.CONV_END_X * M], [L2.CONV_LOAD_X * M], base, eye(4), CARRIER_NAME)
    if not mech.Valid():
        raise RuntimeError("BuildMechanism failed for the carrier pallet")
    mech.setJoints([x_load])
    cell.carrier = mech
    fr = RDK.AddFrame(CARRIER_FRAME)
    B.mount_on_flange(RDK, fr, mech, 1, invH(base) * carrier_frame(0.0))
    cell.carrier_frame = fr
    return mech


def add_parts(RDK, cell):
    tris = spool_part_tris()
    color = dict(flange=FLANGE_C, elbow=PIPE_C, pipe=PIPE_C)
    movable, decor = part_placements()
    for name, (kind, F) in movable.items():
        obj = B.add_shape_object(RDK, name, [(tris[kind], color[kind])])
        obj.setParent(cell.logistics)
        obj.setPose(F)
        cell.parts[name] = obj
        cell.part_start[name] = F
    for name, (kinds, F) in decor.items():
        obj = B.add_shape_object(RDK, name, [(tris[k], color[k]) for k in kinds])
        obj.setParent(cell.logistics)
        obj.setPose(F)
        cell.objects[name] = obj


def add_macros2(RDK, cell):
    """Stage-1 macros (ArcOn/ArcOff re-targeted to the elbow part, View_*) + stage-2 macros, all from GEN_DIR."""
    files = B.write_macros()
    files.update(write_macros2(cell))
    for name, path in files.items():
        item = RDK.AddFile(path)
        item.setName(name)
        cell.macros[name] = item


def _ik_list(q):
    try:
        ql = list(q.tolist()) if isinstance(q, Mat) else list(q or [])
    except Exception:
        return None
    return [float(v) for v in ql[:6]] if len(ql) >= 6 else None


def solve_ik(robot, pose, seed, tool, base):
    """Joints (deg) of `robot` reaching TCP `pose` (world mm) near `seed`, base pose given; None if unreachable."""
    q = _ik_list(robot.SolveIK(pose, list(seed), tool, invH(base)))
    if q is not None:
        robot.setJoints(q)
    return q


def robot_target(RDK, cell, who, name, wp):
    """Cartesian target of the handler / tack robot in the Logistics frame; the IK solution near the plan seed (the
    configuration RoboDK keeps for the target) is stored with it.  Handler targets are solved with the carriage at
    wp['x'] - the programs move the carriage there first."""
    if name in cell.targets:
        return cell.targets[name]
    if who == "handler":
        robot = cell.handler
        cell.htrack.setJoints([wp["x"]])
        q = solve_ik(robot, wp["pose"], wp["seed"], gripper_tcp(), handler_base_world(wp["x"]))
    else:
        robot = cell.tack
        q = solve_ik(robot, wp["pose"], wp["seed"], B.tcp_pose(), tack_base_world())
    t = RDK.AddTarget(name, cell.logistics, robot)
    t.setAsCartesianTarget()
    t.setJoints(q if q is not None else list(wp["seed"]))
    t.setPose(wp["pose"])
    cell.targets[name] = t
    cell.target_info[name] = dict(wp, robot=who, joints=q)
    if q is None:
        msg = "%s target %s: no IK solution near the plan seed (carriage x=%s)" % (who, name, wp.get("x"))
        cell.problems.append(msg)
        print(msg)
    return t


class Prog2(B.Prog):
    """Program of a stage-2 robot (handler / tack robot): Logistics frame, its own tool."""

    def __init__(self, RDK, cell, name, robot, tool, frame):
        B.Prog.__init__(self, RDK, cell, name, robot, is_robot=False)
        self.frame, self.tool = frame, tool
        self.item.setPoseFrame(frame)
        self.item.setPoseTool(tool)

    def joint_target(self, name, joints):
        t = self.cell.targets.get(name)
        if t is None:
            t = self.RDK.AddTarget(name, self.frame, self.mech)
            t.setAsJointTarget()
            t.setJoints(list(joints))
            self.cell.targets[name] = t
        return t


def track_program(RDK, cell, x, label):
    """HTrack_<label>: moves the handler carriage to x (mm)."""
    name = "HTrack_" + label
    if name not in cell.programs:
        p = B.Prog(RDK, cell, name, cell.htrack, is_robot=False)
        p.item.setSpeedJoints(HANDLER_TRACK_MMS)
        p.movej(p.joint_target(name + "_T", [float(x)]))
        cell.track_programs[name] = float(x)
    elif abs(cell.track_programs[name] - x) > 1e-6:
        raise ValueError("track label %s used for two positions" % label)
    return name


def realize_program(RDK, cell, who, name, steps, wps):
    """Turn choreography steps (see handler_plan) into a RoboDK program of the handler / tack robot."""
    if who == "handler":
        robot, tool, speeds, home = cell.handler, cell.gripper, HANDLER_SPEEDS, ("HandlerHome", HANDLER_Q_HOME)
    else:
        robot, tool, speeds, home = cell.tack, cell.tack_torch, TACK_SPEEDS, ("TackHome", TACK_Q_HOME)
    p = Prog2(RDK, cell, name, robot, tool, cell.logistics)
    for st in steps:
        kind = st[0]
        if kind == "comment":
            p.comment(st[1])
        elif kind == "speed":
            lin, jnt, rnd = speeds[st[1]]
            p.item.setSpeed(lin, jnt)
            p.item.setRounding(rnd)
        elif kind == "track":
            p.call(track_program(RDK, cell, st[1], st[2]))
        elif kind in ("movej", "movel"):
            t = robot_target(RDK, cell, who, st[1], wps[st[1]])
            (p.movej if kind == "movej" else p.movel)(t)
        elif kind == "home":
            p.movej(p.joint_target(*home))
        elif kind == "call":
            p.call(st[1])
        elif kind == "view":
            if "View2_" + st[1] in cell.macros:
                p.call("View2_" + st[1])
        elif kind == "do":
            p.item.setDO(st[1], st[2])
        elif kind == "pause":
            p.item.Pause(st[1])
        else:
            raise ValueError("unknown step %r" % (st,))
    cell.steps[name] = steps
    return p


def add_stage2_mechanism_programs(RDK, cell):
    """Carrier (Conv_*), faceplate load / unload orientation, Conveyor_QC."""
    for name, x in (("Conv_ToLoad", L2.CONV_LOAD_X), ("Conv_ToQC", L2.CONV_QC_X), ("Conv_ToEnd", L2.CONV_END_X)):
        p = B.Prog(RDK, cell, name, cell.carrier, is_robot=False)
        p.item.setSpeedJoints(CONV_SPEED_MMS)
        p.movej(p.joint_target(name + "_T", [x * M]))
    for name, dg in (("Pos_RotLoad", L2.POS_LOAD_ROT), ("Pos_RotUnload", L2.POS_UNLOAD_ROT)):
        p = B.Prog(RDK, cell, name, cell.rot, is_robot=False)
        p.item.setSpeedJoints(B.POS_INDEX_DEGS)
        p.movej(p.joint_target(name + "_T", [float(dg)]))
    q = B.Prog(RDK, cell, "Conveyor_QC", cell.carrier, is_robot=False)
    q.comment("Carrier to the QC arch: laser marking of the spool ID, weld-profile scan across seam B, then to the end")
    q.item.setDO("QCTower", 1)
    q.call("Conv_ToQC")
    for shot, io, ms in (("S2_14_mark", "LaserMarker", 2000), ("S2_15_scan", "ProfileScan", 2700)):
        if "View2_" + shot in cell.macros:
            q.call("View2_" + shot)
        q.item.setDO(io, 1)
        if io == "LaserMarker":
            q.call("QC_MarkScan")
        q.item.Pause(ms)
        q.item.setDO(io, 0)
    q.item.setDO("QCTower", 2)
    q.call("Conv_ToEnd")
    q.item.setDO("QCTower", 0)


def add_stage2_robot_programs(RDK, cell):
    wps, progs = handler_plan(cell.plan)
    for name, steps in progs:
        realize_program(RDK, cell, "handler", name, steps, wps)
    twps, tprogs = tack_plan(cell.plan)
    for name, steps in tprogs:
        realize_program(RDK, cell, "tack", name, steps, twps)
    # Cycle2_Home and the main program
    h = Prog2(RDK, cell, "Cycle2_Home", cell.handler, cell.gripper, cell.logistics)
    h.comment("Start state of the full cycle: parts in the kit / buffer, all machines home, faceplate at %.0f deg"
              % L2.POS_LOAD_ROT)
    for c in CYCLE2_HOME:
        h.call(c)
    for io in ("GripperClosed", "StationClamps", "GapOK", "LightCurtainMuting", "TackArc", "WeldArc", "LaserScan"):
        h.item.setDO(io, 0)
    d = Prog2(RDK, cell, "DemoCycle2", cell.handler, cell.gripper, cell.logistics)
    d.comment("Full cycle: kitting -> tacking -> loading -> DemoCycle (stage-1 welding) -> unloading -> conveyor + QC "
              "-> storage")
    for kind, arg in DEMO_CYCLE2:
        if kind == "call":
            d.call(arg)
        elif kind == "view":
            if "View2_" + arg in cell.macros:
                d.call("View2_" + arg)
        elif kind == "pause":
            d.item.Pause(arg)


def set_start_state(cell):
    cell.htrack.setJoints([HANDLER_HOME_X])
    cell.handler.setJoints(HANDLER_Q_HOME)
    cell.tack.setJoints(TACK_Q_HOME)
    cell.carrier.setJoints([L2.CONV_LOAD_X * M])
    B.set_state(cell, B.ROBOT_HOME_Y, 0.0, L2.POS_LOAD_ROT)
    cell.robot.setJoints(B.ROBOT_Q_HOME)


def update_programs(cell, names):
    for name in names:
        prog = cell.programs[name]
        prog.ShowTargets(False)
        try:
            ok, t_s, dist, frac, problems = prog.Update(rl.COLLISION_OFF)
            if frac < 1.0:
                msg = "Program %s: %.0f%% of the path is valid - %s" % (name, frac * 100, problems)
                cell.problems.append(msg)
                print(msg)
        except Exception as exc:
            print("Program %s: Update failed (%s)" % (name, exc))


STAGE1_ROBOT_PROGRAMS = ("Robot_Transit", "Robot_ApproachA", "Robot_ScanA", "Robot_WeldA", "Robot_ApproachB1",
                         "Robot_WeldB_S1", "Robot_WeldB_S2", "Robot_ApproachB2", "Robot_WeldB_S3", "Robot_WeldB_S4",
                         "Robot_WeldA_MoveL")


def build(RDK, plan=None):
    """Build the full-cycle station in the connected RoboDK instance; returns the Cell2 with all items."""
    saved = (B.GEN_DIR, B.ARC_ON_SRC, B.ARC_OFF_SRC)
    B.GEN_DIR = GEN_DIR                           # stage-1 generated files (torch.stl, macros) go to our folder
    B.ARC_ON_SRC = ARC_ON2_SRC.format(part=PART_NAMES["elbow"])
    B.ARC_OFF_SRC = ARC_OFF2_SRC
    try:
        os.makedirs(GEN_DIR, exist_ok=True)
        cell = Cell2()
        cell.plan = load_plan() if plan is None else plan
        RDK.Render(False)
        if REPLACE_OPEN_STATION:
            close_open_station(RDK)
        cell.station = RDK.AddStation(STATION_NAME)
        cell.frame = RDK.AddFrame("Cell")
        cell.frame.setPose(eye(4))
        # ---- stage 1: the welding cell (without the one-piece spool)
        B.add_environment(RDK, cell)
        B.add_track(RDK, cell)
        B.add_robot(RDK, cell)
        B.add_torch(RDK, cell)
        B.add_positioner(RDK, cell)
        add_faceplate_frame(RDK, cell)
        # ---- stage 2: logistics zone
        cell.logistics = RDK.AddFrame(LOGISTICS_FRAME)
        cell.logistics.setPose(eye(4))
        add_static(RDK, cell)
        add_handler_track(RDK, cell)
        add_handler(RDK, cell)
        add_tack_robot(RDK, cell)
        add_conveyor(RDK, cell)
        add_parts(RDK, cell)
        # ---- macros, cameras, programs
        add_macros2(RDK, cell)
        B.add_cameras(RDK, cell)
        B.add_mechanism_programs(RDK, cell)
        B.add_robot_programs(RDK, cell)
        B.add_demo_cycle(RDK, cell)
        if ADD_CURVE_FOLLOW:
            B.add_curve_follow_projects(RDK, cell)
        add_stage2_mechanism_programs(RDK, cell)
        add_stage2_robot_programs(RDK, cell)
        set_start_state(cell)
        # handler programs move the carriage through sub-program calls, which Update() does not simulate: only the
        # welding-robot and the tack-robot programs are checked here
        update_programs(cell, STAGE1_ROBOT_PROGRAMS + ("TackWeld",))
        RDK.setSimulationSpeed(1.0)
        shots = {s["name"]: s for s in stage2_shots()}
        s = shots.get("S2_01_wide")
        RDK.setViewPose(B.view_pose(s["cam"], s["aim"]) if s else B.view_pose((-15000, -9000, 9000), (-4500, 0, 800)))
        RDK.Render(True)
        if SAVE_RDK:
            RDK.Save(os.path.join(HERE, STATION_NAME + ".rdk"), cell.station)
        return cell
    finally:
        B.GEN_DIR, B.ARC_ON_SRC, B.ARC_OFF_SRC = saved


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--export-plan" in argv:
        path = export_plan(use_cache="--fresh" not in argv)
        print("plan snapshot written: " + path)
        return
    RDK = rl.Robolink()
    cell = build(RDK)
    print("Station '%s' built: %d programs, %d targets, %d problems. Run 'DemoCycle2' for the full cycle "
          "(Tools > Record to capture a video)." % (STATION_NAME, len(cell.programs), len(cell.targets), len(cell.problems)))
    for p in cell.problems:
        print("  - " + p)


if __name__ == "__main__":
    main()
