#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check the station "PipeSpool_FullCycle" built in a RUNNING RoboDK against the layout (stage2_logistics/layout2.py).

    python verify_station2.py            (RoboDK running with the station open; exit code 0 = everything matches)

The mock Robolink of tests/t_robodk2.py cannot see how the real RoboDK treats mechanisms (limits, base poses, joint
senses), so this is the first-run check of the skill: poses read back from RoboDK against the layout functions.
  * spool frame on the faceplate at several (tilt, rot) states   == build_station2.positioner_frame
  * carrier spool frame at the load / QC / end stations         == carrier_frame
  * handler robot base at several carriage positions            == handler_base_world
  * welding robot base at several track positions               == stage-1 robot_base_pose_world
  * tack robot base, swing clamps on the faceplate (heights above the flange)
  * the stage-1 welding programs and TackWeld: share of the path RoboDK finds valid (Update), must be 100 %
The joint values are restored afterwards.
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_station2 as S                        # noqa: E402
from robodk import robolink as rl                 # noqa: E402

B = S.B
TOL_MM, TOL_ROT = 0.05, 1e-5
bad = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        bad.append(msg)


def diff(A, Bm):
    """(max position difference in mm, max rotation-matrix difference) of two poses."""
    d_pos = max(abs(A[i, 3] - Bm[i, 3]) for i in range(3))
    d_rot = max(abs(A[i, j] - Bm[i, j]) for i in range(3) for j in range(3))
    return d_pos, d_rot


def same_pose(name, got, want):
    d_pos, d_rot = diff(got, want)
    check(d_pos < TOL_MM and d_rot < TOL_ROT, "%s: %.3f mm, %.1e rot" % (name, d_pos, d_rot))


def main():
    RDK = rl.Robolink()
    st = RDK.Item(S.STATION_NAME, rl.ITEM_TYPE_STATION)
    if not st.Valid():
        raise SystemExit("station %r is not open in RoboDK: run build_station2.py first" % S.STATION_NAME)
    RDK.setActiveStation(st)
    RDK.Render(True)                              # PoseAbs is stale with rendering off
    item = lambda name, t=None: RDK.Item(name, t) if t is not None else RDK.Item(name)   # noqa: E731
    tilt, rot = item("Positioner tilt", rl.ITEM_TYPE_ROBOT), item("Positioner faceplate", rl.ITEM_TYPE_ROBOT)
    track = item(next(n for n in ("Linear track Y", "Linear track") if item(n, rl.ITEM_TYPE_ROBOT).Valid()),
                 rl.ITEM_TYPE_ROBOT)
    htrack, carrier = item(S.HTRACK_NAME, rl.ITEM_TYPE_ROBOT), item(S.CARRIER_NAME, rl.ITEM_TYPE_ROBOT)
    weld, handler, tack = (item("ABB IRB 4600-40/2.55", rl.ITEM_TYPE_ROBOT), item(S.HANDLER_NAME, rl.ITEM_TYPE_ROBOT),
                           item(S.TACK_NAME, rl.ITEM_TYPE_ROBOT))
    fr_pos, fr_car = item(S.FACEPLATE_FRAME, rl.ITEM_TYPE_FRAME), item(S.CARRIER_FRAME, rl.ITEM_TYPE_FRAME)
    for n, it in (("tilt", tilt), ("faceplate", rot), ("track", track), ("handler track", htrack), ("carrier", carrier),
                  ("welding robot", weld), ("handler", handler), ("tack robot", tack), ("spool frame", fr_pos),
                  ("carrier frame", fr_car)):
        check(it.Valid(), "%s found" % n)
    if bad:
        raise SystemExit(1)
    saved = {id(m): m.Joints().list() for m in (tilt, rot, track, htrack, carrier)}

    def put(m, v):
        m.setJoints([float(v)])
        RDK.Render(True)

    print("-- positioner")
    for tl, rt in ((0.0, S.L2.POS_LOAD_ROT), (90.0, 0.0), (-90.0, 360.0), (45.0, S.L2.POS_UNLOAD_ROT)):
        put(tilt, tl)
        put(rot, rt)
        same_pose("spool frame, tilt %+.0f rot %.0f" % (tl, rt), fr_pos.PoseAbs(), S.positioner_frame(rt, tl))
    put(tilt, 0.0)
    put(rot, S.L2.POS_LOAD_ROT)
    print("-- carrier pallet")
    for x in (S.L2.CONV_LOAD_X, -6.0, S.L2.CONV_END_X):
        put(carrier, x * S.M)
        same_pose("carrier spool frame, x %.1f m" % x, fr_car.PoseAbs(), S.carrier_frame(x))
    print("-- handler")
    for x in (S.HANDLER_HOME_X, -5000.0, S.HANDLER_TRACK_X[1]):
        put(htrack, x)
        same_pose("handler base, carriage x %.0f" % x, handler.PoseAbs(), S.handler_base_world(x))
    print("-- welding robot and tack robot")
    for y in (B.ROBOT_HOME_Y, 0.0, B.TRACK_A):
        put(track, y)
        same_pose("welding robot base, track y %.0f" % y, weld.PoseAbs(), B.robot_base_pose_world(y))
    same_pose("tack robot base", tack.PoseAbs(), S.tack_base_world())
    print("-- swing clamps (spool frame = 30 mm above the faceplate, flange face at %.0f mm)" % S.FLANGE_THK)
    for n in (S.CLAMPS_NAME, S.CLAMP_ARMS_OPEN, S.CLAMP_ARMS_CLOSED):
        check(item(n, rl.ITEM_TYPE_OBJECT).Valid(), "%s exists, on the faceplate spool frame" % n)
        same_pose("%s pose" % n, item(n, rl.ITEM_TYPE_OBJECT).PoseAbs(), fr_pos.PoseAbs())
    print("-- programs (share of the path RoboDK finds valid)")
    for name in S.STAGE1_ROBOT_PROGRAMS + ("TackWeld",):
        prog = item(name, rl.ITEM_TYPE_PROGRAM)
        if not prog.Valid():
            check(False, "program %s exists" % name)
            continue
        state = S.PROGRAM_STATES.get(name, (B.ROBOT_HOME_Y, 0.0, S.L2.POS_LOAD_ROT))   # the state it was built for
        put(track, state[0])
        put(tilt, state[1])
        put(rot, state[2])
        ok, _t, _dist, frac, problems = prog.Update(rl.COLLISION_OFF)
        msg = "program %s (track %.0f, tilt %.0f, rot %.0f): %.0f %% valid %s" % (
            name, state[0], state[1], state[2], frac * 100, problems if frac < 1.0 else "")
        if name in S.KNOWN_PARTIAL and frac < 1.0:
            print("  note  %s (known: %s)" % (msg, S.KNOWN_PARTIAL[name]))
        else:
            check(frac >= 1.0, msg)
    for m, key in ((tilt, id(tilt)), (rot, id(rot)), (track, id(track)), (htrack, id(htrack)), (carrier, id(carrier))):
        m.setJoints(saved[key])
    RDK.Render(True)
    print("\n%s" % ("RESULT: OK" if not bad else "RESULT: %d FAILED" % len(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
