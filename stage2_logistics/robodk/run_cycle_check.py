#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the whole DemoCycle2 in a RUNNING RoboDK (simulation) and check where everything ended up.

    python run_cycle_check.py [speed]       speed = simulation speed ratio (default 300; the cycle is minutes at 1x)

The station "PipeSpool_FullCycle" must be open (build_station2.py).  After the run:
  * the spool (flange, elbow, pipe) is in the storage bay layout2.STORAGE_TARGET, flange 2 on the assembly station;
  * the swing clamps are open, the gripper is open, the handler / tack / carrier / positioner are home;
  * the number of instructions RoboDK could not execute (RunCode returns the valid ones).
The program is stopped after MAX_MINUTES.  Cycle2_Home runs at the end, so the station is left at its start state.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_station2 as S                        # noqa: E402
from robodk import robolink as rl                 # noqa: E402

MAX_MINUTES = 25
bad = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg, flush=True)
    if not cond:
        bad.append(msg)


def main():
    speed = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
    RDK = rl.Robolink()
    st = RDK.Item(S.STATION_NAME, rl.ITEM_TYPE_STATION)
    if not st.Valid():
        raise SystemExit("station %r is not open in RoboDK: run build_station2.py first" % S.STATION_NAME)
    RDK.setActiveStation(st)
    RDK.Render(True)                              # the macros read PoseAbs, which is stale with rendering off
    RDK.setRunMode(rl.RUNMODE_SIMULATE)
    RDK.setSimulationSpeed(speed)
    prog = RDK.Item("DemoCycle2", rl.ITEM_TYPE_PROGRAM)
    if not prog.Valid():
        raise SystemExit("program DemoCycle2 not found")
    t0 = time.time()
    n_ok = prog.RunCode()
    print("DemoCycle2 started at speed x%.0f; RoboDK reports %s executable instructions" % (speed, n_ok), flush=True)
    last = 0
    while prog.Busy():
        time.sleep(1.0)
        el = time.time() - t0
        if el - last >= 30:
            last = el
            print("   ... running, %.0f s" % el, flush=True)
        if el > MAX_MINUTES * 60:
            prog.Stop()
            check(False, "DemoCycle2 did not finish in %d minutes (stopped)" % MAX_MINUTES)
            break
    print("DemoCycle2 finished after %.0f s" % (time.time() - t0), flush=True)
    RDK.Render(True)

    P = S.PART_NAMES
    item = lambda n, t=rl.ITEM_TYPE_OBJECT: RDK.Item(n, t)            # noqa: E731

    def pose_ok(name, want, tol_mm=0.5, tol_rot=1e-3):
        got = item(name).PoseAbs()
        d_pos = max(abs(got[i, 3] - want[i, 3]) for i in range(3))
        d_rot = max(abs(got[i, j] - want[i, j]) for i in range(3) for j in range(3))
        check(d_pos < tol_mm and d_rot < tol_rot, "%s: %.2f mm, %.1e rot from the target" % (name, d_pos, d_rot))

    bay = S.storage_frame(*S.L2.STORAGE_TARGET)
    for k in ("flange", "elbow", "pipe"):
        pose_ok(P[k], bay)
    pose_ok(P["flange2"], S.station_frame())
    check(not item(S.CLAMP_ARMS_CLOSED).Visible() and item(S.CLAMP_ARMS_OPEN).Visible(),
          "swing clamps open (arms closed hidden, arms open shown)")
    check(RDK.getParam("GRIPPER") == "OPEN" and RDK.getParam("POS_CLAMPS") == "OPEN",
          "gripper %r, clamps %r" % (RDK.getParam("GRIPPER"), RDK.getParam("POS_CLAMPS")))
    htrack = item(S.HTRACK_NAME, rl.ITEM_TYPE_ROBOT)
    check(abs(htrack.Joints().list()[0] - S.HANDLER_HOME_X) < 1.0, "handler carriage home (x %.0f)" % htrack.Joints().list()[0])
    rot = item("Positioner faceplate", rl.ITEM_TYPE_ROBOT)
    print("   faceplate at %.1f deg, carrier at x %.0f" % (rot.Joints().list()[0],
                                                          item(S.CARRIER_NAME, rl.ITEM_TYPE_ROBOT).Joints().list()[0]))
    home = RDK.Item("Cycle2_Home", rl.ITEM_TYPE_PROGRAM)
    if home.Valid():
        home.RunCode()
        while home.Busy():
            time.sleep(0.5)
        print("station back at its start state (Cycle2_Home)", flush=True)
    print("\n%s" % ("RESULT: OK" if not bad else "RESULT: %d FAILED" % len(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
