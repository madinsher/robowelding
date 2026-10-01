"""Unit test for stage2_logistics/robodk/build_station2.py without RoboDK.

  1. py_compile of the builder and of every generated macro;
  2. unit conversions (m <-> mm, rad <-> deg, pose <-> xyz+rotation vector) round-trip;
  3. the frames rebuilt with robomath equal plan2's numpy functions (station_frame, carrier_frame, grasp_frame,
     spool_on_positioner, kin.planar_frame of the kit / buffer / storage frames, handler_base, tack_base, GRIP_T,
     TORCH_T, TRAVEL_Z, SWING_X) - needs numpy + bpy like the other stage-2 tests, skipped otherwise;
  4. the plan snapshot plan2_robodk.json: the planned TCP poses at the choreography events coincide with the
     builder's waypoints (a stale snapshot after a layout2 change fails here -> run build_station2.py --export-plan);
  5. every Cartesian target lies inside the nominal reach sphere of its robot (IRB 6700-150/3.20: 3.2 m, IRB 1600:
     1.45 m, wrist centre from the J1 axis at shoulder height) and every handler target's carriage x is inside the
     track travel;
  6. a full build() against a mock Robolink (the stage-1 mock of demo_video/tests/t_robodk_math.py, extended with
     absolute poses through mechanisms): items, mechanisms, mounting, programs, macros, DemoCycle2 order;
  7. a simulation of DemoCycle2: the programs are walked through their calls, mechanism programs move the carriage /
     carrier / positioner, the GripperClose / GripperOpen / ResetParts macros are EXECUTED against a small part-world
     simulator; the parts must end where the video puts them (spool in the storage bay, flange 2 on the station);
  8. the same build in a RoboDK-like python without bpy / numpy (subprocess), a fallback build without the snapshot
     and an idempotent re-build.

Run:  python3 stage2_logistics/tests/t_robodk2.py        (needs the `robodk` package: pip install robodk)
"""
import hashlib
import importlib.util
import math
import os
import py_compile
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE2 = os.path.dirname(HERE)
REPO = os.path.dirname(STAGE2)
DEMO = os.path.join(REPO, "demo_video")
RDK_DIR = os.path.join(STAGE2, "robodk")
for p in (RDK_DIR, STAGE2):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from robodk import robolink as rl
    from robodk import robomath as rm
    from robodk.robomath import Mat, eye, invH, rotz, transl
except ImportError:
    print("t_robodk2: the `robodk` python package is required (pip install robodk)")
    sys.exit(2)

import build_station2 as S2  # noqa: E402

B = S2.B
_spec = importlib.util.spec_from_file_location("t_robodk_math", os.path.join(DEMO, "tests", "t_robodk_math.py"))
TM = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TM)

try:                                   # plan2 needs numpy + bpy (the stage-2 environment)
    import plan2
    import kin as K
except Exception as _exc:              # pragma: no cover
    plan2 = None
    print("plan2 not importable (%s): plan2 consistency checks skipped" % _exc)

FAILS = []


def close(a, b, tol=1e-6):
    return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))


def same_pose(A, Bm, tol_mm=1e-6, tol_rad=1e-9):
    return rm.norm(rm.subs3(A.Pos(), Bm.Pos())) <= tol_mm and rm.pose_angle_between(A, Bm) <= tol_rad + 1e-7


def np_mm(T):
    """numpy pose (m) -> Mat (mm)."""
    return S2.mat_mm(T.tolist())


# ================================================================ 1-2. compile + conversions
def check_compile(path):
    """py_compile into a temporary .pyc (nothing written next to the sources)."""
    fd, tmp = tempfile.mkstemp(suffix=".pyc")
    os.close(fd)
    try:
        py_compile.compile(path, cfile=tmp, doraise=True)
    finally:
        os.remove(tmp)


def test_compile():
    check_compile(os.path.join(RDK_DIR, "build_station2.py"))
    check_compile(os.path.abspath(__file__))


def test_conversions():
    T = transl(1234.5, -678.9, 2222.0) * B.roty(0.7) * B.rotx(-1.1) * rotz(2.9)
    back = S2.mat_mm(S2.mat_m(T))
    assert same_pose(back, T, 1e-9, 1e-12)
    assert close(S2.mat_m(T)[0][3:], [1.2345]) and close(S2.mat_m(T)[2][3:], [2.222])
    q = [0.1, -1.2, 2.3, -3.1, 1.57, 6.0]
    assert close(S2.rad(S2.deg(q)), q, 1e-12) and close(S2.deg([math.pi]), [180.0], 1e-12)
    wps, _ = S2.handler_plan(S2.load_plan())
    twps, _ = S2.tack_plan(S2.load_plan())
    for name, wp in list(wps.items()) + list(twps.items()):
        P = wp["pose"]
        assert P.isHomogeneous(), name
        assert same_pose(rm.TxyzRxyz_2_Pose(rm.Pose_2_TxyzRxyz(P)), P, 1e-6, 1e-9), name     # xyz + rotation vector
        assert same_pose(S2.mat_mm(S2.mat_m(P)), P, 1e-9, 1e-12), name


# ================================================================ 3. frames = plan2's functions
def test_frames_vs_plan2():
    if plan2 is None:
        print("   (skipped)")
        return
    L2 = S2.L2
    pairs = [("station", S2.station_frame(), plan2.station_frame()),
             ("positioner load", S2.positioner_frame(L2.POS_LOAD_ROT), plan2.spool_on_positioner(0.0, L2.POS_LOAD_ROT)),
             ("positioner unload", S2.positioner_frame(L2.POS_UNLOAD_ROT), plan2.spool_on_positioner(0.0, L2.POS_UNLOAD_ROT)),
             ("positioner weld", S2.positioner_frame(-20.0, 90.0), plan2.spool_on_positioner(90.0, -20.0)),
             ("tack base", S2.tack_base_world(), plan2.tack_base()),
             ("gripper tcp", S2.gripper_tcp(), plan2.GRIP_T),
             ("torch tcp", B.tcp_pose(), plan2.TORCH_T)]
    for x in (L2.CONV_LOAD_X, L2.CONV_QC_X, L2.CONV_END_X, 0.0):
        pairs.append(("carrier %.2f" % x, S2.carrier_frame(x), plan2.carrier_frame(x)))
    for name in L2.GRASP:
        pairs.append(("grasp " + name, S2.grasp_frame(name), plan2.grasp_frame(name)))
    for i, (o, xd) in enumerate(L2.KIT_FLANGES):
        pairs.append(("kit flange %d" % i, S2.kit_flange_frame(i), K.planar_frame(o, xd)))
    for i, (o, xd) in enumerate(L2.KIT_ELBOWS):
        pairs.append(("kit elbow %d" % i, S2.kit_elbow_frame(i), K.planar_frame(o, xd)))
    for i in range(len(L2.PIPE_BUFFER_X)):
        pairs.append(("buffer %d" % i, S2.buffer_pipe_frame(i), K.planar_frame(*L2.pipe_buffer_frame(i))))
    for t in (0, 1):
        for b in range(len(L2.STORAGE["bay_y"][t])):
            pairs.append(("storage %d %d" % (t, b), S2.storage_frame(t, b), K.planar_frame(*L2.storage_frame(t, b))))
    for x in L2.HANDLER_TRACK_X + (L2.HANDLER_HOME_X,):
        pairs.append(("handler base %.2f" % x, S2.handler_base_world(x * 1000.0), plan2.handler_base(x)))
    for name, mine, ref in pairs:
        assert same_pose(mine, np_mm(ref), 1e-6, 1e-9), (name, mine, ref)
    assert abs(S2.TRAVEL_Z - plan2.TRAVEL_Z * 1000) < 1e-9 and abs(S2.SWING_X - plan2.SWING_X * 1000) < 1e-9
    # handler track base: mechanism Z = world +X, so the joint value is the world x of the J1 axis
    assert close(S2.handler_track_base().VZ(), [1.0, 0.0, 0.0]) and close(S2.carrier_track_base().VZ(), [1.0, 0.0, 0.0])
    print("   %d frames equal plan2" % len(pairs))


# ================================================================ 4. snapshot vs builder waypoints
def test_snapshot():
    plan = S2.load_plan()
    assert plan, "plan2_robodk.json missing: run python3 stage2_logistics/robodk/build_station2.py --export-plan"
    for key in ("events", "handler", "tack", "n_frames", "weld_offset", "plan_hash"):
        assert key in plan, key
    assert abs(plan["travel_z"] * 1000 - S2.TRAVEL_Z) < 1e-9 and abs(plan["swing_x"] * 1000 - S2.SWING_X) < 1e-9
    wps, _ = S2.handler_plan(plan)
    n = 0
    for name, wp in wps.items():
        ev = plan["handler"].get(wp["event"])
        assert ev is not None, (name, wp["event"])
        assert close(wp["seed"], S2.deg(ev["q"]), 1e-9), name
        if wp["exact"]:
            T = S2.mat_mm(ev["tcp"])
            d = rm.norm(rm.subs3(T.Pos(), wp["pose"].Pos()))
            a = math.degrees(rm.pose_angle_between(T, wp["pose"]))
            assert d < 5.0 and a < 1.0, (name, wp["event"], round(d, 2), round(a, 3))
            assert abs(ev["x"] * 1000 - wp["x"]) < 1.0, (name, ev["x"], wp["x"])        # same carriage position
            n += 1
    twps, _ = S2.tack_plan(plan)
    tk = {t["key"]: t for t in plan["tack"]}
    for name, wp in twps.items():
        if wp["exact"]:
            assert same_pose(wp["pose"], S2.mat_mm(tk[name.split("_")[1]]["tcp"]), 1e-6, 1e-9), name
            n += 1
    assert n >= 18, n
    if plan2 is not None:
        cur = plan2._hash()
        if cur != plan.get("plan_hash"):
            print("   NOTE: plan2 changed since the snapshot (%s -> %s); the layout-dependent poses still match. "
                  "Refresh with: python3 stage2_logistics/robodk/build_station2.py --export-plan"
                  % (plan.get("plan_hash"), cur))
    missing = [t["key"] for t in S2.tack_nominal() if t["key"] not in tk]
    print("   %d exact waypoints match the planned TCP poses (plan %s, %d tacks planned%s)"
          % (n, plan["plan_hash"], len(tk), (", nominal: " + ",".join(missing)) if missing else ""))


# ================================================================ 5. reach + track travel
def test_reach_and_track():
    plan = S2.load_plan()
    wps, _ = S2.handler_plan(plan)
    twps, _ = S2.tack_plan(plan)
    lo, hi = S2.HANDLER_TRACK_X
    worst = {"handler": (0.0, ""), "tack": (0.0, "")}
    for who, table in (("handler", wps), ("tack", twps)):
        for name, wp in table.items():
            if who == "handler":
                assert lo - 1e-6 <= wp["x"] <= hi + 1e-6, (name, wp["x"])
            d, reach = S2.reach_of(who, wp["pose"], wp["x"])
            assert d <= reach, (who, name, round(d), reach)
            if d > worst[who][0]:
                worst[who] = (d, name)
    # the loading entry: the carriage runs from SWING_X to the positioner station with the arm frozen, the TCP moves
    # straight along +X from the pre-entry pose (outside the fence) to Load_Entry
    pre, ent = wps["Load_PreEntry"], wps["Load_Entry"]
    dx = ent["x"] - pre["x"]
    assert same_pose(transl(dx, 0, 0) * pre["pose"], ent["pose"], 1e-6, 1e-9)
    assert pre["pose"].Pos()[0] < S2.L2.CELL_FENCE_X0 * 1000 and pre["x"] == S2.SWING_X
    print("   max wrist-centre distance: handler %.0f mm (%s) of %.0f, tack %.0f mm (%s) of %.0f"
          % (worst["handler"][0], worst["handler"][1], S2.ROBOT_SPECS["handler"]["reach"],
             worst["tack"][0], worst["tack"][1], S2.ROBOT_SPECS["tack"]["reach"]))


# ================================================================ 6. mock Robolink
class MockItem2(TM.MockItem):
    valid = True
    mtype = None

    def Valid(self, check_deleted=False):
        return self.valid

    def PoseAbs(self):
        return self.link.abs_pose(self)

    def SolveIK(self, pose, joints_approx=None, tool=None, reference=None):
        assert pose.isHomogeneous() and tool is not None and reference is not None
        self.link.ik_calls.append((self.name, joints_approx))
        if joints_approx is None:
            return Mat([0.0, -10.0, 20.0, 0.0, 60.0, 0.0])
        return Mat([float(v) for v in list(joints_approx)[:6]])      # echo the seed

    def Childs(self):
        return [it for it in self.link.items if it.parent is self]


class MockRobolink2(TM.MockRobolink):
    def __init__(self, lib):
        TM.MockRobolink.__init__(self, lib)
        self.ik_calls = []
        self.station = MockItem2(self, "station", rl.ITEM_TYPE_STATION)
        self.invalid = MockItem2(self, "", 0)
        self.invalid.valid = False

    def _new(self, name, itype, ndof=0):
        it = MockItem2(self, name, itype, ndof)
        self.items.append(it)
        self.names[name] = it
        return it

    def Item(self, name, itemtype=None):
        it = self.names.get(name)
        if it is not None and (itemtype is None or it.itype == itemtype):
            return it
        return self.invalid

    def BuildMechanism(self, mtype, objs, params, jb, jh, senses, lo, hi, base=None, tool=None, name="", robot=None):
        it = TM.MockRobolink.BuildMechanism(self, mtype, objs, params, jb, jh, senses, lo, hi, base, tool, name, robot)
        it.mtype = mtype
        it.joints = list(jh)
        return it

    # absolute poses: a robot / mechanism sits at its parent frame (or at its BuildMechanism base); children of a
    # 1-axis mechanism ride on its flange = base * Tz(j) (1T) or base * Rz(j) (1R)
    def base_pose(self, robot):
        if robot.parent is not None and robot.parent.itype == rl.ITEM_TYPE_FRAME:
            return self.abs_pose(robot.parent)
        return robot.pose

    def attach_pose(self, parent):
        if parent is None or parent.itype == rl.ITEM_TYPE_STATION:
            return eye(4)
        if parent.itype == rl.ITEM_TYPE_ROBOT:
            base = self.base_pose(parent)
            if parent.mtype == rl.MAKE_ROBOT_1T:
                return base * transl(0, 0, parent.joints[0])
            if parent.mtype == rl.MAKE_ROBOT_1R:
                return base * rotz(parent.joints[0] * math.pi / 180.0)
            return base
        return self.abs_pose(parent)

    def abs_pose(self, item):
        if item.itype == rl.ITEM_TYPE_ROBOT:
            return self.base_pose(item)
        return self.attach_pose(item.parent) * item.pose


def make_lib():
    lib = tempfile.mkdtemp(prefix="rdklib2_")
    os.makedirs(os.path.join(lib, "ABB"))
    for f in ("ABB-IRB-4600-40-2.55.robot", "ABB-IRB-6700-150-3.20.robot", "ABB-IRB-6700-200-2.60.robot",
              "ABB-IRB-1600-6-1.45.robot", "ABB-IRB-1600-10-1.2.robot"):
        open(os.path.join(lib, "ABB", f), "w").close()
    return lib


def dir_digest(path):
    h = hashlib.sha1()
    for root, _dirs, files in sorted(os.walk(path)):
        for f in sorted(files):
            with open(os.path.join(root, f), "rb") as fh:
                h.update(f.encode() + fh.read())
    return h.hexdigest()


EXPECTED_STAGE2_PROGRAMS = [
    "Handler_Home", "KitFlange", "KitElbow", "KitPipe", "TackWeld", "Tack_Home", "PickTackedSpool", "LoadPositioner",
    "Handler_Park", "UnloadPositioner", "ToConveyor", "Conveyor_QC", "KitFlange2", "StoreSpool", "Cycle2_Home",
    "DemoCycle2", "Conv_ToLoad", "Conv_ToQC", "Conv_ToEnd", "Pos_RotLoad", "Pos_RotUnload"]
EXPECTED_STAGE1_PROGRAMS = ["DemoCycle", "Cell_Home", "Robot_Home", "Robot_Transit", "Robot_WeldA", "Robot_WeldB_S4",
                            "Track_Home", "Pos_TiltUp", "Pos_Index180", "Pos_RotStart"]
EXPECTED_MACROS = ["ArcOn", "ArcOff", "GripperClose", "GripperOpen", "ClampsClose", "ClampsOpen", "TackArcOn", "TackArcOff",
                   "ResetParts", "QC_MarkScan"]
DEMO2_ORDER = ["Cycle2_Home", "KitFlange", "KitElbow", "KitPipe", "TackWeld", "PickTackedSpool", "LoadPositioner",
               "Handler_Park", "DemoCycle", "Pos_RotUnload", "UnloadPositioner", "ToConveyor", "Conveyor_QC",
               "KitFlange2", "StoreSpool", "Handler_Home"]


def build_mock(plan=None):
    RDK = MockRobolink2(make_lib())
    S2.SAVE_RDK = False
    cell = S2.build(RDK, plan)
    return RDK, cell


def calls_of(prog):
    return [c for k, c in prog.log if k == "call"]


def test_mock_build():
    demo_gen = os.path.join(DEMO, "robodk", "generated")
    before = dir_digest(demo_gen) if os.path.isdir(demo_gen) else None
    gen_dir_before = B.GEN_DIR
    RDK, cell = build_mock()
    assert B.GEN_DIR == gen_dir_before, "stage-1 module state not restored"
    if before is not None:
        assert dir_digest(demo_gen) == before, "demo_video/robodk/generated was modified"
    # ---- robots, tools, mechanisms
    assert cell.handler.name == S2.HANDLER_NAME and cell.tack.name == S2.TACK_NAME and cell.robot.name == "ABB IRB 4600-40/2.55"
    assert cell.gripper.itype == rl.ITEM_TYPE_TOOL and cell.gripper.parent is cell.handler and cell.gripper.name == S2.GRIPPER_NAME
    assert cell.tack_torch.itype == rl.ITEM_TYPE_TOOL and cell.tack_torch.parent is cell.tack
    assert any(k == "setPoseTool" and same_pose(a[0], S2.gripper_tcp()) for k, a in cell.gripper.log)
    assert any(k == "setPoseTool" and same_pose(a[0], B.tcp_pose()) for k, a in cell.tack_torch.log)
    assert cell.htrack.mtype == rl.MAKE_ROBOT_1T and close(cell.htrack.limits[0] + cell.htrack.limits[1], list(S2.HANDLER_TRACK_X))
    assert cell.carrier.mtype == rl.MAKE_ROBOT_1T and close(cell.carrier.limits[0] + cell.carrier.limits[1],
                                                           [S2.L2.CONV_END_X * 1000, S2.L2.CONV_LOAD_X * 1000])
    # ---- mounting: handler base rides the carriage, spool frames ride the faceplate / carrier
    for x in list(S2.HANDLER_TRACK_X) + [S2.HANDLER_HOME_X]:
        cell.htrack.joints = [x]
        assert same_pose(cell.handler.PoseAbs(), S2.handler_base_world(x), 1e-6, 1e-9), x
        assert close(cell.handler.PoseAbs().Pos(), [x, S2.HANDLER_TRACK_Y, S2.HANDLER_TOP_Z], 1e-6)
    cell.htrack.joints = [S2.HANDLER_HOME_X]
    assert same_pose(cell.tack.PoseAbs(), S2.tack_base_world(), 1e-6, 1e-9)
    for x in (S2.L2.CONV_LOAD_X, S2.L2.CONV_QC_X, S2.L2.CONV_END_X):
        cell.carrier.joints = [x * 1000]
        assert same_pose(cell.carrier_frame.PoseAbs(), S2.carrier_frame(x), 1e-6, 1e-9), x
    cell.carrier.joints = [S2.L2.CONV_LOAD_X * 1000]
    for tilt, rot in ((0.0, S2.L2.POS_LOAD_ROT), (90.0, -20.0), (-90.0, 360.0), (0.0, S2.L2.POS_UNLOAD_ROT)):
        cell.tilt.joints, cell.rot.joints = [tilt], [rot]
        assert same_pose(cell.faceplate_frame.PoseAbs(), S2.positioner_frame(rot, tilt), 1e-6, 1e-9), (tilt, rot)
    cell.tilt.joints, cell.rot.joints = [0.0], [S2.L2.POS_LOAD_ROT]
    # ---- objects
    names = set(RDK.names)
    movable, decor = S2.part_placements()
    objs = list(S2.logistics_static_objects()) + ["Handler track bed", "Handler track carriage", "Output conveyor",
                                                  "Carrier pallet", S2.CLAMPS_NAME, S2.CLAMP_ARMS_OPEN,
                                                  S2.CLAMP_ARMS_CLOSED] + list(movable) + list(decor)
    for n in objs:
        assert n in names and RDK.names[n].itype == rl.ITEM_TYPE_OBJECT, n
    for n, (_kind, F) in movable.items():
        it = RDK.names[n]
        assert it.parent is cell.logistics and same_pose(it.pose, F), n
    # swing clamps of the flange: bodies and both arm states ride on the faceplate; the arms start open
    assert cell.clamps.parent is cell.faceplate_frame
    assert all(a.parent is cell.faceplate_frame for a in cell.clamp_arms.values())
    assert ("setVisible", (True,)) in cell.clamp_arms["open"].log
    assert ("setVisible", (False,)) in cell.clamp_arms["closed"].log
    bodies, arms_open, arms_closed = S2.clamp_shapes()
    pts = lambda shapes: [p for tris, _ in shapes for tri in tris for p in tri]              # noqa: E731
    rad = lambda p: math.hypot(p[0], p[1])                                                   # noqa: E731
    r_flange = S2.L1.FLANGE_OD * 1000 / 2
    assert min(rad(p) for p in pts(arms_open)) > r_flange + 20, "an open arm must be clear of the flange being lowered"
    assert min(rad(p) for p in pts(bodies)) > 215 + 10, "the clamp bodies stand outside the fixture ring"
    assert min(rad(p) for p in pts(arms_closed)) < r_flange - 15, "a closed arm reaches over the flange rim"
    assert min(p[2] for p in pts(arms_closed) if rad(p) < r_flange) >= S2.FLANGE_THK - 1e-6, "a closed arm lies on the flange"
    assert max(p[2] for p in pts(arms_closed)) <= S2.FLANGE_THK + 20, "clamped arms are no higher than a bolt head"
    assert max(p[2] for p in pts(bodies)) < S2.FLANGE_THK, "the clamp bodies stay below the flange face"
    assert S2.B.FLANGE_BOLTS is True, "build() must restore the stage-1 flag (the stage-1 station keeps its bolts)"
    for w in ("A", "B"):
        assert cell.curves[w].parent is cell.faceplate_frame
    # ---- programs and macros
    for n in EXPECTED_STAGE2_PROGRAMS + EXPECTED_STAGE1_PROGRAMS:
        assert n in cell.programs, n
    shots = [s["name"] for s in S2.stage2_shots()]
    assert len(shots) >= 10, shots
    expected_macros = set(EXPECTED_MACROS) | {"View_" + s[0] for s in B.SHOTS} | {"View2_" + s for s in shots}
    assert set(cell.macros) == expected_macros, set(cell.macros) ^ expected_macros
    for n in cell.macros:
        path = os.path.join(S2.GEN_DIR, n + ".py")
        assert os.path.exists(path), path
        check_compile(path)
    with open(os.path.join(S2.GEN_DIR, "ArcOn.py"), encoding="utf-8") as fh:
        assert S2.PART_NAMES["elbow"] in fh.read()
    for f in ("gripper.stl", "torch.stl"):
        assert os.path.getsize(os.path.join(S2.GEN_DIR, f)) > 1000, f
    for pname, prog in cell.programs.items():
        for c in calls_of(prog):
            assert c in cell.programs or c in cell.macros, "%s calls unknown %s" % (pname, c)
    demo = calls_of(cell.programs["DemoCycle2"])
    idx = [demo.index(c) for c in DEMO2_ORDER]
    assert idx == sorted(idx), demo
    assert demo[0] == "View2_S2_01_wide"
    home = calls_of(cell.programs["Cycle2_Home"])
    assert home == S2.CYCLE2_HOME, home
    # stage-2 robot programs belong to the right robots
    for n in ("KitFlange", "LoadPositioner", "StoreSpool", "DemoCycle2", "Cycle2_Home"):
        assert cell.programs[n].robot is cell.handler, n
    assert cell.programs["TackWeld"].robot is cell.tack and cell.programs["Conveyor_QC"].robot is cell.carrier
    assert cell.programs["Pos_RotLoad"].robot is cell.rot and cell.targets["Pos_RotLoad_T"].joints == [S2.L2.POS_LOAD_ROT]
    assert cell.targets["Pos_RotUnload_T"].joints == [S2.L2.POS_UNLOAD_ROT]
    for n, x in (("Conv_ToLoad", S2.L2.CONV_LOAD_X), ("Conv_ToQC", S2.L2.CONV_QC_X), ("Conv_ToEnd", S2.L2.CONV_END_X)):
        assert cell.programs[n].robot is cell.carrier and close(cell.targets[n + "_T"].joints, [x * 1000]), n
    qc = calls_of(cell.programs["Conveyor_QC"])
    assert qc.index("Conv_ToQC") < qc.index("QC_MarkScan") < qc.index("Conv_ToEnd"), qc
    # tack welding: 6 tacks = approach (MoveJ back), MoveL in, short arc, MoveL back
    tw = cell.programs["TackWeld"].log
    n_tacks = len(S2.L2.TACKS_A) + len(S2.L2.TACKS_B)
    assert calls_of(cell.programs["TackWeld"]).count("TackArcOn") == n_tacks == 6
    assert sum(1 for k, _ in tw if k == "MoveL") == 2 * n_tacks and sum(1 for k, _ in tw if k == "MoveJ") == n_tacks + 2
    assert sum(1 for k, a in tw if k == "Pause" and a == (S2.TACK_ARC_MS,)) == n_tacks
    # load / unload: clamps, muting, gripper in the right order (clamp before the gripper lets go, and back)
    lp = calls_of(cell.programs["LoadPositioner"])
    assert lp.index("HTrack_Swing") < lp.index("HTrack_Positioner") < lp.index("ClampsClose") < lp.index("GripperOpen")
    up = calls_of(cell.programs["UnloadPositioner"])
    assert up.index("HTrack_Positioner") < up.index("GripperClose") < up.index("ClampsOpen")
    muting = [a for k, a in cell.programs["LoadPositioner"].log if k == "setDO" and a[0] == "LightCurtainMuting"]
    assert muting == [("LightCurtainMuting", 1), ("LightCurtainMuting", 0)]
    # every stage-2 Cartesian target: pose = waypoint, joints = IK near the plan seed (the mock IK echoes the seed)
    plan = S2.load_plan()
    wps, _ = S2.handler_plan(plan)
    twps, _ = S2.tack_plan(plan)
    assert set(cell.target_info) == set(wps) | set(twps)
    for n, info in cell.target_info.items():
        wp = wps.get(n) or twps[n]
        t = cell.targets[n]
        assert t.robot is (cell.handler if n in wps else cell.tack), n
        assert same_pose(t.pose, wp["pose"]) and close(t.joints, wp["seed"], 1e-9), n
        assert ("setAsCartesianTarget", ()) in t.log, n
    assert cell.problems == [], cell.problems
    assert cell.targets["HandlerHome"].joints == S2.HANDLER_Q_HOME and cell.targets["TackHome"].joints == S2.TACK_Q_HOME
    # start state
    assert cell.htrack.joints == [S2.HANDLER_HOME_X] and cell.rot.joints == [S2.L2.POS_LOAD_ROT] and cell.tilt.joints == [0.0]
    assert cell.carrier.joints == [S2.L2.CONV_LOAD_X * 1000]
    print("   mock build: %d items, %d programs, %d targets, %d macros"
          % (len(RDK.items), len(cell.programs), len(cell.targets), len(cell.macros)))
    return RDK, cell


# ================================================================ 7. DemoCycle2 simulation with the real macros
class World:
    """Tiny part-world simulator the generated macros run against (robodk.robolink replaced by a fake module)."""

    def __init__(self, cell):
        self.cell = cell
        self.x = S2.HANDLER_HOME_X          # handler carriage
        self.tcp = None                     # handler gripper TCP (world)
        self.carrier_x = S2.L2.CONV_LOAD_X * 1000
        self.tilt, self.rot = 0.0, S2.L2.POS_LOAD_ROT
        self.params, self.messages, self.visible = {}, [], {}
        self.items = {}
        for name in (S2.LOGISTICS_FRAME, S2.FACEPLATE_FRAME, S2.CARRIER_FRAME):
            self.add(name, rl.ITEM_TYPE_FRAME)
        self.add(S2.GRIPPER_NAME, rl.ITEM_TYPE_TOOL)
        self.add(S2.HANDLER_NAME, rl.ITEM_TYPE_ROBOT)
        for name in (S2.CLAMPS_NAME, S2.CLAMP_ARMS_OPEN, S2.CLAMP_ARMS_CLOSED):
            self.add(name, rl.ITEM_TYPE_OBJECT, self.items[S2.FACEPLATE_FRAME])
        for name in cell.part_start:
            self.add(name, rl.ITEM_TYPE_OBJECT, self.items[S2.LOGISTICS_FRAME], cell.part_start[name])

    def add(self, name, itype, parent=None, pose=None):
        self.items[name] = SimItem(self, name, itype, parent, pose)

    def frame_abs(self, item):
        if item.name == S2.FACEPLATE_FRAME:
            return S2.positioner_frame(self.rot, self.tilt)
        if item.name == S2.CARRIER_FRAME:
            return S2.carrier_frame(self.carrier_x / 1000.0)
        if item.name == S2.GRIPPER_NAME:
            return self.tcp
        if item.name == S2.HANDLER_NAME:
            return S2.handler_base_world(self.x)
        if item.name == S2.LOGISTICS_FRAME:
            return eye(4)
        return None

    def module(self):
        mod = types.ModuleType("robodk.robolink")
        for k in dir(rl):
            if k.startswith(("ITEM_TYPE_", "SPRAY_", "INSTRUCTION_")):
                setattr(mod, k, getattr(rl, k))
        world = self
        mod.Robolink = lambda *a, **kw: SimRDK(world)
        return mod

    def run_macro(self, name):
        path = os.path.join(S2.GEN_DIR, name + ".py")
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        real = sys.modules["robodk.robolink"]
        sys.modules["robodk.robolink"] = self.module()
        try:
            exec(compile(src, path, "exec"), {"__name__": "__main__"})
        finally:
            sys.modules["robodk.robolink"] = real

    def pose(self, name):
        return self.items[name].PoseAbs()

    def held(self):
        return sorted(n for n, it in self.items.items() if it.parent is self.items[S2.GRIPPER_NAME])


class SimItem:
    def __init__(self, world, name, itype, parent=None, pose=None):
        self.world, self.name, self.itype, self.parent = world, name, itype, parent
        self.pose = pose if pose is not None else eye(4)

    def Valid(self):
        return self.itype != 0

    def Name(self):
        return self.name

    def Type(self):
        return self.itype

    def Parent(self):
        return self.parent

    def Pose(self):
        return self.pose

    def setPose(self, pose):
        self.pose = pose

    def setParent(self, parent):
        self.parent = parent

    def PoseAbs(self):
        fixed = self.world.frame_abs(self)
        if fixed is not None:
            return fixed
        return (self.parent.PoseAbs() if self.parent is not None else eye(4)) * self.pose

    def setParentStatic(self, parent):
        self.pose = invH(parent.PoseAbs()) * self.PoseAbs()
        self.parent = parent

    def Childs(self):
        return [it for it in self.world.items.values() if it.parent is self]

    def setVisible(self, v, *a):
        self.world.visible[self.name] = bool(v)

    # robot / tool
    def Joints(self):
        return Mat([0.0] * 6)

    def SolveFK(self, joints, tool=None, reference=None):
        return invH(S2.handler_base_world(self.world.x)) * self.world.tcp * invH(S2.gripper_tcp())

    def PoseTool(self):
        return S2.gripper_tcp()


class SimRDK:
    def __init__(self, world):
        self.world = world

    def Item(self, name, itype=None):
        it = self.world.items.get(name)
        if it is None or (itype is not None and it.itype != itype):
            return SimItem(self.world, name, 0)
        return it

    def ShowMessage(self, msg, popup=True):
        self.world.messages.append(msg)

    def setParam(self, k, v):
        self.world.params[k] = v

    def getParam(self, k):
        return self.world.params.get(k)

    def Spray_Clear(self, *a):
        return 0


def simulate(cell, world):
    """Walk DemoCycle2 through its calls; returns the list of (event, detail) of the gripper actions."""
    mech = {id(cell.htrack): "x", id(cell.carrier): "carrier_x", id(cell.tilt): "tilt", id(cell.rot): "rot"}
    log = []
    log_clamps = world.clamp_log = []

    def run(pname, depth=0):
        assert depth < 12, pname
        prog = cell.programs[pname]
        attr = mech.get(id(prog.robot))
        for k, a in prog.log:
            if k == "call":
                if a in cell.programs:
                    run(a, depth + 1)
                elif a in ("GripperClose", "GripperOpen", "ResetParts", "ClampsClose", "ClampsOpen"):
                    world.run_macro(a)
                    if a in ("ClampsClose", "ClampsOpen"):
                        log_clamps.append((a, pname, world.visible.get(S2.CLAMP_ARMS_CLOSED),
                                           world.visible.get(S2.CLAMP_ARMS_OPEN)))
                    if a in ("GripperClose", "GripperOpen"):
                        log.append((a, pname, tuple(world.held())))
            elif k in ("MoveJ", "MoveL"):
                t = cell.targets[a]
                if attr is not None:
                    setattr(world, attr, t.joints[0])
                elif prog.robot is cell.handler and a in cell.target_info:
                    info = cell.target_info[a]
                    assert abs(info["x"] - world.x) < 1e-6, "%s: %s needs carriage x=%.0f, carriage at %.0f" % (
                        pname, a, info["x"], world.x)
                    world.tcp = info["pose"]
                elif prog.robot is cell.handler:
                    world.tcp = None                       # joint target (home): TCP not modelled, nothing held
                    assert not world.held(), (pname, a)
    run("DemoCycle2")
    return log


def test_cycle_simulation(RDK=None, cell=None):
    if cell is None:
        RDK, cell = build_mock()
    world = World(cell)
    # shuffle the parts first: ResetParts (in Cycle2_Home) must put them back
    world.items[S2.PART_NAMES["pipe"]].pose = transl(123, 456, 789)
    log = simulate(cell, world)
    assert not world.messages, world.messages
    P = S2.PART_NAMES
    spool = tuple(sorted((P["flange"], P["elbow"], P["pipe"])))
    expected = [("GripperClose", "KitFlange", (P["flange"],)), ("GripperOpen", "KitFlange", ()),
                ("GripperClose", "KitElbow", (P["elbow"],)), ("GripperOpen", "KitElbow", ()),
                ("GripperClose", "KitPipe", (P["pipe"],)), ("GripperOpen", "KitPipe", ()),
                ("GripperClose", "PickTackedSpool", spool), ("GripperOpen", "LoadPositioner", ()),
                ("GripperClose", "UnloadPositioner", spool), ("GripperOpen", "ToConveyor", ()),
                ("GripperClose", "KitFlange2", (P["flange2"],)), ("GripperOpen", "KitFlange2", ()),
                ("GripperClose", "StoreSpool", spool), ("GripperOpen", "StoreSpool", ())]
    assert log == expected, log
    # final poses: the welded spool in the storage bay, flange 2 on the assembly station, all snapped exactly
    Fb = S2.storage_frame(*S2.L2.STORAGE_TARGET)
    for n in spool:
        it = world.items[n]
        assert it.parent is world.items[S2.LOGISTICS_FRAME] and same_pose(it.PoseAbs(), Fb, 1e-6, 1e-9), n
    assert same_pose(world.pose(P["flange2"]), S2.station_frame(), 1e-6, 1e-9)
    # the clamps: closed after the spool is placed, open before it is lifted and at the end of the cycle
    assert [c for c in world.clamp_log if c[1] != "Cycle2_Home"] == [
        ("ClampsClose", "LoadPositioner", True, False), ("ClampsOpen", "UnloadPositioner", False, True)], world.clamp_log
    assert world.visible.get(S2.CLAMP_ARMS_CLOSED) is False and world.visible.get(S2.CLAMP_ARMS_OPEN) is True
    assert world.params.get("POS_CLAMPS") == "OPEN" and world.params.get("GRIPPER") == "OPEN"
    assert world.x == S2.HANDLER_HOME_X and world.rot == S2.L2.POS_UNLOAD_ROT and world.carrier_x == S2.L2.CONV_END_X * 1000
    print("   DemoCycle2 simulated: %d gripper actions, spool stored at bay %s, flange 2 on the station"
          % (len(log), S2.L2.STORAGE_TARGET))


def test_positioner_handover():
    """Parts released on the faceplate follow the positioner through the whole stage-1 DemoCycle (rot 180 -> -20 ->
    360, tilt +-90) and are picked up at rot 540 exactly where the unload targets expect them."""
    RDK, cell = build_mock()
    world = World(cell)
    fr = world.items[S2.FACEPLATE_FRAME]
    for n in (S2.PART_NAMES["flange"], S2.PART_NAMES["elbow"], S2.PART_NAMES["pipe"]):
        world.items[n].parent = fr
        world.items[n].pose = eye(4)
    world.rot, world.tilt = S2.L2.POS_UNLOAD_ROT, 0.0    # DemoCycle ends at rot 360 / tilt 0, Pos_RotUnload -> 540
    G = world.pose(S2.PART_NAMES["pipe"]) * S2.grasp_frame("spool")
    wps, _ = S2.handler_plan(S2.load_plan())
    P = wps["Load_Place"]["pose"]
    assert rm.norm(rm.subs3(G.Pos(), P.Pos())) < 1e-6 and min(rm.pose_angle_between(G, P),
                                                             rm.pose_angle_between(G * rotz(math.pi), P)) < 1e-9


# ================================================================ 8. RoboDK-like python (no bpy, no numpy), fallback, idempotence
NO_BPY_SNIPPET = """
import sys
sys.modules["bpy"] = None
sys.modules["numpy"] = None
sys.path.insert(0, %r)
import t_robodk2 as T
assert T.plan2 is None and not hasattr(sys.modules["tools"], "__file__")    # layout2 imported through the stub
RDK, cell = T.build_mock()
assert cell.problems == [] and "DemoCycle2" in cell.programs
T.test_cycle_simulation(RDK, cell)
print("NO_BPY_OK")
"""


def test_without_bpy():
    """The builder must run in RoboDK's python: no bpy and no numpy (layout2 through the stub `tools`, plan data from
    the snapshot)."""
    import subprocess
    r = subprocess.run([sys.executable, "-c", NO_BPY_SNIPPET % HERE], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and "NO_BPY_OK" in r.stdout, (r.stdout[-2000:], r.stderr[-2000:])


def test_fallback_and_idempotent():
    RDK, cell = build_mock(plan={})                      # no snapshot: nominal tacks, home seeds
    assert cell.problems == []
    tacks = [n for n in cell.target_info if n.startswith("Tack_") and not n.endswith("_back")]
    assert len(tacks) == 6
    assert all(close(cell.target_info[n]["seed"], S2.HANDLER_Q_HOME) for n in cell.target_info if not n.startswith("Tack_"))
    for n in tacks:
        d, reach = S2.reach_of("tack", cell.target_info[n]["pose"])
        assert d <= reach, (n, d)
    first = (sorted(cell.programs), sorted(cell.targets), sorted(cell.macros))
    listing = sorted(os.listdir(S2.GEN_DIR))
    cell2 = S2.build(RDK, {})                           # same RoboDK: the old station is closed and rebuilt
    assert ("CloseStation", ()) in RDK.calls
    assert (sorted(cell2.programs), sorted(cell2.targets), sorted(cell2.macros)) == first
    assert sorted(os.listdir(S2.GEN_DIR)) == listing
    # rebuild with the snapshot again so generated/ matches the default build
    build_mock()


if __name__ == "__main__":
    tests = [test_compile, test_conversions, test_frames_vs_plan2, test_snapshot, test_reach_and_track,
             test_fallback_and_idempotent, test_positioner_handover, test_without_bpy]
    for fn in tests:
        fn()
        print("ok ", fn.__name__)
    RDK_, cell_ = test_mock_build()
    print("ok  test_mock_build")
    test_cycle_simulation(RDK_, cell_)
    print("ok  test_cycle_simulation")
    print("all stage-2 robodk tests passed")
