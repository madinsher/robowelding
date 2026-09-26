"""Unit test for robodk/build_station.py without RoboDK: pure maths + a full build() against a mock Robolink.

The constants of build_station.py (mm / deg) are checked against cell/layout.py (metres) and against the
choreography of cell/animation.py (timeline, ROT_A0/ROT_A1, TILT_B1/TILT_B2, TRACK_A/B1/B2, TRANSIT_TCP, lifts,
SHOTS): animation.py imports bpy, so its top-level assignments are extracted with `ast` and evaluated here.

Run:  python3 tests/t_robodk_math.py
"""
import ast
import json
import math
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "robodk"))

from robodk import robolink as rl
from robodk import robomath as rm
from robodk.robomath import Mat, eye, invH, rotz
import build_station as B
from cell import layout as L


def close(a, b, tol=1e-6):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def anim_source():
    with open(os.path.join(ROOT, "cell", "animation.py"), encoding="utf-8") as fh:
        return fh.read()


def anim_constants():
    """Top-level assignments of cell/animation.py evaluated without bpy (only those that need L / math)."""
    ns = {"L": L, "math": math}
    for node in ast.parse(anim_source()).body:
        if isinstance(node, ast.Assign):
            try:
                exec(compile(ast.Module(body=[node], type_ignores=[]), "animation.py", "exec"), ns)
            except Exception:
                pass
    return ns


# ---------------------------------------------------------------- 1. constants mirror cell/layout.py + animation.py
def test_constants():
    assert close([B.POS_TILT_AXIS_Z, B.POS_FACEPLATE_OFFSET, B.POS_FIXTURE_THICK, B.POS_TILT_HALF_WIDTH, B.POS_FACEPLATE_RADIUS],
                 [L.POS_TILT_AXIS_Z * 1000, L.POS_FACEPLATE_OFFSET * 1000, L.POS_FIXTURE_THICK * 1000,
                  L.POS_TILT_HALF_WIDTH * 1000, L.POS_FACEPLATE_RADIUS * 1000])
    assert close(B.POS_PEDESTAL_FOOTPRINT, [v * 1000 for v in L.POS_PEDESTAL_FOOTPRINT])
    assert abs(B.POS_TOWER_X - 680.0) < 1e-9                       # towers at x = +/-680 (positioner.py: hw + 0.06)
    assert close([B.PIPE_OD, B.ELBOW_R, B.SEAM_A_Z, B.SEAM_RADIUS], [L.PIPE_OD * 1000, L.ELBOW_R * 1000, L.SEAM_A_Z * 1000, L.SEAM_RADIUS * 1000])
    assert close(B.SEAM_B_CENTER, [v * 1000 for v in L.SEAM_B_CENTER])
    assert close([B.TRACK_X, B.TRACK_TOP_Z, B.ROBOT_HOME_Y], [L.TRACK_X * 1000, L.TRACK_TOP_Z * 1000, L.ROBOT_HOME_Y * 1000])
    assert close([B.TRACK_Y_MIN, B.TRACK_Y_MAX], [L.TRACK_Y_MIN * 1000, L.TRACK_Y_MAX * 1000])
    assert abs(B.ROBOT_YAW_DEG - math.degrees(L.ROBOT_YAW)) < 1e-9
    assert close([B.TORCH_BRACKET_LEN, B.TORCH_NECK_LEN, B.TORCH_HEAD_LEN, B.TORCH_STICKOUT],
                 [L.TORCH_BRACKET_LEN * 1000, L.TORCH_NECK_LEN * 1000, L.TORCH_HEAD_LEN * 1000, L.TORCH_STICKOUT * 1000])
    assert abs(B.TORCH_BEND_DEG - math.degrees(L.TORCH_BEND_ANGLE)) < 1e-9
    q_deg = [math.degrees(q) for q in L.ROBOT_Q_HOME]
    assert close(B.ROBOT_Q_HOME, q_deg, 0.1), (B.ROBOT_Q_HOME, q_deg)
    assert close(B.FENCE_X, [v * 1000 for v in L.FENCE_X]) and close(B.FENCE_Y, [v * 1000 for v in L.FENCE_Y])
    assert close(B.FUME_HOOD["pos"], [v * 1000 for v in L.FUME_HOOD["pos"]])
    assert abs(B.TORCH_CLEANER["height"] - L.TORCH_CLEANER["height"] * 1000) < 1e-6

    # choreography numbers = cell/animation.py
    A = anim_constants()
    assert A["N_FRAMES"] == 1104 and A["T"]["end"][1] == 1104
    assert abs(B.ROT_A0 - A["ROT_A0"]) < 1e-9 and abs(B.ROT_A1 - A["ROT_A1"]) < 1e-9 and B.ROT_A1 == 360.0
    assert abs(B.TILT_B1 - A["TILT_B1"]) < 1e-9 and abs(B.TILT_B2 - A["TILT_B2"]) < 1e-9
    assert close([B.TRACK_A, B.TRACK_B1, B.TRACK_B2, B.SEAM_B_Y],
                 [A["TRACK_A"] * 1000, A["TRACK_B1"] * 1000, A["TRACK_B2"] * 1000, A["SEAM_B_Y"] * 1000])
    assert close(B.TRANSIT_TCP, [v * 1000 for v in A["TRANSIT_TCP"]])
    assert abs(B.LIFT_APPROACH - A["LIFT_APPROACH"] * 1000) < 1e-6 and abs(B.LIFT_PARK - A["LIFT_PARK"] * 1000) < 1e-6
    assert B.TILT_LIMITS[0] <= B.TILT_B2 and B.TILT_LIMITS[1] >= B.TILT_B1
    src = anim_source()
    m = re.search(r"sgn \* \(([\d.]+) \* s - ([\d.]+)\)", src)              # sector sweep: -4 .. +94 deg
    assert m and abs(B.SECTOR_DEG - float(m.group(1))) < 1e-9 and abs(-B.SECTOR_A0 - float(m.group(2))) < 1e-9
    m = re.search(r'radial\("via", S1, ([\d.]+)\)', src)                     # via point retract
    assert m and abs(B.VIA_RETRACT - float(m.group(1)) * 1000) < 1e-6
    m = re.search(r"a_top \+ sgn \* ([\d.]+)\)\s*# via", src)
    assert m and abs(B.VIA_ANGLE - float(m.group(1))) < 1e-9
    m = re.search(r"x = -([\d.]+) \+ ([\d.]+) \* \(0\.5", src)             # laser sweep -60..+60 mm
    assert m and abs(B.SCAN_HALF - float(m.group(1)) * 1000) < 1e-6
    m = re.search(r"TA_scan = offset\(TA, \(0, 0, ([\d.]+)\)\)", src)      # laser height 35 mm
    assert m and abs(B.SCAN_HEIGHT - float(m.group(1)) * 1000) < 1e-6
    # camera shots: names, frame ranges, start camera / aim positions and lenses
    shots = A["SHOTS"]
    assert len(B.SHOTS) == len(shots) == 12
    for mine, ref in zip(B.SHOTS, shots):
        name, f0, f1, cam, aim, lens = mine
        assert name == ref[0] and f0 == ref[1] and f1 == ref[2], (mine, ref)
        assert close(cam, [v * 1000 for v in ref[3]], 1e-6) and close(aim, [v * 1000 for v in ref[5]], 1e-6), (mine, ref)
        assert lens == ref[7]
    with open(os.path.join(ROOT, "post", "storyboard.json"), encoding="utf-8") as fh:
        sb = json.load(fh)
    assert sb["frames"] == A["N_FRAMES"] and B.SHOTS[-1][2] == sb["frames"]


# ---------------------------------------------------------------- 2. kinematic maths
def test_spool_pose():
    # tilt +90 / rot 360: pipe along +X, seam B centre at (381, 761, 1350), axis X
    T = B.spool_pose(90.0, B.ROT_A1)
    cB, axB, _, _ = B.seam_frame_world("B", T)
    assert close(cB, [381.0, 761.0, 1350.0], 1e-6), cB
    assert close(axB, [1.0, 0.0, 0.0]), axB
    # seam A at tilt +90: centre (0, 380, 1350), axis = faceplate normal = +Y
    cA, axA, _, _ = B.seam_frame_world("A", B.spool_pose(90.0, B.ROT_A0))
    assert close(cA, [0.0, 380.0, 1350.0]) and close(axA, [0.0, 1.0, 0.0]), (cA, axA)
    # after the 180 deg index (tilt -90) the faceplate faces -Y, seam B centre at (381, -761, 1350)
    cB2, axB2, _, _ = B.seam_frame_world("B", B.spool_pose(-90.0, B.ROT_A1))
    assert close(cB2, [381.0, -761.0, 1350.0]) and close(axB2, [1.0, 0.0, 0.0])
    # loading position: flange axis up, spool 30 mm above the faceplate
    T0 = B.spool_pose(0.0, 0.0)
    _, axA0, _, _ = B.seam_frame_world("A", T0)
    assert close(axA0, [0.0, 0.0, 1.0]) and close(T0.Pos(), [0.0, 0.0, 1350.0 + 250.0 + 30.0])
    # the tilt mechanism (1R about its base Z, sense +1) reproduces spool_pose: at joint +90 the faceplate
    # mechanism base is at (0, 250, 1350) with its Z along world +Y (cell/positioner.set_tilt convention)
    assert close(B.tilt_base_pose().VZ(), [-1.0, 0.0, 0.0])            # tilt axis = world X (mechanism Z = -X)
    rel = invH(B.tilt_base_pose()) * B.rot_base_pose_world()
    for tilt in (90.0, -90.0, 37.0):
        F = B.tilt_base_pose() * rotz(math.radians(tilt)) * rel
        S = F * rotz(math.radians(B.ROT_A0)) * B.transl(0, 0, B.POS_FIXTURE_THICK)
        ref = B.spool_pose(tilt, B.ROT_A0)
        assert close(S.Pos(), ref.Pos()) and close(S.VZ(), ref.VZ()) and close(S.VX(), ref.VX()), tilt
    F = B.tilt_base_pose() * rotz(math.radians(90.0)) * rel
    assert close(F.Pos(), [0.0, 250.0, 1350.0]) and close(F.VZ(), [0.0, 1.0, 0.0])
    # the robot base rides the track: mechanism Z = world +Y, base at (2000, y, 450) yaw 180
    rb = B.track_base_pose() * B.transl(0, 0, 761.0) * invH(B.track_base_pose()) * B.robot_base_pose_world(0.0)
    assert close(rb.Pos(), [2000.0, 761.0, 450.0]) and close(rb.VX(), [-1.0, 0.0, 0.0])


def test_tcp():
    T = B.tcp_pose()
    s, c = math.sin(math.radians(45)), math.cos(math.radians(45))
    assert close(T.Pos(), [185 * s, 0.0, 300 + 185 * c]), T.Pos()
    assert close(T.VZ(), [s, 0.0, c])           # wire direction = bent head direction


def test_seam_points():
    rows = B.seam_points_local("B", 36)
    assert len(rows) == 37 and close(rows[0], rows[-1])
    c = B.SEAM_B_CENTER
    for r in rows:
        p, n = r[:3], r[3:]
        assert abs(rm.norm(n) - 1) < 1e-9 and abs(n[0]) < 1e-9          # normal perpendicular to the +X axis
        assert abs(rm.norm(rm.subs3(p, c)) - B.SEAM_RADIUS) < 1e-9


def test_torch_target():
    p, n, t = [0, 0, 0], [0, 0, 1], [1, 0, 0]
    T = B.torch_target(p, n, t, [-1.0, 0.3, 0.9], 10.0)
    z, x, y = T.VZ(), T.VX(), T.VY()
    assert abs(rm.dot(z, n) + math.cos(math.radians(10))) < 1e-9
    assert abs(rm.dot(z, t) - math.sin(math.radians(10))) < 1e-9
    assert abs(rm.dot(x, z)) < 1e-9 and abs(rm.dot(x, y)) < 1e-9 and abs(rm.norm(y) - 1) < 1e-9
    lean_proj = rm.subs3([-1.0, 0.3, 0.9], rm.mult3(z, rm.dot([-1.0, 0.3, 0.9], z)))
    assert abs(rm.dot(x, rm.normalize3(lean_proj)) - 1.0) < 1e-9      # X = lean projected on the plane normal to Z
    R = B.radial(T, [0, 0, 1], 120.0)
    assert close(R.Pos(), [0, 0, 120.0]) and close(R.VZ(), z)


def test_choreography():
    ch = B.choreography()
    R = B.SEAM_RADIUS
    s10 = math.sin(math.radians(10))
    # seam A: torch static at the top of the seam (0, 380, 1487), pointing down, pushed toward -X
    # (the part rotates about +Y, its top surface moves toward +X, so the torch travels -X relative to the part)
    TA = ch["weldA"]
    assert close(TA.Pos(), [0.0, 380.0, 1350.0 + R], 1e-6), TA.Pos()
    assert TA.VZ()[2] < -0.98 and abs(TA.VZ()[0] + s10) < 1e-9 and abs(TA.VZ()[1]) < 1e-9, TA.VZ()
    assert TA.VX()[0] > 0.5                                    # torch body leaning toward the robot (+X)
    assert abs(ch["approachA"].Pos()[2] - TA.Pos()[2] - B.LIFT_PARK) < 1e-9
    # laser scan: 7 points, +/-60 mm along Y (the seam axis) 35 mm above the top of the seam
    scan = ch["scan"]
    assert len(scan) == 7
    assert close(scan[0].Pos(), [0.0, 380.0 - B.SCAN_HALF, 1350.0 + R + B.SCAN_HEIGHT], 1e-6), scan[0].Pos()
    assert close(scan[-1].Pos(), [0.0, 380.0 + B.SCAN_HALF, 1350.0 + R + B.SCAN_HEIGHT], 1e-6)
    for T in scan:
        assert close(T.VZ(), TA.VZ()) and close(T.VX(), TA.VX())
    # transit pose: torch straight down, tip at (1350, track y, 2050), body leaning +X
    for key, y in (("transit", B.TRACK_A), ("transitB1", B.TRACK_B1), ("transitB2", B.TRACK_B2)):
        T = ch[key]
        assert close(T.Pos(), [1350.0, y, 2050.0]) and close(T.VZ(), [0, 0, -1.0]) and close(T.VX(), [1.0, 0, 0]), key
    # seam B sectors: circle in the plane x = 381 around (381, +/-761, 1350); each from 12 o'clock (-4 deg) to
    # +94 deg toward -Y (sectors 1, 3) or +Y (sectors 2, 4)
    for key, cy, d in (("sector1", 761.0, -1), ("sector2", 761.0, 1), ("sector3", -761.0, -1), ("sector4", -761.0, 1)):
        poses, normals = ch[key], ch[key + "_n"]
        assert len(poses) == len(normals) == B.SECTOR_POINTS + 1

        def ang(p):
            return math.degrees(math.atan2(d * (p[1] - cy), p[2] - 1350.0))
        assert abs(ang(poses[0].Pos()) - B.SECTOR_A0) < 1e-6, (key, ang(poses[0].Pos()))
        assert abs(ang(poses[-1].Pos()) - B.SECTOR_A1) < 1e-6, (key, ang(poses[-1].Pos()))
        assert (poses[1].Pos()[1] - poses[0].Pos()[1]) * d > 0, key    # direction toward -Y / +Y
        for T, n in zip(poses, normals):
            p = T.Pos()
            r = rm.subs3(p, [381.0, cy, 1350.0])
            assert abs(p[0] - 381.0) < 1e-6 and abs(rm.norm(r) - R) < 1e-6, (key, p)
            assert close(n, rm.normalize3(r))                          # outward normal
            assert rm.dot(T.VZ(), n) < -0.98                           # tool Z into the surface (toward the axis)
            assert abs(rm.dot(T.VZ(), n) + math.cos(math.radians(10))) < 1e-9   # 10 deg push angle
        # via point: 1:30 o'clock on the sector side, retracted 160 mm along the normal, no push angle
        V = ch[key + "_via"]
        pv = V.Pos()
        rv = rm.subs3(pv, [381.0, cy, 1350.0])
        assert abs(ang(pv) - B.VIA_ANGLE) < 1e-6 and abs(rm.norm(rv) - (R + B.VIA_RETRACT)) < 1e-6, (key, pv)
        assert rm.dot(V.VZ(), rm.normalize3(rv)) < -0.999999
        # radial approach: 120 mm along the start normal, orientation unchanged
        A = B.radial(poses[0], normals[0], B.LIFT_APPROACH)
        assert abs(rm.norm(rm.subs3(A.Pos(), [381.0, cy, 1350.0])) - (R + B.LIFT_APPROACH)) < 1e-6
        assert close(A.VZ(), poses[0].VZ())
    # sectors 1/2 (3/4) start at the same 12 o'clock point mirrored by 4 deg and end at 3 / 9 o'clock (+4 deg past)
    assert abs(ch["sector1"][-1].Pos()[2] - ch["sector2"][-1].Pos()[2]) < 1e-6
    assert ch["sector1"][-1].Pos()[2] < 1350.0                          # 94 deg: just below the seam centre
    assert len(ch["weldA_movel"]) == len(ch["weldA_movel_n"]) == 4 * B.SECTOR_POINTS + 1


def test_view_and_camera():
    cam, aim = (-3800, -6200, 2500), (900, 0, 1300)
    V = B.view_pose(cam, aim)
    a = B.xform_point(V, list(aim))
    d = rm.norm(rm.subs3(list(aim), list(cam)))
    assert close(a, [0.0, 0.0, -d], 1e-6), a                  # aim straight ahead (OpenGL: -Z)
    up = B.xform_vec(V, [0, 0, 1])
    assert up[1] > 0.9                                         # world Z is up on screen
    C = B.camera_frame_pose(cam, aim)
    assert close(C.Pos(), list(map(float, cam)))
    assert rm.dot(C.VZ(), rm.normalize3(rm.subs3(list(aim), list(cam)))) > 0.999999
    assert C.VY()[2] < 0                                       # +Y down
    assert abs(B.lens_to_fov_deg(32) - 35.1) < 0.2
    for name, f0, f1, cam, aim, lens in B.SHOTS:               # every storyboard shot gives a proper view pose
        V = B.view_pose(cam, aim)
        assert V.isHomogeneous() and B.xform_point(V, list(aim))[2] < 0


def test_meshes():
    tris = B.torch_mesh_triangles()
    assert len(tris) > 200
    tip = B.tcp_pose().Pos()
    far = max(rm.norm(p) for tri in tris for p in tri)
    assert abs(far - rm.norm(tip)) < 2.0                        # the wire tip is the farthest vertex
    box = B.box_tris((0, 0, 0), (2, 4, 6))
    assert len(box) == 12
    assert all(abs(p[0]) == 1 and abs(p[1]) == 2 and abs(p[2]) == 3 for tri in box for p in tri)
    m = B.tris_to_mat(box)
    assert m.size() == (3, 36)
    moved = B.tris_transform(box, B.transl(10, 0, 0))
    assert all(abs(p[0] - 10) == 1 for tri in moved for p in tri)
    assert B.fence_segments(0, 10, [(2, 3), (8, 12)]) == [(0, 2), (3, 8)]
    # positioner geometry: bearing towers at x = +/-680 on the tilt axis height, cradle hubs at +/-540
    xs = sorted({round(p[0]) for tris_, _ in B.positioner_static_shapes() for tri in tris_ for p in tri})
    assert -680 - 130 in xs and 680 + 130 in xs, xs               # tower boxes (260 wide) centred at +/-680
    hub = {(round(p[0]), round(p[2])) for tris_, _ in B.positioner_cradle_shapes() for tri in tris_ for p in tri}
    assert (440, 1350) in hub and (640, 1350) in hub and (-640, 1350) in hub   # trunnion hubs on the tilt axis at +/-540
    for tris_, _ in B.positioner_cradle_shapes():                    # cradle arms clear the faceplate (|x| > R)
        for tri in tris_:
            for p in tri:
                assert p[2] <= 1350 + 250 - 60 or abs(p[0]) > B.POS_FACEPLATE_RADIUS or abs(p[1]) > 200, p


# ---------------------------------------------------------------- 3. full build against a mock Robolink
class MockItem:
    def __init__(self, link, name, itype, ndof=0):
        self.link, self.name, self.itype, self.ndof = link, name, itype, ndof
        self.parent = None
        self.pose = eye(4)
        self.joints = [0.0] * max(ndof, 1)
        self.log = []                       # program instructions / calls

    # generic
    def Valid(self, check_deleted=False):
        return True

    def Type(self):
        return self.itype

    def Name(self):
        return self.name

    def setName(self, n):
        self.name = n
        self.link.names[n] = self
        return self

    def Parent(self):
        return self.parent or self.link.station

    def setParent(self, parent):
        self.parent = parent
        return self

    def setPose(self, pose):
        assert isinstance(pose, Mat) and pose.isHomogeneous()
        self.pose = pose
        return self

    def PoseAbs(self):
        return self.pose

    def SolveFK(self, joints, tool=None, reference=None):
        assert len(joints) == self.ndof
        return eye(4)

    def SolveIK(self, pose, joints_approx=None, tool=None, reference=None):
        assert pose.isHomogeneous()
        return Mat([0.0, -10.0, 20.0, 0.0, 60.0, 0.0])

    def setJoints(self, joints):
        assert len(list(joints)) in (self.ndof, 6), (self.name, joints)
        self.joints = list(joints)
        self.log.append(("setJoints", tuple(joints)))
        return self

    def setJointsHome(self, joints):
        return self

    def __getattr__(self, attr):
        # every other setter/instruction is logged (setVisible, setColor, setSpeed, RunInstruction, ...)
        if attr.startswith("_"):
            raise AttributeError(attr)

        def fn(*args, **kw):
            self.log.append((attr, args))
            if attr == "Update":
                return (len(self.log), 10.0, 1000.0, 1.0, "")
            if attr == "setMachiningParameters":
                return MockItem(self.link, self.name + " prog", rl.ITEM_TYPE_PROGRAM), 1.0
            return self
        return fn

    def MoveJ(self, target):
        assert isinstance(target, MockItem) and target.itype == rl.ITEM_TYPE_TARGET
        self.log.append(("MoveJ", target.name))

    def MoveL(self, target):
        assert isinstance(target, MockItem) and target.itype == rl.ITEM_TYPE_TARGET
        self.log.append(("MoveL", target.name))

    def RunInstruction(self, code, run_type=0):
        self.log.append(("call" if run_type == rl.INSTRUCTION_CALL_PROGRAM else "comment", code))
        return 0


class MockRobolink:
    def __init__(self, lib):
        self.lib = lib
        self.names = {}
        self.station = MockItem(self, "station", rl.ITEM_TYPE_STATION)
        self.items = []
        self.calls = []

    def _new(self, name, itype, ndof=0):
        it = MockItem(self, name, itype, ndof)
        self.items.append(it)
        self.names[name] = it
        return it

    def getParam(self, p, str_type=True):
        return self.lib if p == "PATH_LIBRARY" else None

    def AddStation(self, name):
        return self._new(name, rl.ITEM_TYPE_STATION)

    def AddFrame(self, name, parent=0):
        it = self._new(name, rl.ITEM_TYPE_FRAME)
        it.parent = parent if parent else None
        return it

    def AddShape(self, payload, add_to=0, override=False):
        assert isinstance(payload, list) and len(payload) % 2 == 0
        for k in range(0, len(payload), 2):
            m, c = payload[k], payload[k + 1]
            assert isinstance(m, Mat) and m.size(0) == 3 and m.size(1) % 3 == 0, m.size()
            assert len(c) == 4
        return self._new("shape", rl.ITEM_TYPE_OBJECT)

    def AddFile(self, path, parent=0):
        assert os.path.exists(path), path
        if path.endswith(".robot"):
            it = self._new(os.path.basename(path), rl.ITEM_TYPE_ROBOT, 6)
        elif path.endswith(".py"):
            it = self._new(os.path.basename(path)[:-3], rl.ITEM_TYPE_PROGRAM_PYTHON)
        elif parent and parent.itype == rl.ITEM_TYPE_ROBOT:
            it = self._new(os.path.basename(path), rl.ITEM_TYPE_TOOL)
            it.parent = parent
        else:
            it = self._new(os.path.basename(path), rl.ITEM_TYPE_OBJECT)
        return it

    def BuildMechanism(self, mtype, objs, params, jb, jh, senses, lo, hi, base=None, tool=None, name="", robot=None):
        ndof = len(objs) - 1
        assert all(o.itype == rl.ITEM_TYPE_OBJECT for o in objs)
        assert len(jh) == len(senses) == len(lo) == len(hi) == ndof
        assert mtype in (rl.MAKE_ROBOT_1R, rl.MAKE_ROBOT_1T)
        assert all(lo[k] <= jb[k] <= hi[k] and lo[k] <= jh[k] <= hi[k] for k in range(ndof)), (name, jb, jh, lo, hi)
        it = self._new(name, rl.ITEM_TYPE_ROBOT, ndof)
        it.pose = base
        it.limits = (list(lo), list(hi))
        return it

    def AddCurve(self, pts, ref=0, add_to_ref=False, proj=0):
        assert all(len(r) == 6 for r in pts)
        return self._new("curve", rl.ITEM_TYPE_OBJECT)

    def AddMachiningProject(self, name, robot=0):
        return self._new(name, rl.ITEM_TYPE_MACHINING)

    def AddProgram(self, name, robot=0):
        assert robot.itype == rl.ITEM_TYPE_ROBOT
        it = self._new(name, rl.ITEM_TYPE_PROGRAM)
        it.robot = robot
        return it

    def AddTarget(self, name, frame=0, robot=0):
        assert frame.itype == rl.ITEM_TYPE_FRAME and robot.itype == rl.ITEM_TYPE_ROBOT
        it = self._new(name, rl.ITEM_TYPE_TARGET, robot.ndof)
        it.robot = robot
        return it

    def __getattr__(self, attr):
        if attr.startswith("_"):
            raise AttributeError(attr)

        def fn(*args, **kw):
            self.calls.append((attr, args))
            if attr == "Cam2D_Add":
                return self._new("cam", rl.ITEM_TYPE_CAMERA)
            return None
        return fn


def test_mock_build():
    lib = tempfile.mkdtemp(prefix="rdklib_")
    os.makedirs(os.path.join(lib, "ABB"))
    open(os.path.join(lib, "ABB", "ABB-IRB-4600-40-2.55.robot"), "w").close()
    B.SAVE_RDK = False
    RDK = MockRobolink(lib)
    cell = B.build(RDK)
    assert cell.robot.name == "ABB IRB 4600-40/2.55" and cell.torch.itype == rl.ITEM_TYPE_TOOL
    # the robot base frame rides on the track flange, the faceplate mechanism on the tilt flange
    assert cell.robot.parent.itype == rl.ITEM_TYPE_FRAME and cell.robot.parent.parent is cell.track
    assert cell.rot.parent.parent is cell.tilt and cell.spool.parent is cell.rot
    assert close(cell.spool.pose.Pos(), [0, 0, B.POS_FIXTURE_THICK])
    # robot base: world (2000, y, 450) yaw 180 -> relative to the track base (Z = world Y)
    rb = B.track_base_pose() * cell.robot.parent.pose
    assert close(rb.Pos(), [2000.0, 0.0, 450.0]) and close(rb.VX(), [-1.0, 0.0, 0.0]), rb
    # tilt mechanism: axis = world X through (0,0,1350); faceplate mechanism base 1600 above the floor, Z up at tilt 0
    assert close(cell.tilt.pose.Pos(), [0.0, 0.0, 1350.0]) and close(cell.tilt.pose.VZ(), [-1.0, 0.0, 0.0])
    assert cell.tilt.limits[0][0] <= -90.0 <= 90.0 <= cell.tilt.limits[1][0]
    fb = B.tilt_base_pose() * cell.rot.parent.pose
    assert close(fb.Pos(), [0.0, 0.0, 1600.0]) and close(fb.VZ(), [0.0, 0.0, 1.0])
    # track / positioner sub-programs drive the right mechanisms to the storyboard stations
    for name, mech, val in (("Track_Home", cell.track, -1900.0), ("Track_ToA", cell.track, 200.0),
                            ("Track_ToB1", cell.track, 761.0), ("Track_ToB2", cell.track, -761.0),
                            ("Pos_TiltUp", cell.tilt, 90.0), ("Pos_Index180", cell.tilt, -90.0), ("Pos_TiltDown", cell.tilt, 0.0),
                            ("Pos_RotStart", cell.rot, -20.0), ("Pos_RotSeamA_2", cell.rot, 360.0)):
        prog = cell.programs[name]
        assert prog.robot is mech, name
        tgt = cell.targets[name + "_T"]
        assert tgt.robot is mech and close(tgt.joints, [val]), (name, tgt.joints)
    # DemoCycle order
    demo = cell.programs["DemoCycle"].log
    calls = [c for k, c in demo if k == "call"]
    order = ["View_C1_wide", "Cell_Home", "View_C2_setup", "Pos_TiltUp", "Track_ToA", "Robot_ApproachA", "View_C3a_laser",
             "Robot_ScanA", "View_C3b_closeA", "Robot_WeldA", "View_C5_reposition", "Track_ToB1", "Robot_ApproachB1",
             "View_C6a_closeB1", "Robot_WeldB_S1", "View_C6b_closeB1", "Robot_WeldB_S2", "View_C7_index", "Pos_Index180",
             "Track_ToB2", "Robot_ApproachB2", "View_C8a_closeB2", "Robot_WeldB_S3", "View_C8b_closeB2", "Robot_WeldB_S4",
             "View_C9_final", "Robot_Home", "Track_Home", "Pos_TiltDown"]
    idx = [calls.index(c) for c in order]
    assert idx == sorted(idx), calls
    for c in calls:
        assert c in cell.programs or c in cell.macros, "DemoCycle calls unknown program " + c
    for c in ("Robot_Transit", "Pos_RotSeamA_1", "Pos_RotSeamA_2", "ArcOn", "ArcOff", "View_C4_midA"):
        assert c in cell.programs or c in cell.macros
    # seam A: approach via the transit pose, weld with the positioner rotating, lift, back to transit
    appA = cell.programs["Robot_ApproachA"].log
    assert [x for x in appA if x[0] == "MoveJ"] == [("MoveJ", "Transit"), ("MoveJ", "ApproachA")]
    assert cell.targets["Transit"].log[0] == ("setAsJointTarget", ())
    wa = cell.programs["Robot_WeldA"].log
    wa_calls = [c for k, c in wa if k == "call"]
    assert wa_calls == ["ArcOn", "Pos_RotSeamA_1", "View_C4_midA", "Pos_RotSeamA_2", "ArcOff"], wa_calls
    moves = [x for x in wa if x[0] in ("MoveJ", "MoveL")]
    assert moves[-2:] == [("MoveL", "LiftA"), ("MoveJ", "Transit")], moves
    scan = cell.programs["Robot_ScanA"].log
    assert sum(1 for k, _ in scan if k == "MoveL") == 7 and ("setDO", ("LaserScan", 1)) in scan
    # seam B: sector 1 (start .. 24 points .. radial lift, then via), sector 2 (approach .. lift, then transit)
    s1 = cell.programs["Robot_WeldB_S1"].log
    assert sum(1 for k, _ in s1 if k == "MoveL") == B.SECTOR_POINTS + 2
    assert [x for x in s1 if x[0] == "MoveJ"] == [("MoveJ", "S1_via")]
    s2 = cell.programs["Robot_WeldB_S2"].log
    assert [x for x in s2 if x[0] == "MoveJ"] == [("MoveJ", "S2_app"), ("MoveJ", "Transit")]
    assert sum(1 for k, _ in s2 if k == "MoveL") == B.SECTOR_POINTS + 2
    s4 = cell.programs["Robot_WeldB_S4"].log
    assert [x for x in s4 if x[0] in ("MoveJ", "MoveL")][-1] == ("MoveJ", "Transit")
    assert ("MoveJ", "S3_via") in cell.programs["Robot_WeldB_S3"].log
    assert [x for x in cell.programs["Robot_ApproachB2"].log if x[0] == "MoveJ"] == [("MoveJ", "ApproachB2")]
    # Cartesian targets carry the poses of the choreography (mm, Cell frame)
    ch = B.choreography()
    for name, T in (("WeldA", ch["weldA"]), ("S1_start", ch["sector1"][0]), ("S3_via", ch["sector3_via"]),
                    ("ScanA_00", ch["scan"][0])):
        t = cell.targets[name]
        assert close(t.pose.Pos(), T.Pos()) and close(t.pose.VZ(), T.VZ()), name
    assert len(cell.targets) > 4 * B.SECTOR_POINTS + 20
    assert set(cell.macros) == {"ArcOn", "ArcOff"} | {"View_" + s[0] for s in B.SHOTS}
    assert os.path.exists(os.path.join(B.GEN_DIR, "torch.stl"))
    views = sorted(os.path.basename(f) for f in os.listdir(B.GEN_DIR) if f.startswith("View_"))
    assert views == sorted("View_%s.py" % s[0] for s in B.SHOTS), views     # stale View_* macros removed
    assert ("setViewPose",) == RDK.calls[-1][:1] or any(k == "setViewPose" for k, _ in RDK.calls)
    print("mock build: %d items, %d programs, %d targets" % (len(RDK.items), len(cell.programs), len(cell.targets)))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok ", name)
    print("all robodk tests passed")
