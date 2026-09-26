"""Unit test for robodk/build_station.py without RoboDK: pure maths + a full build() against a mock Robolink.

Run:  python3 tests/t_robodk_math.py
"""
import math
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "robodk"))

from robodk import robolink as rl
from robodk import robomath as rm
from robodk.robomath import Mat, eye, invH
import build_station as B
from cell import layout as L


def close(a, b, tol=1e-6):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


# ---------------------------------------------------------------- 1. constants mirror cell/layout.py
def test_constants():
    assert close([B.POS_TILT_AXIS_Z, B.POS_FACEPLATE_OFFSET, B.POS_FIXTURE_THICK],
                 [L.POS_TILT_AXIS_Z * 1000, L.POS_FACEPLATE_OFFSET * 1000, L.POS_FIXTURE_THICK * 1000])
    assert close([B.PIPE_OD, B.ELBOW_R, B.SEAM_A_Z, B.SEAM_RADIUS], [L.PIPE_OD * 1000, L.ELBOW_R * 1000, L.SEAM_A_Z * 1000, L.SEAM_RADIUS * 1000])
    assert close(B.SEAM_B_CENTER, [v * 1000 for v in L.SEAM_B_CENTER])
    assert close([B.TRACK_X, B.TRACK_TOP_Z, B.ROBOT_HOME_Y], [L.TRACK_X * 1000, L.TRACK_TOP_Z * 1000, L.ROBOT_HOME_Y * 1000])
    assert abs(B.ROBOT_YAW_DEG - math.degrees(L.ROBOT_YAW)) < 1e-9
    assert close([B.TORCH_BRACKET_LEN, B.TORCH_NECK_LEN, B.TORCH_HEAD_LEN, B.TORCH_STICKOUT],
                 [L.TORCH_BRACKET_LEN * 1000, L.TORCH_NECK_LEN * 1000, L.TORCH_HEAD_LEN * 1000, L.TORCH_STICKOUT * 1000])
    assert abs(B.TORCH_BEND_DEG - math.degrees(L.TORCH_BEND_ANGLE)) < 1e-9
    q_deg = [math.degrees(q) for q in L.ROBOT_Q_HOME]
    assert close(B.ROBOT_Q_HOME, q_deg, 0.5)
    assert close(B.FENCE_X, [v * 1000 for v in L.FENCE_X]) and close(B.FENCE_Y, [v * 1000 for v in L.FENCE_Y])


# ---------------------------------------------------------------- 2. kinematic maths
def test_spool_pose():
    # tilt 90 / rot 450: pipe along +Y, seam B centre at (761, 381, 1350), seam A centre at (380, 0, 1350)
    T = B.spool_pose(90.0, B.ROT_A1)
    cB, axB, _, _ = B.seam_frame_world("B", T)
    assert close(cB, [761.0, 381.0, 1350.0], 1e-6), cB
    assert close(axB, [0.0, 1.0, 0.0]), axB
    cA, axA, _, _ = B.seam_frame_world("A", B.spool_pose(90.0, B.ROT_A0))
    assert close(cA, [380.0, 0.0, 1350.0]) and close(axA, [1.0, 0.0, 0.0])
    # after the 180 deg index the pipe points to -Y
    cB2, _, _, _ = B.seam_frame_world("B", B.spool_pose(90.0, B.ROT_B2))
    assert close(cB2, [761.0, -381.0, 1350.0])
    # loading position: flange axis up
    _, axA0, _, _ = B.seam_frame_world("A", B.spool_pose(0.0, 0.0))
    assert close(axA0, [0.0, 0.0, 1.0])


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


def test_choreography():
    ch = B.choreography()
    TA = ch["weldA"]
    assert close(TA.Pos(), [380.0, 0.0, 1350.0 + B.SEAM_RADIUS], 1e-6), TA.Pos()
    assert TA.VZ()[2] < -0.98                                  # torch pointing down into the top of the seam
    assert TA.VX()[0] > 0.5                                    # torch body leaning toward the robot (+X)
    assert ch["approachA"].Pos()[2] - TA.Pos()[2] == B.LIFT_APPROACH
    assert len(ch["scan"]) == 7 and abs(ch["scan"][0].Pos()[0] - (380 - B.SCAN_HALF)) < 1e-9
    for key, cy, sign in (("sector1", 381.0, 1), ("sector2", 381.0, -1), ("sector3", -381.0, 1), ("sector4", -381.0, -1)):
        poses = ch[key]
        p0, p1, pe = poses[0].Pos(), poses[1].Pos(), poses[-1].Pos()
        assert abs(p0[0] - 761.0) < 1e-6 and abs(p0[1] - cy) < 1e-6 and abs(p0[2] - (1350 + B.SEAM_RADIUS)) < 1e-6, (key, p0)
        assert (p1[0] - p0[0]) * sign > 0, key                  # sector direction toward +X / -X
        ang = math.degrees(math.atan2(pe[2] - 1350.0, (pe[0] - 761.0) * sign))
        assert abs(ang - (90 - B.SECTOR_DEG)) < 1e-6, (key, ang)   # ends 2 deg past 3 / 9 o'clock
        for T in poses:                                       # tool Z points into the surface (toward the axis)
            r = rm.subs3(T.Pos(), [761.0, cy, 1350.0])
            assert rm.dot(T.VZ(), rm.normalize3(r)) < -0.98
    assert len(ch["weldA_movel"]) == 4 * B.SECTOR_POINTS + 1


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
    assert B.fence_segments(0, 10, [(2, 3), (8, 12)]) == [(0, 2), (3, 8)]


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
        it = self._new(name, rl.ITEM_TYPE_ROBOT, ndof)
        it.pose = base
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
        return self._new(name, rl.ITEM_TYPE_TARGET, robot.ndof)

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
    # faceplate mechanism base: 1600 above the floor, Z up at tilt 0
    fb = B.tilt_base_pose() * cell.rot.parent.pose
    assert close(fb.Pos(), [0.0, 0.0, 1600.0]) and close(fb.VZ(), [0.0, 0.0, 1.0])
    demo = cell.programs["DemoCycle"].log
    calls = [c for k, c in demo if k == "call"]
    order = ["View_C1_wide", "Cell_Home", "Pos_TiltUp", "Track_ToA", "Robot_ApproachA", "Robot_ScanA", "Robot_WeldA",
             "Track_ToB1", "Robot_WeldB_S1", "Robot_WeldB_S2", "Pos_Index180", "Track_ToB2", "Robot_WeldB_S3",
             "Robot_WeldB_S4", "Track_Home", "Pos_TiltDown"]
    idx = [calls.index(c) for c in order]
    assert idx == sorted(idx), calls
    for c in calls:
        assert c in cell.programs or c in cell.macros, "DemoCycle calls unknown program " + c
    wa = cell.programs["Robot_WeldA"].log
    assert ("call", "ArcOn") in wa and ("call", "Pos_RotSeamA_2") in wa and ("call", "ArcOff") in wa
    s1 = cell.programs["Robot_WeldB_S1"].log
    assert sum(1 for k, _ in s1 if k == "MoveL") == B.SECTOR_POINTS + 2
    assert len(cell.targets) > 4 * B.SECTOR_POINTS + 20
    assert set(cell.macros) == {"ArcOn", "ArcOff"} | {"View_" + s[0] for s in B.SHOTS}
    assert os.path.exists(os.path.join(B.GEN_DIR, "torch.stl"))
    assert ("setViewPose",) == RDK.calls[-1][:1] or any(k == "setViewPose" for k, _ in RDK.calls)
    print("mock build: %d items, %d programs, %d targets" % (len(RDK.items), len(cell.programs), len(cell.targets)))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok ", name)
    print("all robodk tests passed")
