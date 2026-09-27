"""Stage-2 choreography planner (pure numpy): one scene timeline for every machine of the logistics cycle.

Timeline (scene frames, 24 fps):
    pre  : kitting (flange, elbow, pipe -> assembly station), clamping, tack welding by the small robot, the handler
           carries the tacked spool through the cell's loading opening onto the positioner, clamp studs rise, the
           positioner turns the spool into the stage-1 start orientation.
    weld : the stage-1 choreography (demo_video/cell/animation.py, 1104 frames) embedded at WELD_OFFSET:
           scene frame = WELD_OFFSET + stage-1 frame (not re-rendered; the edit splices in the stage-1 video).
    post : the welding robot parks, the positioner turns the spool to the unloading orientation, the handler unloads it
           onto a carrier pallet of the output conveyor, laser marking + weld-profile scan at the QC arch, the handler
           starts the next kit (flange to the station) and stores the finished spool in the stepped rack; the AGV arrives.

Every channel is a numpy array over scene frames (index = frame - 1).  Writes follow the stage-1 convention: a move
writes its frames and holds its end value to the end of the timeline, so a channel is scripted in chronological order.
Carried parts follow the gripper TCP through a relative transform frozen at the grasp frame (no jump), and after a
release they follow whatever holds them (fixed pose, positioner, carrier) through a transform frozen at that frame.

    P = plan2.solve()          # cached in stage2_logistics/out/plan2_<hash>.npz
    P["handler_q"], P["parts"]["flange"], P["events"]["load_contact"], ...
"""
import glob
import hashlib
import math
import os

import numpy as np

import tools  # noqa: F401
import kin as K
import layout2 as L2
from cell import layout as L
from cell import robot_build as RB

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = tools.DEMO
OUT = os.path.join(HERE, "out")
FPS = L.FPS
DEG = math.pi / 180.0
N_MAX = 9000

# joint speed limits used to time the moves (deg/s, conservative fractions of the real robots' maxima)
# average axis speeds used to time moves; with smoothstep timing the peak is 1.5x -> <= 6 deg/frame at 24 fps
HANDLER_VMAX = np.radians([90.0, 90.0, 90.0, 95.0, 95.0, 95.0])
VMAX_TCP = 1.6                            # m/s: TCP path speed and the swing speed of a carried part (0.8 m lever)
VMAX_TCP_EMPTY = 2.4                      # m/s: same with an empty gripper
HANDLER_VTRACK = 1.6                      # m/s carriage
TACK_VMAX = np.radians([95.0, 95.0, 95.0, 95.0, 95.0, 95.0])
LIN_SPEED = 0.45                          # m/s straight-line approach / retract moves (short); long lines up to 1.2 m/s
TRAVEL_Z = 1.55                           # TCP height for transfers with a part in the gripper
SWING_X = -4.80                           # carriage x where the handler swings a spool toward the cell opening
TUCK_R = 0.95                             # m: TCP distance from J1 in the folded travel pose
TUCK_DX = 1.0                             # m: carriage travels longer than this use the folded travel pose
STAGE1_N = 1104

PARTS = ("flange", "elbow", "pipe", "flange2")
GRIP_T = K.tr(z=L2.GRIP_TCP_Z)
TORCH_T = RB.tool_transform()


# ============================================================================ stage-1 arrays
def stage1_arrays():
    """Stage-1 per-frame arrays (index = stage-1 frame - 1): Q (welding robot), track, tilt, rot, arc_on, laser_on.
    Q/arc/laser come from the stage-1 IK cache (demo_video/out/anim_cache_<key>.npz, written by animation.build);
    the positioner / track schedules are recomputed exactly as animation.build does."""
    from cell import animation as A
    path = os.path.join(DEMO, "out", f"anim_cache_{A._cache_key()}.npz")
    if not os.path.exists(path):
        raise RuntimeError(f"stage-1 trajectory cache missing ({path}); run stage2_logistics/build2.py once "
                           f"(it builds the stage-1 animation) or demo_video/tests/t_solve.py")
    z = np.load(path)
    n = A.N_FRAMES
    tilt = np.zeros(n)
    rot = np.full(n, A.ROT_A0)
    track = np.full(n, L.ROBOT_HOME_Y)

    def ramp(arr, span, v0, v1, ease=True):
        a, b = span
        for f in range(a, b + 1):
            s = A.frac(f, span)
            s = A.smoothstep(s) if ease else s
            arr[f - 1] = v0 + (v1 - v0) * s
        arr[b:] = v1
    ramp(tilt, A.T["tilt_up"], 0.0, A.TILT_B1)
    ramp(rot, A.T["weld_A"], A.ROT_A0, A.ROT_A1, ease=False)
    ramp(tilt, A.T["index_180"], A.TILT_B1, A.TILT_B2)
    ramp(tilt, A.T["tilt_down"], A.TILT_B2, 0.0)
    ramp(track, A.T["track_to_A"], L.ROBOT_HOME_Y, A.TRACK_A)
    ramp(track, A.T["track_to_B1"], A.TRACK_A, A.TRACK_B1)
    ramp(track, A.T["track_to_B2"], A.TRACK_B1, A.TRACK_B2)
    ramp(track, A.T["track_home"], A.TRACK_B2, L.ROBOT_HOME_Y)
    return dict(Q=z["Q"].copy(), track=track, tilt=tilt, rot=rot, arc_on=z["arc_on"].copy(),
                laser_on=z["laser_on"].copy(), timeline=dict(A.T))


def spool_on_positioner(tilt_deg, rot_deg):
    from cell import animation as A
    return A.spool_world(tilt_deg, rot_deg)


# ============================================================================ frames of the stations
def station_frame():
    return K.planar_frame(L2.STATION_ORIGIN, L2.STATION_XDIR)


def carrier_frame(x):
    return K.planar_frame((x, L2.CONV_SPOOL_Y, L2.CARRIER["seat_z"]), (0.0, 1.0))


def grasp_frame(name):
    o, x, z, _ = L2.GRASP[name]
    return K.frame(x, z, o)


def handler_base(x):
    return K.tr(x, L2.HANDLER_TRACK_Y, L2.HANDLER_TRACK_TOP_Z) @ K.rotz(L2.HANDLER_YAW)


def tack_base():
    return K.tr(L2.TACK_BASE[0], L2.TACK_BASE[1], L2.TACK_BASE_Z) @ K.rotz(L2.TACK_YAW)


def _ramp(arr, f0, f1, v1, ease=K.smoothstep):
    """Write arr over frames f0..f1 from its value at f0 to v1 (eased), then hold v1 to the end."""
    v0 = np.array(arr[f0 - 1], dtype=float)
    v1 = np.asarray(v1, dtype=float)
    for f in range(f0, f1 + 1):
        s = ease((f - f0) / float(max(1, f1 - f0)))
        arr[f - 1] = v0 + (v1 - v0) * s
    arr[f1:] = v1


def _set_from(arr, f, v):
    arr[f - 1:] = v


# ============================================================================ planner
class Planner:
    def __init__(self):
        n = N_MAX
        self.n = n
        self.H = K.scaled_arm(L2.HANDLER_SCALE)
        self.TK = K.scaled_arm(L2.TACK_SCALE)
        self.hx = np.full(n, L2.HANDLER_HOME_X)
        self.hq = np.tile(np.array(L2.HANDLER_Q_HOME, float), (n, 1))
        self.hgrip = np.ones(n)
        self.tq = np.tile(np.array(L2.TACK_Q_HOME, float), (n, 1))
        self.tarc = np.zeros(n)
        self.wq = np.tile(np.array(L.ROBOT_Q_HOME, float), (n, 1))
        self.wtrack = np.full(n, L.ROBOT_HOME_Y)
        self.tilt = np.zeros(n)
        self.rot = np.full(n, L2.POS_LOAD_ROT)
        self.stud = np.zeros(n)
        self.clamps = {k: np.zeros(n) for k in L2.STATION_CLAMPS}
        self.stn_lamp = np.zeros(n)
        self.carrier_x = np.full(n, L2.CONV_LOAD_X)
        self.roller = np.zeros(n)
        self.qc_y = np.full(n, L2.QC_ARCH["y0"] + 0.25)
        self.marker = np.zeros(n)
        self.mark_reveal = np.zeros(n)
        self.mark_hot = np.zeros(n)
        self.tower = np.zeros(n)          # 0 off, 1 amber, 2 green
        self.display = np.zeros(n)        # 0 idle, 1 marking, 2 scanning, 3 OK
        self.muting = np.zeros(n)
        self.agv = np.tile(np.array([L2.AGV["path"][0][0], L2.AGV["path"][0][1], math.pi / 2]), (n, 1))
        self.agv_lift = np.zeros(n)
        self.agv_beacon = np.zeros(n)
        self.tack_s = {}
        self.tack_hot = {}
        for j, angs in (("A", L2.TACKS_A), ("B", L2.TACKS_B)):
            for i in range(len(angs)):
                self.tack_s[(j, i)] = np.zeros(n)
                self.tack_hot[(j, i)] = np.zeros(n)
        self.part_states = {p: [] for p in PARTS}
        self.events = {}
        self.iv = {k: [] for k in ("handler", "tack", "positioner", "clamp", "tack_arc", "conveyor", "qc_move",
                                   "marker", "scan", "agv", "stud", "gripper", "contact")}
        self.ik_fail = []
        self.hc = 1          # handler cursor (last frame of the previous handler action)
        self.loaded = False  # a part is in the gripper (slower transfers)
        self.tc = 1          # tack robot cursor
        self.tack_points = None

    # ------------------------------------------------------------------ helpers
    def ev(self, name, f):
        self.events[name] = int(f)
        return int(f)

    def h_state(self, f):
        return self.hx[f - 1], self.hq[f - 1].copy()

    def h_tcp(self, f):
        x, q = self.h_state(f)
        return self.H.fk(q, handler_base(x), GRIP_T)

    def h_ik(self, T, x, seed, f=0, tag=""):
        """6-DoF IK for the gripper TCP; tries T and T rotated 180 deg about the tool Z (symmetric tongs)."""
        B = handler_base(x)
        p = np.linalg.inv(B) @ T[:, 3]
        j1 = math.atan2(p[1], p[0])
        seeds = [np.asarray(seed, float)]
        for base in ((0.35, 0.2, 0.0, 1.0, 0.0), (0.6, -0.2, 0.0, 1.2, 0.0), (0.1, 0.5, 0.0, 0.9, 0.0),
                     (0.8, -0.5, 0.0, 1.1, 0.0), (-0.2, 0.7, 0.0, 1.0, 0.0)):
            seeds.append(np.array([j1, *base]))
        best = None
        for flip in (False, True):
            TT = T @ K.rotz(math.pi) if flip else T
            q, ok, err = K.ik6(self.H, TT, seeds[0], B, GRIP_T, seeds[1:], iters=300)
            q = K.unwrap_to(q, seed)
            cost = (0 if ok else 100) + float(np.sum(np.abs(q - np.asarray(seed, float)) * [1.0, 1, 1, 0.6, 0.6, 0.4]))
            if best is None or cost < best[0]:
                best = (cost, q, ok, err)
        if not best[2]:
            self.ik_fail.append(("handler", f, tag, float(best[3])))
        return best[1]

    def h_move(self, x1, q1, via=(), dur=None, min_dur=10, tag=""):
        """Joint-space move (piecewise linear through via (x, q) points, smoothstep timing) from the cursor."""
        f0 = self.hc
        x0, q0 = self.h_state(f0)
        pts = [(x0, q0)] + [(vx, K.unwrap_to(vq, q0)) for vx, vq in via] + [(x1, K.unwrap_to(q1, q0))]
        # time share of each segment ~ its slowest axis (joint or carriage) at the speed limits
        cost = []
        for i in range(len(pts) - 1):
            (xa, qa), (xb, qb) = pts[i], pts[i + 1]
            Ts = [self.H.fk(qa + (qb - qa) * u, handler_base(xa + (xb - xa) * u), GRIP_T) for u in np.linspace(0, 1, 7)]
            path = sum(float(np.linalg.norm(Ts[k + 1][:3, 3] - Ts[k][:3, 3])) for k in range(6))
            swing = sum(K.rot_angle(Ts[k], Ts[k + 1]) for k in range(6)) * 0.8
            vt = VMAX_TCP if self.loaded else VMAX_TCP_EMPTY
            cost.append(max(float(np.max(np.abs(qb - qa) / HANDLER_VMAX)), abs(xb - xa) / HANDLER_VTRACK,
                            path / vt, swing / vt) + 1e-4)
        ctot = sum(cost)
        if dur is None:
            dur = max(min_dur, int(math.ceil(ctot * 1.5 * FPS)))
        f1 = f0 + dur
        cum = np.concatenate([[0.0], np.cumsum(cost)])
        m = len(pts) - 1
        for f in range(f0, f1 + 1):
            c = K.smoothstep((f - f0) / float(dur)) * ctot
            i = min(int(np.searchsorted(cum, c, side="right")) - 1, m - 1)
            u = min(1.0, max(0.0, (c - cum[i]) / cost[i]))
            self.hx[f - 1] = pts[i][0] * (1 - u) + pts[i + 1][0] * u
            self.hq[f - 1] = pts[i][1] * (1 - u) + pts[i + 1][1] * u
        self.hx[f1:] = pts[-1][0]
        self.hq[f1:] = pts[-1][1]
        self.iv["handler"].append((f0, f1))
        self.hc = f1
        return f1

    def h_line(self, T1, x1=None, dur=None, min_dur=8, tag=""):
        """Straight-line TCP move (orientation slerp) from the current pose to T1; the carriage moves linearly to x1."""
        f0 = self.hc
        x0, q = self.h_state(f0)
        x1 = x0 if x1 is None else x1
        T0 = self.h_tcp(f0)
        if dur is None:
            d = float(np.linalg.norm(T1[:3, 3] - T0[:3, 3]))
            a = K.rot_angle(T0, T1)
            v = min(1.2, LIN_SPEED + 0.4 * d)
            dur = max(min_dur, int(math.ceil(1.5 * max(d / v, a / math.radians(90), abs(x1 - x0) / HANDLER_VTRACK) * FPS)))
        f1 = f0 + dur
        for f in range(f0 + 1, f1 + 1):
            s = K.smoothstep((f - f0) / float(dur))
            x = x0 + (x1 - x0) * s
            T = K.blend_pose(T0, T1, s)
            q = self.h_ik_near(T, x, q, f, tag)
            self.hx[f - 1] = x
            self.hq[f - 1] = q
        self.hx[f1:] = self.hx[f1 - 1]
        self.hq[f1:] = self.hq[f1 - 1]
        self.iv["handler"].append((f0, f1))
        self.hc = f1
        return f1

    def h_ik_near(self, T, x, q_prev, f, tag):
        q, ok, err = self.H.ik(T, q_prev, handler_base(x), GRIP_T, iters=120)
        if not ok:
            q, ok, err = K.ik6(self.H, T, q_prev, handler_base(x), GRIP_T, [q_prev + 0.05], iters=400)
        if not ok:
            self.ik_fail.append(("handler_line", f, tag, float(err)))
        return K.unwrap_to(q, q_prev)

    def h_wait(self, n):
        self.hc += n
        return self.hc

    def h_grip(self, s1, dur=8):
        f0 = self.hc
        _ramp(self.hgrip, f0, f0 + dur, s1)
        self.iv["gripper"].append((f0, f0 + dur))
        self.hc = f0 + dur
        return self.hc

    # ------------------------------------------------------------------ part bookkeeping
    def part_pose_fixed(self, part, f, T):
        self.part_states[part].append((f, "fixed", np.array(T)))

    def attach(self, part, f):
        """From frame f on, `part` follows the handler TCP (relative transform frozen at f)."""
        self.part_states[part].append((f, "tcp", None))
        self.loaded = True

    def release(self, part, f, holder="fixed", T_nominal=None):
        """From frame f on, the part rests at its nominal placement pose T_nominal (world) and stays there (holder
        'fixed') or follows the positioner / the carrier through the relative transform frozen at f.  Snapping to the
        nominal pose removes the sub-millimetre IK residual so later grasps hit the part exactly."""
        self.part_states[part].append((f, holder, None if T_nominal is None else np.array(T_nominal)))
        self.loaded = False

    # ------------------------------------------------------------------ handler composite actions
    def above(self, T, dz):
        return K.tr(z=dz) @ T

    def pick(self, part, T_part, grasp, x, lift=0.25, approach_dur=None, travel_via=True, tag="", not_before=0):
        """Move above the grasp, (wait until not_before), descend, close, lift.  Returns the frame after the lift."""
        G = T_part @ grasp_frame(grasp)
        seed = self.hq[self.hc - 1]
        q_up = self.h_ik(self.above(G, lift), x, seed, self.hc, tag + ":above")
        via = []
        if travel_via:
            via = self._travel_via(x, q_up)
        self.h_move(x, q_up, via=via, dur=approach_dur, tag=tag)
        if self.hc < not_before:
            self.hc = not_before
        w0 = self.hc
        self.h_line(G, tag=tag + ":down")
        self.ev(f"{tag}_grip", self.hc)
        self.h_grip(L2.GRASP[grasp][3])
        self.attach(part, self.hc)
        self.h_line(self.above(G, lift), tag=tag + ":lift")
        self.iv["contact"].append((w0, self.hc))
        return self.hc

    def place(self, parts, T_part_target, grasp, x, lift=0.25, holder="fixed", tag="",
              approach_dur=None, travel_via=True, open_to=1.0):
        """Carry to above the target, descend, open, retract.  parts: list of attached part names."""
        G = T_part_target @ grasp_frame(grasp)
        seed = self.hq[self.hc - 1]
        q_up = self.h_ik(self.above(G, lift), x, seed, self.hc, tag + ":above")
        via = self._travel_via(x, q_up) if travel_via else []
        self.h_move(x, q_up, via=via, dur=approach_dur, tag=tag)
        w0 = self.hc
        self.h_line(G, tag=tag + ":down")
        self.ev(f"{tag}_contact", self.hc)
        for p in parts:
            self.release(p, self.hc, holder, T_part_target)
        self.h_grip(open_to)
        self.h_line(self.above(G, lift), tag=tag + ":retract")
        self.iv["contact"].append((w0, self.hc))
        return self.hc

    def _tuck(self, x, phi, z, seed, f):
        """Travel ('tucked') pose: TCP TUCK_R in front of the J1 axis along the track direction phi (0: +X, pi: -X),
        at height z, gripper down -> the arm and a carried part stay above the track, clear of the side stations."""
        d = np.array([math.cos(phi), math.sin(phi), 0.0])
        T = K.frame(d, (0.0, 0.0, -1.0), np.array([x, L2.HANDLER_TRACK_Y, z]) + TUCK_R * d)
        return self.h_ik(T, x, seed, f, "tuck")

    def _travel_via(self, x_target, q_target):
        """Via points for a transfer.  Long carriage travels (> TUCK_DX): lift, fold the arm along the track axis,
        travel, unfold above the target.  Short ones: lift to the travel height, move, come down onto the target
        approach pose."""
        f = self.hc
        x0, q0 = self.h_state(f)
        T0 = self.h_tcp(f)
        Tt = self.H.fk(q_target, handler_base(x_target), GRIP_T)
        via = []
        if abs(x_target - x0) > TUCK_DX:
            z_travel = max(TRAVEL_Z, T0[2, 3], Tt[2, 3])
            Tup = T0.copy()
            Tup[2, 3] = z_travel
            q_up = self.h_ik(Tup, x0, q0, f, "via_up")
            # fold toward the travel direction; for -X travel pick the +-180 side of the target so that the final
            # unfold at the target is short (J1 cannot cross +-180 deg): the long turn happens at the source instead
            if x_target > x0:
                phi = 0.0
            else:
                phi = math.pi - 0.03 if q_target[0] >= 0 else -math.pi + 0.03
            # fold in the current direction, then turn J1 only (constant radius), then travel
            q_tk0 = self._tuck(x0, q0[0] + L2.HANDLER_YAW, z_travel, q_up, f)
            q_tk = q_tk0.copy()
            q_tk[0] = phi - L2.HANDLER_YAW
            via += [(x0, q_up), (x0, q_tk0), (x0, q_tk), (x_target, q_tk)]
            Ttop = Tt.copy()
            Ttop[2, 3] = z_travel
            via.append((x_target, self.h_ik(Ttop, x_target, q_tk, f, "via_top")))
            return via
        if T0[2, 3] < TRAVEL_Z - 0.05 or abs(x_target - x0) > 0.4:
            z_travel = max(TRAVEL_Z, T0[2, 3], Tt[2, 3])      # climb first, then travel, then descend
            Tup = T0.copy()
            Tup[2, 3] = z_travel
            q_up = self.h_ik(Tup, x0, q0, f, "via_up")
            via.append((x0, q_up))
            Ttop = Tt.copy()
            Ttop[2, 3] = z_travel
            q_top = self.h_ik(Ttop, x_target, q_up, f, "via_top")
            via.append((x_target, q_top))
        return via

    # ------------------------------------------------------------------ tack robot
    def t_state(self, f):
        return self.tq[f - 1].copy()

    def t_move(self, q1, dur=None, min_dur=10):
        f0 = self.tc
        q0 = self.t_state(f0)
        q1 = K.unwrap_to(q1, q0)
        if dur is None:
            t = float(np.max(np.abs(q1 - q0) / TACK_VMAX)) * 1.5
            dur = max(min_dur, int(math.ceil(t * FPS)))
        _ramp(self.tq, f0, f0 + dur, q1)
        self.iv["tack"].append((f0, f0 + dur))
        self.tc = f0 + dur
        return self.tc

    def solve_tacks(self):
        """Torch targets for the 6 tacks: for each nominal angle try angle offsets / torch tilts / leans until the
        5-DoF IK (free spin about the torch axis) converges.  Returns list of dict(key, T, q, q_back)."""
        from cell import animation as A
        Fs = station_frame()
        R = L2.PIPE_R
        B = tack_base()
        rng = np.random.default_rng(11)
        out = []
        q_prev = np.array(L2.TACK_Q_HOME, float)
        # seam B first: its top tack has no room to back the torch off far, so it is approached straight from the
        # home pose (from above); the moves between the other tacks go through clear poses 0.12-0.20 m off the seam
        specs = [("B", i, b) for i, b in enumerate(L2.TACKS_B)] + [("A", i, a) for i, a in enumerate(L2.TACKS_A)]
        for joint, i, nominal in specs:
            found = None
            for dang in (0.0, 10.0, -10.0, 20.0, -20.0):
                ang = math.radians(nominal + dang)
                if joint == "A":
                    n_l = np.array([math.cos(ang), math.sin(ang), 0.0])
                    p_l = np.array([0.0, 0.0, L2.SEAM_A_Z]) + R * n_l
                    t_l = np.array([-n_l[1], n_l[0], 0.0])
                    bias = np.array([0.0, 0.0, 1.0])
                    ups = (1.0, 0.5, 2.0)
                else:
                    n_l = np.array([0.0, math.cos(ang), math.sin(ang)])
                    p_l = np.array([L2.SEAM_B_X, 0.0, L2.PIPE_AXIS_Z]) + R * n_l
                    t_l = np.array([0.0, -n_l[2], n_l[1]])
                    bias = np.array([1.0, 0.0, 0.0])
                    ups = (0.35, 0.7, 0.0)
                p = (Fs @ np.r_[p_l, 1.0])[:3]
                for up in ups:
                    n = Fs[:3, :3] @ (n_l + up * bias)
                    n /= np.linalg.norm(n)
                    tt = Fs[:3, :3] @ t_l
                    lean = np.array([L2.TACK_BASE[0] - p[0], L2.TACK_BASE[1] - p[1], 0.0])
                    lean = lean / np.linalg.norm(lean) + np.array([0.0, 0.0, 0.9])
                    Tt = A.target_frame(p, n, tt, lean, push_deg=0.0)
                    pb = np.linalg.inv(B) @ np.r_[p, 1.0]
                    j1 = math.atan2(pb[1], pb[0])
                    seeds = [q_prev] + [np.array([j1, rng.uniform(-0.6, 0.8), rng.uniform(-0.3, 1.0), rng.uniform(-1, 1),
                                                  rng.uniform(0.3, 1.6), 0.0]) for _ in range(10)]
                    for q0 in seeds:
                        q, ok, err = self.TK.ik(Tt, q0, B, TORCH_T, iters=250, free_spin=True, q_ref=[j1, 0, 0, 0, 1.2, 0])
                        if ok:
                            Tb = K.tr(*(-0.07 * Tt[:3, 2])) @ Tt           # 70 mm back along the torch axis
                            qb, okb, _ = self.TK.ik(Tb, q, B, TORCH_T, iters=200, free_spin=True, q_ref=q)
                            okc = False
                            for dc in (0.20, 0.16, 0.12):                  # clear pose for the moves between tacks
                                Tc = K.tr(*(-dc * Tt[:3, 2])) @ Tt
                                qc, okc, _ = self.TK.ik(Tc, qb, B, TORCH_T, iters=200, free_spin=True, q_ref=qb)
                                if okc:
                                    break
                            if okb and not okc:                            # no room straight back: back off and up
                                for dc, up in ((0.07, 0.12), (0.07, 0.20), (0.10, 0.10), (0.05, 0.08)):
                                    Tc = K.tr(*(-dc * Tt[:3, 2] + np.array([0.0, 0.0, up]))) @ Tt
                                    qc, okc, _ = self.TK.ik(Tc, qb, B, TORCH_T, iters=200, free_spin=True, q_ref=qb)
                                    if okc:
                                        break
                                if not okc:
                                    qc, okc = qb, True
                            if okb and okc:
                                # entry from above for the first tack (the home pose is behind the elbow)
                                Th = K.tr(0.0, 0.0, 0.25) @ K.tr(*(-0.07 * Tt[:3, 2])) @ Tt
                                qh, okh, _ = self.TK.ik(Th, qb, B, TORCH_T, iters=200, free_spin=True, q_ref=qb)
                                found = dict(key=(joint, i), T=Tt, q=K.unwrap_to(q, q_prev), q_back=K.unwrap_to(qb, q_prev),
                                             q_clear=K.unwrap_to(qc, q_prev), q_high=K.unwrap_to(qh, q_prev) if okh else None,
                                             angle=nominal + dang)
                                break
                    if found:
                        break
                if found:
                    break
            if not found:
                self.ik_fail.append(("tack", 0, f"{joint}{i}", 1.0))
                continue
            q_prev = found["q"]
            out.append(found)
        return out

    def tack_sequence(self, f_start):
        """Tack robot: home -> 6 tacks (back, in, arc, out) -> home.  Returns the end frame."""
        if self.tack_points is None:
            self.tack_points = self.solve_tacks()
        self.tc = f_start
        self.ev("tack_start", f_start)
        for k, tp in enumerate(self.tack_points):
            if k == 0 and tp.get("q_high") is not None:
                self.t_move(tp["q_high"], min_dur=12)
            self.t_move(tp["q_clear"], min_dur=10)
            self.t_move(tp["q_back"], min_dur=6)
            self.t_move(tp["q"], min_dur=6)
            a0 = self.tc + 1
            a1 = a0 + 8
            self.tarc[a0 - 1:a1] = 1.0
            self.iv["tack_arc"].append((a0, a1))
            s = self.tack_s[tp["key"]]
            h = self.tack_hot[tp["key"]]
            _ramp(s, a0, a1, 1.0, ease=lambda t: t)
            _set_from(h, a0, 1.0)
            _ramp(h, a1, a1 + 40, 0.0, ease=lambda t: t)
            if k == 0:
                self.ev("first_tack", a0)
            self.ev(f"tack_{tp['key'][0]}{tp['key'][1]}", a0)
            self.tc = a1 + 1
            self.t_move(tp["q_back"], min_dur=6)
            self.t_move(tp["q_clear"], min_dur=6)
        self.ev("last_tack_end", self.tc)
        self.t_move(np.array(L2.TACK_Q_HOME, float), min_dur=18)
        self.ev("tack_home", self.tc)
        return self.tc

    # ------------------------------------------------------------------ station / conveyor / QC helpers
    def clamp(self, names, s1, f0, dur=10):
        for nme in names:
            _ramp(self.clamps[nme], f0, f0 + dur, s1)
        self.iv["clamp"].append((f0, f0 + dur))
        return f0 + dur

    def carrier_move(self, x1, f0):
        x0 = self.carrier_x[f0 - 1]
        d = abs(x1 - x0)
        dur = max(24, int(math.ceil((d / L2.CONV_SPEED + 0.8) * FPS)))
        _ramp(self.carrier_x, f0, f0 + dur, x1)
        for f in range(f0, f0 + dur + 1):
            self.roller[f - 1] = self.roller[f0 - 2] + abs(self.carrier_x[f - 1] - x0)
        self.roller[f0 + dur:] = self.roller[f0 + dur - 1]
        self.iv["conveyor"].append((f0, f0 + dur))
        return f0 + dur

    def qc_move(self, y1, f0, dur=None):
        y0 = self.qc_y[f0 - 1]
        if dur is None:
            dur = max(12, int(math.ceil(1.5 * abs(y1 - y0) / 0.4 * FPS)))
        _ramp(self.qc_y, f0, f0 + dur, y1)
        self.iv["qc_move"].append((f0, f0 + dur))
        return f0 + dur


# ============================================================================ the script
def _script(P, S1):
    H = P.H
    Fs = station_frame()
    home_q = np.array(L2.HANDLER_Q_HOME, float)

    # ---------------------------------------------------------------- initial state
    kit_f = [K.planar_frame(o, x) for o, x in L2.KIT_FLANGES]
    kit_e = [K.planar_frame(o, x) for o, x in L2.KIT_ELBOWS]
    kit_p = [K.planar_frame(*L2.pipe_buffer_frame(i)) for i in range(len(L2.PIPE_BUFFER_X))]
    P.part_pose_fixed("flange", 1, kit_f[0])
    P.part_pose_fixed("elbow", 1, kit_e[0])
    P.part_pose_fixed("pipe", 1, kit_p[0])
    P.part_pose_fixed("flange2", 1, kit_f[1])
    P.ev("start", 1)

    # ---------------------------------------------------------------- kitting: flange
    P.hc = 20
    P.pick("flange", kit_f[0], "flange", L2.KIT_FLANGES[0][0][0], tag="kit_flange")
    P.place(["flange"], Fs, "flange", L2.STATION_ORIGIN[0], tag="stn_flange")
    P.clamp(["flange_l", "flange_r"], 1.0, P.hc + 2)
    P.ev("clamp_flange", P.hc + 2)
    # ---------------------------------------------------------------- elbow
    P.pick("elbow", kit_e[0], "elbow", L2.KIT_ELBOWS[0][0][0], lift=0.30, tag="kit_elbow")
    P.place(["elbow"], Fs, "elbow", L2.STATION_ORIGIN[0], lift=0.30, tag="stn_elbow")
    P.clamp(["elbow"], 1.0, P.hc + 2)
    P.ev("clamp_elbow", P.hc + 2)
    # ---------------------------------------------------------------- pipe
    P.pick("pipe", kit_p[0], "pipe", L2.PIPE_BUFFER_X[0], tag="buf_pipe")
    P.place(["pipe"], Fs, "pipe", L2.STATION_ORIGIN[0], tag="stn_pipe")
    f = P.clamp(["pipe"], 1.0, P.hc + 2)
    P.ev("clamp_pipe", P.hc + 2)
    _set_from(P.stn_lamp, f + 6, 1.0)
    P.ev("gap_ok", f + 6)
    # handler steps back to a waiting pose next to the station (clear of the tack robot)
    q_wait = P.h_ik(K.tr(L2.STATION_ORIGIN[0] - 0.9, 1.1, TRAVEL_Z) @ K.rotx(math.pi), L2.STATION_ORIGIN[0] - 0.9,
                    P.hq[P.hc - 1], P.hc, "wait_station")
    P.h_move(L2.STATION_ORIGIN[0] - 0.9, q_wait)
    # ---------------------------------------------------------------- tacking
    t_end = P.tack_sequence(max(P.hc, f + 8) + 2)
    f = P.clamp(list(L2.STATION_CLAMPS), 0.0, t_end - 12)
    P.ev("unclamp", t_end - 12)
    _set_from(P.stn_lamp, t_end - 12, 0.0)
    # ---------------------------------------------------------------- pick the tacked spool
    P.hc = max(P.hc, t_end - 30)
    P.pick("pipe", Fs, "spool", L2.STATION_ORIGIN[0], lift=0.35, tag="stn_spool")
    for p in ("flange", "elbow"):
        P.attach(p, P.events["stn_spool_grip"] + 8)
    # swing toward the opening at a safe radius, then enter along X
    x_st = L2.STATION_ORIGIN[0]
    x_sw = SWING_X
    Tpos = spool_on_positioner(0.0, L2.POS_LOAD_ROT)
    G_pos = Tpos @ grasp_frame("spool")
    G_app = P.above(G_pos, L2.POS_APPROACH_DZ)
    G_pre = G_app.copy()
    G_pre[0, 3] = x_sw + 1.40                                 # outside the fence line: x = -3.4
    G_hi = G_pre.copy()
    q_hi_station = P.h_ik(K.tr(0, -0.55, 0) @ K.tr(z=2.30 - (Fs @ grasp_frame("spool"))[2, 3]) @ Fs @ grasp_frame("spool"),
                          x_st, P.hq[P.hc - 1], P.hc, "lift_high")
    P.h_move(x_st, q_hi_station, min_dur=24)
    q_pre = P.h_ik(G_hi, x_sw, q_hi_station, P.hc, "pre_entry")
    P.h_move(x_sw, q_pre, min_dur=30)
    P.ev("pre_entry", P.hc)
    _set_from(P.muting, P.hc - 6, 1.0)
    P.ev("muting_on", P.hc - 6)
    P.h_line(G_app, x1=L2.HANDLER_X_AT_POSITIONER, tag="entry")
    P.ev("entry_done", P.hc)
    w0 = P.hc
    P.h_line(G_pos, tag="lower_on_faceplate")
    fc = P.ev("load_contact", P.hc)
    for p in ("flange", "elbow", "pipe"):
        P.release(p, fc, "positioner", Tpos)
    _ramp(P.stud, fc + 2, fc + 14, 1.0)
    P.iv["stud"].append((fc + 2, fc + 14))
    P.ev("studs_up", fc + 14)
    P.hc = fc + 14
    P.h_grip(1.0)
    P.h_line(G_app, tag="unload_retract")
    P.iv["contact"].append((w0, P.hc))
    P.h_line(G_pre, x1=x_sw, tag="exit")
    f_out = P.ev("handler_out", P.hc)
    _set_from(P.muting, f_out + 4, 0.0)
    # positioner turns the spool into the stage-1 start orientation (ROT_A0) once the gripper left the sweep circle
    from cell import animation as A
    fr0 = f_out - 20
    fr1 = fr0 + 60
    _ramp(P.rot, fr0, fr1, A.ROT_A0)
    P.iv["positioner"].append((fr0, fr1))
    P.ev("pos_rotate_start", fr0)
    P.ev("ready", fr1)
    # handler parks (ready pose for the unloading) while welding
    q_idle = P.h_ik(K.tr(-4.4 + 1.2, 0.0, 2.0) @ K.rotx(math.pi) @ K.rotz(math.pi), -4.4, P.hq[P.hc - 1], P.hc, "idle")
    P.h_move(-4.4, q_idle)
    P.ev("handler_idle", P.hc)

    # ---------------------------------------------------------------- embed stage 1
    off = fr1 - 90
    P.weld_offset = off
    for arr, src in ((P.wq, S1["Q"]), (P.wtrack, S1["track"]), (P.tilt, S1["tilt"]), (P.rot, S1["rot"])):
        a = off + 97
        arr[a - 1:off + STAGE1_N] = src[96:STAGE1_N]
        arr[off + STAGE1_N:] = src[STAGE1_N - 1]
    P.ev("weld_first", off + 97)
    P.ev("weld_last", off + 930)

    # ---------------------------------------------------------------- post: unloading
    f_tilt0 = off + 1060
    P.ev("tilt_home", f_tilt0)
    _ramp(P.rot, f_tilt0 + 2, f_tilt0 + 50, L2.POS_UNLOAD_ROT)
    P.iv["positioner"].append((f_tilt0 + 2, f_tilt0 + 50))
    f_unl = P.ev("unload_orient", f_tilt0 + 50)
    # handler: approach the opening while the positioner turns (stays outside the fence line)
    P.hc = max(P.hc, f_tilt0 - 40)
    q_pre2 = P.h_ik(G_pre, x_sw, P.hq[P.hc - 1], P.hc, "pre_entry2")
    P.h_move(x_sw, q_pre2)
    P.hc = max(P.hc, f_unl + 2)
    _set_from(P.muting, P.hc - 4, 1.0)
    P.h_line(G_app, x1=L2.HANDLER_X_AT_POSITIONER, tag="entry2")
    w0 = P.hc
    P.h_line(G_pos, tag="descend_unload")
    fg = P.ev("unload_grip", P.hc)
    P.h_grip(0.0)
    for p in ("flange", "elbow", "pipe"):
        P.attach(p, P.hc + 12)
    _ramp(P.stud, P.hc, P.hc + 12, 0.0)
    P.iv["stud"].append((P.hc, P.hc + 12))
    P.hc += 12
    P.ev("studs_down", P.hc)
    P.h_line(G_app, tag="lift_off")
    P.iv["contact"].append((w0, P.hc))
    P.h_line(G_pre, x1=x_sw, tag="exit2")
    _set_from(P.muting, P.hc + 4, 0.0)
    P.ev("handler_out2", P.hc)
    # swing toward the conveyor side at the retracted carriage position (the spool stays clear of the fence posts)
    q_sw = P.hq[P.hc - 1].copy()
    q_sw[0] -= math.radians(75.0)
    q_sw[5] -= math.radians(75.0)
    P.h_move(x_sw, q_sw)
    # ---------------------------------------------------------------- onto the carrier pallet
    Fc = carrier_frame(L2.CONV_LOAD_X)
    P.place(["flange", "elbow", "pipe"], Fc, "spool", L2.CONV_LOAD_X, lift=0.25, holder="carrier", tag="conv_load")
    f_conv = P.hc + 4
    # ---------------------------------------------------------------- conveyor -> QC arch
    f = P.carrier_move(L2.CONV_QC_X, f_conv)
    P.ev("conv_start", f_conv)
    P.ev("at_qc", f)
    _set_from(P.tower, f_conv, 1.0)
    mark_y = L2.CONV_SPOOL_Y + L2.QC_MARK_LOCAL[0]
    scan_y0 = L2.CONV_SPOOL_Y + L2.SEAM_B_X + 0.12
    scan_y1 = L2.CONV_SPOOL_Y + L2.SEAM_B_X - 0.10
    fq = P.qc_move(mark_y, f - 30)
    f = max(f, fq) + 4
    P.ev("mark_start", f)
    _set_from(P.display, f, 1.0)
    _set_from(P.marker, f, 1.0)
    _ramp(P.mark_reveal, f, f + 48, 1.0, ease=lambda t: t)
    _set_from(P.mark_hot, f, 1.0)
    _set_from(P.marker, f + 48, 0.0)
    _ramp(P.mark_hot, f + 48, f + 80, 0.0, ease=lambda t: t)
    P.iv["marker"].append((f, f + 48))
    f += 48
    P.ev("mark_end", f)
    f = P.qc_move(scan_y0, f + 2, dur=16)
    _set_from(P.display, f, 2.0)
    P.ev("scan_start", f)
    f1 = P.qc_move(scan_y1, f + 2, dur=64)
    P.iv["scan"].append((f + 1, f1))
    P.ev("scan_end", f1)
    _set_from(P.display, f1 + 2, 3.0)
    _set_from(P.tower, f1 + 2, 2.0)
    P.ev("qc_ok", f1 + 2)
    fq = P.qc_move(L2.QC_ARCH["y0"] + 0.25, f1 + 20)
    f = P.carrier_move(L2.CONV_END_X, f1 + 30)
    _set_from(P.tower, f1 + 30, 0.0)
    _set_from(P.display, f1 + 60, 0.0)
    P.ev("at_end", f)
    f_at_end = f

    # ---------------------------------------------------------------- meanwhile: next kit (flange 2 -> station)
    P.hc = P.hc + 2
    P.pick("flange2", kit_f[1], "flange", L2.KIT_FLANGES[1][0][0], tag="kit_flange2")
    P.place(["flange2"], Fs, "flange", L2.STATION_ORIGIN[0], tag="stn_flange2")
    P.clamp(["flange_l", "flange_r"], 1.0, P.hc + 2)
    P.ev("clamp_flange2", P.hc + 2)
    # ---------------------------------------------------------------- storage
    Fend = carrier_frame(L2.CONV_END_X)
    P.pick("pipe", Fend, "spool", L2.CONV_END_X, lift=0.30, tag="conv_end", not_before=f_at_end + 6)
    for p in ("flange", "elbow"):
        P.attach(p, P.events["conv_end_grip"] + 8)
    t, b = L2.STORAGE_TARGET
    Fb = K.planar_frame(*L2.storage_frame(t, b))
    # vertical descent between the tier-1 V-posts: approach from above their heads
    P.place(["flange", "elbow", "pipe"], Fb, "spool", L2.HANDLER_TRACK_X[0], lift=1.15, tag="store")
    # next cycle: fold, turn toward the kit pallet and go for the second elbow (the video ends on the way)
    q_f = P._tuck(L2.HANDLER_TRACK_X[0], 0.0, TRAVEL_Z, P.hq[P.hc - 1], P.hc)
    P.h_move(L2.HANDLER_TRACK_X[0], q_f)
    Te = K.planar_frame(*L2.KIT_ELBOWS[1]) @ grasp_frame("elbow")
    q_e = P.h_ik(P.above(Te, 0.30), L2.KIT_ELBOWS[1][0][0], q_f, P.hc, "next_elbow")
    P.h_move(L2.KIT_ELBOWS[1][0][0], q_e)
    P.ev("handler_home_end", P.hc)
    # ---------------------------------------------------------------- AGV arrives behind the rack
    fa = P.events["store_contact"] - 40
    (x0, y0), (x1, y1) = L2.AGV["path"][0], L2.AGV["path"][1]
    dur = int(math.ceil(abs(y1 - y0) / 0.9 * FPS * 1.3))
    _ramp(P.agv, fa, fa + dur, np.array([x1, y1, math.pi / 2]))
    _set_from(P.agv_beacon, fa - 12, 1.0)
    P.iv["agv"].append((fa, fa + dur))
    P.ev("agv_arrive", fa + dur)
    _set_from(P.agv_beacon, fa + dur + 24, 0.0)
    end = max(P.hc, fa + dur + 40)
    P.ev("end", end)
    return end


# ============================================================================ part poses
def _part_poses(P, n):
    """Evaluate every part's world pose per frame from its state list."""
    out = {}
    tcp = np.array([P.H.fk(P.hq[i], handler_base(P.hx[i]), GRIP_T) for i in range(n)])
    for part, states in P.part_states.items():
        states = sorted(states, key=lambda s: s[0])
        poses = np.zeros((n, 4, 4))
        cur = None
        si = 0
        for i in range(n):
            f = i + 1
            while si < len(states) and states[si][0] <= f:
                fs, kind, data = states[si]
                prev = poses[i - 1] if i > 0 else None
                ref = data if data is not None else prev
                if kind == "fixed":
                    cur = ("fixed", ref)
                elif kind == "tcp":
                    cur = ("tcp", np.linalg.inv(tcp[i]) @ prev)
                elif kind == "positioner":
                    Tp = spool_on_positioner(P.tilt[i], P.rot[i])
                    cur = ("positioner", np.linalg.inv(Tp) @ ref)
                elif kind == "carrier":
                    Tc = carrier_frame(P.carrier_x[i])
                    cur = ("carrier", np.linalg.inv(Tc) @ ref)
                si += 1
            kind = cur[0]
            if kind == "fixed":
                poses[i] = cur[1]
            elif kind == "tcp":
                poses[i] = tcp[i] @ cur[1]
            elif kind == "positioner":
                poses[i] = spool_on_positioner(P.tilt[i], P.rot[i]) @ cur[1]
            elif kind == "carrier":
                poses[i] = carrier_frame(P.carrier_x[i]) @ cur[1]
        out[part] = poses
    return out


# ============================================================================ public
DATA = os.path.join(HERE, "data")          # the plan shipped with the repository (same code -> same file name)


def _hash():
    """Hash of every source the plan depends on, line endings normalised (a Windows checkout with CRLF gets the same
    hash, so it loads the shipped plan instead of re-solving - the render and the edit then use identical frames)."""
    h = hashlib.sha1()
    files = [os.path.join(HERE, fn) for fn in ("plan2.py", "layout2.py", "kin.py")]
    files += [os.path.join(DEMO, "cell", fn) for fn in ("animation.py", "layout.py", "robot_urdf.py", "robot_build.py")]
    files.append(os.path.join(DEMO, "assets", "abb_irb4600_40_255", "urdf", "robot_description.urdf"))
    for fn in files:
        with open(fn, "rb") as fh:
            h.update(fh.read().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]


def solve(use_cache=True, verbose=True):
    """Run (or load) the plan.  Returns a dict of numpy arrays + events / intervals / metadata.
    Looks for plan2_<hash>.npz in stage2_logistics/data (shipped) and stage2_logistics/out (local cache)."""
    os.makedirs(OUT, exist_ok=True)
    name = f"plan2_{_hash()}.npz"
    path = os.path.join(OUT, name)
    if use_cache:
        for p in (os.path.join(DATA, name), path):
            if os.path.exists(p):
                z = np.load(p, allow_pickle=True)
                return z["plan"].item()
    S1 = stage1_arrays()
    P = Planner()
    end = _script(P, S1)
    n = end
    parts = _part_poses(P, n)
    plan = dict(
        n_frames=n, weld_offset=P.weld_offset,
        handler_x=P.hx[:n], handler_q=P.hq[:n], handler_grip=P.hgrip[:n],
        tack_q=P.tq[:n], tack_arc=P.tarc[:n],
        welder_q=P.wq[:n], welder_track=P.wtrack[:n], tilt=P.tilt[:n], rot=P.rot[:n], stud=P.stud[:n],
        clamps={k: v[:n] for k, v in P.clamps.items()}, station_lamp=P.stn_lamp[:n],
        carrier_x=P.carrier_x[:n], roller=P.roller[:n],
        qc_y=P.qc_y[:n], marker=P.marker[:n], mark_reveal=P.mark_reveal[:n], mark_hot=P.mark_hot[:n],
        tower=P.tower[:n], display=P.display[:n], muting=P.muting[:n],
        agv=P.agv[:n], agv_lift=P.agv_lift[:n], agv_beacon=P.agv_beacon[:n],
        tack_s={f"{k[0]}{k[1]}": v[:n] for k, v in P.tack_s.items()},
        tack_hot={f"{k[0]}{k[1]}": v[:n] for k, v in P.tack_hot.items()},
        tack_points=[dict(key=f"{t['key'][0]}{t['key'][1]}", angle=t["angle"]) for t in (P.tack_points or [])],
        parts=parts, events=dict(P.events), intervals={k: list(v) for k, v in P.iv.items()},
        ik_fail=list(P.ik_fail), stage1_arc_on=S1["arc_on"], stage1_laser_on=S1["laser_on"],
    )
    np.savez_compressed(path, plan=np.array(plan, dtype=object))
    if verbose:
        print(f"[plan2] {n} frames, weld offset {P.weld_offset}, IK failures {len(P.ik_fail)}: {P.ik_fail[:6]}")
    return plan


def ship():
    """Copy the current plan into stage2_logistics/data/ (committed) and drop older shipped plans."""
    import shutil
    solve()
    name = f"plan2_{_hash()}.npz"
    os.makedirs(DATA, exist_ok=True)
    for old in glob.glob(os.path.join(DATA, "plan2_*.npz")):
        if os.path.basename(old) != name:
            os.remove(old)
    src = os.path.join(OUT, name)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(DATA, name))
    return os.path.join(DATA, name)


if __name__ == "__main__":
    import sys
    if "--ship" in sys.argv:
        print("shipped", ship())
        sys.exit(0)
    p = solve(use_cache="--fresh" not in sys.argv)
    ev = sorted(p["events"].items(), key=lambda kv: kv[1])
    for k, v in ev:
        print(f"  {v:5d}  {k}")
    print("frames", p["n_frames"], "weld offset", p["weld_offset"], "ik fails", p["ik_fail"])
