"""Storyboard / choreography: per-frame joint values for robot, track and positioner, weld progress, cameras.

Layout (see layout.py): positioner tilt axis parallel to the track (world X); tilt +90 -> faceplate faces +Y.
Seam A is welded by rotating the part 380° with the torch static at 12 o'clock; the pipe leg sweeps a disc in the
plane y ~ 0.76 that the robot never crosses.  Seam B is welded by the robot in four 90° sectors (two per side), the
180° index being a tilt flip +90 -> -90 (flange over the top).  Before every track / positioner motion the robot parks
in a transit pose (wrist at x ~ 1.35) outside the sweep of the part.
All motion is computed in numpy per frame and keyframed with LINEAR interpolation (transfers use smoothstep timing).
"""
import math
import os
import bpy
import numpy as np
import mathutils
from . import layout as L
from . import geom as G
from . import robot_build as RB
from . import positioner as P

DEG = math.pi / 180.0
FPS = L.FPS
IK_FAILS = []
N_FRAMES = 1104

# ------------------------------------------------------------------ timeline (frames, inclusive)
T = dict(
    rest=(1, 96),
    tilt_up=(97, 150),
    track_to_A=(100, 170),
    robot_to_A=(120, 190),
    laser=(193, 236),
    to_weld_A=(236, 240),
    weld_A=(241, 456), arc_A=(244, 453),
    lift_A=(457, 470),
    to_transit_A=(471, 492),
    track_to_B1=(486, 540),
    robot_to_B1=(496, 537),
    sector1=(541, 624), arc_1=(544, 622),
    back_to_top1=(625, 648),
    sector2=(649, 708), arc_2=(652, 706),
    lift_B1=(709, 720),
    to_transit_B1=(721, 740),
    index_180=(742, 800),
    track_to_B2=(742, 805),
    robot_to_B2=(780, 807),
    sector3=(811, 864), arc_3=(814, 862),
    back_to_top2=(865, 888),
    sector4=(889, 930), arc_4=(892, 928),
    lift_B2=(931, 942),
    to_transit_B2=(943, 968),
    robot_home=(969, 1030),
    track_home=(975, 1040),
    tilt_down=(985, 1060),
    end=(1061, 1104),
)
ROT_A0 = -20.0               # faceplate angle at the start of seam A
ROT_A1 = ROT_A0 + 380.0      # = 360 -> pipe leg along +X (toward the robot) for the seam-B sectors
TILT_B1, TILT_B2 = 90.0, -90.0
SEAM_B_Y = L.POS_FACEPLATE_OFFSET + L.POS_FIXTURE_THICK + L.SEAM_B_CENTER[2]   # |y| of seam B at tilt +/-90
TRACK_A = 0.2                # track station for seam A (arm slightly ahead of the seam plane, clear of the pipe sweep)
TRACK_B1, TRACK_B2 = SEAM_B_Y, -SEAM_B_Y
TRANSIT_TCP = (1.35, 2.05)   # (x, z) of the parked torch tip relative to the world (y = track station)
LIFT_APPROACH = 0.12         # radial approach / retract distance at sector starts/ends
LIFT_PARK = 0.15


def smoothstep(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def frac(f, span):
    a, b = span
    if b == a:
        return 1.0
    return (f - a) / float(b - a)


# ------------------------------------------------------------------ geometry helpers
def spool_world(tilt_deg, rot_deg):
    """World transform of the spool root for given positioner angles (tilt about world X, see positioner.set_tilt)."""
    zt = L.POS_TILT_AXIS_Z
    c, s = math.cos(-tilt_deg * DEG), math.sin(-tilt_deg * DEG)
    Rx = np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.0]])
    c2, s2 = math.cos(rot_deg * DEG), math.sin(rot_deg * DEG)
    Rz = np.array([[c2, -s2, 0, 0], [s2, c2, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])
    Tt = np.eye(4); Tt[2, 3] = zt
    Tf = np.eye(4); Tf[2, 3] = L.POS_FACEPLATE_OFFSET
    Tm = np.eye(4); Tm[2, 3] = L.POS_FIXTURE_THICK
    return Tt @ Rx @ Tf @ Rz @ Tm


def seam_frame(which, T_spool):
    """(centre_world, axis_world, u_world, v_world) of a seam circle: point(a) = c + R (cos a u + sin a v)."""
    if which == "A":
        c = np.array([0, 0, L.SEAM_A_Z, 1.0]); ax = np.array([0, 0, 1.0]); u = np.array([1.0, 0, 0]); v = np.array([0, 1.0, 0])
    else:
        c = np.array([*L.SEAM_B_CENTER, 1.0]); ax = np.array([1.0, 0, 0]); u = np.array([0, 1.0, 0]); v = np.array([0, 0, 1.0])
    R = T_spool[:3, :3]
    return (T_spool @ c)[:3], R @ ax, R @ u, R @ v


def seam_point(sf, a_deg):
    c, ax, u, v = sf
    n = math.cos(a_deg * DEG) * u + math.sin(a_deg * DEG) * v
    p = c + L.SEAM_RADIUS * n
    t = np.cross(ax, n)
    return p, n, t


def top_angle(sf):
    c, ax, u, v = sf
    return math.degrees(math.atan2(v[2], u[2]))


def robot_lean(p, track_y, up=0.9):
    d = np.array([L.TRACK_X - p[0], track_y - p[1], 0.0])
    d /= np.linalg.norm(d)
    return d + np.array([0, 0, up])


def target_frame(p, n, t, lean, push_deg=10.0):
    n = n / np.linalg.norm(n); t = t / np.linalg.norm(t)
    z = -n * math.cos(push_deg * DEG) + t * math.sin(push_deg * DEG)
    x = lean - np.dot(lean, z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    Tm = np.eye(4); Tm[:3, 0] = x; Tm[:3, 1] = y; Tm[:3, 2] = z; Tm[:3, 3] = p
    return Tm


def offset(Tm, d):
    Tl = Tm.copy(); Tl[:3, 3] = Tl[:3, 3] + np.asarray(d, dtype=float); return Tl


# ---- quaternion helpers for Cartesian moves with orientation blending
def mat2quat(R):
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2; w = 0.25 * S; x = (m[2, 1] - m[1, 2]) / S; y = (m[0, 2] - m[2, 0]) / S; z = (m[1, 0] - m[0, 1]) / S
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        S = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2; w = (m[2, 1] - m[1, 2]) / S; x = 0.25 * S; y = (m[0, 1] + m[1, 0]) / S; z = (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2; w = (m[0, 2] - m[2, 0]) / S; x = (m[0, 1] + m[1, 0]) / S; y = 0.25 * S; z = (m[1, 2] + m[2, 1]) / S
    else:
        S = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2; w = (m[1, 0] - m[0, 1]) / S; x = (m[0, 2] + m[2, 0]) / S; y = (m[1, 2] + m[2, 1]) / S; z = 0.25 * S
    q = np.array([w, x, y, z]); return q / np.linalg.norm(q)


def quat2mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def slerp(q0, q1, s):
    d = float(np.dot(q0, q1))
    if d < 0:
        q1 = -q1; d = -d
    if d > 0.9995:
        q = q0 + s * (q1 - q0); return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - s) * th) * q0 + math.sin(s * th) * q1) / math.sin(th)


def blend_pose(T0, T1, s):
    Tm = np.eye(4)
    Tm[:3, :3] = quat2mat(slerp(mat2quat(T0[:3, :3]), mat2quat(T1[:3, :3]), s))
    Tm[:3, 3] = T0[:3, 3] * (1 - s) + T1[:3, 3] * s
    return Tm


def _cache_key():
    import hashlib
    h = hashlib.sha1()
    here = os.path.dirname(os.path.abspath(__file__))
    for fn in ("animation.py", "layout.py", "robot_urdf.py", "robot_build.py"):
        with open(os.path.join(here, fn), "rb") as fh:
            data = fh.read()
        if fn == "animation.py":       # camera definitions do not affect the trajectory
            data = data.split(b"# ------------------------------------------------------------------ cameras")[0]
        h.update(data)
    return h.hexdigest()[:12]


# ------------------------------------------------------------------ main
def build(scene, robot, pos, spool):
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"anim_cache_{_cache_key()}.npz")
    arm = robot["arm"]
    TOOL = RB.tool_transform()
    nF = N_FRAMES
    frames = np.arange(1, nF + 1)
    tilt = np.zeros(nF); rot = np.full(nF, ROT_A0); track = np.full(nF, L.ROBOT_HOME_Y)
    Q = np.tile(np.array(L.ROBOT_Q_HOME, dtype=float), (nF, 1))
    arc_on = np.zeros(nF)
    laser_on = np.zeros(nF)

    # ---- positioner & track schedules
    def ramp(arr, span, v0, v1, ease=True):
        a, b = span
        for f in range(a, b + 1):
            s = frac(f, span); s = smoothstep(s) if ease else s
            arr[f - 1] = v0 + (v1 - v0) * s
        arr[b:] = v1
    ramp(tilt, T["tilt_up"], 0.0, TILT_B1)
    ramp(rot, T["weld_A"], ROT_A0, ROT_A1, ease=False)
    ramp(tilt, T["index_180"], TILT_B1, TILT_B2)
    ramp(tilt, T["tilt_down"], TILT_B2, 0.0)
    ramp(track, T["track_to_A"], L.ROBOT_HOME_Y, TRACK_A)
    ramp(track, T["track_to_B1"], TRACK_A, TRACK_B1)
    ramp(track, T["track_to_B2"], TRACK_B1, TRACK_B2)
    ramp(track, T["track_home"], TRACK_B2, L.ROBOT_HOME_Y)

    q_home = np.array(L.ROBOT_Q_HOME, dtype=float)
    cached = None
    if os.path.exists(cache):
        z = np.load(cache)
        cached = dict(Q=z["Q"], arc_on=z["arc_on"], laser_on=z["laser_on"], slots=z["slots"])
    weld_beads = []    # (bead_key, slot, (f0, f1))
    targets = {}       # frame -> 4x4 TCP target (world)

    if cached is None:
        Q_REF = np.array([0.0, 0.0, 0.0, 0.0, 1.2, 0.0])      # wrist bent ~70° positive, q4/q6 near zero

        def base_at(f):
            return RB.base_matrix(track[f - 1])

        def solve(Tm, q0, f):
            q, ok, err = arm.ik(Tm, q0, base_at(f), TOOL, free_spin=True, q_ref=Q_REF)
            if not ok:
                q, ok, err = arm.ik(Tm, q_home, base_at(f), TOOL, iters=400, free_spin=True, q_ref=Q_REF)
            if not ok:
                IK_FAILS.append((f, err))
            return q

        # ---------------- seam A: torch static at 12 o'clock, part rotates about +Y (faceplate axis)
        TsA = spool_world(TILT_B1, ROT_A0)
        sfA = seam_frame("A", TsA)
        cA, axA = sfA[0], sfA[1]
        pA = cA + L.SEAM_RADIUS * np.array([0, 0, 1.0])
        nA = np.array([0, 0, 1.0])
        v_surface = np.cross(axA, pA - cA)                     # surface velocity for increasing rot
        tA = -v_surface / np.linalg.norm(v_surface)            # torch travel relative to the part
        TA = target_frame(pA, nA, tA, robot_lean(pA, TRACK_A), push_deg=10.0)
        TA_scan = offset(TA, (0, 0, 0.035))
        for f in range(T["weld_A"][0], T["weld_A"][1] + 1):
            w = 0.0025 * math.sin(2 * math.pi * 2.5 * (f - T["weld_A"][0]) / FPS)   # weaving along the seam axis
            targets[f] = offset(TA, axA * w)
        a, b = T["laser"]
        for f in range(a, b + 1):
            s = frac(f, (a, b))
            x = -0.06 + 0.12 * (0.5 - 0.5 * math.cos(math.pi * s))          # sweep -60..+60 mm across the joint
            targets[f] = offset(TA_scan, axA * x)
            laser_on[f - 1] = 1.0
        for f in range(T["to_weld_A"][0], T["to_weld_A"][1] + 1):
            s = smoothstep(frac(f, T["to_weld_A"]))
            targets[f] = blend_pose(offset(TA_scan, axA * 0.06), TA, s)
        arc_on[T["arc_A"][0] - 1:T["arc_A"][1]] = 1.0
        weld_beads.append(("A", 0, T["arc_A"]))

        # ---------------- seam B sectors (pipe along +X); each returns start/end targets and normals
        def sector(span, arc_span, tilt_deg, track_y, toward_plus_y, slot):
            Ts = spool_world(tilt_deg, ROT_A1)
            sf = seam_frame("B", Ts)
            a_top = top_angle(sf)
            p5, _, _ = seam_point(sf, a_top + 5.0)
            sgn = 1.0 if ((p5[1] > sf[0][1]) == toward_plus_y) else -1.0
            f0, f1 = span
            info = {}
            for f in range(f0, f1 + 1):
                s = frac(f, span)
                ang = a_top + sgn * (98.0 * s - 4.0)          # -4° .. +94°: sectors overlap at 12, 3 and 9 o'clock
                p, n, t = seam_point(sf, ang)
                w = 0.0025 * math.sin(2 * math.pi * 2.5 * (f - f0) / FPS)
                p = p + w * sf[1]
                targets[f] = target_frame(p, n, t * sgn, robot_lean(p, track_y), push_deg=10.0)
                if f == f0:
                    info["start"] = (targets[f], n)
                if f == f1:
                    info["end"] = (targets[f], n)
            pv, nv, tv = seam_point(sf, a_top + sgn * 45.0)   # via point for the return move (1:30 o'clock, retracted)
            info["via"] = (target_frame(pv, nv, tv * sgn, robot_lean(pv, track_y), 0.0), nv)
            arc_on[arc_span[0] - 1:arc_span[1]] = 1.0
            weld_beads.append(("B", slot, arc_span))
            return info

        S1 = sector(T["sector1"], T["arc_1"], TILT_B1, TRACK_B1, False, 0)
        S2 = sector(T["sector2"], T["arc_2"], TILT_B1, TRACK_B1, True, 1)
        S3 = sector(T["sector3"], T["arc_3"], TILT_B2, TRACK_B2, False, 2)
        S4 = sector(T["sector4"], T["arc_4"], TILT_B2, TRACK_B2, True, 3)

        def radial(info_key, info, d):
            Tm, n = info[info_key]
            return offset(Tm, n * d)

        # ---------------- transit (park) pose: torch down, tip at TRANSIT_TCP, outside the part sweep
        def transit_target(track_y):
            p = np.array([TRANSIT_TCP[0], track_y, TRANSIT_TCP[1]])
            return target_frame(p, np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), np.array([1.0, 0, 0.9]), 0.0)
        q_transit = solve(transit_target(TRACK_A), q_home, T["to_transit_A"][1])

        # ---------------- key poses
        q_appA = solve(offset(TA, (0, 0, 0.15)), q_transit, T["robot_to_A"][1])
        q_scan0 = solve(targets[T["laser"][0]], q_appA, T["laser"][0])
        q_appB1 = solve(radial("start", S1, LIFT_APPROACH), q_transit, T["robot_to_B1"][1])
        q_via1 = solve(radial("via", S1, 0.16), q_appB1, T["back_to_top1"][0] + 12)
        q_s2_app = solve(radial("start", S2, LIFT_APPROACH), q_via1, T["back_to_top1"][1])
        q_appB2 = solve(radial("start", S3, LIFT_APPROACH), q_transit, T["robot_to_B2"][1])
        q_via2 = solve(radial("via", S3, 0.16), q_appB2, T["back_to_top2"][0] + 12)
        q_s4_app = solve(radial("start", S4, LIFT_APPROACH), q_via2, T["back_to_top2"][1])

        def joint_move(span, q0, q1, via=None):
            a, b = span
            pts = [q0] + (via or []) + [q1]
            for f in range(a, b + 1):
                s = smoothstep(frac(f, span)) * (len(pts) - 1)
                i = min(int(s), len(pts) - 2); u = s - i
                Q[f - 1] = pts[i] * (1 - u) + pts[i + 1] * u

        def cart_move(span, T0, T1, q_seed):
            a, b = span
            q = q_seed
            for f in range(a, b + 1):
                q = solve(blend_pose(T0, T1, smoothstep(frac(f, span))), q, f); Q[f - 1] = q
            return q

        def hold(span, q):
            a, b = span; Q[a - 1:b] = q

        def track_seg(span, q):
            """per-frame IK on `targets` for a welding/scanning span, warm-started from q."""
            for f in range(span[0], span[1] + 1):
                q = solve(targets[f], q, f); Q[f - 1] = q
            return q

        # ---------------- sequence
        hold((1, T["robot_to_A"][0]), q_home)
        joint_move(T["robot_to_A"], q_home, q_appA, via=[q_transit])
        hold((T["robot_to_A"][1], T["laser"][0] - 2), q_appA)
        for f in range(T["laser"][0] - 2, T["laser"][0]):
            s = smoothstep(frac(f, (T["laser"][0] - 2, T["laser"][0])))
            Q[f - 1] = q_appA * (1 - s) + q_scan0 * s
        q = track_seg((T["laser"][0], T["weld_A"][1]), q_appA)
        q = cart_move(T["lift_A"], targets[T["weld_A"][1]], offset(TA, (0, 0, 0.15)), q)
        joint_move(T["to_transit_A"], q, q_transit)
        hold((T["to_transit_A"][1], T["robot_to_B1"][0]), q_transit)
        joint_move(T["robot_to_B1"], q_transit, q_appB1)
        q = cart_move((T["robot_to_B1"][1], T["sector1"][0]), radial("start", S1, LIFT_APPROACH), S1["start"][0], q_appB1)
        q = track_seg(T["sector1"], q)
        a, b = T["back_to_top1"]
        q = cart_move((a, a + 6), S1["end"][0], radial("end", S1, LIFT_APPROACH), q)
        joint_move((a + 6, b - 4), q, q_s2_app, via=[q_via1])
        q = cart_move((b - 4, T["sector2"][0]), radial("start", S2, LIFT_APPROACH), S2["start"][0], q_s2_app)
        q = track_seg(T["sector2"], q)
        q = cart_move(T["lift_B1"], S2["end"][0], radial("end", S2, LIFT_PARK), q)
        joint_move(T["to_transit_B1"], q, q_transit)
        hold((T["to_transit_B1"][1], T["robot_to_B2"][0]), q_transit)
        joint_move(T["robot_to_B2"], q_transit, q_appB2)
        q = cart_move((T["robot_to_B2"][1], T["sector3"][0]), radial("start", S3, LIFT_APPROACH), S3["start"][0], q_appB2)
        q = track_seg(T["sector3"], q)
        a, b = T["back_to_top2"]
        q = cart_move((a, a + 6), S3["end"][0], radial("end", S3, LIFT_APPROACH), q)
        joint_move((a + 6, b - 4), q, q_s4_app, via=[q_via2])
        q = cart_move((b - 4, T["sector4"][0]), radial("start", S4, LIFT_APPROACH), S4["start"][0], q_s4_app)
        q = track_seg(T["sector4"], q)
        q = cart_move(T["lift_B2"], S4["end"][0], radial("end", S4, LIFT_PARK), q)
        q = cart_move(T["to_transit_B2"], radial("end", S4, LIFT_PARK), transit_target(TRACK_B2), q)
        joint_move(T["robot_home"], q, q_home)
        hold((T["robot_home"][1], nF), q_home)

        np.savez(cache, Q=Q, arc_on=arc_on, laser_on=laser_on,
                 slots=np.array([[{"A": 0, "B": 1}[k], sl, sp[0], sp[1]] for (k, sl, sp) in weld_beads]))
    else:
        Q[:] = cached["Q"]; arc_on[:] = cached["arc_on"]; laser_on[:] = cached["laser_on"]
        weld_beads = [("AB"[int(r[0])], int(r[1]), (int(r[2]), int(r[3]))) for r in cached["slots"]]

    # ---- keyframe everything
    for i, f in enumerate(frames):
        P.set_tilt(pos, tilt[i], frame=int(f))
        P.set_rot(pos, rot[i], frame=int(f))
        RB.set_track(robot, float(track[i]), frame=int(f))
        RB.set_q(robot, Q[i], frame=int(f))
    _linear(pos["tilt"]); _linear(pos["rot"]); _linear(robot["carriage"])
    for e in robot["joints"]:
        _linear(e)

    # ---- arc empty with arc_on property (constant interpolation)
    arc = G.empty("arc_point", collection=robot["collection"], size=0.03)
    arc.parent = robot["tcp"]; arc.matrix_parent_inverse = mathutils.Matrix.Identity(4)
    arc["arc_on"] = 0.0
    prev = None
    for i, f in enumerate(frames):
        v = float(arc_on[i])
        if v != prev or i == 0 or i == nF - 1:
            arc["arc_on"] = v; arc.keyframe_insert('["arc_on"]', frame=int(f))
        prev = v
    _constant(arc)
    weld_intervals = [span for (_, _, span) in weld_beads]
    laser_intervals = [T["laser"]]

    # ---- weld bead progress (numeric, from the actual TCP path in the bead's local frame)
    bpy.context.view_layer.update()
    for key, slot, (f0, f1) in weld_beads:
        bead = spool["beads"][key]
        us = []
        for f in range(f0, f1 + 1):
            scene.frame_set(f)
            Mi = bead.matrix_world.inverted()
            p = Mi @ robot["tcp"].matrix_world.translation
            us.append(math.atan2(p.y, p.x) / (2 * math.pi))
        us = np.unwrap(np.array(us) * 2 * math.pi) / (2 * math.pi)
        d = us[-1] - us[0]
        direction = 1.0 if d >= 0 else -1.0
        start = us[0] % 1.0
        bead[f"w{slot}_start"] = start; bead[f"w{slot}_dir"] = direction
        bead.keyframe_insert(f'["w{slot}_start"]', frame=1); bead.keyframe_insert(f'["w{slot}_dir"]', frame=1)
        bead[f"w{slot}_prog"] = 0.0; bead.keyframe_insert(f'["w{slot}_prog"]', frame=f0 - 1)
        for k, f in enumerate(range(f0, f1 + 1)):
            bead[f"w{slot}_prog"] = float(abs(us[k] - us[0]) + 0.004)
            bead.keyframe_insert(f'["w{slot}_prog"]', frame=f)
        bead[f"w{slot}_hot"] = 0.0; bead.keyframe_insert(f'["w{slot}_hot"]', frame=f0 - 1)
        bead[f"w{slot}_hot"] = 1.0; bead.keyframe_insert(f'["w{slot}_hot"]', frame=f0)
        bead.keyframe_insert(f'["w{slot}_hot"]', frame=f1)
        bead[f"w{slot}_hot"] = 0.0; bead.keyframe_insert(f'["w{slot}_hot"]', frame=f1 + 40)
        _linear(bead)
        tint = bpy.data.objects.get(f"spool_tint_{key}")
        if tint is not None:
            tint["tint_on"] = tint.get("tint_on", 0.0)
            tint.keyframe_insert('["tint_on"]', frame=f0)
            tint["tint_on"] = 1.0; tint.keyframe_insert('["tint_on"]', frame=f1); _linear(tint)
    scene.frame_set(1)
    return dict(weld_intervals=weld_intervals, laser_intervals=laser_intervals, arc=arc, frames=nF, Q=Q,
                track=track, tilt=tilt, rot=rot, timeline=T)


def _linear(ob):
    ad = ob.animation_data
    if ad and ad.action:
        for fc in ad.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = 'LINEAR'


def _constant(ob):
    ad = ob.animation_data
    if ad and ad.action:
        for fc in ad.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = 'CONSTANT'


# ------------------------------------------------------------------ cameras
SHOTS = [
    # (name, f0, f1, cam_from, cam_to, aim_from, aim_to, lens_mm, fstop)
    ("C1_wide", 1, 96, (-4.6, -7.2, 4.0), (-3.9, -6.4, 3.7), (0.9, 0.0, 1.1), (0.9, 0.0, 1.1), 28, 0),
    ("C2_setup", 97, 192, (-0.2, 3.1, 2.5), (0.2, 2.7, 2.35), (0.6, 0.2, 1.35), (0.5, 0.2, 1.4), 30, 0),
    ("C3a_laser", 193, 240, (0.75, 1.55, 1.95), (0.70, 1.50, 1.90), (0.0, 0.40, 1.47), (0.0, 0.40, 1.47), 85, 4.0),
    ("C3b_closeA", 241, 300, (0.9, 1.35, 2.0), (0.8, 1.25, 1.95), (0.0, 0.40, 1.47), (0.0, 0.40, 1.47), 60, 5.6),
    ("C4_midA", 301, 456, (1.4, 0.95, 2.8), (1.3, 0.75, 2.75), (0.05, 0.42, 1.45), (0.05, 0.42, 1.45), 45, 8.0),
    ("C5_reposition", 457, 540, (4.4, -3.2, 3.4), (4.0, -2.6, 3.2), (1.4, 0.3, 1.0), (1.4, 0.6, 1.0), 30, 0),
    ("C6a_closeB1", 541, 648, (1.3, -0.55, 2.05), (1.2, -0.45, 2.0), (0.42, 0.76, 1.42), (0.42, 0.76, 1.42), 42, 6.3),
    ("C6b_closeB1", 649, 708, (1.3, 2.05, 2.05), (1.2, 1.95, 2.0), (0.42, 0.76, 1.42), (0.42, 0.76, 1.42), 42, 6.3),
    ("C7_index", 709, 810, (-3.4, -1.8, 2.4), (-3.0, -1.2, 2.3), (0.4, 0.0, 1.4), (0.4, 0.0, 1.4), 32, 0),
    ("C8a_closeB2", 811, 888, (1.3, -2.05, 2.05), (1.2, -1.95, 2.0), (0.42, -0.76, 1.42), (0.42, -0.76, 1.42), 42, 6.3),
    ("C8b_closeB2", 889, 930, (1.3, 0.55, 2.05), (1.2, 0.45, 2.0), (0.42, -0.76, 1.42), (0.42, -0.76, 1.42), 42, 6.3),
    ("C9_final", 931, 1104, (5.4, -6.2, 4.0), (4.6, -5.2, 3.4), (0.8, -0.2, 1.1), (0.8, -0.2, 1.1), 30, 0),
]


def build_cameras(scene, collection=None):
    col = collection or bpy.data.collections.new("Cameras")
    if collection is None:
        scene.collection.children.link(col)
    cams = []
    for (name, f0, f1, c0, c1, a0, a1, lens, fstop) in SHOTS:
        cd = bpy.data.cameras.new(name)
        cd.lens = lens
        cd.sensor_width = 36
        cd.clip_start = 0.05
        cam = bpy.data.objects.new(name, cd)
        col.objects.link(cam)
        aim = G.empty(name + "_aim", collection=col, size=0.1)
        for f, c, a in ((f0, c0, a0), (f1, c1, a1)):
            cam.location = c; cam.keyframe_insert("location", frame=f)
            aim.location = a; aim.keyframe_insert("location", frame=f)
        for ob in (cam, aim):
            for fc in ob.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = 'BEZIER'; kp.easing = 'EASE_IN_OUT'
        con = cam.constraints.new('TRACK_TO')
        con.target = aim; con.track_axis = 'TRACK_NEGATIVE_Z'; con.up_axis = 'UP_Y'
        if fstop > 0:
            cd.dof.use_dof = True; cd.dof.focus_object = aim; cd.dof.aperture_fstop = fstop
        m = scene.timeline_markers.new(name, frame=f0)
        m.camera = cam
        cams.append(cam)
    scene.camera = cams[0]
    return cams
