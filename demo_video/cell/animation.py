"""Storyboard / choreography: per-frame joint values for robot, track and positioner, weld progress, cameras.

All motion is computed in numpy per frame and keyframed with LINEAR interpolation (transfers use smoothstep timing).
"""
import math
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
N_FRAMES = 1008

# ------------------------------------------------------------------ timeline (frames, inclusive)
T = dict(
    rest=(1, 96),
    tilt_up=(97, 150),
    track_to_A=(100, 170),
    robot_to_A=(120, 190),
    laser=(193, 236),
    to_weld_A=(236, 240),
    weld_A=(241, 456),
    arc_A=(244, 453),
    lift_A=(457, 470),
    track_to_B1=(471, 528),
    robot_to_B1=(471, 528),
    sector1=(529, 612), arc_1=(532, 610),
    back_to_top1=(613, 636),
    sector2=(637, 696), arc_2=(640, 694),
    lift_B1=(697, 712),
    index_180=(704, 752),
    track_to_B2=(700, 768),
    robot_to_B2=(742, 768),
    sector3=(769, 822), arc_3=(772, 820),
    back_to_top2=(823, 846),
    sector4=(847, 888), arc_4=(850, 886),
    lift_B2=(889, 900),
    robot_home=(900, 950),
    track_home=(905, 960),
    tilt_down=(908, 960),
    end=(961, 1008),
)
ROT_A0 = 70.0                # positioner faceplate angle at the start of seam A
ROT_A1 = ROT_A0 + 380.0      # 450 -> pipe along +Y
ROT_B2 = ROT_A1 - 180.0      # pipe along -Y
SEAM_B_Y = L.SEAM_B_CENTER[0]     # |y| of seam B when the pipe lies along the track
DY_B = 1.1                    # lateral track offset of the robot for the seam-B sectors (diagonal approach)
TRACK_B1 = SEAM_B_Y + DY_B
TRACK_B2 = -(SEAM_B_Y + DY_B)


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
    zt = L.POS_TILT_AXIS_Z
    c, s = math.cos(tilt_deg * DEG), math.sin(tilt_deg * DEG)
    Ry = np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1.0]])
    c2, s2 = math.cos(rot_deg * DEG), math.sin(rot_deg * DEG)
    Rz = np.array([[c2, -s2, 0, 0], [s2, c2, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])
    Tt = np.eye(4); Tt[2, 3] = zt
    Tf = np.eye(4); Tf[2, 3] = L.POS_FACEPLATE_OFFSET
    Tm = np.eye(4); Tm[2, 3] = L.POS_FIXTURE_THICK
    return Tt @ Ry @ Tf @ Rz @ Tm


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


def robot_lean(p, track_y):
    d = np.array([L.TRACK_X - p[0], track_y - p[1], 0.0])
    d /= np.linalg.norm(d)
    return d + np.array([0, 0, 0.9])


def target_frame(p, n, t, lean, push_deg=10.0):
    n = n / np.linalg.norm(n); t = t / np.linalg.norm(t)
    z = -n * math.cos(push_deg * DEG) + t * math.sin(push_deg * DEG)
    x = lean - np.dot(lean, z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    Tm = np.eye(4); Tm[:3, 0] = x; Tm[:3, 1] = y; Tm[:3, 2] = z; Tm[:3, 3] = p
    return Tm


def lifted(Tm, dz):
    Tl = Tm.copy(); Tl[:3, 3] = Tl[:3, 3] + np.array([0, 0, dz]); return Tl


# ------------------------------------------------------------------ main
def _cache_key():
    import hashlib, os
    h = hashlib.sha1()
    here = os.path.dirname(os.path.abspath(__file__))
    for fn in ("animation.py", "layout.py", "robot_urdf.py", "robot_build.py"):
        with open(os.path.join(here, fn), "rb") as fh:
            data = fh.read()
        if fn == "animation.py":       # camera definitions do not affect the trajectory
            data = data.split(b"# ------------------------------------------------------------------ cameras")[0]
        h.update(data)
    return h.hexdigest()[:12]


def build(scene, robot, pos, spool):
    import os
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
    ramp(tilt, T["tilt_up"], 0.0, 90.0)
    ramp(rot, T["weld_A"], ROT_A0, ROT_A1, ease=False)
    ramp(rot, T["index_180"], ROT_A1, ROT_B2)
    ramp(tilt, T["tilt_down"], 90.0, 0.0)
    ramp(track, T["track_to_A"], L.ROBOT_HOME_Y, 0.0)
    ramp(track, T["track_to_B1"], 0.0, TRACK_B1)
    ramp(track, T["track_to_B2"], TRACK_B1, TRACK_B2)
    ramp(track, T["track_home"], TRACK_B2, L.ROBOT_HOME_Y)

    # ---- robot: IK targets per frame for welding / scanning; joint-space transfers elsewhere
    q_home = np.array(L.ROBOT_Q_HOME, dtype=float)
    cached = None
    if os.path.exists(cache):
        z = np.load(cache)
        cached = dict(Q=z["Q"], arc_on=z["arc_on"], laser_on=z["laser_on"], slots=z["slots"])
    targets = {}       # frame -> 4x4 TCP target (world)
    weld_beads = []    # (bead_key, slot, (f0, f1))

    def base_at(f):
        return RB.base_matrix(track[f - 1])

    if cached is None:
        # seam A (tilt 90 during the weld)
        TsA = spool_world(90.0, ROT_A0)
        sfA = seam_frame("A", TsA)
        pA, nA, tA = seam_point(sfA, top_angle(sfA))
        # relative surface motion: the part rotates +rot about +X world (faceplate axis); torch fixed. Travel direction of the
        # torch relative to the part is opposite to the surface velocity at the top: v_surface = omega x r
        TA = target_frame(pA, nA, -tA, robot_lean(pA, 0.0), push_deg=10.0)
        TA_scan = lifted(TA, 0.035)
        for f in range(T["weld_A"][0], T["weld_A"][1] + 1):
            w = 0.0025 * math.sin(2 * math.pi * 2.5 * (f - T["weld_A"][0]) / FPS)   # weaving along the seam axis (X)
            Tw = TA.copy(); Tw[:3, 3] = Tw[:3, 3] + np.array([w, 0, 0])
            targets[f] = Tw
        # laser scan: traverse across the seam 35 mm above it
        a, b = T["laser"]
        for f in range(a, b + 1):
            s = frac(f, (a, b))
            x = -0.06 + 0.12 * (0.5 - 0.5 * math.cos(math.pi * s))          # sweep -60..+60 mm across (along X)
            Tw = TA_scan.copy(); Tw[:3, 3] = Tw[:3, 3] + np.array([x, 0, 0]); targets[f] = Tw
            laser_on[f - 1] = 1.0
        for f in range(T["to_weld_A"][0], T["to_weld_A"][1] + 1):
            s = smoothstep(frac(f, T["to_weld_A"]))
            Tw = TA_scan.copy(); Tw[:3, 3] = TA_scan[:3, 3] * (1 - s) + TA[:3, 3] * s + np.array([0.06 * (1 - s), 0, 0]); targets[f] = Tw
        arc_on[T["arc_A"][0] - 1:T["arc_A"][1]] = 1.0
        weld_beads.append(("A", 0, T["arc_A"]))

        # seam B sectors
        def sector(span, arc_span, rot_deg, track_y, toward_plus_x, slot):
            Ts = spool_world(90.0, rot_deg)
            sf = seam_frame("B", Ts)
            a_top = top_angle(sf)
            p5, _, _ = seam_point(sf, a_top + 5.0)
            sgn = 1.0 if ((p5[0] > sf[0][0]) == toward_plus_x) else -1.0
            f0, f1 = span
            for f in range(f0, f1 + 1):
                s = frac(f, span)
                ang = a_top + sgn * 92.0 * s
                p, n, t = seam_point(sf, ang)
                w = 0.0025 * math.sin(2 * math.pi * 2.5 * (f - f0) / FPS)
                p = p + w * sf[1]
                targets[f] = target_frame(p, n, t * sgn, robot_lean(p, track_y), push_deg=10.0)
            arc_on[arc_span[0] - 1:arc_span[1]] = 1.0
            weld_beads.append(("B", slot, arc_span))
            return targets[f0], targets[f1]

        s1_start, s1_end = sector(T["sector1"], T["arc_1"], ROT_A1, TRACK_B1, True, 0)
        s2_start, s2_end = sector(T["sector2"], T["arc_2"], ROT_A1, TRACK_B1, False, 1)
        s3_start, s3_end = sector(T["sector3"], T["arc_3"], ROT_B2, TRACK_B2, True, 2)
        s4_start, s4_end = sector(T["sector4"], T["arc_4"], ROT_B2, TRACK_B2, False, 3)

        # ---- transfers (Cartesian lifts + joint-space moves).  Solve IK for key poses first.
        Q_REF = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])      # wrist bent ~57° positive, q4/q6 near zero
        def solve(Tm, q0, f):
            q, ok, err = arm.ik(Tm, q0, base_at(f), TOOL, free_spin=True, q_ref=Q_REF)
            if not ok:
                q, ok, err = arm.ik(Tm, q_home, base_at(f), TOOL, iters=400, free_spin=True, q_ref=Q_REF)
            if not ok:
                IK_FAILS.append((f, err))
            return q
        # key poses
        q_appA = solve(lifted(TA, 0.15), q_home, T["robot_to_A"][1])
        q_scan0 = solve(targets[T["laser"][0]], q_appA, T["laser"][0])
        q_liftA = solve(lifted(TA, 0.15), q_appA, T["lift_A"][1])
        q_appB1 = solve(lifted(s1_start, 0.12), q_liftA, T["robot_to_B1"][1])
        q_s1_lift = solve(lifted(s1_end, 0.06), q_appB1, T["back_to_top1"][0])
        q_s2_app = solve(lifted(s2_start, 0.06), q_s1_lift, T["back_to_top1"][1])
        q_s2_lift = solve(lifted(s2_end, 0.15), q_s2_app, T["lift_B1"][1])
        q_appB2 = solve(lifted(s3_start, 0.12), q_appB1, T["robot_to_B2"][1])
        q_s3_lift = solve(lifted(s3_end, 0.06), q_appB2, T["back_to_top2"][0])
        q_s4_app = solve(lifted(s4_start, 0.06), q_s3_lift, T["back_to_top2"][1])
        q_s4_lift = solve(lifted(s4_end, 0.15), q_s4_app, T["lift_B2"][1])

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
                s = smoothstep(frac(f, span))
                Tm = T0.copy(); Tm[:3, 3] = T0[:3, 3] * (1 - s) + T1[:3, 3] * s
                # slerp-free orientation blend: use T0 orientation for s<0.5 else T1 (close anyway)
                Tm[:3, :3] = T0[:3, :3] if s < 0.5 else T1[:3, :3]
                q = solve(Tm, q, f); Q[f - 1] = q
            return q
        def hold(span, q):
            a, b = span; Q[a - 1:b] = q

        # rest
        hold((1, T["robot_to_A"][0]), q_home)
        joint_move(T["robot_to_A"], q_home, q_appA)
        hold((T["robot_to_A"][1], T["laser"][0]), q_appA)
        # scan + weld A: per-frame IK
        q = q_appA
        for f in range(T["laser"][0] - 2, T["weld_A"][1] + 1):
            if f in targets:
                q = solve(targets[f], q, f); Q[f - 1] = q
            elif f < T["laser"][0]:
                # descend from approach to scan start
                s = smoothstep(frac(f, (T["laser"][0] - 2, T["laser"][0])))
                Q[f - 1] = q_appA * (1 - s) + q_scan0 * s
        q_weldA_end = Q[T["weld_A"][1] - 1]
        cart_move(T["lift_A"], targets[T["weld_A"][1]], lifted(TA, 0.15), q_weldA_end)
        joint_move(T["robot_to_B1"], Q[T["lift_A"][1] - 1], q_appB1)
        # sector 1
        q = cart_move((T["sector1"][0] - 3, T["sector1"][0]), lifted(s1_start, 0.12), s1_start, q_appB1)
        for f in range(T["sector1"][0], T["sector1"][1] + 1):
            q = solve(targets[f], q, f); Q[f - 1] = q
        q = cart_move((T["back_to_top1"][0], T["back_to_top1"][0] + 6), s1_end, lifted(s1_end, 0.06), q)
        joint_move((T["back_to_top1"][0] + 6, T["back_to_top1"][1] - 4), q, q_s2_app)
        q = cart_move((T["back_to_top1"][1] - 4, T["back_to_top1"][1]), lifted(s2_start, 0.06), s2_start, q_s2_app)
        for f in range(T["sector2"][0], T["sector2"][1] + 1):
            q = solve(targets[f], q, f); Q[f - 1] = q
        q = cart_move(T["lift_B1"], s2_end, lifted(s2_end, 0.15), q)
        hold((T["lift_B1"][1], T["robot_to_B2"][0]), q)
        joint_move(T["robot_to_B2"], q, q_appB2)
        q = cart_move((T["sector3"][0] - 3, T["sector3"][0]), lifted(s3_start, 0.12), s3_start, q_appB2)
        for f in range(T["sector3"][0], T["sector3"][1] + 1):
            q = solve(targets[f], q, f); Q[f - 1] = q
        q = cart_move((T["back_to_top2"][0], T["back_to_top2"][0] + 6), s3_end, lifted(s3_end, 0.06), q)
        joint_move((T["back_to_top2"][0] + 6, T["back_to_top2"][1] - 4), q, q_s4_app)
        q = cart_move((T["back_to_top2"][1] - 4, T["back_to_top2"][1]), lifted(s4_start, 0.06), s4_start, q_s4_app)
        for f in range(T["sector4"][0], T["sector4"][1] + 1):
            q = solve(targets[f], q, f); Q[f - 1] = q
        q = cart_move(T["lift_B2"], s4_end, lifted(s4_end, 0.15), q)
        joint_move(T["robot_home"], q, q_home)
        hold((T["robot_home"][1], nF), q_home)

        np.savez(cache, Q=Q, arc_on=arc_on, laser_on=laser_on,
                 slots=np.array([[ {"A": 0, "B": 1}[k], sl, sp[0], sp[1]] for (k, sl, sp) in weld_beads]))
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
    return dict(weld_intervals=weld_intervals, laser_intervals=laser_intervals, arc=arc, frames=nF, Q=Q, track=track, tilt=tilt, rot=rot, timeline=T)


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
    ("C1_wide", 1, 96, (-3.6, -6.4, 3.4), (-2.9, -5.8, 3.15), (0.9, 0.0, 1.15), (0.9, 0.0, 1.2), 32, 0),
    ("C2_setup", 97, 192, (3.6, -3.8, 2.3), (3.1, -3.1, 2.05), (0.7, -0.3, 1.35), (0.5, 0.0, 1.45), 35, 0),
    ("C3_closeA", 193, 330, (1.45, -1.15, 1.85), (1.25, -0.95, 1.78), (0.40, 0.0, 1.47), (0.40, 0.0, 1.47), 60, 5.6),
    ("C4_midA", 331, 456, (1.9, 1.7, 1.55), (2.1, 1.35, 1.65), (0.55, 0.0, 1.45), (0.55, 0.0, 1.45), 45, 8.0),
    ("C5_reposition", 457, 528, (2.9, -2.3, 2.5), (2.6, -2.0, 2.3), (0.8, 0.2, 1.45), (0.8, 0.4, 1.45), 35, 0),
    ("C6_closeB1", 529, 696, (1.55, -0.95, 1.95), (1.4, -0.8, 1.9), (0.74, 0.38, 1.45), (0.74, 0.38, 1.45), 55, 6.3),
    ("C7_index", 697, 768, (-1.3, -3.5, 2.6), (-0.9, -3.15, 2.45), (0.55, 0.1, 1.25), (0.6, -0.1, 1.25), 32, 0),
    ("C8_closeB2", 769, 888, (1.55, 0.95, 1.95), (1.4, 0.8, 1.9), (0.74, -0.38, 1.45), (0.74, -0.38, 1.45), 55, 6.3),
    ("C9_final", 889, 1008, (3.9, -5.8, 3.1), (3.4, -5.1, 2.9), (0.4, -0.3, 1.25), (0.35, -0.2, 1.3), 32, 0),
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
