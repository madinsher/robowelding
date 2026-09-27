"""Assembly & tack station of the logistics zone: welded fixture table with the DN250 spool fixture.

The spool stands on the station in its "standing" pose (layout2 module docstring): station frame
    F = kin.planar_frame(STATION_ORIGIN, STATION_XDIR)
(origin = flange back-face centre on the seat, local +X = pipe leg = world -Y toward the handler track, local +Y =
world +X = tack-robot side).  Everything is built in station-local coordinates under the empty `stn_root` placed at F.

Fixture (station-local, spool frame):
* welded table STATION_TABLE: 20 mm nitrided top plate (top at local z = -0.02, world 0.78) with a D28 hole grid,
  square-tube apron, six legs with levelling feet, stretchers;
* flange seat: machined round plate (r 0.235, top at z = 0) with a recess (r 0.195) for the raised face and the
  stage-1 bolt-hole discs (both stand 2 mm proud of the flange back face), 3 centring pins just
  outside the flange rim (30 / 150 / 270 deg), 6 socket-head screws;
* V-block under the elbow at local x = 0.30 (groove tilted along the elbow tangent, apex solved numerically so both
  bronze liners touch the elbow's outer surface), V-block under the pipe at x = 0.86 (90 deg V touching the pipe),
  pipe end stop (bronze face plate at x = PIPE_END_X, adjusting screw) - the stop sets the root gap of seam B;
* 4 pneumatic lever clamps (names of layout2.STATION_CLAMPS): a lever on a pivot pin (swing axis = the clamp's local
  X), a swivel pad on a threaded spindle, an ISO cylinder on a trunnion pin whose rod eye drives the lever tail.
  s = 1 (closed): the pad faces touch the part (GAP); s = 0 (open): the lever is swung back by OPEN (100-105 deg).
  Refinement of the layout2 pivots (the names are kept): all four clamps stand on the local -Y side (world -X),
  so the +Y side of seams A and B stays free for the tack torch, and outside the open handler gripper (|y| <= 0.32
  over its jaw span) at every station grasp:
      flange_l / flange_r  radial clamps at 210 / 330 deg pressing on the flange top face (r 0.189, between bolt holes)
      elbow                pad on the elbow's upper -Y quadrant at x = 0.285 (pushes the elbow onto its V-block)
      pipe                 pad on the pipe's upper -Y quadrant at x = 0.925 (outside the pipe grasp corridor)
  The layout2 GAP_SENSOR point is moved out to y = -0.43 for the same reason: the laser gap sensor looks down at the
  upper -Y quadrant of seam B from 0.37 m (visible above the pipe in the front shots), on a bracket post (top
  z = 0.73, world 1.53 m) that the handler carries the pipe over (plan2: pipe placed with lift 0.55).
* station lamp (set_lamp: 0 off, 1 green) on its own slim post at the front -Y corner next to the operator panel,
  wired to the valve-terminal duct (visible from the front shots; low, clear of the handler's approach paths).
* pneumatic valve terminal (4 valves) under the -Y apron, PVC duct along the apron, PU tubes (2 per clamp) to the
  cylinders, M12 sensor / lamp cables, air supply hose; operator panel on the front apron; earth clamp on the +Y edge.

Public API (DESIGN.md section 3):
    build(collection=None) -> dict(collection, clamps, lamp, table, frame, root, objects, sensor, supports)
        clamps[name] = dict(root, lever, cyl, rod, spec)   lever.rotation_euler[0], cyl.rotation_euler[0], rod.location[1]
    set_clamp(st, name, s, frame=None)     s = 0 open, 1 closed (keys the lever / cylinder / rod of that clamp)
    set_lamp(st, state, frame=None)        0 off, 1 green (object property 'lamp_on', CONSTANT key)
    clamp_specs()                          pure-python clamp geometry (station-local), clamp_points(spec, s)
    obstacles() -> list[dict(name, center, size, yaw)]   conservative world AABBs (pure python)
"""
import math

import numpy as np
import bpy
import mathutils

import tools  # noqa: F401  (demo_video on sys.path)
from cell import geom as G
from cell import layout as L
from cell import materials
from cell import environment as E
import cassette as C
import kin
import layout2 as L2

NAME = "AssemblyStation"
PREFIX = "stn"
GAP = 0.0003                       # design clearance part <-> support / pad (visually touching)
DEG = math.pi / 180.0

FRAME = kin.planar_frame(L2.STATION_ORIGIN, L2.STATION_XDIR)
R_PIPE = L2.PIPE_R
R_BEND = L.ELBOW_R
Z_A = L2.SEAM_A_Z
Z_P = L2.PIPE_AXIS_Z
FL_THK = L.FLANGE_THK
FL_R = L.FLANGE_OD / 2

# ------------------------------------------------------------------ table (station-local)
TOP = L2.STATION_TABLE["top_z"] - L2.STATION_ORIGIN[2]      # -0.02: table top (seat plate stands on it)
FLOOR = -L2.STATION_ORIGIN[2]                                # -0.80
PLATE_T = 0.02
APRON_H, APRON_W = 0.10, 0.05
LEG = 0.08


def _world_to_local(p):
    return (kin.inv(FRAME) @ np.array([p[0], p[1], p[2], 1.0]))[:3]


def _table_rect():
    cx, cy = L2.STATION_TABLE["center"]
    sx, sy = L2.STATION_TABLE["size"]
    pts = [_world_to_local((cx + dx * sx / 2, cy + dy * sy / 2, 0.0)) for dx in (-1, 1) for dy in (-1, 1)]
    return (min(p[0] for p in pts), max(p[0] for p in pts)), (min(p[1] for p in pts), max(p[1] for p in pts))


TX, TY = _table_rect()                                       # x (-0.60, 1.10), y (-0.55, 0.55)

# ------------------------------------------------------------------ supports
SEAT_R = 0.235                     # seat plate (layout2 flange_seat radius 0.215 + room for the pins and screws)
SEAT_RECESS = (0.195, 0.007)       # recess for the flange raised face (r 0.176, 2 mm proud) and the stage-1 bolt-hole
                                   # discs (PCD 355, D26, also 2 mm proud of the back face: r <= 0.1905)
PIN_R, PIN_H = 0.0065, 0.018
PIN_POS = FL_R + 0.0015 + PIN_R    # centring pins just outside the flange rim
PIN_ANG = (30.0, 150.0, 270.0)
SEAT_SCREW_ANG = (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)

EV_X = L2.STATION_SUPPORTS["elbow_vblock"]["center"][0]      # 0.30
EV_LEN, EV_W, EV_BETA, EV_DEPTH = 0.045, 0.100, 40.0 * DEG, 0.075
PV_X = L2.STATION_SUPPORTS["pipe_vblock"]["center"][0]       # 0.86
PV_LEN, PV_W, PV_BETA, PV_DEPTH = 0.050, 0.115, 45.0 * DEG, 0.070
LINER_T = 0.008
STOP_X = L2.PIPE_END_X + GAP                                  # stop face
STOP_PLATE = dict(t=0.014, y=0.10, z=(Z_P - R_PIPE - 0.015, Z_P + 0.035))

# ------------------------------------------------------------------ clamps
PAD_T = 0.010                      # swivel pad incl. its 3 mm rubber face
SPINDLE_L = 0.040
LEVER_T = 0.022                    # lever thickness along the swing axis
POST_A = (-0.078, -0.028)          # clamp post / brackets beside the lever plane (clamp-local a range)
POST_U = (-0.07, 0.01)
CYL_R, CAP, CAP_L, EYE = 0.021, 0.046, 0.028, 0.012
ROD_R = 0.008
CLAMP_DESIGN = {
    "flange_l": dict(kind="flange", ang=210.0, pad_r=0.189, pivot_r=0.37, pad=0.011, open=105.0),
    "flange_r": dict(kind="flange", ang=330.0, pad_r=0.189, pivot_r=0.37, pad=0.011, open=105.0),
    "elbow": dict(kind="elbow", x=0.285, psi=-50.0, pivot=(-0.365, 0.42), pad=0.016, open=100.0),
    "pipe": dict(kind="pipe", x=0.925, psi=-40.0, pivot=(-0.30, 0.46), pad=0.016, open=100.0),
}
assert set(CLAMP_DESIGN) == set(L2.STATION_CLAMPS)

# ------------------------------------------------------------------ gap sensor, valve terminal, panel
SENSOR = dict(pos=(L2.GAP_SENSOR["local"][0], -0.43, 0.66), aim_b=120.0, post=(0.46, -0.47), post_top=0.73)
LAMP = dict(post=(1.045, -0.50), post_top=0.55)                   # status lamp post at the front -Y corner
# valve terminal hung under the -Y apron (inside the table footprint: the pipe buffer rack stands 0.10 m beyond the
# table's -Y edge), PVC duct on the apron's outer face above it; the port stubs rise from the valves into the duct
ZAB = TOP - PLATE_T - APRON_H                                        # apron bottom
VALVE = dict(x=(-0.49, -0.22), y=(TY[0] + 0.004, TY[0] + 0.069), z=(ZAB - 0.12, ZAB))
DUCT = dict(x=(VALVE["x"][0], TX[1] - 0.08), y=(TY[0] - 0.05, TY[0]), z=(TOP - PLATE_T - 0.10, TOP - PLATE_T - 0.005))
PANEL = dict(x=(TX[1], TX[1] + 0.075), y=(-0.45, -0.19), z=(TOP - 0.23, TOP - 0.05))
EARTH = (0.62, TY[1])              # welding earth clamp on the +Y table edge


# ============================================================================ pure-python geometry
def elbow_surface(x0, x1, n_th=500, n_a=720):
    """Outer surface points of the stage-1 elbow (spool-local) with x in [x0, x1]."""
    th = np.linspace(0.0, math.pi / 2, n_th)[:, None]
    a = np.linspace(0.0, 2 * math.pi, n_a, endpoint=False)[None, :]
    cx, cz = R_BEND * (1 - np.cos(th)), Z_A + R_BEND * np.sin(th)
    px = cx + R_PIPE * np.cos(a) * (-np.cos(th))
    py = R_PIPE * np.sin(a) * np.ones_like(th)
    pz = cz + R_PIPE * np.cos(a) * np.sin(th)
    m = (px >= x0) & (px <= x1)
    return np.stack([px[m], py[m], pz[m]], axis=1)


def elbow_distance(p):
    """Signed distance (m) from spool-local points (n,3) to the elbow's outer surface (quarter torus region only;
    +inf outside it)."""
    p = np.atleast_2d(np.asarray(p, float))
    dx, dz = p[:, 0] - R_BEND, p[:, 2] - Z_A
    rho = np.hypot(dx, dz)
    d = np.hypot(rho - R_BEND, p[:, 1]) - R_PIPE
    inside_range = (dx <= 1e-9) & (dz >= -1e-9)
    return np.where(inside_range, d, np.inf)


def pipe_distance(p):
    """Signed distance to the pipe's outer surface (x within the pipe, else +inf)."""
    p = np.atleast_2d(np.asarray(p, float))
    d = np.hypot(p[:, 1], p[:, 2] - Z_P) - R_PIPE
    return np.where((p[:, 0] >= L2.SEAM_B_X) & (p[:, 0] <= L2.PIPE_END_X), d, np.inf)


def elbow_vblock():
    """Frame and apex of the elbow V-block: groove along the elbow's intrados tangent at EV_X (tilt gamma about -Y),
    apex offset along the block normal such that the liners touch the elbow (+GAP).  Returns dict(O, g, n, gamma,
    apex) with the block frame axes g (groove), y, n (up) and the apex at O + apex * n."""
    th_i = math.acos((R_BEND - EV_X) / (R_BEND - R_PIPE))
    gamma = math.pi / 2 - th_i
    O = np.array([EV_X, 0.0, Z_A + (R_BEND - R_PIPE) * math.sin(th_i)])
    g = np.array([math.cos(gamma), 0.0, math.sin(gamma)])
    n = np.array([-math.sin(gamma), 0.0, math.cos(gamma)])
    pts = elbow_surface(EV_X - 0.08, EV_X + 0.08) - O
    qx, qy, qz = pts @ g, pts[:, 1], pts @ n
    m = (np.abs(qx) <= EV_LEN / 2) & (np.abs(qy) <= EV_W)
    apex = float(np.min(qz[m] - np.abs(qy[m]) * math.tan(EV_BETA) - GAP / math.cos(EV_BETA)))
    return dict(O=O, g=g, n=n, gamma=gamma, apex=apex)


def pipe_vblock_apex():
    return Z_P - (R_PIPE + GAP) / math.cos(PV_BETA)


def _rot2(p, a):
    c, s = math.cos(a), math.sin(a)
    return np.array([c * p[0] - s * p[1], s * p[0] + c * p[1]])


def clamp_specs():
    """Clamp geometry (pure python).  Per clamp: frame R (columns a, u, v in station-local coords: a = swing axis,
    u = toward the part, v = up), pivot (station-local), and in clamp (a, u, v) coordinates at s = 1: contact point P,
    outward surface normal n, pad back centre Q, spindle top S, tail pin T, cylinder pin Cp (u, v), open angle,
    cylinder body / rod lengths."""
    out = {}
    for name, d in CLAMP_DESIGN.items():
        if d["kind"] == "flange":
            ang = d["ang"] * DEG
            radial = np.array([math.cos(ang), math.sin(ang), 0.0])
            u = -radial
            P = np.array([d["pad_r"] * radial[0], d["pad_r"] * radial[1], FL_THK])
            n = np.array([0.0, 0.0, 1.0])
            zp = FL_THK + GAP + PAD_T + SPINDLE_L
            pivot = np.array([d["pivot_r"] * radial[0], d["pivot_r"] * radial[1], zp])
        elif d["kind"] == "elbow":
            psi = d["psi"] * DEG
            ct = (R_BEND - d["x"]) / (R_BEND + R_PIPE * math.cos(psi))
            th = math.acos(ct)
            Cc = np.array([R_BEND * (1 - ct), 0.0, Z_A + R_BEND * math.sin(th)])
            e1 = np.array([-ct, 0.0, math.sin(th)])
            n = math.cos(psi) * e1 + math.sin(psi) * np.array([0.0, 1.0, 0.0])
            P = Cc + R_PIPE * n
            u = np.array([0.0, 1.0, 0.0])
            pivot = np.array([P[0], d["pivot"][0], d["pivot"][1]])
        else:
            psi = d["psi"] * DEG
            n = np.array([0.0, math.sin(psi), math.cos(psi)])
            P = np.array([d["x"], 0.0, Z_P]) + R_PIPE * n
            u = np.array([0.0, 1.0, 0.0])
            pivot = np.array([d["x"], d["pivot"][0], d["pivot"][1]])
        v = np.array([0.0, 0.0, 1.0])
        a = np.cross(u, v)
        R = np.stack([a, u, v], axis=1)
        Pc = R.T @ (P - pivot)
        nc = R.T @ n
        Q = Pc + nc * (GAP + PAD_T)
        n2 = np.array([0.0, nc[1], nc[2]])
        n2 /= np.linalg.norm(n2)
        S = np.array([0.0, Q[1], Q[2]]) + n2 * SPINDLE_L
        alpha = math.atan2(S[2], S[1])
        Lt = 0.045 if d["kind"] == "flange" else 0.05
        tail = alpha - math.pi / 2 if d["kind"] == "flange" else alpha + math.pi
        T = Lt * np.array([math.cos(tail), math.sin(tail)])
        Cp = np.array([-0.20, T[1]]) if d["kind"] == "flange" else np.array([-0.13, -0.24])
        op = d["open"] * DEG
        ds = [float(np.linalg.norm(_rot2(T, t) - Cp)) for t in np.linspace(0.0, op, 41)]
        Lb = min(ds) - 0.035
        Lr = max(ds) - Lb + 0.02
        out[name] = dict(name=name, kind=d["kind"], R=R, pivot=pivot, P=Pc, n=nc, Q=Q, S=S, n2=n2, T=T, Cp=Cp,
                         alpha=alpha, open=op, Lb=Lb, Lr=Lr, pad_r=d["pad"],
                         lever_h=0.028 if d["kind"] == "flange" else 0.032)
    return out


def clamp_state(cs, s):
    """(lever angle, cylinder angle, rod extension) for command s (0 open .. 1 closed)."""
    s = min(max(float(s), 0.0), 1.0)
    th = (1.0 - s) * cs["open"]
    T = _rot2(cs["T"], th)
    dv = T - cs["Cp"]
    return th, math.atan2(dv[1], dv[0]), float(np.linalg.norm(dv))


def clamp_points(cs, s):
    """Key points (station-local, (n,3)) of the moving clamp parts at command s: lever outline, spindle, pad rim."""
    th = clamp_state(cs, s)[0]
    h = cs["lever_h"] / 2 + 0.004
    pts2 = []                                            # (u, v) in the closed pose
    for p in (np.zeros(2), cs["S"][1:], cs["T"]):
        for du in (-h, h):
            for dv in (-h, h):
                pts2.append(p + np.array([du, dv]))
    Qp = cs["Q"][1:]
    for k in range(12):                                  # pad rim (approximated in the lever plane)
        ang = 2 * math.pi * k / 12
        t = np.array([-cs["n2"][2], cs["n2"][1]])
        pts2.append(Qp + cs["pad_r"] * math.cos(ang) * t - cs["n2"][1:] * (PAD_T * 0.5 * (1 + math.sin(ang))))
    out = []
    for p in pts2:
        q = _rot2(p, th)
        for a in (-LEVER_T / 2 - 0.012, LEVER_T / 2 + 0.012):
            out.append(cs["pivot"] + cs["R"] @ np.array([a, q[0], q[1]]))
    return np.array(out)


def _box_pts(lo, hi):
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def clamp_static_points(cs):
    """Corner points (station-local) of a clamp's static mount (post, base plate, cylinder bracket, cylinder sweep)."""
    R, pv = cs["R"], cs["pivot"]
    v_top = TOP - pv[2]                                   # table top in clamp v
    u0 = min(cs["Cp"][0] - 0.035, POST_U[0] - 0.04)
    pts = [_box_pts((POST_A[0] - 0.025, u0, v_top), (0.055, POST_U[1] + 0.025, 0.03 + 0.01))]
    for s in np.linspace(0.0, 1.0, 9):                   # cylinder body + rod sweep
        _, phi, ext = clamp_state(cs, s)
        dirv = np.array([math.cos(phi), math.sin(phi)])
        for t in (-EYE - 0.01, ext + 0.02):
            q = cs["Cp"] + t * dirv
            for a in (-CAP / 2 - 0.004, CAP / 2 + 0.004):
                for w in (-CAP / 2 - 0.004, CAP / 2 + 0.004):
                    pts.append([[a, q[0] - w * dirv[1], q[1] + w * dirv[0]]])
    loc = []
    for block in pts:
        for p in np.atleast_2d(block):
            loc.append(pv + R @ np.asarray(p, float))
    return np.array(loc)


# ============================================================================ bpy helpers
def _mats():
    return dict(
        frame=materials.get("painted", color="#2B2F36", roughness=0.5),
        blue=materials.get("painted", color="#1F4E8C", roughness=0.42),
        top=materials.get("cast_iron"),
        steel=materials.get("machined_steel"),
        chrome=materials.get("bevel_steel"),
        dark=materials.get("dark_metal"),
        black=materials.get("black_plastic"),
        rubber=materials.get("rubber"),
        alu=materials.get("painted", color="#AEB2B7", roughness=0.32, coat=0.0),
        lever=materials.get("painted", color="#33363A", roughness=0.45, coat=0.0),
        brass=materials.get("brass"),
        copper=materials.get("copper"),
        bronze=_bronze(),
        yellow=materials.get("safety_yellow"),
        red=materials.get("safety_red"),
        cable=materials.get("cable_black"),
        tube_blue=materials.get("painted", color="#2F74C0", roughness=0.35, coat=0.0),
        grey=materials.get("painted", color="#B9BCBE", roughness=0.5),
        sensor=materials.get("painted", color="#1E2A44", roughness=0.35),
        glass=materials.get("glass_dark"),
        green_btn=materials.get("painted", color="#2ECC40", roughness=0.3, coat=0.4),
        white_btn=materials.get("painted", color="#E8E8E8", roughness=0.3, coat=0.4),
        led=materials.get("emissive", color="#30FF40", strength=8.0),
        laser=materials.get("emissive", color="#FF2A10", strength=6.0),
        lamp=C._state_mat("stn_lamp_green", (0.19, 1.0, 0.25), 14.0, "lamp_on"),
    )


def _bronze():
    m = bpy.data.materials.get("stn_bronze")
    if m is None:
        m = bpy.data.materials.new("stn_bronze")
        m.use_nodes = True
        b = m.node_tree.nodes["Principled BSDF"]
        b.inputs["Base Color"].default_value = (0.50, 0.33, 0.15, 1.0)
        b.inputs["Metallic"].default_value = 1.0
        b.inputs["Roughness"].default_value = 0.42
    return m


def _attach(objs, parent):
    """Parent objects (whose location/rotation are given in the parent's frame) with an identity parent inverse."""
    for ob in objs:
        ob.parent = parent
        ob.matrix_parent_inverse = mathutils.Matrix.Identity(4)


def _empty(col, name, parent, M=None, size=0.05):
    e = G.empty(name, collection=col, size=size)
    e.parent = parent
    e.matrix_parent_inverse = mathutils.Matrix.Identity(4)
    e.matrix_basis = M if M is not None else mathutils.Matrix.Identity(4)
    return e


def _rot_to(d):
    """Euler rotation that maps local +Z onto direction d."""
    return mathutils.Vector(tuple(float(x) for x in d)).to_track_quat('Z', 'Y').to_euler()


def _disc(b, name, r, t, mat, center, direction, segs=24, chamfer=0.0):
    """Flat cylinder (axis = direction), its base face at center."""
    prof = [(0, 0), (r, 0), (r, t - chamfer), (r - chamfer, t), (0, t)] if chamfer > 0 else [(0, 0), (r, 0), (r, t), (0, t)]
    return E._rev(b, name, prof, mat, tuple(center), _rot_to(direction), segs=segs)


# ============================================================================ table
def _table(b, M, lm):
    x0, x1 = TX
    y0, y1 = TY
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    zt = TOP - PLATE_T
    top = C._bx(b, f"{PREFIX}_table_top", (x1 - x0, y1 - y0, PLATE_T), (cx, cy, TOP - PLATE_T / 2), M["top"], bev=0.003)
    # D28 hole grid, 100 mm pitch (one mesh of flat dark discs just above the plate)
    verts, faces = [], []
    nseg = 12
    xs = np.arange(x0 + 0.05, x1 - 0.02, 0.10)
    ys = np.arange(y0 + 0.05, y1 - 0.02, 0.10)
    for hx in xs:
        for hy in ys:
            if math.hypot(hx, hy) < SEAT_R - 0.02:
                continue
            k0 = len(verts)
            verts += [(hx + 0.014 * math.cos(2 * math.pi * i / nseg), hy + 0.014 * math.sin(2 * math.pi * i / nseg), TOP + 0.0004)
                      for i in range(nseg)]
            faces.append(tuple(range(k0, k0 + nseg)))
    E._mesh(b, f"{PREFIX}_table_holes", verts, faces, M["black"])
    # apron (square tubes under the plate), legs, feet, stretchers
    za = zt - APRON_H / 2
    w = APRON_W
    C._bx(b, f"{PREFIX}_apron_xn", (x1 - x0, w, APRON_H), (cx, y0 + w / 2, za), M["frame"], bev=0.004)
    C._bx(b, f"{PREFIX}_apron_xp", (x1 - x0, w, APRON_H), (cx, y1 - w / 2, za), M["frame"], bev=0.004)
    C._bx(b, f"{PREFIX}_apron_yn", (w, y1 - y0 - 2 * w, APRON_H), (x0 + w / 2, cy, za), M["frame"], bev=0.004)
    C._bx(b, f"{PREFIX}_apron_yp", (w, y1 - y0 - 2 * w, APRON_H), (x1 - w / 2, cy, za), M["frame"], bev=0.004)
    C._bx(b, f"{PREFIX}_apron_mid", (w, y1 - y0 - 2 * w, APRON_H), (cx, cy, za), M["frame"], bev=0.004)
    legs_x = (x0 + LEG / 2 + 0.01, cx, x1 - LEG / 2 - 0.01)
    legs_y = (y0 + LEG / 2 + 0.01, y1 - LEG / 2 - 0.01)
    zf = FLOOR + 0.045
    for i, lx in enumerate(legs_x):
        for j, ly in enumerate(legs_y):
            C._bx(b, f"{PREFIX}_leg{i}{j}", (LEG, LEG, zt - APRON_H - zf + 0.001), (lx, ly, (zt - APRON_H + zf) / 2),
                  M["frame"], bev=0.004)
            E._box(b, f"{PREFIX}_leg{i}{j}_cap", (LEG + 0.02, LEG + 0.02, 0.008), (lx, ly, zf - 0.004), M["frame"])
            E._rod(b, f"{PREFIX}_leg{i}{j}_stud", 0.011, 0.03, M["steel"], (lx, ly, zf - 0.023), segs=10)
            E._rev(b, f"{PREFIX}_leg{i}{j}_foot", [(0, 0), (0.042, 0), (0.042, 0.008), (0.03, 0.016), (0, 0.016)], M["dark"],
                   (lx, ly, FLOOR), segs=20)
    zs = FLOOR + 0.17
    for j, ly in enumerate(legs_y):
        C._bx(b, f"{PREFIX}_stretch_x{j}", (x1 - x0 - 0.12, 0.05, 0.05), (cx, ly, zs), M["frame"], bev=0.003)
    for i, lx in enumerate((legs_x[0], legs_x[2])):
        C._bx(b, f"{PREFIX}_stretch_y{i}", (0.05, y1 - y0 - 0.12, 0.05), (lx, cy, zs), M["frame"], bev=0.003)
    # station label on the front apron (+X local = world -Y, faces the handler track)
    C._label(b, f"{PREFIX}_label_station", lm, 0, (0.30, 0.06), (x1 + 0.0015, 0.12, za),
             (math.pi / 2, 0.0, math.pi / 2), backing=M["frame"])
    return top


def _panel(b, M, lm):
    """Operator panel (push-button box) on the front apron: start / stop / reset, key switch, e-stop."""
    (px0, px1), (py0, py1), (pz0, pz1) = PANEL["x"], PANEL["y"], PANEL["z"]
    pc = ((px0 + px1) / 2, (py0 + py1) / 2, (pz0 + pz1) / 2)
    C._bx(b, f"{PREFIX}_panel_box", (px1 - px0, py1 - py0, pz1 - pz0), pc, M["grey"], bev=0.006)
    xf = px1
    E._box(b, f"{PREFIX}_panel_face", (0.003, py1 - py0 - 0.03, pz1 - pz0 - 0.05), (xf + 0.0015, pc[1], pc[2] - 0.01), M["frame"])
    xd = (1.0, 0.0, 0.0)
    zb = pc[2] + 0.015
    items = [("start", M["green_btn"], py0 + 0.045), ("stop", M["red"], py0 + 0.095), ("reset", M["white_btn"], py0 + 0.145)]
    for nm, mat, yy in items:
        _disc(b, f"{PREFIX}_panel_{nm}_ring", 0.016, 0.006, M["steel"], (xf, yy, zb), xd, segs=20)
        _disc(b, f"{PREFIX}_panel_{nm}", 0.011, 0.014, mat, (xf, yy, zb), xd, segs=16, chamfer=0.002)
    _disc(b, f"{PREFIX}_panel_key", 0.014, 0.012, M["dark"], (xf, py0 + 0.20, zb), xd, segs=16)
    E._box(b, f"{PREFIX}_panel_key_tab", (0.012, 0.004, 0.018), (xf + 0.016, py0 + 0.20, zb), M["steel"])
    for k, yy in enumerate((py0 + 0.045, py0 + 0.095, py0 + 0.145)):
        E._box(b, f"{PREFIX}_panel_tag{k}", (0.002, 0.03, 0.01), (xf + 0.001, yy, zb - 0.032), M["white_btn"])
    E._box(b, f"{PREFIX}_panel_led", (0.004, 0.012, 0.006), (xf + 0.002, py1 - 0.03, zb + 0.035), M["led"])
    # e-stop on the panel's -Y end face (yellow collar, red mushroom)
    _disc(b, f"{PREFIX}_panel_estop_ring", 0.03, 0.012, M["yellow"], (pc[0], py0, pc[2]), (0.0, -1.0, 0.0), segs=24)
    E._rev(b, f"{PREFIX}_panel_estop", [(0, 0), (0.012, 0), (0.012, 0.012), (0.024, 0.016), (0.024, 0.026), (0.018, 0.031), (0, 0.032)],
           M["red"], (pc[0], py0 - 0.012, pc[2]), _rot_to((0.0, -1.0, 0.0)), segs=24)
    C._label(b, f"{PREFIX}_label_panel", lm, 5, (0.12, 0.028), (xf + 0.0015, pc[1] + 0.03, pz1 - 0.016),
             (math.pi / 2, 0.0, math.pi / 2))


# ============================================================================ supports
def _seat(b, M):
    rr, rd = SEAT_RECESS
    prof = [(0.0, TOP), (SEAT_R, TOP), (SEAT_R, -0.003), (SEAT_R - 0.003, -GAP), (rr, -GAP), (rr, -rd), (0.0, -rd)]
    E._rev(b, f"{PREFIX}_seat", prof, M["steel"], segs=96, smooth=False)
    E._rev(b, f"{PREFIX}_seat_recess", [(0.0, -rd), (rr - 0.0005, -rd), (rr - 0.0005, -rd + 0.0004), (0.0, -rd + 0.0004)],
           M["dark"], segs=64, smooth=False)
    for k, a in enumerate(PIN_ANG):
        c = (PIN_POS * math.cos(a * DEG), PIN_POS * math.sin(a * DEG), -GAP)
        E._rev(b, f"{PREFIX}_seat_pin{k}", [(0, 0), (PIN_R, 0), (PIN_R, PIN_H - 0.004), (PIN_R - 0.003, PIN_H), (0, PIN_H)],
               M["chrome"], c, segs=16)
    for k, a in enumerate(SEAT_SCREW_ANG):
        c = (0.2245 * math.cos(a * DEG), 0.2245 * math.sin(a * DEG), -GAP)
        E._rev(b, f"{PREFIX}_seat_screw{k}", [(0, 0), (0.0065, 0), (0.0065, 0.005), (0.004, 0.005), (0.004, 0.0035), (0, 0.0035)],
               M["dark"], c, segs=12)


def _v_prism(b, name, apex, beta, w, flat, depth, length, mat, liner_mat, center, rot, span=None):
    """V-block (outline in the (y, z) plane, extruded along local X), design V surface = liner tops, apex at z = apex
    (block coords), bottom at apex - depth.  Returns the objects."""
    tb = math.tan(beta)
    za = apex - LINER_T / math.cos(beta)
    zb = apex - depth
    body = [(-w, zb), (w, zb), (w, za + tb * w), (flat, za + tb * flat), (-flat, za + tb * flat), (-w, za + tb * w)]
    objs = [E._prism(b, name, body, length, mat, center, rot, axis='X')]
    ua, ub = span or (flat + 0.01, w)
    for k, s in enumerate((1, -1)):
        q = [(ua, za + tb * ua), (ub, za + tb * ub), (ub, apex + tb * ub), (ua, apex + tb * ua)]
        if s < 0:
            q = [(-u, v) for u, v in q][::-1]
        objs.append(E._prism(b, f"{name}_liner{k}", q, length, liner_mat, center, rot, axis='X'))
    return objs


def _foot(b, M, name, cx, cy, sx, sy):
    """Bolted foot plate on the table top with 4 socket-head screws."""
    E._box(b, name, (sx, sy, 0.012), (cx, cy, TOP + 0.006), M["frame"])
    for k, (dx, dy) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
        E._rev(b, f"{name}_bolt{k}", [(0, 0), (0.0065, 0), (0.0065, 0.006), (0, 0.006)], M["dark"],
               (cx + dx * (sx / 2 - 0.013), cy + dy * (sy / 2 - 0.013), TOP + 0.012), segs=10)


def _elbow_vblock(b, M):
    vb = elbow_vblock()
    O, gam = vb["O"], vb["gamma"]
    rot = (0.0, -gam, 0.0)                                    # block X -> groove direction (rising toward +x)
    objs = _v_prism(b, f"{PREFIX}_ev", vb["apex"], EV_BETA, EV_W, 0.012, EV_DEPTH, EV_LEN, M["blue"], M["bronze"],
                    tuple(O), rot, span=(0.018, EV_W))
    # riser: from the table up into the block bottom (lowest corner of the tilted bottom face)
    zb = vb["apex"] - EV_DEPTH
    xr = O[0] - zb * math.sin(gam)                            # under the centre of the tilted bottom face
    lowest = O[2] + zb * math.cos(gam) - (EV_LEN / 2) * math.sin(gam)
    top = lowest + 0.02
    C._bx(b, f"{PREFIX}_ev_riser", (0.04, 0.12, top - TOP), (xr, 0.0, (top + TOP) / 2), M["blue"], bev=0.004)
    for k, s in enumerate((-1, 1)):                           # gussets along the riser
        E._prism(b, f"{PREFIX}_ev_gusset{k}", [(0.0, TOP + 0.012), (0.05, TOP + 0.012), (0.0, TOP + 0.11)], 0.012, M["blue"],
                 (xr + s * 0.02, 0.0, 0.0), (0.0, 0.0, 0.0 if s > 0 else math.pi), axis='Y')
    _foot(b, M, f"{PREFIX}_ev_foot", xr, 0.0, 0.15, 0.17)
    return objs


def _pipe_vblock(b, M):
    apex = pipe_vblock_apex()
    objs = _v_prism(b, f"{PREFIX}_pv", apex, PV_BETA, PV_W, 0.012, PV_DEPTH, PV_LEN, M["blue"], M["bronze"],
                    (PV_X, 0.0, 0.0), (0.0, 0.0, 0.0), span=(0.018, PV_W))
    top = apex - PV_DEPTH + 0.01
    C._bx(b, f"{PREFIX}_pv_riser", (0.06, 0.13, top - TOP), (PV_X, 0.0, (top + TOP) / 2), M["blue"], bev=0.004)
    for k, s in enumerate((-1, 1)):
        E._prism(b, f"{PREFIX}_pv_gusset{k}", [(0.0, TOP + 0.012), (0.06, TOP + 0.012), (0.0, TOP + 0.14)], 0.012, M["blue"],
                 (PV_X + s * 0.03, 0.0, 0.0), (0.0, 0.0, 0.0 if s > 0 else math.pi), axis='Y')
    _foot(b, M, f"{PREFIX}_pv_foot", PV_X, 0.0, 0.16, 0.18)
    return objs


def _pipe_stop(b, M):
    t, hy = STOP_PLATE["t"], STOP_PLATE["y"]
    z0, z1 = STOP_PLATE["z"]
    x0 = STOP_X
    objs = [E._box(b, f"{PREFIX}_stop_face", (t, 2 * hy, z1 - z0), (x0 + t / 2, 0.0, (z0 + z1) / 2), M["bronze"])]
    xb = x0 + t
    C._bx(b, f"{PREFIX}_stop_block", (0.04, 2 * hy - 0.02, z1 - z0 - 0.01), (xb + 0.02, 0.0, (z0 + z1) / 2), M["blue"], bev=0.004)
    xp = xb + 0.04
    C._bx(b, f"{PREFIX}_stop_post", (0.05, 0.10, z1 - TOP), (xp + 0.025, 0.0, (z1 + TOP) / 2), M["blue"], bev=0.004)
    for k, s in enumerate((-1, 1)):
        E._prism(b, f"{PREFIX}_stop_gusset{k}", [(0.0, TOP + 0.012), (0.07, TOP + 0.012), (0.0, TOP + 0.20)], 0.012, M["blue"],
                 (xp + 0.05, s * 0.035, 0.0), (0.0, 0.0, 0.0), axis='Y')
    _foot(b, M, f"{PREFIX}_stop_foot", xp + 0.045, 0.0, 0.14, 0.16)
    # adjusting screw with lock nut through the block (sets the root gap)
    zs = Z_P - 0.04
    E._rod(b, f"{PREFIX}_stop_screw", 0.009, 0.11, M["steel"], (xb + 0.055, 0.045, zs), (0.0, math.pi / 2, 0.0), segs=12)
    C._hex(b, f"{PREFIX}_stop_nut", 0.015, 0.012, M["dark"], (xp + 0.001, 0.045, zs), (0.0, math.pi / 2, 0.0))
    C._hex(b, f"{PREFIX}_stop_screw_head", 0.013, 0.01, M["dark"], (xp + 0.06, 0.045, zs), (0.0, math.pi / 2, 0.0))
    return objs


# ============================================================================ clamps
def _clamp(col, M, lm, name, cs, parent, k):
    """One lever clamp.  Clamp-local coordinates: X = a (swing axis), Y = u (toward the part), Z = v (up), origin at
    the pivot; the lever is built in its closed pose (rotation 0)."""
    Rm = mathutils.Matrix(np.eye(4).tolist())
    for i in range(3):
        for j in range(3):
            Rm[i][j] = float(cs["R"][i, j])
        Rm[i][3] = float(cs["pivot"][i])
    root = _empty(col, f"{PREFIX}_clamp_{name}", parent, Rm, size=0.04)
    pre = f"{PREFIX}_cl_{name}"
    v_top = TOP - float(cs["pivot"][2])                  # table top in clamp v
    Cu, Cv = float(cs["Cp"][0]), float(cs["Cp"][1])
    a0, a1 = POST_A
    am = (a0 + a1) / 2
    # ---- static mount: base plate, post, pivot pin, cylinder bracket + trunnion pin
    bs = E._B(col)
    u0 = min(Cu - 0.035, POST_U[0] - 0.035)
    pa0, pa1 = a0 - 0.02, 0.045
    E._box(bs, f"{pre}_base", (pa1 - pa0, POST_U[1] + 0.02 - u0, 0.012), ((pa0 + pa1) / 2, (u0 + POST_U[1] + 0.02) / 2, v_top + 0.006),
           M["frame"])
    for j, uu in enumerate((u0 + 0.013, POST_U[1] + 0.007)):
        for i, aa in enumerate((pa0 + 0.012, pa1 - 0.012)):
            E._rev(bs, f"{pre}_base_bolt{i}{j}", [(0, 0), (0.0065, 0), (0.0065, 0.006), (0, 0.006)], M["dark"],
                   (aa, uu, v_top + 0.012), segs=10)
    ph = 0.03 - (v_top + 0.012)
    C._bx(bs, f"{pre}_post", (a1 - a0, POST_U[1] - POST_U[0], ph), (am, (POST_U[0] + POST_U[1]) / 2, v_top + 0.012 + ph / 2),
          M["blue"], bev=0.003)
    E._rod(bs, f"{pre}_pin", 0.0085, a1 + LEVER_T / 2 + 0.012 - a0 + 0.004, M["chrome"],
           ((a0 - 0.004 + a1 + LEVER_T / 2 + 0.012) / 2, 0.0, 0.0), (0.0, math.pi / 2, 0.0), segs=12)
    C._hex(bs, f"{pre}_pin_nut", 0.012, 0.008, M["dark"], (LEVER_T / 2 + 0.004, 0.0, 0.0), (0.0, math.pi / 2, 0.0))
    ab0, ab1 = -0.052, -0.029                             # cylinder bracket beside the cylinder eye
    if Cv - v_top < 0.10:                                 # low cylinder: bracket block from the base plate
        E._box(bs, f"{pre}_cbracket", (ab1 - ab0, 0.05, Cv + 0.02 - v_top - 0.012), ((ab0 + ab1) / 2, Cu, (Cv + 0.02 + v_top + 0.012) / 2),
               M["blue"])
    else:                                                 # arm from the post
        E._box(bs, f"{pre}_cbracket", (a1 - a0, POST_U[0] - (Cu - 0.03), 0.045), (am, (POST_U[0] + Cu - 0.03) / 2, Cv), M["blue"])
        E._box(bs, f"{pre}_cbracket2", (ab1 - ab0, 0.05, 0.045), ((ab0 + ab1) / 2, Cu, Cv), M["blue"])
    E._rod(bs, f"{pre}_cpin", 0.007, EYE + 0.004 - ab0 + 0.004, M["chrome"], ((ab0 - 0.004 + EYE + 0.004) / 2, Cu, Cv),
           (0.0, math.pi / 2, 0.0), segs=10)
    C._label(bs, f"{pre}_tag", lm, 1 + k, (0.04, 0.02), (a0 - 0.0015, (POST_U[0] + POST_U[1]) / 2, ph / 2 + v_top),
             (math.pi / 2, 0.0, -math.pi / 2))
    _attach(bs.objs, root)
    # ---- lever (empty) + parts
    lever = _empty(col, f"{pre}_lever", root, size=0.03)
    lever.rotation_mode = 'XYZ'
    bl = E._B(col)
    S, T = cs["S"], cs["T"]
    h = cs["lever_h"]
    nose_len = float(np.hypot(S[1], S[2]))
    E._box(bl, f"{pre}_arm", (LEVER_T, nose_len + 0.02, h), (0.0, S[1] / 2, S[2] / 2), M["lever"], (cs["alpha"], 0.0, 0.0))
    tl = float(np.hypot(T[0], T[1]))
    ta = math.atan2(T[1], T[0])
    E._box(bl, f"{pre}_tail", (LEVER_T, tl + 0.018, h * 0.9), (0.0, T[0] / 2, T[1] / 2), M["lever"], (ta, 0.0, 0.0))
    E._rod(bl, f"{pre}_boss", 0.021, LEVER_T + 0.004, M["lever"], (0.0, 0.0, 0.0), (0.0, math.pi / 2, 0.0), segs=20)
    E._rod(bl, f"{pre}_nose", h / 2 + 0.003, LEVER_T + 0.002, M["lever"], (0.0, float(S[1]), float(S[2])), (0.0, math.pi / 2, 0.0), segs=16)
    # spindle through the nose, two lock nuts, swivel pad on the part
    n2 = cs["n2"]
    Qp = np.array([0.0, cs["Q"][1], cs["Q"][2]])
    top_pt = np.array([0.0, S[1], S[2]]) + n2 * 0.028
    E._bar(bl, f"{pre}_spindle", tuple(Qp - n2 * 0.002), tuple(top_pt), 0, M["steel"], radius=0.0075, segs=10)
    for j, off in enumerate((h / 2 + 0.006, -h / 2 - 0.006)):
        c = np.array([0.0, S[1], S[2]]) + n2 * off - n2 * 0.005
        C._hex(bl, f"{pre}_nut{j}", 0.013, 0.01, M["dark"], tuple(c), _rot_to(n2))
    n = cs["n"]
    P = cs["P"]
    face = P + n * GAP
    _disc(bl, f"{pre}_pad_rubber", cs["pad_r"], 0.003, M["rubber"], tuple(face), n, segs=20)
    _disc(bl, f"{pre}_pad", cs["pad_r"] + 0.001, PAD_T - 0.003, M["bronze"], tuple(face + n * 0.003), n, segs=20, chamfer=0.002)
    E._rev(bl, f"{pre}_pad_ball", [(0, 0), (0.009, 0), (0.009, 0.006), (0.006, 0.009), (0, 0.009)], M["steel"],
           tuple(face + n * (PAD_T - 0.001)), _rot_to(n), segs=12)
    _attach(bl.objs, lever)
    # ---- cylinder (empty at the trunnion pin, local +Y toward the rod eye) and rod (empty at the rod eye)
    cyl = _empty(col, f"{pre}_cyl", root, mathutils.Matrix.Translation((0.0, Cu, Cv)), size=0.03)
    cyl.rotation_mode = 'XYZ'
    bc = E._B(col)
    Lb = cs["Lb"]
    E._rod(bc, f"{pre}_cyl_eye", 0.014, 0.024, M["dark"], (0.0, 0.0, 0.0), (0.0, math.pi / 2, 0.0), segs=16)
    E._box(bc, f"{pre}_cyl_tongue", (0.024, EYE + 0.004, 0.022), (0.0, EYE / 2 + 0.002, 0.0), M["dark"])
    C._bx(bc, f"{pre}_cyl_rear", (CAP, CAP_L, CAP), (0.0, EYE + CAP_L / 2, 0.0), M["black"], bev=0.003)
    C._bx(bc, f"{pre}_cyl_front", (CAP, CAP_L, CAP), (0.0, Lb - CAP_L / 2, 0.0), M["black"], bev=0.003)
    blen = Lb - EYE - 2 * CAP_L
    E._rod(bc, f"{pre}_cyl_barrel", CYL_R, blen + 0.002, M["alu"], (0.0, EYE + CAP_L + blen / 2, 0.0), (math.pi / 2, 0.0, 0.0), segs=24)
    for i, (sa, sv) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
        E._rod(bc, f"{pre}_cyl_tie{i}", 0.0028, blen + 2 * CAP_L - 0.004, M["steel"],
               (sa * (CAP / 2 - 0.006), EYE + CAP_L + blen / 2, sv * (CAP / 2 - 0.006)), (math.pi / 2, 0.0, 0.0), segs=6)
    for i, yy in enumerate((EYE + CAP_L / 2, Lb - CAP_L / 2)):          # air fittings on top of the caps
        E._rod(bc, f"{pre}_cyl_fit{i}", 0.006, 0.016, M["brass"], (0.0, yy, CAP / 2 + 0.008), segs=10)
        E._rod(bc, f"{pre}_cyl_fit{i}_b", 0.0055, 0.018, M["brass"], (0.0, yy - 0.008 if i == 0 else yy - 0.008, CAP / 2 + 0.016),
               (math.pi / 2, 0.0, 0.0), segs=10)
    _attach(bc.objs, cyl)
    fittings = [_empty(col, f"{pre}_fit{i}", cyl, mathutils.Matrix.Translation((0.0, yy - 0.017, CAP / 2 + 0.016)), size=0.01)
                for i, yy in enumerate((EYE + CAP_L / 2, Lb - CAP_L / 2))]
    _, phi, ext = clamp_state(cs, 1.0)
    rod = _empty(col, f"{pre}_rod", cyl, mathutils.Matrix.Translation((0.0, ext, 0.0)), size=0.02)
    br = E._B(col)
    Lr = cs["Lr"]
    E._rod(br, f"{pre}_rod_shaft", ROD_R, Lr, M["chrome"], (0.0, -Lr / 2 - 0.012, 0.0), (math.pi / 2, 0.0, 0.0), segs=12)
    C._hex(br, f"{pre}_rod_nut", 0.012, 0.01, M["dark"], (0.0, -0.030, 0.0), (-math.pi / 2, 0.0, 0.0))
    for j, s in enumerate((-1, 1)):                       # rod-end fork around the lever tail
        E._box(br, f"{pre}_fork{j}", (0.006, 0.036, 0.026), (s * (LEVER_T / 2 + 0.004), -0.006, 0.0), M["steel"])
    E._box(br, f"{pre}_fork_base", (LEVER_T + 0.014, 0.012, 0.026), (0.0, -0.024, 0.0), M["steel"])
    E._rod(br, f"{pre}_fork_pin", 0.0055, LEVER_T + 0.02, M["chrome"], (0.0, 0.0, 0.0), (0.0, math.pi / 2, 0.0), segs=10)
    _attach(br.objs, rod)
    handle = dict(root=root, lever=lever, cyl=cyl, rod=rod, spec=cs, fittings=fittings,
                  objects=bs.objs + bl.objs + bc.objs + br.objs)
    _apply_clamp(handle, 0.0)
    return handle


def _apply_clamp(c, s):
    th, phi, ext = clamp_state(c["spec"], s)
    c["lever"].rotation_euler[0] = th
    c["cyl"].rotation_euler[0] = phi
    c["rod"].location[1] = ext


# ============================================================================ gap sensor, lamp, pneumatics
def _gap_sensor(b, M):
    sx, sy, sz = SENSOR["pos"]
    b_ang = SENSOR["aim_b"] * DEG
    tgt = np.array([L2.SEAM_B_X, R_PIPE * math.cos(b_ang), Z_P + R_PIPE * math.sin(b_ang)])     # seam B, -Y upper quadrant
    d = tgt - np.array([sx, sy, sz])
    d /= np.linalg.norm(d)
    px, py = SENSOR["post"]
    ptop = SENSOR["post_top"]
    C._bx(b, f"{PREFIX}_gs_post", (0.05, 0.05, ptop - TOP), (px, py, (ptop + TOP) / 2), M["blue"], bev=0.004)
    _foot(b, M, f"{PREFIX}_gs_foot", px, py, 0.12, 0.12)
    E._box(b, f"{PREFIX}_gs_post_cap", (0.058, 0.058, 0.006), (px, py, ptop + 0.003), M["frame"])
    # horizontal arm from the post top to above the head, short hanger (clamp block) holding the head behind its lens
    ax, ay = sx - px, sy - py
    E._box(b, f"{PREFIX}_gs_arm", (math.hypot(ax, ay) + 0.03, 0.03, 0.03), ((px + sx) / 2, (py + sy) / 2, ptop - 0.015),
           M["frame"], (0.0, 0.0, math.atan2(ay, ax)))
    hz0, hz1 = sz + 0.028, ptop - 0.03
    E._box(b, f"{PREFIX}_gs_hanger", (0.034, 0.024, hz1 - hz0), (sx, sy - 0.018, (hz0 + hz1) / 2), M["frame"])
    rot = mathutils.Vector(tuple(d)).to_track_quat('Z', 'X').to_euler()
    head = C._bx(b, f"{PREFIX}_gs_head", (0.05, 0.06, 0.09), (sx, sy, sz), M["sensor"], rot=tuple(rot), bev=0.004)
    R3 = rot.to_matrix()
    lens_c = mathutils.Vector((sx, sy, sz)) + R3 @ mathutils.Vector((0.0, 0.0, 0.046))
    E._box(b, f"{PREFIX}_gs_lens", (0.03, 0.04, 0.004), tuple(lens_c), M["glass"], tuple(rot))
    lab_c = mathutils.Vector((sx, sy, sz)) + R3 @ mathutils.Vector((0.0, -0.0305, 0.0))
    E._box(b, f"{PREFIX}_gs_label", (0.035, 0.002, 0.05), tuple(lab_c), M["white_btn"], tuple(rot))
    return head, np.array([sx, sy, sz]), tgt


def _lamp(b, M):
    """Status lamp on a slim post (foot plate bolted to the table) at the front -Y corner."""
    px, py = LAMP["post"]
    z0 = LAMP["post_top"]
    C._bx(b, f"{PREFIX}_lamp_post", (0.04, 0.04, z0 - TOP), (px, py, (z0 + TOP) / 2), M["blue"], bev=0.003)
    _foot(b, M, f"{PREFIX}_lamp_foot", px, py, 0.09, 0.09)
    E._rev(b, f"{PREFIX}_lamp_base", [(0, 0), (0.032, 0), (0.032, 0.03), (0.028, 0.034), (0, 0.034)], M["black"], (px, py, z0), segs=20)
    lamp = E._rev(b, f"{PREFIX}_lamp", [(0, 0), (0.029, 0), (0.029, 0.05), (0.022, 0.062), (0.0, 0.066)], M["lamp"],
                  (px, py, z0 + 0.034), segs=24)
    E._rev(b, f"{PREFIX}_lamp_cap", [(0, 0), (0.026, 0), (0.026, 0.006), (0, 0.009)], M["black"], (px, py, z0 + 0.1), segs=20)
    lamp["lamp_on"] = 0.0
    return lamp


def _valve_terminal(b, M, lm):
    """Valve terminal hung under the -Y apron (node + 4 valve slices + end plate), ports / coils / LEDs on its -Y face,
    and the PVC cable duct on the apron's outer face above it, running along the apron to the front."""
    (x0, x1), (y0, y1), (z0, z1) = VALVE["x"], VALVE["y"], VALVE["z"]
    zc = (z0 + z1) / 2
    xn = x0 + 0.06
    C._bx(b, f"{PREFIX}_vt_node", (xn - x0, y1 - y0, z1 - z0), ((x0 + xn) / 2, (y0 + y1) / 2, zc), M["grey"], bev=0.004)
    E._box(b, f"{PREFIX}_vt_node_led", (0.02, 0.004, 0.008), ((x0 + xn) / 2, y0 - 0.001, z1 - 0.02), M["led"])
    E._rod(b, f"{PREFIX}_vt_m12", 0.008, 0.02, M["dark"], ((x0 + xn) / 2, y0 - 0.008, z0 + 0.03), (math.pi / 2, 0.0, 0.0), segs=10)
    C._label(b, f"{PREFIX}_label_vt", lm, 6, (0.045, 0.018), ((x0 + xn) / 2, y0 - 0.0015, zc + 0.01), (math.pi / 2, 0.0, 0.0))
    for k in range(4):
        xv = valve_x(k)
        C._bx(b, f"{PREFIX}_vt_valve{k}", (0.045, y1 - y0 - 0.004, z1 - z0 - 0.004), (xv, (y0 + y1) / 2 + 0.002, zc), M["alu"], bev=0.003)
        E._box(b, f"{PREFIX}_vt_coil{k}", (0.036, 0.014, 0.026), (xv, y0 - 0.005, z1 - 0.022), M["black"])
        E._box(b, f"{PREFIX}_vt_led{k}", (0.008, 0.003, 0.005), (xv + 0.008, y0 - 0.0125, z1 - 0.016), M["led"])
        for j in range(2):
            p = port(k, j)
            E._rod(b, f"{PREFIX}_vt_port{k}{j}", 0.0065, 0.014, M["black"], (p[0], y0 - 0.006, p[2]), (math.pi / 2, 0.0, 0.0), segs=10)
    C._bx(b, f"{PREFIX}_vt_end", (0.02, y1 - y0, z1 - z0), (x1 - 0.01, (y0 + y1) / 2, zc), M["grey"], bev=0.003)
    for k in range(2):
        E._rod(b, f"{PREFIX}_vt_silencer{k}", 0.007, 0.028, M["brass"], (x1 - 0.01, y0 - 0.013, z0 + 0.03 + 0.03 * k),
               (math.pi / 2, 0.0, 0.0), segs=10)
    (dx0, dx1), (dy0, dy1), (dz0, dz1) = DUCT["x"], DUCT["y"], DUCT["z"]
    duct = materials.get("painted", color="#8E9194", roughness=0.55, coat=0.0)
    E._box(b, f"{PREFIX}_duct", (dx1 - dx0, dy1 - dy0 - 0.003, dz1 - dz0 - 0.006), ((dx0 + dx1) / 2, (dy0 + dy1) / 2 + 0.0015, (dz0 + dz1) / 2 - 0.003), duct)
    E._box(b, f"{PREFIX}_duct_lid", (dx1 - dx0 + 0.004, dy1 - dy0 + 0.004, 0.008), ((dx0 + dx1) / 2, (dy0 + dy1) / 2, dz1 - 0.004), duct)
    E._box(b, f"{PREFIX}_duct_end", (0.004, dy1 - dy0, dz1 - dz0), (dx1 + 0.002, (dy0 + dy1) / 2, (dz0 + dz1) / 2), duct)


def valve_x(k):
    return VALVE["x"][0] + 0.06 + 0.0475 * (k + 0.5)


def port(k, j):
    """Station-local tip of port j (0 = A upper, 1 = B lower) of valve k on the valve terminal's -Y face."""
    z0, z1 = VALVE["z"]
    return np.array([valve_x(k), VALVE["y"][0] - 0.013, z0 + 0.045 - 0.022 * j])


def fitting(cs, i, s):
    """Station-local position of cylinder fitting i (0 rear, 1 front: tube end) and the tube's leaving direction."""
    _, phi, _ = clamp_state(cs, s)
    yy = (EYE + CAP_L / 2, cs["Lb"] - CAP_L / 2)[i] - 0.017
    zz = CAP / 2 + 0.016
    c, sn = math.cos(phi), math.sin(phi)
    f = np.array([0.0, cs["Cp"][0] + yy * c - zz * sn, cs["Cp"][1] + yy * sn + zz * c])
    back = np.array([0.0, -c, -sn])
    return cs["pivot"] + cs["R"] @ f, cs["R"] @ back


CLAMP_ORDER = ("flange_l", "flange_r", "elbow", "pipe")


def tube_paths(s=0.0):
    """Control points (station-local) of the pneumatic tubes for clamp state s (the tubes are built at s = 0 and their
    last two points follow the cylinder fittings).  Returns dict name -> (points, clamp name or None, fitting index)."""
    specs = clamp_specs()
    zt = TOP + 0.0045
    ye = TY[0]
    out = {}
    for k, name in enumerate(CLAMP_ORDER):
        cs = specs[name]
        for j in range(2):
            end, back = fitting(cs, j, s)
            prev = end + back * 0.035 + np.array([0.0, 0.0, 0.004])
            behind = end + back * 0.09
            xe = float(behind[0]) + 0.012 * (j - 0.5)
            p = port(k, j)
            sg = 1.0 - 2.0 * j                         # stub up from the port into the duct bottom (B passes outside A)
            stub = [p, p + np.array([0.006 * sg, -0.010 - 0.012 * j, 0.010]),
                    np.array([p[0] + 0.011 * sg, ye - 0.018 - 0.016 * j, DUCT["z"][0] + 0.012])]
            out[f"{name}_{'ab'[j]}_stub"] = (stub, None, None)
            pts = [np.array([xe, ye - 0.028, DUCT["z"][1] - 0.012])]      # leaves the duct at the clamp, over the edge
            pts += [np.array([xe, ye - 0.022, TOP + 0.008]), np.array([xe, ye + 0.02, zt])]
            pts += [np.array([behind[0], behind[1], zt + 0.006]), behind, prev, end]
            out[f"{name}_{'ab'[j]}"] = (pts, name, j)
    sx, sy, sz = SENSOR["pos"]
    px, py = SENSOR["post"]
    xc = px - 0.031                                    # down the post's -X face
    out["sensor_cable"] = ([np.array([sx, sy - 0.046, sz - 0.005]), np.array([sx + 0.012, sy - 0.066, sz - 0.03]),
                            np.array([xc, py - 0.012, sz - 0.08]), np.array([xc, py - 0.012, TOP + 0.09]),
                            np.array([xc, ye + 0.012, TOP + 0.02]), np.array([xc, ye - 0.022, TOP + 0.008]),
                            np.array([xc, ye - 0.028, DUCT["z"][1] - 0.012])], None, None)
    lx, ly = LAMP["post"]                              # lamp: down the post's -Y face, over the edge into the duct end
    out["lamp_cable"] = ([np.array([lx, ly - 0.0235, LAMP["post_top"] - 0.06]), np.array([lx, ly - 0.0235, TOP + 0.08]),
                          np.array([lx, ly - 0.032, TOP + 0.03]), np.array([lx - 0.008, ye - 0.013, TOP + 0.012]),
                          np.array([DUCT["x"][1] - 0.012, ye - 0.028, DUCT["z"][1] - 0.012])], None, None)
    x0 = VALVE["x"][0]                                 # air supply from the node's bottom down to the floor
    yv, zv = VALVE["y"][0], VALVE["z"][0]
    out["air_supply"] = ([np.array([x0 + 0.03, yv + 0.022, zv + 0.004]), np.array([x0 + 0.03, yv + 0.012, zv - 0.05]),
                          np.array([x0 + 0.01, ye - 0.035, FLOOR + 0.30]), np.array([x0 + 0.06, ye - 0.045, FLOOR + 0.008])], None, None)
    return out


def _tube(col, name, pts, radius, mat, res=8):
    """PU tube / cable: bezier curve through pts (station-local, AUTO handles) with a round bevel."""
    cu = bpy.data.curves.new(name, 'CURVE')
    cu.dimensions = '3D'
    cu.resolution_u = res
    cu.bevel_depth = radius
    cu.bevel_resolution = 2
    cu.use_fill_caps = True
    sp = cu.splines.new('BEZIER')
    sp.bezier_points.add(len(pts) - 1)
    for p, bp in zip(pts, sp.bezier_points):
        bp.co = tuple(float(v) for v in p)
        bp.handle_left_type = bp.handle_right_type = 'AUTO'
    ob = bpy.data.objects.new(name, cu)
    col.objects.link(ob)
    ob.data.materials.append(mat)
    return ob


def _hook(ob, idx, empty):
    """Hook bezier point idx of curve object ob (already parented / placed) to empty."""
    m = ob.modifiers.new(f"hook{idx}", 'HOOK')
    m.object = empty
    m.vertex_indices_set([3 * idx, 3 * idx + 1, 3 * idx + 2])
    m.matrix_inverse = empty.matrix_world.inverted() @ ob.matrix_world
    return m


def _tubes(col, M, clamps):
    """Tubes / cables from tube_paths(0) (clamps are built open).  Returns [objects], [(object, [(index, empty)])]."""
    objs, hooks = [], []
    for name, (pts, cl, j) in tube_paths(0.0).items():
        if name in ("sensor_cable", "lamp_cable"):
            objs.append(_tube(col, f"{PREFIX}_{name}", pts, 0.0035, M["cable"]))
        elif name == "air_supply":
            objs.append(_tube(col, f"{PREFIX}_{name}", pts, 0.006, M["tube_blue"]))
        else:
            mat = M["tube_blue"] if "_a" in name[-7:] else M["black"]
            ob = _tube(col, f"{PREFIX}_tube_{name}", pts, 0.004, mat)
            objs.append(ob)
            if cl is not None:
                fit = clamps[cl]["fittings"][j]
                hooks.append((ob, [(len(pts) - 2, fit), (len(pts) - 1, fit)]))
    return objs, hooks


def _earth_clamp(b, M):
    """Welding earth (ground) clamp on the +Y table edge (tack-robot side) with its cable down to the floor."""
    x, y = EARTH
    E._box(b, f"{PREFIX}_earth_jaw", (0.05, 0.03, 0.012), (x, y - 0.012, TOP + 0.006), M["copper"])
    E._box(b, f"{PREFIX}_earth_body", (0.05, 0.02, 0.07), (x, y + 0.01, TOP - 0.02), M["copper"])
    E._box(b, f"{PREFIX}_earth_lower", (0.05, 0.03, 0.012), (x, y - 0.012, TOP - PLATE_T - 0.006), M["copper"])
    E._rod(b, f"{PREFIX}_earth_screw", 0.006, 0.05, M["steel"], (x, y - 0.012, TOP - PLATE_T - 0.03), segs=8)
    return [(x, y + 0.02, TOP - 0.03)]


# ============================================================================ public API
def build(collection=None):
    col = collection or bpy.data.collections.new(NAME)
    if collection is None:
        bpy.context.scene.collection.children.link(col)
    M = _mats()
    lm = C._label_mat("stn_label", ["ST-10 FIT-UP & TACK · DN250", "CL1", "CL2", "CL3", "CL4", "FIXTURE ST-10", "VT-10"])
    root = G.empty(f"{PREFIX}_root", collection=col, size=0.3)
    root.matrix_world = G.M(FRAME)
    b = E._B(col)
    table = _table(b, M, lm)
    _panel(b, M, lm)
    _seat(b, M)
    ev = _elbow_vblock(b, M)
    pv = _pipe_vblock(b, M)
    stop = _pipe_stop(b, M)
    head, s_pos, s_tgt = _gap_sensor(b, M)
    lamp = _lamp(b, M)
    _valve_terminal(b, M, lm)
    earth = _earth_clamp(b, M)
    _attach(b.objs, root)
    specs = clamp_specs()
    clamps = {}
    for k, name in enumerate(L2.STATION_CLAMPS):
        clamps[name] = _clamp(col, M, lm, name, specs[name], root, k)
    tubes, hooks = _tubes(col, M, clamps)
    ex, ey, ez = earth[0]
    tubes.append(_tube(col, f"{PREFIX}_earth_cable", [np.array([ex, ey, ez]), np.array([ex, ey + 0.04, ez - 0.1]),
                                                       np.array([ex - 0.05, ey + 0.06, FLOOR + 0.25]),
                                                       np.array([ex - 0.12, ey + 0.25, FLOOR + 0.012])], 0.009, M["cable"]))
    _attach(tubes, root)
    bpy.context.view_layer.update()
    for ob, hk in hooks:
        for idx, e in hk:
            _hook(ob, idx, e)
    objs = [o for o in col.all_objects]
    return dict(collection=col, clamps=clamps, lamp=lamp, table=table, frame=FRAME.copy(), root=root, objects=objs,
                sensor=dict(head=head, pos=s_pos, target=s_tgt),
                supports=dict(elbow_vblock=ev, pipe_vblock=pv, pipe_stop=stop))


def set_clamp(st, name, s, frame=None):
    """Clamp `name`: s = 0 open .. 1 closed.  Keys lever / cylinder rotation_euler[0] and rod location[1]."""
    c = st["clamps"][name]
    _apply_clamp(c, s)
    if frame is not None:
        c["lever"].keyframe_insert("rotation_euler", index=0, frame=frame)
        c["cyl"].keyframe_insert("rotation_euler", index=0, frame=frame)
        c["rod"].keyframe_insert("location", index=1, frame=frame)


def set_lamp(st, state, frame=None):
    """Station lamp: 0 off, 1 green (object property 'lamp_on', CONSTANT key when frame is given)."""
    lamp = st["lamp"]
    lamp["lamp_on"] = float(state)
    if frame is not None:
        C.key_constant(lamp, '["lamp_on"]', frame)


# ============================================================================ obstacles (pure python)
def _local_box_to_world(name, lo, hi):
    pts = _box_pts(lo, hi)
    w = (FRAME @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
    lo_w, hi_w = w.min(axis=0), w.max(axis=0)
    return dict(name=name, center=tuple(float(v) for v in (lo_w + hi_w) / 2), size=tuple(float(v) for v in hi_w - lo_w), yaw=0.0)


def _aabb(pts, pad=0.0):
    pts = np.asarray(pts, float)
    return pts.min(axis=0) - pad, pts.max(axis=0) + pad


def obstacles():
    """Conservative world AABBs of the station (table, fixture, clamps over their whole swing, sensor, panel, valve
    terminal).  Pure python (numpy)."""
    out = []
    p = 0.01
    out.append(_local_box_to_world(f"{PREFIX}_table", (TX[0] - p, TY[0] - p, FLOOR), (TX[1] + p, TY[1] + p, TOP + 0.001)))
    out.append(_local_box_to_world(f"{PREFIX}_panel", (TX[1], PANEL["y"][0] - 0.055, PANEL["z"][0] - p),
                                   (PANEL["x"][1] + 0.03, PANEL["y"][1] + p, PANEL["z"][1] + p)))
    out.append(_local_box_to_world(f"{PREFIX}_seat", (-SEAT_R - p, -SEAT_R - p, TOP), (SEAT_R + p, SEAT_R + p, PIN_H + 0.002)))
    vb = elbow_vblock()
    corners = []
    for gx in (-EV_LEN / 2, EV_LEN / 2):
        for yy in (-EV_W, EV_W):
            for zz in (vb["apex"] - EV_DEPTH, vb["apex"] + math.tan(EV_BETA) * EV_W):
                corners.append(vb["O"] + gx * vb["g"] + np.array([0.0, yy, 0.0]) + zz * vb["n"])
    lo, hi = _aabb(corners, 0.005)
    xr = vb["O"][0] - (vb["apex"] - EV_DEPTH) * math.sin(vb["gamma"])       # riser / foot centre
    out.append(_local_box_to_world(f"{PREFIX}_elbow_vblock", (min(lo[0], xr - 0.085), min(lo[1], -0.095), TOP),
                                   (max(hi[0], xr + 0.085), max(hi[1], 0.095), hi[2])))
    apex = pipe_vblock_apex()
    out.append(_local_box_to_world(f"{PREFIX}_pipe_vblock", (PV_X - 0.10, -PV_W - p, TOP), (PV_X + 0.10, PV_W + p, apex + math.tan(PV_BETA) * PV_W + 0.004)))
    z0, z1 = STOP_PLATE["z"]
    out.append(_local_box_to_world(f"{PREFIX}_pipe_stop", (STOP_X - 0.001, -STOP_PLATE["y"] - p, TOP), (STOP_X + 0.20, STOP_PLATE["y"] + p, z1 + p)))
    for name, cs in clamp_specs().items():
        pts = np.concatenate([clamp_points(cs, s) for s in np.linspace(0.0, 1.0, 13)])
        lo, hi = _aabb(pts, 0.012)
        out.append(_local_box_to_world(f"{PREFIX}_clamp_{name}_lever", lo, hi))
        lo, hi = _aabb(clamp_static_points(cs), 0.008)
        out.append(_local_box_to_world(f"{PREFIX}_clamp_{name}_mount", (lo[0], lo[1], TOP), hi))
    sx, sy, sz = SENSOR["pos"]
    px, py = SENSOR["post"]
    out.append(_local_box_to_world(f"{PREFIX}_gap_sensor", (min(sx - 0.06, px - 0.07), min(sy, py) - 0.07, TOP),
                                   (max(sx + 0.06, px + 0.07), max(sy + 0.06, py + 0.07), max(SENSOR["post_top"] + 0.015, sz + 0.06))))
    out.append(_local_box_to_world(f"{PREFIX}_gap_sensor_head", (sx - 0.07, sy - 0.07, sz - 0.07), (sx + 0.07, sy + 0.07, sz + 0.07)))
    lx, ly = LAMP["post"]
    out.append(_local_box_to_world(f"{PREFIX}_lamp", (lx - 0.055, ly - 0.055, TOP), (lx + 0.055, ly + 0.055, LAMP["post_top"] + 0.115)))
    out.append(_local_box_to_world(f"{PREFIX}_valve_terminal", (VALVE["x"][0] - p, DUCT["y"][0] - 0.035, VALVE["z"][0] - p),
                                   (DUCT["x"][1] + p, VALVE["y"][1] + p, TOP)))
    paths0, paths1 = tube_paths(0.0), tube_paths(1.0)
    for name, (pts, _, _) in paths0.items():
        allp = np.concatenate([np.array(pts), np.array(paths1[name][0])])
        lo, hi = _aabb(allp, 0.032)                         # bevel radius + bezier overshoot
        out.append(_local_box_to_world(f"{PREFIX}_tube_{name}", lo, hi))
    ex, ey = EARTH
    out.append(_local_box_to_world(f"{PREFIX}_earth_clamp", (ex - 0.04, ey - 0.04, TOP - PLATE_T - 0.06), (ex + 0.04, ey + 0.04, TOP + 0.02)))
    out.append(_local_box_to_world(f"{PREFIX}_earth_cable", (ex - 0.14, ey, FLOOR), (ex + 0.02, ey + 0.29, TOP - 0.02)))
    return out
