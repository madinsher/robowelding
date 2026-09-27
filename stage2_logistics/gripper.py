"""Tong gripper (clamp tongs with V-prism jaws) for the stage-2 handler robot.

Tool frame = tool0 of the robot: +Z from the robot flange toward the part, X = jaw hinge axis (the gripped pipe's axis),
Y = closing direction.  TCP = Tz(GRIP_TCP_Z) = axis of the gripped cylinder.

Stack along +Z: adapter plate -> swivel (rotary union: air + signals; its stator carries the dress-pack connector on
the -X side) -> blue cross-beam housing with a valve terminal on its +X face, an M12 junction box on its -X cover, two
jaw clevises and two cylinder brackets -> two jaws hinged on pins parallel to X at (y, z) = (+-HINGE_Y, HINGE_Z).
Each jaw: two dark cheek plates, a steel V-prism with two urethane pads.  Each jaw is driven by its own ISO pneumatic
cylinder (pivot under the housing, rod eye on a cross pin between the cheek plates).

Swivel: the handler turns J6 by up to 180 deg (the tongs are symmetric), so the dress pack must not turn with the
flange.  `g["stator"]` lists the objects that stay fixed to the robot wrist (swivel body, torque lug, connector and the
`connector` empty); a robot re-parents them to the parent of its J6 joint (handler_robot.build does), everything else
turns with tool0.  Nothing of the rotor (adapter, housing, valve terminal, hoses) enters the stator's band
(z < HOUSING z0, r > ADAPTER r), so the rotor can turn freely.

Jaw geometry: the pads lie on the two tangent lines of a circle r = PIPE_R about the TCP axis, touching it at
PHI_U above and PHI_L below the equator (form closure: the lower contacts are below the pipe axis), so at s = 0 the
pads touch a DN250 pipe exactly (a mesh vertex row lies on each contact line).  s = 1 opens each jaw by OPEN_ANGLE about
its hinge, enough to pass a cylinder of diameter 0.36 m approaching from above (tool -Z side).  The flange grasp
(GRASP['flange']: weld-neck hub gripped at 75 mm above the flange face) cannot close to s = 0 - the lower pad tips
rest on the hub cone - so s is calibrated piecewise: s = GRASP['flange'][3] (0.05) is exactly the jaw angle where the
pads touch the hub, see jaw_angle().

    g = build(tool0, collection, name="grp")    # dict(root, tcp, jaws, parts, cylinders, rods, connector, stator)
    set_open(g, s, frame=None)                  # s in [0, 1]
"""
import math

import bpy
import mathutils
import numpy as np

import tools  # noqa: F401
from cell import geom as G
from cell import layout as L
from cell import materials
import layout2 as L2

DEG = math.pi / 180.0
R_PIPE = L2.PIPE_R
TCP_Z = L2.GRIP_TCP_Z
PAD_HW = L2.GRIP_PAD_WIDTH / 2          # pads span |x| <= PAD_HW
BLOCK_HW = PAD_HW - 0.001               # steel prism body (pads overhang it by 1 mm)
CHEEK_X = (BLOCK_HW, BLOCK_HW + 0.016)  # cheek plates of each jaw (|x| range)
CLEVIS_X = (CHEEK_X[1] + 0.002, CHEEK_X[1] + 0.018)
PHI_U = 38.0 * DEG                      # upper pad contact, above the pipe equator
PHI_L = 10.0 * DEG                      # lower pad contact, below the equator
PAD_T = 0.008                           # urethane pad thickness
PAD_EXT_U, PAD_EXT_L = 0.010, 0.005     # pad length beyond the contact line
PAD_RELIEF = 0.006                      # gap between the two pads at the V apex
BLOCK_BACK = 0.212
HINGE_Y, HINGE_Z = 0.25, 0.13
OPEN_ANGLE = 20.0 * DEG                 # jaw rotation at s = 1
CYL_PIVOT = (0.07, 0.168)               # (y, z) cylinder pivot under the housing (+Y side)
ROD_PIN = (0.20, 0.25)                  # (y, z) rod-eye cross pin on the +Y jaw at s = 0
HOUSING = dict(x=0.075, y=0.205, z=(0.078, 0.150))
ROTARY = dict(r=0.085, z=(0.018, 0.078))
ADAPTER = dict(r=0.10, z=(0.0, 0.018))
CONN_Z = 0.049                          # centre height of the stator's connector box / torque lug (above the adapter screws)
VALVE = dict(t=0.022, w=0.09, h=0.05, y=0.06)    # valve terminal on the housing's +X face: thickness (x), width (y), height (z)
JBOX_Y = 0.05                           # M12 junction box on the -X cover (between the nameplate and the cover screws)
S_FLANGE = L2.GRASP["flange"][3]
SENSOR_Y = 0.14


# ------------------------------------------------------------------ 2D jaw profile (+Y jaw, s = 0, (y, z) tool coords)
def _v(a):
    return np.asarray(a, dtype=float)


def jaw_profile():
    """Polygons of the +Y jaw at s = 0 in (y, z) tool coordinates: block (steel V-prism), pad_u / pad_l (urethane,
    index 1 of each is the contact point), cheek (side plate incl. hinge boss); plus the contact points and apex."""
    O = _v((0.0, TCP_Z))
    Pu = O + R_PIPE * _v((math.cos(PHI_U), -math.sin(PHI_U)))
    Pl = O + R_PIPE * _v((math.cos(PHI_L), math.sin(PHI_L)))
    bis, half = (PHI_U - PHI_L) / 2, (PHI_U + PHI_L) / 2
    A = O + R_PIPE / math.cos(half) * _v((math.cos(bis), -math.sin(bis)))
    du = (Pu - A) / np.linalg.norm(Pu - A)
    dl = (Pl - A) / np.linalg.norm(Pl - A)
    nu, nl = (O - Pu) / R_PIPE, (O - Pl) / R_PIPE          # pad surface normals (toward the pipe axis)
    Ue, Le = Pu + PAD_EXT_U * du, Pl + PAD_EXT_L * dl
    a1, a2 = Pu - PAD_T * nu, Pl - PAD_T * nl
    t = np.linalg.solve(np.array([du, -dl]).T, a2 - a1)
    As = a1 + t[0] * du                                      # apex of the steel V behind the pads
    Ue_s, Le_s = Ue - PAD_T * nu, Le - PAD_T * nl
    zt = Ue_s[1] - 0.010
    block = [Ue_s, As, Le_s, Le_s + _v((0.022, -0.004)), _v((BLOCK_BACK, Le_s[1] - 0.030)), _v((BLOCK_BACK, zt)),
             _v((Ue_s[0] + 0.008, zt))]
    Au, Al = A + PAD_RELIEF * du, A + PAD_RELIEF * dl
    pad_u = [Au, Pu, Ue, Ue - PAD_T * nu, Pu - PAD_T * nu, Au - PAD_T * nu]
    pad_l = [Al, Pl, Le, Le - PAD_T * nl, Pl - PAD_T * nl, Al - PAD_T * nl]
    H = _v((HINGE_Y, HINGE_Z))
    rb = 0.032
    cheek = [H + rb * _v((math.cos(a), math.sin(a))) for a in np.linspace(math.pi, 2 * math.pi, 13)]
    cheek += [_v((0.222, 0.36)), _v((0.207, 0.425)), _v((0.170, 0.432)), _v((0.160, 0.362)), _v((0.150, 0.318)),
              _v((0.205, 0.200))]
    return dict(block=block, pad_u=pad_u, pad_l=pad_l, cheek=cheek, Pu=Pu, Pl=Pl, apex=A, hinge=H)


def _rot2(p, c, a):
    """Rotate the (y, z) point p about c by angle a (rotation about +X)."""
    ca, sa = math.cos(a), math.sin(a)
    d = _v(p) - c
    return c + _v((ca * d[0] - sa * d[1], sa * d[0] + ca * d[1]))


def _sample(poly, n=24):
    pts = []
    for i in range(len(poly)):
        a, b = _v(poly[i]), _v(poly[(i + 1) % len(poly)])
        for k in range(n):
            pts.append(a + (b - a) * (k / n))
    return pts


def _flange_radius(zp):
    """Outer radius of the stage-1 weld-neck flange (spool.py profile) at flange-local height zp; 0 outside."""
    r_o = L.PIPE_OD / 2
    hub = r_o + 0.028
    zf = L.FLANGE_THK
    if zp < 0.0 or zp > L.SEAM_A_Z:
        return 0.0
    if zp <= zf:
        return L.FLANGE_OD / 2
    if zp <= zf + 0.012:
        return hub + 0.012 - 0.012 * (zp - zf) / 0.012
    if zp <= L.SEAM_A_Z - 0.035:
        return hub + (r_o - hub) * (zp - zf - 0.012) / (L.SEAM_A_Z - 0.035 - zf - 0.012)
    return r_o


def flange_contact_angle():
    """Smallest jaw rotation (rad) at which no jaw part (pads, prism, cheek plates) intersects the flange held at
    GRASP['flange'] - i.e. the angle where the pads come to rest on the hub.  Pure python."""
    prof = jaw_profile()
    H = prof["hinge"]
    zg = L2.GRASP["flange"][0][2]
    sets = [(_sample(prof["block"]), 0.0), (_sample(prof["pad_u"]), 0.0), (_sample(prof["pad_l"]), 0.0),
            (_sample(prof["cheek"]), CHEEK_X[0])]

    def clear(a):
        for pts, xm in sets:
            for p in pts:
                q = _rot2(p, H, -a)
                R = _flange_radius(zg + (TCP_Z - q[1]))
                if R > 0.0 and math.hypot(xm, q[0]) < R - 1e-6:
                    return False
        return True
    lo, hi = 0.0, OPEN_ANGLE
    if clear(lo):
        return 0.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if clear(mid):
            hi = mid
        else:
            lo = mid
    return hi


_ANG_F = None


def jaw_angle(s):
    """Jaw opening angle (rad) for the normalised command s in [0, 1]: piecewise linear through
    (0, 0) - (S_FLANGE, flange contact angle) - (1, OPEN_ANGLE)."""
    global _ANG_F
    if _ANG_F is None:
        _ANG_F = flange_contact_angle()
    s = min(max(float(s), 0.0), 1.0)
    if 0.0 < S_FLANGE < 1.0 and 0.0 < _ANG_F < OPEN_ANGLE:
        if s <= S_FLANGE:
            return _ANG_F * s / S_FLANGE
        return _ANG_F + (OPEN_ANGLE - _ANG_F) * (s - S_FLANGE) / (1.0 - S_FLANGE)
    return OPEN_ANGLE * s


def _cylinder_state(side, a):
    """(angle about X, length) of the cylinder on side +1 (+Y jaw) / -1 for jaw opening angle a."""
    H = _v((HINGE_Y, HINGE_Z))
    C = _rot2(ROD_PIN, H, -a)
    dy, dz = side * C[0] - side * CYL_PIVOT[0], C[1] - CYL_PIVOT[1]
    return math.atan2(dz, dy), math.hypot(dy, dz)


# ------------------------------------------------------------------ mesh helpers
def _mats():
    return dict(
        blue=materials.get("painted", color="#1F4E8C", roughness=0.42),
        frame=materials.get("painted", color="#2B2F36", roughness=0.5),
        steel=materials.get("machined_steel"),
        chrome=materials.get("bevel_steel"),
        dark=materials.get("dark_metal"),
        black=materials.get("black_plastic"),
        alu=materials.get("painted", color="#AEB2B7", roughness=0.32, coat=0.0),
        motor=materials.get("painted", color="#33363A", roughness=0.5, coat=0.0),
        pad=materials.get("painted", color="#B5541B", roughness=0.65, coat=0.0),
        brass=materials.get("brass"),
        cable=materials.get("cable_black"),
        hose=materials.get("painted", color="#1E5AA8", roughness=0.4, coat=0.0),
        label=materials.get("painted", color="#C9CCD0", roughness=0.35, coat=0.0),
        led=materials.get("emissive", color="#FF8A1E", strength=6.0),
        yellow=materials.get("safety_yellow"),
    )


def _finish(ob, mat, bevel=0.0, parent=None, local=None):
    ob.data.materials.append(mat)
    if bevel > 0:
        m = ob.modifiers.new("bevel", 'BEVEL')
        m.width = bevel
        m.segments = 2
        m.limit_method = 'ANGLE'
        m.harden_normals = False
    if parent is not None:
        ob.parent = parent
        ob.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        ob.matrix_basis = local if local is not None else mathutils.Matrix.Identity(4)
    return ob


def _extrude(name, poly, x0, x1, col, origin=(0.0, 0.0), mirror=False):
    """Prism: 2D polygon [(y, z), ...] extruded along X from x0 to x1, vertices relative to origin (y, z);
    mirror flips y (the -Y jaw)."""
    sy = -1.0 if mirror else 1.0
    pts = [(sy * (p[0] - origin[0]), p[1] - origin[1]) for p in poly]
    n = len(pts)
    verts = [(x0, y, z) for y, z in pts] + [(x1, y, z) for y, z in pts]
    faces = [tuple(range(n)), tuple(range(2 * n - 1, n - 1, -1))]
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, n + i, n + j, j))
    ob = G.new_object(name, verts, faces, col, smooth=False)
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(ob.data)
    bm.free()
    ob.data.update()
    return ob


def _cyl(name, r, length, col, axis='Z', verts=24, center=(0, 0, 0)):
    """Cylinder mesh along the given local axis, centred at center (mesh coordinates, object at the origin)."""
    rot = {'X': (0, math.pi / 2, 0), 'Y': (math.pi / 2, 0, 0), 'Z': (0, 0, 0)}[axis]
    ob = G.cylinder(name, r, length, (0, 0, 0), rot, vertices=verts, collection=col)
    M = mathutils.Matrix.Translation(center) @ mathutils.Euler(rot).to_matrix().to_4x4()
    ob.rotation_euler = (0, 0, 0)
    ob.data.transform(M)
    ob.data.update()
    return ob


def _box(name, size, center, col):
    ob = G.box(name, size, (0, 0, 0), col)
    ob.data.transform(mathutils.Matrix.Translation(center))
    ob.data.update()
    return ob


def _curve(name, pts, radius, mat, col, parent):
    cu = bpy.data.curves.new(name, 'CURVE')
    cu.dimensions = '3D'
    cu.bevel_depth = radius
    cu.bevel_resolution = 3
    cu.resolution_u = 8
    sp = cu.splines.new('BEZIER')
    sp.bezier_points.add(len(pts) - 1)
    for p, xyz in zip(sp.bezier_points, pts):
        p.co = xyz
        p.handle_left_type = p.handle_right_type = 'AUTO'
    ob = bpy.data.objects.new(name, cu)
    col.objects.link(ob)
    ob.data.materials.append(mat)
    ob.parent = parent
    ob.matrix_parent_inverse = mathutils.Matrix.Identity(4)
    return ob


# ------------------------------------------------------------------ build
def build(tool0, collection, name="grp"):
    col = collection
    M = _mats()
    root = G.empty(f"{name}_root", collection=col, size=0.08)
    root.parent = tool0
    root.matrix_parent_inverse = mathutils.Matrix.Identity(4)
    root.matrix_basis = mathutils.Matrix.Identity(4)
    tcp = G.empty(f"{name}_tcp", collection=col, size=0.08, display='ARROWS')
    tcp.parent = root
    tcp.matrix_parent_inverse = mathutils.Matrix.Identity(4)
    tcp.matrix_basis = mathutils.Matrix.Translation((0.0, 0.0, TCP_Z))
    parts = []

    def add(ob, mat, bevel=0.0, parent=root, local=None):
        _finish(ob, mat, bevel, parent, local)
        parts.append(ob)
        return ob

    # ---- adapter plate (rotor: turns with tool0)
    z0, z1 = ADAPTER["z"]
    add(_cyl(f"{name}_adapter", ADAPTER["r"], z1 - z0, col, center=(0, 0, (z0 + z1) / 2), verts=48), M["steel"], 0.0015)
    for k in range(8):
        a = (k + 0.5) * math.pi / 4
        add(_cyl(f"{name}_adapter_screw{k}", 0.0075, 0.006, col, center=(0.092 * math.cos(a), 0.092 * math.sin(a), z1 + 0.003), verts=12), M["dark"])
    # ---- swivel stator (fixed to the wrist on a robot): body, band, torque lug, dress-pack connector on the -X side
    r0, r1 = ROTARY["z"]
    zc = CONN_Z
    stator = []
    stator.append(add(_cyl(f"{name}_rotary", ROTARY["r"], r1 - r0, col, center=(0, 0, (r0 + r1) / 2), verts=48), M["motor"], 0.003))
    stator.append(add(_cyl(f"{name}_rotary_band", ROTARY["r"] + 0.002, 0.012, col, center=(0, 0, r0 + 0.022), verts=48), M["blue"]))
    stator.append(add(_box(f"{name}_rotary_drive", (0.05, 0.07, 0.042), (0.0, -ROTARY["r"] - 0.012, zc), col), M["motor"], 0.003))
    stator.append(add(_cyl(f"{name}_rotary_drive_cap", 0.018, 0.012, col, axis='Y', center=(0.0, -ROTARY["r"] - 0.053, zc)), M["dark"]))
    # dress-pack connector on the -X side (the handler's hose ends here)
    stator.append(add(_box(f"{name}_connector_box", (0.032, 0.07, 0.042), (-ROTARY["r"] - 0.012, 0.0, zc), col), M["dark"], 0.002))
    stator.append(add(_cyl(f"{name}_connector_plug0", 0.013, 0.016, col, axis='X', center=(-ROTARY["r"] - 0.034, 0.0, zc), verts=16), M["black"]))
    stator.append(add(_cyl(f"{name}_connector_plug1", 0.008, 0.012, col, axis='X', center=(-ROTARY["r"] - 0.032, 0.026, zc), verts=12), M["dark"]))
    connector = G.empty(f"{name}_connector", collection=col, size=0.03)      # dress-pack hose end, enters along +X
    connector.parent = root
    connector.matrix_parent_inverse = mathutils.Matrix.Identity(4)
    connector.matrix_basis = mathutils.Matrix.Translation((-ROTARY["r"] - 0.044, 0.0, zc))
    stator.append(connector)

    # ---- housing (cross beam), cover + nameplate, clevises, hinge pins, cylinder brackets, jaw sensors
    hx, hy = HOUSING["x"], HOUSING["y"]
    hz0, hz1 = HOUSING["z"]
    add(_box(f"{name}_housing", (2 * hx, 2 * hy, hz1 - hz0), (0, 0, (hz0 + hz1) / 2), col), M["blue"], 0.006)
    add(_box(f"{name}_housing_cover", (0.004, 0.24, 0.05), (-hx - 0.002, 0.0, (hz0 + hz1) / 2), col), M["blue"], 0.0015)
    add(_box(f"{name}_nameplate", (0.002, 0.07, 0.03), (-hx - 0.005, -0.06, (hz0 + hz1) / 2), col), M["label"])
    for k, (yy, zz) in enumerate(((-0.105, -0.018), (0.105, -0.018), (-0.105, 0.018), (0.105, 0.018))):
        add(_cyl(f"{name}_cover_screw{k}", 0.004, 0.004, col, axis='X', center=(-hx - 0.005, yy, (hz0 + hz1) / 2 + zz), verts=10), M["dark"])
    for side in (1, -1):
        sd = "R" if side > 0 else "L"
        H = (side * HINGE_Y, HINGE_Z)
        rc = 0.036
        poly = [(0.17, hz0 + 0.012)]
        poly += [(HINGE_Y + rc * math.cos(a), HINGE_Z + rc * math.sin(a)) for a in np.linspace(-math.pi / 2, math.pi / 2, 13)]
        poly += [(0.20, HINGE_Z + rc), (0.17, hz1)]
        for sx in (1, -1):
            add(_extrude(f"{name}_clevis_{sd}{'p' if sx > 0 else 'n'}", poly, sx * CLEVIS_X[0], sx * CLEVIS_X[1], col, mirror=side < 0),
                M["frame"], 0.002)
        add(_cyl(f"{name}_hinge_pin_{sd}", 0.012, 2 * CLEVIS_X[1] + 0.012, col, axis='X', center=(0, H[0], H[1]), verts=20), M["steel"])
        for sx in (1, -1):
            add(_cyl(f"{name}_hinge_nut_{sd}{'p' if sx > 0 else 'n'}", 0.019, 0.012, col, axis='X',
                     center=(sx * (CLEVIS_X[1] + 0.006), H[0], H[1]), verts=6), M["dark"])
        # cylinder pivot bracket (two lugs + pin) under the housing
        by, bz = side * CYL_PIVOT[0], CYL_PIVOT[1]
        for sx in (1, -1):
            add(_box(f"{name}_cyl_bracket_{sd}{'p' if sx > 0 else 'n'}", (0.008, 0.024, bz - hz1 + 0.008),
                     (sx * 0.026, by, (hz1 + bz + 0.008) / 2), col), M["frame"], 0.0015)
        add(_cyl(f"{name}_cyl_pivot_{sd}", 0.007, 0.066, col, axis='X', center=(0, by, bz), verts=16), M["steel"])
        # jaw position sensor (M12 inductive, orange LED) under the housing's -X edge, looking at the jaw boss
        sy = side * SENSOR_Y
        add(_box(f"{name}_sensor_bracket_{sd}", (0.03, 0.03, 0.004), (-hx - 0.004, sy, hz1 + 0.002), col), M["frame"])
        add(_cyl(f"{name}_sensor_{sd}", 0.006, 0.05, col, axis='Y', center=(-hx + 0.006, sy + side * 0.01, hz1 + 0.010), verts=16), M["steel"])
        add(_cyl(f"{name}_sensor_led_{sd}", 0.0062, 0.006, col, axis='Y', center=(-hx + 0.006, sy - side * 0.018, hz1 + 0.010), verts=16), M["led"])
    # valve terminal with two solenoids on the housing's +X face (rotor side: it must stay out of the swivel band)
    vx0, vzc = hx + VALVE["t"] / 2, (hz0 + hz1) / 2
    add(_box(f"{name}_valve", (VALVE["t"], VALVE["w"], VALVE["h"]), (vx0, VALVE["y"], vzc), col), M["alu"], 0.002)
    for k, dy in enumerate((-0.022, 0.022)):
        yy = VALVE["y"] + dy
        add(_box(f"{name}_valve_coil{k}", (0.018, 0.028, 0.034), (hx + VALVE["t"] + 0.009, yy, vzc - 0.004), col), M["black"], 0.002)
        add(_cyl(f"{name}_valve_plug{k}", 0.006, 0.008, col, axis='X', center=(hx + VALVE["t"] + 0.022, yy, vzc - 0.004), verts=12), M["led"])
        add(_cyl(f"{name}_valve_fitting{k}", 0.0045, 0.008, col, axis='Z', center=(vx0, VALVE["y"] + 1.4 * dy, vzc + VALVE["h"] / 2 + 0.004), verts=12),
            M["brass"])
    # M12 junction box for the jaw sensors on the -X cover (rotor side; signals pass the swivel's slip ring)
    jb = (-hx - 0.004 - 0.009, JBOX_Y, vzc)
    add(_box(f"{name}_jbox", (0.018, 0.05, 0.034), jb, col), M["black"], 0.002)
    for k, dy in enumerate((-0.011, 0.011)):
        add(_cyl(f"{name}_jbox_plug{k}", 0.0055, 0.01, col, axis='Z', center=(jb[0], JBOX_Y + dy, vzc + 0.022), verts=12), M["dark"])
    # ---- jaws
    prof = jaw_profile()
    Hn = prof["hinge"]
    jaws, cyls, rods = [], [], []
    for side in (-1, 1):
        sd = "R" if side > 0 else "L"
        jaw = G.empty(f"{name}_jaw_{sd}", collection=col, size=0.05)
        jaw.parent = root
        jaw.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        jaw.location = (0.0, side * HINGE_Y, HINGE_Z)
        jaw.rotation_mode = 'XYZ'
        mir = side < 0
        o = (Hn[0], Hn[1])
        add(_extrude(f"{name}_prism_{sd}", prof["block"], -BLOCK_HW, BLOCK_HW, col, o, mir), M["steel"], 0.0015, jaw)
        for k in ("u", "l"):
            add(_extrude(f"{name}_pad_{sd}{k}", prof["pad_" + k], -PAD_HW, PAD_HW, col, o, mir), M["pad"], 0.0008, jaw)
        for sx in (1, -1):
            add(_extrude(f"{name}_cheek_{sd}{'p' if sx > 0 else 'n'}", prof["cheek"], sx * CHEEK_X[0], sx * CHEEK_X[1], col, o, mir),
                M["frame"], 0.002, jaw)
            for k, (yy, zz) in enumerate(((0.19, 0.335), (0.19, 0.395))):
                add(_cyl(f"{name}_cheek_bolt_{sd}{'p' if sx > 0 else 'n'}{k}", 0.008, 0.006, col, axis='X',
                         center=(sx * (CHEEK_X[1] + 0.003), side * (yy - Hn[0]), zz - Hn[1]), verts=6), M["dark"], 0.0, jaw)
        add(_cyl(f"{name}_rod_pin_{sd}", 0.008, 2 * CHEEK_X[1] + 0.008, col, axis='X',
                 center=(0, side * (ROD_PIN[0] - Hn[0]), ROD_PIN[1] - Hn[1]), verts=16), M["steel"], 0.0, jaw)
        # small warning stripe on the jaw back (pinch point)
        add(_box(f"{name}_jaw_stripe_{sd}", (2 * BLOCK_HW - 0.004, 0.002, 0.03), (0, side * (BLOCK_HW * 0 + BLOCK_BACK + 0.001 - Hn[0]), 0.35 - Hn[1]), col),
            M["yellow"], 0.0, jaw)
        jaws.append(jaw)
        # ---- pneumatic cylinder (ISO 15552 look): body pivots about X at CYL_PIVOT, rod eye on the jaw cross pin
        cyl = G.empty(f"{name}_cyl_{sd}", collection=col, size=0.04)
        cyl.parent = root
        cyl.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        cyl.location = (0.0, side * CYL_PIVOT[0], CYL_PIVOT[1])
        cyl.rotation_mode = 'XYZ'
        add(_box(f"{name}_cyl_eye_{sd}", (0.02, 0.03, 0.02), (0, 0.003, 0), col), M["dark"], 0.002, cyl)
        add(_box(f"{name}_cyl_cap_rear_{sd}", (0.05, 0.024, 0.05), (0, 0.030, 0), col), M["alu"], 0.003, cyl)
        add(_cyl(f"{name}_cyl_tube_{sd}", 0.021, 0.058, col, axis='Y', center=(0, 0.071, 0), verts=32), M["alu"], 0.0, cyl)
        add(_box(f"{name}_cyl_cap_front_{sd}", (0.05, 0.022, 0.05), (0, 0.111, 0), col), M["alu"], 0.003, cyl)
        for k, (xx, zz) in enumerate(((0.019, 0.019), (-0.019, 0.019), (0.019, -0.019), (-0.019, -0.019))):
            add(_cyl(f"{name}_cyl_tierod_{sd}{k}", 0.003, 0.104, col, axis='Y', center=(xx, 0.070, zz), verts=8), M["steel"], 0.0, cyl)
        for k, yy in enumerate((0.030, 0.111)):
            add(_cyl(f"{name}_cyl_port_{sd}{k}", 0.005, 0.012, col, axis='X', center=(0.030, yy, 0.0), verts=12), M["brass"], 0.0, cyl)
        rod = G.empty(f"{name}_rod_{sd}", collection=col, size=0.03)
        rod.parent = cyl
        rod.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        add(_cyl(f"{name}_rod_shaft_{sd}", 0.008, 0.14, col, axis='Y', center=(0, -0.075, 0), verts=20), M["chrome"], 0.0, rod)
        add(_cyl(f"{name}_rod_nut_{sd}", 0.012, 0.012, col, axis='Y', center=(0, -0.022, 0), verts=6), M["dark"], 0.0, rod)
        add(_box(f"{name}_rod_eye_{sd}", (0.022, 0.03, 0.022), (0, -0.004, 0), col), M["dark"], 0.002, rod)
        cyls.append(cyl)
        rods.append(rod)
        # air hose: valve fitting (under the valve, +X side) -> cylinder rear-cap port on its +X side (the port moves only a
        # few mm with the cylinder); the hose to the -Y cylinder runs under the housing, outside the cylinder brackets
        by = side * CYL_PIVOT[0]
        ang, _ = _cylinder_state(side, jaw_angle(0.5))
        port = (0.036, by + 0.030 * math.cos(ang), CYL_PIVOT[1] + 0.030 * math.sin(ang))
        fy = VALVE["y"] + 1.4 * (0.022 if side > 0 else -0.022)
        fz = vzc + VALVE["h"] / 2 + 0.008
        if side > 0:
            pts = [(vx0, fy, fz), (vx0 - 0.004, fy, fz + 0.016), (0.056, port[1] + 0.004, port[2] - 0.002), port]
        else:
            pts = [(vx0, fy, fz), (vx0 - 0.004, fy - 0.012, hz1 + 0.014), (0.064, 0.0, hz1 + 0.022), (0.060, by, port[2] - 0.004),
                   (0.047, port[1] + 0.006, port[2]), port]
        parts.append(_curve(f"{name}_hose_{sd}", pts, 0.004, M["hose"], col, root))
        # sensor cable: sensor -> along the bottom edge of the -X face -> junction box plug
        sy = side * SENSOR_Y
        py = JBOX_Y + (0.011 if side > 0 else -0.011)
        cz = hz1 - 0.006
        if side > 0:
            pts = [(-hx + 0.006, sy - 0.024, hz1 + 0.010), (-hx - 0.004, sy - 0.036, hz1 + 0.004), (-hx - 0.0045, py + 0.018, cz),
                   (jb[0], py, vzc + 0.036), (jb[0], py, vzc + 0.027)]
        else:
            pts = [(-hx + 0.006, sy + 0.024, hz1 + 0.010), (-hx - 0.004, sy + 0.036, hz1 + 0.004), (-hx - 0.0045, -0.05, cz),
                   (-hx - 0.0045, py - 0.018, cz), (jb[0], py, vzc + 0.036), (jb[0], py, vzc + 0.027)]
        parts.append(_curve(f"{name}_sensor_cable_{sd}", pts, 0.0035, M["cable"], col, root))
    g = dict(root=root, tcp=tcp, jaws=jaws, parts=parts, cylinders=cyls, rods=rods, connector=connector, stator=stator, name=name)
    set_open(g, 1.0)
    bpy.context.view_layer.update()
    return g


def set_open(g, s, frame=None):
    """s = 0 closed on a DN250 pipe, GRASP['flange'][3] = resting on the flange hub, 1 = fully open."""
    a = jaw_angle(s)
    for jaw, cyl, rod in zip(g["jaws"], g["cylinders"], g["rods"]):
        side = 1.0 if jaw.location[1] > 0 else -1.0
        jaw.rotation_euler[0] = -side * a
        ang, length = _cylinder_state(side, a)
        cyl.rotation_euler[0] = ang                     # Rx(ang) turns the cylinder's local +Y onto pivot -> pin
        rod.location[1] = length
        if frame is not None:
            jaw.keyframe_insert("rotation_euler", index=0, frame=frame)
            cyl.keyframe_insert("rotation_euler", index=0, frame=frame)
            rod.keyframe_insert("location", index=1, frame=frame)


def obstacles():
    """The gripper moves with the robot: no static geometry."""
    return []
