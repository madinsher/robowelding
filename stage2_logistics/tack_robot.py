"""Tack-welding robot of the assembly station: IRB 4600 x TACK_SCALE (IRB 1600 class) on a fixed pedestal with the
stage-1 MIG torch, a wire feeder on the upper arm, a torch hose package, a wire conduit and a compact power source.

    build(collection=None) -> dict(collection, arm, base, joints, tool0, tcp, arc, torch, links, root, feeder, hose)
        arm   = kin.scaled_arm(TACK_SCALE) (URDFArm)          base = empty 'tack_base' at base_world()
        joints = [tack_J1..tack_J6] (rotation_axis_angle[0] = q, robot_build.set_q compatible)
        tool0 = 'tack_tool0'; tcp = 'tack_tcp' = tool0 @ robot_build.tool_transform()
        arc   = 'tack_arc' (child of tcp, identity: +Z into the work like stage-1 'arc_point'), property 'arc_on'
        torch = robot_build._build_torch(...) result, every object renamed 'tack_torch_*'
    set_q(rob, q, frame=None)
    set_arc(rob, on, frame=None)        'arc_on' 0/1 on the arc empty, CONSTANT key
    base_world() -> numpy 4x4           world pose of the robot base (TACK_BASE, TACK_BASE_Z, TACK_YAW)
    obstacles() -> list[dict(name, center, size, yaw)]   pedestal + static base casting, power source trolley, cables

Layout (world): pedestal centred on the J1 axis at TACK_BASE (top plate at TACK_BASE_Z); the power source trolley with
its argon cylinder stands between the pedestal and the +Y fence (x -3.28 .. -2.60, y 2.95 .. 3.25), behind the robot
(J1 = 0 faces -X, toward the station), outside the cell fence hazard stripe (x < -2.58).  Floor cables run from the
power source to the robot base and (earth return) toward the station table.
"""
import math

import bpy
import mathutils
import numpy as np

import tools  # noqa: F401  (demo_video on sys.path)
from cell import geom as G
from cell import materials
from cell import robot_build as RB
import handler_robot as HR
import kin
import layout2 as L2

NAME = "TackRobot"
PREFIX = "tack"
SCALE = L2.TACK_SCALE
TOOL = RB.tool_transform()

PED_PLATE = (0.56, 0.56, 0.025)             # floor plate (x, y, t)
PED_COLUMN = (0.34, 0.34)                   # square column
PED_TOP = (0.50, 0.42, 0.03)                # machined top plate (x, y, t); top = TACK_BASE_Z
PS_C = (-3.00, 3.10)                        # power source trolley centre (world)
PS_BODY = (0.52, 0.26, 0.50)                # body (x, y, z) on the trolley
PS_TROLLEY_Z = 0.14                         # trolley plate top
GAS_C, GAS_R, GAS_H = (-2.685, 3.10), 0.085, 1.22
EARTH_END = (-3.66, 1.98)                   # earth cable end on the floor, next to the station table (+Y side)
HOSE_R = 0.016                              # torch hose radius (stage-1 hose_tube node group)


def base_world():
    return kin.tr(L2.TACK_BASE[0], L2.TACK_BASE[1], L2.TACK_BASE_Z) @ kin.rotz(L2.TACK_YAW)


# ------------------------------------------------------------------ helpers
def _mats():
    return dict(
        frame=materials.get("painted", color="#2B2F36", roughness=0.5),
        blue=materials.get("painted", color="#1F4E8C", roughness=0.42),
        steel=materials.get("machined_steel"),
        dark=materials.get("dark_metal"),
        black=materials.get("black_plastic"),
        rubber=materials.get("rubber"),
        cable=materials.get("cable_black"),
        yellow=materials.get("safety_yellow"),
        red=materials.get("safety_red"),
        brass=materials.get("brass"),
        copper=materials.get("copper"),
        welder_red=materials.get("painted", color="#B01818", roughness=0.35, coat=0.25),
        feeder=materials.get("painted", color="#B3B3B3", roughness=0.4),
        gas=materials.get("painted", color="#2E6B3A", roughness=0.45),
        label=materials.get("painted", color="#C9CCD0", roughness=0.35, coat=0.0),
        screen=materials.get("emissive", color="#8FD0FF", strength=1.5),
        led=materials.get("emissive", color="#30FF40", strength=8.0),
        hose=RB._hose_material(),
    )


def _local(ob, parent, M=None):
    ob.parent = parent
    ob.matrix_parent_inverse = mathutils.Matrix.Identity(4)
    ob.matrix_basis = M if M is not None else mathutils.Matrix.Identity(4)
    return ob


def _box(name, size, center, col, mat, parent, bevel=0.0, rot=None):
    ob = G.box(name, size, (0, 0, 0), col, bevel=bevel)
    ob.data.materials.append(mat)
    M = mathutils.Matrix.Translation(center)
    if rot is not None:
        M = M @ mathutils.Euler(rot).to_matrix().to_4x4()
    return _local(ob, parent, M)


def _cyl(name, r, h, center, col, mat, parent, rot=(0, 0, 0), verts=24):
    ob = G.cylinder(name, r, h, (0, 0, 0), (0, 0, 0), vertices=verts, collection=col)
    ob.data.materials.append(mat)
    return _local(ob, parent, mathutils.Matrix.Translation(center) @ mathutils.Euler(rot).to_matrix().to_4x4())


def _rev(name, profile, center, col, mat, parent, segs=24, rot=(0, 0, 0)):
    ob = G.revolve(name, profile, segments=segs, axis='Z', close=True, collection=col)
    ob.data.materials.append(mat)
    return _local(ob, parent, mathutils.Matrix.Translation(center) @ mathutils.Euler(rot).to_matrix().to_4x4())


def _static_curve(name, pts, radius, mat, col, res=8):
    """Bezier tube through world points (AUTO handles)."""
    cu = bpy.data.curves.new(name, 'CURVE')
    cu.dimensions = '3D'
    cu.resolution_u = res
    cu.bevel_depth = radius
    cu.bevel_resolution = 2
    cu.use_fill_caps = True
    sp = cu.splines.new('BEZIER')
    sp.bezier_points.add(len(pts) - 1)
    for p, xyz in zip(sp.bezier_points, pts):
        p.co = tuple(float(v) for v in xyz)
        p.handle_left_type = p.handle_right_type = 'AUTO'
    ob = bpy.data.objects.new(name, cu)
    col.objects.link(ob)
    ob.data.materials.append(mat)
    return ob


def _hooked_curve(name, anchors, col, radius=None, mat=None, tube_group=None):
    """Bezier curve whose control points (and handles) are hooked to anchor empties.  anchors: list of
    (parent, local offset, local tangent, (handle in, handle out)); the curve object stays at the world origin."""
    cu = bpy.data.curves.new(name + "_curve", 'CURVE')
    cu.dimensions = '3D'
    cu.resolution_u = 12
    if radius is not None:
        cu.bevel_depth = radius
        cu.bevel_resolution = 3
        cu.use_fill_caps = True
    sp = cu.splines.new('BEZIER')
    sp.bezier_points.add(len(anchors) - 1)
    ob = bpy.data.objects.new(name, cu)
    col.objects.link(ob)
    empties = []
    for i, (par, off, tan, hl) in enumerate(anchors):
        e = G.empty(f"{name}_anchor{i}", collection=col, size=0.03)
        _local(e, par, mathutils.Matrix.Translation(off))
        empties.append(e)
    bpy.context.view_layer.update()
    for i, (e, (par, off, tan, hl)) in enumerate(zip(empties, anchors)):
        p = sp.bezier_points[i]
        co = e.matrix_world.translation.copy()
        t = (par.matrix_world.to_3x3() @ mathutils.Vector(tan)).normalized()
        p.handle_left_type = p.handle_right_type = 'FREE'
        p.co = co
        p.handle_left = co - t * hl[0]
        p.handle_right = co + t * hl[1]
    for i, e in enumerate(empties):
        m = ob.modifiers.new(f"hook{i}", 'HOOK')
        m.object = e
        m.vertex_indices_set([3 * i, 3 * i + 1, 3 * i + 2])
        m.matrix_inverse = e.matrix_world.inverted()
        m.center = e.matrix_world.translation
    if tube_group is not None:
        gm = ob.modifiers.new("tube", 'NODES')
        gm.node_group = tube_group
    elif mat is not None:
        ob.data.materials.append(mat)
    return ob, empties


# ------------------------------------------------------------------ parts
def _pedestal(col, root, M):
    """Floor plate with anchors, square column with gussets, machined top plate (top = TACK_BASE_Z).  root: empty at
    the J1 axis on the floor, world-aligned."""
    zt = L2.TACK_BASE_Z
    px, py, pt = PED_PLATE
    cx, cy = PED_COLUMN
    tx, ty, tt = PED_TOP
    objs = [_box(f"{PREFIX}_ped_plate", (px, py, pt), (0, 0, pt / 2), col, M["frame"], root, bevel=0.004)]
    objs.append(_box(f"{PREFIX}_ped_column", (cx, cy, zt - tt - pt), (0, 0, pt + (zt - tt - pt) / 2), col, M["frame"], root, bevel=0.006))
    objs.append(_box(f"{PREFIX}_ped_top", (tx, ty, tt), (0, 0, zt - tt / 2), col, M["steel"], root, bevel=0.003))
    gz = 0.16
    for k, (dx, dy) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
        # triangular gusset standing on the floor plate against the column face
        L_ = (px - cx) / 2 - 0.01
        c = mathutils.Vector((dx * (cx / 2 + L_ / 2), dy * (cy / 2 + L_ / 2), pt))
        ang = math.atan2(dy, dx)
        verts = [(-L_ / 2, -0.006, 0), (L_ / 2, -0.006, 0), (-L_ / 2, -0.006, gz), (-L_ / 2, 0.006, 0), (L_ / 2, 0.006, 0), (-L_ / 2, 0.006, gz)]
        faces = [(0, 1, 2), (3, 5, 4), (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0)]
        g = G.new_object(f"{PREFIX}_ped_gusset{k}", verts, faces, col, smooth=False)
        g.data.materials.append(M["frame"])
        _local(g, root, mathutils.Matrix.Translation(c) @ mathutils.Matrix.Rotation(ang, 4, 'Z'))
        objs.append(g)
    for k, (dx, dy) in enumerate(((1, 1), (-1, 1), (-1, -1), (1, -1))):        # chemical anchors: washer + nut + stud
        c = (dx * (px / 2 - 0.045), dy * (py / 2 - 0.045))
        objs.append(_cyl(f"{PREFIX}_ped_washer{k}", 0.022, 0.004, (c[0], c[1], pt + 0.002), col, M["steel"], root, verts=16))
        objs.append(_cyl(f"{PREFIX}_ped_nut{k}", 0.016, 0.014, (c[0], c[1], pt + 0.011), col, M["dark"], root, verts=6))
        objs.append(_cyl(f"{PREFIX}_ped_stud{k}", 0.008, 0.03, (c[0], c[1], pt + 0.02), col, M["steel"], root, verts=10))
    for k, (dx, dy) in enumerate(((1, 1), (-1, 1), (-1, -1), (1, -1))):        # robot base screws in the top plate
        c = (dx * (tx / 2 - 0.03), dy * (ty / 2 - 0.03))
        objs.append(_cyl(f"{PREFIX}_ped_screw{k}", 0.011, 0.008, (c[0], c[1], zt + 0.004), col, M["dark"], root, verts=6))
    # hazard band and nameplate on the column (+X face toward the cell / camera side, -X face toward the station)
    for k, s in enumerate((1, -1)):
        objs.append(_box(f"{PREFIX}_ped_band{k}", (0.002, cy - 0.04, 0.05), (s * (cx / 2 + 0.001), 0.0, zt - tt - 0.05), col,
                         M["yellow"], root))
    objs.append(_box(f"{PREFIX}_ped_label", (0.002, 0.12, 0.06), (-(cx / 2 + 0.0015), 0.0, 0.16), col, M["label"], root))
    return objs


def _feeder(col, j3, M):
    """Compact wire feeder on top of the upper-arm housing (link_3): body, red lid, spool cover, euro connector,
    conduit inlet.  Returns dict(box, outlet (local point), inlet (local point))."""
    z0 = 0.192                      # link_3 top in the J3 frame (scaled mesh) + clearance
    sx, sy, sz = 0.22, 0.14, 0.15
    c = (0.03, 0.02, z0 + sz / 2)
    box = _box(f"{PREFIX}_feeder", (sx, sy, sz), c, col, M["feeder"], j3, bevel=0.008)
    parts = [box]
    parts.append(_box(f"{PREFIX}_feeder_lid", (sx - 0.03, sy - 0.03, 0.012), (c[0], c[1], z0 + sz + 0.006), col, M["red"], j3, bevel=0.003))
    parts.append(_box(f"{PREFIX}_feeder_foot", (sx - 0.04, sy - 0.02, 0.012), (c[0], c[1], z0 - 0.004), col, M["dark"], j3))
    parts.append(_cyl(f"{PREFIX}_feeder_cover", 0.052, 0.008, (c[0] - 0.01, c[1] + sy / 2 + 0.004, c[2] + 0.005), col, M["black"], j3,
                      rot=(math.pi / 2, 0, 0), verts=32))
    parts.append(_cyl(f"{PREFIX}_feeder_knob", 0.012, 0.012, (c[0] - 0.01, c[1] + sy / 2 + 0.013, c[2] + 0.005), col, M["dark"], j3,
                      rot=(math.pi / 2, 0, 0), verts=12))
    outlet = (c[0] + sx / 2 + 0.018, c[1], c[2] - 0.03)
    parts.append(_cyl(f"{PREFIX}_feeder_euro", 0.022, 0.03, (c[0] + sx / 2 + 0.008, c[1], c[2] - 0.03), col, M["brass"], j3,
                      rot=(0, math.pi / 2, 0), verts=20))
    inlet = (c[0] - sx / 2 - 0.012, c[1], c[2] + 0.02)
    parts.append(_cyl(f"{PREFIX}_feeder_inlet", 0.014, 0.022, (c[0] - sx / 2 - 0.004, c[1], c[2] + 0.02), col, M["black"], j3,
                      rot=(0, math.pi / 2, 0), verts=16))
    parts.append(_box(f"{PREFIX}_feeder_label", (0.08, 0.002, 0.04), (c[0] + 0.03, c[1] - sy / 2 - 0.001, c[2] + 0.02), col,
                      M["label"], j3))
    return dict(box=box, parts=parts, outlet=outlet, inlet=inlet)


def _torch_hose(col, arm, torch, feeder, M):
    """Torch hose package: feeder euro connector -> two clamps on the forearm -> over the wrist -> torch bracket side
    (anchors like stage-1 robot_build._build_hose, scaled forearm geometry)."""
    J = arm["joints"]
    zc = 0.058 + 0.022                  # hose centre above the forearm axis (clamp blocks on the forearm top)
    anchors = [
        (J[2], feeder["outlet"], (1.0, 0.0, -0.3), (0.04, 0.07)),
        (J[3], (0.20, 0.0, zc), (1.0, 0.0, 0.0), (0.08, 0.08)),
        (J[3], (0.43, 0.0, zc), (1.0, 0.0, 0.0), (0.07, 0.07)),
        (J[4], (0.0, 0.0, 0.080), (1.0, 0.0, 0.0), (0.05, 0.05)),
        (torch["root"], (-0.052, 0.0, 0.055), (0.45, 0.0, 1.0), (0.05, 0.03)),
    ]
    ob, empties = _hooked_curve(f"{PREFIX}_hose", anchors, col, tube_group=_tube_group())
    clamps = []
    tops = {0.20: 0.018, 0.43: 0.046}   # forearm (link_4) top at y = 0 in the J4 frame (scaled mesh)
    for k, x in enumerate((0.20, 0.43)):
        h = zc - HOSE_R - tops[x] + 0.004
        clamps.append(_box(f"{PREFIX}_hose_clamp{k}", (0.035, 0.035, h), (x, 0.0, tops[x] - 0.004 + h / 2), col, M["dark"], J[3], bevel=0.003))
        strap = G.revolve(f"{PREFIX}_hose_strap{k}", [(HOSE_R + 0.001, -0.01), (HOSE_R + 0.004, -0.01), (HOSE_R + 0.004, 0.01),
                                                       (HOSE_R + 0.001, 0.01)], segments=20, axis='X', collection=col)
        strap.data.materials.append(M["dark"])
        clamps.append(_local(strap, J[3], mathutils.Matrix.Translation((x, 0.0, zc))))
    return dict(curve=ob, anchors=empties, clamps=clamps)


def _tube_group():
    """Stage-1 corrugated hose sweep (robot_build._hose_tube node group 'hose_tube', radius RB.HOSE_RADIUS)."""
    ng = bpy.data.node_groups.get("hose_tube")
    if ng is None:
        cu = bpy.data.curves.new("tack_tmp_hose", 'CURVE')
        tmp = bpy.data.objects.new("tack_tmp_hose", cu)
        RB._hose_tube(tmp, RB._hose_material())
        ng = bpy.data.node_groups.get("hose_tube")
        bpy.data.objects.remove(tmp)
        bpy.data.curves.remove(cu)
    return ng


def _conduit(col, arm, feeder, base_box_pt, M):
    """Wire conduit + control cable bundle from the connector box on the base to the feeder inlet, hooked to the base,
    link_1 (J1), the back of the lower arm (J2) and the feeder (J3)."""
    J = arm["joints"]
    anchors = [
        (arm["base"], base_box_pt, (0.0, 0.0, 1.0), (0.03, 0.06)),
        (J[0], (-0.16, 0.10, 0.40), (0.25, 0.0, 1.0), (0.08, 0.10)),
        (J[1], (-0.105, -0.11, 0.36), (0.0, 0.0, 1.0), (0.12, 0.12)),
        (J[2], feeder["inlet"], (1.0, 0.0, 0.15), (0.12, 0.03)),
    ]
    ob, empties = _hooked_curve(f"{PREFIX}_conduit", anchors, col, radius=0.011, mat=M["cable"])
    clamp = _box(f"{PREFIX}_conduit_clamp", (0.03, 0.035, 0.035), (-0.078 - 0.012, -0.11, 0.36), col, M["dark"], J[1], bevel=0.003)
    return dict(curve=ob, anchors=empties, clamp=clamp)


def _power_source(col, M):
    """Compact MIG power source (stage-1 welder red) on a castor trolley with an argon cylinder at its +X end."""
    root = G.empty(f"{PREFIX}_ps_root", collection=col, size=0.2)
    root.location = (PS_C[0], PS_C[1], 0.0)
    bx, by, bz = PS_BODY
    zt = PS_TROLLEY_Z
    objs = [_box(f"{PREFIX}_ps_trolley", (bx + 0.10, by + 0.06, 0.03), (0.04, 0.0, zt - 0.015), col, M["dark"], root, bevel=0.003)]
    for k, (dx, dy) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1))):
        cx, cy = 0.04 + dx * (bx / 2 + 0.02), dy * (by / 2 - 0.01)
        objs.append(_box(f"{PREFIX}_ps_fork{k}", (0.03, 0.02, 0.05), (cx, cy, zt - 0.055), col, M["dark"], root))
        objs.append(_cyl(f"{PREFIX}_ps_castor{k}", 0.042, 0.028, (cx, cy, 0.042), col, M["rubber"], root, rot=(math.pi / 2, 0, 0), verts=16))
    objs.append(_box(f"{PREFIX}_ps_body", (bx, by, bz - 0.12), (0.0, 0.0, zt + 0.12 + (bz - 0.12) / 2), col, M["welder_red"], root, bevel=0.01))
    objs.append(_box(f"{PREFIX}_ps_body_low", (bx + 0.01, by + 0.01, 0.12), (0.0, 0.0, zt + 0.06), col, M["black"], root, bevel=0.006))
    yf = -by / 2                                     # front panel faces -Y (toward the robot / station side)
    objs.append(_box(f"{PREFIX}_ps_front", (bx - 0.08, 0.008, bz - 0.20), (0.0, yf - 0.004, zt + 0.12 + (bz - 0.12) / 2), col, M["black"], root))
    objs.append(_box(f"{PREFIX}_ps_display", (0.12, 0.004, 0.06), (-0.08, yf - 0.009, zt + bz - 0.10), col, M["screen"], root))
    objs.append(_cyl(f"{PREFIX}_ps_knob", 0.022, 0.02, (0.08, yf - 0.018, zt + bz - 0.10), col, M["black"], root, rot=(math.pi / 2, 0, 0), verts=20))
    objs.append(_box(f"{PREFIX}_ps_led", (0.012, 0.004, 0.008), (0.16, yf - 0.009, zt + bz - 0.07), col, M["led"], root))
    for k in range(3):                                # dinse sockets / control connector
        objs.append(_cyl(f"{PREFIX}_ps_socket{k}", 0.02, 0.03, (-0.12 + 0.12 * k, yf - 0.012, zt + 0.20), col,
                         M["brass"] if k < 2 else M["dark"], root, rot=(math.pi / 2, 0, 0), verts=16))
    objs.append(_box(f"{PREFIX}_ps_handle", (bx - 0.10, 0.025, 0.025), (0.0, 0.0, zt + bz + 0.07), col, M["black"], root, bevel=0.004))
    for k, dx in enumerate((-(bx / 2 - 0.08), bx / 2 - 0.08)):
        objs.append(_box(f"{PREFIX}_ps_handle_leg{k}", (0.025, 0.025, 0.07), (dx, 0.0, zt + bz + 0.035), col, M["black"], root))
    # argon cylinder in a cradle at the +X end, chain, regulator + gauge
    gx = GAS_C[0] - PS_C[0]
    objs.append(_rev(f"{PREFIX}_gas", [(0, 0), (GAS_R, 0), (GAS_R, GAS_H - 0.14), (0.055, GAS_H - 0.03), (0.035, GAS_H), (0, GAS_H)],
                     (gx, 0.0, zt), col, M["gas"], root, segs=28))
    objs.append(_cyl(f"{PREFIX}_gas_valve", 0.016, 0.08, (gx, 0.0, zt + GAS_H + 0.04), col, M["brass"], root, verts=12))
    objs.append(_box(f"{PREFIX}_gas_reg", (0.06, 0.05, 0.05), (gx - 0.03, 0.0, zt + GAS_H + 0.08), col, M["brass"], root))
    objs.append(_cyl(f"{PREFIX}_gas_gauge", 0.026, 0.02, (gx - 0.03, -0.035, zt + GAS_H + 0.09), col, M["steel"], root,
                     rot=(math.pi / 2, 0, 0), verts=16))
    objs.append(_box(f"{PREFIX}_gas_cradle", (0.04, 0.22, 0.05), (gx - GAS_R - 0.01, 0.0, zt + 0.55), col, M["dark"], root))
    objs.append(_cyl(f"{PREFIX}_gas_chain", GAS_R + 0.004, 0.012, (gx, 0.0, zt + 0.80), col, M["steel"], root, verts=24))
    return root, objs


def _floor_cables(col, M, base_box_world):
    """Power / control cable from the power source front to the robot base connector box, gas hose to the conduit,
    earth return cable toward the station table."""
    ps = np.array([PS_C[0], PS_C[1] - PS_BODY[1] / 2 - 0.03, PS_TROLLEY_Z + 0.20])
    b = np.array(base_box_world)
    objs = []
    for k, (dx, r) in enumerate(((-0.12, 0.012), (0.0, 0.009))):
        p0 = ps + np.array([dx, 0.0, 0.0])
        pts = [p0, p0 + np.array([0.0, -0.08, -0.12]), (p0[0], p0[1] - 0.14, 0.012), (b[0] + 0.25 + 0.03 * k, b[1] + 0.25, 0.012),
               (b[0] + 0.17, b[1] + 0.03 * k, 0.012), (b[0] + 0.06, b[1] + 0.03 * k - 0.015, 0.05),
               (b[0] + 0.02, b[1] + 0.03 * k - 0.015, L2.TACK_BASE_Z - 0.06), (b[0], b[1] + 0.03 * k - 0.015, b[2] - 0.03)]
        objs.append(_static_curve(f"{PREFIX}_cable{k}", pts, r, M["cable"], col))
    p0 = ps + np.array([0.12, 0.0, 0.0])
    ex, ey = EARTH_END
    pts = [p0, p0 + np.array([0.0, -0.07, -0.13]), (p0[0], p0[1] - 0.16, 0.013), (L2.TACK_BASE[0] + 0.45, L2.TACK_BASE[1] - 0.10, 0.013),
           (L2.TACK_BASE[0] + 0.20, L2.TACK_BASE[1] - 0.52, 0.013), (ex + 0.12, ey + 0.05, 0.013), (ex, ey, 0.013)]
    objs.append(_static_curve(f"{PREFIX}_earth_cable", pts, 0.013, M["cable"], col))
    lug = G.box(f"{PREFIX}_earth_lug", (0.06, 0.03, 0.02), (ex - 0.03, ey, 0.01), col, bevel=0.003)
    lug.data.materials.append(M["copper"])
    objs.append(lug)
    return objs


# ------------------------------------------------------------------ public API
def build(collection=None):
    col = collection
    if col is None:
        col = bpy.data.collections.new(NAME)
        bpy.context.scene.collection.children.link(col)
    M = _mats()
    root = G.empty(f"{PREFIX}_root", collection=col, size=0.3)
    root.location = (L2.TACK_BASE[0], L2.TACK_BASE[1], 0.0)
    bpy.context.view_layer.update()
    ped = _pedestal(col, root, M)
    arm = HR.build_arm(PREFIX, SCALE, base_world(), col)
    G.set_parent(arm["base"], root, keep_world=True)
    tool0 = arm["tool0"]
    # torch (stage-1 geometry, full size: the IRB 1600-class robot carries the same torch), objects renamed tack_torch_*
    before = set(bpy.data.objects)
    torch = RB._build_torch(col, tool0)
    for ob in set(bpy.data.objects) - before:
        base = ob.name.split(".")[0]
        ob.name = f"{PREFIX}_{base}" if not base.startswith(PREFIX + "_") else base
        if ob.data is not None:
            ob.data.name = ob.name
    tcp = G.empty(f"{PREFIX}_tcp", collection=col, size=0.06, display='ARROWS')
    _local(tcp, tool0, G.M(TOOL))
    arc = G.empty(f"{PREFIX}_arc", collection=col, size=0.03)
    _local(arc, tcp)
    arc["arc_on"] = 0.0
    feeder = _feeder(col, arm["joints"][2], M)
    # connector box on the rear of the base casting (static), cables to the power source
    s = SCALE
    box_pt = (-0.235, 0.0, 0.075)                    # base frame (the casting's rear face is at x = -0.229)
    cbox = _box(f"{PREFIX}_base_connector", (0.05, 0.14, 0.08), (box_pt[0] - 0.02, 0.0, box_pt[2]), col, M["dark"], arm["base"], bevel=0.004)
    hose = _torch_hose(col, arm, torch, feeder, M)
    conduit = _conduit(col, arm, feeder, (box_pt[0] - 0.02, 0.035, box_pt[2] + 0.04), M)
    ps_root, ps_objs = _power_source(col, M)
    bpy.context.view_layer.update()
    bw = arm["base"].matrix_world @ mathutils.Vector((box_pt[0] - 0.045, -0.03, box_pt[2]))
    cables = _floor_cables(col, M, tuple(bw))
    rob = dict(collection=col, arm=arm["arm"], base=arm["base"], joints=arm["joints"], tool0=tool0, tcp=tcp, arc=arc,
               torch=torch, links=arm["links"], root=root, pedestal=ped, feeder=feeder, hose=hose, conduit=conduit,
               connector=cbox, power_source=dict(root=ps_root, objects=ps_objs), cables=cables, scale=s)
    set_q(rob, L2.TACK_Q_HOME)
    bpy.context.view_layer.update()
    return rob


def set_q(rob, q, frame=None):
    for j, e in enumerate(rob["joints"]):
        aa = e.rotation_axis_angle
        e.rotation_axis_angle = (float(q[j]), aa[1], aa[2], aa[3])
        if frame is not None:
            e.keyframe_insert("rotation_axis_angle", index=0, frame=frame)


def set_arc(rob, on, frame=None):
    """Arc switch: property 'arc_on' (0/1) on the arc empty; CONSTANT key when frame is given."""
    arc = rob["arc"]
    arc["arc_on"] = float(on)
    if frame is not None:
        arc.keyframe_insert('["arc_on"]', frame=frame)
        fc = arc.animation_data.action.fcurves.find('["arc_on"]')
        if fc is not None:
            for kp in fc.keyframe_points:
                kp.interpolation = 'CONSTANT'


# ------------------------------------------------------------------ obstacles (pure python)
def obstacles():
    """Conservative world AABBs of the static geometry: pedestal (+ the fixed base casting and its connector box),
    power source trolley with the argon cylinder, floor cables (thin boxes)."""
    out = []

    def add(name, lo, hi):
        out.append(dict(name=name, center=tuple((lo[k] + hi[k]) / 2 for k in range(3)),
                        size=tuple(hi[k] - lo[k] for k in range(3)), yaw=0.0))
    x, y = L2.TACK_BASE
    px, py, _ = PED_PLATE
    add(f"{PREFIX}_pedestal", (x - px / 2 - 0.005, y - py / 2 - 0.005, 0.0), (x + px / 2 + 0.005, y + py / 2 + 0.005, L2.TACK_BASE_Z + 0.01))
    # base casting (J1 turns link_1 above it): base frame x [-0.229, 0.156] (+ connector box to -0.31), |y| <= 0.146,
    # z <= 0.111; yaw = pi maps base x onto world -x
    c, s_ = math.cos(L2.TACK_YAW), math.sin(L2.TACK_YAW)
    pts = [(x + c * bx - s_ * by, y + s_ * bx + c * by) for bx in (-0.31, 0.16) for by in (-0.15, 0.15)]
    add(f"{PREFIX}_base_casting", (min(p[0] for p in pts) - 0.01, min(p[1] for p in pts) - 0.01, L2.TACK_BASE_Z),
        (max(p[0] for p in pts) + 0.01, max(p[1] for p in pts) + 0.01, L2.TACK_BASE_Z + 0.125))
    bx, by, bz = PS_BODY
    add(f"{PREFIX}_power_source", (PS_C[0] - bx / 2 - 0.07, PS_C[1] - by / 2 - 0.07, 0.0),
        (PS_C[0] + bx / 2 + 0.13, PS_C[1] + by / 2 + 0.035, PS_TROLLEY_Z + bz + 0.10))
    add(f"{PREFIX}_gas", (GAS_C[0] - GAS_R - 0.03, GAS_C[1] - GAS_R - 0.03, 0.0),
        (GAS_C[0] + GAS_R + 0.005, GAS_C[1] + GAS_R + 0.005, PS_TROLLEY_Z + GAS_H + 0.13))
    add(f"{PREFIX}_floor_cables", (EARTH_END[0] - 0.07, EARTH_END[1] - 0.03, 0.0), (PS_C[0] + 0.20, PS_C[1] - PS_BODY[1] / 2, 0.03))
    return out
