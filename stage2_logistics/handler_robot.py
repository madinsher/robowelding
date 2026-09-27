"""Handler (loader) robot of the logistics zone: IRB 4600 x HANDLER_SCALE on a floor track along X, dress pack, tong gripper.

    build_arm(prefix, scale, base_world, collection, colours=None) -> dict(arm, base, joints, tool0, links)
        generic scaled IRB 4600 (also used by the tack robot): link meshes scaled by `scale`, joint origins as
        kin.scaled_arm(scale); joint empties <prefix>_J1..J6 rotate by rotation_axis_angle[0] (robot_build.set_q works).
    build(collection=None) -> dict(collection, arm, carriage, base, joints, tool0, links, gripper, chain_bend, hose, swivel)
    set_track(rob, x, frame=None)   carriage x (= robot J1 axis x); also moves the drag-chain bend
    set_q(rob, q, frame=None)
    obstacles()                     static track geometry (world AABBs)

Track (world, y = HANDLER_TRACK_Y): bed with levelling feet, two linear rails, gear rack, end stops with rubber
buffers inside HANDLER_TRACK_BED; a galvanised trough on the +Y side with an energy chain whose U-bend travels at half
the carriage speed (curve hooked to an empty keyed by set_track, swept into a band by geometry nodes).  Carriage:
body, top plate at HANDLER_TRACK_TOP_Z, guide blocks, rack-and-pinion drive (gearbox + servo motor) on the -Y side,
chain bracket on the +Y side, yellow buffers.  Everything stays inside |y| < 0.75.
"""
import math

import bpy
import mathutils

import tools  # noqa: F401
from cell import geom as G
from cell import materials
from cell import robot_build as RB
import gripper as GR
import kin
import layout2 as L2

PREFIX = "hnd"
COLLECTION = "Handler"
DEG = math.pi / 180.0

BED_X = (L2.HANDLER_TRACK_BED[0] + 0.06, L2.HANDLER_TRACK_BED[1] - 0.06)    # bed between the end stops
BED_HW, BED_H = 0.47, 0.24
RAIL_Y, RAIL_TOP = 0.32, 0.29
RACK_Y = -0.44
CARRIAGE_X = (-0.53, 0.36)          # carriage body x range relative to the J1 axis (buffers add 0.02 each side)
CARRIAGE_HW = 0.55
TROUGH_Y, TROUGH_W, TROUGH_H = 0.62, 0.22, 0.17
CHAIN_W, CHAIN_H, CHAIN_R = 0.16, 0.06, 0.06      # energy chain width, link height, bend radius (centre line)
CHAIN_Z0 = 0.005 + CHAIN_H / 2                    # lower run centre height
CHAIN_OFF = -0.10                                 # chain end on the carriage, x relative to the J1 axis
CHAIN_PITCH = 0.075
CHAIN_FIX_X = (L2.HANDLER_TRACK_X[0] + L2.HANDLER_TRACK_X[1]) / 2 + CHAIN_OFF
CHAIN_LEN = CHAIN_FIX_X - (L2.HANDLER_TRACK_X[0] + CHAIN_OFF) + math.pi * CHAIN_R + 0.10
HOSE_R = 0.022


def chain_bend_x(x):
    """x of the chain U-bend (lower/upper run junction) for carriage x; the bend faces -X."""
    c = x + CHAIN_OFF
    return (CHAIN_FIX_X + c - CHAIN_LEN + math.pi * CHAIN_R) / 2


# ------------------------------------------------------------------ helpers
def _mats():
    return dict(
        frame=materials.get("painted", color="#2B2F36", roughness=0.5),
        blue=materials.get("painted", color="#1F4E8C", roughness=0.42),
        motor=materials.get("painted", color="#33363A", roughness=0.5, coat=0.0),
        steel=materials.get("machined_steel"),
        dark=materials.get("dark_metal"),
        black=materials.get("black_plastic"),
        rubber=materials.get("rubber"),
        galv=materials.get("galvanized"),
        yellow=materials.get("safety_yellow"),
        cable=materials.get("cable_black"),
        label=materials.get("painted", color="#C9CCD0", roughness=0.35, coat=0.0),
    )


def _box(name, size, center, col, mat, bevel=0.0, parent=None):
    ob = G.box(name, size, center, col, bevel=bevel)
    ob.data.materials.append(mat)
    if parent is not None:
        G.set_parent(ob, parent)
    return ob


def _cyl(name, r, h, center, col, mat, rot=(0, 0, 0), verts=32, parent=None):
    ob = G.cylinder(name, r, h, center, rot, vertices=verts, collection=col)
    ob.data.materials.append(mat)
    if parent is not None:
        G.set_parent(ob, parent)
    return ob


def _prism_x(name, poly, x0, x1, col, mat, y0=0.0):
    """2D polygon [(y, z)] extruded along world X from x0 to x1, shifted by y0."""
    n = len(poly)
    verts = [(x0, y0 + y, z) for y, z in poly] + [(x1, y0 + y, z) for y, z in poly]
    faces = [tuple(range(n)), tuple(range(2 * n - 1, n - 1, -1))] + [(i, n + i, n + (i + 1) % n, (i + 1) % n) for i in range(n)]
    ob = G.new_object(name, verts, faces, col, smooth=False)
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(ob.data)
    bm.free()
    ob.data.materials.append(mat)
    return ob


def _local(child, parent, M=None):
    child.parent = parent
    child.matrix_parent_inverse = mathutils.Matrix.Identity(4)
    child.matrix_basis = M if M is not None else mathutils.Matrix.Identity(4)


# ------------------------------------------------------------------ generic scaled arm
def build_arm(prefix, scale, base_world, collection, colours=None):
    """IRB 4600 x scale.  base_world: numpy 4x4 world pose of the base frame (parent the returned `base` to a carriage
    or pedestal afterwards with keep_world).  colours: {link name: material or material-name} overrides
    (default: base_link and link_6 grey, link_1..link_5 ABB orange)."""
    col = collection
    arm = kin.scaled_arm(scale)
    orange, grey = materials.get("abb_orange"), materials.get("abb_grey")
    cmap = {arm.base_link: grey, "link_1": orange, "link_2": orange, "link_3": orange, "link_4": orange,
            "link_5": orange, "link_6": grey}
    for k, v in (colours or {}).items():
        cmap[k] = materials.get(v) if isinstance(v, str) else v
    S = mathutils.Matrix.Scale(scale, 4)

    def mesh(link, name):
        info = arm.links[link]
        o = RB._load_mesh(name, info["mesh"], col)
        o.data.transform(S)
        o.data.update()
        o.data.materials.append(cmap.get(link, orange))
        return o, info

    base = G.empty(f"{prefix}_base", collection=col, size=0.3 * scale)
    base.matrix_world = G.M(base_world)
    links = {}
    bl, info = mesh(arm.base_link, f"{prefix}_{arm.base_link}")
    _local(bl, base, G.M(info["vis_origin"]))
    links[arm.base_link] = bl
    joints = []
    parent = base
    tool0 = None
    for j in arm.chain:
        if j["type"] != "revolute":
            e = G.empty(f"{prefix}_{j['child']}", collection=col, size=0.15 * scale)
            _local(e, parent, G.M(j["origin"]))
            parent = e
            tool0 = e
            continue
        e = G.empty(f"{prefix}_J{len(joints) + 1}", collection=col, size=0.2 * scale)
        e.parent = parent
        e.matrix_parent_inverse = G.M(j["origin"])
        e.rotation_mode = 'AXIS_ANGLE'
        e.rotation_axis_angle = (0.0, *j["axis"])
        joints.append(e)
        if arm.links[j["child"]]["mesh"]:
            lo, info = mesh(j["child"], f"{prefix}_{j['child']}")
            _local(lo, e, G.M(info["vis_origin"]))
            links[j["child"]] = lo
        parent = e
    tool0.name = f"{prefix}_tool0"
    # ABB-style grey nameplate on the outer (-Y) face of the lower arm
    npl = G.box(f"{prefix}_nameplate", (0.13 * scale, 0.004, 0.055 * scale), (0, 0, 0), col)
    npl.data.materials.append(materials.get("painted", color="#C9CCD0", roughness=0.35, coat=0.0))
    _local(npl, joints[1], mathutils.Matrix.Translation((0.0, -0.2545 * scale, 0.55 * scale)) @ mathutils.Matrix.Rotation(0.02, 4, 'X'))
    bpy.context.view_layer.update()
    return dict(arm=arm, base=base, joints=joints, tool0=tool0, links=links, scale=scale, collection=col)


def set_q(rob, q, frame=None):
    for j, e in enumerate(rob["joints"]):
        aa = e.rotation_axis_angle
        e.rotation_axis_angle = (float(q[j]), aa[1], aa[2], aa[3])
        if frame is not None:
            e.keyframe_insert("rotation_axis_angle", index=0, frame=frame)


def set_track(rob, x, frame=None):
    rob["carriage"].location[0] = float(x)
    rob["chain_bend"].location[0] = chain_bend_x(x)
    if frame is not None:
        rob["carriage"].keyframe_insert("location", index=0, frame=frame)
        rob["chain_bend"].keyframe_insert("location", index=0, frame=frame)


def base_world(x):
    """World pose of the handler robot base for carriage x (numpy 4x4)."""
    return kin.tr(x, L2.HANDLER_TRACK_Y, L2.HANDLER_TRACK_TOP_Z) @ kin.rotz(L2.HANDLER_YAW)


# ------------------------------------------------------------------ track
def _build_track(col, M):
    x0, x1 = BED_X
    xc, L = (x0 + x1) / 2, x1 - x0
    y = L2.HANDLER_TRACK_Y
    objs = []
    objs.append(_box(f"{PREFIX}_bed", (L, 2 * BED_HW, BED_H - 0.02), (xc, y, 0.02 + (BED_H - 0.02) / 2), col, M["frame"], 0.012))
    # bolted rail seats (steel strips under the rails) and cross covers every 1.2 m
    for s in (-1, 1):
        objs.append(_box(f"{PREFIX}_rail_seat{'p' if s > 0 else 'n'}", (L - 0.04, 0.10, 0.012), (xc, y + s * RAIL_Y, BED_H + 0.006), col, M["dark"]))
    objs.append(_box(f"{PREFIX}_rack_seat", (L - 0.08, 0.06, 0.012), (xc, y + RACK_Y, BED_H + 0.006), col, M["dark"]))
    n_cov = int(L / 1.2)
    for k in range(n_cov + 1):
        cx = x0 + 0.1 + k * (L - 0.2) / n_cov
        objs.append(_box(f"{PREFIX}_bed_joint{k}", (0.012, 2 * BED_HW + 0.004, BED_H - 0.03), (cx, y, 0.02 + (BED_H - 0.03) / 2), col, M["dark"]))
    n_feet = int(L / 1.0) + 1
    for k in range(n_feet):
        fx = x0 + 0.12 + k * (L - 0.24) / (n_feet - 1)
        for s in (-1, 1):
            objs.append(_cyl(f"{PREFIX}_foot{k}{'p' if s > 0 else 'n'}", 0.05, 0.02, (fx, y + s * (BED_HW + 0.03), 0.01), col, M["dark"], verts=24))
            objs.append(_box(f"{PREFIX}_foot_lug{k}{'p' if s > 0 else 'n'}", (0.10, 0.06, 0.012), (fx, y + s * (BED_HW + 0.02), 0.026), col, M["frame"]))
            objs.append(_cyl(f"{PREFIX}_foot_bolt{k}{'p' if s > 0 else 'n'}", 0.014, 0.03, (fx, y + s * (BED_HW + 0.035), 0.04), col, M["steel"], verts=6))
    for s in (-1, 1):
        objs.append(_box(f"{PREFIX}_rail{'p' if s > 0 else 'n'}", (L - 0.06, 0.055, RAIL_TOP - BED_H - 0.012), (xc, y + s * RAIL_Y, (RAIL_TOP + BED_H + 0.012) / 2), col, M["steel"]))
    objs.append(_box(f"{PREFIX}_rack", (L - 0.10, 0.035, 0.038), (xc, y + RACK_Y, BED_H + 0.012 + 0.019), col, M["dark"]))
    # side cover strip with hazard look (yellow) along the bed top edge
    for s in (-1, 1):
        objs.append(_box(f"{PREFIX}_bed_stripe{'p' if s > 0 else 'n'}", (L - 0.02, 0.004, 0.05), (xc, y + s * (BED_HW + 0.002), BED_H - 0.04), col, M["yellow"]))
    # end stops with rubber buffers
    for tag, xs, face in (("n", L2.HANDLER_TRACK_BED[0] + 0.03, 1), ("p", L2.HANDLER_TRACK_BED[1] - 0.03, -1)):
        objs.append(_box(f"{PREFIX}_stop_{tag}", (0.06, 0.90, 0.42), (xs, y, 0.21), col, M["yellow"], 0.006))
        for s in (-1, 1):
            objs.append(_cyl(f"{PREFIX}_stop_buffer_{tag}{'p' if s > 0 else 'n'}", 0.045, 0.02, (xs + face * 0.04, y + s * 0.28, 0.40), col, M["rubber"],
                             rot=(0, math.pi / 2, 0), verts=24))
    # cable-chain trough (open U channel) with glide ledges for the upper run
    t = 0.004
    u = [(-TROUGH_W / 2, 0.0), (TROUGH_W / 2, 0.0), (TROUGH_W / 2, TROUGH_H), (TROUGH_W / 2 - t, TROUGH_H), (TROUGH_W / 2 - t, t),
         (-TROUGH_W / 2 + t, t), (-TROUGH_W / 2 + t, TROUGH_H), (-TROUGH_W / 2, TROUGH_H)]
    tr = _prism_x(f"{PREFIX}_trough", u, x0, x1, col, M["galv"], y0=y + TROUGH_Y)
    objs.append(tr)
    zl = CHAIN_Z0 + 2 * CHAIN_R - CHAIN_H / 2          # upper-run underside rests on the ledges
    for s in (-1, 1):
        led = [(0.0, 0.0), (0.022, 0.0), (0.022, 0.004), (0.004, 0.004), (0.004, 0.03), (0.0, 0.03)]
        led = [(s * (TROUGH_W / 2 - t) - s * p[0], zl - 0.004 + p[1]) for p in led]
        if s < 0:
            led = led[::-1]
        objs.append(_prism_x(f"{PREFIX}_trough_ledge{'p' if s > 0 else 'n'}", led, x0 + 0.02, x1 - 0.02, col, M["galv"], y0=y + TROUGH_Y))
    # fixed end of the chain: clamp bracket + cable exit to the floor duct
    objs.append(_box(f"{PREFIX}_chain_fix", (0.10, CHAIN_W + 0.02, 0.07), (CHAIN_FIX_X + 0.05, y + TROUGH_Y, 0.04), col, M["dark"], 0.004))
    objs.append(_box(f"{PREFIX}_floor_duct", (0.30, 0.012, 0.10), (CHAIN_FIX_X + 0.10, y + TROUGH_Y + TROUGH_W / 2 + 0.006, 0.05), col, M["galv"]))
    return objs


def _chain_material():
    m = bpy.data.materials.get(f"{PREFIX}_chain")
    if m is not None:
        return m
    m = bpy.data.materials.new(f"{PREFIX}_chain")
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.018, 0.018, 0.02, 1.0)
    b.inputs["Roughness"].default_value = 0.55
    b.inputs["Specular IOR Level"].default_value = 0.4
    at = nt.nodes.new("ShaderNodeAttribute")
    at.attribute_name = "chain_u"
    div = nt.nodes.new("ShaderNodeMath"); div.operation = 'DIVIDE'; div.inputs[1].default_value = CHAIN_PITCH
    nt.links.new(at.outputs["Fac"], div.inputs[0])
    fr = nt.nodes.new("ShaderNodeMath"); fr.operation = 'FRACT'
    nt.links.new(div.outputs[0], fr.inputs[0])
    # rounded link-joint groove at every pitch + a lighter pin boss in the middle of each link side
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.elements[0].position = 0.0; cr.elements[0].color = (0.0, 0.0, 0.0, 1)
    e = cr.elements.new(0.06); e.color = (1.0, 1.0, 1.0, 1)
    e = cr.elements.new(0.94); e.color = (1.0, 1.0, 1.0, 1)
    cr.elements[-1].position = 1.0; cr.elements[-1].color = (0.0, 0.0, 0.0, 1)
    nt.links.new(fr.outputs[0], ramp.inputs["Fac"])
    bump = nt.nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.8; bump.inputs["Distance"].default_value = 0.004
    nt.links.new(ramp.outputs["Color"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    mix = nt.nodes.new("ShaderNodeMix"); mix.data_type = 'RGBA'; mix.blend_type = 'MIX'
    mix.inputs[6].default_value = (0.004, 0.004, 0.005, 1.0)
    mix.inputs[7].default_value = (0.03, 0.03, 0.033, 1.0)
    nt.links.new(ramp.outputs["Color"], mix.inputs["Factor"])
    nt.links.new(mix.outputs[2], b.inputs["Base Color"])
    return m


def _chain_band_group(mat):
    ng = bpy.data.node_groups.get(f"{PREFIX}_chain_band")
    if ng is not None:
        return ng
    ng = bpy.data.node_groups.new(f"{PREFIX}_chain_band", 'GeometryNodeTree')
    ng.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    n = ng.nodes
    gin, gout = n.new("NodeGroupInput"), n.new("NodeGroupOutput")
    res = n.new("GeometryNodeResampleCurve"); res.mode = 'LENGTH'; res.inputs["Length"].default_value = 0.01
    nrm = n.new("GeometryNodeSetCurveNormal"); nrm.mode = 'FREE'; nrm.inputs["Normal"].default_value = (0.0, 1.0, 0.0)
    par = n.new("GeometryNodeSplineParameter")
    st = n.new("GeometryNodeStoreNamedAttribute"); st.data_type = 'FLOAT'; st.domain = 'POINT'; st.inputs["Name"].default_value = "chain_u"
    quad = n.new("GeometryNodeCurvePrimitiveQuadrilateral")
    quad.inputs["Width"].default_value = CHAIN_W
    quad.inputs["Height"].default_value = CHAIN_H
    c2m = n.new("GeometryNodeCurveToMesh"); c2m.inputs["Fill Caps"].default_value = True
    setm = n.new("GeometryNodeSetMaterial"); setm.inputs["Material"].default_value = mat
    lk = ng.links.new
    lk(gin.outputs["Geometry"], res.inputs["Curve"]); lk(res.outputs["Curve"], nrm.inputs["Curve"])
    lk(nrm.outputs["Curve"], st.inputs["Geometry"]); lk(par.outputs["Length"], st.inputs["Value"])
    lk(st.outputs["Geometry"], c2m.inputs["Curve"]); lk(quad.outputs["Curve"], c2m.inputs["Profile Curve"])
    lk(c2m.outputs["Mesh"], setm.inputs["Geometry"]); lk(setm.outputs["Geometry"], gout.inputs["Geometry"])
    return ng


def _build_chain(col, carriage, M):
    """Energy chain: bezier path fixed end -> lower run -> U-bend (hooked to the bend empty) -> upper run -> carriage."""
    y = L2.HANDLER_TRACK_Y + TROUGH_Y
    z0, z1 = CHAIN_Z0, CHAIN_Z0 + 2 * CHAIN_R
    x = carriage.location[0]
    xb = chain_bend_x(x)
    bend = G.empty(f"{PREFIX}_chain_bend", collection=col, size=0.1)
    bend.location = (xb, y, 0.0)
    end = G.empty(f"{PREFIX}_chain_end", collection=col, size=0.08)
    _local(end, carriage, mathutils.Matrix.Translation((CHAIN_OFF, y, z1)))
    bpy.context.view_layer.update()
    cu = bpy.data.curves.new(f"{PREFIX}_chain_path", 'CURVE')
    cu.dimensions = '3D'
    cu.resolution_u = 16
    sp = cu.splines.new('BEZIER')
    k = 0.5523 * CHAIN_R
    h = 0.02
    V = mathutils.Vector
    pts = [(V((CHAIN_FIX_X, y, z0)), V((h, 0, 0)), V((-h, 0, 0))),
           (V((xb, y, z0)), V((h, 0, 0)), V((-k, 0, 0))),
           (V((xb - CHAIN_R, y, (z0 + z1) / 2)), V((0, 0, -k)), V((0, 0, k))),
           (V((xb, y, z1)), V((-k, 0, 0)), V((h, 0, 0))),
           (V((x + CHAIN_OFF, y, z1)), V((-h, 0, 0)), V((h, 0, 0)))]
    sp.bezier_points.add(len(pts) - 1)
    for p, (co, hl, hr) in zip(sp.bezier_points, pts):
        p.handle_left_type = p.handle_right_type = 'FREE'
        p.co, p.handle_left, p.handle_right = co, co + hl, co + hr
    ob = bpy.data.objects.new(f"{PREFIX}_chain", cu)
    col.objects.link(ob)
    for i, e in ((1, bend), (2, bend), (3, bend), (4, end)):
        m = ob.modifiers.new(f"hook{i}", 'HOOK')
        m.object = e
        m.vertex_indices_set([3 * i, 3 * i + 1, 3 * i + 2])
        m.matrix_inverse = e.matrix_world.inverted()
        m.center = e.matrix_world.translation
    gm = ob.modifiers.new("band", 'NODES')
    gm.node_group = _chain_band_group(_chain_material())
    # chain end block + bracket on the carriage (carriage-local coordinates)
    parts = []
    blk = G.box(f"{PREFIX}_chain_end_block", (0.08, CHAIN_W + 0.012, CHAIN_H + 0.01), (0, 0, 0), col, bevel=0.003)
    blk.data.materials.append(M["dark"])
    _local(blk, carriage, mathutils.Matrix.Translation((CHAIN_OFF + 0.04, y, z1)))
    plate = G.box(f"{PREFIX}_chain_bracket_h", (0.12, 0.20, 0.012), (0, 0, 0), col, bevel=0.002)
    plate.data.materials.append(M["frame"])
    _local(plate, carriage, mathutils.Matrix.Translation((CHAIN_OFF + 0.04, y - 0.01, z1 + CHAIN_H / 2 + 0.011)))
    vert = G.box(f"{PREFIX}_chain_bracket_v", (0.12, 0.012, 0.30 - (z1 + CHAIN_H / 2) + 0.02), (0, 0, 0), col, bevel=0.002)
    vert.data.materials.append(M["frame"])
    _local(vert, carriage, mathutils.Matrix.Translation((CHAIN_OFF + 0.04, CARRIAGE_HW - 0.07 + 0.006 + 0.075, (0.30 + z1 + CHAIN_H / 2) / 2 + 0.01)))
    parts += [blk, plate, vert]
    return dict(curve=ob, bend=bend, end=end, parts=parts)


# ------------------------------------------------------------------ carriage
def _build_carriage(col, M, x):
    car = G.empty(f"{PREFIX}_carriage", collection=col, size=0.4)
    car.location = (x, L2.HANDLER_TRACK_Y, 0.0)
    xa, xb = CARRIAGE_X
    xm, lx = (xa + xb) / 2, xb - xa
    top = L2.HANDLER_TRACK_TOP_Z
    parts = []

    def add(ob):
        _local(ob, car, ob.matrix_basis.copy())      # the object was created at its carriage-local position
        parts.append(ob)
        return ob
    ob = G.box(f"{PREFIX}_carriage_body", (lx, 2 * CARRIAGE_HW, top - 0.02 - 0.30), (xm, 0, 0.30 + (top - 0.32) / 2), col, bevel=0.012)
    ob.data.materials.append(M["frame"]); add(ob)
    ob = G.box(f"{PREFIX}_carriage_plate", (lx, 0.84, 0.02), (xm, 0, top - 0.01), col, bevel=0.003)
    ob.data.materials.append(M["steel"]); add(ob)
    for tag, xs in (("f", xb + 0.01), ("r", xa - 0.01)):
        ob = G.box(f"{PREFIX}_carriage_buffer_{tag}", (0.02, 0.80, 0.16), (xs, 0, 0.42), col, bevel=0.004)
        ob.data.materials.append(M["yellow"]); add(ob)
    for k, (gx, gy) in enumerate(((xa + 0.16, RAIL_Y), (xb - 0.16, RAIL_Y), (xa + 0.16, -RAIL_Y), (xb - 0.16, -RAIL_Y))):
        ob = G.box(f"{PREFIX}_guide_block{k}", (0.15, 0.10, 0.05), (gx, gy, 0.29), col, bevel=0.004)
        ob.data.materials.append(M["steel"]); add(ob)
    # rack-and-pinion drive on the -Y side: gearbox hanging beside the body, servo motor on top
    ob = G.box(f"{PREFIX}_drive_gearbox", (0.18, 0.12, 0.20), (0.12, -CARRIAGE_HW - 0.06, 0.40), col, bevel=0.008)
    ob.data.materials.append(M["blue"]); add(ob)
    ob = G.cylinder(f"{PREFIX}_drive_motor", 0.062, 0.22, (0.12, -CARRIAGE_HW - 0.06, 0.61), vertices=32, collection=col)
    ob.data.materials.append(M["motor"]); add(ob)
    ob = G.cylinder(f"{PREFIX}_drive_motor_cap", 0.066, 0.03, (0.12, -CARRIAGE_HW - 0.06, 0.735), vertices=32, collection=col)
    ob.data.materials.append(M["dark"]); add(ob)
    ob = G.box(f"{PREFIX}_drive_motor_plug", (0.05, 0.04, 0.05), (0.12 + 0.07, -CARRIAGE_HW - 0.06, 0.66), col, bevel=0.003)
    ob.data.materials.append(M["dark"]); add(ob)
    ob = G.box(f"{PREFIX}_carriage_label", (0.16, 0.003, 0.08), (xm - 0.2, -CARRIAGE_HW - 0.0015, 0.44), col)
    ob.data.materials.append(M["label"]); add(ob)
    # cable from the chain bracket to the robot base connector (static on the carriage)
    cu = bpy.data.curves.new(f"{PREFIX}_base_cable", 'CURVE')
    cu.dimensions = '3D'
    cu.bevel_depth = 0.02
    cu.bevel_resolution = 4
    cu.resolution_u = 10
    sp = cu.splines.new('BEZIER')
    xr = -0.401 * L2.HANDLER_SCALE                  # rear face of the robot base (connector housing)
    pts = [(CHAIN_OFF + 0.06, TROUGH_Y + 0.04, CHAIN_Z0 + 2 * CHAIN_R + CHAIN_H / 2 + 0.02), (CHAIN_OFF - 0.08, TROUGH_Y - 0.02, 0.46),
           (xr + 0.02, 0.40, top + 0.09), (xr - 0.07, 0.12, top + 0.13), (xr + 0.005, 0.0, top + 0.13)]
    sp.bezier_points.add(len(pts) - 1)
    for p, xyz in zip(sp.bezier_points, pts):
        p.co = xyz
        p.handle_left_type = p.handle_right_type = 'AUTO'
    cab = bpy.data.objects.new(f"{PREFIX}_base_cable", cu)
    col.objects.link(cab)
    cab.data.materials.append(M["cable"])
    add(cab)
    return car, parts


# ------------------------------------------------------------------ dress pack
def _hose_group(radius):
    """Geometry nodes: resample the hooked curve, store arc length 'hose_u' (corrugation), sweep a circle."""
    name = f"{PREFIX}_hose_tube"
    ng = bpy.data.node_groups.get(name)
    if ng is not None:
        return ng
    ng = bpy.data.node_groups.new(name, 'GeometryNodeTree')
    ng.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    n = ng.nodes
    gin, gout = n.new("NodeGroupInput"), n.new("NodeGroupOutput")
    res = n.new("GeometryNodeResampleCurve"); res.mode = 'LENGTH'; res.inputs["Length"].default_value = 0.01
    par = n.new("GeometryNodeSplineParameter")
    st = n.new("GeometryNodeStoreNamedAttribute"); st.data_type = 'FLOAT'; st.domain = 'POINT'; st.inputs["Name"].default_value = "hose_u"
    circ = n.new("GeometryNodeCurvePrimitiveCircle"); circ.inputs["Resolution"].default_value = 16; circ.inputs["Radius"].default_value = radius
    c2m = n.new("GeometryNodeCurveToMesh"); c2m.inputs["Fill Caps"].default_value = True
    sm = n.new("GeometryNodeSetShadeSmooth")
    setm = n.new("GeometryNodeSetMaterial"); setm.inputs["Material"].default_value = RB._hose_material()
    lk = ng.links.new
    lk(gin.outputs["Geometry"], res.inputs["Curve"]); lk(res.outputs["Curve"], st.inputs["Geometry"])
    lk(par.outputs["Length"], st.inputs["Value"]); lk(st.outputs["Geometry"], c2m.inputs["Curve"])
    lk(circ.outputs["Curve"], c2m.inputs["Profile Curve"]); lk(c2m.outputs["Mesh"], sm.inputs["Geometry"])
    lk(sm.outputs["Geometry"], setm.inputs["Geometry"]); lk(setm.outputs["Geometry"], gout.inputs["Geometry"])
    return ng


def _build_dress_pack(col, arm, gripper, M):
    """Hose bundle hooked to anchors: link_1 top -> back of the lower arm -> behind / over the upper-arm housing ->
    two forearm clamps -> over the wrist -> gripper connector.  Anchor offsets are in the unscaled IRB 4600 link
    frames (x scale), tangents in the same frames; the hooks carry points + handles rigidly with their link.  The last
    anchor is the connector on the gripper's swivel stator, fixed to J5 (see _mount_swivel): the bundle never winds with J6."""
    s = arm["scale"]
    J = arm["joints"]
    r = HOSE_R / s                      # hose radius in unscaled units
    anchors = [
        # parent, offset (unscaled), tangent, handle lengths (in, out; unscaled), clamp support direction or None.
        # The long out-handle of the upper lower-arm clamp / in-handle on the upper-arm housing keep the bundle off the
        # elbow housing when J3 bends; the long handles at the forearm end and over the wrist keep it off the wrist
        # housing and the flange when J5 bends past 90 deg (checked along the whole plan trajectory).
        (J[0], (-0.17, -0.05, 0.60), (0.0, -0.35, 1.0), (0.08, 0.08), None),
        (J[1], (-0.12 - 0.035 - r, -0.19, 0.32), (0.0, 0.0, 1.0), (0.14, 0.14), (1.0, 0.0, 0.0)),
        (J[1], (-0.11 - 0.035 - r, -0.19, 0.86), (0.0, 0.0, 1.0), (0.14, 0.17), (1.0, 0.0, 0.0)),
        (J[2], (-0.18 - 0.03 - r, -0.06, 0.24), (0.35, 0.0, 1.0), (0.13, 0.10), (1.0, 0.0, 0.0)),
        (J[2], (0.02, 0.0, 0.332 + 0.03 + r), (1.0, 0.0, 0.0), (0.10, 0.10), (0.0, 0.0, -1.0)),
        (J[3], (0.35, 0.0, 0.055 + 0.04 + r), (1.0, 0.0, 0.0), (0.12, 0.12), (0.0, 0.0, -1.0)),
        (J[3], (0.75, 0.0, 0.073 + 0.04 + r), (1.0, 0.0, 0.0), (0.10, 0.13), (0.0, 0.0, -1.0)),
        (J[4], (0.0, 0.0, 0.13), (1.0, 0.0, 0.0), (0.13, 0.06), None),
        (gripper["connector"], None, (1.0, 0.0, 0.0), (0.05, 0.05), None),
    ]
    cu = bpy.data.curves.new(f"{PREFIX}_hose_curve", 'CURVE')
    cu.dimensions = '3D'
    cu.resolution_u = 12
    sp = cu.splines.new('BEZIER')
    ob = bpy.data.objects.new(f"{PREFIX}_hose", cu)
    col.objects.link(ob)
    sp.bezier_points.add(len(anchors) - 1)
    empties = []
    for i, (par, off, tan, hl, sup) in enumerate(anchors):
        if off is None:
            empties.append(par)
            continue
        e = G.empty(f"{PREFIX}_hose_anchor{i}", collection=col, size=0.04)
        _local(e, par, mathutils.Matrix.Translation([v * s for v in off]))
        empties.append(e)
    bpy.context.view_layer.update()
    for i, (e, (par, off, tan, hl, sup)) in enumerate(zip(empties, anchors)):
        p = sp.bezier_points[i]
        co = e.matrix_world.translation
        t = (par.matrix_world.to_3x3() @ mathutils.Vector(tan)).normalized()
        p.handle_left_type = p.handle_right_type = 'FREE'
        p.co = co
        p.handle_left = co - t * hl[0] * s
        p.handle_right = co + t * hl[1] * s
    for i, e in enumerate(empties):
        m = ob.modifiers.new(f"hook{i}", 'HOOK')
        m.object = e
        m.vertex_indices_set([3 * i, 3 * i + 1, 3 * i + 2])
        m.matrix_inverse = e.matrix_world.inverted()
        m.center = e.matrix_world.translation
    gm = ob.modifiers.new("tube", 'NODES')
    gm.node_group = _hose_group(HOSE_R)
    # clamps: a block between the link surface and the hose + a strap ring around the hose
    clamps = []
    for i, (par, off, tan, hl, sup) in enumerate(anchors):
        if sup is None or off is None:
            continue
        t = mathutils.Vector(tan).normalized()
        d = mathutils.Vector(sup).normalized()
        side = t.cross(d)
        R3 = mathutils.Matrix((t, side, d)).transposed()          # columns: tangent, side, support
        reach = 0.045 * s + HOSE_R          # hose centre -> link surface (anchors use 30..40 mm gaps, unscaled)
        c = mathutils.Vector(off) * s
        blk = G.box(f"{PREFIX}_hose_clamp{i}", (0.05 * s, 0.045 * s, reach), (0, 0, 0), col, bevel=0.003)
        blk.data.materials.append(M["dark"])
        _local(blk, par, mathutils.Matrix.Translation(c + d * (reach / 2)) @ R3.to_4x4())
        strap = G.revolve(f"{PREFIX}_hose_strap{i}", [(HOSE_R + 0.001, -0.014), (HOSE_R + 0.005, -0.014), (HOSE_R + 0.005, 0.014), (HOSE_R + 0.001, 0.014)],
                          segments=24, axis='X', collection=col)
        strap.data.materials.append(M["dark"])
        _local(strap, par, mathutils.Matrix.Translation(c) @ R3.to_4x4())
        clamps += [blk, strap]
    # hose connector box on top of link_1 where the bundle leaves the base
    cb = G.box(f"{PREFIX}_hose_base_box", (0.10 * s, 0.10 * s, 0.05 * s), (0, 0, 0), col, bevel=0.004)
    cb.data.materials.append(M["dark"])
    _local(cb, J[0], mathutils.Matrix.Translation(mathutils.Vector((-0.17, -0.05, 0.565 + 0.025)) * s))
    clamps.append(cb)
    return dict(curve=ob, anchors=empties, clamps=clamps)


# ------------------------------------------------------------------ swivel mount
# anti-rotation bracket of the gripper swivel, in tool0 coordinates at q6 = 0 (link_5 dimensions in unscaled URDF units):
# a foot bolted to the -X face of the wrist housing (link_5) and a bar down to the stator's connector box.  The bar stays
# outside the adapter plate (r 0.10) that turns with J6, and inside the dress-pack hose that runs further out on -X.
SWIVEL_FOOT = dict(z=-0.117, x_in=-0.062, h=0.022, w=0.032)     # foot centre z and inner face x (unscaled), height, width
SWIVEL_BAR = dict(x=(-0.115, -0.105), w=0.030)                  # bar x range (tool frame, m) and width along y


def _mount_swivel(col, arm, grip, M):
    """Hold the gripper's swivel stator (swivel body, torque lug, dress-pack connector) on the wrist: re-parent it to J5
    (the parent of J6) and add the anti-rotation bracket, so J6 can turn the gripper half a turn (the planner uses
    q6 up to 180 deg) without winding the hose through the flange."""
    s = arm["scale"]
    tool0, j5 = arm["tool0"], arm["joints"][4]
    zf, xin = SWIVEL_FOOT["z"] * s, SWIVEL_FOOT["x_in"] * s
    xo, xi = SWIVEL_BAR["x"]
    zb0 = zf - SWIVEL_FOOT["h"] / 2
    zb1 = GR.CONN_Z - 0.021                      # top face of the connector box (tool -z side)
    out = []

    def add(ob, mat, local):
        ob.data.materials.append(mat)
        _local(ob, tool0, local)
        out.append(ob)
        return ob
    T = mathutils.Matrix.Translation
    add(G.box(f"{PREFIX}_swivel_foot", (xin - xo, SWIVEL_FOOT["w"], SWIVEL_FOOT["h"]), (0, 0, 0), col, bevel=0.002), M["frame"],
        T(((xin + xo) / 2, 0.0, zf)))
    add(G.box(f"{PREFIX}_swivel_bar", (xi - xo, SWIVEL_BAR["w"], zb1 - zb0), (0, 0, 0), col, bevel=0.002), M["frame"],
        T(((xo + xi) / 2, 0.0, (zb0 + zb1) / 2)))
    rx = mathutils.Matrix.Rotation(math.pi / 2, 4, 'Y')
    for k, (yy, zz) in enumerate(((-0.008, zf), (0.008, zf), (0.0, zb1 - 0.014))):
        add(G.cylinder(f"{PREFIX}_swivel_bolt{k}", 0.0055, 0.005, (0, 0, 0), (0, 0, 0), vertices=6, collection=col), M["dark"],
            T((xo - 0.0025, yy, zz)) @ rx)
    bpy.context.view_layer.update()
    for ob in list(grip["stator"]) + out:
        G.set_parent(ob, j5, keep_world=True)
    return out


# ------------------------------------------------------------------ build
def build(collection=None):
    col = collection or bpy.data.collections.new(COLLECTION)
    if collection is None:
        bpy.context.scene.collection.children.link(col)
    M = _mats()
    track = _build_track(col, M)
    x = L2.HANDLER_HOME_X
    car, cparts = _build_carriage(col, M, x)
    bpy.context.view_layer.update()
    arm = build_arm(PREFIX, L2.HANDLER_SCALE, base_world(x), col)
    G.set_parent(arm["base"], car)
    chain = _build_chain(col, car, M)
    grip = GR.build(arm["tool0"], col, name="grp")
    swivel = _mount_swivel(col, arm, grip, M)       # arm still at q = 0 here (build_arm), so q6 = 0
    hose = _build_dress_pack(col, arm, grip, M)
    rob = dict(collection=col, arm=arm["arm"], carriage=car, base=arm["base"], joints=arm["joints"], tool0=arm["tool0"],
               links=arm["links"], gripper=grip, chain_bend=chain["bend"], chain=chain, hose=hose, track=track,
               carriage_parts=cparts, swivel=swivel, scale=L2.HANDLER_SCALE)
    set_track(rob, x)
    set_q(rob, L2.HANDLER_Q_HOME)
    GR.set_open(grip, 1.0)
    bpy.context.view_layer.update()
    return rob


# ------------------------------------------------------------------ static obstacles (pure python)
def obstacles():
    """Conservative world AABBs of the static track geometry (bed incl. feet/rails/rack, end stops, trough + chain)."""
    x0, x1 = L2.HANDLER_TRACK_BED
    y = L2.HANDLER_TRACK_Y
    bx0, bx1 = BED_X
    out = [
        dict(name=f"{PREFIX}_bed", center=((bx0 + bx1) / 2, y, 0.16), size=(bx1 - bx0, 2 * (BED_HW + 0.08), 0.32), yaw=0.0),
        dict(name=f"{PREFIX}_stop_n", center=(x0 + 0.04, y, 0.23), size=(0.08, 0.92, 0.46), yaw=0.0),
        dict(name=f"{PREFIX}_stop_p", center=(x1 - 0.04, y, 0.23), size=(0.08, 0.92, 0.46), yaw=0.0),
        dict(name=f"{PREFIX}_trough", center=((bx0 + bx1) / 2, y + TROUGH_Y + 0.005, 0.11),
             size=(bx1 - bx0, TROUGH_W + 0.02, 0.22), yaw=0.0),          # incl. the floor-duct plate at y 0.742
    ]
    return out
