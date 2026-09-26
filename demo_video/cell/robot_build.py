"""ABB IRB 4600 assembly in Blender: linear track, robot links as a parented empty chain, MIG torch, hose package.

Joint empties J1..J6: world = parent_world @ joint_origin @ Rot(axis, q).  We store the constant joint origin in
`matrix_parent_inverse` and animate `rotation_axis_angle` (axis fixed = URDF joint axis, angle = q).
"""
import math
import os
import bpy
import numpy as np
import mathutils
from . import layout as L
from . import geom as G
from . import materials
from .robot_urdf import URDFArm

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF = os.path.join(HERE, "assets", "abb_irb4600_40_255", "urdf", "robot_description.urdf")
MESH_ROOT = os.path.join(HERE, "assets", "abb_irb4600_40_255")
HOSE_WAVE_SCALE = 35.0      # wave-texture scale on the arc length in metres: band pitch = 0.314 / scale ~ 9 mm
HOSE_RADIUS = 0.016
_FOREARM_TOP = {0.35: 0.055, 0.75: 0.073}   # forearm (link_4) top surface z at y = 0 for the two hose clamps, from the STL


def tool_transform():
    """TCP pose in the tool0 frame (numpy 4x4): bracket + neck along +Z, 45° bend toward +X, head + stickout."""
    from .robot_kr5 import tr, rx
    def ry(t):
        c, s = math.cos(t), math.sin(t)
        return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1.0]])
    T = tr(z=L.TORCH_BRACKET_LEN + L.TORCH_NECK_LEN) @ ry(L.TORCH_BEND_ANGLE) @ tr(z=L.TORCH_HEAD_LEN + L.TORCH_STICKOUT)
    return T


def load_arm():
    return URDFArm(URDF, MESH_ROOT)


def base_matrix(track_y):
    """World pose of the robot base for a given track position."""
    c, s = math.cos(L.ROBOT_YAW), math.sin(L.ROBOT_YAW)
    return np.array([[c, -s, 0, L.TRACK_X], [s, c, 0, track_y], [0, 0, 1, L.TRACK_TOP_Z], [0, 0, 0, 1.0]])


def build(collection=None):
    col = collection or bpy.data.collections.new("Robot")
    if collection is None:
        bpy.context.scene.collection.children.link(col)
    arm = load_arm()
    orange = materials.get("abb_orange")
    grey = materials.get("abb_grey")
    dark = materials.get("dark_metal")

    # ---------------- linear track (7th axis) along Y
    ymin, ymax = L.TRACK_Y_MIN, L.TRACK_Y_MAX
    length = ymax - ymin + 1.2
    ycen = (ymin + ymax) / 2
    bed = G.box("track_bed", (0.9, length, 0.22), (L.TRACK_X, ycen, 0.11), col, bevel=0.01)
    bed.data.materials.append(materials.get("painted", color="#2B2F36", roughness=0.5))
    for sx in (-0.3, 0.3):
        rail = G.box(f"track_rail{'L' if sx < 0 else 'R'}", (0.06, length - 0.1, 0.05), (L.TRACK_X + sx, ycen, 0.245), col)
        rail.data.materials.append(materials.get("machined_steel"))
    rack = G.box("track_rack", (0.04, length - 0.2, 0.04), (L.TRACK_X - 0.42, ycen, 0.24), col)
    rack.data.materials.append(dark)
    # cable chain trough
    trough = G.box("track_trough", (0.22, length, 0.16), (L.TRACK_X + 0.62, ycen, 0.08), col)
    trough.data.materials.append(materials.get("galvanized"))
    for k in range(int(length / 0.12)):
        y = ymin - 0.5 + k * 0.12
        seg = G.box(f"track_chain{k}", (0.16, 0.1, 0.09), (L.TRACK_X + 0.62, y, 0.14), col)
        seg.data.materials.append(materials.get("black_plastic"))
    # end stops / bumpers
    for y in (ymin - 0.55, ymax + 0.55):
        st = G.box(f"track_stop{y:+.0f}", (0.8, 0.06, 0.32), (L.TRACK_X, y, 0.16), col)
        st.data.materials.append(materials.get("safety_yellow"))

    # carriage
    carriage = G.empty("track_carriage", collection=col, size=0.4)
    carriage.location = (L.TRACK_X, L.ROBOT_HOME_Y, 0.0)
    cbody = G.box("carriage_body", (0.86, 1.05, L.TRACK_TOP_Z - 0.27), (L.TRACK_X, L.ROBOT_HOME_Y, 0.27 + (L.TRACK_TOP_Z - 0.27) / 2), col, bevel=0.01)
    cbody.data.materials.append(materials.get("painted", color="#2B2F36", roughness=0.5))
    cplate = G.box("carriage_plate", (0.78, 0.78, 0.03), (L.TRACK_X, L.ROBOT_HOME_Y, L.TRACK_TOP_Z - 0.015), col)
    cplate.data.materials.append(materials.get("machined_steel"))
    cmot = G.cylinder("carriage_motor", 0.07, 0.25, (L.TRACK_X - 0.5, L.ROBOT_HOME_Y + 0.3, 0.36), (0, math.pi / 2, 0), collection=col)
    cmot.data.materials.append(dark)
    for o in (cbody, cplate, cmot):
        G.set_parent(o, carriage)

    # ---------------- robot base + joint chain
    base = G.empty("robot_base", collection=col, size=0.3)
    base.matrix_world = G.M(base_matrix(L.ROBOT_HOME_Y))
    G.set_parent(base, carriage)
    joints = []
    parent = base
    link_objs = {}
    # base link mesh
    info = arm.links[arm.base_link]
    bl = _load_mesh("abb_base_link", info["mesh"], col)
    bl.data.materials.append(grey)
    bl.matrix_world = base.matrix_world @ G.M(info["vis_origin"])
    G.set_parent(bl, base)
    link_objs[arm.base_link] = bl
    colours = {"link_1": orange, "link_2": orange, "link_3": orange, "link_4": orange, "link_5": orange, "link_6": grey}
    for j in arm.chain:
        if j["type"] != "revolute":
            # fixed joint (tool0)
            e = G.empty(f"robot_{j['child']}", collection=col, size=0.15)
            e.parent = parent
            e.matrix_parent_inverse = mathutils.Matrix.Identity(4)
            e.matrix_basis = G.M(j["origin"])
            parent = e
            joints.append(e)
            continue
        e = G.empty(f"robot_J{len(joints) + 1}", collection=col, size=0.2)
        e.parent = parent
        e.matrix_parent_inverse = G.M(j["origin"])
        e.rotation_mode = 'AXIS_ANGLE'
        e.rotation_axis_angle = (0.0, *j["axis"])
        joints.append(e)
        info = arm.links[j["child"]]
        if info["mesh"]:
            lo = _load_mesh(f"abb_{j['child']}", info["mesh"], col)
            lo.data.materials.append(colours.get(j["child"], orange))
            lo.parent = e
            lo.matrix_parent_inverse = mathutils.Matrix.Identity(4)
            lo.matrix_basis = G.M(info["vis_origin"])
            link_objs[j["child"]] = lo
        parent = e
    # small grey nameplate on the outer face of the lower arm (link_2)
    np_ = G.box("robot_nameplate", (0.13, 0.004, 0.055), (0, 0, 0), col)
    np_.data.materials.append(materials.get("painted", color="#C9CCD0", roughness=0.35, coat=0.0))
    np_.parent = joints[1]; np_.matrix_parent_inverse = mathutils.Matrix.Identity(4)
    np_.matrix_basis = mathutils.Matrix.Translation((0.0, -0.2545, 0.55)) @ mathutils.Matrix.Rotation(0.02, 4, 'X')   # side face y=-0.252 at z=0.55, tapering 0.02 rad
    tool0 = parent  # fixed tool0 empty
    tool0.name = "robot_tool0"
    # ---------------- torch
    torch = _build_torch(col, tool0)
    tcp = G.empty("robot_tcp", collection=col, size=0.08, display='ARROWS')
    tcp.parent = tool0
    tcp.matrix_parent_inverse = mathutils.Matrix.Identity(4)
    tcp.matrix_basis = G.M(tool_transform())
    # wire feeder on link_3 (upper arm) and hose package
    feeder = _build_feeder(col, joints[2])
    hose = _build_hose(col, joints, torch, feeder)
    bpy.context.view_layer.update()
    return dict(collection=col, arm=arm, carriage=carriage, base=base, joints=joints[:6], tool0=tool0, tcp=tcp,
                torch=torch, links=link_objs, hose=hose)


def _load_mesh(name, path, col):
    bpy.ops.wm.stl_import(filepath=path)
    o = bpy.context.object
    o.name = name
    if max(o.dimensions) > 20:      # mm file
        o.scale = (0.001,) * 3
        bpy.ops.object.transform_apply(scale=True)
    bpy.ops.object.shade_smooth_by_angle(angle=math.radians(30))
    for c in list(o.users_collection):
        c.objects.unlink(o)
    col.objects.link(o)
    return o


def _build_torch(col, tool0):
    """Torch geometry in the tool0 frame (+Z out of the flange): anti-collision bracket, straight neck, smooth swept
    45° elbow (torus segment), head with insulator ring + copper gas nozzle, and a laser seam-tracking sensor whose
    lens window looks at the TCP, bracketed to a clamp ring on the neck."""
    dark = materials.get("dark_metal")
    black = materials.get("black_plastic")
    brass = materials.get("brass")
    copper = materials.get("copper")
    root = G.empty("torch_root", collection=col, size=0.05)
    root.parent = tool0
    root.matrix_parent_inverse = mathutils.Matrix.Identity(4)
    parts = []
    # anti-collision bracket: flange disc + cylinder + shock sensor ring
    p = G.cylinder("torch_flange", 0.06, 0.015, (0, 0, 0.0075), collection=col); p.data.materials.append(materials.get("machined_steel")); parts.append(p)
    p = G.cylinder("torch_bracket", 0.045, L.TORCH_BRACKET_LEN - 0.02, (0, 0, 0.015 + (L.TORCH_BRACKET_LEN - 0.02) / 2), collection=col); p.data.materials.append(black); parts.append(p)
    p = G.cylinder("torch_sensor_ring", 0.05, 0.03, (0, 0, L.TORCH_BRACKET_LEN - 0.005), collection=col); p.data.materials.append(materials.get("safety_yellow")); parts.append(p)
    # straight neck with clamp, then a smooth elbow (bend radius RB) tangent to both the neck and the head
    z0 = L.TORCH_BRACKET_LEN
    zb = z0 + L.TORCH_NECK_LEN
    a = L.TORCH_BEND_ANGLE
    RB = 0.06
    tl = RB * math.tan(a / 2)          # tangent length from the bend vertex on each leg
    rn = 0.019
    p = G.cylinder("torch_neck_clamp", 0.03, 0.06, (0, 0, z0 + 0.03), collection=col); p.data.materials.append(dark); parts.append(p)
    p = G.cylinder("torch_neck", rn, L.TORCH_NECK_LEN - tl + 0.002, (0, 0, z0 + (L.TORCH_NECK_LEN - tl + 0.002) / 2), collection=col); p.data.materials.append(dark); parts.append(p)
    elbow = G.torus_sweep("torch_elbow", RB, a, lambda t: [rn], seg_bend=16, seg_tube=32, collection=col)
    elbow.location = (0, 0, zb - tl)
    elbow.data.materials.append(dark); parts.append(elbow)
    d = mathutils.Vector((math.sin(a), 0, math.cos(a)))
    hl = L.TORCH_HEAD_LEN
    def along(t):
        return mathutils.Vector((0, 0, zb)) + d * t
    rot = (0, a, 0)
    h0 = tl - 0.002
    p = G.cylinder("torch_head", 0.017, hl * 0.6 - h0, along((h0 + hl * 0.6) / 2), rot, collection=col); p.data.materials.append(dark); parts.append(p)
    p = G.cylinder("torch_insulator", L.TORCH_NOZZLE_R + 0.0065, 0.018, along(hl * 0.58 + 0.009), rot, vertices=32, collection=col); p.data.materials.append(black); parts.append(p)
    p = G.cylinder("torch_gas_nozzle", L.TORCH_NOZZLE_R + 0.004, hl * 0.42, along(hl * 0.79), rot, vertices=32, collection=col); p.data.materials.append(copper); parts.append(p)
    p = G.cylinder("torch_contact_tip", 0.0045, 0.03, along(hl - 0.01), rot, vertices=16, collection=col); p.data.materials.append(brass); parts.append(p)
    p = G.cylinder("torch_wire", 0.0008, L.TORCH_STICKOUT + 0.02, along(hl + L.TORCH_STICKOUT / 2 - 0.005), rot, vertices=8, collection=col); p.data.materials.append(brass); parts.append(p)
    # seam-tracking laser sensor: body axis aimed at the TCP, lens window on the end face, bracket + clamp ring on the neck
    tcp = mathutils.Vector(tool_transform()[:3, 3].tolist())
    sc = mathutils.Vector((0.052, 0.0, zb - 0.05))
    dv = (tcp - sc).normalized()
    theta = math.atan2(dv.x, dv.z)
    Rs = mathutils.Matrix.Rotation(theta, 4, 'Y')
    s = G.box("torch_laser_sensor", (0.05, 0.035, 0.09), (0, 0, 0), col, bevel=0.004)
    s.data.materials.append(materials.get("painted", color="#1E2A44", roughness=0.35)); parts.append(s)
    lens = G.box("torch_laser_lens", (0.02, 0.025, 0.004), (0, 0, 0), col)
    lens.data.materials.append(materials.get("glass_dark")); parts.append(lens)
    label = G.box("torch_laser_label", (0.03, 0.0015, 0.05), (0, 0, 0), col)
    label.data.materials.append(materials.get("painted", color="#D9DBDD", roughness=0.4, coat=0.0)); parts.append(label)
    placed = [(s, mathutils.Matrix.Identity(4)), (lens, mathutils.Matrix.Translation((0, 0, 0.045))),
              (label, mathutils.Matrix.Translation((0, 0.018, 0.0)))]
    ring = G.cylinder("torch_sensor_clamp", 0.024, 0.018, (0, 0, zb - 0.05), collection=col); ring.data.materials.append(dark); parts.append(ring)
    br = G.box("torch_sensor_bracket", (0.05, 0.016, 0.012), (0.026, 0, zb - 0.05), col); br.data.materials.append(dark); parts.append(br)
    sensor = G.empty("torch_sensor", collection=col, size=0.03)
    sensor.matrix_world = G.M(tool_transform()) @ mathutils.Matrix.Translation((0.0, 0.0, -0.045))
    sensor.parent = root; sensor.matrix_parent_inverse = mathutils.Matrix.Identity(4)
    for p in parts:
        p.parent = root
        p.matrix_parent_inverse = mathutils.Matrix.Identity(4)
    for p, m in placed:
        p.matrix_basis = mathutils.Matrix.Translation(sc) @ Rs @ m
    bpy.context.view_layer.update()
    return dict(root=root, parts=parts, sensor=sensor)


def _build_feeder(col, j3):
    """Wire feeder box mounted on top of link_3 (upper arm)."""
    f = G.box("wire_feeder", (0.34, 0.2, 0.22), (0, 0, 0), col, bevel=0.01)
    f.data.materials.append(materials.get("painted", color="#B3B3B3", roughness=0.4))
    f.parent = j3
    f.matrix_parent_inverse = mathutils.Matrix.Identity(4)
    f.matrix_basis = mathutils.Matrix.Translation((0.25, 0.0, 0.33))
    lid = G.box("wire_feeder_lid", (0.3, 0.16, 0.02), (0, 0, 0), col)
    lid.data.materials.append(materials.get("safety_red"))
    lid.parent = f; lid.matrix_parent_inverse = mathutils.Matrix.Identity(4); lid.matrix_basis = mathutils.Matrix.Translation((0, 0, 0.12))
    out = G.empty("feeder_outlet", collection=col, size=0.05)
    out.parent = f; out.matrix_parent_inverse = mathutils.Matrix.Identity(4); out.matrix_basis = mathutils.Matrix.Translation((0.18, 0.0, 0.05))
    return dict(box=f, outlet=out)


def _hose_material():
    """Black corrugated hose: cable_black look plus a wave bump along the hose (arc length attribute "hose_u", metres,
    written by the geometry-nodes modifier of the hose)."""
    m = bpy.data.materials.get("hose_corrugated")
    if m is not None:
        return m
    m = bpy.data.materials.new("hose_corrugated")
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.02, 0.02, 0.022, 1.0)
    b.inputs["Roughness"].default_value = 0.6
    b.inputs["Specular IOR Level"].default_value = 0.35
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "hose_u"
    xyz = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(attr.outputs["Fac"], xyz.inputs["X"])
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.wave_type = 'BANDS'
    wave.bands_direction = 'X'
    wave.wave_profile = 'SIN'
    wave.inputs["Scale"].default_value = HOSE_WAVE_SCALE
    wave.inputs["Distortion"].default_value = 0.0
    wave.inputs["Detail"].default_value = 0.0
    nt.links.new(xyz.outputs["Vector"], wave.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.4
    bump.inputs["Distance"].default_value = 0.002
    nt.links.new(wave.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    return m


def _hose_tube(ob, mat):
    """Geometry-nodes modifier: resample the (hooked) curve by length, store the arc length as "hose_u", sweep a circle
    profile of HOSE_RADIUS along it and assign the hose material (replaces the curve bevel, whose UV u is not
    proportional to length so the corrugation pitch would change from segment to segment)."""
    ng = bpy.data.node_groups.get("hose_tube")
    if ng is None:
        ng = bpy.data.node_groups.new("hose_tube", 'GeometryNodeTree')
        ng.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
        ng.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
        n = ng.nodes
        gin, gout = n.new("NodeGroupInput"), n.new("NodeGroupOutput")
        res = n.new("GeometryNodeResampleCurve"); res.mode = 'LENGTH'; res.inputs["Length"].default_value = 0.008
        par = n.new("GeometryNodeSplineParameter")
        st = n.new("GeometryNodeStoreNamedAttribute"); st.data_type = 'FLOAT'; st.domain = 'POINT'; st.inputs["Name"].default_value = "hose_u"
        circ = n.new("GeometryNodeCurvePrimitiveCircle"); circ.inputs["Resolution"].default_value = 16; circ.inputs["Radius"].default_value = HOSE_RADIUS
        c2m = n.new("GeometryNodeCurveToMesh"); c2m.inputs["Fill Caps"].default_value = True
        sm = n.new("GeometryNodeSetShadeSmooth")
        setm = n.new("GeometryNodeSetMaterial"); setm.inputs["Material"].default_value = mat
        lk = ng.links.new
        lk(gin.outputs["Geometry"], res.inputs["Curve"]); lk(res.outputs["Curve"], st.inputs["Geometry"])
        lk(par.outputs["Length"], st.inputs["Value"]); lk(st.outputs["Geometry"], c2m.inputs["Curve"])
        lk(circ.outputs["Curve"], c2m.inputs["Profile Curve"]); lk(c2m.outputs["Mesh"], sm.inputs["Geometry"])
        lk(sm.outputs["Geometry"], setm.inputs["Geometry"]); lk(setm.outputs["Geometry"], gout.inputs["Geometry"])
    m = ob.modifiers.new("tube", 'NODES')
    m.node_group = ng
    return m


def _build_hose(col, joints, torch, feeder):
    """Hose package (torch cable) as a bezier curve hooked to anchors on the arm and the torch.

    Anchors: feeder outlet -> two clamps on top of the forearm (link_4) -> over the wrist pitch axis (link_5) -> torch
    root.  Each control point's handles are set along the local arm direction of its anchor, and the hook modifiers
    carry point + handles rigidly with the anchor, so the hose keeps resting on the arm for any joint angles.
    """
    dark = materials.get("dark_metal")
    cu = bpy.data.curves.new("hose_curve", 'CURVE')
    cu.dimensions = '3D'
    cu.resolution_u = 12
    sp = cu.splines.new('BEZIER')
    ob = bpy.data.objects.new("hose", cu)
    col.objects.link(ob)
    # (parent, local offset, local tangent direction, handle length)
    anchors = [
        (feeder["outlet"], (0, 0, 0), (1.0, 0, -0.35), 0.09),
        (joints[3], (0.35, 0.0, _FOREARM_TOP[0.35] + 0.04 + HOSE_RADIUS * 0.75), (1.0, 0, 0), 0.12),
        (joints[3], (0.75, 0.0, _FOREARM_TOP[0.75] + 0.04 + HOSE_RADIUS * 0.75), (1.0, 0, 0), 0.10),
        (joints[4], (0.0, 0.0, 0.13), (1.0, 0, 0), 0.07),
        (torch["root"], (-0.052, 0.0, 0.055), (0.45, 0, 1.0), 0.06),    # dives into the side of the anti-collision bracket
    ]
    sp.bezier_points.add(len(anchors) - 1)
    empties = []
    for i, (par, off, tan, hl) in enumerate(anchors):
        e = G.empty(f"hose_anchor{i}", collection=col, size=0.04)
        e.parent = par
        e.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        e.matrix_basis = mathutils.Matrix.Translation(off)
        empties.append(e)
    bpy.context.view_layer.update()
    for i, (e, (par, off, tan, hl)) in enumerate(zip(empties, anchors)):
        p = sp.bezier_points[i]
        co = e.matrix_world.translation
        t = (par.matrix_world.to_3x3() @ mathutils.Vector(tan)).normalized()
        p.handle_left_type = p.handle_right_type = 'FREE'
        p.co = co
        p.handle_left = co - t * hl
        p.handle_right = co + t * hl
    for i, e in enumerate(empties):
        m = ob.modifiers.new(f"hook{i}", 'HOOK')
        m.object = e
        m.vertex_indices_set([3 * i, 3 * i + 1, 3 * i + 2])
        m.matrix_inverse = e.matrix_world.inverted()
        m.center = e.matrix_world.translation
    _hose_tube(ob, _hose_material())        # after the hooks: hooks move the control points, the tube is built from them
    # clamp blocks on the forearm (hose rests on them) with a strap ring around the hose
    clamps = []
    for k, x in enumerate((0.35, 0.75)):
        zt = _FOREARM_TOP[x]
        c = G.box(f"hose_clamp{k}", (0.05, 0.05, 0.04), (0, 0, 0), col, bevel=0.003); c.data.materials.append(dark)
        c.parent = joints[3]; c.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        c.matrix_basis = mathutils.Matrix.Translation((x, 0.0, zt + 0.02))
        r = HOSE_RADIUS
        strap = G.revolve(f"hose_strap{k}", [(r + 0.001, -0.012), (r + 0.004, -0.012), (r + 0.004, 0.012), (r + 0.001, 0.012)], segments=24, axis='X', collection=col)
        strap.data.materials.append(dark)
        strap.parent = joints[3]; strap.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        strap.matrix_basis = mathutils.Matrix.Translation((x, 0.0, zt + 0.04 + r * 0.75))
        clamps += [c, strap]
    return dict(curve=ob, anchors=empties, clamps=clamps)


def set_q(robot, q, frame=None):
    for j, e in enumerate(robot["joints"]):
        aa = e.rotation_axis_angle
        e.rotation_axis_angle = (float(q[j]), aa[1], aa[2], aa[3])
        if frame is not None:
            e.keyframe_insert("rotation_axis_angle", index=0, frame=frame)


def set_track(robot, y, frame=None):
    robot["carriage"].location[1] = y
    if frame is not None:
        robot["carriage"].keyframe_insert("location", index=1, frame=frame)
