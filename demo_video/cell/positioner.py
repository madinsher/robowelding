"""2-axis tilt/rotate welding positioner with a U-cradle and a bolted faceplate.

Kinematics: `tilt` empty rotates about world X through (0,0,POS_TILT_AXIS_Z) (tilt axis parallel to the track;
set_tilt(+90) makes the faceplate face +Y); `rot` empty (child of tilt) at local (0,0,POS_FACEPLATE_OFFSET)
rotates about its local Z.
The spool root should be parented to `rot` at local z = POS_FIXTURE_THICK.

Static structure (world frame): base plate with anchor bolts, a welded bed spanning the two bearing towers, the centre
pedestal (POS_PEDESTAL_FOOTPRINT) with a cover plate and nameplate, blue bearing caps with a bolted end cover, and the
tilt drive (gearbox on the right cap, motor hanging below it, terminal box, corrugated cable down to a gland).
"""
import math
import bpy
import mathutils
from . import layout as L
from . import geom as G
from . import materials


def _mat(name, fallback, **kw):
    try:
        return materials.get(name, **kw)
    except KeyError:
        return materials.get(fallback)


def build(collection=None):
    col = collection or bpy.data.collections.new("Positioner")
    if collection is None:
        bpy.context.scene.collection.children.link(col)
    blue = materials.get("painted", color="#26558F", roughness=0.42, coat=0.05)     # machine blue (slightly desaturated)
    dark = materials.get("dark_metal")
    steel = materials.get("machined_steel")
    cast = _mat("cast_iron", "machined_steel")
    yellow = materials.get("safety_yellow")
    motor_grey = materials.get("painted", color="#33363A", roughness=0.5, coat=0.0)
    plate_grey = materials.get("painted", color="#C9CCD0", roughness=0.35, coat=0.0)
    cable = materials.get("cable_black")

    fx, fy, fz = L.POS_PEDESTAL_FOOTPRINT
    zt = L.POS_TILT_AXIS_Z
    hw = L.POS_TILT_HALF_WIDTH
    tx = hw + 0.06                     # tower centre |x|
    tw = 0.26                          # tower thickness (x)
    z_bed = 0.03 + 0.30                # top of the welded bed
    bed_x = 2 * tx + tw                # bed spans tower to tower

    # ---------------------------------------------------------------- base plate, bed, pedestal
    base = G.box("pos_baseplate", (2 * hw + 0.5, fy + 0.25, 0.03), (0, 0, 0.015), col, bevel=0.005); base.data.materials.append(dark)
    for i, (sx, sy) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):      # anchor bolts at the base-plate corners
        x, y = sx * (hw + 0.25 - 0.07), sy * (fy / 2 + 0.125 - 0.07)
        w = G.cylinder(f"pos_anchor_washer{i}", 0.026, 0.005, (x, y, 0.0325), vertices=24, collection=col); w.data.materials.append(steel)
        b = G.cylinder(f"pos_anchor_bolt{i}", 0.015, 0.018, (x, y, 0.044), vertices=6, collection=col); b.data.materials.append(dark)
    bed = G.box("pos_bed", (bed_x, fy, z_bed - 0.03), (0, 0, (z_bed + 0.03) / 2), col, bevel=0.015); bed.data.materials.append(blue)
    ped = G.box("pos_pedestal", (fx, fy, fz), (0, 0, z_bed + fz / 2), col, bevel=0.02); ped.data.materials.append(blue)
    # yellow hazard band around the bed
    stripe = G.box("pos_stripe", (bed_x + 0.02, fy + 0.02, 0.05), (0, 0, 0.03 + 0.06), col); stripe.data.materials.append(yellow)
    # front (loading side, -X) cover plate with screws + nameplate
    cx = -fx / 2 - 0.004
    cov = G.box("pos_cover", (0.008, 0.5, 0.40), (cx, 0, z_bed + 0.25), col, bevel=0.003); cov.data.materials.append(blue)
    for i, (sy, sz) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
        s = G.cylinder(f"pos_cover_screw{i}", 0.007, 0.005, (cx - 0.005, sy * 0.225, z_bed + 0.25 + sz * 0.175), (0, math.pi / 2, 0), vertices=12, collection=col)
        s.data.materials.append(dark)
    np_ = G.box("pos_nameplate", (0.004, 0.15, 0.085), (cx, 0.31, z_bed + fz - 0.12), col); np_.data.materials.append(plate_grey)
    for i, (sy, sz) in enumerate(((-1, -1), (1, 1))):
        r = G.cylinder(f"pos_nameplate_rivet{i}", 0.003, 0.003, (cx - 0.003, 0.31 + sy * 0.065, z_bed + fz - 0.12 + sz * 0.033), (0, math.pi / 2, 0), vertices=8, collection=col)
        r.data.materials.append(steel)

    # ---------------------------------------------------------------- bearing towers + caps
    for s in (-1, 1):
        sd = 'L' if s < 0 else 'R'
        t = G.box(f"pos_tower{sd}", (tw, 0.42, zt - z_bed), (s * tx, 0, (zt + z_bed) / 2), col, bevel=0.015); t.data.materials.append(blue)
        head = G.cylinder(f"pos_tower_head{sd}", 0.21, tw + 0.006, (s * tx, 0, zt), (0, math.pi / 2, 0), vertices=64, collection=col); head.data.materials.append(blue)   # 3 mm proud of the tower faces (no coplanar faces)
        # bearing cap: protrudes 0.03 from the tower's outer face
        cap = G.cylinder(f"pos_bearing{sd}", 0.15, 0.2, (s * (tx + 0.06), 0, zt), (0, math.pi / 2, 0), collection=col); cap.data.materials.append(blue)
        if s < 0:   # bolted end cover on the free (left) bearing
            xf = -(tx + 0.16)
            cover = G.cylinder("pos_bearing_cover", 0.10, 0.02, (xf - 0.01, 0, zt), (0, math.pi / 2, 0), vertices=48, collection=col); cover.data.materials.append(dark)
            for k in range(8):
                a = 2 * math.pi * k / 8
                b = G.cylinder(f"pos_bearing_bolt{k}", 0.008, 0.008, (xf - 0.024, 0.06 * math.cos(a), zt + 0.06 * math.sin(a)), (0, math.pi / 2, 0), vertices=6, collection=col)
                b.data.materials.append(steel)
    # tilt drive on the right cap: gearbox, motor hanging below it, terminal box, corrugated cable down to a gland on the bed
    gx = tx + 0.16 + 0.07
    gb = G.box("pos_tilt_gearbox", (0.14, 0.30, 0.34), (gx, 0, zt), col, bevel=0.01); gb.data.materials.append(blue)
    adapter = G.cylinder("pos_tilt_adapter", 0.115, 0.03, (tx + 0.16 + 0.015, 0, zt), (0, math.pi / 2, 0), collection=col); adapter.data.materials.append(dark)
    zm = zt - 0.17 - 0.16
    mot = G.cylinder("pos_tilt_motor", 0.085, 0.32, (gx, 0, zm), collection=col); mot.data.materials.append(motor_grey)
    fan = G.cylinder("pos_tilt_motor_fan", 0.092, 0.04, (gx, 0, zm - 0.16 - 0.01), collection=col); fan.data.materials.append(dark)
    mf = G.cylinder("pos_tilt_motor_flange", 0.10, 0.02, (gx, 0, zt - 0.17 - 0.01), collection=col); mf.data.materials.append(dark)
    tb = G.box("pos_tilt_terminal", (0.09, 0.06, 0.08), (gx, 0.085 + 0.03, zm + 0.04), col, bevel=0.004); tb.data.materials.append(motor_grey)
    gland_z = 0.2
    gl = G.cylinder("pos_cable_gland", 0.018, 0.03, (bed_x / 2 + 0.012, 0.12, gland_z), (0, math.pi / 2, 0), vertices=24, collection=col); gl.data.materials.append(dark)
    xt = bed_x / 2 + 0.018            # cable runs on the tower's outer face
    _cable("pos_cable_tilt", [(gx, 0.145, zm + 0.0), (gx, 0.15, zm - 0.14), (xt, 0.14, 0.62), (xt, 0.13, 0.36), (bed_x / 2 + 0.03, 0.12, gland_z)], col, cable)
    for i, z in enumerate((0.62, 0.40)):
        c = G.box(f"pos_cable_clip{i}", (0.014, 0.03, 0.02), (bed_x / 2 + 0.008, 0.135, z), col); c.data.materials.append(dark)

    # ---------------------------------------------------------------- tilt joint
    tilt = G.empty("pos_tilt", collection=col, size=0.3)
    tilt.location = (0, 0, zt)
    tilt.rotation_mode = 'XYZ'

    # U-cradle: two trunnion hubs + arms + cross beam carrying the rotary drive
    parts = []
    for s in (-1, 1):
        sd = 'L' if s < 0 else 'R'
        hub = G.cylinder(f"pos_hub{sd}", 0.14, 0.2, (s * (hw - 0.08), 0, zt), (0, math.pi / 2, 0), collection=col)
        hub.data.materials.append(steel); parts.append(hub)
        arm = G.box(f"pos_arm{sd}", (0.12, 0.22, L.POS_FACEPLATE_OFFSET + 0.12), (s * (hw - 0.14), 0, zt + (L.POS_FACEPLATE_OFFSET - 0.06) / 2), col, bevel=0.01)
        arm.data.materials.append(blue); parts.append(arm)
    cross = G.box("pos_cross", (2 * hw - 0.2, 0.26, 0.12), (0, 0, zt + L.POS_FACEPLATE_OFFSET - 0.14), col, bevel=0.01)
    cross.data.materials.append(blue); parts.append(cross)
    drive = G.cylinder("pos_rot_drive", 0.2, 0.16, (0, 0, zt + L.POS_FACEPLATE_OFFSET - 0.06), collection=col)
    drive.data.materials.append(motor_grey); parts.append(drive)
    rm = (0.28, -0.22, zt + L.POS_FACEPLATE_OFFSET - 0.1)
    rmot = G.cylinder("pos_rot_motor", 0.075, 0.28, rm, (math.pi / 2, 0, 0.0), collection=col)
    rmot.data.materials.append(motor_grey); parts.append(rmot)
    rfan = G.cylinder("pos_rot_motor_fan", 0.08, 0.03, (rm[0], rm[1] - 0.14 - 0.01, rm[2]), (math.pi / 2, 0, 0.0), collection=col)
    rfan.data.materials.append(dark); parts.append(rfan)
    rtb = G.box("pos_rot_terminal", (0.06, 0.07, 0.05), (rm[0] + 0.075 + 0.02, rm[1] - 0.02, rm[2]), col, bevel=0.003)
    rtb.data.materials.append(motor_grey); parts.append(rtb)
    # cable from the rotary motor along the right cradle arm to a gland beside the trunnion hub (rotary feed-through)
    ax = hw - 0.14                      # right arm centre x; its -Y face is at y = -0.11
    rc = _cable("pos_cable_rot", [(rm[0] + 0.125, rm[1] - 0.02, rm[2] - 0.02), (ax - 0.04, rm[1], zt + 0.09), (ax - 0.01, -0.17, zt + 0.02), (ax, -0.125, zt)], col, cable)
    parts.append(rc)
    rg = G.cylinder("pos_hub_gland", 0.014, 0.02, (ax, -0.118, zt), (math.pi / 2, 0, 0), vertices=16, collection=col)
    rg.data.materials.append(dark); parts.append(rg)
    for p in parts:
        G.set_parent(p, tilt)

    # ---------------------------------------------------------------- rotary joint + faceplate
    rot = G.empty("pos_rot", collection=col, size=0.25)
    rot.matrix_world = G.M([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, zt + L.POS_FACEPLATE_OFFSET], [0, 0, 0, 1]])
    G.set_parent(rot, tilt)
    rot.rotation_mode = 'XYZ'
    R = L.POS_FACEPLATE_RADIUS
    zf = zt + L.POS_FACEPLATE_OFFSET
    face = G.revolve("pos_faceplate", [(0.0, -0.06), (R - 0.01, -0.06), (R, -0.05), (R, -0.006), (R - 0.006, 0.0), (0.0, 0.0)], segments=96, collection=col)
    face.data.materials.append(cast)
    face.location = (0, 0, zf)
    # radial T-slots and a centre bore as real recesses (boolean cut, applied once at build time) with a dark floor
    slot_r0, slot_r1, slot_w, slot_d = 0.06, R - 0.025, 0.016, 0.012
    cutters = []
    for k in range(8):      # cutters are rotated in mesh space: the exact boolean ignores rotated object transforms here
        a = k * math.pi / 4
        rm_ = (slot_r0 + slot_r1) / 2
        c = G.box(f"pos_tslot_cut{k}", (slot_r1 - slot_r0, slot_w, 2 * slot_d), (0, 0, 0), col)
        c.data.transform(mathutils.Matrix.Translation((rm_ * math.cos(a), rm_ * math.sin(a), zf)) @ mathutils.Matrix.Rotation(a, 4, 'Z'))
        c.data.update(); cutters.append(c)
    c = G.cylinder("pos_bore_cut", 0.045, 2 * slot_d, (0, 0, zf), vertices=48, collection=col); cutters.append(c)
    _cut(face, cutters)
    for k in range(8):
        a = k * math.pi / 4
        rm_ = (slot_r0 + slot_r1) / 2
        fl = G.box(f"pos_tslot{k}", (slot_r1 - slot_r0 - 0.002, slot_w - 0.002, 0.003), (0, 0, 0), col)
        fl.data.materials.append(dark)
        _parent_local(fl, rot, (rm_ * math.cos(a), rm_ * math.sin(a), -slot_d + 0.001), (0, 0, a))
    fl = G.cylinder("pos_bore_floor", 0.044, 0.003, (0, 0, 0), vertices=48, collection=col); fl.data.materials.append(dark)
    _parent_local(fl, rot, (0, 0, -slot_d + 0.001))
    # fixture adapter ring (bolted to faceplate, carries the spool flange) + 6 dark flange bolts with washers
    ring = G.revolve("pos_fixture_ring", [(0.13, 0.0), (0.215, 0.0), (0.215, L.POS_FIXTURE_THICK), (0.13, L.POS_FIXTURE_THICK)], segments=96, collection=col)
    ring.data.materials.append(yellow)
    zb = L.POS_FIXTURE_THICK + L.FLANGE_THK
    for k in range(0, L.FLANGE_HOLES, 2):
        a = 2 * math.pi * (k + 0.5) / L.FLANGE_HOLES
        x, y = L.FLANGE_PCD / 2 * math.cos(a), L.FLANGE_PCD / 2 * math.sin(a)
        w = G.cylinder(f"pos_washer{k}", 0.024, 0.003, (0, 0, 0), vertices=24, collection=col); w.data.materials.append(steel)
        _parent_local(w, rot, (x, y, zb + 0.0015))
        b = G.cylinder(f"pos_bolt{k}", 0.016, 0.016, (0, 0, 0), vertices=6, collection=col); b.data.materials.append(dark)
        _parent_local(b, rot, (x, y, zb + 0.003 + 0.008))
    for o in (face, ring):
        o.location = rot.matrix_world.translation
        G.set_parent(o, rot)
    # mount for the spool: empty at the fixture top surface
    mount = G.empty("pos_mount", collection=col, size=0.15)
    mount.matrix_world = G.M([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, zt + L.POS_FACEPLATE_OFFSET + L.POS_FIXTURE_THICK], [0, 0, 0, 1]])
    G.set_parent(mount, rot)
    return dict(collection=col, tilt=tilt, rot=rot, mount=mount)


def _cable(name, points, col, mat, radius=0.012):
    """Corrugated black cable: bezier curve with a round bevel through the given world points."""
    cu = bpy.data.curves.new(name, 'CURVE')
    cu.dimensions = '3D'
    cu.bevel_depth = radius
    cu.bevel_resolution = 4
    cu.resolution_u = 10
    sp = cu.splines.new('BEZIER')
    sp.bezier_points.add(len(points) - 1)
    for p, xyz in zip(sp.bezier_points, points):
        p.co = xyz
        p.handle_left_type = p.handle_right_type = 'AUTO'
    ob = bpy.data.objects.new(name, cu)
    col.objects.link(ob)
    ob.data.materials.append(mat)
    return ob


def _cut(target, cutters):
    """Boolean-subtract the cutter objects from target (applied immediately) and delete the cutters."""
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action='DESELECT')
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    for c in cutters:
        m = target.modifiers.new("cut", 'BOOLEAN')
        m.operation = 'DIFFERENCE'
        m.solver = 'EXACT'
        m.object = c
        bpy.ops.object.modifier_apply(modifier=m.name)
    for c in cutters:
        bpy.data.objects.remove(c, do_unlink=True)
    for p in target.data.polygons:
        p.use_smooth = True
    bpy.ops.object.shade_smooth_by_angle(angle=math.radians(35))


def _parent_local(child, parent, loc, rot=(0, 0, 0)):
    """Parent with an explicit local transform (no world-keeping)."""
    child.parent = parent
    child.matrix_parent_inverse = mathutils.Matrix.Identity(4)
    child.matrix_basis = mathutils.Matrix.Translation(loc) @ mathutils.Euler(rot).to_matrix().to_4x4()


def set_tilt(pos, deg, frame=None):
    """tilt in degrees: 0 = faceplate up, +90 = faceplate faces +Y, -90 = faces -Y (rotation about world X)."""
    pos["tilt"].rotation_euler[0] = -math.radians(deg)
    if frame is not None:
        pos["tilt"].keyframe_insert("rotation_euler", index=0, frame=frame)


def set_rot(pos, deg, frame=None):
    pos["rot"].rotation_euler[2] = math.radians(deg)
    if frame is not None:
        pos["rot"].keyframe_insert("rotation_euler", index=2, frame=frame)
