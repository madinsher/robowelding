"""2-axis tilt/rotate welding positioner with a U-cradle and a bolted faceplate.

Kinematics: `tilt` empty rotates about world X through (0,0,POS_TILT_AXIS_Z) (tilt axis parallel to the track;
set_tilt(+90) makes the faceplate face +Y); `rot` empty (child of tilt) at local (0,0,POS_FACEPLATE_OFFSET)
rotates about its local Z.
The spool root should be parented to `rot` at local z = POS_FIXTURE_THICK.
"""
import math
import bpy
from . import layout as L
from . import geom as G
from . import materials


def build(collection=None):
    col = collection or bpy.data.collections.new("Positioner")
    if collection is None:
        bpy.context.scene.collection.children.link(col)
    blue = materials.get("painted", color="#1F4E8C", roughness=0.42)     # machine blue
    dark = materials.get("dark_metal")
    steel = materials.get("machined_steel")
    yellow = materials.get("safety_yellow")

    fx, fy, fz = L.POS_PEDESTAL_FOOTPRINT
    zt = L.POS_TILT_AXIS_Z
    hw = L.POS_TILT_HALF_WIDTH

    # pedestal: base plate + box + two bearing towers
    base = G.box("pos_baseplate", (2 * hw + 0.5, fy + 0.25, 0.03), (0, 0, 0.015), col, bevel=0.005); base.data.materials.append(dark)
    ped = G.box("pos_pedestal", (fx, fy, fz), (0, 0, fz / 2 + 0.03), col, bevel=0.02); ped.data.materials.append(blue)
    towers = []
    for s in (-1, 1):
        t = G.box(f"pos_tower{'L' if s < 0 else 'R'}", (0.26, 0.5, zt - 0.1), (s * (hw + 0.06), 0, (zt - 0.1) / 2 + 0.03), col, bevel=0.02)
        t.data.materials.append(blue); towers.append(t)
        cap = G.cylinder(f"pos_bearing{'L' if s < 0 else 'R'}", 0.19, 0.28, (s * (hw + 0.06), 0, zt), (0, math.pi / 2, 0), collection=col)
        cap.data.materials.append(dark)
    # tilt drive motor on the right tower
    mot = G.cylinder("pos_tilt_motor", 0.11, 0.35, (hw + 0.06 + 0.32, 0, zt), (0, math.pi / 2, 0), collection=col); mot.data.materials.append(dark)
    gb = G.box("pos_tilt_gearbox", (0.14, 0.3, 0.3), (hw + 0.06 + 0.2, 0, zt), col, bevel=0.01); gb.data.materials.append(blue)

    # tilt joint
    tilt = G.empty("pos_tilt", collection=col, size=0.3)
    tilt.location = (0, 0, zt)
    tilt.rotation_mode = 'XYZ'

    # U-cradle: two trunnion hubs + arms + cross beam carrying the rotary drive
    parts = []
    for s in (-1, 1):
        hub = G.cylinder(f"pos_hub{'L' if s < 0 else 'R'}", 0.14, 0.2, (s * (hw - 0.08), 0, zt), (0, math.pi / 2, 0), collection=col)
        hub.data.materials.append(steel); parts.append(hub)
        arm = G.box(f"pos_arm{'L' if s < 0 else 'R'}", (0.12, 0.22, L.POS_FACEPLATE_OFFSET + 0.12), (s * (hw - 0.14), 0, zt + (L.POS_FACEPLATE_OFFSET - 0.06) / 2), col, bevel=0.01)
        arm.data.materials.append(blue); parts.append(arm)
    cross = G.box("pos_cross", (2 * hw - 0.2, 0.26, 0.12), (0, 0, zt + L.POS_FACEPLATE_OFFSET - 0.14), col, bevel=0.01)
    cross.data.materials.append(blue); parts.append(cross)
    drive = G.cylinder("pos_rot_drive", 0.2, 0.16, (0, 0, zt + L.POS_FACEPLATE_OFFSET - 0.06), collection=col)
    drive.data.materials.append(dark); parts.append(drive)
    rmot = G.cylinder("pos_rot_motor", 0.075, 0.28, (0.28, -0.22, zt + L.POS_FACEPLATE_OFFSET - 0.1), (math.pi / 2, 0, 0.0), collection=col)
    rmot.data.materials.append(dark); parts.append(rmot)
    for p in parts:
        G.set_parent(p, tilt)

    # rotary joint + faceplate
    rot = G.empty("pos_rot", collection=col, size=0.25)
    rot.matrix_world = G.M([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, zt + L.POS_FACEPLATE_OFFSET], [0, 0, 0, 1]])
    G.set_parent(rot, tilt)
    rot.rotation_mode = 'XYZ'
    R = L.POS_FACEPLATE_RADIUS
    face = G.revolve("pos_faceplate", [(0.0, -0.06), (R - 0.01, -0.06), (R, -0.05), (R, -0.006), (R - 0.006, 0.0), (0.0, 0.0)], segments=96, collection=col)
    face.data.materials.append(steel)
    # T-slots as dark thin boxes
    for k in range(4):
        a = k * math.pi / 4
        sl = G.box(f"pos_tslot{k}", (2 * R - 0.06, 0.016, 0.004), (0, 0, 0), col)
        sl.data.materials.append(dark)
        _parent_local(sl, rot, (0, 0, -0.001), (0, 0, a))
    # fixture adapter ring (bolted to faceplate, carries the spool flange) + bolts
    ring = G.revolve("pos_fixture_ring", [(0.13, 0.0), (0.215, 0.0), (0.215, L.POS_FIXTURE_THICK), (0.13, L.POS_FIXTURE_THICK)], segments=96, collection=col)
    ring.data.materials.append(yellow)
    for k in range(L.FLANGE_HOLES):
        a = 2 * math.pi * (k + 0.5) / L.FLANGE_HOLES
        x, y = L.FLANGE_PCD / 2 * math.cos(a), L.FLANGE_PCD / 2 * math.sin(a)
        b = G.cylinder(f"pos_bolt{k}", 0.019, 0.02, (0, 0, 0), vertices=6, collection=col)
        b.data.materials.append(steel)
        _parent_local(b, rot, (x, y, L.POS_FIXTURE_THICK + L.FLANGE_THK + 0.01))
    for o in (face, ring):
        o.location = rot.matrix_world.translation
        G.set_parent(o, rot)
    # mount for the spool: empty at the fixture top surface
    mount = G.empty("pos_mount", collection=col, size=0.15)
    mount.matrix_world = G.M([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, zt + L.POS_FACEPLATE_OFFSET + L.POS_FIXTURE_THICK], [0, 0, 0, 1]])
    G.set_parent(mount, rot)
    # yellow/black hazard stripe on the pedestal front edge
    stripe = G.box("pos_stripe", (fx + 0.02, fy + 0.02, 0.05), (0, 0, 0.03 + 0.06), col); stripe.data.materials.append(yellow)
    # cable loops
    return dict(collection=col, tilt=tilt, rot=rot, mount=mount)


def _parent_local(child, parent, loc, rot=(0, 0, 0)):
    """Parent with an explicit local transform (no world-keeping)."""
    import mathutils
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
