"""Visual check of the RoboDK target maths: draw the generated torch targets on the Blender spool/positioner.

Renders (Blender, Cycles, 1280x720):
  out/robodk_seamA.png   tilt +90 / rot -20: static seam-A target (cyan), approach + laser-scan targets (yellow),
                         the MoveL alternative around seam A (green) and the transit pose (magenta)
  out/robodk_seamB1.png  tilt +90 / rot 360 (pipe along +X): sectors 1 (cyan) and 2 (orange), radial approach and
                         via targets (yellow)
  out/robodk_seamB2.png  tilt -90 (after the 180 deg index): sectors 3 and 4, same colours
  out/robodk_torch.png   the generated torch STL with its TCP marker
Tool Z arrows end at the TCP; the short red stub is tool X (torch body direction).
Run:  python3 tests/t_robodk_targets.py        (needs Blender's bpy; renders ~1 min)
"""
import math
import os
import sys

import bpy
import mathutils

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "robodk"))
from cell import spool, positioner, geom as G, materials
import build_station as B

OUT = os.path.join(ROOT, "out")
CYAN, ORANGE, YELLOW, GREEN, MAGENTA = (0.1, 0.9, 1.0), (1.0, 0.5, 0.1), (1.0, 0.9, 0.1), (0.4, 1.0, 0.4), (1.0, 0.2, 0.9)


def arrow(name, T, length=0.06, color=CYAN, col=None):
    """Cylinder along the tool +Z ending at the TCP (mm pose -> metres)."""
    p = mathutils.Vector([v / 1000.0 for v in T.Pos()])
    z = mathutils.Vector(T.VZ())
    x = mathutils.Vector(T.VX())
    for axis, ln, r, c in ((z, length, 0.004, color), (x, 0.03, 0.002, (1.0, 0.2, 0.2))):
        ob = G.cylinder(name + ("_z" if axis is z else "_x"), r, ln, vertices=8, collection=col)
        m = materials.get("painted", color="#%02X%02X%02X" % tuple(int(255 * v) for v in c))
        ob.data.materials.append(m)
        start = p - axis * ln if axis is z else p
        mid = start + axis * (ln / 2)
        ob.matrix_world = mathutils.Matrix.Translation(mid) @ axis.to_track_quat('Z', 'Y').to_matrix().to_4x4()


def setup(tilt, rot):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    pos = positioner.build()
    sp = spool.build()
    G.set_parent(sp["root"], pos["mount"], keep_world=False)
    positioner.set_tilt(pos, tilt)
    positioner.set_rot(pos, rot)
    bpy.ops.mesh.primitive_plane_add(size=20)
    bpy.context.object.data.materials.append(materials.get("concrete"))
    bpy.ops.object.light_add(type='SUN', location=(3, 2, 6), rotation=(0.5, 0.4, 0))
    bpy.context.object.data.energy = 3
    bpy.ops.object.light_add(type='AREA', location=(2, -3, 3))
    bpy.context.object.data.energy = 1500
    bpy.context.object.data.size = 3
    w = bpy.data.worlds.new("W")
    sc.world = w
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (0.3, 0.33, 0.38, 1)
    col = bpy.data.collections.new("Targets")
    sc.collection.children.link(col)
    return sc, col


def render(sc, loc, tgt, path, lens=45):
    bpy.ops.object.camera_add(location=loc)
    cam = bpy.context.object
    cam.data.lens = lens
    sc.camera = cam
    d = cam.location - mathutils.Vector(tgt)
    cam.rotation_euler = d.to_track_quat('Z', 'Y').to_euler()
    sc.render.resolution_x, sc.render.resolution_y = 1280, 720
    sc.render.engine = 'CYCLES'
    sc.cycles.samples = 24
    sc.cycles.use_denoising = True
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)


def sectors(col, ch, keys):
    """Two sectors of seam B with their radial approach / via targets."""
    for key, c in zip(keys, (CYAN, ORANGE)):
        poses, normals = ch[key], ch[key + "_n"]
        for k, T in enumerate(poses[::2]):
            arrow("%s_%d" % (key, k), T, 0.05, c, col)
        arrow(key + "_app", B.radial(poses[0], normals[0], B.LIFT_APPROACH), 0.04, YELLOW, col)
        arrow(key + "_lift", B.radial(poses[-1], normals[-1], B.LIFT_APPROACH), 0.04, YELLOW, col)
        arrow(key + "_via", ch[key + "_via"], 0.05, YELLOW, col)


ch = B.choreography()

# --- seam A state: tilt +90 (faceplate faces +Y), rot -20: static torch target, approach, laser scan, transit
sc, col = setup(B.TILT_B1, B.ROT_A0)
arrow("weldA", ch["weldA"], 0.08, CYAN, col)
arrow("appA", ch["approachA"], 0.05, YELLOW, col)
for k, T in enumerate(ch["scan"]):
    arrow("scan%d" % k, T, 0.03, YELLOW, col)
for k, T in enumerate(ch["weldA_movel"][::4]):
    arrow("amovel%d" % k, T, 0.05, GREEN, col)
arrow("transit", ch["transit"], 0.10, MAGENTA, col)
render(sc, (1.9, -1.4, 2.3), (0.45, 0.35, 1.55), os.path.join(OUT, "robodk_seamA.png"), 40)

# --- seam B, sectors 1-2: tilt +90, rot 360 (pipe along +X, seam centre (0.381, 0.761, 1.35))
sc, col = setup(B.TILT_B1, B.ROT_A1)
sectors(col, ch, ("sector1", "sector2"))
arrow("transitB1", ch["transitB1"], 0.10, MAGENTA, col)
render(sc, (1.9, -0.5, 2.2), (0.5, 0.72, 1.45), os.path.join(OUT, "robodk_seamB1.png"), 45)

# --- seam B, sectors 3-4: tilt -90 after the 180 deg index (seam centre (0.381, -0.761, 1.35))
sc, col = setup(B.TILT_B2, B.ROT_A1)
sectors(col, ch, ("sector3", "sector4"))
arrow("transitB2", ch["transitB2"], 0.10, MAGENTA, col)
render(sc, (1.9, 0.5, 2.2), (0.5, -0.72, 1.45), os.path.join(OUT, "robodk_seamB2.png"), 45)

# --- torch STL alone with the TCP marker
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
stl = os.path.join(B.GEN_DIR, "torch.stl")
if not os.path.exists(stl):
    B.write_stl(stl, B.torch_mesh_triangles(), "MIG torch")
bpy.ops.wm.stl_import(filepath=stl, global_scale=0.001)
torch = bpy.context.object
torch.data.materials.append(materials.get("dark_metal"))
bpy.ops.object.shade_smooth_by_angle(angle=math.radians(40))
col = bpy.data.collections.new("Targets")
sc.collection.children.link(col)
arrow("tcp", B.tcp_pose(), 0.04, CYAN, col)
bpy.ops.object.light_add(type='SUN', location=(1, -1, 2), rotation=(0.6, 0.3, 0.2))
bpy.context.object.data.energy = 4
w = bpy.data.worlds.new("W")
sc.world = w
w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (0.5, 0.52, 0.55, 1)
render(sc, (0.75, -1.25, 0.55), (0.07, 0.0, 0.24), os.path.join(OUT, "robodk_torch.png"), 50)
print("rendered robodk_seamA.png, robodk_seamB1.png, robodk_seamB2.png, robodk_torch.png")
