"""Visual check of the RoboDK target maths: draw the generated torch targets on the Blender spool/positioner.

Renders out/robodk_seamA.png, out/robodk_seamB.png (tool Z arrows at every target, cyan = weld, yellow =
approach/scan) and out/robodk_torch.png (the generated torch STL with its TCP marker).
Run:  python3 tests/t_robodk_targets.py
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


def arrow(name, T, length=0.06, color=(0.1, 0.9, 1.0), col=None):
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
    return sc


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


ch = B.choreography()

# --- seam A state: tilt 90, rot 70: static torch target, approach, laser scan
sc = setup(90, B.ROT_A0)
col = bpy.data.collections.new("Targets")
sc.collection.children.link(col)
arrow("weldA", ch["weldA"], 0.08, col=col)
arrow("appA", ch["approachA"], 0.05, (1.0, 0.9, 0.1), col)
for k, T in enumerate(ch["scan"]):
    arrow("scan%d" % k, T, 0.03, (1.0, 0.9, 0.1), col)
for k, T in enumerate(ch["weldA_movel"][::4]):
    arrow("amovel%d" % k, T, 0.05, (0.4, 1.0, 0.4), col)
render(sc, (1.5, -1.3, 1.9), (0.4, 0.0, 1.4), os.path.join(OUT, "robodk_seamA.png"), 50)

# --- seam B state: tilt 90, rot 450 (pipe along +Y): sectors 1 and 2
sc = setup(90, B.ROT_A1)
col = bpy.data.collections.new("Targets")
sc.collection.children.link(col)
for key, c in (("sector1", (0.1, 0.9, 1.0)), ("sector2", (1.0, 0.5, 0.1))):
    for k, T in enumerate(ch[key][::2]):
        arrow("%s_%d" % (key, k), T, 0.05, c, col)
render(sc, (1.7, -0.9, 2.0), (0.76, 0.38, 1.4), os.path.join(OUT, "robodk_seamB.png"), 50)

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
arrow("tcp", B.tcp_pose(), 0.04, col=col)
bpy.ops.object.light_add(type='SUN', location=(1, -1, 2), rotation=(0.6, 0.3, 0.2))
bpy.context.object.data.energy = 4
w = bpy.data.worlds.new("W")
sc.world = w
w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (0.5, 0.52, 0.55, 1)
render(sc, (0.75, -1.25, 0.55), (0.07, 0.0, 0.24), os.path.join(OUT, "robodk_torch.png"), 50)
print("rendered robodk_seamA.png, robodk_seamB.png, robodk_torch.png")
