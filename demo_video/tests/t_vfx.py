"""Stand-in scene for cell.vfx: a steel cylinder, an arc empty orbiting its top, EEVEE stills.

    xvfb-run -a python3 tests/t_vfx.py            # stills f5/40/80/115 at 1280x720 + 1080p timing + preview mp4
    xvfb-run -a python3 tests/t_vfx.py stills     # only the stills
    xvfb-run -a python3 tests/t_vfx.py preview    # only the 24-frame preview (frames 30-53, 640x360)
    xvfb-run -a python3 tests/t_vfx.py timing     # only one 1920x1080 frame, timed
"""
import math
import os
import subprocess
import sys
import time

import bpy
from mathutils import Vector, Matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cell import materials, vfx, layout as L, geom as G

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
CYL_R, CYL_Z = 0.137, 1.0
ORBIT_R = 0.14

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.frame_start, sc.frame_end = 1, 120
sc.render.fps = L.FPS

# ---- stand-in geometry
cyl = G.cylinder("Pipe", CYL_R, 1.6, location=(0, 0, CYL_Z), rotation=(math.pi / 2, 0, 0), vertices=128)
cyl.data.materials.append(materials.get("steel_pipe"))
bpy.ops.mesh.primitive_plane_add(size=12)
floor = bpy.context.object
floor.data.materials.append(materials.get("concrete"))
bpy.ops.mesh.primitive_plane_add(size=12, location=(-3, 0, 3), rotation=(0, math.pi / 2, 0))
wall = bpy.context.object
wall.data.materials.append(materials.get("wall_panel"))
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.4))
stand = bpy.context.object
stand.scale = (0.3, 1.2, 0.8)
stand.data.materials.append(materials.get("painted", color="#3A5A8C"))

# ---- lighting
bpy.ops.object.light_add(type='AREA', location=(1.5, -2.5, 3.2))
key = bpy.context.object
key.data.energy = 250
key.data.size = 2.5
key.rotation_euler = (Vector((1.5, -2.5, 3.2)) - Vector((0, 0, 1))).to_track_quat('Z', 'Y').to_euler()
bpy.ops.object.light_add(type='SUN', rotation=(0.9, 0.3, 0.8))
bpy.context.object.data.energy = 1.2
w = bpy.data.worlds.new("W"); sc.world = w; w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (0.12, 0.13, 0.15, 1)
w.node_tree.nodes["Background"].inputs[1].default_value = 0.4

# ---- arc empty: orbit the cylinder top from 60 deg to 120 deg over frames 1..120, +Z toward the axis
arc = G.empty("ArcTip", size=0.05)
arc.rotation_mode = 'QUATERNION'
arc["arc_on"] = 0.0
for f in range(1, 121):
    a = math.radians(60 + 60 * (f - 1) / 119)
    p = Vector((ORBIT_R * math.cos(a), 0.0, CYL_Z + ORBIT_R * math.sin(a)))
    d = (Vector((0, 0, CYL_Z)) - p).normalized()
    arc.location = p
    arc.rotation_quaternion = d.to_track_quat('Z', 'Y')
    arc.keyframe_insert("location", frame=f)
    arc.keyframe_insert("rotation_quaternion", frame=f)
    arc["arc_on"] = 1.0 if 10 <= f <= 110 else 0.0
    arc.keyframe_insert('["arc_on"]', frame=f)
for fc in arc.animation_data.action.fcurves:
    if fc.data_path.startswith('["'):
        for kp in fc.keyframe_points:
            kp.interpolation = 'CONSTANT'

# ---- VFX under test
fx = vfx.build(arc, [(10, 110)])
vfx.setup_compositor(sc)

# ---- camera at (0.9,-0.8,1.35) looking at the empty's midway position (angle 90 deg = top of the pipe)
mid = Vector((0.0, 0.0, CYL_Z + ORBIT_R))
bpy.ops.object.camera_add(location=(0.9, -0.8, 1.35))
cam = bpy.context.object
cam.data.lens = 50
cam.rotation_euler = (cam.location - mid).to_track_quat('Z', 'Y').to_euler()
sc.camera = cam

sc.render.engine = 'BLENDER_EEVEE_NEXT'
sc.eevee.taa_render_samples = 16
sc.render.image_settings.file_format = 'PNG'
bpy.context.view_layer.update()
vfx.bake_particles(sc)


def render(frame, res, path, samples=16):
    sc.frame_set(frame)
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.eevee.taa_render_samples = samples
    sc.render.filepath = path
    t = time.time()
    bpy.ops.render.render(write_still=True)
    return time.time() - t


what = sys.argv[1] if len(sys.argv) > 1 else "all"
if what in ("all", "stills"):
    for f in (5, 40, 80, 115):
        dt = render(f, (1280, 720), os.path.join(OUT, f"vfx_f{f:03d}.png"))
        print(f"still frame {f}: {dt:.1f} s")
if what in ("all", "timing"):
    dt = render(60, (1920, 1080), os.path.join(OUT, "vfx_1080p_f060.png"))
    print(f"1920x1080 frame 60 (16 samples): {dt:.1f} s")
if what in ("all", "preview"):
    pdir = os.path.join(OUT, "vfx_preview_frames")
    os.makedirs(pdir, exist_ok=True)
    t0 = time.time()
    for f in range(30, 54):
        render(f, (640, 360), os.path.join(pdir, f"p{f:03d}.png"), samples=8)
    print(f"preview 24 frames: {time.time() - t0:.1f} s")
    subprocess.run(["/usr/bin/ffmpeg", "-y", "-loglevel", "error", "-framerate", str(L.FPS), "-start_number", "30",
                    "-i", os.path.join(pdir, "p%03d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                    os.path.join(OUT, "vfx_preview.mp4")], check=True)
    print("wrote", os.path.join(OUT, "vfx_preview.mp4"))
