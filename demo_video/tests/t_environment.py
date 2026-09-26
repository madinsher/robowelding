"""Environment test: hall + fence + equipment + lighting with stand-ins, 3 EEVEE stills.

    xvfb-run -a python3 tests/t_environment.py [--res W H] [--samples N] [--cam wide|medium|reverse ...]
"""
import bpy, sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mathutils
from cell import environment, materials, geom as G

ap = argparse.ArgumentParser()
ap.add_argument("--res", nargs=2, type=int, default=(1280, 720))
ap.add_argument("--samples", type=int, default=32)
ap.add_argument("--cam", nargs="*", default=["wide", "medium", "reverse"])
ap.add_argument("--engine", default="EEVEE")
args = ap.parse_args()

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene

t0 = time.time()
env = environment.build()
lit = environment.build_lighting(sc)
print(f"environment: {len(env['objects'])} objects, {len(lit['lights'])} lights, built in {time.time() - t0:.1f}s")

# stand-ins: positioner (grey cylinder) and robot (orange box)
col = bpy.data.collections.new("StandIns"); sc.collection.children.link(col)
cyl = G.cylinder("standin_positioner", 0.45, 1.2, location=(0, 0, 0.6), collection=col)
cyl.data.materials.append(materials.get("abb_grey"))
box = G.box("standin_robot", (0.8, 0.8, 1.8), location=(2.0, 0, 0.9), collection=col)
box.data.materials.append(materials.get("abb_orange"))

tris = sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in bpy.data.objects if o.type == 'MESH')
print(f"triangles: {tris}")

sc.render.resolution_x, sc.render.resolution_y = args.res
sc.render.resolution_percentage = 100
if args.engine == "EEVEE":
    sc.render.engine = 'BLENDER_EEVEE_NEXT'
    sc.eevee.taa_render_samples = args.samples
else:
    sc.render.engine = 'CYCLES'; sc.cycles.samples = args.samples; sc.cycles.use_denoising = True
sc.view_settings.view_transform = 'AgX'
sc.view_settings.look = 'AgX - Medium High Contrast'
sc.view_settings.exposure = 0.0

CAMS = dict(wide=((-4.5, -6.5, 2.6), (0.8, 0, 1.2), 22),
            medium=((3.2, -3.4, 2.0), (0.5, 0, 1.0), 32),
            reverse=((-1.5, 2.5, 1.8), (0.8, 0, 1.0), 28))
for name in args.cam:
    loc, tgt, lens = CAMS[name]
    cam_data = bpy.data.cameras.new(name); cam_data.lens = lens; cam_data.clip_end = 200
    cam = bpy.data.objects.new("cam_" + name, cam_data); col.objects.link(cam)
    cam.location = loc
    cam.rotation_euler = (mathutils.Vector(loc) - mathutils.Vector(tgt)).to_track_quat('Z', 'Y').to_euler()
    sc.camera = cam
    sc.render.filepath = os.path.join(OUT, f"environment_{name}.png")
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    print(f"rendered {sc.render.filepath} in {time.time() - t0:.1f}s")
