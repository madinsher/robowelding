"""Environment test: hall + fence + equipment + lighting around the real positioner / spool / robot, EEVEE stills.

    xvfb-run -a python3 tests/t_environment.py [--res W H] [--samples N] [--cam wide medium reverse ...] [--standins]

Emulates build.setup_render for EEVEE: 8 TAA samples, no screen-space raytracing / fast GI, shadows only from
env_light_key (and ArcLight, absent here).  Stills go to out/environment_<cam>.png.
"""
import bpy, sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mathutils
from cell import environment, materials, geom as G, positioner, spool, layout as L

ap = argparse.ArgumentParser()
ap.add_argument("--res", nargs=2, type=int, default=(1280, 720))
ap.add_argument("--samples", type=int, default=8)
ap.add_argument("--cam", nargs="*", default=["wide", "medium", "reverse"])
ap.add_argument("--engine", default="EEVEE")
ap.add_argument("--standins", action="store_true", help="orange box instead of the robot (faster build)")
ap.add_argument("--prefix", default="environment")
args = ap.parse_args()

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene

t0 = time.time()
pos = positioner.build()
sp = spool.build()
G.set_parent(sp["root"], pos["mount"], keep_world=False)
positioner.set_tilt(pos, 90.0)          # faceplate faces +Y, pipe leg along +X (seam-B sector orientation)
positioner.set_rot(pos, 0.0)
if args.standins:
    col = bpy.data.collections.new("StandIns"); sc.collection.children.link(col)
    box = G.box("standin_robot", (0.8, 0.8, 1.8), location=(L.TRACK_X, L.ROBOT_HOME_Y, 0.9 + L.TRACK_TOP_Z), collection=col)
    box.data.materials.append(materials.get("abb_orange"))
else:
    from cell import robot_build
    rob = robot_build.build()
    robot_build.set_q(rob, L.ROBOT_Q_HOME)
    robot_build.set_track(rob, L.ROBOT_HOME_Y)
print(f"machines built in {time.time() - t0:.1f}s")
t0 = time.time()
env = environment.build()
lit = environment.build_lighting(sc)
print(f"environment: {len(env['objects'])} objects, {len(lit['lights'])} lights, built in {time.time() - t0:.1f}s")
tris = sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in bpy.data.objects if o.type == 'MESH')
print(f"triangles: {tris}")

# ---- render settings (mirror build.setup_render)
sc.render.resolution_x, sc.render.resolution_y = args.res
sc.render.resolution_percentage = 100
sc.render.image_settings.file_format = 'PNG'
sc.render.image_settings.color_mode = 'RGB'
sc.view_settings.view_transform = 'AgX'
sc.view_settings.look = 'AgX - Medium High Contrast'
sc.view_settings.exposure = 0.0
if args.engine == "EEVEE":
    sc.render.engine = 'BLENDER_EEVEE_NEXT'
    sc.eevee.taa_render_samples = args.samples
    sc.eevee.use_shadows = True
    sc.eevee.use_raytracing = False
    sc.eevee.use_fast_gi = False
    sc.eevee.shadow_ray_count = 2
    sc.eevee.shadow_step_count = 4
    sc.eevee.shadow_resolution_scale = 0.25
    sc.eevee.use_volumetric_shadows = False
    keep = {"env_light_key", "ArcLight"}
    for ob in bpy.data.objects:
        if ob.type == 'LIGHT':
            if ob.name in keep:
                ob.data.use_shadow = True
                ob.data.shadow_maximum_resolution = max(ob.data.shadow_maximum_resolution, 0.015)
            else:
                ob.data.use_shadow = False
else:
    sc.render.engine = 'CYCLES'; sc.cycles.samples = args.samples; sc.cycles.use_denoising = True

CAMS = dict(wide=((-4.5, -6.5, 2.6), (0.8, 0, 1.2), 22),
            medium=((3.2, -3.4, 2.0), (0.5, 0, 1.0), 32),
            reverse=((-1.5, 2.5, 1.8), (0.8, 0, 1.0), 28),
            cleaner=((0.6, -4.6, 1.6), (2.0, -2.55, 0.8), 45),
            corner=((-3.0, 5.0, 1.7), (2.5, 0.5, 0.9), 35))
ccol = bpy.data.collections.new("TestCams"); sc.collection.children.link(ccol)
for name in args.cam:
    loc, tgt, lens = CAMS[name]
    cam_data = bpy.data.cameras.new(name); cam_data.lens = lens; cam_data.clip_end = 200
    cam = bpy.data.objects.new("cam_" + name, cam_data); ccol.objects.link(cam)
    cam.location = loc
    cam.rotation_euler = (mathutils.Vector(loc) - mathutils.Vector(tgt)).to_track_quat('Z', 'Y').to_euler()
    sc.camera = cam
    sc.render.filepath = os.path.join(OUT, f"{args.prefix}_{name}.png")
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    print(f"rendered {sc.render.filepath} in {time.time() - t0:.1f}s")
