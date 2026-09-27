"""Shared test harness for stage-2 module checks: empty scene with a floor and neutral lights, look-at cameras,
Cycles/EEVEE stills, footprint checks.  Stills go to stage2_logistics/out/tests/.

    import harness as H
    H.new_scene()                      # factory-empty scene, metric, 24 fps, floor + 3 area lights + world
    ... build your module ...
    H.still("t_conveyor_a", cam=(-2, -6, 3), aim=(-5.5, -2.3, 0.8), lens=30)       # Cycles 16 spp 640x360
    H.check_inside(objs, box_min, box_max)                                          # footprint assertion
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE2 = os.path.dirname(HERE)
if STAGE2 not in sys.path:
    sys.path.insert(0, STAGE2)
import tools  # noqa: E402,F401  (demo_video on sys.path)
import bpy  # noqa: E402
import mathutils  # noqa: E402
from cell import materials  # noqa: E402

OUT = os.path.join(STAGE2, "out", "tests")


def new_scene(floor=True, lights=True):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.fps = 24
    sc.unit_settings.system = 'METRIC'
    if floor:
        bpy.ops.mesh.primitive_plane_add(size=60, location=(-4.0, 0.0, 0.0))
        fl = bpy.context.object
        fl.name = "test_floor"
        fl.data.materials.append(materials.get("concrete"))
    if lights:
        w = bpy.data.worlds.new("TestWorld")
        sc.world = w
        w.use_nodes = True
        w.node_tree.nodes["Background"].inputs[0].default_value = (0.35, 0.37, 0.40, 1.0)
        w.node_tree.nodes["Background"].inputs[1].default_value = 0.6
        for i, (loc, e) in enumerate((((-2.0, -6.0, 7.0), 2500.0), ((-9.0, 5.0, 6.0), 1500.0), ((2.0, 4.0, 5.0), 900.0))):
            ld = bpy.data.lights.new(f"test_light_{i}", 'AREA')
            ld.energy = e
            ld.size = 4.0
            ob = bpy.data.objects.new(f"test_light_{i}", ld)
            sc.collection.objects.link(ob)
            ob.location = loc
            d = mathutils.Vector((-5.0, 0.0, 0.5)) - mathutils.Vector(loc)
            ob.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    return sc


def camera(name, cam, aim, lens=35.0):
    cd = bpy.data.cameras.new(name)
    cd.lens = lens
    cd.sensor_width = 36
    cd.clip_start = 0.05
    ob = bpy.data.objects.new(name, cd)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = cam
    d = mathutils.Vector(aim) - mathutils.Vector(cam)
    ob.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    return ob


def still(name, cam, aim, lens=35.0, res=(640, 360), engine='CYCLES', samples=16, frame=None):
    """Render one still to out/tests/<name>.png and return the path."""
    sc = bpy.context.scene
    if frame is not None:
        sc.frame_set(frame)
    sc.camera = camera("cam_" + name, cam, aim, lens)
    sc.render.engine = engine
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = 'PNG'
    sc.view_settings.view_transform = 'AgX'
    sc.view_settings.look = 'AgX - Medium High Contrast'
    if engine == 'CYCLES':
        sc.cycles.device = 'CPU'
        sc.cycles.samples = samples
        sc.cycles.use_denoising = True
        sc.cycles.max_bounces = 4
    else:
        sc.eevee.taa_render_samples = samples
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name + ".png")
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    print("[still]", path, flush=True)
    return path


def world_bbox(objs):
    """(min, max) world-space AABB over the evaluated mesh objects in objs (modifiers applied)."""
    dg = bpy.context.evaluated_depsgraph_get()
    lo = [1e9] * 3
    hi = [-1e9] * 3
    for ob in objs:
        if ob.type != 'MESH':
            continue
        ev = ob.evaluated_get(dg)
        for c in ev.bound_box:
            p = ev.matrix_world @ mathutils.Vector(c)
            for k in range(3):
                lo[k] = min(lo[k], p[k])
                hi[k] = max(hi[k], p[k])
    return lo, hi


def check_inside(objs, box_min, box_max, tol=1e-3, label=""):
    lo, hi = world_bbox(objs)
    ok = all(lo[k] >= box_min[k] - tol and hi[k] <= box_max[k] + tol for k in range(3))
    print(f"[bbox]{' ' + label if label else ''} {[round(v, 3) for v in lo]} .. {[round(v, 3) for v in hi]} "
          f"inside {list(box_min)}..{list(box_max)}: {ok}", flush=True)
    return ok


def n_tris(objs):
    dg = bpy.context.evaluated_depsgraph_get()
    n = 0
    for ob in objs:
        if ob.type == 'MESH':
            me = ob.evaluated_get(dg).to_mesh()
            n += sum(len(p.vertices) - 2 for p in me.polygons)
            ob.evaluated_get(dg).to_mesh_clear()
    return n


def collection_objects(col):
    return list(col.all_objects)
