"""Spool on the positioner: bead half-welded and hot, heat tint, shop markings.

    python3 tests/t_spool_pos.py [--only N]                 Cycles, simple test lights
    xvfb-run -a python3 tests/t_spool_pos.py --eevee --env   EEVEE with the real hall + environment lighting (final look)
    ... --swatches                                           extra still: spheres of the machine/pipe materials
"""
import bpy, sys, os, time
import mathutils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cell import spool, positioner, geom as G, materials

EEVEE = "--eevee" in sys.argv
ENV = "--env" in sys.argv
ONLY = int(sys.argv[sys.argv.index("--only") + 1]) if "--only" in sys.argv else None
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
pos = positioner.build()
sp = spool.build()
G.set_parent(sp["root"], pos["mount"], keep_world=False)
positioner.set_tilt(pos, 90)
positioner.set_rot(pos, 0)
# bead A: welded from the top (u=0.75) backwards over 55 % of the circumference, arc still on -> hot tail behind it
b = sp["beads"]["A"]; b["w0_start"] = 0.30; b["w0_prog"] = 0.55; b["w0_dir"] = -1.0; b["w0_hot"] = 1.0
# bead B: 60 % welded, arc at the upper-outer side
b = sp["beads"]["B"]; b["w0_start"] = 0.0; b["w0_prog"] = 0.60; b["w0_dir"] = 1.0; b["w0_hot"] = 1.0
for t in sp["tints"].values():
    t["tint_on"] = 1.0
for b in sp["beads"].values():
    b.update_tag()               # ID-prop writes from Python do not tag the depsgraph; frame_set/keyframes do
bpy.context.view_layer.update()
dg = bpy.context.evaluated_depsgraph_get(); dg.update()
for k in ("A", "B"):
    ev = sp["tints"][k].evaluated_get(dg)
    assert abs(ev["w0"][1] - sp["beads"][k]["w0_prog"]) < 1e-6, ("tint driver not following bead", k, list(ev["w0"]))
    ev = sp["beads"][k].evaluated_get(dg)
    assert abs(ev["w0"][3] - 1.0) < 1e-6, ("bead packed props not driven", k, list(ev["w0"]))
print("progress drivers ok")

if ENV:
    from cell import environment
    environment.build()
    environment.build_lighting(sc)
else:
    bpy.ops.mesh.primitive_plane_add(size=20)
    bpy.context.object.data.materials.append(materials.get("concrete"))
    bpy.ops.object.light_add(type='SUN', location=(3, 2, 6), rotation=(0.5, 0.4, 0)); bpy.context.object.data.energy = 2.0
    bpy.context.object.name = "env_light_key"
    bpy.ops.object.light_add(type='AREA', location=(2, -3, 3)); bpy.context.object.data.energy = 900; bpy.context.object.data.size = 3
    bpy.ops.object.light_add(type='AREA', location=(-1, 3, 3)); bpy.context.object.data.energy = 600; bpy.context.object.data.size = 3
    w = bpy.data.worlds.new("W"); sc.world = w; w.use_nodes = True; w.node_tree.nodes["Background"].inputs[0].default_value = (0.25, 0.27, 0.31, 1)
bpy.context.view_layer.update()

SHOTS = [  # (cam_from, aim, lens)  -- like the storyboard close-ups
    ((0.9, 1.35, 2.0), (0.0, 0.40, 1.47), 42),      # seam A close-up ~1.3 m (C3b/C4)
    ((1.3, 2.05, 2.05), (0.42, 0.76, 1.42), 42),    # seam B from the +Y side, pipe marking (C6b)
    ((-1.3, 2.3, 1.9), (0.25, 0.55, 1.42), 35),     # elbow back marking + both seams
]
sc.render.resolution_x = 1280; sc.render.resolution_y = 720
if EEVEE:   # same cost-saving settings as build.py setup_render (no GPU here)
    sc.render.engine = 'BLENDER_EEVEE_NEXT'; sc.eevee.taa_render_samples = 8; sc.eevee.use_raytracing = False
    sc.eevee.use_shadows = True; sc.eevee.shadow_resolution_scale = 0.25
    sc.eevee.shadow_ray_count = 2; sc.eevee.shadow_step_count = 4
    for ob in bpy.data.objects:
        if ob.type == 'LIGHT':
            ob.data.use_shadow = ob.name in ("env_light_key", "ArcLight")
            if ob.data.use_shadow:
                ob.data.shadow_maximum_resolution = max(ob.data.shadow_maximum_resolution, 0.015)
else:
    sc.render.engine = 'CYCLES'; sc.cycles.samples = 32; sc.cycles.use_denoising = True
sc.view_settings.view_transform = 'AgX'
for i, (loc, tgt, lens) in enumerate(SHOTS):
    if ONLY is not None and i != ONLY:
        continue
    bpy.ops.object.camera_add(location=loc); cam = bpy.context.object; cam.data.lens = lens; sc.camera = cam
    d = cam.location - mathutils.Vector(tgt); cam.rotation_euler = d.to_track_quat('Z', 'Y').to_euler()
    sc.render.filepath = os.path.join(OUT, f"spool_{'eevee' if EEVEE else 'cycles'}{'_env' if ENV else ''}_{i}.png")
    t0 = time.time(); bpy.ops.render.render(write_still=True)
    print(f"rendered {sc.render.filepath} in {time.time() - t0:.1f}s")
if "--swatches" in sys.argv:    # material check: cast_iron vs machined_steel (faceplate), abb_orange, steel_pipe
    names = ["cast_iron", "machined_steel", "abb_orange", "steel_pipe", "abb_grey", "galvanized"]
    for k, n in enumerate(names):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.11, location=(-1.2 + 0.26 * k, 1.35, 1.6), segments=48, ring_count=24)
        ob = bpy.context.object; bpy.ops.object.shade_smooth(); ob.data.materials.append(materials.get(n))
    bpy.ops.object.camera_add(location=(-0.55, 3.2, 1.8)); cam = bpy.context.object; cam.data.lens = 40; sc.camera = cam
    d = cam.location - mathutils.Vector((-0.55, 1.35, 1.6)); cam.rotation_euler = d.to_track_quat('Z', 'Y').to_euler()
    sc.render.filepath = os.path.join(OUT, f"spool_{'eevee' if EEVEE else 'cycles'}{'_env' if ENV else ''}_swatches.png")
    bpy.ops.render.render(write_still=True); print("rendered", sc.render.filepath, names)
print("tris:", sum(len(o.data.polygons) for o in bpy.data.objects if o.type == 'MESH'))
