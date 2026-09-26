"""VFX test in the real cell: positioner (tilt +90) + spool + IK-posed robot + hall lighting, no animation module.

Timeline (120 frames): laser seam search 1..40 (TCP 35 mm above seam A, sweep +/-60 mm along X), move down,
arc on 50..110 while the part rotates 100 deg, lift, arc off 111..120 (smoke fades, everything must be off).

    xvfb-run -a python3 tests/t_vfx.py                       # all shots at 960x540, 8 samples
    xvfb-run -a python3 tests/t_vfx.py --shots close mid     # subset: close close2 mid laser laser_wide index off timing
    xvfb-run -a python3 tests/t_vfx.py --shots close2 --hide smoke --tag _nosmoke   # attribute an effect
    xvfb-run -a python3 tests/t_vfx.py --shots close2 --set ARC_LIGHT_POWER=100 --tag _p100   # try a tunable
    xvfb-run -a python3 tests/t_vfx.py --res 1920 1080 --shots timing   # 1080p cost with / without the plume
    xvfb-run -a python3 tests/t_vfx.py --no-env               # faster: fallback lighting instead of the hall
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import bpy
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cell import layout as L, geom as G, materials, spool, positioner, robot_build as RB, environment, vfx  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
DEG = math.pi / 180
ap = argparse.ArgumentParser()
ap.add_argument("--res", nargs=2, type=int, default=(960, 540))
ap.add_argument("--samples", type=int, default=8)
ap.add_argument("--shots", nargs="*", default=["close", "close2", "mid", "laser", "laser_wide", "index", "off"])
ap.add_argument("--no-env", action="store_true")
ap.add_argument("--hide", nargs="*", default=[], help="vfx objects to hide for the render: smoke glow core light spark_emitter")
ap.add_argument("--tag", default="", help="suffix for the output file names")
ap.add_argument("--set", nargs="*", default=[], help="override vfx tunables, e.g. ARC_LIGHT_POWER=120 SMOKE_DENSITY=80")
args = ap.parse_args()
import ast  # noqa: E402
for kv in args.set:
    k, v = kv.split("=", 1)
    assert hasattr(vfx, k), k
    setattr(vfx, k, ast.literal_eval(v))

LASER = (1, 40)
ARC = (50, 110)
N = 120

# ------------------------------------------------------------------ scene
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.fps = L.FPS
sc.frame_start, sc.frame_end = 1, N
t0 = time.time()
pos = positioner.build()
sp = spool.build()
G.set_parent(sp["root"], pos["mount"], keep_world=False)
rob = RB.build()
print(f"machines built in {time.time() - t0:.1f}s")

# positioner: tilt +90 (faceplate faces +Y); rotate 100 deg during the arc
positioner.set_tilt(pos, 90.0, frame=1)
positioner.set_rot(pos, 70.0, frame=ARC[0] - 2)
positioner.set_rot(pos, 170.0, frame=ARC[1] + 2)
for fc in pos["rot"].animation_data.action.fcurves:
    for kp in fc.keyframe_points:
        kp.interpolation = 'LINEAR'

# ------------------------------------------------------------------ robot: IK on seam A top point
bpy.context.view_layer.update()
TOP = np.array([0.0, L.POS_FACEPLATE_OFFSET + L.POS_FIXTURE_THICK + L.SEAM_A_Z, L.POS_TILT_AXIS_Z + L.SEAM_RADIUS])
print("seam A top point", TOP.round(3))
arm, TOOL = rob["arm"], RB.tool_transform()
QREF = np.array([0, 0, 0, 0, 1.2, 0])
TRACK_Y = 0.0


def target(p, n, t, lean, push=10.0):
    n = n / np.linalg.norm(n); t = t / np.linalg.norm(t)
    z = -n * math.cos(push * DEG) + t * math.sin(push * DEG)
    x = lean - np.dot(lean, z) * z; x /= np.linalg.norm(x); y = np.cross(z, x)
    T = np.eye(4); T[:3, 0] = x; T[:3, 1] = y; T[:3, 2] = z; T[:3, 3] = p
    return T


def lean(p):
    d = np.array([L.TRACK_X - p[0], TRACK_Y - p[1], 0.0]); d /= np.linalg.norm(d)
    return d + np.array([0, 0, 0.9])


n_up, t_x = np.array([0, 0, 1.0]), np.array([1.0, 0, 0])
targets = {}
for f in range(1, N + 1):
    if f <= LASER[1]:
        s = (f - LASER[0]) / (LASER[1] - LASER[0])
        x = -0.06 + 0.12 * (0.5 - 0.5 * math.cos(math.pi * s))
        p = TOP + np.array([x, 0, 0.035])
    elif f < ARC[0] - 2:
        s = (f - LASER[1]) / (ARC[0] - 2 - LASER[1])
        p = TOP + np.array([0.06 * (1 - s), 0, 0.035 * (1 - s)])
    elif f <= ARC[1] + 2:
        w = 0.0025 * math.sin(2 * math.pi * 2.5 * (f - ARC[0]) / L.FPS)
        p = TOP + np.array([w, 0, 0])
    else:
        s = (f - ARC[1] - 2) / (N - ARC[1] - 2)
        p = TOP + np.array([0, 0, 0.15 * s])
    targets[f] = target(p, n_up, t_x, lean(p))
base = RB.base_matrix(TRACK_Y)
q = np.array(L.ROBOT_Q_HOME)
fails = 0
for f in range(1, N + 1):
    q, ok, err = arm.ik(targets[f], q, base, TOOL, free_spin=True, q_ref=QREF, iters=200)
    fails += (not ok)
    RB.set_q(rob, q, frame=f)
RB.set_track(rob, TRACK_Y, frame=1)
print(f"IK: {fails} failures")
arc = G.empty("arc_point", collection=rob["collection"], size=0.03)
arc.parent = rob["tcp"]
import mathutils  # noqa: E402
arc.matrix_parent_inverse = mathutils.Matrix.Identity(4)

# weld bead A progress (same convention as animation.py: u = angle/2pi in the bead frame)
bpy.context.view_layer.update()
bead = sp["beads"]["A"]
us = []
for f in range(ARC[0], ARC[1] + 1):
    sc.frame_set(f)
    pl = bead.matrix_world.inverted() @ Vector(TOP)
    us.append((math.atan2(pl.y, pl.x) / (2 * math.pi)) % 1.0)
d = us[1] - us[0]; d -= round(d)
bead["w0_start"] = us[0]; bead["w0_dir"] = 1.0 if d > 0 else -1.0
bead.keyframe_insert('["w0_start"]', frame=1); bead.keyframe_insert('["w0_dir"]', frame=1)
bead["w0_prog"] = 0.0; bead.keyframe_insert('["w0_prog"]', frame=ARC[0] - 1)
for k, f in enumerate(range(ARC[0], ARC[1] + 1)):
    du = (us[k] - us[0]) * bead["w0_dir"]; du -= math.floor(du)
    bead["w0_prog"] = float(du + 0.004); bead.keyframe_insert('["w0_prog"]', frame=f)
bead["w0_hot"] = 0.0; bead.keyframe_insert('["w0_hot"]', frame=ARC[0] - 1)
bead["w0_hot"] = 1.0; bead.keyframe_insert('["w0_hot"]', frame=ARC[0]); bead.keyframe_insert('["w0_hot"]', frame=ARC[1])
bead["w0_hot"] = 0.0; bead.keyframe_insert('["w0_hot"]', frame=ARC[1] + 40)

# ------------------------------------------------------------------ environment + lighting
t0 = time.time()
if not args.no_env:
    environment.build()
    environment.build_lighting(sc)
else:
    bpy.ops.mesh.primitive_plane_add(size=40)
    bpy.context.object.data.materials.append(materials.get("concrete"))
    bpy.ops.object.light_add(type='AREA', location=(0.8, 0, 6.2)); k = bpy.context.object
    k.name = "env_light_key"; k.data.name = "env_light_key"; k.data.energy = 550; k.data.size = 3.5
    w = bpy.data.worlds.new("W"); sc.world = w; w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (0.34, 0.38, 0.44, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = 0.12
print(f"environment built in {time.time() - t0:.1f}s")

# ------------------------------------------------------------------ VFX under test
t0 = time.time()
fx = vfx.build(arc, [ARC])
line = vfx.laser_line(rob["torch"]["sensor"], [LASER])
vfx.setup_compositor(sc)
print(f"vfx built in {time.time() - t0:.1f}s")

# ------------------------------------------------------------------ render settings (as build.setup_render for EEVEE)
sc.render.engine = 'BLENDER_EEVEE_NEXT'
sc.render.resolution_x, sc.render.resolution_y = args.res
sc.render.resolution_percentage = 100
sc.render.image_settings.file_format = 'PNG'
sc.render.image_settings.color_mode = 'RGB'
sc.view_settings.view_transform = 'AgX'
sc.view_settings.look = 'AgX - Medium High Contrast'
sc.eevee.taa_render_samples = args.samples
sc.eevee.use_shadows = True
sc.eevee.use_raytracing = False
sc.eevee.use_fast_gi = False
sc.eevee.shadow_ray_count = 2
sc.eevee.shadow_step_count = 4
sc.eevee.shadow_resolution_scale = 0.25
sc.eevee.use_volumetric_shadows = False
for ob in bpy.data.objects:
    if ob.type == 'LIGHT':
        if ob.name in ("env_light_key", "ArcLight"):
            ob.data.use_shadow = True
            ob.data.shadow_maximum_resolution = 0.008 if ob.name == "ArcLight" else max(ob.data.shadow_maximum_resolution, 0.015)
        else:
            ob.data.use_shadow = False
bpy.context.view_layer.update()
t0 = time.time()
vfx.bake_particles(sc)
print(f"particles baked in {time.time() - t0:.1f}s")

# ------------------------------------------------------------------ off-state check (finding 5)
for f in (5, 45, 111, 118):
    sc.frame_set(f)
    dg = bpy.context.evaluated_depsgraph_get()
    core, glow, light, smoke = (fx[k].evaluated_get(dg) for k in ("core", "glow", "light", "smoke"))
    print(f"frame {f:3d}: core scale {tuple(round(v, 4) for v in core.scale)} glow scale {tuple(round(v, 4) for v in glow.scale)} "
          f"arc light {light.data.energy:.1f} W smoke {smoke['smoke']:.2f} laser scale {tuple(round(v, 2) for v in line.evaluated_get(dg).scale)} "
          f"laser spot {bpy.data.objects['LaserSpot'].evaluated_get(dg).data.energy:.0f} W")

# ------------------------------------------------------------------ cameras
CAMS = {}


def cam(name, loc, tgt, lens):
    cd = bpy.data.cameras.new(name); cd.lens = lens; cd.sensor_width = 36; cd.clip_start = 0.02; cd.clip_end = 200
    ob = bpy.data.objects.new("cam_" + name, cd); sc.collection.objects.link(ob)
    ob.location = loc
    ob.rotation_euler = (Vector(loc) - Vector(tgt)).to_track_quat('Z', 'Y').to_euler()
    CAMS[name] = ob
    return ob


A = Vector(TOP)
# the seam hides behind the flange (r 0.2) from -Y and behind the rotary drive motor from +X/-Y: look from -X, high
cam("close", A + Vector((-0.40, -0.36, 0.41)), A + Vector((0.0, 0.0, 0.01)), 60)
cam("close2", A + Vector((0.55, -0.15, 0.42)), A + Vector((0.0, 0.0, 0.01)), 50)
cam("mid", A + Vector((-1.7, -0.3, 0.55)), A + Vector((0.0, 0.0, 0.22)), 35)     # from -X: the faceplate (r 0.32) hides the seam from -Y
cam("laser_wide", A + Vector((-0.9, -0.9, 0.7)), A + Vector((0.0, 0.0, 0.03)), 50)
cam("index", (1.9, -3.6, 2.3), (0.3, 0.0, 1.0), 32)

SHOTS = dict(close=("close", 80), close2=("close2", 75), mid=("mid", 90), laser=("close", 20), laser_start=("close", 3), laser_end=("close", 38), laser_wide=("laser_wide", 30),
             index=("index", 118), off=("mid", 116))

# laser projection sanity: centre sample of the first key (sensor frame; +Z into the part) and the z spread across the line
kb = line.data.shape_keys.key_blocks[1]
zs = [kb.data[i].co.z for i in range(len(kb.data))]
print(f"laser line ({kb.name}): centre {tuple(round(v, 4) for v in kb.data[len(kb.data) // 2].co)}  z range {min(zs):.4f}..{max(zs):.4f}")


def render(cam_name, frame, path):
    sc.camera = CAMS[cam_name]
    sc.frame_set(frame)
    sc.render.filepath = path
    t = time.time()
    bpy.ops.render.render(write_still=True)
    dt = time.time() - t
    print(f"rendered {path} (frame {frame}) in {dt:.1f}s", flush=True)
    return dt


for k in args.hide:
    fx[k].hide_render = True
for shot in args.shots:
    if shot == "timing":
        dt1 = render("mid", 90, os.path.join(OUT, "vfx_timing_plume.png"))
        fx["smoke"].hide_render = True
        dt2 = render("mid", 90, os.path.join(OUT, "vfx_timing_noplume.png"))
        fx["smoke"].hide_render = False
        print(f"plume cost at {args.res[0]}x{args.res[1]}: {dt1 - dt2:+.1f}s per frame")
        continue
    if shot == "index":
        # the finding-5 check: same view with the VFX collection excluded, then included
        fx["collection"].hide_render = True
        render("index", 118, os.path.join(OUT, "vfx_index_novfx.png"))
        fx["collection"].hide_render = False
    c, f = SHOTS[shot]
    render(c, f, os.path.join(OUT, f"vfx_{shot}{args.tag}.png"))
