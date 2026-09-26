#!/usr/bin/env python3
"""Build the welding-cell scene and render.

    python3 build.py --still 300 [--res 960 540] [--engine CYCLES]
    python3 build.py --preview           (every 8th frame, 640x360, EEVEE via xvfb-run)
    xvfb-run -a python3 build.py --render [--start 1 --end 1008] [--res 1920 1080]
    python3 build.py --export-stl robodk/spool.stl
    python3 build.py --save out/cell.blend
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bpy  # noqa: E402
from cell import layout as L, geom as G, materials, spool, positioner, robot_build, animation  # noqa: E402

try:
    from cell import environment
except Exception as e:  # module may still be under construction
    environment = None
    print("[build] environment module unavailable:", e)
try:
    from cell import vfx
except Exception as e:
    vfx = None
    print("[build] vfx module unavailable:", e)


ARC_LIGHT_POWER = 90.0      # W; the hall is lit with a few hundred W, 500 W bleached the part around the arc


def build_scene(with_env=True, with_vfx=True):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.fps = L.FPS
    sc.frame_start, sc.frame_end = 1, animation.N_FRAMES
    sc.unit_settings.system = 'METRIC'
    t0 = time.time()
    pos = positioner.build()
    sp = spool.build()
    G.set_parent(sp["root"], pos["mount"], keep_world=False)
    rob = robot_build.build()
    print(f"[build] machines built in {time.time() - t0:.1f}s")
    t0 = time.time()
    anim = animation.build(sc, rob, pos, sp)
    if animation.IK_FAILS:
        print(f"[build] WARNING: {len(animation.IK_FAILS)} IK failures, first: {animation.IK_FAILS[:5]}")
    cams = animation.build_cameras(sc)
    print(f"[build] animation solved in {time.time() - t0:.1f}s")
    env = None
    if with_env and environment is not None:
        t0 = time.time()
        env = environment.build()
        environment.build_lighting(sc)
        print(f"[build] environment built in {time.time() - t0:.1f}s")
    else:
        _fallback_lighting(sc)
    fx = None
    if with_vfx and vfx is not None:
        t0 = time.time()
        vfx.ARC_LIGHT_POWER = ARC_LIGHT_POWER
        fx = vfx.build(anim["arc"], anim["weld_intervals"])
        vfx.laser_line(rob["torch"]["sensor"], anim["laser_intervals"])
        vfx.setup_compositor(sc)
        print(f"[build] vfx built in {time.time() - t0:.1f}s")
    ntris = sum(len(o.data.polygons) for o in bpy.data.objects if o.type == 'MESH')
    print(f"[build] mesh polygons: {ntris}")
    return dict(scene=sc, pos=pos, spool=sp, robot=rob, anim=anim, cams=cams, env=env, vfx=fx)


def _fallback_lighting(sc):
    bpy.ops.mesh.primitive_plane_add(size=40)
    bpy.context.object.data.materials.append(materials.get("concrete"))
    bpy.ops.object.light_add(type='SUN', location=(3, 2, 8), rotation=(0.5, 0.4, 0.2)); bpy.context.object.data.energy = 3
    for loc in ((3, -4, 6), (-3, 3, 6), (4, 3, 5)):
        bpy.ops.object.light_add(type='AREA', location=loc); bpy.context.object.data.energy = 3000; bpy.context.object.data.size = 4
    w = bpy.data.worlds.new("W"); sc.world = w; w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (0.25, 0.28, 0.33, 1)


def setup_render(sc, engine, res, samples=None):
    sc.render.engine = engine
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_mode = 'RGB'
    sc.render.film_transparent = False
    sc.view_settings.view_transform = 'AgX'
    sc.view_settings.look = 'AgX - Medium High Contrast'
    if engine == 'CYCLES':
        sc.cycles.device = 'CPU'
        sc.cycles.samples = samples or 24
        sc.cycles.use_denoising = True
        sc.cycles.denoiser = 'OPENIMAGEDENOISE'
        sc.cycles.max_bounces = 4
        sc.render.use_persistent_data = True
    else:
        sc.eevee.taa_render_samples = samples or 16
        sc.eevee.use_shadows = True
        sc.eevee.use_raytracing = True
        sc.eevee.ray_tracing_options.resolution_scale = '2'
        sc.eevee.shadow_ray_count = 2
        sc.eevee.shadow_step_count = 4
        sc.eevee.use_volumetric_shadows = False
        sc.eevee.volumetric_tile_size = '16'


def apply_fx_off(sc, spec):
    """Selectively disable expensive features (profiling / low-quality previews)."""
    offs = {x.strip() for x in spec.split(",") if x.strip()}
    if not offs:
        return
    if "raytracing" in offs:
        sc.eevee.use_raytracing = False
    if "fastgi" in offs:
        sc.eevee.use_fast_gi = False
    if "shadows" in offs:
        sc.eevee.use_shadows = False
    if "bayshadows" in offs:      # shadows only from the key light and the arc
        keep = {"env_light_key", "ArcLight"}
        for ob in bpy.data.objects:
            if ob.type == 'LIGHT' and ob.name not in keep:
                ob.data.use_shadow = False
    if "softshadow" in offs:
        for lt in bpy.data.lights:
            lt.shadow_soft_size = 0.0
    if "volumetric" in offs:
        for ob in bpy.data.objects:
            if ob.type == 'MESH' and any(m and m.node_tree and any(n.type in ('VOLUME_PRINCIPLED', 'VOLUME_SCATTER', 'VOLUME_ABSORPTION') for n in m.node_tree.nodes) for m in ob.data.materials):
                ob.hide_render = True
    if "dof" in offs:
        for cd in bpy.data.cameras:
            cd.dof.use_dof = False
    if "glare" in offs:
        sc.use_nodes = False
    if "particles" in offs:
        for ob in bpy.data.objects:
            for mod in ob.modifiers:
                if mod.type == 'PARTICLE_SYSTEM':
                    mod.show_render = False
    print("[build] fx off:", sorted(offs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--still", type=int, help="render a single frame")
    ap.add_argument("--stills", type=str, help="comma-separated frames to render")
    ap.add_argument("--preview", action="store_true", help="sparse low-res preview (every 8th frame)")
    ap.add_argument("--render", action="store_true", help="full frame sequence render")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=animation.N_FRAMES)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--res", type=int, nargs=2, default=None)
    ap.add_argument("--engine", default="BLENDER_EEVEE_NEXT")
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--no-env", action="store_true")
    ap.add_argument("--no-vfx", action="store_true")
    ap.add_argument("--export-stl", type=str)
    ap.add_argument("--save", type=str)
    ap.add_argument("--outdir", type=str, default=os.path.join(HERE, "out", "frames"))
    ap.add_argument("--camera", type=str, help="force a camera (name) for stills")
    ap.add_argument("--fx-off", type=str, default="", help="comma list: raytracing,fastgi,volumetric,dof,glare,shadows,particles,softshadow")
    a = ap.parse_args()

    if a.export_stl:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        sp = spool.build()
        spool.export_stl(sp, os.path.abspath(a.export_stl))
        print("exported", a.export_stl)
        return

    S = build_scene(with_env=not a.no_env, with_vfx=not a.no_vfx)
    sc = S["scene"]
    if a.save:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.save))
        print("saved", a.save)

    res = tuple(a.res) if a.res else (L.RES_FINAL if a.render else L.RES_PREVIEW)
    if a.preview:
        res = tuple(a.res) if a.res else (640, 360)
        a.step = a.step if a.step != 1 else 8
    setup_render(sc, a.engine, res, a.samples)
    apply_fx_off(sc, a.fx_off)
    os.makedirs(a.outdir, exist_ok=True)

    frames = []
    if a.still is not None:
        frames = [a.still]
    elif a.stills:
        frames = [int(x) for x in a.stills.split(",")]
    elif a.preview or a.render:
        frames = list(range(a.start, a.end + 1, a.step))
    if a.camera:
        for m in list(sc.timeline_markers):
            sc.timeline_markers.remove(m)
        sc.camera = bpy.data.objects[a.camera]
    if frames and vfx is not None and not a.no_vfx and (len(frames) == 1 or frames[0] > 1 or a.step > 1) and hasattr(vfx, "bake_particles"):
        t0 = time.time()
        vfx.bake_particles(sc)
        print(f"[build] particles baked in {time.time() - t0:.1f}s")
    t_all = time.time()
    for i, f in enumerate(frames):
        sc.frame_set(f)
        sc.render.filepath = os.path.join(a.outdir, f"frame_{f:04d}.png")
        if os.path.exists(sc.render.filepath) and a.render:
            continue
        t0 = time.time()
        bpy.ops.render.render(write_still=True)
        print(f"[render] frame {f} ({i + 1}/{len(frames)}) {time.time() - t0:.1f}s -> {sc.render.filepath}", flush=True)
    if frames:
        print(f"[render] done {len(frames)} frames in {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
