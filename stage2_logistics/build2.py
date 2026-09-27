#!/usr/bin/env python3
"""Build the stage-2 scene (stage-1 welding cell + logistics zone + full-cycle choreography) and render it.

    python3 build2.py --still 1332 [--res 960 540] [--engine CYCLES]     one frame (scene frame number)
    python3 build2.py --stills 115,512,1332                             several frames
    python3 build2.py --preview [--step 6]                              every 6th frame of the edit, 640x360
    python3 build2.py --render [--worker 0/2] [--first F --last F]      all frames of the edit (edl.py), 1920x1080,
                                                                        resumable: existing PNGs are skipped
    python3 build2.py --shot S2_08_load --render                        only the frames of one shot
    python3 build2.py --shot S2_08_load                                 the middle frame of one shot (test still)
    python3 build2.py --sketch                                          layout sketches (top + 3/4) -> deliverables/
    python3 build2.py --save out/stage2.blend                           save the scene for inspection in Blender
    blender -b --python build2.py -- --render                           same options with the Blender app

EEVEE (BLENDER_EEVEE_NEXT) is the default engine: it matches the look of the stage-1 video that the edit splices in.
Frames are written as <outdir>/frame_NNNN.png with NNNN = scene frame; post/compose2.py assembles the video.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import tools  # noqa: E402  (demo_video on sys.path)
import bpy  # noqa: E402
import numpy as np  # noqa: E402

from cell import layout as L, geom as G, spool, positioner, robot_build, animation, environment, vfx  # noqa: E402
import layout2 as L2  # noqa: E402
import plan2  # noqa: E402

OUT = os.path.join(HERE, "out")
SHADOW_LIGHTS = {"env_light_key", "ArcLight", "env2_light_key"}


def _argv():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]


# ============================================================================ scene
def build_scene(with_env=True, with_vfx=True, verbose=True):
    t_all = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.fps = L.FPS
    sc.unit_settings.system = 'METRIC'
    sc.frame_start, sc.frame_end = 1, animation.N_FRAMES
    # ---- stage-1 machines + its choreography (bead progress etc. is computed on the stage-1 timeline)
    import fonts2
    fonts2.patch_stage1()                      # stage-1 paint markings with the shipped fonts (any OS)
    pos = positioner.build()
    sp = spool.build(name="spool")
    G.set_parent(sp["root"], pos["mount"], keep_world=False)
    rob = robot_build.build()
    t0 = time.time()
    anim1 = animation.build(sc, rob, pos, sp)
    if verbose:
        print(f"[build2] stage-1 choreography in {time.time() - t0:.1f}s", flush=True)
    # ---- plan
    t0 = time.time()
    P = plan2.solve(verbose=verbose)
    sc.frame_start, sc.frame_end = 1, P["n_frames"]
    if verbose:
        print(f"[build2] plan: {P['n_frames']} frames, weld offset {P['weld_offset']}, IK failures {len(P['ik_fail'])}"
              f" ({time.time() - t0:.1f}s)", flush=True)
    # ---- stage-2 equipment
    import parts, handler_robot, tack_robot, assembly_station, cassette, conveyor, marking_qc, storage, environment2
    t0 = time.time()
    sc.frame_set(1)
    bpy.context.view_layer.update()
    ang = {t["key"]: t["angle"] for t in P["tack_points"]}
    tacks_a = [ang.get(f"A{i}", a) for i, a in enumerate(L2.TACKS_A)]
    tacks_b = [ang.get(f"B{i}", b) for i, b in enumerate(L2.TACKS_B)]
    prt = parts.split(sp, tacks_a=tacks_a, tacks_b=tacks_b)      # tacks where the planner actually put them
    hnd = handler_robot.build()
    tck = tack_robot.build()
    stn = assembly_station.build()
    kit = cassette.build()
    conv = conveyor.build(n_carriers=1)
    qc = marking_qc.build()
    mark = marking_qc.build_mark(prt["pipe"])
    sto = storage.build()
    extra = _extra_parts(prt, sto)
    env = env2 = None
    if with_env:
        env = environment.build()
        environment.build_lighting(sc)
        env2 = environment2.build()
        environment2.build_lighting(sc)
    else:
        env2 = environment2.build()
    if verbose:
        print(f"[build2] equipment + environment in {time.time() - t0:.1f}s", flush=True)
    # ---- animation
    import animation2
    t0 = time.time()
    M = dict(pos=pos, robot=rob, spool=sp, parts=prt, extra=extra, handler=hnd, tack=tck, station=stn,
             conveyor=conv, qc=qc, mark=mark, storage=sto, env2=env2, kit=kit)
    animation2.apply(sc, P, M)
    if verbose:
        print(f"[build2] keyframes in {time.time() - t0:.1f}s", flush=True)
    # ---- effects
    if with_vfx:
        t0 = time.time()
        tack_iv = [tuple(iv) for iv in P["intervals"]["tack_arc"]]
        fx = vfx.build(tck["arc"], tack_iv)
        _limit_particle_caches(fx)
        # the stage-1 arc effects at their embedded frames: the last seam's sparks and fume are still in the air when
        # the edit cuts from the stage-1 video to the first "post" shot
        off = P["weld_offset"]
        fx1 = vfx.build(anim1["arc"], [(a + off, b + off) for a, b in anim1["weld_intervals"]])
        _limit_particle_caches(fx1)
        scan_iv = [tuple(iv) for iv in P["intervals"]["scan"]]
        if scan_iv:
            marking_qc.scan_line(qc, scan_iv)
        vfx.setup_compositor(sc)
        if verbose:
            print(f"[build2] vfx in {time.time() - t0:.1f}s", flush=True)
    import cameras2
    cams = cameras2.build(sc, P["events"])
    if verbose:
        npoly = sum(len(o.data.polygons) for o in bpy.data.objects if o.type == 'MESH')
        print(f"[build2] scene ready in {time.time() - t_all:.1f}s, {npoly} polygons", flush=True)
    sc.frame_set(1)
    return dict(scene=sc, plan=P, modules=M, cams=cams)


def _extra_parts(prt, sto):
    """Loose parts that stay in the kit cassette / pipe buffer, the next kit's flange (animated by the plan) and the
    finished spools already in the storage rack.  Returns {name: (root, pose)} for animation2 (pose = plan part name
    or a static 4x4)."""
    import kin as K
    import parts
    out = {}
    kit_a = parts.build_kit(prt, "kitA")      # the next kit: its flange is picked in the "post" part
    out["flange2"] = (kit_a["flange"], "flange2")
    out["elbow2"] = (kit_a["elbow"], K.planar_frame(*L2.KIT_ELBOWS[1]))
    out["pipe2"] = (kit_a["pipe"], K.planar_frame(*L2.pipe_buffer_frame(1)))
    for t, b in L2.STORAGE_FILLED:
        root = parts.build_finished(prt, f"fin_{t}{b}")
        out[f"fin_{t}{b}"] = (root, K.planar_frame(*L2.storage_frame(t, b)))
    return out


def _delete_tree(root):
    for ob in list(root.children_recursive) + [root]:
        bpy.data.objects.remove(ob, do_unlink=True)


def _limit_particle_caches(fx):
    """Restrict every spark system's point cache to its own interval (+ lifetime) so baking is cheap."""
    em = fx.get("spark_emitter")
    if em is None:
        return
    for ps in em.particle_systems:
        st = ps.settings
        pc = ps.point_cache
        pc.frame_start = max(1, int(st.frame_start) - 1)
        pc.frame_end = int(st.frame_end + st.lifetime * (1 + st.lifetime_random) + 3)


# ============================================================================ render settings
def setup_render(sc, engine, res, samples=None, quality="stage1", gpu=None):
    """Stage-1 look (build.setup_render) + shadows for the logistics key light.  quality='high' adds shadows from
    every hall light and screen-space ray tracing (slower, slightly different look from the stage-1 footage)."""
    sys.path.insert(0, tools.DEMO)
    import build as B1
    B1.setup_render(sc, engine, res, samples)
    if engine != 'CYCLES':
        for ob in bpy.data.objects:
            if ob.type == 'LIGHT' and (ob.name in SHADOW_LIGHTS or ob.name.startswith("ArcLight")):
                ob.data.use_shadow = True
                ob.data.shadow_maximum_resolution = 0.008 if ob.name.startswith("ArcLight") else max(ob.data.shadow_maximum_resolution, 0.015)
        if quality == "high":
            sc.eevee.use_raytracing = True
            sc.eevee.use_fast_gi = True
            sc.eevee.shadow_resolution_scale = 1.0
            for ob in bpy.data.objects:
                if ob.type == 'LIGHT':
                    ob.data.use_shadow = True
    elif gpu:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = gpu
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type == gpu
        sc.cycles.device = 'GPU'


def _stamp_plan(outdir, keep_stale=False):
    """Frames of one render folder must come from one plan: write plan2's source hash into <outdir>/.plan_id and
    refuse to mix frames of another plan (resumed renders would otherwise silently keep stale frames)."""
    os.makedirs(outdir, exist_ok=True)
    stamp = os.path.join(outdir, ".plan_id")
    pid = plan2._IMPORT_HASH
    if os.path.exists(stamp):
        old = open(stamp, encoding="utf-8").read().strip()
        has_frames = any(n.startswith("frame_") for n in os.listdir(outdir))
        if old != pid and has_frames and not keep_stale:
            raise SystemExit(f"[build2] {outdir} holds frames of another plan ({old}, now {pid}): the choreography or"
                             f" the edit changed. Move/delete those frames (or use a new --outdir), or pass --keep-stale.")
    with open(stamp, "w", encoding="utf-8") as fh:
        fh.write(pid + "\n")


def render_frames(sc, frames, outdir, skip_existing=True, bake=True):
    os.makedirs(outdir, exist_ok=True)
    todo = [f for f in frames if not (skip_existing and os.path.exists(os.path.join(outdir, f"frame_{f:04d}.png")))]
    print(f"[render] {len(todo)} of {len(frames)} frames to render -> {outdir}", flush=True)
    if not todo:
        return
    if bake:
        t0 = time.time()
        vfx.bake_particles(sc)
        print(f"[render] particles baked in {time.time() - t0:.1f}s", flush=True)
    t_all = time.time()
    for i, f in enumerate(todo):
        sc.frame_set(f)
        sc.render.filepath = os.path.join(outdir, f"frame_{f:04d}.png")
        t0 = time.time()
        bpy.ops.render.render(write_still=True)
        if i == 0:
            _report_gpu(sc)
        print(f"[render] frame {f} ({i + 1}/{len(todo)}) {time.time() - t0:.1f}s", flush=True)
    print(f"[render] done {len(todo)} frames in {(time.time() - t_all) / 60:.1f} min", flush=True)


def _report_gpu(sc):
    try:
        import gpu
        print(f"[render] engine {sc.render.engine}, OpenGL/Vulkan renderer: {gpu.platform.renderer_get()} "
              f"({gpu.platform.vendor_get()})", flush=True)
    except Exception as e:  # background Cycles has no GPU module context
        print(f"[render] engine {sc.render.engine} (gpu info unavailable: {e})", flush=True)


# ============================================================================ sketches
def sketches(sc, outdir, engine, samples):
    """Layout sketches at frame 1: orthographic top view and a 3/4 view of the whole area."""
    import mathutils
    os.makedirs(outdir, exist_ok=True)
    for m in list(sc.timeline_markers):
        sc.timeline_markers.remove(m)
    views = [
        ("layout_top", (-4.0, 0.0, 20.0), (-4.0, 0.0, 0.0), dict(ortho=17.5)),
        ("layout_34", (-16.0, -11.0, 10.5), (-4.2, 0.2, 0.4), dict(lens=28)),
    ]
    paths = []
    for name, loc, aim, opt in views:
        cd = bpy.data.cameras.new(name)
        if "ortho" in opt:
            cd.type = 'ORTHO'
            cd.ortho_scale = opt["ortho"]
        else:
            cd.lens = opt["lens"]
        cd.clip_end = 200
        cam = bpy.data.objects.new(name, cd)
        sc.collection.objects.link(cam)
        cam.location = loc
        d = mathutils.Vector(aim) - mathutils.Vector(loc)
        cam.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
        if "ortho" in opt:
            cam.rotation_euler = (0.0, 0.0, 0.0)
        sc.camera = cam
        sc.frame_set(1)
        p = os.path.join(outdir, name + ".png")
        sc.render.filepath = p
        bpy.ops.render.render(write_still=True)
        paths.append(p)
        print("[sketch]", p, flush=True)
    return paths


def _check_edl(E):
    """The montage (post/compose2.py) reads post/edl.json: warn loudly if this scene's edit differs from it."""
    import json
    import edl as EDL
    try:
        saved = json.load(open(EDL.EDL_JSON, encoding="utf-8"))
    except OSError:
        print("[build2] WARNING: post/edl.json missing - run: python stage2_logistics/edl.py", flush=True)
        return
    sig = lambda e: [(s["src"], s.get("shot"), s["f0"], s["f1"]) for s in e["segments"]]
    if sig(saved) != sig(E):
        print("[build2] WARNING: the edit of this scene differs from post/edl.json (code or plan changed).\n"
              "          Run  python stage2_logistics/edl.py  before composing, or the frames will not match.", flush=True)


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--still", type=int)
    ap.add_argument("--stills", type=str)
    ap.add_argument("--preview", action="store_true", help="every --step-th frame of the edit at 640x360")
    ap.add_argument("--render", action="store_true", help="every frame of the edit (edl.py) at 1920x1080")
    ap.add_argument("--shot", type=str, help="restrict --preview/--render to one shot (cameras2 name); alone: its middle frame")
    ap.add_argument("--first", type=int, default=None, help="restrict to scene frames >= first")
    ap.add_argument("--last", type=int, default=None, help="restrict to scene frames <= last")
    ap.add_argument("--worker", type=str, default=None, help="k/K: render the k-th of K contiguous blocks")
    ap.add_argument("--step", type=int, default=6)
    ap.add_argument("--res", type=int, nargs=2, default=None)
    ap.add_argument("--engine", default="BLENDER_EEVEE_NEXT", help="BLENDER_EEVEE_NEXT (default, stage-1 look) or CYCLES")
    ap.add_argument("--gpu", default=None, help="Cycles only: OPTIX | CUDA | HIP | METAL | ONEAPI")
    ap.add_argument("--samples", type=int, default=None, help="EEVEE TAA samples (stage 1: 8; 16-32 on a GPU)")
    ap.add_argument("--quality", default="stage1", choices=("stage1", "high"))
    ap.add_argument("--no-env", action="store_true")
    ap.add_argument("--no-vfx", action="store_true")
    ap.add_argument("--fx-off", type=str, default="", help="as demo_video/build.py --fx-off (profiling)")
    ap.add_argument("--save", type=str)
    ap.add_argument("--sketch", action="store_true")
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--camera", type=str, help="force a camera (object name) for stills")
    ap.add_argument("--keep-stale", action="store_true", help="resume into a folder rendered for another plan (not advised)")
    a = ap.parse_args(_argv())

    S = build_scene(with_env=not a.no_env, with_vfx=not a.no_vfx)
    sc = S["scene"]
    if a.save:
        # with the render look applied (engine, AgX, shadows, samples) so the .blend reproduces the video
        setup_render(sc, a.engine, tuple(a.res or (1920, 1080)), a.samples, a.quality, a.gpu)
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.save))
        print("saved", a.save)
    if a.sketch:
        setup_render(sc, a.engine, tuple(a.res or (1920, 1080)), a.samples or 16, a.quality, a.gpu)
        sketches(sc, a.outdir or os.path.join(HERE, "deliverables"), a.engine, a.samples)
        return

    import edl as EDL
    E = EDL.build_edl(S["plan"])
    _check_edl(E)
    frames = []
    if a.still is not None:
        frames = [a.still]
    elif a.stills:
        frames = [int(x) for x in a.stills.split(",") if x.strip()]
    elif a.shot and not (a.preview or a.render):
        seg = [s for s in E["segments"] if s["src"] == "s2" and s["shot"] == a.shot]
        if not seg:
            raise SystemExit(f"unknown shot {a.shot}; shots: {[s['shot'] for s in E['segments'] if s['src'] == 's2']}")
        frames = [(seg[0]["f0"] + seg[0]["f1"]) // 2]
    elif a.preview or a.render:
        segs = [s for s in E["segments"] if s["src"] == "s2" and (a.shot is None or s["shot"] == a.shot)]
        if a.preview:
            frames = sorted({f for s in segs for f in range(s["f0"], s["f1"] + 1, a.step)} | {s["f1"] for s in segs})
        else:
            frames = sorted({f for s in segs for f in range(s["f0"], s["f1"] + 1)})
    if a.first is not None:
        frames = [f for f in frames if f >= a.first]
    if a.last is not None:
        frames = [f for f in frames if f <= a.last]
    if a.worker:
        k, K_ = (int(v) for v in a.worker.split("/"))
        per = (len(frames) + K_ - 1) // K_
        frames = frames[k * per:(k + 1) * per]
    if not frames:
        print("[build2] nothing to render (use --still/--stills/--preview/--render/--sketch)")
        return
    res = tuple(a.res) if a.res else ((640, 360) if a.preview else (1920, 1080) if a.render else (960, 540))
    setup_render(sc, a.engine, res, a.samples, a.quality, a.gpu)
    sys.path.insert(0, tools.DEMO)
    import build as B1
    B1.apply_fx_off(sc, a.fx_off)
    if a.camera:
        for m in list(sc.timeline_markers):
            sc.timeline_markers.remove(m)
        sc.camera = bpy.data.objects[a.camera]
    outdir = a.outdir or os.path.join(OUT, "frames" if a.render else "preview_frames" if a.preview else "stills")
    if a.render or a.preview:
        _stamp_plan(outdir, a.keep_stale)
    render_frames(sc, frames, outdir, skip_existing=bool(a.render or a.preview), bake=not a.no_vfx)


if __name__ == "__main__":
    main()
