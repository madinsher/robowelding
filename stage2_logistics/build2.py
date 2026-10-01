#!/usr/bin/env python3
"""Build the stage-2 scene (stage-1 welding cell + logistics zone + full-cycle choreography) and render it.

Every command takes --edition ru|en (editions.py, default ru): the language of the texts drawn in the scene (i18n2),
the logo (branding2), the edit (post/edl_<ed>.json, written by edl.py) and the default output folders
(out/frames_<ed>, out/preview_<ed>, out/stills_<ed>).
    Both editions render all 2152 frames of the edit: the finished stage-1 video has its own titles burned in, so
    the welding part is rendered from this scene with the stage-1 cameras (cameras2.S1_SHOTS, scene frame = stage-1
    frame + WELD_OFFSET).  (editions.py stage1="video" would splice the stage-1 video instead: 1402 frames.)

    python3 build2.py --still 1332 [--res 960 540] [--engine CYCLES]     one frame (scene frame number)
    python3 build2.py --stills 115,512,1332                             several frames
    python3 build2.py --preview [--step 6]                              every 6th frame of the edit, 640x360
    python3 build2.py --render [--worker 0/2] [--first F --last F]      all frames of the edit (edl.py), 1920x1080,
                                                                        resumable: existing PNGs are skipped
    python3 build2.py --edition en --render                             the English edition (stage-1 shots included)
    python3 build2.py --shot S2_08_load --render                        only the frames of one shot
    python3 build2.py --shot S2_08_load                                 the middle frame of one shot (test still)
    python3 build2.py --edition en --shot S1_C3a_laser                  a stage-1 welding shot
    python3 build2.py --sketch                                          layout sketches (top + 3/4) -> deliverables/
                                                                        (en: layout_top_en.png, layout_34_en.png)
    python3 build2.py --save out/stage2.blend                           save the scene for inspection in Blender
    blender -b --python build2.py -- --render                           same options with the Blender app

EEVEE (BLENDER_EEVEE_NEXT) is the default engine: it matches the look of the stage-1 video (spliced into the ru edit,
re-rendered for the en one with the stage-1 render settings).
Frames are written as <outdir>/frame_NNNN.png with NNNN = scene frame; <outdir>/.plan_id = "<plan hash> <edition>"
keeps one folder from mixing frames of two plans or two editions.  post/compose2.py assembles the video.
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

from cell import layout as L, geom as G, spool, positioner, robot_build, animation, environment, vfx  # noqa: E402
import layout2 as L2  # noqa: E402
import plan2  # noqa: E402
import pos_clamps  # noqa: E402

OUT = os.path.join(HERE, "out")
SHADOW_LIGHTS = {"env_light_key", "ArcLight", "env2_light_key"}


def _argv():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]


# ============================================================================ scene
def build_scene(with_env=True, with_vfx=True, verbose=True, edition=None):
    """The whole stage-2 scene of an edition (editions.py name, None -> editions.DEFAULT).
    Returns dict(scene, plan, modules, cams, edition)."""
    import editions
    import i18n2
    import branding2
    ed = editions.get(edition)
    # language + logo first: every module below that draws a texture (signs, HMI screens, logo plates) reads them
    i18n2.set_lang(ed["lang"])
    branding2.set_brand(ed["brand"])
    if verbose:
        # ASCII only (not the Cyrillic ed["label"]): Windows encodes a redirected stdout (> render_ru.log, Tee-Object)
        # with the locale code page, and cp1252 cannot write Cyrillic - the ru build would abort on this print
        print(f"[build2] edition {ed['name']}: lang {ed['lang']}, brand {ed['brand']}, stage 1 {ed['stage1']}",
              flush=True)
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
    clamps = pos_clamps.build(pos)             # swing clamps on the faceplate: they hold the flange
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
    M = dict(pos=pos, clamps=clamps, robot=rob, spool=sp, parts=prt, extra=extra, handler=hnd, tack=tck, station=stn,
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
        # the stage-1 effects at their embedded frames (scene = stage-1 frame + WELD_OFFSET): the en edition renders
        # the stage-1 shots from this scene, and in both editions the last seam's sparks and fume are still in the
        # air when the edit cuts to the first "post" shot.  The arc is built on the stage-1 frames, then its keys
        # move by WELD_OFFSET: its flicker (10 Hz sinusoid of the absolute frame + seeded noise) then repeats the
        # stage-1 video frame by frame (built on the shifted frames it runs 240 degrees out of phase).
        off = P["weld_offset"]
        fx1 = vfx.build(anim1["arc"], anim1["weld_intervals"])
        _shift_arc_fx(fx1, off)
        _limit_particle_caches(fx1)
        # seam-tracking laser of the stage-1 scan (shot C3a): the fan is ray cast onto the scene at every scan frame,
        # so it is built on the embedded frames, after the keyframes have put the loaded spool on the positioner
        vfx.laser_line(rob["torch"]["sensor"], [(a + off, b + off) for a, b in anim1["laser_intervals"]])
        scan_iv = [tuple(iv) for iv in P["intervals"]["scan"]]
        if scan_iv:
            marking_qc.scan_line(qc, scan_iv)
        vfx.setup_compositor(sc)
        if verbose:
            print(f"[build2] vfx in {time.time() - t0:.1f}s", flush=True)
    import cameras2
    # every shot: the stage-1 shots (S1_*, event "weld_offset") are in the edit of the "render" editions; a "video"
    # edition splices the stage-1 video over their frames, so their markers are never reached by its renders
    cams = cameras2.build(sc, cameras2.events_of(P), shots=cameras2.ALL_SHOTS)
    if verbose:
        npoly = sum(len(o.data.polygons) for o in bpy.data.objects if o.type == 'MESH')
        print(f"[build2] scene ready in {time.time() - t_all:.1f}s, {npoly} polygons", flush=True)
    sc.frame_set(1)
    return dict(scene=sc, plan=P, modules=M, cams=cams, edition=ed)


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


def _shift_arc_fx(fx, off):
    """Move an arc effect built by vfx.build on the stage-1 timeline by `off` frames: the keys of the light energy,
    the core / glow scale and the fume fade, and the emission range of every spark / droplet system."""
    import animation2
    for idb in (fx["light"].data, fx["core"], fx["glow"], fx["smoke"]):
        animation2.shift_keys(idb, off)
    for ps in fx["spark_emitter"].particle_systems:
        st = ps.settings
        st.frame_end = st.frame_end + off          # end first: setting a start beyond the end drags the end along
        st.frame_start = st.frame_start + off


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


def _stamp_plan(outdir, edition, keep_stale=False):
    """Frames of one render folder must come from one plan and one edition: write "<plan2 source hash> <edition>"
    into <outdir>/.plan_id and refuse to add frames of another plan (a resumed render would silently keep stale
    frames) or of another edition (the texts and logos of the other language) to a folder that holds frames.
    --keep-stale overrides a plan mismatch only.  A stamp without an edition predates the editions: Russian."""
    import editions
    os.makedirs(outdir, exist_ok=True)
    stamp = os.path.join(outdir, ".plan_id")
    pid, ed = plan2._IMPORT_HASH, editions.get(edition)["name"]
    if os.path.exists(stamp):
        tok = open(stamp, encoding="utf-8").read().split()
        old_pid = tok[0] if tok else ""
        old_ed = tok[1] if len(tok) > 1 else "ru"          # stamped before the editions existed
        has_frames = any(n.startswith("frame_") for n in os.listdir(outdir))
        if old_ed != ed and has_frames:
            raise SystemExit(f"[build2] {outdir} holds frames of the {old_ed!r} edition, this run renders {ed!r}: "
                             f"use the {ed} folders (default out/frames_{ed}, out/preview_{ed}) or another --outdir.")
        if old_pid != pid and has_frames and not keep_stale:
            raise SystemExit(f"[build2] {outdir} holds frames of another plan ({old_pid}, now {pid}): the choreography"
                             f" or the edit changed. Move/delete those frames (or use a new --outdir), or pass"
                             f" --keep-stale.")
    with open(stamp, "w", encoding="utf-8") as fh:
        fh.write(f"{pid} {ed}\n")


def render_frames(sc, frames, outdir, skip_existing=True, bake=True):
    """Render `frames` to <outdir>/frame_NNNN.png.  bake=True (every run with effects) bakes the spark caches first:
    besides the sparks themselves this keeps the robots on their keyframes - an unbaked spark system that resimulates
    after a frame jump re-evaluates its emitter's parent chain (the welding / tack robot carrying the torch) at the
    particles' birth times and leaves it there (Blender's evaluate_emitter_anim; measured on this scene: 6 of 150
    random jumps into the arcs put the welding robot up to 0.24 m off its keys)."""
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
def _overhead_objects(sc, zmin=2.8):
    """Static meshes hanging above the floor equipment (fume hood, duct, hall lamps): hidden in the top view."""
    import mathutils
    out = []
    for ob in sc.objects:
        if ob.type != 'MESH' or ob.hide_render or ob.animation_data or (ob.parent and ob.parent.animation_data):
            continue
        z = min((ob.matrix_world @ mathutils.Vector(c)).z for c in ob.bound_box)
        if z > zmin:
            out.append(ob)
    return out


def sketches(sc, outdir, engine, samples, suffix=""):
    """Layout sketches at frame 1: orthographic top view and a 3/4 view of the whole area.  The hall roof deck is at
    9 m (trusses from 8 m): the top view clips everything above 7.4 m and hides the overhead equipment, the 3/4 camera
    stands inside the hall under the trusses.  Files: <outdir>/layout_top<suffix>.png, layout_34<suffix>.png."""
    import mathutils
    os.makedirs(outdir, exist_ok=True)
    for m in list(sc.timeline_markers):
        sc.timeline_markers.remove(m)
    sc.frame_set(1)
    views = [
        ("layout_top", (-4.0, 0.0, 20.0), (-4.0, 0.0, 0.0), dict(ortho=17.5, clip_start=20.0 - 7.4)),
        ("layout_34", (-15.5, -11.5, 7.4), (-4.4, 0.2, 0.4), dict(lens=22)),
    ]
    paths = []
    for name, loc, aim, opt in views:
        cd = bpy.data.cameras.new(name)
        if "ortho" in opt:
            cd.type = 'ORTHO'
            cd.ortho_scale = opt["ortho"]
        else:
            cd.lens = opt["lens"]
        cd.clip_start = opt.get("clip_start", 0.1)
        cd.clip_end = 200
        cam = bpy.data.objects.new(name, cd)
        sc.collection.objects.link(cam)
        cam.location = loc
        d = mathutils.Vector(aim) - mathutils.Vector(loc)
        cam.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
        hidden = []
        if "ortho" in opt:
            cam.rotation_euler = (0.0, 0.0, 0.0)
            hidden = _overhead_objects(sc)
            print("[sketch] hidden overhead objects:", ", ".join(sorted(o.name for o in hidden)), flush=True)
        for ob in hidden:
            ob.hide_render = True
        sc.camera = cam
        sc.frame_set(1)
        p = os.path.join(outdir, name + suffix + ".png")
        sc.render.filepath = p
        bpy.ops.render.render(write_still=True)
        for ob in hidden:
            ob.hide_render = False
        paths.append(p)
        print("[sketch]", p, flush=True)
    return paths


def _check_edl(E):
    """The montage (post/compose2.py) reads the edition's post/edl_<ed>.json: warn loudly if this scene's edit (E,
    edl.build_edl of the same edition) differs from it."""
    import json
    import editions
    ed = editions.get(E["edition"])
    path = ed["edl_json"]
    rel = os.path.relpath(path, os.path.dirname(HERE))
    cmd = f"python stage2_logistics/edl.py --edition {ed['name']}"
    try:
        saved = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        print(f"[build2] WARNING: {rel} missing or unreadable - run:  {cmd}", flush=True)
        return
    sig = lambda e: (e.get("edition", "ru"), [(s["src"], s.get("shot"), s["f0"], s["f1"]) for s in e["segments"]])
    if sig(saved) != sig(E):
        print(f"[build2] WARNING: the edit of this scene differs from {rel} (code or plan changed).\n"
              f"          Run  {cmd}  before composing, or the frames will not match.", flush=True)


def _edit_checks(E, a, ed):
    """The requested --shot must be a rendered segment of the edition's edit (S1_* only where the stage-1 shots are
    rendered); the saved post/edl_<ed>.json must match E (not for --sketch).  Returns the edit's s2 segments."""
    import editions
    segs = [s for s in E["segments"] if s["src"] == "s2"]
    if a.shot and a.shot not in {s["shot"] for s in segs}:
        why = (f" (the stage-1 shots are spliced from the stage-1 video in the {ed['name']} edition, rendered only in"
               f" {', '.join(n for n in editions.names() if editions.get(n)['stage1'] == 'render')})"
               if a.shot.startswith("S1_") and ed["stage1"] == "video" else "")
        raise SystemExit(f"unknown shot {a.shot} in the {ed['name']} edit{why}; shots: {[s['shot'] for s in segs]}")
    if not a.sketch:
        _check_edl(E)
    return segs


# ============================================================================ main
def main():
    import editions
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    editions.add_argument(ap)
    ap.add_argument("--still", type=int, help="render one scene frame (default outdir out/stills_<edition>)")
    ap.add_argument("--stills", type=str, help="comma-separated scene frames")
    ap.add_argument("--preview", action="store_true", help="every --step-th frame of the edit at 640x360 "
                    "(default outdir out/preview_<edition>)")
    ap.add_argument("--render", action="store_true", help="every frame of the edition's edit (post/edl_<edition>.json)"
                    " at 1920x1080 (default outdir out/frames_<edition>): 2152 frames per edition")
    ap.add_argument("--shot", type=str, help="restrict --preview/--render to one shot of the edition's edit (cameras2 "
                    "name; S1_* = the stage-1 welding shots); alone: its middle frame")
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
    ap.add_argument("--sketch", action="store_true", help="layout sketches -> deliverables/ (en: *_en.png)")
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--camera", type=str, help="force a camera (object name) for stills")
    ap.add_argument("--keep-stale", action="store_true", help="resume into a folder rendered for another plan (not "
                    "advised; frames of another edition are always refused)")
    a = ap.parse_args(_argv())
    ed = editions.get(a.edition)
    # the edit needs only the plan: check it and the requested shot before the minutes-long scene build.  A plan that
    # must be solved again (no plan2_<hash>.npz shipped or cached for these sources) needs the stage-1 trajectory
    # cache, which only the scene build writes (animation.build, the first run on a fresh machine): then the checks
    # wait for the scene
    import edl as EDL
    try:
        P0 = plan2.solve(verbose=True)
    except RuntimeError:                       # plan2.stage1_arrays: no stage-1 trajectory cache
        print("[build2] no plan for these sources and no stage-1 trajectory cache yet: the scene build writes the"
              " cache and solves the plan, the edit is checked after it", flush=True)
        P0 = None
    E = EDL.build_edl(P0, ed["name"]) if P0 is not None else None
    segs = _edit_checks(E, a, ed) if E is not None else None

    S = build_scene(with_env=not a.no_env, with_vfx=not a.no_vfx, edition=ed["name"])
    sc = S["scene"]
    if E is None:
        E = EDL.build_edl(S["plan"], ed["name"])
        segs = _edit_checks(E, a, ed)
    if a.save:
        # with the render look applied (engine, AgX, shadows, samples) so the .blend reproduces the video
        setup_render(sc, a.engine, tuple(a.res or (1920, 1080)), a.samples, a.quality, a.gpu)
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.save))
        print("saved", a.save)
    if a.sketch:
        setup_render(sc, a.engine, tuple(a.res or (1920, 1080)), a.samples or 16, a.quality, a.gpu)
        suffix = "" if ed["name"] == editions.DEFAULT else "_" + ed["name"]
        sketches(sc, a.outdir or os.path.join(HERE, "deliverables"), a.engine, a.samples, suffix)
        return

    frames = []
    if a.still is not None:
        frames = [a.still]
    elif a.stills:
        frames = [int(x) for x in a.stills.split(",") if x.strip()]
    elif a.shot and not (a.preview or a.render):
        seg = [s for s in segs if s["shot"] == a.shot]
        frames = [(seg[0]["f0"] + seg[0]["f1"]) // 2]
    elif a.preview or a.render:
        sel = [s for s in segs if a.shot is None or s["shot"] == a.shot]
        if a.preview:
            frames = sorted({f for s in sel for f in range(s["f0"], s["f1"] + 1, a.step)} | {s["f1"] for s in sel})
        else:
            frames = sorted({f for s in sel for f in range(s["f0"], s["f1"] + 1)})
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
    outdir = a.outdir or (ed["frames_dir"] if a.render else ed["preview_dir"] if a.preview else ed["stills_dir"])
    if a.render or a.preview:
        _stamp_plan(outdir, ed["name"], a.keep_stale)
    render_frames(sc, frames, outdir, skip_existing=bool(a.render or a.preview), bake=not a.no_vfx)


if __name__ == "__main__":
    main()
