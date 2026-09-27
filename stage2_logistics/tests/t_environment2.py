"""Check environment2.py together with the stage-1 environment: contract, no overlap with stage-1 geometry (fence posts,
bays, light curtains, hazard stripes, door rail), nothing inside the layout2 equipment footprints / handler corridor /
grasp corridors, clearance against the other modules' obstacles(), obstacles() coverage, set_muting keys, light setup;
optional exposure check (top-down floor luminance, stage-1 look, env2 lights off / on); renders Cycles stills (high 3/4
overview of the whole zone, view from inside the zone toward the cell opening with the muting lamps on, walkway with
the cabinet / HMI / personnel door, close-up of a muting post).

    cd /home/user/robowelding && python3 stage2_logistics/tests/t_environment2.py [--no-render]
    cd /home/user/robowelding && xvfb-run -a python3 stage2_logistics/tests/t_environment2.py --no-render --exposure --eevee
"""
import importlib
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import harness as H  # noqa: E402
import bpy  # noqa: E402
import numpy as np  # noqa: E402
import layout2 as L2  # noqa: E402
import kin  # noqa: E402
import environment2 as E2  # noqa: E402
from cell import environment as E1  # noqa: E402
from t_cassette import FAIL, check, world_mesh, bvh, aabb  # noqa: E402

TOL = 1e-3


def boxes_overlap(lo_a, hi_a, lo_b, hi_b, tol=TOL):
    return all(lo_a[k] < hi_b[k] - tol and lo_b[k] < hi_a[k] - tol for k in range(3))


def box_lohi(o):
    c, s = o["center"], o["size"]
    return [c[k] - s[k] / 2 for k in range(3)], [c[k] + s[k] / 2 for k in range(3)]


def footprints():
    """Equipment footprints from layout2 (world boxes, z up to 3 m): nothing of environment2 may enter them."""
    out = []
    st = L2.STATION_TABLE
    out.append(("station_table", (st["center"][0] - st["size"][0] / 2, st["center"][1] - st["size"][1] / 2, 0.0),
                (st["center"][0] + st["size"][0] / 2, st["center"][1] + st["size"][1] / 2, 3.0)))
    kp = L2.KIT_PALLET
    out.append(("kit_pallet", (kp["center"][0] - kp["size"][0] / 2, kp["center"][1] - kp["size"][1] / 2, 0.0),
                (kp["center"][0] + kp["size"][0] / 2, kp["center"][1] + kp["size"][1] / 2, 3.0)))
    pb = L2.PIPE_BUFFER
    out.append(("pipe_buffer", (pb["center"][0] - pb["size"][0] / 2, pb["center"][1] - pb["size"][1] / 2, 0.0),
                (pb["center"][0] + pb["size"][0] / 2, pb["center"][1] + pb["size"][1] / 2, 3.0)))
    cv = L2.CONVEYOR
    out.append(("conveyor", (cv["x0"], cv["y"] - cv["width"] / 2, 0.0), (cv["x1"], cv["y"] + cv["width"] / 2, 3.0)))
    qc = L2.QC_ARCH
    out.append(("qc_arch", (qc["x"] - 0.35, qc["y0"] - 0.12, 0.0), (qc["x"] + 0.35, qc["y1"] + 0.12, 3.0)))
    # the rack stays within the storage_back opening width (+ margin); at the -X end it starts past the 0.2 m base plates
    # of the fence-line light-curtain columns (storage.BACK_X keeps its back rail 1 cm off them)
    hy = E2.STORAGE_HALF_Y
    out.append(("storage", (L2.LOG_FENCE_X0 + 0.10, -hy, 0.0), (E2.STORAGE_FRONT_X, hy, 3.0)))
    out.append(("handler_bed", (L2.HANDLER_TRACK_BED[0], -0.75, 0.0), (L2.HANDLER_TRACK_BED[1], 0.75, 3.5)))
    tx, ty = L2.TACK_BASE
    out.append(("tack_pedestal", (tx - 0.35, ty - 0.35, 0.0), (tx + 0.35, ty + 0.35, 3.0)))
    return out


def grasp_points():
    """World grasp points of every place where the handler grips something (station, kit, buffer, storage, conveyor)."""
    pts = []
    Fst = kin.planar_frame(L2.STATION_ORIGIN, L2.STATION_XDIR)
    for key in ("flange", "elbow", "pipe", "spool"):
        pts.append(("station_" + key, Fst @ np.array([*L2.GRASP[key][0], 1.0])))
    for i, (o, xd) in enumerate(L2.KIT_FLANGES):
        pts.append((f"kit_flange{i}", kin.planar_frame(o, xd) @ np.array([*L2.GRASP["flange"][0], 1.0])))
    for i, (o, xd) in enumerate(L2.KIT_ELBOWS):
        pts.append((f"kit_elbow{i}", kin.planar_frame(o, xd) @ np.array([*L2.GRASP["elbow"][0], 1.0])))
    for i in range(len(L2.PIPE_BUFFER_X)):
        pts.append((f"buffer{i}", kin.planar_frame(*L2.pipe_buffer_frame(i)) @ np.array([*L2.GRASP["pipe"][0], 1.0])))
    for t in (0, 1):
        for i in range(len(L2.STORAGE["bay_y"][t])):
            pts.append((f"bay{t}{i}", kin.planar_frame(*L2.storage_frame(t, i)) @ np.array([*L2.GRASP["spool"][0], 1.0])))
    for x in (L2.CONV_LOAD_X, L2.CONV_QC_X, L2.CONV_END_X):
        F = kin.planar_frame((x, L2.CONV_SPOOL_Y, L2.CARRIER["seat_z"]), (0.0, 1.0))
        pts.append((f"conv{x}", F @ np.array([*L2.GRASP["spool"][0], 1.0])))
    return pts


def main():
    render = "--no-render" not in sys.argv
    H.new_scene(floor=False, lights=False)
    sc = bpy.context.scene
    env1 = E1.build()
    E1.build_lighting(sc)
    s1_objs = [o for o in env1["objects"] if o.type == 'MESH']
    s1_lights = {o.name: (o.data.energy, tuple(o.location), o.data.use_shadow) for o in bpy.data.objects if o.type == 'LIGHT'}
    world_before = (tuple(sc.world.node_tree.nodes["Background"].inputs[0].default_value),
                    sc.world.node_tree.nodes["Background"].inputs[1].default_value)
    env = E2.build()
    lt = E2.build_lighting(sc)
    bpy.context.view_layer.update()

    # ---------------------------------------------------------------- contract
    check(set(env) >= {"collection", "objects", "muting"}, "build() keys collection / objects / muting")
    check(env["collection"].name == "Environment2", "collection name Environment2")
    objs = list(env["collection"].all_objects)
    check(all(o.name.startswith("env2_") for o in objs), "all objects prefixed env2_")
    check(set(objs) == set(env["objects"]), f"objects list = collection ({len(objs)} objects)")
    meshes = [o for o in objs if o.type == 'MESH']
    tris = H.n_tris(meshes)
    check(tris < 150000, f"triangles {tris} < 150k")
    check(len(env["muting"]) == 2 and all("muting_on" in o for o in env["muting"]), "two muting lamps with 'muting_on'")
    names = [o.name for o in lt["lights"]]
    check(names[:1] == ["env2_light_key"] and all(n.startswith("env2_light_") for n in names), f"lights {names}")
    key = bpy.data.objects["env2_light_key"]
    k = L2.LOG_KEY_LIGHT
    check(key.data.type == 'AREA' and key.data.use_shadow and abs(key.data.energy - k["energy"]) < 1e-6
          and np.allclose(tuple(key.location), k["pos"]) and abs(key.data.size - k["size"]) < 1e-6, "env2_light_key per LOG_KEY_LIGHT, shadowed")
    check(all(not o.data.use_shadow for o in lt["lights"] if o.name != "env2_light_key"), "fill light(s) shadowless")
    s1_after = {o.name: (o.data.energy, tuple(o.location), o.data.use_shadow) for o in bpy.data.objects if o.type == 'LIGHT' and not o.name.startswith("env2_")}
    check(s1_after == s1_lights, "stage-1 lights untouched")
    world_after = (tuple(sc.world.node_tree.nodes["Background"].inputs[0].default_value),
                   sc.world.node_tree.nodes["Background"].inputs[1].default_value)
    check(world_after == world_before, "world untouched")

    # ---------------------------------------------------------------- stage-1 corner posts not duplicated, fence closes on them
    posts = E2.fence_posts()
    x1 = L2.CELL_FENCE_X0
    check(not any(abs(p[0] - x1) < 1e-6 and abs(abs(p[1]) - 3.4) < 1e-6 for p in posts), "no post at the stage-1 corners (-2.4, +-3.4)")
    runs = dict(E2.fence_runs())
    ends = {(round(p[0], 3), round(p[1], 3)) for pts in runs.values() for p in (pts[0], pts[-1])}
    check((x1, L2.LOG_FENCE_Y[1]) in ends and (x1, L2.LOG_FENCE_Y[0]) in ends, "the +-Y runs end on the stage-1 corner posts")
    # openings and door are free of bays
    bays = [(pts[i], pts[i + 1]) for pts in runs.values() for i in range(len(pts) - 1)]

    def bay_crosses(side, a, b):
        for p, q in bays:
            if side in ('+Y', '-Y') and abs(p[1] - (3.4 if side == '+Y' else -3.4)) < 1e-6:
                lo, hi = sorted((p[0], q[0]))
                if lo < b - 1e-6 and hi > a + 1e-6:
                    return True
            if side == '-X' and abs(p[0] - L2.LOG_FENCE_X0) < 1e-6:
                lo, hi = sorted((p[1], q[1]))
                if lo < b - 1e-6 and hi > a + 1e-6:
                    return True
        return False
    for o in L2.LOG_FENCE_OPENINGS:
        a, b = (o["x0"], o["x1"]) if o["side"] in ('+Y', '-Y') else (o["y0"], o["y1"])
        check(not bay_crosses(o["side"], a, b), f"opening {o['name']} free of fence bays")
    d = L2.PERSONNEL_DOOR
    check(not bay_crosses(d["side"], d["center_x"] - d["width"] / 2, d["center_x"] + d["width"] / 2), "personnel door opening free of bays")
    check(len(E2.curtain_posts()) == 2 * sum(1 for o in L2.LOG_FENCE_OPENINGS if o.get("curtain")), "2 light-curtain columns per opening")

    # ---------------------------------------------------------------- no overlap with stage-1 geometry
    zone_s1 = [o for o in s1_objs if aabb(o)[0][0] < -1.8]            # stage-1 objects near the logistics side
    s1_boxes = {o.name: aabb(o) for o in zone_s1}
    my_boxes = {o.name: aabb(o) for o in meshes}
    cand = [(a, b) for a, (la, ha) in my_boxes.items() for b, (lb, hb) in s1_boxes.items() if boxes_overlap(la, ha, lb, hb)]
    real = []
    for a, b in cand:
        ta, _ = bvh([bpy.data.objects[a]])
        tb, _ = bvh([bpy.data.objects[b]])
        if ta.overlap(tb):
            real.append((a, b))
    check(not real, f"no env2 mesh intersects stage-1 geometry ({len(zone_s1)} stage-1 objects near the zone; {real[:8]})")
    info = sorted({b for _, b in cand})
    if cand:
        print(f"[info] AABB contacts with stage-1 (touching, no intersection): {info[:10]}")
    # the post under the stage-1 door-rail bracket carries it (bracket bottom = post top)
    if "env_door_rail_bkt0" in bpy.data.objects:
        lo, hi = aabb(bpy.data.objects["env_door_rail_bkt0"])
        under = [n for n, (l2, h2) in my_boxes.items() if n.startswith("env2_fpost") and l2[0] < hi[0] and h2[0] > lo[0]
                 and l2[1] < hi[1] and h2[1] > lo[1] and abs(h2[2] - lo[2]) < 2e-3]
        check(bool(under), f"stage-1 door-rail bracket rests on a new post ({under})")

    # ---------------------------------------------------------------- zones: footprints, corridors, cell fence
    # floor paint objects hold many separate quads: check them per polygon, everything else per object
    pieces = []
    for o in meshes:
        if o.name.startswith(("env2_paint_", "env2_hatch_", "env2_hz_")):
            v, _ = world_mesh(o)
            for poly in o.data.polygons:
                pv = [o.matrix_world @ o.data.vertices[i].co for i in poly.vertices]
                pieces.append((o.name, [min(p[k] for p in pv) for k in range(3)], [max(p[k] for p in pv) for k in range(3)]))
        else:
            pieces.append((o.name, *my_boxes[o.name]))
    bad = []
    for name, lo, hi in footprints():
        for n, l2, h2 in pieces:
            flat = h2[2] - l2[2] < 0.01
            lo_ = (lo[0], lo[1], -1.0) if flat else lo           # paint counts even though it lies on the floor
            if boxes_overlap(l2, h2, lo_, hi, tol=TOL if not flat else -1e-6):
                bad.append((name, n))
    check(not bad, f"nothing inside the equipment footprints / handler corridor ({bad[:8]})")
    gp = grasp_points()
    bad = [(g, n) for g, p in gp for n, (l2, h2) in my_boxes.items()
           if h2[2] > p[2] and math.hypot(max(l2[0] - p[0], 0, p[0] - h2[0]), max(l2[1] - p[1], 0, p[1] - h2[1])) < 0.20]
    check(not bad, f"grasp corridors (r 0.20 above {len(gp)} grasp points) free ({bad[:6]})")
    lo, hi = H.world_bbox(meshes)
    inside_cell = [n for n, (l2, h2) in my_boxes.items() if h2[0] > -2.43 + TOL and not n.startswith("env2_fbay_")]
    check(not inside_cell, f"nothing but the joining bays reaches the cell fence (x > -2.43): {inside_cell[:6]}")
    check(lo[0] > -14.0 and hi[0] < -2.2, f"env2 extent x [{lo[0]:.2f}, {hi[0]:.2f}]")
    mx, my = E2.MUTE_POS
    mute = [o for o in meshes if o.name.startswith("env2_mute")]
    ml, mh = H.world_bbox(mute)
    check(mh[0] < -2.58 - 0.02 and abs(mx + 2.6) < 0.15 and abs(my - 1.35) < 0.05, f"muting posts outside the stage-1 stripe (x max {mh[0]:.3f})")
    # hatch inside the requested transfer zone and clear of the handler bed
    hl, hh = H.world_bbox([bpy.data.objects["env2_hatch_transfer"]])
    check(hl[0] >= -3.1 - TOL and hh[0] <= -2.5 + TOL and max(abs(hl[1]), abs(hh[1])) <= 1.3 + TOL, "transfer hatching within x[-3.1,-2.5] |y|<1.3")

    # ---------------------------------------------------------------- other modules: my geometry vs theirs
    # floor paint must stay outside every other module's obstacle box (paint under equipment = wrong outline);
    # solid parts: an overlap of the (conservative) boxes is confirmed on the real geometry of that module (built here
    # only when needed, then excluded from the view layer so the stills show environment2 alone)
    mine = E2.obstacles()
    check(all(set(o) >= {"name", "center", "size", "yaw"} for o in mine), f"obstacles(): {len(mine)} boxes")
    clash, box_only = [], []
    flat_pieces = [p for p in pieces if p[0].startswith(("env2_paint_", "env2_hatch_", "env2_hz_"))]
    solids = [o for o in meshes if not o.name.startswith(("env2_paint_", "env2_hatch_", "env2_hz_"))]
    for mod in ("handler_robot", "cassette", "assembly_station", "tack_robot", "conveyor", "marking_qc", "storage"):
        try:
            m_ = importlib.import_module(mod)
            other = m_.obstacles()
        except Exception as e:  # module not written yet
            print(f"[info] {mod}.obstacles() unavailable ({type(e).__name__}: {e})")
            continue
        cand = []
        for o in other:
            lb, hb = box_lohi(o)
            for n, l2, h2 in flat_pieces:
                if boxes_overlap(l2, h2, lb, hb, tol=-1e-6) and (n, o["name"]) not in clash:
                    clash.append((n, o["name"]))
            if any(boxes_overlap(*box_lohi(m), lb, hb) for m in mine):
                cand.append((lb, hb))
        if not cand:
            continue
        built = m_.build()
        bpy.context.view_layer.update()
        theirs = [o for o in built["collection"].all_objects if o.type == 'MESH']
        near = [o for o in theirs if any(boxes_overlap(*aabb(o), lb, hb, tol=0.0) for lb, hb in cand)]
        t_boxes = {o.name: aabb(o) for o in near}
        for a in solids:
            la, ha = my_boxes[a.name]
            if not any(boxes_overlap(la, ha, lb, hb, tol=0.0) for lb, hb in cand):
                continue
            ta = None
            for o in near:
                if boxes_overlap(la, ha, *t_boxes[o.name], tol=0.0):
                    ta = ta or bvh([a])[0]
                    if ta.overlap(bvh([o])[0]):
                        clash.append((a.name, o.name))
                    else:
                        box_only.append((a.name, o.name))
        bpy.context.view_layer.layer_collection.children[built["collection"].name].exclude = True
    if box_only:
        print(f"[info] env2 parts inside another module's obstacle box but clear of its geometry: {box_only[:8]}")
    check(not clash, f"env2 geometry clear of the other modules (paint vs their boxes, solids vs their meshes) ({clash[:8]})")

    # ---------------------------------------------------------------- obstacles() coverage
    boxes = [box_lohi(o) for o in mine]

    def inside_any(p):
        return any(all(lo_[q] - TOL <= p[q] <= hi_[q] + TOL for q in range(3)) for lo_, hi_ in boxes)
    miss = {}
    for o in meshes:
        v, _ = world_mesh(o)
        n = sum(1 for p in v if not inside_any(p))
        if n:
            miss[o.name] = n
    check(not miss, f"obstacle boxes cover every env2 vertex (outside: {sorted(miss.items())[:8]})")

    # ---------------------------------------------------------------- set_muting
    E2.set_muting(env, 1.0, frame=10)
    E2.set_muting(env, 0.0, frame=40)
    lamp = env["muting"][0]
    fcs = [fc for fc in lamp.animation_data.action.fcurves]
    check([fc.data_path for fc in fcs] == ['["muting_on"]'], f"set_muting keys only 'muting_on' ({[fc.data_path for fc in fcs]})")
    check(all(kp.interpolation == 'CONSTANT' for fc in fcs for kp in fc.keyframe_points), "muting keys CONSTANT")
    sc.frame_set(39)
    check(all(abs(o["muting_on"] - 1.0) < 1e-6 for o in env["muting"]), "muting on at frame 39")
    sc.frame_set(40)
    check(all(abs(o["muting_on"]) < 1e-6 for o in env["muting"]), "muting off at frame 40")

    # ---------------------------------------------------------------- exposure of the cell: stage-1 lights vs + env2
    if "--exposure" in sys.argv:
        exposure_check(sc, lt)

    if render:
        sc.frame_set(20)          # muting on
        H.still("t_env2_overview", cam=(-14.2, -9.0, 7.1), aim=(-6.4, 0.2, 0.3), lens=24, samples=16)
        H.still("t_env2_opening", cam=(-6.9, -1.25, 2.1), aim=(-2.4, 0.15, 1.05), lens=26, samples=16)
        H.still("t_env2_walkway", cam=(-6.2, -7.6, 2.4), aim=(-4.6, -3.6, 1.0), lens=26, samples=16)
    print("RESULT:", "PASS" if not FAIL else f"FAIL ({len(FAIL)})", flush=True)
    return 0 if not FAIL else 1


def exposure_check(sc, lt):
    """Top-down orthographic view of the cell + logistics floor (camera below the trusses) rendered with the stage-1
    look (demo_video/build.setup_render; EEVEE with --eevee, needs xvfb-run, else Cycles), env2 lights off / on:
    the cell floor patches must not get noticeably brighter; the logistics floor should end up about as bright as
    the cell floor."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "demo_video"))
    import build as B1
    engine = 'BLENDER_EEVEE_NEXT' if "--eevee" in sys.argv else 'CYCLES'
    cd = bpy.data.cameras.new("t_env2_top")
    cd.type = 'ORTHO'
    cd.ortho_scale = 18.0
    cam = bpy.data.objects.new("t_env2_top", cd)
    sc.collection.objects.link(cam)
    cam.location = (-4.0, 0.0, 6.9)
    sc.camera = cam
    W, Hh = 480, 240
    B1.setup_render(sc, engine, (W, Hh), 4 if engine != 'CYCLES' else 16)
    key = bpy.data.objects["env2_light_key"]
    key.data.use_shadow = True
    key.data.shadow_maximum_resolution = max(key.data.shadow_maximum_resolution, 0.015)
    patches = dict(cell_track=(1.2, 3.8, -1.6, -0.6), cell_load=(-0.8, 1.5, 1.8, 3.0), cell_east=(2.4, 3.8, 0.6, 1.4),
                   log_centre=(-6.6, -5.4, -0.9, 0.9), log_station=(-4.8, -3.6, -1.2, 0.6), log_storage=(-10.8, -9.4, -1.0, 1.0),
                   log_conveyor=(-8.5, -3.5, -3.2, -1.6))
    res = {}
    for on in (False, True):
        for o in lt["lights"]:
            o.hide_render = not on
        p = os.path.join(H.OUT, f"t_env2_exposure_{'on' if on else 'off'}.png")
        sc.render.filepath = p
        bpy.ops.render.render(write_still=True)
        img = bpy.data.images.load(p)
        a = np.array(img.pixels[:]).reshape(img.size[1], img.size[0], 4)[..., :3]
        lum = 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
        bpy.data.images.remove(img)

        def patch(x0, x1, y0, y1):
            c0, c1 = int((x0 + 13) / 18 * W), int((x1 + 13) / 18 * W)
            r0, r1 = int((y0 + 4.5) / 9 * Hh), int((y1 + 4.5) / 9 * Hh)
            return float(lum[r0:r1, c0:c1].mean())
        res[on] = {k: patch(*v) for k, v in patches.items()}
    for o in lt["lights"]:
        o.hide_render = False
    for k in patches:
        print(f"[exposure] {engine} {k:13s} off {res[False][k]:.3f}  on {res[True][k]:.3f}  x{res[True][k] / res[False][k]:.3f}")
    worst = max(res[True][k] / res[False][k] for k in patches if k.startswith("cell"))
    check(worst < 1.06, f"cell floor brightness with the env2 lights x{worst:.3f} (< 1.06)")
    cell = np.mean([res[True][k] for k in patches if k.startswith("cell")])
    logi = np.mean([res[True][k] for k in patches if k.startswith("log")])
    print(f"[exposure] logistics floor / cell floor = {logi / cell:.3f}")
    if engine != 'CYCLES':
        check(0.8 < logi / cell < 1.2, f"logistics floor about as bright as the cell floor (x{logi / cell:.3f})")


if __name__ == "__main__":
    sys.exit(main())
