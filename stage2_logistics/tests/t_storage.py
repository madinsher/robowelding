"""Check storage.py: contract, zone footprints, finished spools on every bay (seat + V-post contact, no interpenetration
between tier-0 / tier-1 spools and the rack), grasp corridors, obstacles() coverage, AGV setters; renders 3 Cycles
stills (rack filled per STORAGE_FILLED with the empty target bay, stepped side view, AGV behind the rack).

    cd /home/user/robowelding && python3 stage2_logistics/tests/t_storage.py [--no-render]
"""
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
import storage  # noqa: E402
from t_cassette import FAIL, check, world_mesh, bvh, overlap, min_dist, aabb, rect_dist, part, grasp_frame, gripper_checks  # noqa: E402

BODY = ["flange", "flange_face", "hole*", "elbow", "elbow_bevel_*", "pipe"]
ALL = ["*"]


def main():
    render = "--no-render" not in sys.argv
    H.new_scene()
    st = storage.build()
    bpy.context.view_layer.update()

    # ---------------------------------------------------------------- contract
    check(set(st) >= {"collection", "rack", "bays", "agv"}, "build() keys")
    check(st["collection"].name == "Storage", "collection name Storage")
    objs = list(st["collection"].all_objects)
    check(all(o.name.startswith("sto_") for o in objs), "all objects prefixed sto_")
    nb = sum(len(v) for v in L2.STORAGE["bay_y"])
    check(len(st["bays"]) == nb, f"{nb} bays")
    check(all(np.allclose(T, kin.planar_frame(*L2.storage_frame(t, i))) for (t, i), T in st["bays"].items()), "bay frames = storage_frame")
    check(set(st["agv"]) >= {"root", "lift", "beacon"}, "agv dict has root / lift / beacon")
    meshes = [o for o in objs if o.type == 'MESH']
    rack = [o for o in st["rack"]["objects"] if o.type == 'MESH']
    agv = [o for o in st["agv"]["objects"] if o.type == 'MESH']
    tris = H.n_tris(meshes)
    check(tris < 150000, f"triangles {tris} < 150k")

    # ---------------------------------------------------------------- footprints
    ob = L2.LOG_FENCE_OPENINGS[2]
    check(H.check_inside(rack, (L2.LOG_FENCE_X0, ob["y0"], 0.0), (L2.HANDLER_TRACK_BED[0] - 0.1, ob["y1"], L2.LOG_FENCE_H + 0.02),
                         label="rack"), "rack inside [fence x0, track bed) x storage_back opening")
    lo, hi = H.world_bbox(rack)
    check(hi[0] < L2.HANDLER_TRACK_BED[0] - 0.1 and hi[0] < min(L2.KIT_PALLET["center"][0] - L2.KIT_PALLET["size"][0] / 2,
                                                                 L2.CONVEYOR["x0"]) - 0.5, "rack clear of the handler bed / kit / conveyor")
    check(lo[0] < L2.LOG_FENCE_X0 + 0.15, f"rack back side reaches the fence line (x = {lo[0]:.3f})")
    alo, ahi = H.world_bbox(agv)
    check(ahi[0] < L2.LOG_FENCE_X0 - 0.3, f"AGV parked outside the fence (x max {ahi[0]:.2f})")
    px, py = L2.AGV["park"]
    check(abs((alo[0] + ahi[0]) / 2 - px) < 0.05 and abs((alo[1] + ahi[1]) / 2 - py) < 0.05, "AGV at AGV['park']")

    # ---------------------------------------------------------------- spools on every bay (fit), then keep STORAGE_FILLED
    spools = {}
    for (t, i), T in sorted(st["bays"].items()):
        spools[(t, i)] = part(f"chk{t}{i}", ALL, T)
    bpy.context.view_layer.update()

    def body(key):
        name = f"chk{key[0]}{key[1]}"
        return [o for o in spools[key] if any(o.name[len(name) + 1:] == k or (k.endswith("*") and o.name[len(name) + 1:].startswith(k[:-1]))
                                            for k in BODY)]

    def by_name(prefix):
        return [o for o in rack if o.name.startswith(prefix)]

    for key in sorted(spools):
        t, i = key
        n = overlap(body(key), rack)
        check(n == 0, f"bay {key}: spool does not intersect the rack ({n})")
        d = min_dist([o for o in body(key) if o.name.endswith("_flange") or o.name.endswith("_flange_face")], by_name(f"sto_seat{t}{i}_ring"))
        check(d < 0.002, f"bay {key}: flange on its seat ring (gap {d * 1000:.2f} mm)")
        d = min_dist([o for o in body(key) if o.name.endswith("_pipe")], by_name(f"sto_vhead{t}{i}"))
        check(d < 0.002, f"bay {key}: pipe on its V-post (gap {d * 1000:.2f} mm)")
        # V-head position: spool-local x of the V-head centre
        vh = [o for o in by_name(f"sto_vhead{t}{i}") if o.name == f"sto_vhead{t}{i}"][0]
        lo_, hi_ = aabb(vh)
        xl = (kin.inv(st["bays"][key]) @ np.array([(lo_[0] + hi_[0]) / 2, (lo_[1] + hi_[1]) / 2, 0.0, 1.0]))[0]
        check(abs(xl - 0.86) < 0.02, f"bay {key}: V-post at local x = {xl:.3f}")
    keys = sorted(spools)
    trees = {k: bvh(body(k))[0] for k in keys}
    ypos = {k: L2.STORAGE["bay_y"][k[0]][k[1]] for k in keys}
    near = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:] if abs(ypos[a] - ypos[b]) < 0.9]
    bad = [(a, b) for a, b in near if trees[a].overlap(trees[b])]
    check(not bad, f"no spool-spool interpenetration between neighbouring bays ({len(near)} pairs, {bad[:4]})")
    d = min(min_dist(body(k0), body(k1), n=1500) for k0, k1 in near if k0[0] != k1[0] and abs(ypos[k0] - ypos[k1]) < 0.5)
    check(d > 0.03, f"tier-0 / tier-1 spool clearance {d * 1000:.0f} mm")
    # tier-1 V-posts vs tier-0 spools
    d = min(min_dist(body(k0), [o for o in rack if o.name.startswith("sto_vpost1")], n=1500) for k0 in keys if k0[0] == 0)
    check(d > 0.03, f"tier-1 V-posts clear the tier-0 spools by {d * 1000:.0f} mm")

    # ---------------------------------------------------------------- grasp corridors (r = 0.20 above the spool grasp point)
    grasps = [(k, T @ np.array([*L2.GRASP["spool"][0], 1.0])) for k, T in st["bays"].items()]
    boxes = {o.name: aabb(o) for o in rack}
    bad = [(k, on) for k, g in grasps for on, (lo_, hi_) in boxes.items() if hi_[2] > g[2] and rect_dist(g[0], g[1], lo_, hi_) < 0.20]
    check(not bad, f"grasp corridors free ({bad[:6]})")
    # gripper jaws (open +-0.32 along world Y, +-0.13 along X around the grasp, down to 0.06 below the pipe) clear of the rack
    bad = []
    for k, g in grasps:
        jlo = (g[0] - 0.13, g[1] - 0.32, g[2] - L2.PIPE_R - 0.06)
        jhi = (g[0] + 0.13, g[1] + 0.32, g[2] + 0.6)
        for on, (lo_, hi_) in boxes.items():
            if all(lo_[q] < jhi[q] and hi_[q] > jlo[q] for q in range(3)):
                bad.append((k, on))
    check(not bad, f"open-jaw envelope at every bay clear of the rack ({bad[:6]})")

    # ---------------------------------------------------------------- the real gripper at every bay (open / closed)
    gl = [(f"{k[0]}{k[1]}", *grasp_frame(T, "spool"), body(k)) for k, T in sorted(st["bays"].items())]
    gripper_checks(gl, rack, {f"{k[0]}{k[1]}": body(k) for k in keys}, "rack")

    # ---------------------------------------------------------------- obstacles()
    obs = storage.obstacles()
    check(all(set(o) >= {"name", "center", "size", "yaw"} for o in obs), f"obstacles(): {len(obs)} boxes")

    def inside_any(p):
        return any(all(abs(p[q] - o["center"][q]) <= o["size"][q] / 2 + 1e-3 for q in range(3)) for o in obs)
    miss = {}
    for o in meshes:
        v, _ = world_mesh(o)
        k = sum(1 for p in v if not inside_any(p))
        if k:
            miss[o.name] = k
    check(not miss, f"obstacle boxes cover rack + parked AGV (outside: {sorted(miss.items())[:8]})")
    bad = []
    for o in obs:
        lo_ = [o["center"][q] - o["size"][q] / 2 for q in range(3)]
        hi_ = [o["center"][q] + o["size"][q] / 2 for q in range(3)]
        bad += [(o["name"], k) for k, g in grasps if hi_[2] > g[2] and rect_dist(g[0], g[1], lo_, hi_) < 0.20]
    check(not bad, f"obstacle boxes keep the grasp corridors free ({bad[:6]})")

    # ---------------------------------------------------------------- setters
    sc = bpy.context.scene
    storage.set_agv(st, -12.6, -6.0, math.pi / 2, frame=1)
    storage.set_agv(st, *L2.AGV["park"], math.pi / 2, frame=49)
    storage.set_agv_lift(st, 0.0, frame=49)
    storage.set_agv_lift(st, 0.08, frame=73)
    storage.set_beacon(st, 1.0, frame=1)
    storage.set_beacon(st, 0.0, frame=60)
    ad = st["agv"]["root"].animation_data
    paths = sorted({(fc.data_path, fc.array_index) for fc in ad.action.fcurves})
    check(paths == [("location", 0), ("location", 1), ("rotation_euler", 2)], f"set_agv keys only x, y, yaw {paths}")
    sc.frame_set(25)
    r = st["agv"]["root"]
    check(-6.0 < r.location[1] < L2.AGV["park"][1] and abs(r.location[0] + 12.6) < 1e-6, f"AGV moving at frame 25 (y = {r.location[1]:.2f})")
    sc.frame_set(59)
    check(abs(st["agv"]["beacon"]["beacon_on"] - 1.0) < 1e-6, "beacon CONSTANT on at frame 59")
    sc.frame_set(73)
    check(abs(st["agv"]["lift"].location[2] - (storage.AGV_DECK_Z + 0.08)) < 1e-6, "lift stroke 0.08 at frame 73")
    sc.frame_set(60)
    check(abs(st["agv"]["beacon"]["beacon_on"]) < 1e-6, "beacon off at frame 60")
    # still pose: AGV parked, table raised a bit, beacon on
    sc.frame_set(55)

    # ---------------------------------------------------------------- stills: STORAGE_FILLED only (target bay stays empty)
    for key in keys:
        if key not in L2.STORAGE_FILLED:
            for o in spools[key]:
                bpy.data.objects.remove(o, do_unlink=True)
    check(L2.STORAGE_TARGET not in L2.STORAGE_FILLED, "target bay is empty")
    if render:
        H.still("t_storage_rack", cam=(-7.3, -2.9, 2.6), aim=(-10.35, 0.1, 0.75), lens=28, frame=55)
        H.still("t_storage_side", cam=(-8.2, -3.3, 1.05), aim=(-10.45, -0.9, 0.85), lens=32, frame=55)
        H.still("t_storage_agv", cam=(-14.6, -1.6, 2.2), aim=(-12.0, 1.2, 0.6), lens=28, frame=55)
    print("RESULT:", "PASS" if not FAIL else f"FAIL ({len(FAIL)})", flush=True)
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
