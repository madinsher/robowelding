"""Check cassette.py: contract, footprints, part fit in the nests / V-cradles (BVH overlap + contact distance), grasp
corridors, obstacles() coverage, setter keys; renders 3 Cycles stills to out/tests/.

    cd /home/user/robowelding && python3 stage2_logistics/tests/t_cassette.py [--no-render]
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import harness as H  # noqa: E402
import bpy  # noqa: E402
import mathutils  # noqa: E402
import numpy as np  # noqa: E402
from mathutils.bvhtree import BVHTree  # noqa: E402
import layout2 as L2  # noqa: E402
import kin  # noqa: E402
import cassette  # noqa: E402
from cell import spool, geom as G  # noqa: E402

FAIL = []


def check(ok, msg):
    print(("[ok]   " if ok else "[FAIL] ") + msg, flush=True)
    if not ok:
        FAIL.append(msg)


# ------------------------------------------------------------------ geometry helpers
def world_mesh(ob):
    """World vertices + triangles (proper tessellation: the V-blocks are concave n-gons)."""
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    me = ev.to_mesh()
    me.calc_loop_triangles()
    mw = ev.matrix_world
    verts = [mw @ v.co for v in me.vertices]
    polys = [tuple(t.vertices) for t in me.loop_triangles]
    ev.to_mesh_clear()
    return verts, polys


def bvh(objs):
    V, P = [], []
    for ob in objs:
        v, p = world_mesh(ob)
        o = len(V)
        V += v
        P += [tuple(i + o for i in f) for f in p]
    return BVHTree.FromPolygons(V, P, epsilon=0.0), V


def overlap(objs_a, objs_b):
    ta, _ = bvh(objs_a)
    tb, _ = bvh(objs_b)
    return len(ta.overlap(tb))


def surface_samples(objs, n=6000, seed=1):
    """Random points on the (world) surface of objs, area-weighted."""
    rng = np.random.default_rng(seed)
    tris = []
    for ob in objs:
        v, p = world_mesh(ob)
        v = np.array([tuple(q) for q in v])
        for f in p:
            for k in range(1, len(f) - 1):
                tris.append((v[f[0]], v[f[k]], v[f[k + 1]]))
    if not tris:
        return np.zeros((0, 3))
    T = np.array(tris)
    area = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    idx = rng.choice(len(T), size=n, p=area / area.sum())
    u, w = rng.random(n), rng.random(n)
    flip = u + w > 1
    u[flip], w[flip] = 1 - u[flip], 1 - w[flip]
    A, B, C = T[idx, 0], T[idx, 1], T[idx, 2]
    return A + (B - A) * u[:, None] + (C - A) * w[:, None]


def min_dist(objs_a, objs_b, n=6000):
    """Minimum surface-to-surface distance (vertices + area samples of each side against the other's BVH)."""
    best = 1e9
    for a, b in ((objs_a, objs_b), (objs_b, objs_a)):
        tb, _ = bvh(b)
        _, va = bvh(a)
        pts = list(va) + [mathutils.Vector(p) for p in surface_samples(a, n)]
        for v in pts:
            hit = tb.find_nearest(v)
            if hit[0] is not None:
                best = min(best, hit[3])
    return best


def aabb(ob):
    v, _ = world_mesh(ob)
    lo = [min(p[k] for p in v) for k in range(3)]
    hi = [max(p[k] for p in v) for k in range(3)]
    return lo, hi


def rect_dist(px, py, lo, hi):
    dx = max(lo[0] - px, 0.0, px - hi[0])
    dy = max(lo[1] - py, 0.0, py - hi[1])
    return math.hypot(dx, dy)


def part(name, keep, T):
    """Stage-1 spool reduced to one part (objects whose name matches keep), root placed at T."""
    sp = spool.build(name=name)
    sp["root"].matrix_world = G.M(T)
    objs = []
    for ob in list(sp["collection"].all_objects):
        if ob.type != 'MESH':
            continue
        short = ob.name[len(name) + 1:]
        if any(short == k or (k.endswith("*") and short.startswith(k[:-1])) for k in keep):
            objs.append(ob)
        else:
            bpy.data.objects.remove(ob, do_unlink=True)
    bpy.context.view_layer.update()
    return objs


def grasp_frame(T_part, key):
    """World grasp frame (TCP) of GRASP[key] for a part / spool frame, and its closed jaw state."""
    o, x, z, s_closed = L2.GRASP[key]
    return T_part @ kin.frame(x, z, o), s_closed


def gripper_at(T_grasp, s, name):
    """The stage-2 gripper (gripper.py) with its TCP at T_grasp, jaws at s; returns (collection, handle, meshes) or None."""
    try:
        import gripper
    except ImportError:
        return None
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    t0 = G.empty(name + "_tool0", collection=col)
    t0.matrix_world = G.M(T_grasp @ kin.tr(0.0, 0.0, -L2.GRIP_TCP_Z))
    g = gripper.build(t0, col, name=name)
    gripper.set_open(g, s)
    bpy.context.view_layer.update()
    return col, g, [o for o in col.all_objects if o.type == 'MESH']


def set_jaws(g, s):
    import gripper
    gripper.set_open(g, s)
    bpy.context.view_layer.update()


def remove_collection(col):
    for o in list(col.all_objects):
        bpy.data.objects.remove(o, do_unlink=True)
    bpy.data.collections.remove(col)


def gripper_checks(grasps, statics, others, label):
    """grasps: [(key, T_grasp, s_closed, own_part_objs)].  The gripper (open s = 1 and closed) at every grasp must not
    touch `statics`; contacts with other loose parts (`others`: dict key -> objs) are reported as info only."""
    for key, Tg, s_closed, own in grasps:
        res = gripper_at(Tg, 1.0, f"gchk_{key}")
        if res is None:
            print("[info] gripper.py not available: gripper compatibility not checked", flush=True)
            return
        col, g, gm = res
        for s in (1.0, s_closed):
            set_jaws(g, s)
            n = overlap(gm, statics)
            check(n == 0, f"{label} {key}: gripper (s = {s:.2f}) clear of the {label} ({n})")
            hits = [k for k, objs in others.items() if objs is not own and overlap(gm, objs)]
            if hits:
                print(f"[info] {label} {key}: gripper (s = {s:.2f}) touches neighbouring parts {hits}", flush=True)
        remove_collection(col)


FLANGE = ["flange", "flange_face", "hole*"]
ELBOW = ["elbow", "elbow_bevel_*"]
PIPE = ["pipe"]


def main():
    render = "--no-render" not in sys.argv
    H.new_scene()
    cas = cassette.build()
    bpy.context.view_layer.update()

    # ---------------------------------------------------------------- contract
    check(set(cas) >= {"collection", "pallet", "nests", "pipe_buffer"}, "build() keys")
    check(cas["collection"].name == "KitCassette", "collection name KitCassette")
    objs = [o for o in cas["collection"].all_objects]
    check(all(o.name.startswith("kit_") for o in objs), "all objects prefixed kit_")
    check(len(cas["nests"]["flange"]) == 2 and len(cas["nests"]["elbow"]) == 2, "2 flange + 2 elbow nests")
    for i, (o, xd) in enumerate(L2.KIT_FLANGES):
        check(np.allclose(cas["nests"]["flange"][i], kin.planar_frame(o, xd)), f"flange nest {i} frame")
    for i, (o, xd) in enumerate(L2.KIT_ELBOWS):
        check(np.allclose(cas["nests"]["elbow"][i], kin.planar_frame(o, xd)), f"elbow nest {i} frame")
    for i in range(len(L2.PIPE_BUFFER_X)):
        check(np.allclose(cas["pipe_buffer"]["slots"][i], kin.planar_frame(*L2.pipe_buffer_frame(i))), f"buffer slot {i} frame")
    meshes = [o for o in objs if o.type == 'MESH']
    tris = H.n_tris(meshes)
    check(tris < 150000, f"triangles {tris} < 150k")
    mats = {m.name for o in meshes for m in o.data.materials if m}
    check(all(bpy.data.materials[m].use_nodes for m in mats), f"{len(mats)} materials, all node-based")

    # ---------------------------------------------------------------- footprints
    pal = [o for o in cas["pallet"]["objects"] if o.type == 'MESH']
    cx, cy = L2.KIT_PALLET["center"]
    sx, sy = L2.KIT_PALLET["size"]
    check(H.check_inside(pal, (cx - sx / 2, cy - sy / 2, 0.0), (cx + sx / 2, cy + sy / 2, 0.65), label="pallet"), "kit pallet inside KIT_PALLET")
    buf = [o for o in cas["pipe_buffer"]["objects"] if o.type in ('MESH', 'CURVE')]
    bx, by = L2.PIPE_BUFFER["center"]
    bsx, bsy = L2.PIPE_BUFFER["size"]
    check(H.check_inside([o for o in buf if o.type == 'MESH'], (bx - bsx / 2, by - bsy / 2, 0.0), (bx + bsx / 2, by + bsy / 2, 1.05),
                         label="buffer"), "pipe buffer inside PIPE_BUFFER")
    lo, hi = H.world_bbox(meshes)
    check(lo[1] > 0.75 and hi[1] < L2.LOG_FENCE_Y[1] - 0.1, "clear of the handler corridor and the +Y fence")

    # ---------------------------------------------------------------- parts in their nests
    parts = {}
    for i, T in enumerate(cas["nests"]["flange"]):
        parts[f"F{i}"] = part(f"chkF{i}", FLANGE, T)
    for i, T in enumerate(cas["nests"]["elbow"]):
        parts[f"E{i}"] = part(f"chkE{i}", ELBOW, T)
    for i, T in enumerate(cas["pipe_buffer"]["slots"]):
        parts[f"P{i}"] = part(f"chkP{i}", PIPE, T)
    bpy.context.view_layer.update()

    def by_name(*prefixes):
        return [o for o in meshes if any(o.name.startswith(p) for p in prefixes)]

    for key, pobjs in parts.items():
        n = overlap(pobjs, meshes)
        check(n == 0, f"part {key}: no interpenetration with the cassette/buffer ({n} tri pairs)")
    for i in range(2):
        d = min_dist(parts[f"F{i}"], by_name(f"kit_fnest{i}_ring"))
        check(d < 0.002, f"flange {i} sits on its seat ring (gap {d * 1000:.2f} mm)")
        d = min_dist(parts[f"E{i}"], by_name(f"kit_enest{i}_cup"))
        check(d < 0.002, f"elbow {i} lower end on the cup floor (gap {d * 1000:.2f} mm)")
        d = min_dist([o for o in parts[f"E{i}"] if o.name.endswith("_elbow")], by_name(f"kit_enest{i}_vpad"))
        check(d < 0.002, f"elbow {i} upper end touches the V-rest (gap {d * 1000:.2f} mm)")
    for i in range(len(L2.PIPE_BUFFER_X)):
        for j in range(2):
            d = min_dist(parts[f"P{i}"], by_name(f"kit_buf_v{j}{i}"))
            check(d < 0.002, f"pipe {i} rests in V-cradle {j} (gap {d * 1000:.2f} mm)")
        d = min_dist(parts[f"P{i}"], by_name("kit_buf_stop_rubber"))
        check(d < 0.004, f"pipe {i} end at the rubber stop (gap {d * 1000:.2f} mm)")

    # ---------------------------------------------------------------- grasp corridors (r = 0.20 above each grasp point)
    grasps = []
    for T in cas["nests"]["flange"]:
        grasps.append(("flange", T @ np.array([*L2.GRASP["flange"][0], 1.0])))
    for T in cas["nests"]["elbow"]:
        grasps.append(("elbow", T @ np.array([*L2.GRASP["elbow"][0], 1.0])))
    for T in cas["pipe_buffer"]["slots"]:
        grasps.append(("pipe", T @ np.array([*L2.GRASP["pipe"][0], 1.0])))
    bad = []
    boxes = {o.name: aabb(o) for o in meshes}
    for name, g in grasps:
        for on, (lo, hi) in boxes.items():
            if hi[2] > g[2] and rect_dist(g[0], g[1], lo, hi) < 0.20:
                bad.append((name, on))
    check(not bad, f"grasp corridors free ({bad[:6]})")

    # ---------------------------------------------------------------- the real gripper at every grasp (open / closed)
    gl = []
    for i, T in enumerate(cas["nests"]["flange"]):
        gl.append((f"F{i}", *grasp_frame(T, "flange"), parts[f"F{i}"]))
    for i, T in enumerate(cas["nests"]["elbow"]):
        gl.append((f"E{i}", *grasp_frame(T, "elbow"), parts[f"E{i}"]))
    for i, T in enumerate(cas["pipe_buffer"]["slots"]):
        gl.append((f"P{i}", *grasp_frame(T, "pipe"), parts[f"P{i}"]))
    gripper_checks(gl, meshes, parts, "kit")

    # ---------------------------------------------------------------- obstacles(): pure data, covers the geometry
    obs = cassette.obstacles()
    check(all(set(o) >= {"name", "center", "size", "yaw"} for o in obs), f"obstacles(): {len(obs)} boxes with name/center/size/yaw")

    def inside_any(p):
        for o in obs:
            c, s = o["center"], o["size"]
            if all(abs(p[k] - c[k]) <= s[k] / 2 + 1e-3 for k in range(3)):
                return True
        return False
    miss = {}
    for o in meshes:
        v, _ = world_mesh(o)
        k = sum(1 for p in v if not inside_any(p))
        if k:
            miss[o.name] = k
    check(not miss, f"obstacle boxes cover the mesh vertices (outside: {sorted(miss.items())[:8]})")
    bad = []
    for o in obs:
        c, s = o["center"], o["size"]
        lo = [c[k] - s[k] / 2 for k in range(3)]
        hi = [c[k] + s[k] / 2 for k in range(3)]
        for name, g in grasps:
            if hi[2] > g[2] and rect_dist(g[0], g[1], lo, hi) < 0.20:
                bad.append((o["name"], name))
    check(not bad, f"obstacle boxes keep the grasp corridors free ({bad[:6]})")

    # ---------------------------------------------------------------- setter
    cassette.set_presence(cas, 1, 0.0, frame=10)
    cassette.set_presence(cas, 1, 1.0, frame=20)
    led = cas["pipe_buffer"]["leds"][1]
    bpy.context.scene.frame_set(15)
    check(abs(led["led_on"]) < 1e-6, "set_presence keys CONSTANT (off at frame 15)")
    bpy.context.scene.frame_set(20)
    check(abs(led["led_on"] - 1.0) < 1e-6, "set_presence on at frame 20")
    bpy.context.scene.frame_set(1)

    # ---------------------------------------------------------------- stills
    if render:
        H.still("t_cassette_kit", cam=(-6.05, 0.55, 1.75), aim=(-7.12, 2.45, 0.30), lens=32)
        H.still("t_cassette_buffer", cam=(-4.55, 0.85, 1.85), aim=(-5.62, 2.55, 0.72), lens=30)
        H.still("t_cassette_elbow", cam=(-6.35, 1.05, 0.95), aim=(-7.1, 2.1, 0.42), lens=40)
    print("RESULT:", "PASS" if not FAIL else f"FAIL ({len(FAIL)})", flush=True)
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
