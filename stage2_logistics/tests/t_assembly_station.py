"""Assembly & tack station check: contract (dict keys, setters key their channels), footprint inside STATION_TABLE,
clear of the neighbours (pipe buffer rack, tack pedestal, handler corridor), fixture vs a stage-1 spool placed at the
station frame (supports and closed clamp pads touch, nothing penetrates the part, no station mesh intersects any spool
mesh incl. the bolt-hole discs), open clamps clear of the grasp corridors and of the open handler gripper, +Y side of
seams A / B free for the tack torch, obstacles() cover the geometry; Cycles stills to out/tests/.

    python3 stage2_logistics/tests/t_assembly_station.py [--no-render]
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness as H  # noqa: E402
import bpy  # noqa: E402
import mathutils  # noqa: E402
import numpy as np  # noqa: E402

import kin  # noqa: E402
import layout2 as L2  # noqa: E402
import assembly_station as AS  # noqa: E402
from cell import geom as G, spool, layout as L  # noqa: E402

RENDER = "--no-render" not in sys.argv
FAILS = []


def check(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg, flush=True)
    if not ok:
        FAILS.append(msg)


def local_verts(objs, st):
    """Evaluated mesh vertices of objs in station-local coordinates (n,3) and the owning object index."""
    dg = bpy.context.evaluated_depsgraph_get()
    inv = st["root"].matrix_world.inverted()
    pts, own = [], []
    for i, ob in enumerate(objs):
        if ob.type not in ('MESH', 'CURVE'):
            continue
        ev = ob.evaluated_get(dg)
        me = ev.to_mesh()
        if me is None:
            continue
        Mx = inv @ ev.matrix_world
        n = len(me.vertices)
        co = np.empty(n * 3)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        R = np.array(Mx.to_3x3())
        t = np.array(Mx.translation)
        pts.append(co @ R.T + t)
        own += [i] * n
        ev.to_mesh_clear()
    return (np.concatenate(pts) if pts else np.zeros((0, 3))), np.array(own)


def gap(ob, surf_local, st):
    """Min (unsigned) distance (m) from part-surface points (station-local) to the evaluated mesh of ob (BVH);
    penetration is checked separately (fixture vertices inside the part)."""
    from mathutils.bvhtree import BVHTree
    dg = bpy.context.evaluated_depsgraph_get()
    bvh = BVHTree.FromObject(ob, dg)
    Mw = st["root"].matrix_world
    Mi = ob.matrix_world.inverted()
    best = 1e9
    for p in surf_local:
        q = Mi @ (Mw @ mathutils.Vector(tuple(p)))
        loc, nor, _, dist = bvh.find_nearest(q)
        if loc is not None:
            best = min(best, dist)
    return best


def world_bvh(ob):
    """(BVH, lo, hi) of the evaluated mesh of ob in world coordinates, or None."""
    from mathutils.bvhtree import BVHTree
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    me = ev.to_mesh()
    if me is None or len(me.vertices) == 0:
        ev.to_mesh_clear()
        return None
    Mw = ev.matrix_world
    V = [tuple(Mw @ v.co) for v in me.vertices]
    F = [tuple(p.vertices) for p in me.polygons]
    ev.to_mesh_clear()
    a = np.array(V)
    return BVHTree.FromPolygons(V, F), a.min(axis=0), a.max(axis=0)


def mesh_overlaps(objs_a, objs_b):
    """Pairs (name_a, name_b) whose evaluated meshes intersect (triangle overlap, AABB pre-filter)."""
    A = [(o.name, world_bvh(o)) for o in objs_a]
    B = [(o.name, world_bvh(o)) for o in objs_b]
    out = []
    for na, ta in A:
        if ta is None:
            continue
        for nb, tb in B:
            if tb is None or np.any(ta[1] > tb[2]) or np.any(tb[1] > ta[2]):
                continue
            if ta[0].overlap(tb[0]):
                out.append((na, nb))
    return out


def flange_distance(p):
    """Signed distance-ish to the flange disc + hub (spool-local): negative inside."""
    r = np.hypot(p[:, 0], p[:, 1])
    z = p[:, 2]
    d_disc = np.maximum.reduce([r - AS.FL_R, -z, z - AS.FL_THK])
    d_hub = np.maximum.reduce([r - (AS.R_PIPE + 0.028), -z, z - AS.Z_A])
    return np.minimum(d_disc, d_hub)


def part_distance(p):
    return np.minimum.reduce([flange_distance(p), AS.elbow_distance(p), AS.pipe_distance(p)])


def main():
    sc = H.new_scene()
    st = AS.build()
    print("built", len(st["objects"]), "objects,", H.n_tris(st["objects"]), "triangles")
    # ---------------------------------------------------------------- contract
    for k in ("collection", "clamps", "lamp", "table", "frame"):
        check(k in st, f"build() returns '{k}'")
    check(set(st["clamps"]) == set(L2.STATION_CLAMPS), "clamp names == layout2.STATION_CLAMPS")
    check(np.allclose(st["frame"], kin.planar_frame(L2.STATION_ORIGIN, L2.STATION_XDIR)), "frame == planar_frame(STATION_ORIGIN, XDIR)")
    check(np.allclose(np.array(st["root"].matrix_world), st["frame"], atol=1e-6), "stn_root at the station frame")
    check(all(o.name.startswith("stn_") for o in st["objects"]), "every object name has the stn_ prefix")
    check(H.n_tris(st["objects"]) < 150000, "triangle budget < 150k")
    for name in L2.STATION_CLAMPS:
        AS.set_clamp(st, name, 0.0, frame=1)
        AS.set_clamp(st, name, 1.0, frame=11)
        c = st["clamps"][name]
        ok = all(ob.animation_data and ob.animation_data.action and ob.animation_data.action.fcurves.find(p, index=i)
                 for ob, p, i in ((c["lever"], "rotation_euler", 0), (c["cyl"], "rotation_euler", 0), (c["rod"], "location", 1)))
        check(ok, f"set_clamp('{name}') keys lever / cylinder / rod")
    AS.set_lamp(st, 0, frame=1)
    AS.set_lamp(st, 1, frame=20)
    fc = st["lamp"].animation_data.action.fcurves.find('["lamp_on"]')
    check(fc is not None and all(kp.interpolation == 'CONSTANT' for kp in fc.keyframe_points), "set_lamp keys lamp_on (CONSTANT)")
    sc.frame_set(20)
    check(abs(st["lamp"]["lamp_on"] - 1.0) < 1e-9, "lamp on at frame 20")
    # ---------------------------------------------------------------- footprint (world)
    cx, cy = L2.STATION_TABLE["center"]
    sx, sy = L2.STATION_TABLE["size"]
    objs = [o for o in st["objects"] if o.type in ('MESH', 'CURVE')]
    for f in (1, 11):
        sc.frame_set(f)
        check(H.check_inside(objs, (cx - sx / 2 - 0.10, cy - sy / 2 - 0.10, 0.0), (cx + sx / 2 + 0.10, cy + sy / 2 + 0.10, 1.75),
                             label=f"station f{f}"), f"station inside its table footprint (+0.1 m) at frame {f}")
    # neighbours (every evaluated vertex, tubes / cables included): the pipe buffer rack stands 0.10 m beyond the table's
    # world -X edge, the tack-robot pedestal on its +X side, the handler track corridor at |y| < 0.75
    Wv = []
    dg = bpy.context.evaluated_depsgraph_get()
    for ob in objs:
        ev = ob.evaluated_get(dg)
        me = ev.to_mesh()
        if me is not None:
            Wv += [tuple(ev.matrix_world @ v.co) for v in me.vertices]
        ev.to_mesh_clear()
    Wv = np.array(Wv)
    buf_x1 = L2.PIPE_BUFFER["center"][0] + L2.PIPE_BUFFER["size"][0] / 2
    ped_x0 = L2.TACK_BASE[0] - 0.30
    check(Wv[:, 0].min() >= buf_x1 + 0.03, f"clear of the pipe buffer footprint x <= {buf_x1:.2f} by >= 3 cm "
          f"(station min x {Wv[:, 0].min():.3f})")
    check(Wv[:, 0].max() <= ped_x0 - 0.05, f"clear of the tack pedestal (station max x {Wv[:, 0].max():.3f})")
    check(Wv[:, 1].min() >= 0.75 + 0.3, f"clear of the handler track corridor (station min y {Wv[:, 1].min():.3f})")
    # ---------------------------------------------------------------- vs the stage-1 spool on the station
    sp = spool.build(name="chk")
    sp["root"].matrix_world = G.M(st["frame"])
    bpy.context.view_layer.update()
    names = list(L2.STATION_CLAMPS)
    fixture = [o for o in objs if not any(o.name.startswith(f"stn_cl_{n}_") for n in names)]
    P, own = local_verts(fixture, st)
    d = part_distance(P)
    worst = int(np.argmin(d))
    check(d[worst] > -2e-4, f"no fixture vertex inside the spool (min {d[worst] * 1000:.2f} mm at {fixture[own[worst]].name})")
    # mesh-level: no station mesh intersects any stage-1 spool mesh (incl. the bolt-hole discs that stand 2 mm proud of
    # the flange back face, the raised face, the beads), clamps closed and open
    spool_objs = [o for o in bpy.data.objects if o.name.startswith("chk_") and o.type == 'MESH']
    for s in (1.0, 0.0):
        for name in L2.STATION_CLAMPS:
            AS.set_clamp(st, name, s)
        bpy.context.view_layer.update()
        ov = mesh_overlaps(objs, spool_objs)
        check(not ov, f"clamps s={s}: no station mesh intersects the spool meshes {ov[:4]}")

    def touch(name, surf, label, tol=1.0e-3):
        """Surface-to-surface gap: dense part-surface samples (station-local) vs the object's BVH."""
        ob = bpy.data.objects[name]
        m = gap(ob, surf, st)
        check(-2e-4 < m < tol, f"{label} touches the part (min gap {m * 1000:.2f} mm)")
    ev_pts = AS.elbow_surface(AS.EV_X - 0.07, AS.EV_X + 0.07)
    th = np.linspace(0, 2 * math.pi, 720, endpoint=False)
    xs = np.linspace(AS.PV_X - 0.04, AS.PV_X + 0.04, 41)
    pv_pts = np.array([(x, AS.R_PIPE * math.cos(t), AS.Z_P + AS.R_PIPE * math.sin(t)) for x in xs for t in th])
    for k in range(2):
        touch(f"stn_ev_liner{k}", ev_pts, f"elbow V-block liner {k}")
        touch(f"stn_pv_liner{k}", pv_pts, f"pipe V-block liner {k}")
    Q, _ = local_verts([o for o in objs if o.name == "stn_stop_face"], st)
    m = float(np.min(Q[:, 0] - L2.PIPE_END_X))
    check(0.0 <= m < 1e-3, f"pipe end stop face at the pipe end (gap {m * 1000:.2f} mm)")
    Q, _ = local_verts([o for o in objs if o.name == "stn_seat"], st)
    top = float(Q[:, 2].max())
    check(-1e-3 < top <= 0.0, f"flange seat top at the flange back face (z {top * 1000:.2f} mm)")
    # clamps: closed pads touch, nothing of a clamp penetrates the part in any state
    for s in (1.0, 0.5, 0.0):
        for name in names:
            AS.set_clamp(st, name, s)
        bpy.context.view_layer.update()
        for name in names:
            sel = [o for o in objs if o.name.startswith(f"stn_cl_{name}_")]
            Q, ow = local_verts(sel, st)
            dd = part_distance(Q)
            i = int(np.argmin(dd))
            check(dd[i] > -2e-4, f"clamp {name} s={s}: no penetration (min {dd[i] * 1000:.2f} mm at {sel[ow[i]].name})")
            if s == 1.0:
                pad = [o for o in sel if o.name.endswith("_pad_rubber")]
                Qp, _ = local_verts(pad, st)
                m = float(np.min(part_distance(Qp)))
                check(-2e-4 < m < 1.5e-3, f"clamp {name} closed: pad touches the part (gap {m * 1000:.2f} mm)")
    # ---------------------------------------------------------------- corridors (clamps open, s = 0)
    sc.frame_set(1)
    for name in names:
        AS.set_clamp(st, name, 0.0)
    bpy.context.view_layer.update()
    Pall, own = local_verts(objs, st)
    part_top = {"flange": 0.028, "elbow": 0.40, "pipe": Z_TOP, "spool": Z_TOP}
    for g, (o, xa, za, _) in L2.GRASP.items():
        c = np.array(o[:2])
        r = np.hypot(Pall[:, 0] - c[0], Pall[:, 1] - c[1])
        bad = (r < 0.20) & (Pall[:, 2] > part_top[g])
        names_bad = sorted({objs[i].name for i in own[bad]})
        check(not bad.any(), f"grasp corridor '{g}' (r 0.20 above the part) free with clamps open {names_bad[:4]}")
    # open handler gripper at each station grasp (tool frame box x +-0.13, y +-0.34, z from -0.45 to +0.06 about TCP)
    states = {"flange": {}, "elbow": {"flange_l": 1, "flange_r": 1}, "pipe": {"flange_l": 1, "flange_r": 1, "elbow": 1}, "spool": {}}
    for g, closed in states.items():
        for name in names:
            AS.set_clamp(st, name, closed.get(name, 0.0))
        bpy.context.view_layer.update()
        Pg, ow = local_verts(objs, st)
        o, xa, za, _ = L2.GRASP[g]
        Tg = kin.frame(xa, za, o)
        q = (kin.inv(Tg) @ np.c_[Pg, np.ones(len(Pg))].T).T[:, :3]
        bad = (np.abs(q[:, 0]) < 0.13) & (np.abs(q[:, 1]) < 0.34) & (q[:, 2] > -0.45) & (q[:, 2] < 0.06)
        names_bad = sorted({objs[i].name for i in ow[bad]})
        check(not bad.any(), f"open gripper envelope at grasp '{g}' free {names_bad[:4]}")
    # tack torch side: +Y (local) of seams A and B free with the clamps closed
    for name in names:
        AS.set_clamp(st, name, 1.0)
    bpy.context.view_layer.update()
    Pc, ow = local_verts(objs, st)
    zoneA = (np.abs(Pc[:, 0]) < 0.30) & (Pc[:, 1] > 0.15) & (Pc[:, 1] < 0.7) & (Pc[:, 2] > 0.035) & (Pc[:, 2] < 0.9)
    zoneB = (Pc[:, 0] > 0.25) & (Pc[:, 0] < 0.65) & (Pc[:, 1] > 0.15) & (Pc[:, 1] < 0.7) & (Pc[:, 2] > 0.30) & (Pc[:, 2] < 1.0)
    check(not (zoneA | zoneB).any(), f"+Y side of seams A / B free (tack torch) {sorted({objs[i].name for i in ow[zoneA | zoneB]})[:4]}")
    # ---------------------------------------------------------------- obstacles cover the geometry (all clamp states)
    obs = AS.obstacles()
    for s in (0.0, 0.5, 1.0):
        for name in names:
            AS.set_clamp(st, name, s)
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        miss = []
        for ob in objs:
            ev = ob.evaluated_get(dg)
            me = ev.to_mesh()
            if me is None:
                continue
            co = np.array([ev.matrix_world @ v.co for v in me.vertices])
            ev.to_mesh_clear()
            if len(co) == 0:
                continue
            inside = np.zeros(len(co), dtype=bool)
            for o in obs:
                c, h = np.array(o["center"]), np.array(o["size"]) / 2 + 1e-4
                inside |= np.all(np.abs(co - c) <= h, axis=1)
            if not inside.all():
                miss.append((ob.name, int((~inside).sum())))
        check(not miss, f"obstacles() cover every vertex at s={s} {miss[:4]}")
    # ---------------------------------------------------------------- stills
    if RENDER:
        for ob in st["objects"]:
            ob.animation_data_clear()                     # the contract keys above would override set_clamp
        sc.frame_set(1)
        for name in names:
            AS.set_clamp(st, name, 1.0)
        AS.set_lamp(st, 1)
        H.still("t_station_closed", cam=(-2.95, 0.95, 2.05), aim=(-4.40, 2.20, 1.05), lens=30)
        H.still("t_station_clamps", cam=(-5.95, 1.35, 1.75), aim=(-4.45, 2.25, 1.10), lens=35)
        for name in names:
            AS.set_clamp(st, name, 0.0)
        AS.set_lamp(st, 0)
        sp["root"].location[2] += 0.40
        H.still("t_station_open", cam=(-5.75, 0.75, 2.35), aim=(-4.40, 2.30, 0.95), lens=30)
    print("RESULT:", "OK" if not FAILS else f"FAIL ({len(FAILS)})")
    return 0 if not FAILS else 1


Z_TOP = L2.SPOOL_TOP_Z

if __name__ == "__main__":
    sys.exit(main())
