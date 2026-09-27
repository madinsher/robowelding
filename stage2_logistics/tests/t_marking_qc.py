"""Check marking_qc.py: contract, one root for the arch (vfx.laser_line skips it), zone footprint, spool clearance
under the beam / heads / posts while the carrier passes (>= 0.08 m) for carriage positions over the whole travel,
scanner sensor frame + vfx.laser_line crossing seam B, laser-mark decal (placement, reveal / hot, writing front and
beam aim), setters, energy-chain drivers (simple expressions), obstacles() coverage; renders 3 Cycles stills (arch
overview, scan line at frame 2, mark half revealed with the marking beam on).

    cd /home/user/robowelding && python3 stage2_logistics/tests/t_marking_qc.py [--no-render]
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
import layout2 as L2  # noqa: E402
import conveyor  # noqa: E402
import marking_qc as QC  # noqa: E402
from cell import geom as G, spool, vfx  # noqa: E402
from t_cassette import FAIL, check, world_mesh, bvh, min_dist, aabb  # noqa: E402

BODY = ("_flange", "_flange_face", "_elbow", "_pipe", "_bead_A", "_bead_B")


def root_of(ob):
    while ob.parent is not None:
        ob = ob.parent
    return ob


def fc_paths(idb):
    ad = idb.animation_data
    return sorted({(fc.data_path, fc.array_index) for fc in ad.action.fcurves}) if ad and ad.action else []


def main():
    render = "--no-render" not in sys.argv
    H.new_scene()
    sc = bpy.context.scene
    cv = conveyor.build(n_carriers=1)
    qc = QC.build()
    bpy.context.view_layer.update()
    A = L2.QC_ARCH

    # ---------------------------------------------------------------- contract
    keys = {"collection", "root", "carriage", "marker", "beam", "scanner", "sensor", "tower", "display"}
    check(set(qc) >= keys, f"build() keys {sorted(keys)}")
    check(qc["collection"].name == "MarkingQC", "collection name MarkingQC")
    objs = list(qc["collection"].all_objects)
    check(all(o.name.startswith("qc_") for o in objs), "all objects prefixed qc_")
    check(qc["root"].name == "qc_root", "root empty is 'qc_root'")
    bad = [o.name for o in objs if root_of(o) is not qc["root"]]
    check(not bad, f"every arch object is under qc_root ({bad[:5]})")
    check(set(qc["tower"]) == {"red", "amber", "green"}, "tower lamps red / amber / green")
    tris = H.n_tris([o for o in objs if o.type == 'MESH'])
    check(tris < 150000, f"triangles {tris} < 150k")

    # ---------------------------------------------------------------- footprint (zone of the arch)
    static = [o for o in qc["objects"] if o.type == 'MESH' and not o.name.startswith("qc_beam")]
    lo, hi = H.world_bbox(static)
    check(lo[1] > L2.LOG_FENCE_Y[0] + 0.03, f"clear of the -Y fence (y min {lo[1]:.3f})")
    check(hi[1] < -0.75 - 0.05, f"clear of the handler track corridor (y max {hi[1]:.3f})")
    check(lo[0] > A["x"] - 0.50 and hi[0] < A["x"] + 0.50, f"arch within x = QC_X +- 0.5 ({lo[0]:.2f} .. {hi[0]:.2f})")
    check(hi[2] < L2.LOG_FENCE_H + 0.5, f"height {hi[2]:.2f} m")
    conv_static = [o for o in cv["objects"] if o.type == 'MESH']
    tq, _ = bvh(static)
    tc, _ = bvh(conv_static)
    check(not tq.overlap(tc), "arch does not intersect the conveyor")

    # ---------------------------------------------------------------- spool on the carrier: fit under the arch
    sp = spool.build(name="chk")
    body = [o for o in sp["collection"].all_objects if o.type == 'MESH' and any(o.name == "chk" + s for s in BODY)]
    head_objs = [o for o in objs if o.type in ('MESH', 'CURVE') and (root_of(o) is qc["root"])
                 and not o.name.startswith(("qc_beam", "qc_mark"))]
    car_objs = [o for o in head_objs if o.parent is qc["carriage"]]
    car_meshes = [o for o in car_objs if o.type == 'MESH']
    y_range = qc["carriage_range"]
    ys = [y_range[0], L2.CONV_SPOOL_Y + L2.QC_MARK_LOCAL[0], L2.CONV_SPOOL_Y + L2.SEAM_B_X, y_range[1]]
    worst = (9.0, None)
    for y in ys:
        QC.set_carriage(qc, y)
        bpy.context.view_layer.update()
        zlo = min(aabb(o)[0][2] for o in car_meshes)
        top = L2.CARRIER["seat_z"] + L2.SPOOL_TOP_Z
        if zlo - top < worst[0]:
            worst = (zlo - top, y)
    check(worst[0] >= 0.08, f"heads / carriage >= 0.08 above the spool top everywhere (min {worst[0]:.3f} m at y {worst[1]:.2f})")
    dmin = (9.0, None)
    for x in (A["x"] + 0.45, A["x"] + 0.2, A["x"], A["x"] - 0.2):
        conveyor.set_carrier(cv, 0, x)
        sp["root"].matrix_world = G.M(conveyor.carrier_frame(x))
        for y in (y_range[0], L2.CONV_SPOOL_Y + L2.SEAM_B_X, y_range[1]):
            QC.set_carriage(qc, y)
            bpy.context.view_layer.update()
            d = min_dist(body, [o for o in head_objs if o.type == 'MESH'], n=1500)
            if d < dmin[0]:
                dmin = (d, (round(x, 2), round(y, 2)))
    check(dmin[0] >= 0.08, f"spool >= 0.08 m from every arch part while passing (min {dmin[0]:.3f} m at carrier x / carriage y {dmin[1]})")
    car_body = [o for o in cv["carriers"][0]["objects"] if o.type == 'MESH']
    conveyor.set_carrier(cv, 0, A["x"])
    bpy.context.view_layer.update()
    d = min_dist(car_body, static, n=1500)
    check(d >= 0.08, f"carrier pallet {d:.3f} m from the arch posts / HMI")

    # ---------------------------------------------------------------- sensor frame, laser line across seam B
    sp["root"].matrix_world = G.M(conveyor.carrier_frame(A["x"]))
    F = conveyor.carrier_frame(A["x"])
    scan_w = F @ np.array([*L2.QC_SCAN_LOCAL, 1.0])
    QC.set_carriage(qc, scan_w[1])
    bpy.context.view_layer.update()
    Ms = qc["sensor"].matrix_world
    zs = Ms.to_3x3() @ mathutils.Vector((0, 0, 1))
    yv = Ms.to_3x3() @ mathutils.Vector((0, 1, 0))
    check(zs.z < -0.999, "sensor +Z looks straight down")
    check(abs(abs(yv.y) - 1.0) < 1e-6, "sensor Y (line direction) along the pipe axis (world Y)")
    win = Ms @ mathutils.Vector(vfx.LASER_WINDOW_FALLBACK)
    lw = [o for o in objs if o.name == "qc_sc_laserwin"][0]
    lwc = lw.matrix_world.translation
    check((win - lwc).length < 0.003, f"vfx fallback window = scanner laser window ({(win - lwc).length * 1000:.1f} mm)")
    check(abs(Ms.translation.x - scan_w[0]) < 1e-6 and abs(Ms.translation.y - scan_w[1]) < 1e-6, "sensor above the seam-B top point")
    check(bpy.data.objects.get("torch_laser_lens") is None, "no stage-1 torch lens in the test scene (fallback window is used)")
    sc.frame_set(1)
    line = vfx.laser_line(qc["sensor"], [(1, 3)])
    sc.frame_set(2)
    dg = bpy.context.evaluated_depsgraph_get()
    ev = line.evaluated_get(dg)
    me = ev.to_mesh()
    pts = [ev.matrix_world @ v.co for v in me.vertices]
    ev.to_mesh_clear()
    yl = [p.y for p in pts]
    zl = [p.z for p in pts]
    check(min(yl) < scan_w[1] - 0.03 and max(yl) > scan_w[1] + 0.03, f"laser line crosses seam B (y {min(yl):.3f} .. {max(yl):.3f}, seam {scan_w[1]:.3f})")
    check(all(abs(p.x - scan_w[0]) < 0.01 for p in pts), "laser line on the pipe crest (x)")
    check(max(zl) < scan_w[2] + 0.01 and min(zl) > scan_w[2] - 0.03, f"laser line on the steel surface (z {min(zl):.3f} .. {max(zl):.3f})")
    near = [p.z for p in pts if abs(p.y - scan_w[1]) < 0.004]
    check(near and max(near) > scan_w[2] + 0.001, f"line rides over the bead (z {max(near) if near else 0:.4f} vs crest {scan_w[2]:.4f})")

    # ---------------------------------------------------------------- decal
    mark = QC.build_mark(sp["root"])
    bpy.context.view_layer.update()
    check(mark.parent is sp["root"] and mark.name.startswith("qc_mark"), "decal 'qc_mark' parented to the pipe root")
    lo_, hi_ = aabb(mark)
    c_w = mathutils.Vector([(lo_[k] + hi_[k]) / 2 for k in range(3)])
    c_l = sp["root"].matrix_world.inverted() @ c_w
    check(abs(c_l.x - L2.QC_MARK_LOCAL[0]) < 0.005 and abs(c_l.y) < 0.002, f"decal centred at QC_MARK_LOCAL (local x {c_l.x:.3f})")
    v, _ = world_mesh(mark)
    Mi = sp["root"].matrix_world.inverted()
    rr = [math.hypot((Mi @ p).y, (Mi @ p).z - L2.PIPE_AXIS_Z) for p in v]
    check(all(L2.PIPE_R < r < L2.PIPE_R + 0.001 for r in rr), f"decal wrapped on the pipe surface (r {min(rr):.4f} .. {max(rr):.4f})")
    xs_ = [(Mi @ p).x for p in v]
    bead_w = 0.0143
    check(min(xs_) > L2.SEAM_B_X + bead_w + 0.005 and max(xs_) < 0.60, f"decal clear of the seam-B bead and the stage-1 marking (x {min(xs_):.3f} .. {max(xs_):.3f})")
    names = {n.attribute_name for n in mark.active_material.node_tree.nodes if n.type == 'ATTRIBUTE'}
    check(names == {"reveal", "hot"}, f"decal shader reads object attributes {sorted(names)}")
    QC.set_marker(qc, 1.0)
    QC.set_mark(mark, 0.5, 1.0)
    QC.set_carriage(qc, L2.CONV_SPOOL_Y + L2.QC_MARK_LOCAL[0])
    bpy.context.view_layer.update()
    front = mark["front"]
    fl = Mi @ front.matrix_world.translation
    check(abs(fl.x - (QC.MARK_X[0] + 0.5 * (QC.MARK_X[1] - QC.MARK_X[0]))) < 1e-4, f"writing front follows reveal (local x {fl.x:.4f})")
    dg = bpy.context.evaluated_depsgraph_get()
    be = qc["beam"].evaluated_get(dg)
    tip = be.matrix_world @ mathutils.Vector((0, 1, 0))
    check((tip - front.matrix_world.translation).length < 0.002, f"marker beam ends at the writing front ({(tip - front.matrix_world.translation).length * 1000:.1f} mm)")
    lens = be.matrix_world.translation
    check(lens.z > tip.z + 0.15, "beam starts at the lens above the pipe")
    spot = qc["spot"]
    check((spot.matrix_world.translation - front.matrix_world.translation).length < 0.02, "marking spot light at the writing front")

    # ---------------------------------------------------------------- setters
    for ob in [qc["carriage"], mark, qc["beam"], qc["beam_glow"], qc["led"], qc["display"], *qc["tower"].values()]:
        if ob.animation_data:
            ob.animation_data_clear()
    QC.set_carriage(qc, -3.0, frame=1)
    QC.set_carriage(qc, -2.4, frame=25)
    check(fc_paths(qc["carriage"]) == [("location", 1)], f"set_carriage keys only location[1] {fc_paths(qc['carriage'])}")
    QC.set_carriage(qc, -9.0)
    check(abs(qc["carriage"].location[1] - qc["carriage_range"][0]) < 1e-6, "set_carriage clamps to the travel range")
    QC.set_marker(qc, 0.0, frame=1)
    QC.set_marker(qc, 1.0, frame=10)
    QC.set_marker(qc, 0.0, frame=20)
    check(fc_paths(qc["beam"]) == [('["on"]', 0)] and fc_paths(qc["led"]) == [('["on"]', 0)], "set_marker keys the 'on' properties")
    check(fc_paths(qc["spot"].data) == [("energy", 0)], "set_marker keys the spot light energy")
    QC.set_mark(mark, 0.0, 0.0, frame=10)
    QC.set_mark(mark, 1.0, 1.0, frame=58)
    check(fc_paths(mark) == [('["hot"]', 0), ('["reveal"]', 0)], f"set_mark keys reveal / hot {fc_paths(mark)}")
    for st, f in (("off", 1), ("amber", 5), ("green", 30), ("off", 60)):
        QC.set_tower(qc, st, frame=f)
    for stt, f in ((0, 1), (1, 10), ("scanning", 30), (3, 50)):
        QC.set_display(qc, stt, frame=f)
    sc.frame_set(15)
    check(qc["beam"]["on"] == 1.0 and abs(qc["spot"].data.energy - QC.SPOT_POWER) < 1e-6, "marker on at frame 15 (CONSTANT)")
    sc.frame_set(34)
    check(abs(qc["carriage"].location[1] - (-2.4)) < 1e-6, "carriage at -2.4 at frame 25+")
    fl = Mi @ front.matrix_world.translation
    check(abs(fl.x - (QC.MARK_X[0] + 0.5 * (QC.MARK_X[1] - QC.MARK_X[0]))) < 2e-3, f"keyed reveal drives the writing front (local x {fl.x:.4f})")
    check(qc["tower"]["green"]["on"] == 1.0 and qc["tower"]["amber"]["on"] == 0.0, "tower green at frame 34")
    check(qc["display"]["state"] == 2.0, "display 'scanning' at frame 34")
    check(0.45 < mark["reveal"] < 0.55, f"reveal half-way at frame 34 ({mark['reveal']:.2f})")
    sc.frame_set(21)
    check(qc["beam"]["on"] == 0.0 and qc["led"]["on"] == 0.0, "marker off at frame 21")

    # ---------------------------------------------------------------- energy chain drivers
    drv = [fc.driver for ob in qc["chain"] for fc in ob.animation_data.drivers]
    check(len(drv) == 4 and all(d.is_simple_expression and d.is_valid for d in drv), "energy-chain drivers are valid simple expressions")
    drv = [fc.driver for fc in front.animation_data.drivers]
    check(len(drv) == 2 and all(d.is_simple_expression and d.is_valid for d in drv), "writing-front drivers are valid simple expressions")
    for y in qc["carriage_range"]:
        QC.set_carriage(qc, y)
        bpy.context.view_layer.update()
        up = [o for o in qc["chain"] if o.name == "qc_chain_upper"][0]
        end = (up.matrix_world @ mathutils.Vector((0, 1, 0))).y
        check(abs(end - y) < 1e-4, f"chain upper run ends at the carriage (y {y:.2f}: {end:.3f})")
        lo_ = min(aabb(o)[0][1] for o in qc["chain"])
        check(lo_ > QC.BEAM_Y[0], f"chain loop stays on the beam (y min {lo_:.3f})")

    # ---------------------------------------------------------------- obstacles()
    obs = QC.obstacles()
    check(all(set(o) >= {"name", "center", "size", "yaw"} for o in obs), f"obstacles(): {len(obs)} boxes")

    def inside_any(p):
        return any(all(abs(p[q] - o["center"][q]) <= o["size"][q] / 2 + 1e-3 for q in range(3)) for o in obs)
    miss = {}
    for y in qc["carriage_range"]:
        QC.set_carriage(qc, y)
        bpy.context.view_layer.update()
        for o in head_objs + qc["chain"]:
            if o.type != 'MESH':
                continue
            v, _ = world_mesh(o)
            k = sum(1 for p in v if not inside_any(p))
            if k:
                miss[o.name] = k
    check(not miss, f"obstacle boxes cover the arch at both carriage ends (outside: {sorted(miss.items())[:8]})")
    for st, x in (("LOAD", L2.CONV_LOAD_X), ("END", L2.CONV_END_X)):
        g = conveyor.carrier_frame(x) @ np.array([*L2.GRASP["spool"][0], 1.0])
        near = [o["name"] for o in obs if abs(o["center"][0] - g[0]) < o["size"][0] / 2 + 0.5]
        check(not near, f"{st}: no arch obstacle within 0.5 m of the grasp point ({near})")

    # ---------------------------------------------------------------- scan_line(): torch lens hidden from the window lookup
    dummy = bpy.data.objects.new("torch_laser_lens", None)
    sc.collection.objects.link(dummy)
    dummy.location = (2.0, 0.0, 1.5)
    QC.set_carriage(qc, scan_w[1])
    bpy.context.view_layer.update()
    ln2 = QC.scan_line(qc, [(40, 40)])
    fan = [o for o in bpy.data.objects if o.name.startswith("LaserFan") and o.parent is qc["sensor"]][-1]
    sc.frame_set(40)
    dg = bpy.context.evaluated_depsgraph_get()
    ev = fan.evaluated_get(dg)
    me = ev.to_mesh()
    w0 = ev.matrix_world @ me.vertices[0].co
    ev.to_mesh_clear()
    check((w0 - lwc_at(qc)).length < 0.003 and dummy.name == "torch_laser_lens", "scan_line(): fan starts at the scanner window, lens name restored")
    for ob in (ln2, fan, dummy):
        bpy.data.objects.remove(ob, do_unlink=True)

    # ---------------------------------------------------------------- stills
    if render:
        for ob in list(bpy.data.objects):
            if ob.name.startswith(("qc_", "conv_")) and ob.animation_data and ob.animation_data.action:
                ob.animation_data.action = None
        for idb in (qc["spot"].data,):
            if idb.animation_data:
                idb.animation_data_clear()
        conveyor.set_carrier(cv, 0, A["x"])
        sp["root"].matrix_world = G.M(conveyor.carrier_frame(A["x"]))
        QC.set_tower(qc, "amber")
        QC.set_display(qc, "marking")
        QC.set_marker(qc, 1.0)
        QC.set_mark(mark, 0.5, 1.0)
        QC.set_carriage(qc, L2.CONV_SPOOL_Y + L2.QC_MARK_LOCAL[0])
        H.still("t_marking_qc_arch", cam=(-3.75, 0.1, 3.0), aim=(-6.0, -2.3, 1.25), lens=30, frame=5)
        H.still("t_marking_qc_mark", cam=(-5.62, -2.02, 1.86), aim=(-6.0, -2.39, 1.43), lens=50, frame=5)
        # scan: line crossing seam B at frame 2, marker off, mark fully revealed and cooled, display 'scanning'
        QC.set_marker(qc, 0.0)
        QC.set_mark(mark, 1.0, 0.0)
        QC.set_display(qc, "scanning")
        QC.set_carriage(qc, scan_w[1])
        H.still("t_marking_qc_scan", cam=(-5.40, -1.65, 1.95), aim=(-6.0, -2.47, 1.40), lens=45, frame=2)
    print("RESULT:", "PASS" if not FAIL else f"FAIL ({len(FAIL)})", flush=True)
    return 0 if not FAIL else 1


def lwc_at(qc):
    lw = bpy.data.objects["qc_sc_laserwin"]
    return lw.matrix_world.translation


if __name__ == "__main__":
    sys.exit(main())
