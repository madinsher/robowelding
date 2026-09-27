"""Check conveyor.py: contract, zone footprint, carrier frame, stage-1 spool fit on a carrier (flange on the seat ring,
V-post touching the pipe bottom, no interpenetration), carrier + spool clear of the static conveyor along the whole
travel, grasp corridors / gripper at the LOAD and END stations, obstacles() coverage, setter keys; renders 2 Cycles
stills (whole conveyor with spools on the carriers, carrier close-up).

    cd /home/user/robowelding && python3 stage2_logistics/tests/t_conveyor.py [--no-render]
"""
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
import conveyor  # noqa: E402
from cell import geom as G  # noqa: E402
from t_cassette import FAIL, check, world_mesh, overlap, min_dist, aabb, rect_dist, part, grasp_frame, gripper_checks  # noqa: E402

BODY = ["flange", "flange_face", "hole*", "elbow", "elbow_bevel_*", "pipe"]
ALL = ["*"]


def body_of(objs, name):
    return [o for o in objs if any(o.name[len(name) + 1:] == k or (k.endswith("*") and o.name[len(name) + 1:].startswith(k[:-1]))
                                   for k in BODY)]


def move_spool(objs, root_name, T):
    bpy.data.objects[root_name].matrix_world = G.M(T)
    bpy.context.view_layer.update()


def main():
    render = "--no-render" not in sys.argv
    H.new_scene()
    cv = conveyor.build()
    bpy.context.view_layer.update()
    CV = L2.CONVEYOR

    # ---------------------------------------------------------------- contract
    check(set(cv) >= {"collection", "carriers", "rollers"}, "build() keys collection / carriers / rollers")
    check(cv["collection"].name == "OutputConveyor", "collection name OutputConveyor")
    objs = list(cv["collection"].all_objects)
    check(all(o.name.startswith("conv_") for o in objs), "all objects prefixed conv_")
    check(len(cv["carriers"]) == 2, "2 carriers by default")
    n_roll = int(round((CV["x1"] - CV["x0"]) / CV["roller_pitch"]))
    check(len(cv["rollers"]) == n_roll, f"{n_roll} rollers")
    for i, x in enumerate((L2.CONV_LOAD_X, L2.CONV_QC_X)):
        T = kin.planar_frame((x, L2.CONV_SPOOL_Y, L2.CARRIER["seat_z"]), (0.0, 1.0))
        check(np.allclose(G.np4(cv["carriers"][i]["root"].matrix_world), T, atol=1e-6), f"carrier {i} root = spool frame at x = {x}")
    static = [o for o in cv["objects"] if o.type == 'MESH']
    carrier_objs = {i: [o for o in c["objects"] if o.type == 'MESH'] for i, c in enumerate(cv["carriers"])}
    meshes = [o for o in objs if o.type == 'MESH']
    tris = H.n_tris(meshes)
    check(tris < 150000, f"triangles {tris} < 150k")

    # ---------------------------------------------------------------- footprints
    hw = CV["width"] / 2
    ok = H.check_inside(static, (CV["x0"] - 0.05, CV["y"] - hw - 0.05, 0.0), (CV["x1"] + 0.05, CV["y"] + hw + 0.05, CV["top_z"] + 0.07),
                        label="conveyor frame")
    check(ok, "static conveyor inside its footprint (+5 cm for sensors / nuts)")
    lo, hi = H.world_bbox(static)
    check(hi[1] < -0.75 - 0.5, f"clear of the handler track corridor (y max {hi[1]:.3f})")
    check(lo[1] > L2.LOG_FENCE_Y[0] + 0.2, f"clear of the -Y fence (y min {lo[1]:.3f})")
    check(hi[0] < L2.CELL_FENCE_X0 - 0.5, f"clear of the cell fence (x max {hi[0]:.3f})")
    sx, sy = L2.CARRIER["size"]
    for i, c in enumerate(cv["carriers"]):
        x = c["root"].matrix_world.translation.x
        check(H.check_inside(carrier_objs[i], (x - sx / 2, CV["y"] - sy / 2, CV["top_z"]), (x + sx / 2, CV["y"] + sy / 2, 1.30),
                             label=f"carrier {i}"), f"carrier {i} inside its footprint {sx} x {sy} above the roller tops")

    # ---------------------------------------------------------------- spool fit on carrier 0 (at LOAD)
    T0 = conveyor.carrier_frame(L2.CONV_LOAD_X)
    sp = part("chk", ALL, T0)
    sbody = body_of(sp, "chk")
    car = carrier_objs[0]
    n = overlap(sbody, car)
    check(n == 0, f"spool does not intersect the carrier ({n})")
    ring = [o for o in car if o.name.endswith("_seat_ring")]
    vh = [o for o in car if "_vhead" in o.name and "bolt" not in o.name]
    d = min_dist([o for o in sbody if o.name.endswith("_flange") or o.name.endswith("_flange_face")], ring)
    check(d < 0.002, f"flange on the seat ring (gap {d * 1000:.2f} mm)")
    d = min_dist([o for o in sbody if o.name.endswith("_pipe")], vh)
    check(d < 0.002, f"pipe on the V-post (gap {d * 1000:.2f} mm)")
    lo_, hi_ = aabb([o for o in vh if o.name.endswith("_vhead")][0])
    xl = (kin.inv(T0) @ np.array([(lo_[0] + hi_[0]) / 2, (lo_[1] + hi_[1]) / 2, 0.0, 1.0]))[0]
    check(abs(xl - 0.86) < 0.02, f"V-post at spool-local x = {xl:.3f}")
    pins = [o for o in car if "_pin" in o.name]
    d = min_dist([o for o in sbody if o.name.endswith("_flange")], pins, n=3000)
    check(0.0005 < d < 0.006, f"centring pins just outside the flange rim ({d * 1000:.1f} mm)")

    # ---------------------------------------------------------------- travel: carrier + spool vs static conveyor
    bad_c, bad_s = [], []
    xs = list(np.arange(L2.CONV_LOAD_X, L2.CONV_END_X - 1e-6, -0.125)) + [L2.CONV_END_X]
    for x in xs:
        conveyor.set_carrier(cv, 0, x)
        move_spool(sp, "chk_root", conveyor.carrier_frame(x))
        if overlap(car, static):
            bad_c.append(round(x, 3))
        if overlap(sbody, static):
            bad_s.append(round(x, 3))
    check(not bad_c, f"carrier clear of the static conveyor along the travel ({len(xs)} positions, hits at {bad_c[:6]})")
    check(not bad_s, f"spool clear of the static conveyor along the travel (hits at {bad_s[:6]})")
    # END: carrier bumper just short of the hard stop
    conveyor.set_carrier(cv, 0, L2.CONV_END_X)
    bpy.context.view_layer.update()
    stop = [o for o in static if o.name.startswith("conv_stop_end_pad")]
    d = min_dist([o for o in car if "_bumper" in o.name and "bolt" not in o.name], stop, n=2000)
    check(0.002 < d < 0.008, f"carrier at END: bumper {d * 1000:.1f} mm from the hard stop")
    # 1 roller pitch clearance to the photo-eye / reflector / guides: carrier - guide gap
    guides = [o for o in static if o.name.startswith("conv_guide") and "_b" not in o.name[len("conv_guide0_0"):]]
    d = min_dist([o for o in car if o.name.endswith("_plate")], guides, n=2000)
    check(0.003 < d < 0.012, f"carrier plate .. side guides {d * 1000:.1f} mm")

    # ---------------------------------------------------------------- grasp corridors + gripper at LOAD / END
    boxes = {o.name: aabb(o) for o in static}
    for st, x in (("LOAD", L2.CONV_LOAD_X), ("END", L2.CONV_END_X)):
        T = conveyor.carrier_frame(x)
        g = T @ np.array([*L2.GRASP["spool"][0], 1.0])
        bad = [on for on, (l_, h_) in boxes.items() if h_[2] > g[2] and rect_dist(g[0], g[1], l_, h_) < 0.20]
        check(not bad, f"{st}: grasp corridor free ({bad[:4]})")
    conveyor.set_carrier(cv, 0, L2.CONV_LOAD_X)
    move_spool(sp, "chk_root", T0)
    gl = [("LOAD", *grasp_frame(T0, "spool"), sbody)]
    gripper_checks(gl, static + car, {"LOAD": sbody}, "conveyor+carrier")

    # ---------------------------------------------------------------- obstacles()
    obs = conveyor.obstacles()
    check(all(set(o) >= {"name", "center", "size", "yaw"} for o in obs), f"obstacles(): {len(obs)} boxes")

    def inside_any(p):
        return any(all(abs(p[q] - o["center"][q]) <= o["size"][q] / 2 + 1e-3 for q in range(3)) for o in obs)
    miss = {}
    for o in static:
        v, _ = world_mesh(o)
        k = sum(1 for p in v if not inside_any(p))
        if k:
            miss[o.name] = k
    check(not miss, f"obstacle boxes cover the static conveyor (outside: {sorted(miss.items())[:8]})")
    for st, x in (("LOAD", L2.CONV_LOAD_X), ("END", L2.CONV_END_X)):
        g = conveyor.carrier_frame(x) @ np.array([*L2.GRASP["spool"][0], 1.0])
        top = max(o["center"][2] + o["size"][2] / 2 for o in obs)
        check(top < g[2] - L2.PIPE_R - 0.06 - 0.1, f"{st}: obstacle tops ({top:.3f}) below the open-jaw envelope")

    # ---------------------------------------------------------------- setters
    sc = bpy.context.scene
    conveyor.set_carrier(cv, 0, L2.CONV_LOAD_X, frame=1)
    conveyor.set_carrier(cv, 0, L2.CONV_QC_X, frame=97)
    conveyor.set_rollers(cv, 0.0, frame=1)
    conveyor.set_rollers(cv, abs(L2.CONV_QC_X - L2.CONV_LOAD_X), frame=97)
    ad = cv["carriers"][0]["root"].animation_data
    paths = sorted({(fc.data_path, fc.array_index) for fc in ad.action.fcurves})
    check(paths == [("location", 0)], f"set_carrier keys only location[0] {paths}")
    rp = sorted({(fc.data_path, fc.array_index) for r in cv["rollers"] for fc in r.animation_data.action.fcurves})
    check(rp == [("rotation_euler", 1)], f"set_rollers keys only rotation_euler[1] {rp}")
    check(all(r.animation_data is not None for r in cv["rollers"]), "every roller keyed")
    sc.frame_set(97)
    r0 = cv["rollers"][0]
    exp = -abs(L2.CONV_QC_X - L2.CONV_LOAD_X) / L2.CONVEYOR["roller_r"]
    check(abs(r0.rotation_euler[1] - exp) < 1e-5, f"roller angle at frame 97 = {r0.rotation_euler[1]:.2f} rad ({exp:.2f})")
    check(abs(cv["carriers"][0]["root"].location[0] - L2.CONV_QC_X) < 1e-6, "carrier at QC at frame 97")
    # rotation about +Y with a decreasing angle: the roller top (0, 0, r) moves with w x r = (w r, 0, 0), w < 0 -> -X
    check(exp < 0 and abs(r0.matrix_world.to_3x3().col[1].y - 1.0) < 1e-6, "roller axis = world Y, top surface moves toward -X")

    # ---------------------------------------------------------------- stills
    if render:
        # carrier 0 with the spool at LOAD, carrier 1 at QC with a second spool
        for a in list(sc.objects):
            if a.animation_data:
                a.animation_data_clear()
        conveyor.set_carrier(cv, 0, L2.CONV_LOAD_X)
        move_spool(sp, "chk_root", T0)
        part("chk2", ALL, conveyor.carrier_frame(L2.CONV_QC_X))
        H.still("t_conveyor_overview", cam=(-2.3, 0.2, 2.9), aim=(-5.6, -2.45, 0.7), lens=26)
        H.still("t_conveyor_carrier", cam=(-3.05, -1.05, 1.55), aim=(-4.2, -2.35, 0.9), lens=32)
        # -X end: carrier 1 against the END hard stop, gear motor + drive guard, photo-eye / reflector
        conveyor.set_carrier(cv, 1, L2.CONV_END_X)
        bpy.data.objects["chk2_root"].matrix_world = G.M(conveyor.carrier_frame(L2.CONV_END_X))
        H.still("t_conveyor_end", cam=(-6.55, -0.55, 1.25), aim=(-7.75, -2.05, 0.55), lens=30)
    print("RESULT:", "PASS" if not FAIL else f"FAIL ({len(FAIL)})", flush=True)
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
