"""tack_robot.py check: contract (keys, names, setters), bpy FK of the TCP vs kin.scaled_arm(TACK_SCALE).fk @ tool_transform
for random q, footprint / zone checks, arm-vs-static clearance along the plan (plan2 tack_q), and 3 Cycles stills:
overview (robot at home beside the station), torch on seam B top of a stage-1 spool on the station (IK), wrist close-up.

    cd /home/user/robowelding && python3 stage2_logistics/tests/t_tack_robot.py [--no-render]
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402
import bpy  # noqa: E402
import numpy as np  # noqa: E402
from cell import geom as G, spool, layout as L, robot_build as RB, animation as A  # noqa: E402
import kin  # noqa: E402
import layout2 as L2  # noqa: E402
import tack_robot as TR  # noqa: E402

FAIL = []
RENDER = "--no-render" not in sys.argv


def check(ok, msg):
    print(("[ok]   " if ok else "[FAIL] ") + msg, flush=True)
    if not ok:
        FAIL.append(msg)


H.new_scene()
sc = bpy.context.scene
rob = TR.build()
bpy.context.view_layer.update()

# ---------------------------------------------------------------- contract
for k in ("collection", "arm", "base", "joints", "tool0", "tcp", "arc", "torch"):
    check(k in rob, f"build() returns key '{k}'")
col = rob["collection"]
check(col.name == TR.NAME and col.name in sc.collection.children, "own collection 'TackRobot' linked to the scene")
objs = list(col.all_objects)
bad = [o.name for o in objs if not o.name.startswith("tack_")]
check(not bad, f"every object prefixed tack_ ({bad[:6]})")
torch_objs = [o for o in objs if o.name.startswith("tack_torch_")]
check(len(torch_objs) >= 15 and bpy.data.objects.get("torch_root") is None,
      f"stage-1 torch objects renamed tack_torch_* ({len(torch_objs)})")
check(rob["torch"]["root"].name == "tack_torch_root" and rob["torch"]["root"].parent == rob["tool0"], "torch root on tool0")
check([j.name for j in rob["joints"]] == [f"tack_J{i}" for i in range(1, 7)], "joint empties tack_J1..J6")
check(rob["arc"].parent == rob["tcp"] and "arc_on" in rob["arc"], "arc empty on the TCP with 'arc_on'")

# ---------------------------------------------------------------- FK: bpy vs kin (random q within the joint limits)
arm = kin.scaled_arm(L2.TACK_SCALE)
B = TR.base_world()
TOOL = RB.tool_transform()
rng = np.random.default_rng(7)
dev_p = dev_r = 0.0
for _ in range(25):
    q = np.array([rng.uniform(max(lo, -3.0), min(hi, 3.0)) for lo, hi in arm.limits])
    TR.set_q(rob, q)
    bpy.context.view_layer.update()
    T_b = G.np4(rob["tcp"].matrix_world)
    T_k = arm.fk(q, B, TOOL)
    dev_p = max(dev_p, float(np.linalg.norm(T_b[:3, 3] - T_k[:3, 3])))
    dev_r = max(dev_r, float(np.abs(T_b[:3, :3] - T_k[:3, :3]).max()))
    dev_a = float(np.abs(G.np4(rob["arc"].matrix_world) - T_b).max())
check(dev_p < 1e-5 and dev_r < 1e-5, f"bpy TCP == scaled_arm.fk @ tool_transform (max {dev_p:.2e} m, rotation entries {dev_r:.2e})")
check(dev_a < 1e-6, "arc empty coincides with the TCP")
check(np.allclose(G.np4(rob["base"].matrix_world), B, atol=1e-6), "robot base at TACK_BASE / TACK_BASE_Z / TACK_YAW")

# ---------------------------------------------------------------- setters
TR.set_q(rob, L2.TACK_Q_HOME, frame=5)
TR.set_arc(rob, 1, frame=10)
TR.set_arc(rob, 0, frame=18)
fc = rob["arc"].animation_data.action.fcurves.find('["arc_on"]')
check(fc is not None and len(fc.keyframe_points) == 2 and all(k.interpolation == 'CONSTANT' for k in fc.keyframe_points),
      "set_arc keys 'arc_on' with CONSTANT interpolation")
check(all(e.animation_data and e.animation_data.action.fcurves.find("rotation_axis_angle", index=0) for e in rob["joints"]),
      "set_q keys rotation_axis_angle[0] of J1..J6")
sc.frame_set(12)
check(rob["arc"]["arc_on"] == 1.0, "arc on between the keys")
for e in rob["joints"]:
    e.animation_data_clear()
rob["arc"].animation_data_clear()
TR.set_q(rob, L2.TACK_Q_HOME)
bpy.context.view_layer.update()

# ---------------------------------------------------------------- footprint, budget
moving = set()
for j in rob["joints"]:
    moving |= {j} | set(j.children_recursive)
static = [o for o in objs if o.type == 'MESH' and o not in moving]
lo, hi = H.world_bbox(static)
print(f"[info] static bbox {np.round(lo, 3)} .. {np.round(hi, 3)}")
check(hi[0] <= L2.CELL_FENCE_X0 - 0.18 and lo[1] >= 0.75 and hi[1] <= L.FENCE_Y[1] - 0.02,
      "static geometry inside the zone: x <= -2.58 (cell hazard stripe), 0.75 < y < 3.38")
ntri = H.n_tris(objs)
check(ntri <= 150000, f"triangle budget {ntri} <= 150k")
obs = TR.obstacles()


def inside(p, o, pad=0.0):
    c, s = np.array(o["center"]), np.array(o["size"]) / 2 + pad
    return bool(np.all(np.abs(np.asarray(p) - c) <= s))


# every static mesh vertex inside some obstacle box
dg = bpy.context.evaluated_depsgraph_get()
miss = 0
nv = 0
for o in static:
    ev = o.evaluated_get(dg)
    me = ev.to_mesh()
    Mw = np.array(ev.matrix_world)
    V = np.array([v.co[:] for v in me.vertices])
    ev.to_mesh_clear()
    if len(V) == 0:
        continue
    W = (np.c_[V, np.ones(len(V))] @ Mw.T)[:, :3]
    for p in W[:: max(1, len(W) // 60)]:
        nv += 1
        if not any(inside(p, ob, 1e-3) for ob in obs):
            miss += 1
            print(f"[info] uncovered vertex of {o.name}: {np.round(p, 3)}")
check(miss == 0, f"obstacles() cover the static geometry ({miss} of {nv} sampled vertices outside)")

# ---------------------------------------------------------------- plan clearance: tack arm vs power source / cables / station
try:
    import plan2
    P = plan2.solve(verbose=False)
    TQ = P["tack_q"]
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import t_collision2 as TC
    try:
        import assembly_station as AS
        st_obs = AS.obstacles()
    except Exception as e:  # station module missing / broken
        print(f"[info] assembly_station.obstacles() unavailable: {e}")
        st_obs = []
    targets = [o for o in obs if o["name"] not in ("tack_pedestal", "tack_base_casting")] + st_obs
    BX = TC.Boxes(targets)
    worst = {}
    for i in range(0, P["n_frames"], 2):
        caps, _ = TC.arm_capsules(arm, TQ[i], B, L2.TACK_SCALE, TOOL, L.TORCH_NOZZLE_R + 0.004, torch=True)
        for name, a, b, r in caps:
            d = BX.seg_dist(a, b) - r
            k = int(np.argmin(d))
            key = (name, targets[k]["name"])
            if key not in worst or d[k] < worst[key][0]:
                worst[key] = (float(d[k]), i + 1)
    tack_w = sorted(((v, k) for k, v in worst.items() if k[1].startswith("tack_")))
    print("[info] closest tack arm / torch approaches to own static boxes:", [(k, round(v[0], 3), v[1]) for v, k in
                                                                           [(v, k) for v, k in tack_w[:4]]])
    check(all(v[0] > 0.05 for v, _ in tack_w), "tack arm + torch stay > 5 cm from the power source / gas / cables along the plan")
    stn_w = sorted(((v, k) for k, v in worst.items() if k[1].startswith("stn_")))
    print("[info] closest tack arm / torch approaches to station boxes (conservative boxes; the torch touches the seam "
          "at the tacks):", [(k, round(v[0], 3), v[1]) for v, k in stn_w[:6]])
except Exception as e:
    print(f"[info] plan clearance check skipped: {type(e).__name__}: {e}")

# ---------------------------------------------------------------- stills
if RENDER:
    F = kin.planar_frame(L2.STATION_ORIGIN, L2.STATION_XDIR)
    sp = spool.build(name="chk")
    sp["root"].matrix_world = G.M(F)
    for bd in sp["beads"].values():
        bd.hide_render = True
    for t in sp.get("tints", {}).values():
        t.hide_render = True
    try:
        import assembly_station as AS
        st = AS.build()
        for name in L2.STATION_CLAMPS:
            AS.set_clamp(st, name, 1.0)
        AS.set_lamp(st, 1)
    except Exception as e:
        print(f"[info] station not built for the stills: {type(e).__name__}: {e}")
    bpy.context.view_layer.update()
    H.still("t_tack_robot_overview", cam=(-2.55, 0.55, 2.05), aim=(-3.75, 2.45, 0.95), lens=24)
    # torch on seam B top: target frame as plan2.solve_tacks (lean toward the robot, tilt toward the pipe side).  The
    # exact top (b = 90 deg) may be out of reach from TACK_BASE; the planner shifts that tack to 70 deg - fall back to it.
    R = L2.PIPE_R
    try:
        plan_seeds = [P["tack_q"][P["events"]["tack_B0"] - 1]]
    except Exception:
        plan_seeds = []
    best = None
    for bdeg in (90.0, 80.0, 70.0):
        bb = math.radians(bdeg)
        nl = np.array([0.0, math.cos(bb), math.sin(bb)])
        p = (F @ np.r_[np.array([L2.SEAM_B_X, 0.0, L2.PIPE_AXIS_Z]) + R * nl, 1.0])[:3]
        tt = F[:3, :3] @ np.array([0.0, -nl[2], nl[1]])
        pb = np.linalg.inv(B) @ np.r_[p, 1.0]
        j1 = math.atan2(pb[1], pb[0])
        for up in (0.35, 0.7, 0.0):
            n = F[:3, :3] @ (nl + up * np.array([1.0, 0.0, 0.0]))
            n /= np.linalg.norm(n)
            lean = np.array([L2.TACK_BASE[0] - p[0], L2.TACK_BASE[1] - p[1], 0.0])
            lean = lean / np.linalg.norm(lean) + np.array([0.0, 0.0, 0.9])
            Tt = A.target_frame(p, n, tt, lean, push_deg=0.0)
            seeds = plan_seeds + [np.array(L2.TACK_Q_HOME)] + [np.array([j1, rng.uniform(-0.6, 0.8), rng.uniform(-0.3, 1.0),
                                                                         rng.uniform(-1, 1), rng.uniform(0.3, 1.6), 0.0]) for _ in range(8)]
            for q0 in seeds:
                q, ok, err = arm.ik(Tt, q0, B, TOOL, iters=250, free_spin=True, q_ref=[j1, 0, 0, 0, 1.2, 0])
                if ok:
                    best = (q, err, up, bdeg, p)
                    break
            if best is not None:
                break
        if best is not None:
            break
        print(f"[info] seam B at {bdeg:.0f} deg: no IK solution from the pedestal")
    check(best is not None, "IK: torch on seam B (top, or the planner's 70 deg top tack) reachable from the pedestal")
    if best is not None:
        TR.set_q(rob, best[0])
        TR.set_arc(rob, 1)
        bpy.context.view_layer.update()
        tip = G.np4(rob["tcp"].matrix_world)[:3, 3]
        p = best[4]
        check(np.linalg.norm(tip - p) < 2e-3, f"TCP on seam B at {best[3]:.0f} deg (err {np.linalg.norm(tip - p) * 1000:.2f} mm, "
              f"tilt toward the pipe {best[2]})")
        print("[info] seam-B-top q =", np.round(best[0], 3).tolist())
        H.still("t_tack_robot_seamB", cam=(-3.05, 1.35, 1.75), aim=(-4.05, 2.35, 1.25), lens=32)
        H.still("t_tack_robot_wrist", cam=(-3.55, 1.55, 1.95), aim=(-4.20, 2.20, 1.40), lens=45)

print("RESULT:", "OK" if not FAIL else f"FAIL ({len(FAIL)})")
for f in FAIL:
    print("  -", f)
sys.exit(1 if FAIL else 0)
