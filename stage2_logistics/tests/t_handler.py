"""handler_robot.py check: build_arm FK (bpy) == kin.scaled_arm FK for scales 1.25 and 0.57 (3 random q each), handler
build contract, track inside its zone, set_track / set_q keys and the drag-chain bend, IK grasp of the spool on the
stage-1 positioner at the loading pose (pads touch the pipe leg), and a second pose placing the spool on the station.

    cd /home/user/robowelding && python3 stage2_logistics/tests/t_handler.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402
import bpy  # noqa: E402
import numpy as np  # noqa: E402
from cell import geom as G, positioner, spool, animation as A  # noqa: E402
import kin  # noqa: E402
import layout2 as L2  # noqa: E402
import handler_robot as HR  # noqa: E402
import gripper as GR  # noqa: E402

FAIL = []
R = L2.PIPE_R


def check(ok, msg):
    print(("[ok]   " if ok else "[FAIL] ") + msg, flush=True)
    if not ok:
        FAIL.append(msg)


def mw(ob):
    return G.np4(ob.matrix_world)


H.new_scene()
sc = bpy.context.scene

# ---------------------------------------------------------------- FK: bpy chain == kin.scaled_arm
rng = np.random.default_rng(7)
tmp = bpy.data.collections.new("FKTest")
sc.collection.children.link(tmp)
for scale in (1.25, 0.57):
    Tb = kin.tr(-6.0, 5.0, 0.4) @ kin.rotz(0.3)
    a = HR.build_arm(f"fk{int(scale * 100)}", scale, Tb, tmp)
    check(set(("arm", "base", "joints", "tool0", "links")) <= set(a), f"build_arm keys (scale {scale})")
    ref = kin.scaled_arm(scale)
    err = 0.0
    for _ in range(3):
        q = [rng.uniform(lo * 0.8, hi * 0.8) for lo, hi in ref.limits]
        HR.set_q(a, q)
        bpy.context.view_layer.update()
        err = max(err, float(np.abs(mw(a["tool0"]) - ref.fk(q, Tb)).max()))
    check(err < 1e-4, f"FK bpy vs kin.scaled_arm({scale}): max |dT| = {err:.2e}")
    bb_lo, bb_hi = H.world_bbox([a["links"]["link_2"]])
    HR.set_q(a, [0] * 6)
    bpy.context.view_layer.update()
    lo, hi = H.world_bbox([a["links"]["link_2"]])
    check(abs((hi[2] - lo[2]) - 1.332 * scale) < 0.02 * scale, f"link_2 mesh scaled (height {hi[2] - lo[2]:.3f} for scale {scale})")
for o in list(tmp.all_objects):
    bpy.data.objects.remove(o, do_unlink=True)
bpy.data.collections.remove(tmp)

# ---------------------------------------------------------------- stage-1 positioner + spool at the loading pose
pos = positioner.build()
sp = spool.build(name="spool")
G.set_parent(sp["root"], pos["mount"], keep_world=False)
positioner.set_tilt(pos, 0.0)
positioner.set_rot(pos, L2.POS_LOAD_ROT)
bpy.context.view_layer.update()
T_sp = A.spool_world(0.0, L2.POS_LOAD_ROT)
check(np.allclose(mw(sp["root"]), T_sp, atol=1e-6), "spool on the faceplate at A.spool_world(0, POS_LOAD_ROT)")

# ---------------------------------------------------------------- handler
rob = HR.build()
for k in ("collection", "arm", "carriage", "base", "joints", "tool0", "links", "gripper"):
    check(k in rob, f"build() returns key '{k}'")
check(rob["collection"].name == "Handler", "collection 'Handler'")
check(all(o.name.startswith(("hnd_", "grp_")) for o in rob["collection"].all_objects), "object names prefixed hnd_ / grp_")
track = rob["track"]
lo, hi = H.world_bbox(track)
x0, x1 = L2.HANDLER_TRACK_BED
check(lo[0] >= x0 - 1e-3 and hi[0] <= x1 + 1e-3 and max(-lo[1], hi[1]) < 0.75, f"track geometry inside x {x0}..{x1}, |y| < 0.75: {np.round(lo, 3)} .. {np.round(hi, 3)}")
for ob in HR.obstacles():
    c, s = np.array(ob["center"]), np.array(ob["size"])
    check(c[0] - s[0] / 2 >= x0 - 1e-6 and c[0] + s[0] / 2 <= x1 + 1e-6 and abs(c[1]) + s[1] / 2 < 0.75, f"obstacle {ob['name']} inside the track corridor")
olo = np.min([np.array(o["center"]) - np.array(o["size"]) / 2 for o in HR.obstacles()], axis=0)
ohi = np.max([np.array(o["center"]) + np.array(o["size"]) / 2 for o in HR.obstacles()], axis=0)
check(all(olo[k] <= lo[k] + 1e-3 and ohi[k] >= hi[k] - 1e-3 for k in range(3)), "obstacles() cover the static track geometry")
# carriage at both travel limits stays clear of the end stops
car_meshes = [o for o in rob["carriage"].children_recursive if o.type == 'MESH' and o.parent == rob["carriage"]]
for xt in L2.HANDLER_TRACK_X:
    HR.set_track(rob, xt)
    bpy.context.view_layer.update()
    clo, chi = H.world_bbox(car_meshes)
    check(clo[0] > x0 + 0.08 and chi[0] < x1 - 0.08, f"carriage at x={xt}: {clo[0]:.3f}..{chi[0]:.3f} clear of the end stops")
    bend = rob["chain_bend"].location[0]
    check(x0 + 0.15 < bend - HR.CHAIN_R and bend < xt + HR.CHAIN_OFF - 0.02, f"chain bend at {bend:.3f} inside the trough and behind the carriage end")
# chain band length is constant (links neither stretch nor slip)
dg = bpy.context.evaluated_depsgraph_get()
lens = []
for xt in (L2.HANDLER_TRACK_X[0], -5.5, L2.HANDLER_TRACK_X[1]):
    HR.set_track(rob, xt)
    bpy.context.view_layer.update()
    ev = rob["chain"]["curve"].evaluated_get(bpy.context.evaluated_depsgraph_get())
    me = ev.to_mesh()
    at = me.attributes.get("chain_u")
    lens.append(max(v.value for v in at.data) if at else -1)
    ev.to_mesh_clear()
check(max(lens) - min(lens) < 0.01 and abs(lens[0] - HR.CHAIN_LEN) < 0.02, f"chain length constant {np.round(lens, 3)} (design {HR.CHAIN_LEN:.3f})")
# setters
HR.set_track(rob, -6.0, frame=1)
HR.set_track(rob, -4.0, frame=11)
HR.set_q(rob, L2.HANDLER_Q_HOME, frame=1)
HR.set_q(rob, (0.3, -0.2, 0.4, 0.1, 1.0, 0.2), frame=11)
fc = {(f.data_path, f.array_index) for f in rob["carriage"].animation_data.action.fcurves}
check(fc == {("location", 0)}, "set_track keys only carriage location[0]")
check(rob["chain_bend"].animation_data is not None, "set_track keys the chain bend")
check(all(j.animation_data and j.animation_data.action for j in rob["joints"]), "set_q keys J1..J6")
sc.frame_set(6)
check(abs(rob["carriage"].location[0] + 5.0) < 1e-6, "carriage interpolates")
for o in [rob["carriage"], rob["chain_bend"]] + rob["joints"]:
    o.animation_data_clear()
sc.frame_set(1)
ntri = H.n_tris([o for o in rob["collection"].all_objects if o.type in ('MESH', 'CURVE')])
check(ntri < 150000, f"handler triangles {ntri}")

# ---------------------------------------------------------------- IK: grasp the spool on the positioner
arm = rob["arm"]
Tz_tcp = kin.tr(z=L2.GRIP_TCP_Z)


def grasp_tool0(T_spool, name="spool"):
    o, x, z, s = L2.GRASP[name]
    return T_spool @ kin.frame(x, z, o) @ kin.inv(Tz_tcp), s


def solve(T_target, x_track, seeds):
    base = HR.base_world(x_track)
    best = None
    for flip in (False, True):
        T = T_target @ (kin.rotz(math.pi) if flip else np.eye(4))
        q, ok, err = kin.ik6(arm, T, L2.HANDLER_Q_HOME, base, np.eye(4), seeds=seeds)
        cost = abs(q[3]) + abs(q[5]) + (0 if ok else 10)
        if best is None or cost < best[3]:
            best = (q, ok, err, cost)
    return best[:3]


seeds = [(0.0, 0.2, 0.2, 0.0, 1.2, 0.0), (0.0, 0.5, -0.2, 0.0, 1.3, 0.0), (0.0, 0.8, -0.5, 0.0, 1.2, 0.0)]
xA = L2.HANDLER_X_AT_POSITIONER
T1, s1 = grasp_tool0(T_sp)
q1, ok1, err1 = solve(T1, xA, seeds)
check(ok1 and err1 < 1e-3, f"IK grasp at the positioner (x={xA}): err {err1:.2e}, q = {np.round(q1, 3)}")
check(all(lo <= v <= hi for v, (lo, hi) in zip(q1, arm.limits)), "IK within joint limits")
HR.set_track(rob, xA)
HR.set_q(rob, q1)
GR.set_open(rob["gripper"], s1)
bpy.context.view_layer.update()
Ttcp = mw(rob["gripper"]["tcp"])
check(np.linalg.norm(Ttcp[:3, 3] - (T_sp @ np.array([*L2.GRASP["spool"][0], 1.0]))[:3]) < 2e-3, "gripper TCP on the pipe-leg grasp point")
# pads touch the pipe leg: distance of pad vertices from the pipe-leg axis
axis_p = (T_sp @ np.array([0.0, 0.0, L2.PIPE_AXIS_Z, 1.0]))[:3]
axis_d = T_sp[:3, 0]
dmin = 1.0
for o in rob["gripper"]["parts"]:
    if "_pad_" not in o.name:
        continue
    for v in o.data.vertices:
        p = np.array((o.matrix_world @ v.co)[:])
        d = p - axis_p
        dmin = min(dmin, float(np.linalg.norm(d - np.dot(d, axis_d) * axis_d)))
check(abs(dmin - R) < 2e-3, f"pads on the pipe leg: min distance to the pipe axis {dmin:.4f} (PIPE_R {R:.4f})")
# swivel: the stator (connector, lug, swivel body, anti-rotation bracket) follows J5 and stays put when J6 turns half a
# turn, the rotor (adapter, housing, jaws) turns with it; the hose end stays on the connector
check(len(rob.get("swivel", [])) >= 2 and all(o.parent == rob["joints"][4] for o in rob["gripper"]["stator"] + rob["swivel"]),
      "swivel stator + anti-rotation bracket parented to J5")
con0, hou0 = mw(rob["gripper"]["connector"]), mw(bpy.data.objects["grp_housing"])
HR.set_q(rob, [*q1[:5], q1[5] + math.pi])
bpy.context.view_layer.update()
con1, hou1 = mw(rob["gripper"]["connector"]), mw(bpy.data.objects["grp_housing"])
check(np.allclose(con0, con1, atol=1e-6) and not np.allclose(hou0[:3, :3], hou1[:3, :3], atol=1e-3),
      "J6 + 180 deg turns the gripper housing, not the stator / dress-pack connector")
ev = rob["hose"]["curve"].evaluated_get(bpy.context.evaluated_depsgraph_get())
me = ev.to_mesh()
hv = np.array([(ev.matrix_world @ v.co)[:] for v in me.vertices])
ev.to_mesh_clear()
dcon = float(np.linalg.norm(hv - con1[:3, 3], axis=1).min())
check(dcon < HR.HOSE_R + 0.002, f"hose ends on the connector (distance {dcon * 1000:.1f} mm)")
HR.set_q(rob, q1)
bpy.context.view_layer.update()
wrist = mw(rob["joints"][4])[:3, 3]
hood_z = A.L.FUME_HOOD["pos"][2] - A.L.FUME_HOOD["size"][2] / 2
check(wrist[2] < hood_z, f"wrist (J5) at z={wrist[2]:.2f} stays under the fume hood (z {hood_z:.2f})")

H.still("t_handler_positioner", cam=(-4.9, -4.3, 3.3), aim=(-1.9, 0.0, 1.35), lens=26)
HR.set_q(rob, [*q1[:5], q1[5] + math.pi])          # symmetric tongs: same grasp with J6 half a turn -> swivel at work
bpy.context.view_layer.update()
H.still("t_handler_grip", cam=(-1.25, -1.45, 2.55), aim=(-0.55, 0.0, 2.2), lens=32)

# ---------------------------------------------------------------- pose 2: placing the spool on the assembly station
F_st = kin.planar_frame((L2.STATION_ORIGIN[0], L2.STATION_ORIGIN[1], L2.STATION_ORIGIN[2] + 0.25), L2.STATION_XDIR)
T2, s2 = grasp_tool0(F_st)
xB = -4.55
q2, ok2, err2 = solve(T2, xB, [(1.3, 0.2, 0.2, 0.0, 1.2, 0.0), (1.4, 0.4, 0.0, 0.0, 1.2, 0.0)] + seeds)
check(ok2 and err2 < 1e-3, f"IK above the station (x={xB}): err {err2:.2e}, q = {np.round(q2, 3)}")
HR.set_track(rob, xB)
HR.set_q(rob, q2)
GR.set_open(rob["gripper"], s2)
bpy.context.view_layer.update()
sp["root"].parent = None
sp["root"].matrix_world = G.M(mw(rob["gripper"]["tcp"]) @ kin.inv(kin.frame(L2.GRASP["spool"][1], L2.GRASP["spool"][2], L2.GRASP["spool"][0])))
positioner.set_rot(pos, 0.0)
bpy.context.view_layer.update()
check(np.allclose(mw(sp["root"]), F_st, atol=2e-3), "carried spool frame = station frame + 0.25 m")
H.still("t_handler_carry", cam=(-2.2, -3.6, 2.9), aim=(-4.4, 1.2, 1.1), lens=26)
print("RESULT:", "PASS" if not FAIL else f"FAIL ({len(FAIL)})", flush=True)
sys.exit(1 if FAIL else 0)
