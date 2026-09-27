"""parts.py check: split the stage-1 spool (mounted on the stage-1 positioner at the loading angle) into flange / elbow /
pipe roots, move them with set_pose, grow tacks, build a kit copy and a finished (welded) copy; render 3 stills.

    cd /home/user/robowelding && python3 stage2_logistics/tests/t_parts.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402
import bpy  # noqa: E402
import numpy as np  # noqa: E402
from cell import geom as G, positioner, spool, materials, layout as L  # noqa: E402
import kin  # noqa: E402
import layout2 as L2  # noqa: E402
import parts as P  # noqa: E402

FAIL = []


def check(ok, msg):
    print(("[ok]   " if ok else "[FAIL] ") + msg, flush=True)
    if not ok:
        FAIL.append(msg)


def mw(ob):
    return G.np4(ob.matrix_world)


H.new_scene()
sc = bpy.context.scene
pos = positioner.build()
sp = spool.build(name="spool")
G.set_parent(sp["root"], pos["mount"], keep_world=False)
positioner.set_rot(pos, L2.POS_LOAD_ROT)
bpy.context.view_layer.update()
before = {o.name: mw(o) for o in sp["root"].children}
T_root = mw(sp["root"])

parts = P.split(sp)
bpy.context.view_layer.update()
check(sp["root"] is None and bpy.data.objects.get("spool_root") is None, "stage-1 spool root removed")
for k in ("flange", "elbow", "pipe", "roots", "tacks"):
    check(k in parts, f"split() returns key '{k}'")
check([r.name for r in parts["roots"]] == ["part_flange", "part_elbow", "part_pipe"], "part root names")
for r in parts["roots"]:
    check(np.allclose(mw(r), T_root, atol=1e-6), f"{r.name} at the spool root pose")
moved = max(float(np.abs(mw(bpy.data.objects[n]) - M).max()) for n, M in before.items())
check(moved < 1e-6, f"world poses of the stage-1 spool objects preserved by split (max dev {moved:.2e})")
expect = {"flange": {"spool_flange", "spool_flange_face", "spool_bead_A", "spool_tint_A"} | {f"spool_hole{k}" for k in range(12)},
          "elbow": {"spool_elbow", "spool_elbow_bevel_A", "spool_elbow_bevel_B", "spool_mark_elbow", "spool_bead_B", "spool_tint_B"},
          "pipe": {"spool_pipe", "spool_mark_pipe"}}
for k, names in expect.items():
    got = {o.name for o in parts[k].children if not o.name.startswith("part_tack")}
    check(got == names, f"{k} children {sorted(got - names)} missing {sorted(names - got)}")
check(len(parts["tacks"]["A"]) == 3 and len(parts["tacks"]["B"]) == 3, "3 + 3 tacks")
check(all(t.parent == parts["flange"] for t in parts["tacks"]["A"]) and all(t.parent == parts["elbow"] for t in parts["tacks"]["B"]),
      "tacks A on the flange, tacks B on the elbow")

# ---- tack positions (layout2 formulas) at a test pose
F = kin.planar_frame(L2.STATION_ORIGIN, L2.STATION_XDIR)
for r in parts["roots"]:
    P.set_pose(r, F, frame=1)
bpy.context.view_layer.update()
R = L2.PIPE_R
dev = 0.0
for i, a in enumerate(L2.TACKS_A):
    p = F @ np.array([R * math.cos(math.radians(a)), R * math.sin(math.radians(a)), L2.SEAM_A_Z, 1.0])
    P.set_tack(parts["tacks"]["A"][i], 1.0)
    bpy.context.view_layer.update()
    dev = max(dev, float(np.linalg.norm(mw(parts["tacks"]["A"][i])[:3, 3] - p[:3])))
for i, b in enumerate(L2.TACKS_B):
    p = F @ np.array([L2.SEAM_B_X, R * math.cos(math.radians(b)), L2.PIPE_AXIS_Z + R * math.sin(math.radians(b)), 1.0])
    P.set_tack(parts["tacks"]["B"][i], 1.0)
    bpy.context.view_layer.update()
    dev = max(dev, float(np.linalg.norm(mw(parts["tacks"]["B"][i])[:3, 3] - p[:3])))
check(dev < 1e-6, f"tack origins on the seam circles (max dev {dev:.2e})")
# place_tack while invisible (scale 0) keeps the frame
tb = parts["tacks"]["B"][2]
P.set_tack(tb, 0.0)
P.place_tack(tb, L2.TACKS_B[2] + 15.0)
P.set_tack(tb, 1.0)
bpy.context.view_layer.update()
bq = math.radians(L2.TACKS_B[2] + 15.0)
p = F @ np.array([L2.SEAM_B_X, R * math.cos(bq), L2.PIPE_AXIS_Z + R * math.sin(bq), 1.0])
check(np.allclose(mw(tb), F @ P.tack_frame("B", L2.TACKS_B[2] + 15.0), atol=1e-6) and np.linalg.norm(mw(tb)[:3, 3] - p[:3]) < 1e-6,
      "place_tack moves a tack along seam B (+15 deg) at scale 0")
P.place_tack(tb, L2.TACKS_B[2])
# tack length along the seam
tk = parts["tacks"]["A"][1]
xs = [v.co.x for v in tk.data.vertices]
check(abs((max(xs) - min(xs)) - L2.TACK_LEN) < 0.002, f"tack length {max(xs) - min(xs):.4f} ~ TACK_LEN {L2.TACK_LEN}")
# tack shape vs the stage-1 joint (tack frame: x along the seam, y = seam axis, z outward; seam circle centre (0, 0, -R)):
# every vertex stays under the finished stage-1 bead (tacks remain at s = 1 after welding), the base lies inside the
# bevel walls / root face (no gap under the tack), and the crown fills the groove above the root
r_o, wall = L.PIPE_OD / 2, L.PIPE_WALL
bev = (wall - 0.0016) / math.tan(math.radians(32))
bw = max(L.BEAD_WIDTH, 2 * bev + 0.004)
nv = len(tk.data.vertices) // 2
worst_bead, worst_base, crown = 1.0, 1.0, 0.0
for i, v in enumerate(tk.data.vertices):
    x, y, z = v.co
    r = math.hypot(x, z + R)
    if abs(y) < bw / 2:
        rb = r_o - 0.0012 + L.BEAD_HEIGHT * max(0.0, math.sin(math.pi * (y + bw / 2) / bw)) ** 0.8
        worst_bead = min(worst_bead, rb - r)
    else:
        worst_bead = min(worst_bead, r_o - r)
    if i >= nv:                                 # base layer: below the bevel wall of the grooved side
        wall_r = r_o - (wall - 0.0016) * max(0.0, 1.0 - abs(y) / bev)
        worst_base = min(worst_base, wall_r - r)
    else:
        crown = max(crown, r - r_o)
check(worst_bead > 0.0001, f"tack inside the finished bead envelope (min margin {worst_bead * 1000:.2f} mm)")
check(worst_base > 0.0002, f"tack base inside the bevel walls (min depth {worst_base * 1000:.2f} mm)")
check(0.001 < crown < L.BEAD_HEIGHT, f"tack crown {crown * 1000:.2f} mm above the pipe surface")

# ---- set_pose round trip + keys
T = kin.tr(-5.0, -0.8, 0.002) @ kin.rotz(0.7)
P.set_pose(parts["flange"], T, frame=10)
bpy.context.view_layer.update()
check(np.allclose(mw(parts["flange"]), T, atol=1e-6), "set_pose -> matrix_world")
fc = {(f.data_path, f.array_index) for f in parts["flange"].animation_data.action.fcurves}
check(("location", 0) in fc and ("rotation_quaternion", 3) in fc, "set_pose keys location + rotation_quaternion")
P.set_tack(parts["tacks"]["B"][0], 0.0, 0.0, frame=20)
P.set_tack(parts["tacks"]["B"][0], 1.0, 1.0, frame=30)
fc = {f.data_path for f in parts["tacks"]["B"][0].animation_data.action.fcurves}
check("scale" in fc and '["hot"]' in fc, "set_tack keys scale + hot")
sc.frame_set(25)
check(0.0 < parts["tacks"]["B"][0].scale[0] < 1.0, "tack grows between keys")

# ---- scene for the stills
sc.frame_set(1)
for r in parts["roots"]:
    P.set_pose(r, F)
for i, t in enumerate(parts["tacks"]["A"] + parts["tacks"]["B"]):
    P.set_tack(t, 1.0 if i != 4 else 0.6, hot=1.0 if i == 4 else 0.0)
# simple test stand under the station frame (test-only)
stand = G.box("test_stand", (0.62, 1.2, 0.78), (L2.STATION_ORIGIN[0], L2.STATION_ORIGIN[1] - 0.45, 0.39), bevel=0.01)
stand.data.materials.append(materials.get("painted", color="#2B2F36", roughness=0.5))

kit = P.build_kit(parts, "kit1")
check(set(kit) == {"flange", "elbow", "pipe"}, "build_kit keys")
kf = [o for o in kit["flange"].children if o.name == "part_kit1_flange_flange"]
check(len(kf) == 1 and kf[0].data == bpy.data.objects["spool_flange"].data, "kit meshes are linked duplicates")
check(not any("bead" in o.name or "tint" in o.name or "tack" in o.name for r in kit.values() for o in r.children),
      "kit has no beads / tints / tacks")
fin = P.build_finished(parts, "fin1")
fb = [o for o in fin.children if "_bead_" in o.name]
check(len(fb) == 2 and all(o.animation_data is None and abs(o["w0"][1] - 1.01) < 1e-6 for o in fb), "finished beads: no drivers, w0 prog 1.01")
ft = [o for o in fin.children if "_tint_" in o.name]
check(len(ft) == 2 and all(o["tint_on"] == 1.0 for o in ft), "finished tints on")

# loose kit parts on the floor: flange face down, elbow standing on its lower end, pipe lying (main parts stay keyed at
# the station frame for the tack still)
P.set_pose(kit["flange"], kin.tr(-5.75, -0.9, 0.002))
P.set_pose(kit["elbow"], kin.tr(-6.30, -0.4, -L2.SEAM_A_Z) @ kin.rotz(-0.6))
P.set_pose(kit["pipe"], kin.tr(-6.2, -1.4, R - L2.PIPE_AXIS_Z) @ kin.rotz(0.2))
P.set_pose(fin, kin.tr(-4.9, 0.4, 0.002) @ kin.rotz(-2.2))
bpy.context.view_layer.update()

# bead visibility of the finished copy: evaluated object attribute read by the shader
check(abs(fb[0]["w0"][1] - 1.01) < 1e-6 and fb[0]["w0"][2] == 1.0, "finished bead w0 = [0, 1.01, 1, 0]")
ntri = H.n_tris([o for o in bpy.data.objects if o.name.startswith("part_")])
print(f"[info] part_ triangles (kit + finished + tacks): {ntri}")
check(P.obstacles() == [], "obstacles() empty (parts are moved)")

H.still("t_parts_station", cam=(-3.55, 1.72, 1.62), aim=(-4.36, 2.28, 1.02), lens=38)
H.still("t_parts_loose", cam=(-3.4, -3.2, 1.9), aim=(-5.6, -0.4, 0.3), lens=32)
H.still("t_parts_finished", cam=(-3.85, -0.45, 1.05), aim=(-5.0, 0.3, 0.33), lens=38)
print("RESULT:", "PASS" if not FAIL else f"FAIL ({len(FAIL)})", flush=True)
sys.exit(1 if FAIL else 0)
