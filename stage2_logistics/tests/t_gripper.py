"""gripper.py check: pad contact on a DN250 pipe at s = 0 (numeric), approach clearance of a D0.36 cylinder at s = 1,
envelope, flange-hub contact at s = GRASP['flange'][3], cylinder linkage (rod eye stays on the jaw pin), setter keys;
stills: three grippers (closed on a pipe / open / holding a loose flange) + a close-up of the pads.

    cd /home/user/robowelding && python3 stage2_logistics/tests/t_gripper.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402
import bpy  # noqa: E402
import mathutils  # noqa: E402
import numpy as np  # noqa: E402
from cell import geom as G, materials, spool  # noqa: E402
import kin  # noqa: E402
import layout2 as L2  # noqa: E402
import gripper as GR  # noqa: E402
import parts as P  # noqa: E402

FAIL = []
R = L2.PIPE_R
TCP = L2.GRIP_TCP_Z


def check(ok, msg):
    print(("[ok]   " if ok else "[FAIL] ") + msg, flush=True)
    if not ok:
        FAIL.append(msg)


def verts_in(ob, frame_inv):
    """Evaluated mesh vertices of ob in the coordinates of frame_inv (mathutils inverse matrix)."""
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    me = ev.to_mesh()
    M = frame_inv @ ev.matrix_world
    out = np.array([(M @ v.co)[:] for v in me.vertices])
    ev.to_mesh_clear()
    return out


def down_tool(x, y, z):
    """tool0 pose pointing down (tool z = world -z, tool x = world x)."""
    return kin.tr(x, y, z) @ kin.rotx(math.pi)


H.new_scene()
col = bpy.data.collections.new("GripperTest")
bpy.context.scene.collection.children.link(col)
tool_a = G.empty("test_tool0_a", matrix=down_tool(-5.6, 0.0, 1.30), collection=col)
g = GR.build(tool_a, col, name="grp")
for k in ("root", "tcp", "jaws", "parts", "stator", "connector"):
    check(k in g, f"build() returns key '{k}'")
check(len(g["jaws"]) == 2, "two jaws")
bpy.context.view_layer.update()
check(np.allclose(G.np4(g["tcp"].matrix_world), down_tool(-5.6, 0.0, 1.30) @ kin.tr(z=TCP), atol=1e-6), "tcp = tool0 @ Tz(GRIP_TCP_Z)")

root_inv = g["root"].matrix_world.inverted()
meshes = [o for o in g["parts"] if o.type == 'MESH']
# ---- swivel: the stator (kept still by the robot while J6 turns) and the rotor must not share any swept volume:
# stator between the adapter plate and the housing; below the housing the rotor stays inside the adapter radius
check("stator" in g and g["connector"] in g["stator"] and all(o in g["parts"] for o in g["stator"] if o.type == 'MESH'),
      "stator listed (swivel body, lug, connector)")
stat = [o for o in g["stator"] if o.type == 'MESH']
sv = np.concatenate([verts_in(o, root_inv) for o in stat])
z_lo, z_hi = GR.ADAPTER["z"][1], GR.HOUSING["z"][0]
check(sv[:, 2].min() >= z_lo - 1e-4 and sv[:, 2].max() <= z_hi + 1e-4,
      f"stator inside the swivel band z [{sv[:, 2].min():.3f}, {sv[:, 2].max():.3f}] of [{z_lo}, {z_hi}]")
rotor_r = 0.0
for s_ in (0.0, 1.0):
    GR.set_open(g, s_)
    bpy.context.view_layer.update()
    for o in meshes:
        if o in stat:
            continue
        v = verts_in(o, root_inv)
        low = v[v[:, 2] < z_hi - 1e-4]
        if len(low):
            rotor_r = max(rotor_r, float(np.hypot(low[:, 0], low[:, 1]).max()))
check(rotor_r <= GR.ADAPTER["r"] + 1e-4, f"rotor parts in the swivel band stay inside the adapter radius ({rotor_r:.4f})")
cz = min(float(verts_in(o, root_inv)[:, 2].min()) for o in g["parts"] if o.type == 'CURVE')
check(cz > z_hi, f"hoses / sensor cables (rotor) stay below the housing top (min z {cz:.3f})")
pads = [o for o in meshes if "_pad_" in o.name]
jaw_side = [o for o in meshes if o not in pads and any(o.parent == j for j in g["jaws"])]
movers = [o for o in meshes if o.parent in g["jaws"] or o.parent in g["cylinders"] or o.parent in g["rods"]]

# ---- s = 0: pads touch r = PIPE_R about the TCP x axis; nothing else inside the pipe
GR.set_open(g, 0.0)
bpy.context.view_layer.update()
dmin = min(float(np.hypot(v[:, 1], v[:, 2] - TCP).min()) for v in (verts_in(o, root_inv) for o in pads))
check(abs(dmin - R) < 2e-4, f"pad contact distance at s=0: {dmin:.5f} (PIPE_R {R:.5f})")
per_pad = [float(np.hypot(v[:, 1], v[:, 2] - TCP).min()) for v in (verts_in(o, root_inv) for o in pads)]
check(all(abs(d - R) < 2e-4 for d in per_pad), f"all 4 pads touch: {[round(d, 5) for d in per_pad]}")
others = [o for o in meshes if o not in pads]
omin = min(float(np.hypot(v[:, 1], v[:, 2] - TCP).min()) for v in (verts_in(o, root_inv) for o in others))
check(omin > R + 0.003, f"no other part inside the pipe at s=0 (min dist {omin:.4f})")
lo = np.min([verts_in(o, root_inv).min(0) for o in meshes], axis=0)
hi = np.max([verts_in(o, root_inv).max(0) for o in meshes], axis=0)
print(f"[info] closed envelope {lo.round(3)} .. {hi.round(3)}")

# ---- s = 1: a D0.36 cylinder along tool X approaching from +Z (below in world) passes
GR.set_open(g, 1.0)
bpy.context.view_layer.update()
cl = 1.0
for o in movers:
    v = verts_in(o, root_inv)
    c = np.where(v[:, 2] >= TCP, np.abs(v[:, 1]) - 0.18, np.hypot(v[:, 1], v[:, 2] - TCP) - 0.18)
    cl = min(cl, float(c.min()))
check(cl > 0.0, f"open jaws pass a D0.36 cylinder (clearance {cl:.4f})")
lo1 = np.min([verts_in(o, root_inv).min(0) for o in meshes], axis=0)
hi1 = np.max([verts_in(o, root_inv).max(0) for o in meshes], axis=0)
print(f"[info] open envelope {lo1.round(3)} .. {hi1.round(3)}")
lo, hi = np.minimum(lo, lo1), np.maximum(hi, hi1)
check(lo[2] >= -1e-4 and hi[2] <= TCP + R + 0.06 and max(-lo[1], hi[1]) <= 0.32 and max(-lo[0], hi[0]) <= 0.13,
      f"envelope z [{lo[2]:.3f}, {hi[2]:.3f}] <= {TCP + R + 0.06:.3f}, |y| <= {max(-lo[1], hi[1]):.3f}, |x| <= {max(-lo[0], hi[0]):.3f}")

# ---- rod eye stays on the jaw cross pin (linkage consistent) for several s
dev = 0.0
for s in (0.0, 0.05, 0.3, 0.7, 1.0):
    GR.set_open(g, s)
    bpy.context.view_layer.update()
    for sd in ("L", "R"):
        eye = bpy.data.objects[f"grp_rod_{sd}"].matrix_world.translation
        pin = bpy.data.objects[f"grp_rod_pin_{sd}"]
        pc = pin.matrix_world @ mathutils.Vector(np.mean([v.co[:] for v in pin.data.vertices], axis=0))
        dev = max(dev, (eye - pc).length)
check(dev < 1e-4, f"cylinder rod eye on the jaw pin for s in [0,1] (max dev {dev:.2e})")

# ---- flange grasp: pads rest on the hub at s = GRASP['flange'][3]
def flange_clear(s):
    GR.set_open(g, s)
    bpy.context.view_layer.update()
    m = 1.0
    zg = L2.GRASP["flange"][0][2]
    for o in movers:
        v = verts_in(o, root_inv)
        # every jaw part is a prism along X: its section nearest to the hub axis is at x = 0 if it spans it
        xeff = 0.0 if v[:, 0].min() < 0.0 < v[:, 0].max() else float(np.abs(v[:, 0]).min())
        for x, y, z in v:
            Rf = GR._flange_radius(zg + (TCP - z))
            if Rf > 0:
                m = min(m, math.hypot(xeff, y) - Rf)
    return m
sF = L2.GRASP["flange"][3]
c_at, c_before = flange_clear(sF), flange_clear(sF * 0.8)
check(-1e-3 < c_at < 2e-3 and c_before < 0, f"flange hub contact at s={sF}: clearance {c_at * 1000:.2f} mm (s={0.8 * sF:.3f}: {c_before * 1000:.1f} mm)")
print(f"[info] jaw angle deg: s=0 {math.degrees(GR.jaw_angle(0)):.2f}, s={sF} {math.degrees(GR.jaw_angle(sF)):.2f}, s=1 {math.degrees(GR.jaw_angle(1)):.2f}")

# ---- setter keys
GR.set_open(g, 1.0, frame=1)
GR.set_open(g, 0.0, frame=12)
ok = all(o.animation_data and o.animation_data.action for o in g["jaws"] + g["cylinders"] + g["rods"])
check(ok, "set_open keys jaws, cylinders and rods")
bpy.context.scene.frame_set(6)
check(0.0 < abs(g["jaws"][1].rotation_euler[0]) < GR.OPEN_ANGLE, "jaw angle interpolates between keys")
for o in g["jaws"] + g["cylinders"] + g["rods"]:
    o.animation_data_clear()
ntri = H.n_tris(meshes)
check(ntri < 150000, f"triangles {ntri}")
check(GR.obstacles() == [], "obstacles() empty (moves with the robot)")

# ---- stills: (a) closed on a pipe, (b) open at the grasp height, (c) holding a loose flange at s = 0.05
GR.set_open(g, 0.0)
tool_b = G.empty("test_tool0_b", matrix=down_tool(-4.75, 0.0, 1.30), collection=col)
gb = GR.build(tool_b, col, name="grp_b")
GR.set_open(gb, 1.0)
tool_c = G.empty("test_tool0_c", matrix=down_tool(-3.95, 0.0, 1.30), collection=col)
gc = GR.build(tool_c, col, name="grp_c")
GR.set_open(gc, sF)
steel = materials.get("steel_pipe")
for i, x in enumerate((-5.6, -4.75)):
    p = G.revolve(f"test_pipe{i}", [(R - 0.0093, -0.3), (R, -0.3), (R, 0.3), (R - 0.0093, 0.3)], segments=96, axis='X', collection=col)
    p.location = (x, 0.0, 1.30 - TCP)
    p.data.materials.append(steel)
stand = G.box("test_stand", (0.12, 0.30, 1.30 - TCP - R), (-4.75 + 0.22, 0.0, (1.30 - TCP - R) / 2), col, bevel=0.005)
stand.data.materials.append(materials.get("painted", color="#2B2F36", roughness=0.5))
sp = spool.build(name="spool")
pp = P.split(sp)
T_tcp_c = G.np4(gc["tcp"].matrix_world)
g_fr = L2.GRASP["flange"]
T_fl = T_tcp_c @ kin.inv(kin.frame(g_fr[1], g_fr[2], g_fr[0]))
P.set_pose(pp["flange"], T_fl)
P.set_pose(pp["elbow"], kin.tr(-3.2, 1.6, -L2.SEAM_A_Z))
P.set_pose(pp["pipe"], kin.tr(-3.0, 2.4, R - L2.PIPE_AXIS_Z))
bpy.context.view_layer.update()
H.still("t_gripper_trio", cam=(-6.55, -2.35, 1.75), aim=(-4.75, 0.0, 0.98), lens=30)
H.still("t_gripper_closed", cam=(-4.95, -0.62, 1.12), aim=(-5.6, 0.0, 1.0), lens=35)
print("RESULT:", "PASS" if not FAIL else f"FAIL ({len(FAIL)})", flush=True)
sys.exit(1 if FAIL else 0)
