"""Numeric clearance check between the robot (link segments) and the spool (centreline polyline) for every frame.
Uses the cached trajectory (out/anim_cache_*.npz); run after tests/t_solve.py."""
import sys, os, math, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cell import layout as L, animation as A, robot_build as RB
arm = RB.load_arm(); TOOL = RB.tool_transform()
cache = sorted(glob.glob(os.path.join("out", f"anim_cache_{A._cache_key()}.npz")))
assert cache, "no cache for current animation.py — run tests/t_solve.py first"
z = np.load(cache[0]); Q = z["Q"]
nF = A.N_FRAMES
# rebuild the positioner/track schedules exactly as animation.build does
tilt = np.zeros(nF); rot = np.full(nF, A.ROT_A0); track = np.full(nF, L.ROBOT_HOME_Y)
def ramp(arr, span, v0, v1, ease=True):
    a, b = span
    for f in range(a, b + 1):
        s = A.frac(f, span); s = A.smoothstep(s) if ease else s
        arr[f - 1] = v0 + (v1 - v0) * s
    arr[b:] = v1
ramp(tilt, A.T["tilt_up"], 0.0, A.TILT_B1); ramp(rot, A.T["weld_A"], A.ROT_A0, A.ROT_A1, ease=False)
ramp(tilt, A.T["index_180"], A.TILT_B1, A.TILT_B2); ramp(tilt, A.T["tilt_down"], A.TILT_B2, 0.0)
ramp(track, A.T["track_to_A"], L.ROBOT_HOME_Y, A.TRACK_A); ramp(track, A.T["track_to_B1"], A.TRACK_A, A.TRACK_B1)
ramp(track, A.T["track_to_B2"], A.TRACK_B1, A.TRACK_B2); ramp(track, A.T["track_home"], A.TRACK_B2, L.ROBOT_HOME_Y)

def spool_polyline(Ts):
    # (x,y,z, radius): fixture ring + flange disc, then the weld-neck hub tapering to the pipe radius at seam A
    pts = [(0, 0, -0.03, 0.2025), (0, 0, L.FLANGE_THK, 0.2025), (0, 0, L.FLANGE_THK + 0.001, 0.165),
           (0, 0, L.SEAM_A_Z - 0.035, L.SEAM_RADIUS + 0.004), (0, 0, L.SEAM_A_Z, L.SEAM_RADIUS)]
    for k in range(1, 13):
        th = math.pi / 2 * k / 12
        pts.append((L.ELBOW_R * (1 - math.cos(th)), 0, L.SEAM_A_Z + L.ELBOW_R * math.sin(th), L.SEAM_RADIUS))
    cx, cy, cz = L.SEAM_B_CENTER
    pts.append((cx + L.PIPE_LEN, cy, cz, L.SEAM_RADIUS))
    out = []
    for x, y, zz, r in pts:
        out.append(((Ts @ np.array([x, y, zz, 1.0]))[:3], r))
    return out

def seg_dist(p1, p2, q1, q2, r1=0.0, r2=0.0):
    """min clearance between robot segment p1p2 (sampled) and a flat-ended cylinder/cone segment q1q2 with radii r1->r2."""
    best = 1e9
    d = q2 - q1; L2 = float(d @ d); Ln = math.sqrt(L2) if L2 > 1e-12 else 1e-6
    for s in np.linspace(0, 1, 9):
        a = p1 + (p2 - p1) * s
        t = 0.0 if L2 < 1e-12 else float((a - q1) @ d) / L2
        axis_pt = q1 + d * min(1.0, max(0.0, t))
        radial = float(np.linalg.norm(a - axis_pt)) if 0.0 <= t <= 1.0 else float(np.linalg.norm(np.cross(a - q1, d) / Ln))
        if 0.0 <= t <= 1.0:
            c = radial - (r1 + (r2 - r1) * t)
        else:
            over = (-t if t < 0 else t - 1.0) * Ln
            r_end = r1 if t < 0 else r2
            c = math.sqrt(over * over + max(0.0, radial - r_end) ** 2)
        best = min(best, c)
    return best

worst = {}
weld_frames = set()
for k in ("arc_A", "arc_1", "arc_2", "arc_3", "arc_4", "laser", "to_weld_A"):
    a, b = A.T[k]; weld_frames.update(range(a - 6, b + 7))
for i in range(nF):
    f = i + 1
    base = RB.base_matrix(track[i]); fr = arm.fk_frames(Q[i], base)
    P = {k: v[:3, 3] for k, v in fr.items()}
    tool0 = fr["tool0"]; neck = (tool0 @ np.array([0, 0, L.TORCH_BRACKET_LEN + L.TORCH_NECK_LEN, 1.0]))[:3]; tcp = (tool0 @ TOOL)[:3, 3]
    segs = [("upper_arm", P["link_2"], P["link_3"], 0.15), ("forearm", P["link_4"], P["link_5"], 0.11), ("wrist", P["link_5"], P["link_6"], 0.09),
            ("torch_neck", P["link_6"], neck, 0.05), ("torch_head", neck, tcp, 0.02)]
    Ts = A.spool_world(tilt[i], rot[i]); poly = spool_polyline(Ts)
    for name, s0, s1, rl in segs:
        m = 1e9
        for (a, ra), (b, rb) in zip(poly[:-1], poly[1:]):
            m = min(m, seg_dist(s0, s1, a, b, ra, rb) - rl)
        allow = -0.03 if (name == "torch_head" and f in weld_frames) else 0.02
        if m < allow:
            worst.setdefault(name, []).append((f, round(m, 3)))
print("frames with clearance below threshold (name: [(frame, clearance m)...]):")
for name, lst in worst.items():
    print(f"  {name}: n={len(lst)} worst={min(lst, key=lambda x: x[1])} frames={[x[0] for x in lst][:12]}{'...' if len(lst)>12 else ''}")
if not worst:
    print("  none — no robot/part interference")
