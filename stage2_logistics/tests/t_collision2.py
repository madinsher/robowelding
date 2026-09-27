"""Numeric clearance check of the stage-2 choreography (numpy only, every frame):

  1. handler links / gripper / carried parts vs static obstacles (every module's obstacles(), the cell fence with
     its loading opening, the positioner towers and fume hood); gripper and carried parts are exempt inside their own
     pick / place contact windows (plan intervals "contact"), the arm links never are;
  2. tack robot (links + torch) vs the handler, the spool on the station (torch tip may touch at the tack frames);
  3. zone rule (layout2.SAFE_X_HANDLER): whenever any handler point or carried part is beyond the cell fence line, the
     welding robot is at home, the positioner is at rest and the muting lamp is on;
  4. handler vs welding robot: never closer than 0.3 m.
Also reports, per check, the worst clearance.

    python3 stage2_logistics/tests/t_collision2.py [--step 1]
"""
import importlib
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np  # noqa: E402
import kin as K  # noqa: E402
import layout2 as L2  # noqa: E402
import plan2  # noqa: E402
from cell import layout as L, robot_build as RB  # noqa: E402

STEP = int(sys.argv[sys.argv.index("--step") + 1]) if "--step" in sys.argv else 1
MARGIN = 0.005         # obstacle boxes are conservative (they enclose the real geometry with ~1 cm padding)


# ---------------------------------------------------------------------------- geometry
def seg_box_dist(p0, p1, box, n=9):
    """Min distance from a segment (sampled) to an oriented box (center, size, yaw about Z)."""
    c = np.asarray(box["center"], float)
    h = np.asarray(box["size"], float) / 2
    yaw = box.get("yaw", 0.0) or 0.0
    cy, sy = math.cos(-yaw), math.sin(-yaw)
    best = 1e9
    for s in np.linspace(0.0, 1.0, n):
        p = p0 + (p1 - p0) * s - c
        q = np.array([cy * p[0] - sy * p[1], sy * p[0] + cy * p[1], p[2]])
        d = np.maximum(np.abs(q) - h, 0.0)
        inside = np.all(np.abs(q) <= h)
        dist = -float(np.min(h - np.abs(q))) if inside else float(np.linalg.norm(d))
        best = min(best, dist)
    return best


class Boxes:
    """Obstacle boxes as arrays for vectorised segment-to-box distances."""

    def __init__(self, obs):
        self.obs = obs
        self.c = np.array([o["center"] for o in obs], float)
        self.h = np.array([o["size"] for o in obs], float) / 2
        yaw = np.array([o.get("yaw", 0.0) or 0.0 for o in obs])
        self.cos, self.sin = np.cos(-yaw), np.sin(-yaw)

    def seg_dist(self, p0, p1, n=9):
        """(M,) min distance from the sampled segment p0-p1 to every box (negative inside)."""
        s = np.linspace(0.0, 1.0, n)[:, None]
        pts = p0[None, :] + (p1 - p0)[None, :] * s                     # (n,3)
        d = pts[:, None, :] - self.c[None, :, :]                        # (n,M,3)
        qx = self.cos[None, :] * d[..., 0] - self.sin[None, :] * d[..., 1]
        qy = self.sin[None, :] * d[..., 0] + self.cos[None, :] * d[..., 1]
        q = np.abs(np.stack([qx, qy, d[..., 2]], axis=-1))              # (n,M,3)
        out = np.maximum(q - self.h[None], 0.0)
        dist = np.linalg.norm(out, axis=-1)
        inside = np.all(q <= self.h[None], axis=-1)
        pen = -np.min(self.h[None] - q, axis=-1)
        dist = np.where(inside, pen, dist)
        return dist.min(axis=0)


def seg_seg_dist(a0, a1, b0, b1):
    d1, d2, r = a1 - a0, b1 - b0, a0 - b0
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    if a < 1e-12 and e < 1e-12:
        return float(np.linalg.norm(r))
    if a < 1e-12:
        s, t = 0.0, min(max(f / e, 0.0), 1.0)
    else:
        c = d1 @ r
        if e < 1e-12:
            t, s = 0.0, min(max(-c / a, 0.0), 1.0)
        else:
            b = d1 @ d2
            den = a * e - b * b
            s = min(max((b * f - c * e) / den, 0.0), 1.0) if den > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0:
                t, s = 0.0, min(max(-c / a, 0.0), 1.0)
            elif t > 1:
                t, s = 1.0, min(max((b - c) / a, 0.0), 1.0)
    return float(np.linalg.norm((a0 + d1 * s) - (b0 + d2 * t)))


TORCH_NECK = L.TORCH_BRACKET_LEN + L.TORCH_NECK_LEN


def arm_capsules(arm, q, base, scale, tool=None, tool_r=0.0, torch=False):
    fr = arm.fk_frames(q, base)
    P = {k: v[:3, 3] for k, v in fr.items()}
    caps = [("upper_arm", P["link_2"], P["link_3"], 0.15 * scale), ("forearm", P["link_4"], P["link_5"], 0.11 * scale),
            ("wrist", P["link_5"], P["tool0"], 0.10 * scale)]
    if tool is not None:
        tcp = (fr["tool0"] @ tool)[:3, 3]
        if torch:     # bent MIG torch: straight neck along tool0 z, then the 45-degree head to the TCP
            neck = (fr["tool0"] @ np.array([0.0, 0.0, TORCH_NECK, 1.0]))[:3]
            caps.append(("torch_neck", P["tool0"], neck, 0.03))
            caps.append(("tool", neck, tcp, tool_r))
        else:
            caps.append(("tool", P["tool0"], tcp, tool_r))
    return caps, fr


# part-local centreline polylines with radii (spool frame)
def part_polyline(part):
    R = L2.PIPE_R
    if part in ("flange", "flange2"):      # disc and weld-neck hub are added as flat capsules (part_segments)
        return []
    if part == "elbow":
        pts = []
        for k in range(0, 13):
            th = math.pi / 2 * k / 12
            pts.append(((L.ELBOW_R * (1 - math.cos(th)), 0, L2.SEAM_A_Z + L.ELBOW_R * math.sin(th)), R))
        return pts
    if part == "pipe":
        return [((L2.SEAM_B_X, 0, L2.PIPE_AXIS_Z), R), ((L2.PIPE_END_X, 0, L2.PIPE_AXIS_Z), R)]
    raise KeyError(part)


def transform_poly(poly, T):
    return [((T @ np.array([*p, 1.0]))[:3], r) for p, r in poly]


def part_segments(part, T):
    """World capsules (a, b, r) of a part: centreline polyline + for flanges the disc as four flat diameters."""
    poly = transform_poly(part_polyline(part), T)
    segs = [(a, b, max(ra, rb)) for (a, ra), (b, rb) in zip(poly[:-1], poly[1:])]
    if part in ("flange", "flange2"):
        # (radius to the capsule end centres, height, capsule radius): disc, then the hub (r 0.137, z 0.028..0.1)
        hub_r = (L2.SEAM_A_Z - L.FLANGE_THK) / 2
        for rr, z, rad in ((L.FLANGE_OD / 2 - L.FLANGE_THK / 2, L.FLANGE_THK / 2, L.FLANGE_THK / 2),
                           (L2.PIPE_R - hub_r, L.FLANGE_THK + hub_r, hub_r)):
            for k in range(4):
                c, s_ = math.cos(k * math.pi / 4), math.sin(k * math.pi / 4)
                a = (T @ np.array([rr * c, rr * s_, z, 1.0]))[:3]
                b = (T @ np.array([-rr * c, -rr * s_, z, 1.0]))[:3]
                segs.append((a, b, rad))
    return segs


# ---------------------------------------------------------------------------- obstacles
def module_obstacles():
    obs = []
    for mod in ("handler_robot", "cassette", "assembly_station", "conveyor", "marking_qc", "storage", "environment2",
                "tack_robot"):
        try:
            m = importlib.import_module(mod)
            got = m.obstacles()
            for o in got:
                o = dict(o)
                o["module"] = mod
                obs.append(o)
        except Exception as e:  # module not built yet
            print(f"[obstacles] {mod}: skipped ({type(e).__name__}: {e})")
    return obs


def stage1_obstacles():
    x0 = L2.CELL_FENCE_X0
    oy = L2.LOAD_OPENING_Y
    y0, y1 = L.FENCE_Y
    obs = [
        dict(name="cell_fence_w1", center=(x0, (y0 - oy) / 2, 1.05), size=(0.08, -oy - y0, 2.1)),
        dict(name="cell_fence_w2", center=(x0, (y1 + oy) / 2, 1.05), size=(0.08, y1 - oy, 2.1)),
        dict(name="light_curtain_post_s", center=(x0, -oy, 0.97), size=(0.10, 0.10, 1.94)),
        dict(name="light_curtain_post_n", center=(x0, oy, 0.97), size=(0.10, 0.10, 1.94)),
        dict(name="fume_hood", center=(0.3, 0.0, 3.05 + 0.175), size=(1.6, 1.6, 0.35)),
        dict(name="pos_towers", center=(0.0, 0.0, (0.33 + 1.35) / 2), size=(1.84, 0.46, 1.35 - 0.33)),
        dict(name="pos_heads", center=(0.0, 0.0, 1.35), size=(1.84, 0.44, 0.44)),
        dict(name="pos_bed", center=(0.0, 0.0, 0.2), size=(1.84, 1.25, 0.4)),
    ]
    for o in obs:
        o.setdefault("yaw", 0.0)
        o["module"] = "stage1"
    return obs


def in_windows(f, windows):
    return any(a <= f <= b for a, b in windows)


# ---------------------------------------------------------------------------- main
def main():
    P = plan2.solve()
    n = P["n_frames"]
    H = K.scaled_arm(L2.HANDLER_SCALE)
    TK = K.scaled_arm(L2.TACK_SCALE)
    W = RB.load_arm()
    GRIP = plan2.GRIP_T
    TORCH = RB.tool_transform()
    obs = module_obstacles() + stage1_obstacles()
    BX = Boxes(obs)
    print(f"{len(obs)} obstacle boxes")
    contact = [tuple(w) for w in P["intervals"].get("contact", [])]
    tack_arcs = [tuple(w) for w in P["intervals"]["tack_arc"]]
    carried = {p: [] for p in P["parts"]}
    # which parts are attached to the gripper at each frame: part moves with the TCP
    tcp = np.array([H.fk(P["handler_q"][i], plan2.handler_base(P["handler_x"][i]), GRIP) for i in range(n)])
    for part, T in P["parts"].items():
        rel = np.einsum("nij,njk->nik", np.linalg.inv(tcp), T)
        same = np.linalg.norm(np.diff(rel[:, :3, 3], axis=0), axis=1) < 1e-6
        moving = np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1) > 1e-6
        att = np.zeros(n, dtype=bool)
        att[1:] = same & moving
        carried[part] = att
    worst = {}
    fails = []
    home_q = np.array(L.ROBOT_Q_HOME)

    def note(key, f, d, limit):
        if key not in worst or d < worst[key][1]:
            worst[key] = (f, d)
        if d < limit:
            fails.append((key, f, round(d, 3)))

    for i in range(0, n, STEP):
        f = i + 1
        x, q = P["handler_x"][i], P["handler_q"][i]
        caps, fr = arm_capsules(H, q, plan2.handler_base(x), L2.HANDLER_SCALE, GRIP, 0.13)
        in_contact = in_windows(f, contact)
        # 1. handler vs static
        for name, a, b, r in caps:
            if name == "tool" and in_contact:
                continue
            dd = BX.seg_dist(a, b) - r
            for k in np.nonzero(dd < 0.25)[0]:
                o = obs[k]
                note(f"handler.{name} vs {o['module']}:{o['name']}", f, float(dd[k]), MARGIN)
        for part, T in P["parts"].items():
            if not carried[part][i] or in_contact:
                continue
            for a, b, r in part_segments(part, T[i]):
                dd = BX.seg_dist(a, b) - r
                for k in np.nonzero(dd < 0.25)[0]:
                    o = obs[k]
                    # parts vs padded support boxes (~12 mm padding): no contact is the criterion
                    note(f"part.{part} vs {o['module']}:{o['name']}", f, float(dd[k]), 0.0)
        # 3. zone rule
        pts = [a for _, a, b, _ in caps] + [b for _, a, b, _ in caps]
        for part, T in P["parts"].items():
            if carried[part][i]:
                pts += [p for a, b, _ in part_segments(part, T[i]) for p in (a, b)]
        xmax = max(p[0] for p in pts)
        if xmax > L2.SAFE_X_HANDLER:
            welder_home = np.allclose(P["welder_q"][i], home_q, atol=1e-3) and abs(P["welder_track"][i] - L.ROBOT_HOME_Y) < 1e-3
            j = min(i + 1, n - 1)
            pos_rest = abs(P["tilt"][j] - P["tilt"][i - 1]) < 1e-6 and abs(P["rot"][j] - P["rot"][i - 1]) < 1e-6
            muting = P["muting"][i] > 0.5
            if not (welder_home and pos_rest and muting):
                fails.append(("zone rule", f, f"xmax={xmax:.2f} welder_home={welder_home} pos_rest={pos_rest} muting={muting}"))
        # 4. handler vs welder
        wc, _ = arm_capsules(W, P["welder_q"][i], RB.base_matrix(P["welder_track"][i]), 1.0, TORCH, 0.03, torch=True)
        dmin = min(seg_seg_dist(a, b, c, d) - ra - rc for _, a, b, ra in caps for _, c, d, rc in wc)
        note("handler vs welder", f, dmin, 0.3)
        # 2. tack robot
        tq = P["tack_q"][i]
        tb = plan2.tack_base()
        tcaps, _ = arm_capsules(TK, tq, tb, L2.TACK_SCALE, TORCH, L.TORCH_NOZZLE_R + 0.004, torch=True)
        dmin = min(seg_seg_dist(a, b, c, d) - ra - rc for _, a, b, ra in caps for _, c, d, rc in tcaps)
        note("handler vs tack robot", f, dmin, 0.05)
        near_arc = any(a - 8 <= f <= b + 8 for a, b in tack_arcs)
        for part in ("flange", "elbow", "pipe"):
            T = P["parts"][part][i]
            if np.linalg.norm(T[:3, 3] - np.array(L2.STATION_ORIGIN)) > 0.05:
                continue
            psegs = part_segments(part, T)
            for name, a, b, r in tcaps:
                # at a tack the torch tip sits on the seam: its capsule end sphere and the parts' end caps overlap by
                # construction; elsewhere the torch must stay clear of the parts
                allow = -0.045 if (name == "tool" and near_arc) else 0.0
                d = min(seg_seg_dist(a, b, pa, pb) - r - rp for pa, pb, rp in psegs)
                note(f"tack.{name} vs {part}", f, d, allow)
    print("worst clearances (m):")
    for k, (f, d) in sorted(worst.items(), key=lambda kv: kv[1][1])[:25]:
        print(f"  {d:7.3f}  frame {f:5d}  {k}")
    print("RESULT:", "OK" if not fails else f"FAIL ({len(fails)} frame-violations)")
    seen = set()
    for key, f, d in fails:
        if key in seen:
            continue
        seen.add(key)
        frames = [ff for kk, ff, _ in fails if kk == key]
        print(f"  - {key}: {len(frames)} frames, first {frames[:6]} ({d})")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
