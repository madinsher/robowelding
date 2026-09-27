"""Continuity checks of the stage-2 plan (numpy only):
  * no IK failures;
  * joint step per frame <= 6 deg for the handler, the tack robot and the welding robot; carriage <= 2 m/s;
  * carried / placed parts never jump: per-frame translation <= 0.10 m and rotation <= 6 deg, in particular at the
    grasp / release frames (hand-over transforms frozen at the event frame);
  * the part is where the gripper is while attached (TCP-to-grasp error < 2 mm at every grip event).

    python3 stage2_logistics/tests/t_continuity2.py
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np  # noqa: E402
import kin as K  # noqa: E402
import plan2  # noqa: E402

LIM_JOINT = math.radians(6.0)
LIM_TRACK = 2.0 / 24.0
LIM_PART_T = 0.10
LIM_PART_R = math.radians(6.0)


def main():
    P = plan2.solve()
    n = P["n_frames"]
    bad = []
    if P["ik_fail"]:
        bad.append(f"IK failures: {P['ik_fail'][:8]}")
    for name, key in (("handler", "handler_q"), ("tack", "tack_q"), ("welder", "welder_q")):
        d = np.abs(np.diff(P[key], axis=0))
        m = d.max(axis=0)
        worst = int(np.argmax(d.max(axis=1))) + 2
        print(f"{name:8s} max joint step/frame (deg): {np.round(np.degrees(m), 2)}  worst at frame {worst}")
        if (m > LIM_JOINT).any():
            bad.append(f"{name} joint step {np.round(np.degrees(m), 2)} deg/frame at frame {worst}")
    dx = np.abs(np.diff(P["handler_x"])).max()
    print(f"handler carriage max step {dx * 24:.2f} m/s")
    if dx > LIM_TRACK:
        bad.append(f"carriage speed {dx * 24:.2f} m/s")
    for part, T in P["parts"].items():
        dt = np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1)
        dr = np.array([K.rot_angle(T[i], T[i + 1]) for i in range(n - 1)])
        it, ir = int(np.argmax(dt)), int(np.argmax(dr))
        print(f"part {part:8s} max step {dt[it]:.3f} m (frame {it + 2}), {math.degrees(dr[ir]):.2f} deg (frame {ir + 2})")
        if dt[it] > LIM_PART_T or dr[ir] > LIM_PART_R:
            bad.append(f"part {part} jumps: {dt[it]:.3f} m at {it + 2}, {math.degrees(dr[ir]):.2f} deg at {ir + 2}")
    # gripper vs part at the grip events
    for ev, f in sorted(P["events"].items(), key=lambda kv: kv[1]):
        if not ev.endswith("_grip"):
            continue
        tcp = plan2.GRIP_T
        x, q = P["handler_x"][f - 1], P["handler_q"][f - 1]
        H = K.scaled_arm(plan2.L2.HANDLER_SCALE)
        Ttcp = H.fk(q, plan2.handler_base(x), tcp)
        errs = []
        for part, T in P["parts"].items():
            for g in ("flange", "elbow", "pipe", "spool"):
                G = T[f - 1] @ plan2.grasp_frame(g)
                errs.append((float(np.linalg.norm(G[:3, 3] - Ttcp[:3, 3])), part, g))
        e = min(errs)
        print(f"grip {ev:18s} frame {f:5d}: nearest grasp {e[1]}/{e[2]} at {e[0] * 1000:.1f} mm")
        if e[0] > 0.002:
            bad.append(f"grip {ev} misses its grasp by {e[0] * 1000:.1f} mm")
    print("RESULT:", "OK" if not bad else "FAIL")
    for b in bad:
        print("  -", b)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
