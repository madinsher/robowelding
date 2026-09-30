"""Pre-render sanity check of the stage-2 camera shots (numpy only, no rendering):

  * the camera is never inside an obstacle box, a robot link capsule or a carried part;
  * the line of sight camera -> aim point is not blocked by a robot link / carried part for more than 25 % of the
    shot's frames (short wipes are fine, a shot looking at the back of an arm is not);
  * the shot's subject points (gripper TCP of the handler, torch tip of the tack robot or of the welding robot) stay
    inside the frame (with a 5 % safety margin) in at least half of the samples.

The stage-1 shots (cameras2.S1_SHOTS, rendered from this scene by the en edition at WELD_OFFSET + stage-1 frame) are
checked too, against what stage 2 adds around the stage-1 cell: the camera outside every obstacle / arm, the line of
sight free of the handler robot, the tack robot and the logistics equipment (static obstacle boxes of the stage-2
modules, crossed anywhere up to 0.25 m before the aim point).  The welding robot and the spool are the stage-1 content
of those shots (the same in the stage-1 video), so they are not counted as blocking there.  The S1 cameras cannot
change for ru (the stage-1 video is spliced there): a failure is a finding to report, not a camera to move.

    python3 stage2_logistics/tests/t_cameras2.py [--shot S2_08_load] [--s1 | --s2]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import cameras2  # noqa: E402
import kin as K  # noqa: E402
import layout2 as L2  # noqa: E402
import plan2  # noqa: E402
from cell import robot_build as RB  # noqa: E402
import t_collision2 as C  # noqa: E402

ONLY = sys.argv[sys.argv.index("--shot") + 1] if "--shot" in sys.argv else None
KINDS = {"--s1": ("s1",), "--s2": ("s2",)}.get(next((a for a in sys.argv if a in ("--s1", "--s2")), None), ("s1", "s2"))
# what each shot is about: "handler" (gripper TCP), "tack" / "welder" (torch tip), or None (overview shots)
SUBJECT = {"S2_02_flange": "handler", "S2_03_elbow": "handler", "S2_04_pipe": "handler", "S2_05_tackA": "tack",
           "S2_06_tackB": "tack", "S2_07_pick": "handler", "S2_08_load": "handler", "S2_11_unload": "handler",
           "S2_12_carrier": "handler", "S2_16_store": "handler",
           # stage-1 close-ups of the seams (C2_setup and C7_index are overviews)
           "S1_C3a_laser": "welder", "S1_C3b_closeA": "welder", "S1_C4_midA": "welder", "S1_C6a_closeB1": "welder",
           "S1_C6b_closeB1": "welder", "S1_C8a_closeB2": "welder", "S1_C8b_closeB2": "welder"}
S1_NAMES = {s[0] for s in cameras2.S1_SHOTS}
LOS_STOP = 0.25        # m: the static line-of-sight test ends this far before the aim point (on the part)


def ease(t):
    # Blender BEZIER + EASE_IN_OUT between two keys ~ smoothstep
    return K.smoothstep(t)


def cam_at(shot, f, f0, f1):
    s = ease((f - f0) / float(max(1, f1 - f0)))
    c = np.array(shot[3]) * (1 - s) + np.array(shot[4]) * s
    a = np.array(shot[5]) * (1 - s) + np.array(shot[6]) * s
    return c, a


def project(p, cam, aim, lens, sensor_w=36.0, aspect=16 / 9):
    """Normalised image coords (-1..1 on each axis) of p, or None if behind the camera."""
    fwd = aim - cam
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    d = p - cam
    z = d @ fwd
    if z <= 0.05:
        return None
    half_w = sensor_w / 2 / lens
    half_h = half_w / aspect
    return np.array([(d @ right) / z / half_w, (d @ up) / z / half_h])


def seg_point_dist(a, b, p):
    ab = b - a
    t = min(max(float((p - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0), 1.0)
    return float(np.linalg.norm(a + ab * t - p))


def main():
    P = plan2.solve()
    ev = cameras2.events_of(P)
    H = K.scaled_arm(L2.HANDLER_SCALE)
    TK = K.scaled_arm(L2.TACK_SCALE)
    W = RB.load_arm()
    TORCH = RB.tool_transform()
    mod_obs = C.module_obstacles()
    obs = mod_obs + C.stage1_obstacles()
    boxes = C.Boxes(mod_obs)
    bad = []
    for shot in cameras2.ALL_SHOTS:
        name, lens = shot[0], shot[7]
        s1 = name in S1_NAMES
        if (ONLY and name != ONLY) or ("s1" if s1 else "s2") not in KINDS:
            continue
        f0, f1 = cameras2.shot_range(shot, ev)
        blocked, n, out_of_frame, inside, crossed = 0, 0, 0, [], {}
        for f in range(f0, f1 + 1, 2):
            i = f - 1
            cam, aim = cam_at(shot, f, f0, f1)
            caps, fr = C.arm_capsules(H, P["handler_q"][i], plan2.handler_base(P["handler_x"][i]), L2.HANDLER_SCALE,
                                      plan2.GRIP_T, 0.13)
            tcaps, tfr = C.arm_capsules(TK, P["tack_q"][i], plan2.tack_base(), L2.TACK_SCALE, TORCH, 0.02, torch=True)
            wcaps, wfr = C.arm_capsules(W, P["welder_q"][i], RB.base_matrix(P["welder_track"][i]), 1.0, TORCH, 0.03,
                                        torch=True)
            pcaps = []
            for part, T in P["parts"].items():
                pcaps += [("part_" + part, a, b, r) for a, b, r in C.part_segments(part, T[i])]
            allcaps = caps + tcaps + wcaps + pcaps
            # camera inside something
            for nm, a, b, r in allcaps:
                if seg_point_dist(a, b, cam) < r + 0.05:
                    inside.append((f, nm))
            for o in obs:
                if C.seg_box_dist(cam, cam, o) < 0.03:
                    inside.append((f, o["name"]))
            # line of sight to the aim point
            n += 1
            subj = SUBJECT.get(name)
            tagged = [("handler", c) for c in caps] + [("tack", c) for c in tcaps]
            if not s1:          # stage-1 shots: the welding robot is the stage-1 content, not an obstruction
                tagged += [("welder", c) for c in wcaps]
            hit = False
            for owner, (nm, a, b, r) in tagged:
                # the subject's own tool (gripper holding the part at the aim point / torch at the seam) is the subject
                if owner == subj and nm in ("tool", "torch_neck"):
                    continue
                if C.seg_seg_dist(cam, aim, a, b) < r * 0.8:
                    hit = True
                    break
            if s1:              # logistics equipment (static stage-2 boxes) between the camera and the part
                d = aim - cam
                end = aim - d / np.linalg.norm(d) * LOS_STOP
                for k in np.nonzero(boxes.seg_dist(cam, end, n=64) < 0.0)[0]:
                    crossed[mod_obs[k]["name"]] = crossed.get(mod_obs[k]["name"], 0) + 1
                    hit = True
            blocked += hit
            # subjects in frame
            subjects = {"handler": [(fr["tool0"] @ plan2.GRIP_T)[:3, 3]], "tack": [(tfr["tool0"] @ TORCH)[:3, 3]],
                        "welder": [(wfr["tool0"] @ TORCH)[:3, 3]]}.get(subj, [])
            for sp in subjects:
                uv = project(sp, cam, aim, lens)
                if uv is None or np.max(np.abs(uv)) > 0.95:
                    out_of_frame += 1
        frac_b = blocked / max(n, 1)
        status = []
        if inside:
            status.append(f"camera inside {sorted(set(nm for _, nm in inside))[:4]} at frames {[f for f, _ in inside][:5]}")
        if frac_b > 0.25:
            status.append(f"line of sight blocked {frac_b:.0%}")
        if crossed:
            status.append(f"logistics equipment in the line of sight: {dict(sorted(crossed.items())[:4])}")
        if SUBJECT.get(name) and out_of_frame > 0.5 * n:
            status.append(f"subject out of frame in {out_of_frame}/{n} samples")
        tag = " (stage 1)" if s1 else ""
        print(f"{name:16s} {f0:5d}-{f1:5d} blocked {frac_b:4.0%} out-of-frame {out_of_frame:3d}/{n:3d}  "
              f"{'; '.join(status) or 'ok'}{tag}")
        if inside or frac_b > 0.25 or (SUBJECT.get(name) and out_of_frame > 0.5 * n):
            bad.append(name)
    print("RESULT:", "OK" if not bad else f"FAIL {bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
