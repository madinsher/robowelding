"""Integration test of the stage-2 scene (bpy, no rendering): build the whole scene (without the hall environment for
speed) and compare what Blender evaluates with the numpy plan on sampled frames:

  * handler gripper TCP  == kin FK of plan handler_q / handler_x          (< 1 mm, < 0.1 deg)
  * tack robot torch TCP == kin FK of plan tack_q                        (< 1 mm)
  * welding robot tool0  == stage-1 FK of plan welder_q / welder_track   (< 1 mm)
  * part roots           == plan part poses                              (< 1 mm, < 0.1 deg)
  * positioner mount     == animation.spool_world(tilt, rot)             (< 1 mm)
  * camera markers: one per shot of the edit, bound to the right camera at the shot's first frame
  * weld beads: invisible before the embedded weld, complete after it (w0 progress of bead A)

    python3 stage2_logistics/tests/t_build2.py [--env]
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np  # noqa: E402
import bpy  # noqa: E402
import build2  # noqa: E402
import kin as K  # noqa: E402
import layout2 as L2  # noqa: E402
import plan2  # noqa: E402
from cell import robot_build as RB, animation as A, geom as G  # noqa: E402


def mw(ob):
    return G.np4(ob.matrix_world)


def main():
    S = build2.build_scene(with_env="--env" in sys.argv, with_vfx=True)
    sc = S["scene"]
    P = S["plan"]
    M = S["modules"]
    ev = P["events"]
    H = K.scaled_arm(L2.HANDLER_SCALE)
    TK = K.scaled_arm(L2.TACK_SCALE)
    W = RB.load_arm()
    TORCH = RB.tool_transform()
    frames = sorted({1, ev["kit_flange_grip"], ev["stn_elbow_contact"], ev["first_tack"] + 3, ev["stn_spool_grip"],
                     ev["entry_done"], ev["load_contact"], ev["studs_up"], ev["weld_first"] + 300, ev["unload_grip"] + 10,
                     ev["conv_load_contact"], ev["mark_start"] + 20, ev["store_contact"], P["n_frames"]})
    bad = []

    def check(name, f, Tb, Tp, tol_t=1e-3, tol_r=math.radians(0.1)):
        dt = float(np.linalg.norm(Tb[:3, 3] - Tp[:3, 3]))
        dr = K.rot_angle(Tb, Tp)
        if dt > tol_t or dr > tol_r:
            bad.append(f"{name} @ {f}: {dt * 1000:.2f} mm, {math.degrees(dr):.3f} deg")
        return dt

    hnd, tck, rob, pos = M["handler"], M["tack"], M["robot"], M["pos"]
    grip_tcp = hnd["gripper"]["tcp"]
    for f in frames:
        sc.frame_set(f)
        i = f - 1
        check("handler tcp", f, mw(grip_tcp), H.fk(P["handler_q"][i], plan2.handler_base(P["handler_x"][i]), plan2.GRIP_T))
        check("tack tcp", f, mw(tck["tcp"]), TK.fk(P["tack_q"][i], plan2.tack_base(), TORCH), tol_r=math.radians(0.2))
        check("welder tool0", f, mw(rob["tool0"]), W.fk(P["welder_q"][i], RB.base_matrix(P["welder_track"][i])))
        check("positioner mount", f, mw(pos["mount"]), A.spool_world(P["tilt"][i], P["rot"][i]))
        for part in ("flange", "elbow", "pipe"):
            check(f"part {part}", f, mw(M["parts"][part]), P["parts"][part][i])
    # markers / cameras
    import edl as EDL
    E = EDL.build_edl(P)
    shots = [s for s in E["segments"] if s["src"] == "s2"]
    marks = {m.frame: m.camera.name for m in sc.timeline_markers if m.camera}
    for s in shots:
        if marks.get(s["f0"]) != s["shot"]:
            bad.append(f"marker at {s['f0']}: {marks.get(s['f0'])} != {s['shot']}")
        sc.frame_set(s["f0"])
        cam = sc.camera
        # the active camera at the shot's frames is taken from the markers at render time
    # beads: before / after the embedded weld
    bead = bpy.data.objects.get("spool_bead_A")
    if bead is not None:
        sc.frame_set(ev["load_contact"])
        p0 = float(bead["w0_prog"])
        sc.frame_set(ev["unload_grip"])
        p1 = float(bead["w0_prog"])
        if not (p0 <= 0.01 and p1 >= 0.99):
            bad.append(f"bead A progress before/after weld: {p0:.3f} / {p1:.3f}")
    print(f"checked {len(frames)} frames x 7 poses, {len(shots)} shot markers")
    print("RESULT:", "OK" if not bad else "FAIL")
    for b in bad[:30]:
        print("  -", b)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
