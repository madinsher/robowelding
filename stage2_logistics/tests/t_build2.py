"""Integration test of the stage-2 scene (bpy, no rendering): build the whole scene of an edition (without the hall
environment for speed) and compare what Blender evaluates with the numpy plan on sampled frames:

  * handler gripper TCP  == kin FK of plan handler_q / handler_x          (< 1 mm, < 0.1 deg)
  * tack robot torch TCP == kin FK of plan tack_q                        (< 1 mm)
  * welding robot tool0  == stage-1 FK of plan welder_q / welder_track   (< 1 mm)
  * part roots           == plan part poses                              (< 1 mm, < 0.1 deg)
  * positioner mount     == animation.spool_world(tilt, rot)             (< 1 mm)
  * camera markers: every stage-2 segment of the edition's edit (en: the stage-1 shots S1_* too) is shown by its own
    camera from its first to its last frame
  * en: the S1_* cameras are the stage-1 shots - at the first, middle and last frame of each the camera location is
    the stage-1 formula (cam_from -> cam_to, Blender BEZIER ~ smoothstep, at stage-1 frame = scene frame -
    WELD_OFFSET) within 1 cm, it looks at the stage-1 aim point (< 0.2 deg), same lens and depth of field
  * stage-1 effects at the embedded frames (both editions): the seam-tracking laser line is visible and lies on the
    loaded spool during the scan (hidden outside it); the stage-1 arc light repeats the stage-1 flicker frame by frame;
    the spark systems emit over the embedded weld intervals and their caches cover them
  * weld beads: invisible before the embedded weld, complete after it (w0 progress of bead A)
  * the language / brand of the scene texts are the edition's (i18n2, branding2)
The particle caches are baked first, as build2.render_frames does before rendering: an unbaked spark system that
resimulates after a frame jump re-evaluates its emitter's parent chain (the robot carrying the torch) at the particles'
birth times and leaves it there (Blender's evaluate_emitter_anim) - the poses would then be off by up to 0.24 m in a
few percent of the jumps into an arc interval, which no render sees.

    python3 stage2_logistics/tests/t_build2.py [--edition ru|en] [--env]
"""
import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np  # noqa: E402
import bpy  # noqa: E402
import build2  # noqa: E402
import editions  # noqa: E402
import kin as K  # noqa: E402
import layout2 as L2  # noqa: E402
import plan2  # noqa: E402
from cell import robot_build as RB, animation as A, geom as G, vfx  # noqa: E402

TOL_CAM = 0.01                 # m, S1 camera location vs the stage-1 formula
TOL_AIM = math.radians(0.2)    # S1 camera view direction vs the stage-1 aim point
TOL_LASER = 0.005              # m, laser line vertices from the spool surface (the strip floats 1.5 mm above it)


def mw(ob):
    return G.np4(ob.matrix_world)


def _root(ob):
    while ob.parent is not None:
        ob = ob.parent
    return ob


def check_s1_cameras(sc, E, off, bad):
    """en: the S1_* cameras against the stage-1 SHOTS literal (demo_video/cell/animation.py), independent of
    cameras2's own event arithmetic."""
    import cameras2
    raw = {"S1_" + s[0]: s for s in cameras2._stage1_shots()}
    n = 0
    for seg in (s for s in E["segments"] if s.get("s1")):
        _, g0, g1, c0, c1, a0, a1, lens, fstop = raw[seg["shot"]]
        cam = bpy.data.objects[seg["shot"]]
        if abs(cam.data.lens - lens) > 1e-6 or cam.data.dof.use_dof != (fstop > 0) or \
                (fstop > 0 and abs(cam.data.dof.aperture_fstop - fstop) > 1e-6):
            bad.append(f"{seg['shot']}: lens {cam.data.lens} / f{cam.data.dof.aperture_fstop} "
                       f"(dof {cam.data.dof.use_dof}) != stage-1 {lens} / f{fstop}")
        for f in (seg["f0"], (seg["f0"] + seg["f1"]) // 2, seg["f1"]):
            sc.frame_set(f)
            s = K.smoothstep((f - off - g0) / float(g1 - g0))
            c = np.array(c0) * (1 - s) + np.array(c1) * s
            aim = np.array(a0) * (1 - s) + np.array(a1) * s
            M = mw(cam)
            d = float(np.linalg.norm(M[:3, 3] - c))
            view = -M[:3, 2] / np.linalg.norm(M[:3, 2])
            want = (aim - M[:3, 3]) / np.linalg.norm(aim - M[:3, 3])
            ang = math.acos(max(-1.0, min(1.0, float(view @ want))))
            n += 1
            if d > TOL_CAM or ang > TOL_AIM:
                bad.append(f"{seg['shot']} @ {f} (stage-1 {f - off}): camera {d * 1000:.1f} mm from the stage-1 "
                           f"path, view {math.degrees(ang):.3f} deg off the stage-1 aim")
    return n


def check_stage1_fx(sc, P, M, bad):
    """Laser line during / outside the scan, stage-1 arc light flicker and spark systems at the embedded frames."""
    off = P["weld_offset"]
    scan = np.nonzero(P["stage1_laser_on"])[0] + 1          # stage-1 frames
    s1_iv = _runs(np.nonzero(P["stage1_arc_on"])[0] + 1)     # stage-1 arc intervals (= vfx weld intervals)
    # ---- laser line
    line = bpy.data.objects.get("LaserLine")
    if line is None:
        bad.append("no LaserLine object (vfx.laser_line of the stage-1 scan)")
    else:
        roots = {r.name for r in M["parts"]["roots"]}
        spool = [o for o in bpy.data.objects if o.type == 'MESH' and _root(o).name in roots and not o.hide_render]
        f_in = off + int(scan[len(scan) // 2])
        for f, on in ((f_in, True), (off + int(scan[0]) - 5, False), (off + int(scan[-1]) + 10, False)):
            sc.frame_set(f)
            sx = min(line.matrix_world.to_scale())
            if on != (sx > 0.5) or (not on and sx > 1e-6):
                bad.append(f"laser line scale {sx:.3f} at scene frame {f} ({'inside' if on else 'outside'} the scan)")
        sc.frame_set(f_in)
        dg = bpy.context.evaluated_depsgraph_get()
        ev = line.evaluated_get(dg)
        me = ev.to_mesh()
        pts = [ev.matrix_world @ v.co for v in me.vertices]
        ev.to_mesh_clear()
        near = 0
        for p in pts:
            best = 1e9
            for o in spool:
                oe = o.evaluated_get(dg)
                ok, loc, _n, _i = oe.closest_point_on_mesh(oe.matrix_world.inverted() @ p, distance=0.05)
                if ok:
                    best = min(best, ((oe.matrix_world @ loc) - p).length)
            near += best < TOL_LASER
        if near < 0.9 * len(pts):
            bad.append(f"laser line at scene frame {f_in}: only {near}/{len(pts)} vertices on the spool surface")
        print(f"laser line @ {f_in}: {near}/{len(pts)} vertices within {TOL_LASER * 1000:.0f} mm of the spool")
    # ---- stage-1 arc light: on exactly at the embedded arc frames, with the stage-1 flicker
    arc1 = bpy.data.objects.get("arc_point")
    lights = [o for o in bpy.data.objects if o.type == 'LIGHT' and o.parent is arc1]
    if arc1 is None or len(lights) != 1:
        bad.append(f"stage-1 arc light: expected one light on arc_point, found {[o.name for o in lights]}")
    else:
        ld = lights[0].data
        flick = dict(vfx._flicker_series(s1_iv, seed=11))
        for f1 in (int(s1_iv[0][0]), int(s1_iv[0][0]) + 40, 350, int(s1_iv[-1][1]), int(s1_iv[0][1]) + 1, 500):
            sc.frame_set(f1 + off)
            want = vfx.ARC_LIGHT_POWER * flick[f1] if f1 in flick else 0.0
            if abs(ld.energy - want) > 1e-3 * vfx.ARC_LIGHT_POWER:
                bad.append(f"stage-1 arc light at stage-1 frame {f1} (scene {f1 + off}): {ld.energy:.2f} W, "
                           f"stage-1 video {want:.2f} W")
    # ---- spark / droplet systems of the stage-1 arc: one pair per embedded weld interval, caches cover them
    ems = [o for o in bpy.data.objects if o.name.startswith("SparkEmitter") and o.parent is arc1]
    if len(ems) != 1:
        bad.append(f"stage-1 spark emitter: {[o.name for o in ems]}")
    else:
        want = sorted((a + off, b + off) for a, b in s1_iv)
        got = sorted({(int(ps.settings.frame_start), int(ps.settings.frame_end)) for ps in ems[0].particle_systems})
        if got != want:
            bad.append(f"stage-1 spark intervals {got} != embedded arc intervals {want}")
        for ps in ems[0].particle_systems:
            st, pc = ps.settings, ps.point_cache
            if pc.frame_start > st.frame_start or pc.frame_end < st.frame_end + st.lifetime:
                bad.append(f"{ps.name}: cache {pc.frame_start}-{pc.frame_end} does not cover emission "
                           f"{st.frame_start:.0f}-{st.frame_end:.0f} + life {st.lifetime:.0f}")


def _runs(frames):
    out = []
    for f in (int(v) for v in frames):
        if out and f == out[-1][1] + 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return [tuple(r) for r in out]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    editions.add_argument(ap)
    ap.add_argument("--env", action="store_true", help="with the hall environment (slower)")
    a = ap.parse_args()
    ed = editions.get(a.edition)
    S = build2.build_scene(with_env=a.env, with_vfx=True, edition=ed["name"])
    sc = S["scene"]
    vfx.bake_particles(sc)             # as render_frames: poses are exact only with baked spark caches (see above)
    P = S["plan"]
    M = S["modules"]
    ev = P["events"]
    off = P["weld_offset"]
    H = K.scaled_arm(L2.HANDLER_SCALE)
    TK = K.scaled_arm(L2.TACK_SCALE)
    W = RB.load_arm()
    TORCH = RB.tool_transform()
    frames = sorted({1, ev["kit_flange_grip"], ev["stn_elbow_contact"], ev["first_tack"] + 3, ev["stn_spool_grip"],
                     ev["entry_done"], ev["load_contact"], ev["studs_up"], ev["weld_first"] + 300, ev["unload_grip"] + 10,
                     ev["conv_load_contact"], ev["mark_start"] + 20, ev["store_contact"], P["n_frames"]})
    bad = []
    import i18n2
    import branding2
    if i18n2.get_lang() != ed["lang"] or branding2.get_brand() != ed["brand"] or S.get("edition", {}).get("name") != ed["name"]:
        bad.append(f"scene language / brand {i18n2.get_lang()} / {branding2.get_brand()} != edition {ed['name']}")

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
    # markers / cameras: the active camera (timeline markers, as the renderer switches them) over each segment
    import edl as EDL
    E = EDL.build_edl(P, ed["name"])
    shots = [s for s in E["segments"] if s["src"] == "s2"]
    marks = sorted((m.frame, m.camera.name) for m in sc.timeline_markers if m.camera)

    def active(f):
        cur = None
        for fr, nm in marks:
            if fr <= f:
                cur = nm
        return cur

    for s in shots:
        for f in (s["f0"], s["f1"]):
            sc.frame_set(f)
            if active(f) != s["shot"] or sc.camera is None or sc.camera.name != s["shot"]:
                bad.append(f"{s['shot']} @ {f}: marker camera {active(f)}, scene camera "
                           f"{sc.camera.name if sc.camera else None}")
    n_s1 = check_s1_cameras(sc, E, off, bad) if ed["stage1"] == "render" else 0
    check_stage1_fx(sc, P, M, bad)
    # beads: before / after the embedded weld
    bead = bpy.data.objects.get("spool_bead_A")
    if bead is not None:
        sc.frame_set(ev["load_contact"])
        p0 = float(bead["w0_prog"])
        sc.frame_set(ev["unload_grip"])
        p1 = float(bead["w0_prog"])
        if not (p0 <= 0.01 and p1 >= 0.99):
            bad.append(f"bead A progress before/after weld: {p0:.3f} / {p1:.3f}")
    print(f"edition {ed['name']}: checked {len(frames)} frames x 7 poses, {len(shots)} shot segments "
          f"({sum(1 for s in shots if s.get('s1'))} stage-1 shots, {n_s1} stage-1 camera samples)")
    print("RESULT:", "OK" if not bad else "FAIL")
    for b in bad[:30]:
        print("  -", b)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
