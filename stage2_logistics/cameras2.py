"""Stage-2 camera shots.  Each shot is tied to choreography events of plan2 (so re-timing the plan keeps the shots on
the action) and is used in this order by the edit (edl.py).  Rules from the stage-1 critique: cameras above the fences
or inside a zone, outside the swing of carried parts and robot arms, no cut while an arm crosses the lens.

SHOTS: (name, (start_event, offset), (end_event, offset), cam_from, cam_to, aim_from, aim_to, lens_mm, fstop, caption_key)
"""
import bpy

import tools  # noqa: F401
from cell import geom as G

PRE_SHOTS = [
    ("S2_01_wide", ("start", 0), ("start", 112), (-9.4, -8.6, 6.4), (-8.3, -7.6, 5.9), (-4.6, 0.4, 0.7), (-4.9, 0.6, 0.8), 24, 0, None),
    ("S2_02_flange", ("stn_flange_contact", -50), ("clamp_flange", 12), (-2.95, 0.55, 2.45), (-3.05, 0.70, 2.35), (-4.35, 2.15, 0.95), (-4.40, 2.25, 0.95), 32, 0, "flange"),
    ("S2_03_elbow", ("stn_elbow_contact", -40), ("clamp_elbow", 8), (-5.75, 0.95, 2.05), (-5.65, 1.05, 2.00), (-4.40, 2.25, 1.10), (-4.40, 2.30, 1.10), 40, 5.6, "elbow"),
    ("S2_04_pipe", ("stn_pipe_contact", -40), ("gap_ok", 14), (-3.05, 0.35, 2.15), (-3.15, 0.45, 2.10), (-4.40, 1.95, 1.15), (-4.40, 2.00, 1.15), 34, 0, "pipe"),
    ("S2_05_tackA", ("tack_A0", -12), ("tack_A0", 84), (-3.75, 1.55, 1.45), (-3.72, 1.62, 1.42), (-4.36, 2.50, 0.93), (-4.36, 2.50, 0.93), 50, 4.0, "tack"),
    # ("S2_06_tackB", ("last_tack_end", -52), ("last_tack_end", 6), (-3.55, 1.05, 1.95), (-3.60, 1.10, 1.92), (-4.36, 2.12, 1.30), (-4.36, 2.12, 1.30), 45, 4.5, None),   (optional shot, not in the default edit)
    ("S2_07_pick", ("stn_spool_grip", -24), ("stn_spool_grip", 66), (-6.9, -1.0, 3.1), (-6.7, -0.9, 3.0), (-4.4, 1.3, 1.5), (-4.1, 0.9, 1.7), 24, 0, "carry"),
    ("S2_08_load", ("entry_done", -56), ("studs_up", 12), (0.95, 1.95, 2.75), (0.90, 1.85, 2.70), (-1.05, 0.00, 1.90), (-0.80, 0.00, 1.80), 28, 0, "load"),
    ("S2_09_ready", ("handler_out", -30), ("ready", 6), (-5.3, 2.30, 3.05), (-5.1, 2.20, 2.95), (-1.0, 0.0, 1.75), (-1.0, 0.0, 1.75), 30, 0, "ready"),
]

POST_SHOTS = [
    ("S2_10_done", ("weld_last", 1), ("weld_last", 72), (-5.4, -2.2, 3.1), (-5.2, -2.05, 3.0), (0.4, -0.3, 1.4), (0.4, -0.3, 1.4), 26, 0, "done"),
    ("S2_11_unload", ("unload_grip", -56), ("studs_down", 8), (0.95, 1.95, 2.75), (0.90, 1.88, 2.72), (-0.90, 0.00, 1.95), (-0.70, 0.00, 1.90), 28, 0, "unload"),
    ("S2_12_carrier", ("conv_load_contact", -52), ("conv_load_contact", 12), (-2.9, -0.9, 2.5), (-3.0, -1.0, 2.4), (-4.3, -2.0, 1.3), (-4.3, -2.2, 1.2), 28, 0, "carrier"),
    # ("S2_13_convey", ("at_qc", -56), ("at_qc", 4), (-4.5, -0.75, 1.95), (-5.7, -0.75, 1.95), (-4.7, -2.4, 1.1), (-6.0, -2.4, 1.2), 26, 0, None),   (optional shot, not in the default edit)
    ("S2_14_mark", ("mark_start", -6), ("mark_end", 6), (-5.28, -2.47, 1.82), (-5.30, -2.44, 1.80), (-6.0, -2.37, 1.42), (-6.0, -2.37, 1.42), 45, 4.0, "mark"),
    ("S2_15_scan", ("scan_start", -4), ("qc_ok", 22), (-6.62, -1.72, 1.98), (-6.58, -1.76, 1.95), (-6.0, -2.47, 1.40), (-6.0, -2.47, 1.40), 42, 4.5, "scan"),
    ("S2_16_store", ("conv_end_grip", 30), ("store_contact", 16), (-9.0, 2.9, 2.6), (-9.2, 2.8, 2.5), (-9.4, -0.6, 1.1), (-9.9, -0.4, 0.9), 24, 0, "store"),
    ("S2_17_final", ("agv_arrive", -104), ("end", 0), (-14.5, 5.8, 6.2), (-13.8, 5.2, 5.8), (-5.0, -0.3, 0.8), (-5.0, -0.3, 0.8), 26, 0, None),
]
SHOTS = PRE_SHOTS + POST_SHOTS


def shot_range(shot, events):
    (e0, o0), (e1, o1) = shot[1], shot[2]
    return events[e0] + o0, events[e1] + o1


def build(scene, events, shots=None, collection=None):
    """Create the cameras (animated from -> to over each shot, ease in/out) and bind them with timeline markers.
    Returns list of (camera object, f0, f1)."""
    shots = shots or SHOTS
    col = collection or bpy.data.collections.new("Cameras2")
    if collection is None:
        scene.collection.children.link(col)
    for m in list(scene.timeline_markers):
        scene.timeline_markers.remove(m)
    out = []
    for (name, a, b, c0, c1, a0, a1, lens, fstop, _cap) in shots:
        f0, f1 = shot_range((name, a, b), events)
        cd = bpy.data.cameras.new(name)
        cd.lens = lens
        cd.sensor_width = 36
        cd.clip_start = 0.05
        cam = bpy.data.objects.new(name, cd)
        col.objects.link(cam)
        aim = G.empty(name + "_aim", collection=col, size=0.1)
        for f, c, t in ((f0, c0, a0), (f1, c1, a1)):
            cam.location = c
            cam.keyframe_insert("location", frame=f)
            aim.location = t
            aim.keyframe_insert("location", frame=f)
        for ob in (cam, aim):
            for fc in ob.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = 'BEZIER'
                    kp.easing = 'EASE_IN_OUT'
        con = cam.constraints.new('TRACK_TO')
        con.target = aim
        con.track_axis = 'TRACK_NEGATIVE_Z'
        con.up_axis = 'UP_Y'
        if fstop > 0:
            cd.dof.use_dof = True
            cd.dof.focus_object = aim
            cd.dof.aperture_fstop = fstop
        m = scene.timeline_markers.new(name, frame=f0)
        m.camera = cam
        out.append((cam, f0, f1))
    scene.camera = out[0][0]
    return out
