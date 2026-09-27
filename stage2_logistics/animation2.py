"""Keyframe the stage-2 plan (plan2.solve()) onto the Blender scene.

* Robot joints, carriages, positioner axes and part poses are baked directly into F-curves (foreach_set: fast even
  for ~4000 frames); the stage-1 keys of the positioner / welding robot are replaced by the stage-2 arrays (the
  stage-1 choreography is embedded in them at WELD_OFFSET).
* Stage-1 keyframed properties that are not re-planned (weld-bead progress, heat tint, arc_on) are shifted by
  WELD_OFFSET so the welded beads appear exactly when the stage-1 welding happens in the stage-2 timeline.
* Every other channel goes through the equipment modules' setters (set_clamp, set_carrier, ...), keyed only at the
  frames where the value changes (constant runs are collapsed), then the interpolation of the F-curves those setters
  touched is set to LINEAR (continuous channels) or CONSTANT (switches).
"""
import math

import bpy
import numpy as np

import tools  # noqa: F401
import layout2 as L2
from cell import layout as L

INTERP = {"CONSTANT": 0, "LINEAR": 1}


# ============================================================================ low-level baking
def _fcurve(idb, data_path, index, frame1_value):
    """Get (creating through keyframe_insert, which also creates action / slot) the F-curve of idb.data_path[index]."""
    ad = idb.animation_data
    if ad is None or ad.action is None or ad.action.fcurves.find(data_path, index=index) is None:
        prop = idb.path_resolve(data_path)
        if hasattr(prop, "__len__") and not isinstance(prop, str):
            prop[index] = frame1_value
        else:
            _assign(idb, data_path, frame1_value)
        idb.keyframe_insert(data_path, index=index, frame=1)
    return idb.animation_data.action.fcurves.find(data_path, index=index)


def _assign(idb, data_path, value):
    if data_path.startswith('["'):
        idb[data_path[2:-2]] = value
    else:
        setattr(idb, data_path, value)


def bake(idb, data_path, index, values, interp="LINEAR", frames=None):
    """Replace the F-curve idb.data_path[index] by keys at `frames` (default 1..len(values)) with `values`."""
    values = np.asarray(values, dtype=float)
    if frames is None:
        frames = np.arange(1, len(values) + 1)
    frames = np.asarray(frames, dtype=float)
    keep = _compress(values)
    fr, va = frames[keep], values[keep]
    fc = _fcurve(idb, data_path, index, float(va[0]))
    kps = fc.keyframe_points
    kps.clear()
    kps.add(len(fr))
    co = np.empty(2 * len(fr))
    co[0::2] = fr
    co[1::2] = va
    kps.foreach_set("co", co)
    kps.foreach_set("interpolation", [INTERP[interp]] * len(fr))
    fc.update()
    return fc


def _compress(v, tol=1e-9):
    """Indices to keep so that LINEAR interpolation reproduces v exactly on integer frames: every frame that differs
    from a neighbour (changing runs) plus the ends of constant runs."""
    n = len(v)
    if n <= 2:
        return np.arange(n)
    d_prev = np.abs(np.diff(v, prepend=v[0] - 1.0))
    d_next = np.abs(np.diff(v, append=v[-1] - 1.0))
    keep = (d_prev > tol) | (d_next > tol)
    # a strictly linear stretch inside a changing run could also be collapsed; keeping it is harmless
    keep[0] = keep[-1] = True
    return np.nonzero(keep)[0]


def bake_vec(idb, data_path, values, interp="LINEAR"):
    values = np.asarray(values, dtype=float)
    for k in range(values.shape[1]):
        bake(idb, data_path, k, values[:, k], interp)


def remove_fcurves(idb, data_path, indices=None):
    ad = idb.animation_data
    if ad is None or ad.action is None:
        return
    for fc in list(ad.action.fcurves):
        if fc.data_path == data_path and (indices is None or fc.array_index in indices):
            ad.action.fcurves.remove(fc)


def shift_keys(idb, offset, data_paths=None):
    """Shift every keyframe (and handles) of idb's action by `offset` frames."""
    ad = idb.animation_data
    if ad is None or ad.action is None:
        return
    for fc in ad.action.fcurves:
        if data_paths is not None and fc.data_path not in data_paths:
            continue
        for kp in fc.keyframe_points:
            kp.co.x += offset
            kp.handle_left.x += offset
            kp.handle_right.x += offset
        fc.update()


# ============================================================================ setter-driven channels
def _snapshot():
    snap = {}
    for act in bpy.data.actions:
        for fc in act.fcurves:
            snap[(act.name, fc.data_path, fc.array_index)] = len(fc.keyframe_points)
    return snap


def key_setter(setter, values, interp="LINEAR", discrete=False):
    """Call setter(value, frame) at the frames where `values` (per scene frame, scalar or row) changes, then set the
    interpolation of the F-curves that the calls created / extended."""
    values = np.asarray(values)
    before = _snapshot()
    if values.ndim == 1:
        keep = _compress(values.astype(float)) if not discrete else _changes(values)
    else:
        mask = np.zeros(len(values), dtype=bool)
        for k in range(values.shape[1]):
            mask[_compress(values[:, k].astype(float))] = True
        keep = np.nonzero(mask)[0]
    for i in keep:
        v = values[i]
        setter(v.tolist() if hasattr(v, "tolist") and np.ndim(v) else (v.item() if hasattr(v, "item") else v), int(i + 1))
    after = _snapshot()
    code = INTERP["CONSTANT" if discrete else interp]
    for act in bpy.data.actions:
        for fc in act.fcurves:
            k = (act.name, fc.data_path, fc.array_index)
            if after.get(k, 0) != before.get(k, 0):
                fc.keyframe_points.foreach_set("interpolation", [code] * len(fc.keyframe_points))
                fc.update()


def _changes(v):
    idx = [0]
    for i in range(1, len(v)):
        if v[i] != v[i - 1]:
            idx.append(i)
    return np.array(idx)


# ============================================================================ poses
def mat_to_loc_quat(T):
    """(n,4,4) -> (n,3) locations, (n,4) quaternions (w,x,y,z) with sign continuity."""
    import kin as K
    loc = T[:, :3, 3].copy()
    q = np.array([K.mat2quat(M[:3, :3]) for M in T])
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return loc, q


def bake_pose(ob, T):
    ob.parent = None
    ob.rotation_mode = 'QUATERNION'
    loc, q = mat_to_loc_quat(np.asarray(T))
    bake_vec(ob, "location", loc)
    bake_vec(ob, "rotation_quaternion", q)


def bake_joints(joints, Q):
    """Robot joint empties (rotation_axis_angle[0] = q, as robot_build.set_q)."""
    for j, e in enumerate(joints):
        remove_fcurves(e, "rotation_axis_angle")
        bake(e, "rotation_axis_angle", 0, np.asarray(Q)[:, j])


# ============================================================================ main entry
STUD_KEYS = [0, 2, 4, 6, 8, 10]


def apply(scene, P, M):
    """Keyframe everything.  M: dict of the built modules:
        pos, robot (stage-1 welder), spool (stage-1 dict), parts (parts.split result), extra (dict name -> root with a
        (n,4,4) pose in P['parts'] or a static 4x4), handler, tack, station, conveyor, qc, mark, storage, env2
    """
    n = P["n_frames"]
    off = P["weld_offset"]
    scene.frame_start, scene.frame_end = 1, n

    # ---------------------------------------------------------------- stage-1 machines: replaced by the plan arrays
    pos, rob = M["pos"], M["robot"]
    remove_fcurves(pos["tilt"], "rotation_euler")
    remove_fcurves(pos["rot"], "rotation_euler")
    bake(pos["tilt"], "rotation_euler", 0, -np.radians(P["tilt"]))
    bake(pos["rot"], "rotation_euler", 2, np.radians(P["rot"]))
    remove_fcurves(rob["carriage"], "location")
    bake(rob["carriage"], "location", 1, P["welder_track"])
    bake_joints(rob["joints"], P["welder_q"])
    # stage-1 keyed properties (bead progress, heat tint, arc switch) move to the embedded weld window
    sp = M["spool"]
    for b in sp["beads"].values():
        shift_keys(b, off)
    for t in sp.get("tints", {}).values():
        shift_keys(t, off)
    arc1 = bpy.data.objects.get("arc_point")
    if arc1 is not None:
        shift_keys(arc1, off)

    # clamp studs (the stage-1 flange bolts on the faceplate): retracted while the faceplate is empty
    for k in STUD_KEYS:
        for name in (f"pos_bolt{k}", f"pos_washer{k}"):
            ob = bpy.data.objects.get(name)
            if ob is None:
                continue
            z0 = ob.location[2]
            ob["stud_z0"] = z0
            bake(ob, "location", 2, z0 - (1.0 - P["stud"]) * L2.STUD_DROP)

    # ---------------------------------------------------------------- parts
    parts = M["parts"]
    for name in ("flange", "elbow", "pipe"):
        bake_pose(parts[name], P["parts"][name])
    for name, (root, pose) in M.get("extra", {}).items():
        T = P["parts"].get(pose) if isinstance(pose, str) else pose
        if T is None:
            continue
        T = np.asarray(T)
        if T.ndim == 2:
            root.parent = None
            root.rotation_mode = 'QUATERNION'
            import kin as K
            root.location = T[:3, 3]
            root.rotation_quaternion = K.mat2quat(T[:3, :3])
        else:
            bake_pose(root, T)
    import parts as PT
    for key, obs in parts["tacks"].items():
        for i, ob in enumerate(obs):
            s = P["tack_s"][f"{key}{i}"]
            h = P["tack_hot"][f"{key}{i}"]
            key_setter(lambda v, f, ob=ob: PT.set_tack(ob, v[0], v[1], frame=f), np.stack([s, h], axis=1))

    # ---------------------------------------------------------------- handler + gripper
    import handler_robot as HR
    import gripper as GR
    hnd = M["handler"]
    bake(hnd["carriage"], "location", 0, P["handler_x"])
    if hnd.get("chain_bend") is not None:             # energy-chain U-bend follows the carriage (handler_robot.set_track)
        bake(hnd["chain_bend"], "location", 0, np.array([HR.chain_bend_x(x) for x in P["handler_x"]]))
    bake_joints(hnd["joints"], P["handler_q"])
    key_setter(lambda v, f: GR.set_open(hnd["gripper"], v, frame=f), P["handler_grip"])

    # ---------------------------------------------------------------- tack robot
    import tack_robot as TR
    tk = M["tack"]
    bake_joints(tk["joints"], P["tack_q"])
    key_setter(lambda v, f: TR.set_arc(tk, v, frame=f), P["tack_arc"], discrete=True)

    # ---------------------------------------------------------------- station
    import assembly_station as AS
    st = M["station"]
    for name, arr in P["clamps"].items():
        key_setter(lambda v, f, name=name: AS.set_clamp(st, name, v, frame=f), arr)
    key_setter(lambda v, f: AS.set_lamp(st, v, frame=f), P["station_lamp"], discrete=True)

    # ---------------------------------------------------------------- conveyor, QC
    import conveyor as CV
    import marking_qc as QC
    cv = M["conveyor"]
    key_setter(lambda v, f: CV.set_carrier(cv, 0, v, frame=f), P["carrier_x"])
    key_setter(lambda v, f: CV.set_rollers(cv, v, frame=f), P["roller"])
    qc = M["qc"]
    key_setter(lambda v, f: QC.set_carriage(qc, v, frame=f), P["qc_y"])
    key_setter(lambda v, f: QC.set_marker(qc, v, frame=f), P["marker"], discrete=True)
    key_setter(lambda v, f: QC.set_mark(M["mark"], v[0], v[1], frame=f), np.stack([P["mark_reveal"], P["mark_hot"]], axis=1))
    names = {0: "off", 1: "amber", 2: "green"}
    key_setter(lambda v, f: QC.set_tower(qc, names[int(v)], frame=f), P["tower"], discrete=True)
    key_setter(lambda v, f: QC.set_display(qc, int(v), frame=f), P["display"], discrete=True)

    # ---------------------------------------------------------------- kit cassette: pipe-buffer presence LEDs
    import cassette as CS
    kit = M.get("kit")
    if kit is not None and hasattr(CS, "set_presence"):
        f_taken = P["events"].get("buf_pipe_grip")
        for slot in range(len(L2.PIPE_BUFFER_X)):
            on = np.ones(n)
            if slot == 0 and f_taken:
                on[f_taken + 6:] = 0.0          # the sensor sees the slot empty once the pipe is lifted
            key_setter(lambda v, f, slot=slot: CS.set_presence(kit, slot, v, frame=f), on, discrete=True)

    # ---------------------------------------------------------------- environment, AGV
    import environment2 as E2
    import storage as SG
    key_setter(lambda v, f: E2.set_muting(M["env2"], v, frame=f), P["muting"], discrete=True)
    sto = M["storage"]
    key_setter(lambda v, f: SG.set_agv(sto, v[0], v[1], v[2], frame=f), P["agv"])
    key_setter(lambda v, f: SG.set_agv_lift(sto, v, frame=f), P["agv_lift"])
    key_setter(lambda v, f: SG.set_beacon(sto, v, frame=f), P["agv_beacon"], discrete=True)
    scene.frame_set(1)
