"""Stage-2 spool parts: the stage-1 spool split into three movable parts (flange, elbow, pipe) + tack welds.

Every part is an empty (`part_flange`, `part_elbow`, `part_pipe`) whose frame IS the spool frame of the assembled spool
(flange back-face centre, local Z up, pipe leg along local +X, see layout2).  A part "in its assembled position at spool
frame F" therefore has world pose F; the planner keys the world pose of each part root per frame with `set_pose`.

    P = split(sp)                      # sp = stage-1 spool.build(name="spool") dict (on the positioner, animated)
    set_pose(P["pipe"], T, frame=f)    # world pose of a part root (numpy 4x4)
    set_tack(P["tacks"]["A"][0], s, hot, frame=f)
    kit = build_kit(P, "kit1")          # loose linked-duplicate parts (no beads / tints / tacks)
    fin = build_finished(P, "fin1")     # one-root welded spool: beads A/B complete, heat tint visible, no glow

Objects of the stage-1 spool keep their names (`spool_bead_A` ... are referenced by the stage-1 animation); the new
objects use the `part_` prefix and live in the `Parts2` collection.
"""
import math

import bpy
import mathutils
import numpy as np

import tools  # noqa: F401  (demo_video on sys.path)
from cell import geom as G
from cell import layout as L1
from cell import materials
import kin
import layout2 as L2

COLLECTION = "Parts2"
PART_KEYS = ("flange", "elbow", "pipe")

# tack weld (m).  Both stage-1 joints are single-bevel V grooves (32 deg bevel on the flange hub / pipe end, square elbow
# end, 1.6 mm root face): a tack fills the groove from just under the root face to a low crown.  Its cross-section is
# the stage-1 bead profile (spool._bead_ring) scaled to TACK_CROWN_K of the bead height and TACK_INSET below it, so the
# tacks (kept at s = 1 after welding) stay hidden under the finished bead; its base lies inside the bevel walls.
_R_O = L1.PIPE_OD / 2
_ROOT_R = _R_O - L1.PIPE_WALL + 0.0016                                   # root-face radius (bottom of the groove)
_BEV = (L1.PIPE_WALL - 0.0016) / math.tan(math.radians(32))             # bevel length along the seam axis (12.3 mm)
TACK_HALF_WIDTH = max(L1.BEAD_WIDTH, 2 * _BEV + 0.004) / 2              # = stage-1 bead half width (14.3 mm)
TACK_CROWN_K = 0.8             # crown height as a fraction of the stage-1 bead height
TACK_INSET = 0.0003            # tack surface below the bead surface
TACK_EDGE_DEPTH = 0.0015       # tack edges (toes) sink this far below the pipe surface
TACK_GROW_POWER = 0.5          # cross-section grows as s ** 0.5 while the length grows as s


# ------------------------------------------------------------------ helpers
def _collection():
    col = bpy.data.collections.get(COLLECTION)
    if col is None:
        col = bpy.data.collections.new(COLLECTION)
    if col.name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(col)
    return col


def _classify(suffix):
    """Stage-1 spool child name without the spool prefix (e.g. '_hole3') -> part key."""
    if suffix in ("_elbow", "_elbow_bevel_A", "_elbow_bevel_B", "_mark_elbow", "_bead_B", "_tint_B"):
        return "elbow"
    if suffix in ("_pipe", "_mark_pipe"):
        return "pipe"
    return "flange"       # _flange, _flange_face, _hole*, _bead_A, _tint_A (and anything unexpected)


def _part_root(name, matrix, col):
    e = G.empty(name, collection=col, size=0.12, display='ARROWS')
    e.rotation_mode = 'QUATERNION'
    e.matrix_world = matrix
    return e


def _attach(child, parent, local):
    """Parent with an explicit local transform (identity parent inverse)."""
    child.parent = parent
    child.matrix_parent_inverse = mathutils.Matrix.Identity(4)
    child.matrix_basis = local


# ------------------------------------------------------------------ tack welds
def tack_frame(seam, angle_deg):
    """Spool-local 4x4 frame of a tack weld: origin on the seam circle (radius PIPE_R), X = seam tangent (direction of
    increasing angle), Y = seam axis, Z = outward surface normal.  seam 'A': point (R cos a, R sin a, SEAM_A_Z);
    seam 'B': point (SEAM_B_X, R cos b, PIPE_AXIS_Z + R sin b)  (layout2 formulas)."""
    a = math.radians(angle_deg)
    if seam == "A":
        c, e1, e2 = np.array([0.0, 0.0, L2.SEAM_A_Z]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    else:
        c, e1, e2 = np.array([L2.SEAM_B_X, 0.0, L2.PIPE_AXIS_Z]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
    n = math.cos(a) * e1 + math.sin(a) * e2
    t = -math.sin(a) * e1 + math.cos(a) * e2
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = t, np.cross(n, t), n, c + L2.PIPE_R * n
    return T


def _tack_radii(u, w):
    """(top, base) radius of the tack at u (along the seam, -1..1) and w (across, -1..1): the top follows the stage-1
    bead profile (lowered), the base is a V below the bevel walls (root at the centre, toes at the edges); toward the
    ends the top dives into the groove (crater) until it meets the base."""
    t = (w + 1.0) / 2.0
    top = (_R_O - 0.0012 + TACK_CROWN_K * L1.BEAD_HEIGHT * max(0.0, math.sin(math.pi * t)) ** 0.8 - TACK_INSET)
    top = max(top, _R_O - TACK_EDGE_DEPTH)
    base = (_ROOT_R - 0.0005) + ((_R_O - TACK_EDGE_DEPTH - 0.0003) - (_ROOT_R - 0.0005)) * abs(w)
    e = max(0.02, max(0.0, 1.0 - u * u) ** 0.4)
    return base + (top - base) * e, base


def _tack_mesh():
    """Closed tack body in the tack frame, curved along the seam circle (radius PIPE_R about the local point (0, 0, -R)):
    filled V groove + low crown (see _tack_radii), TACK_LEN long, 2 * TACK_HALF_WIDTH wide."""
    me = bpy.data.meshes.get("part_tack")
    if me is not None:
        return me
    R = L2.PIPE_R
    half_ang = L2.TACK_LEN / 2 / R
    nu, nw = 16, 12
    verts = []
    for layer in (0, 1):                       # 0 = crown surface, 1 = base (inside the bevel walls / root face)
        for i in range(nu + 1):
            u = -1.0 + 2.0 * i / nu
            phi = u * half_ang
            for j in range(nw + 1):
                w = -1.0 + 2.0 * j / nw
                rr = _tack_radii(u, w)[layer]
                verts.append((rr * math.sin(phi), w * TACK_HALF_WIDTH, rr * math.cos(phi) - R))
    off = (nu + 1) * (nw + 1)
    idx = lambda layer, i, j: layer * off + i * (nw + 1) + j
    faces = []
    for i in range(nu):
        for j in range(nw):
            faces.append((idx(0, i, j), idx(0, i + 1, j), idx(0, i + 1, j + 1), idx(0, i, j + 1)))
            faces.append((idx(1, i, j), idx(1, i, j + 1), idx(1, i + 1, j + 1), idx(1, i + 1, j)))
    for i in range(nu):             # side strips along w = -1 / +1
        faces.append((idx(0, i, 0), idx(1, i, 0), idx(1, i + 1, 0), idx(0, i + 1, 0)))
        faces.append((idx(0, i, nw), idx(0, i + 1, nw), idx(1, i + 1, nw), idx(1, i, nw)))
    for j in range(nw):             # end strips u = -1 / +1
        faces.append((idx(0, 0, j), idx(0, 0, j + 1), idx(1, 0, j + 1), idx(1, 0, j)))
        faces.append((idx(0, nu, j), idx(1, nu, j), idx(1, nu, j + 1), idx(0, nu, j + 1)))
    me = bpy.data.meshes.new("part_tack")
    me.from_pydata(verts, [], faces)
    me.update()
    me.validate()
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    me.materials.append(_tack_material())
    return me


def _tack_material():
    """Dull grey tack metal (as the stage-1 weld bead, but always opaque) + hot emission from the object property
    'hot' (0..1): dark red -> orange -> yellow-white."""
    m = bpy.data.materials.get("part_tack")
    if m is not None:
        return m
    m = bpy.data.materials.new("part_tack")
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.16, 0.15, 0.145, 1.0)
    b.inputs["Metallic"].default_value = 0.8
    b.inputs["Roughness"].default_value = 0.62
    att = nt.nodes.new("ShaderNodeAttribute")
    att.attribute_type = 'OBJECT'
    att.attribute_name = "hot"
    col, p = materials._heat_ramp(nt, att.outputs["Fac"], power=1.5)
    nt.links.new(col, b.inputs["Emission Color"])
    es = nt.nodes.new("ShaderNodeMath")
    es.operation = 'MULTIPLY'
    es.inputs[1].default_value = 22.0
    nt.links.new(p, es.inputs[0])
    nt.links.new(es.outputs[0], b.inputs["Emission Strength"])
    # fine ripples / spatter texture
    tc = nt.nodes.new("ShaderNodeTexCoord")
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 900.0
    nz.inputs["Detail"].default_value = 2.0
    nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.5
    bump.inputs["Distance"].default_value = 0.0004
    nt.links.new(nz.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    return m


def _build_tacks(roots, col, tacks_a, tacks_b):
    me = _tack_mesh()
    out = dict(A=[], B=[])
    for seam, angles, parent in (("A", tacks_a, roots["flange"]), ("B", tacks_b, roots["elbow"])):
        for i, ang in enumerate(angles):
            ob = bpy.data.objects.new(f"part_tack_{seam}{i}", me)
            col.objects.link(ob)
            _attach(ob, parent, G.M(tack_frame(seam, ang)))
            ob["seam"] = seam
            ob["angle"] = float(ang)
            ob["hot"] = 0.0
            set_tack(ob, 0.0)
            out[seam].append(ob)
    return out


def place_tack(obj, angle_deg):
    """Move an existing tack along its seam (the planner may shift a tack by up to +-20 deg for reach)."""
    loc, rot, _ = G.M(tack_frame(obj["seam"], angle_deg)).decompose()
    obj.location = loc                      # keep the growth scale (may be 0) untouched
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = rot
    obj["angle"] = float(angle_deg)


# ------------------------------------------------------------------ public API
def split(sp, tacks_a=None, tacks_b=None):
    """Split the stage-1 spool into three part roots at the spool root's current world pose (see module docstring).

    Returns dict(flange, elbow, pipe, roots, tacks, meshes, beads, tints):
      flange/elbow/pipe = part root empties;  roots = [flange, elbow, pipe];
      tacks = dict(A=[3 obj], B=[3 obj]) (children of flange / elbow, invisible until set_tack);
      meshes = dict(part -> [mesh objects]) incl. beads/tints; beads/tints = dict(A=obj, B=obj).
    The stage-1 root empty is deleted and sp["root"] is set to None."""
    root = sp["root"]
    prefix = root.name[:-len("_root")] if root.name.endswith("_root") else root.name
    col = _collection()
    bpy.context.view_layer.update()
    Mw = root.matrix_world.copy()
    Mi = Mw.inverted()
    roots = {k: _part_root(f"part_{k}", Mw, col) for k in PART_KEYS}
    meshes = {k: [] for k in PART_KEYS}
    for ch in sorted(root.children, key=lambda o: o.name):
        suffix = ch.name[len(prefix):] if ch.name.startswith(prefix) else ch.name
        key = _classify(suffix)
        local = Mi @ ch.matrix_world
        _attach(ch, roots[key], local)
        meshes[key].append(ch)
    bpy.data.objects.remove(root, do_unlink=True)
    sp["root"] = None
    bpy.context.view_layer.update()
    tacks = _build_tacks(roots, col, L2.TACKS_A if tacks_a is None else tacks_a, L2.TACKS_B if tacks_b is None else tacks_b)
    beads = {k: bpy.data.objects.get(f"{prefix}_bead_{k}") for k in ("A", "B")}
    tints = {k: bpy.data.objects.get(f"{prefix}_tint_{k}") for k in ("A", "B")}
    return dict(flange=roots["flange"], elbow=roots["elbow"], pipe=roots["pipe"],
                roots=[roots[k] for k in PART_KEYS], tacks=tacks, meshes=meshes, beads=beads, tints=tints)


def set_pose(root, T, frame=None):
    """World pose (numpy 4x4) of a part root -> location + rotation_quaternion (sign kept continuous)."""
    T = np.asarray(T, dtype=float)
    if root.rotation_mode != 'QUATERNION':
        root.rotation_mode = 'QUATERNION'
    if root.parent is not None:
        P = G.np4(root.parent.matrix_world @ root.matrix_parent_inverse)
        T = kin.inv(P) @ T
    q = kin.mat2quat(T[:3, :3])
    q_prev = np.array(root.rotation_quaternion[:])
    if float(np.dot(q, q_prev)) < 0.0:
        q = -q
    root.location = [float(v) for v in T[:3, 3]]
    root.rotation_quaternion = [float(v) for v in q]
    if frame is not None:
        root.keyframe_insert("location", frame=frame)
        root.keyframe_insert("rotation_quaternion", frame=frame)


def set_tack(obj, s, hot=0.0, frame=None):
    """Tack growth s in [0, 1] (0 = invisible, 1 = full TACK_LEN bead) and glow hot in [0, 1]."""
    s = min(max(float(s), 0.0), 1.0)
    w = s ** TACK_GROW_POWER
    obj.scale = (s, w, w)
    obj["hot"] = min(max(float(hot), 0.0), 1.0)
    if frame is not None:
        obj.keyframe_insert("scale", frame=frame)
        obj.keyframe_insert('["hot"]', frame=frame)


def _copy_part(src_root, meshes, root_name, col, skip=()):
    bpy.context.view_layer.update()
    root = _part_root(root_name, src_root.matrix_world.copy(), col)
    out = []
    for ob in meshes:
        if ob.name in skip:
            continue
        c = ob.copy()                       # linked duplicate: shares the mesh datablock
        c.name = f"{root_name}_{ob.name.split('_', 1)[-1]}"
        col.objects.link(c)
        _attach(c, root, ob.matrix_basis.copy())
        out.append(c)
    return root, out


def _kit_name(name):
    return name if name.startswith("part_") else f"part_{name}"


def build_kit(main, name):
    """Another set of loose blanks (flange, elbow, pipe): linked duplicates of the main part meshes without beads,
    heat tints and tacks; same root frames (spool frame), created at the main roots' current world poses."""
    col = _collection()
    base = _kit_name(name)
    skip = {o.name for o in list(main["beads"].values()) + list(main["tints"].values()) if o is not None}
    out = {}
    for k in PART_KEYS:
        root, _ = _copy_part(main[k], main["meshes"][k], f"{base}_{k}", col, skip)
        out[k] = root
    return out


def build_finished(main, name):
    """A finished welded spool under one root (spool frame, at the main flange root's current pose): linked duplicates
    of all part meshes, beads A and B complete, heat tint visible, no glow, no tacks.  Returns the root empty."""
    col = _collection()
    base = _kit_name(name)
    bpy.context.view_layer.update()
    root = _part_root(base, main["flange"].matrix_world.copy(), col)
    ring_names = {o.name for o in list(main["beads"].values()) + list(main["tints"].values()) if o is not None}
    for k in PART_KEYS:
        for ob in main["meshes"][k]:
            c = ob.copy()
            c.name = f"{base}_{ob.name.split('_', 1)[-1]}"
            col.objects.link(c)
            _attach(c, root, ob.matrix_basis.copy())
            if ob.name in ring_names:
                c.animation_data_clear()        # drop the progress drivers / keyed props of the stage-1 rings
                c["w0"] = [0.0, 1.01, 1.0, 0.0]     # [start, prog, dir, hot]: whole ring welded, cold
                for i in range(1, materials.N_ARCS):
                    c[f"w{i}"] = [0.0, 0.0, 0.0, 0.0]
                for i in range(materials.N_ARCS):
                    for key in materials.PROP_KEYS:
                        if f"w{i}_{key}" in c:
                            c[f"w{i}_{key}"] = 1.01 if (i == 0 and key == "prog") else (1.0 if (i == 0 and key == "dir") else 0.0)
                if "tint_on" in c:
                    c["tint_on"] = 1.0
    return root


def obstacles():
    """Parts are moved by the planner: no static geometry."""
    return []
