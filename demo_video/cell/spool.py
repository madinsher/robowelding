"""DN250 pipe spool: weld-neck flange + 90° LR elbow + straight pipe, with two weld-bead rings.

Spool local frame: flange back face on z=0, flange axis +Z, elbow bends toward +X, pipe along +X.
"""
import math
import bpy
import numpy as np
import mathutils
from . import layout as L
from . import geom as G
from . import materials


def _bevel_profile_outer(r_o, r_i, z_end, direction, root_face=0.0016, angle=math.radians(32)):
    """Return profile points for a bevelled pipe end.  direction=+1: bevel at the +h end."""
    # V-bevel: outer surface chamfered from (r_o, z_end - d*t) to (r_i + root_face, z_end)
    t = (r_o - r_i - root_face) / math.tan(angle)
    return t


def build(collection=None, name="spool"):
    col = collection or bpy.data.collections.new("Spool")
    if col.name not in bpy.context.scene.collection.children and collection is None:
        bpy.context.scene.collection.children.link(col)
    root = G.empty(f"{name}_root", collection=col, size=0.2)

    r_o = L.PIPE_OD / 2
    r_i = r_o - L.PIPE_WALL
    steel = materials.get("steel_pipe")
    bevel = materials.get("bevel_steel")
    machined = materials.get("machined_steel")
    ang = math.radians(32)
    bev = (r_o - r_i - 0.0016) / math.tan(ang)      # axial length of the bevel

    # --- weld-neck flange: revolved profile around Z (z from 0 to hub end at SEAM_A_Z)
    zf = L.FLANGE_THK
    hub_r0 = r_o + 0.028      # hub base radius
    prof = [
        (r_i, 0.0), (L.FLANGE_OD / 2 - 0.004, 0.0), (L.FLANGE_OD / 2, 0.004),
        (L.FLANGE_OD / 2, zf - 0.003), (L.FLANGE_OD / 2 - 0.003, zf),
        (hub_r0 + 0.012, zf), (hub_r0, zf + 0.012),
        (r_o, L.SEAM_A_Z - 0.035), (r_o, L.SEAM_A_Z - bev),
        (r_i + 0.0016, L.SEAM_A_Z), (r_i, L.SEAM_A_Z),
    ]
    flange = G.revolve(f"{name}_flange", prof, segments=128, collection=col)
    flange.data.materials.append(steel)
    flange.data.materials.append(bevel)
    # assign bevel material to the chamfer faces (last two profile segments)
    _assign_by_profile(flange, len(prof), {len(prof) - 3, len(prof) - 2}, 1)
    # raised face ring on the back
    rf = G.revolve(f"{name}_flange_face", [(r_i + 0.002, -0.002), (0.176, -0.002), (0.176, 0.0), (r_i + 0.002, 0.0)], segments=96, collection=col)
    rf.data.materials.append(machined)
    # bolt holes as dark discs (cheap) on both faces
    for k in range(L.FLANGE_HOLES):
        a = 2 * math.pi * (k + 0.5) / L.FLANGE_HOLES
        x, y = L.FLANGE_PCD / 2 * math.cos(a), L.FLANGE_PCD / 2 * math.sin(a)
        h = G.cylinder(f"{name}_hole{k}", L.FLANGE_HOLE_D / 2, zf + 0.004, location=(x, y, zf / 2), vertices=24, collection=col)
        h.data.materials.append(materials.get("black_plastic"))
        G.set_parent(h, root)
    for o in (flange, rf):
        G.set_parent(o, root)

    # --- elbow: sweep from (0,0,SEAM_A_Z) heading +Z bending toward +X
    def radii(t):
        return [r_o, r_i]
    elbow = G.torus_sweep(f"{name}_elbow", L.ELBOW_R, math.pi / 2, radii, seg_bend=40, seg_tube=96, collection=col)
    elbow.location = (0, 0, L.SEAM_A_Z)
    elbow.data.materials.append(steel)
    G.set_parent(elbow, root)
    # bevel rings at both elbow ends (small revolved chamfer pieces, bright metal)
    for end in ("A", "B"):
        pr = [(r_i, 0.0), (r_i + 0.0016, 0.0), (r_o, bev), (r_i, bev)]
        if end == "A":
            ob = G.revolve(f"{name}_elbow_bevel_A", pr, segments=96, collection=col)
            ob.location = (0, 0, L.SEAM_A_Z)
        else:
            pr = [(r_i, 0.0), (r_i, -bev), (r_o, -bev), (r_i + 0.0016, 0.0)]
            ob = G.revolve(f"{name}_elbow_bevel_B", pr, segments=96, axis='X', collection=col)
            ob.location = L.SEAM_B_CENTER
        ob.data.materials.append(bevel)
        G.set_parent(ob, root)

    # --- straight pipe along +X from SEAM_B_CENTER
    cx, cy, cz = L.SEAM_B_CENTER
    prof_p = [(r_i, 0.0), (r_i + 0.0016, 0.0), (r_o, bev), (r_o, L.PIPE_LEN - 0.004), (r_o - 0.004, L.PIPE_LEN), (r_i, L.PIPE_LEN)]
    pipe = G.revolve(f"{name}_pipe", prof_p, segments=128, axis='X', collection=col)
    pipe.location = (cx, cy, cz)
    pipe.data.materials.append(steel)
    pipe.data.materials.append(bevel)
    _assign_by_profile(pipe, len(prof_p), {0, 1}, 1)
    G.set_parent(pipe, root)

    # --- weld beads (rings) with shader-driven progress. Bead object local Z = seam axis.
    beads = {}
    beads["A"] = _bead_ring(f"{name}_bead_A", r_o, col)
    beads["A"].location = (0, 0, L.SEAM_A_Z)
    beads["B"] = _bead_ring(f"{name}_bead_B", r_o, col)
    beads["B"].location = L.SEAM_B_CENTER
    beads["B"].rotation_euler = (0, math.pi / 2, 0)      # local Z -> spool +X
    for b in beads.values():
        G.set_parent(b, root)
    # heat-tint ring under each bead (subtle straw/blue oxide band on the base metal)
    for key, b in beads.items():
        ht = _heat_tint_ring(f"{name}_tint_{key}", r_o, col)
        ht.matrix_world = b.matrix_world.copy()
        G.set_parent(ht, root)

    # thick chalk-style marking on the pipe (like the shop photos): skip geometry, just a decal-free look
    return dict(root=root, collection=col, flange=flange, elbow=elbow, pipe=pipe, beads=beads)


def _assign_by_profile(ob, n_prof, seg_indices, mat_index):
    """Revolve faces are emitted per (segment, profile-index k); set material for given k."""
    me = ob.data
    # faces are created in order: for i in segments: for k in range(n_prof) (closed) -> index = i*n + k, minus skipped
    # geom.revolve skips degenerate quads only when r==0 which we do not use here.
    for p in me.polygons:
        k = p.index % n_prof
        if k in seg_indices:
            p.material_index = mat_index


def _bead_ring(name, r_o, col):
    w = L.BEAD_WIDTH
    h = L.BEAD_HEIGHT
    prof = []
    n = 10
    for i in range(n + 1):
        t = i / n
        z = -w / 2 + w * t
        r = r_o - 0.0012 + h * math.sin(math.pi * t) ** 0.8 * 1.0
        prof.append((r, z))
    prof.append((r_o - 0.004, w / 2))
    prof.append((r_o - 0.004, -w / 2))
    ob = G.revolve(name, prof, segments=160, collection=col)
    ob.data.materials.append(materials.get("weld_bead"))
    materials.init_bead_props(ob)
    return ob


def _heat_tint_ring(name, r_o, col):
    ob = G.revolve(name, [(r_o + 0.0004, -0.032), (r_o + 0.0004, 0.032), (r_o + 0.0003, 0.032), (r_o + 0.0003, -0.032)], segments=128, collection=col)
    m = bpy.data.materials.get("heat_tint")
    if m is None:
        m = bpy.data.materials.new("heat_tint")
        m.use_nodes = True
        nt = m.node_tree
        b = nt.nodes["Principled BSDF"]
        b.inputs["Metallic"].default_value = 0.8
        b.inputs["Roughness"].default_value = 0.45
        tc = nt.nodes.new("ShaderNodeTexCoord")
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        nt.links.new(tc.outputs["Object"], sep.inputs[0])
        ab = nt.nodes.new("ShaderNodeMath"); ab.operation = 'ABSOLUTE'
        nt.links.new(sep.outputs["Z"], ab.inputs[0])
        mr = nt.nodes.new("ShaderNodeMapRange"); mr.inputs["From Min"].default_value = 0.009; mr.inputs["From Max"].default_value = 0.032
        nt.links.new(ab.outputs[0], mr.inputs["Value"])
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        cr = ramp.color_ramp
        cr.elements[0].position = 0.0; cr.elements[0].color = (0.12, 0.11, 0.1, 1)
        e = cr.elements.new(0.3); e.color = (0.14, 0.16, 0.3, 1)         # blue oxide
        e = cr.elements.new(0.6); e.color = (0.3, 0.2, 0.09, 1)          # straw
        cr.elements[-1].position = 1.0; cr.elements[-1].color = (0.09, 0.075, 0.065, 1)
        nt.links.new(mr.outputs["Result"], ramp.inputs["Fac"])
        nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
        # alpha fades to the base metal at the edge; driven by "welded" — simply keep visible always but subtle
        alpha = nt.nodes.new("ShaderNodeMapRange"); alpha.inputs["From Min"].default_value = 0.0; alpha.inputs["From Max"].default_value = 1.0
        alpha.inputs["To Min"].default_value = 0.55; alpha.inputs["To Max"].default_value = 0.0
        nt.links.new(mr.outputs["Result"], alpha.inputs["Value"])
        # visibility gated by object prop "tint_on" (0..1)
        att = nt.nodes.new("ShaderNodeAttribute"); att.attribute_type = 'OBJECT'; att.attribute_name = "tint_on"
        mul = nt.nodes.new("ShaderNodeMath"); mul.operation = 'MULTIPLY'
        nt.links.new(alpha.outputs["Result"], mul.inputs[0]); nt.links.new(att.outputs["Fac"], mul.inputs[1])
        nt.links.new(mul.outputs[0], b.inputs["Alpha"])
        m.surface_render_method = 'DITHERED'
    ob.data.materials.append(m)
    ob["tint_on"] = 0.0
    return ob


def seam_circle_local(which):
    """Return (centre, axis) of a seam in spool-local coordinates."""
    if which == "A":
        return np.array([0.0, 0.0, L.SEAM_A_Z]), np.array([0.0, 0.0, 1.0])
    return np.array(L.SEAM_B_CENTER), np.array([1.0, 0.0, 0.0])


def export_stl(spool, path):
    """Export the spool body (without beads) as one STL in the spool-local frame (mm) for RoboDK."""
    objs = [spool["flange"], spool["elbow"], spool["pipe"]]
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.wm.stl_export(filepath=path, export_selected_objects=True, global_scale=1000.0, use_scene_unit=False, apply_modifiers=True)
