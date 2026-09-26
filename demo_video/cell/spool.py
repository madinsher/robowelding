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
    # heat-tint ring under each bead (straw/blue oxide band with dark HAZ on the base metal); its material reads the
    # same w{i}_* progress props as the bead, so the band appears behind the arc — drivers copy them from the bead.
    tints = {}
    for key, b in beads.items():
        ht = _heat_tint_ring(f"{name}_tint_{key}", r_o, col)
        ht.matrix_world = b.matrix_world.copy()
        G.set_parent(ht, root)
        materials.init_bead_props(ht, source=b)      # packed w{i} arrays driven from the bead's scalar props
        tints[key] = ht

    # paint-marker job markings like the shop photos: one on the pipe leg (+Z side), one on the elbow back (extrados)
    marks = _markings(name, r_o, col, root)
    return dict(root=root, collection=col, flange=flange, elbow=elbow, pipe=pipe, beads=beads, tints=tints, marks=marks)


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
    # the cap must cover the whole groove mouth (2 x bevel length) or a bright strip of chamfer shows beside the bead
    r_i = r_o - L.PIPE_WALL
    bev = (r_o - r_i - 0.0016) / math.tan(math.radians(32))
    w = max(L.BEAD_WIDTH, 2 * bev + 0.004)
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


TINT_HALF_WIDTH = 0.045


def _heat_tint_ring(name, r_o, col):
    hw = TINT_HALF_WIDTH
    ob = G.revolve(name, [(r_o + 0.0004, -hw), (r_o + 0.0004, hw), (r_o + 0.0003, hw), (r_o + 0.0003, -hw)], segments=128, collection=col)
    ob.data.materials.append(materials.get("heat_tint", half_width=hw, inner=0.015))
    ob["tint_on"] = 0.0
    return ob


# ------------------------------------------------------------------ shop markings
_FONTS = [
    "/usr/share/fonts/truetype/freefont/FreeSansBoldOblique.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def _marker_image(name, lines, size=(1024, 512), seed=3, grids=()):
    """RGBA bpy image with hand-written paint-marker text.  lines = [(text, (r,g,b), height_px, x, y, angle_deg)],
    grids = [(x, y, w, h, nx, ny, (r,g,b), stroke_px)] hand-drawn tally/QC grids."""
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    rng = np.random.default_rng(seed)
    W, H = size
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for text, rgb, hpx, x0, y0, ang in lines:
        font = None
        for f in _FONTS:
            try:
                font = ImageFont.truetype(f, int(hpx)); break
            except OSError:
                continue
        if font is None:
            font = ImageFont.load_default(size=int(hpx))
        x = x0
        top = font.getbbox("Hg")[1]                  # common baseline for every glyph (a hyphen must not float up)
        for ch in text:
            if ch == " ":
                x += hpx * 0.35; continue
            # each character on its own tile, jittered and rotated a little: a hand, not a font
            bb = font.getbbox(ch)
            cw, chh = bb[2] - bb[0] + int(hpx * 0.5), int(hpx * 1.6)
            tile = Image.new("RGBA", (cw, chh), (0, 0, 0, 0))
            d = ImageDraw.Draw(tile)
            ox, oy = int(hpx * 0.25) - bb[0], int(hpx * 0.3) - top
            # a marker stroke: draw the glyph a few times with sub-pixel offsets, opaque core, slightly uneven paint
            for dx, dy in ((0, 0), (1.5, 0.5), (-1, 1), (0.5, -1.5)):
                d.text((ox + dx, oy + dy), ch, font=font, fill=(*rgb, 255), stroke_width=max(1, int(hpx * 0.02)), stroke_fill=(*rgb, 255))
            rot = ang + rng.uniform(-6, 6)
            tile = tile.rotate(rot, resample=Image.BICUBIC, expand=False)
            px = int(x - hpx * 0.25 + rng.uniform(-1, 1) * hpx * 0.04)
            py = int(y0 - hpx * 0.3 + rng.uniform(-1, 1) * hpx * 0.06 - (x - x0) * math.tan(math.radians(ang)))
            img.alpha_composite(tile, (max(0, px), max(0, py)))
            x += (bb[2] - bb[0]) + hpx * 0.1 + rng.uniform(-1, 1) * hpx * 0.03
    d = ImageDraw.Draw(img)
    for gx, gy, gw, gh, gnx, gny, rgb, sw in grids:
        def jl(x0, y0, x1, y1):
            j = lambda: rng.uniform(-1, 1) * sw * 1.5
            d.line((x0 + j(), y0 + j(), x1 + j(), y1 + j()), fill=(*rgb, 255), width=int(sw))
        for k in range(gnx + 1):
            x = gx + gw * k / gnx; jl(x, gy, x, gy + gh)
        for k in range(gny + 1):
            y = gy + gh * k / gny; jl(gx, y, gx + gw, y)
        # inspector's tick in the first cell
        cw, chh = gw / gnx, gh / gny
        jl(gx + cw * 0.2, gy + chh * 0.55, gx + cw * 0.45, gy + chh * 0.85)
        jl(gx + cw * 0.45, gy + chh * 0.85, gx + cw * 0.9, gy + chh * 0.15)
    # paint sits on rough mill scale: break the coverage with fine noise and soften the edges
    a = np.asarray(img, dtype=np.float32) / 255.0
    n = rng.random((H // 4, W // 4)).astype(np.float32)
    n = np.asarray(Image.fromarray((n * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR), dtype=np.float32) / 255.0
    a[..., 3] *= np.clip(0.55 + 0.75 * n, 0.0, 1.0)
    img = Image.fromarray((a * 255).astype(np.uint8), "RGBA").filter(ImageFilter.GaussianBlur(0.7))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr[::-1]                      # bpy pixel rows run bottom-up
    bi = bpy.data.images.get(name)
    if bi is None:
        bi = bpy.data.images.new(name, W, H, alpha=True)
    bi.colorspace_settings.name = 'sRGB'
    bi.alpha_mode = 'STRAIGHT'
    bi.pixels.foreach_set(arr.ravel())
    bi.pack()
    return bi


def _patch_object(name, pts, uvs, nu, nv, col, mat, outward):
    """Quad grid mesh from an (nu+1)*(nv+1) point list with per-vertex UVs; outward(p) gives the surface normal side."""
    faces = []
    for i in range(nu):
        for j in range(nv):
            a = i * (nv + 1) + j
            faces.append((a, a + 1, a + nv + 2, a + nv + 1))
    ob = G.new_object(name, pts, faces, col, smooth=True)
    me = ob.data
    uvl = me.uv_layers.new(name="UVMap")
    for poly in me.polygons:
        for li in poly.loop_indices:
            uvl.data[li].uv = uvs[me.loops[li].vertex_index]
    me.materials.append(mat)
    p0 = me.polygons[0]
    if p0.normal.dot(outward(p0.center)) < 0:
        me.flip_normals()
    return ob


def _markings(name, r_o, col, root):
    r = r_o + 0.0005
    yellow = (240, 205, 70)
    white = (235, 235, 225)
    # --- pipe leg: job number + FINAL, patch along X centred at ELBOW_R + 0.35, +Z side of the pipe
    img_p = _marker_image(f"{name}_mark_pipe", [
        ("JC-53  SPL-02", yellow, 128, 40, 150, 3.0),
        ("FINAL", white, 160, 420, 345, -2.0),
    ], size=(1280, 512), seed=7)
    cx, cy, cz = L.SEAM_B_CENTER
    xc = L.ELBOW_R + 0.35
    wx, span = 0.26, math.radians(56)     # 260 mm x ~130 mm
    nu, nv = 16, 12
    pts, uvs = [], []
    for i in range(nu + 1):
        x = xc + wx / 2 - wx * i / nu              # u runs toward -X (reads left-to-right seen from the +Z side, top = -Y)
        for j in range(nv + 1):
            a = -span / 2 + span * j / nv
            pts.append((x, cy + r * math.sin(a), cz + r * math.cos(a)))
            uvs.append((i / nu, 1.0 - j / nv))
    mark_p = _patch_object(f"{name}_mark_pipe", pts, uvs, nu, nv, col, materials.get("marker_decal", image=img_p.name),
                           outward=lambda p: mathutils.Vector((0.0, p.y - cy, p.z - cz)))
    G.set_parent(mark_p, root)
    # --- elbow back (extrados around 45 deg): heat number + a small QC grid with a tick
    img_e = _marker_image(f"{name}_mark_elbow", [
        ("HT 4471", yellow, 108, 40, 118, 2.0),
        ("DN250", white, 92, 640, 128, -3.0),
        ("QC", white, 96, 660, 330, 4.0),
    ], size=(1024, 512), seed=11, grids=[(110, 270, 300, 180, 3, 2, white, 9)])
    R = L.ELBOW_R
    th0, th1 = math.radians(30), math.radians(62)     # ~213 mm of arc length on the extrados
    span_e = math.radians(50)
    pts, uvs = [], []
    for i in range(nu + 1):
        th = th1 - (th1 - th0) * i / nu              # u runs from the pipe end toward the flange
        ccx, ccz = R * (1 - math.cos(th)), R * math.sin(th)
        nx, nz = -math.cos(th), math.sin(th)
        for j in range(nv + 1):
            a = -span_e / 2 + span_e * j / nv
            pts.append((ccx + r * math.cos(a) * nx, r * math.sin(a), L.SEAM_A_Z + ccz + r * math.cos(a) * nz))
            uvs.append((i / nu, 1.0 - j / nv))
    def outward_e(p):
        th = math.atan2(p.z - L.SEAM_A_Z, R - p.x)
        return p - mathutils.Vector((R * (1 - math.cos(th)), 0.0, L.SEAM_A_Z + R * math.sin(th)))
    mark_e = _patch_object(f"{name}_mark_elbow", pts, uvs, nu, nv, col, materials.get("marker_decal", image=img_e.name), outward=outward_e)
    G.set_parent(mark_e, root)
    return dict(pipe=mark_p, elbow=mark_e)


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
