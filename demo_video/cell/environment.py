"""Static surroundings of the welding cell: hall, safety fence, peripheral equipment, dressing, lighting.

Everything here is static geometry built from the footprints in `cell.layout`.  Nothing is
placed inside the keep-clear box x[-1.3,1.9] y[-2.4,2.4] z[0,2.5] (except the fume hood above
z = 2.5) nor inside the track corridor x[1.5,2.6] y[-2.4,2.4]; the positioner, robot, track and
spool are built by other modules.

Public API
    build(collection=None) -> dict(collection=..., objects=[...])
    build_lighting(scene)  -> dict(world=..., lights=[...])
"""
import math
import bpy
import bmesh
import mathutils
from . import layout as L
from . import geom as G
from . import materials

DEG = L.DEG
HALL_X, HALL_Y, HALL_H = L.HALL_SIZE
HX, HY = HALL_X / 2.0, HALL_Y / 2.0
ROOF_Z = HALL_H                    # underside of the roof deck
TRUSS_BOT_Z, TRUSS_TOP_Z = 8.0, 8.75
CRANE_RAIL_Z = 6.7                 # top of the crane runway rail
LAMP_Z = 7.5                       # high-bay lamp discs / area lights
LAMP_GRID = [(x, y) for y in (-7.0, 7.0) for x in (-12.0, -4.0, 4.0, 12.0)]
KEY_LIGHT_POS = (0.8, 0.0, 6.2)


# ============================================================================ builder / mesh helpers
class _B:
    """Collects the objects of one build() call."""

    def __init__(self, col):
        self.col = col
        self.objs = []

    def reg(self, ob, mat=None):
        if mat is not None:
            ob.data.materials.append(mat)
        self.objs.append(ob)
        return ob


def _recalc(ob):
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(ob.data)
    bm.free()


def _mesh(b, name, verts, faces, mat, location=(0, 0, 0), rotation=(0, 0, 0), smooth=False):
    ob = G.new_object(name, verts, faces, b.col, smooth=smooth)
    ob.location = location
    ob.rotation_euler = rotation
    return b.reg(ob, mat)


def _box(b, name, size, center, mat, rotation=(0, 0, 0)):
    """Axis-aligned box (in its local frame) of `size`, centred at `center`, then rotated."""
    x, y, z = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
    v = [(-x, -y, -z), (x, -y, -z), (x, y, -z), (-x, y, -z), (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)]
    f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return _mesh(b, name, v, f, mat, center, rotation)


def _prism(b, name, poly, length, mat, center, rotation=(0, 0, 0), axis='Z'):
    """Extrude a CCW 2D polygon [(u, v), ...] along a local axis, centred on the origin.

    axis 'Z': (u, v) -> (x, y);  axis 'X': (u, v) -> (y, z);  axis 'Y': (u, v) -> (x, z).
    """
    n = len(poly)
    h = length / 2.0
    if axis == 'Y':
        poly = poly[::-1]          # mirrored mapping: reverse to keep faces outward
    verts = []
    for w in (-h, h):
        for u, v in poly:
            if axis == 'Z':
                verts.append((u, v, w))
            elif axis == 'X':
                verts.append((w, u, v))
            else:
                verts.append((u, w, v))
    faces = [tuple(range(n))[::-1], tuple(range(n, 2 * n))]
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j, n + i))
    ob = _mesh(b, name, verts, faces, mat, center, rotation)
    _recalc(ob)
    return ob


def _plane(b, name, w, h, mat, location, rotation=(0, 0, 0)):
    """Quad in the local XZ plane, corner at the origin, normal -Y."""
    v = [(0, 0, 0), (w, 0, 0), (w, 0, h), (0, 0, h)]
    return _mesh(b, name, v, [(0, 1, 2, 3)], mat, location, rotation)


def _rev(b, name, profile, mat, center=(0, 0, 0), rotation=(0, 0, 0), segs=24, axis='Z', smooth=True):
    """Solid of revolution; profile [(r, h)...] traversed with the outer surface going +h."""
    ob = G.revolve(name, profile, segments=segs, axis=axis, close=True, collection=b.col, smooth=smooth)
    ob.location = center
    ob.rotation_euler = rotation
    return b.reg(ob, mat)


def _rod(b, name, r, length, mat, center, rotation=(0, 0, 0), segs=16, axis='Z'):
    return _rev(b, name, [(0, -length / 2), (r, -length / 2), (r, length / 2), (0, length / 2)], mat, center, rotation, segs, axis)


def _pipe(b, name, ro, ri, length, mat, center, rotation=(0, 0, 0), segs=32, axis='Z'):
    return _rev(b, name, [(ro, -length / 2), (ro, length / 2), (ri, length / 2), (ri, -length / 2)], mat, center, rotation, segs, axis)


def _dir_rotation(p0, p1):
    d = mathutils.Vector(p1) - mathutils.Vector(p0)
    return d.length, d.to_track_quat('Z', 'Y').to_euler()


def _bar(b, name, p0, p1, thick, mat, radius=None, segs=12):
    """Straight member from p0 to p1: square (thick) box or round (radius) rod."""
    length, rot = _dir_rotation(p0, p1)
    c = tuple((p0[i] + p1[i]) / 2.0 for i in range(3))
    if radius is not None:
        return _rod(b, name, radius, length, mat, c, rot, segs)
    return _box(b, name, (thick, thick, length), c, mat, rot)


def _ibeam_poly(h, w, tf, tw):
    return [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, -h / 2 + tf), (tw / 2, -h / 2 + tf), (tw / 2, h / 2 - tf),
            (w / 2, h / 2 - tf), (w / 2, h / 2), (-w / 2, h / 2), (-w / 2, h / 2 - tf), (-tw / 2, h / 2 - tf),
            (-tw / 2, -h / 2 + tf), (-w / 2, -h / 2 + tf)]


def _u_poly(h, w, t):
    """Open-top U profile (cable tray)."""
    return [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (w / 2 - t, h / 2), (w / 2 - t, -h / 2 + t),
            (-w / 2 + t, -h / 2 + t), (-w / 2 + t, h / 2), (-w / 2, h / 2)]


def _channel_poly(h, w, t):
    return [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, -h / 2 + t), (-w / 2 + t, -h / 2 + t),
            (-w / 2 + t, h / 2 - t), (w / 2, h / 2 - t), (w / 2, h / 2), (-w / 2, h / 2)]


# ============================================================================ local materials
def _nodes(m):
    nt = m.node_tree
    return nt, nt.nodes["Principled BSDF"]


def _sep_object_coords(nt):
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs[0])
    return sep


def _math(nt, op, a, b=None, clamp=False):
    n = nt.nodes.new("ShaderNodeMath")
    n.operation = op
    n.use_clamp = clamp
    for k, v in enumerate((a, b)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            n.inputs[k].default_value = v
        else:
            nt.links.new(v, n.inputs[k])
    return n.outputs[0]


def _line_mask(nt, coord, pitch, width):
    """1 where `coord` is within +-width/2 of a multiple of `pitch`, else 0."""
    f = _math(nt, 'FRACT', _math(nt, 'DIVIDE', coord, pitch))
    d = _math(nt, 'ABSOLUTE', _math(nt, 'SUBTRACT', f, 0.5))
    return _math(nt, 'GREATER_THAN', d, 0.5 - width / pitch / 2.0)


def _tint_base_color(nt, bsdf, mask, color):
    """Insert a Mix on the Base Color input: where mask = 1 use `color`."""
    src = None
    for lk in nt.links:
        if lk.to_socket == bsdf.inputs["Base Color"]:
            src = lk.from_socket
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = 'RGBA'
    if src is not None:
        nt.links.new(src, mix.inputs[6])
    else:
        mix.inputs[6].default_value = bsdf.inputs["Base Color"].default_value[:]
    mix.inputs[7].default_value = (*color, 1)
    nt.links.new(mask, mix.inputs[0])
    nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])


def _world_z(nt):
    """World-space Z of the shading point (so gradients line up across separately placed objects)."""
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geo.outputs["Position"], sep.inputs[0])
    return sep


def _scale_base_color(nt, bsdf, color, ref):
    """The shared materials drive Base Color from a texture chain, so setting the socket default is a no-op:
    multiply whatever feeds it by color/ref instead (ref ~ the chain's mean value)."""
    src = None
    for lk in nt.links:
        if lk.to_socket == bsdf.inputs["Base Color"]:
            src = lk.from_socket
    if src is None:
        bsdf.inputs["Base Color"].default_value = (*color, 1)
        return
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = 'RGBA'
    mix.blend_type = 'MULTIPLY'
    mix.inputs[0].default_value = 1.0
    nt.links.new(src, mix.inputs[6])
    mix.inputs[7].default_value = (color[0] / ref, color[1] / ref, color[2] / ref, 1)
    nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])


def _mat_floor():
    """Dark, dusty shop concrete: 6 m expansion-joint grid, tyre / grinding-dust streaks, lighter dust drifts and a
    darker worn zone around the cell (the reference photos: brownish-grey, no clean white patches)."""
    key = "env_floor"
    m = bpy.data.materials.get(key)
    if m:
        return m
    m = materials.get("concrete").copy()
    m.name = key
    nt, bsdf = _nodes(m)
    _scale_base_color(nt, bsdf, (0.095, 0.088, 0.08), 0.36)     # concrete chain averages ~0.36
    bsdf.inputs["Roughness"].default_value = 0.78
    bsdf.inputs["Specular IOR Level"].default_value = 0.18
    sep = _sep_object_coords(nt)

    def streaks(name, sx, sy, scale, thresh, amount, color):
        n = nt.nodes.new("ShaderNodeTexNoise")
        n.label = name
        n.inputs["Scale"].default_value = scale
        n.inputs["Detail"].default_value = 3.0
        comb = nt.nodes.new("ShaderNodeCombineXYZ")
        nt.links.new(_math(nt, 'MULTIPLY', sep.outputs["X"], sx), comb.inputs["X"])
        nt.links.new(_math(nt, 'MULTIPLY', sep.outputs["Y"], sy), comb.inputs["Y"])
        nt.links.new(comb.outputs[0], n.inputs["Vector"])
        mask = _math(nt, 'MULTIPLY', _math(nt, 'GREATER_THAN', n.outputs["Fac"], thresh), amount)
        _tint_base_color(nt, bsdf, mask, color)

    # tyre / drag streaks along X (forklift aisle direction) and a finer set along Y, dark grinding-dust smears
    streaks("tyre_x", 0.25, 3.0, 1.0, 0.50, 0.55, (0.055, 0.05, 0.045))
    streaks("tyre_y", 3.0, 0.3, 1.3, 0.56, 0.45, (0.06, 0.055, 0.05))
    streaks("grind", 1.0, 1.0, 2.2, 0.60, 0.35, (0.05, 0.048, 0.045))
    # lighter grey dust drifts (large blotches)
    streaks("dust", 1.0, 1.0, 0.35, 0.55, 0.45, (0.15, 0.14, 0.125))
    # worn / dirtier zone around the cell (radius ~6 m, soft edge, noise-broken)
    dx = _math(nt, 'SUBTRACT', sep.outputs["X"], 0.8)
    dist = _math(nt, 'SQRT', _math(nt, 'ADD', _math(nt, 'MULTIPLY', dx, dx), _math(nt, 'MULTIPLY', sep.outputs["Y"], sep.outputs["Y"])))
    worn = _math(nt, 'MULTIPLY', _math(nt, 'SUBTRACT', 1.0, _math(nt, 'DIVIDE', dist, 7.0), clamp=True), 0.5)
    _tint_base_color(nt, bsdf, worn, (0.06, 0.055, 0.05))
    # expansion joints
    joints = _math(nt, 'MAXIMUM', _line_mask(nt, sep.outputs["X"], L.COLUMN_SPACING, 0.014),
                   _line_mask(nt, sep.outputs["Y"], L.COLUMN_SPACING, 0.014))
    _tint_base_color(nt, bsdf, joints, (0.04, 0.037, 0.034))
    return m


def _mat_wall():
    """Corrugated grey sandwich panel: vertical ribs (bump along local X), faint horizontal panel seams and a
    dirt / scuff gradient on the lower panels (world Z below ~1.5 m above the plinth)."""
    key = "env_wall"
    m = bpy.data.materials.get(key)
    if m:
        return m
    m = materials.get("wall_panel").copy()
    m.name = key
    nt, bsdf = _nodes(m)
    for n in nt.nodes:                       # _noise_variation keeps the base colour in an RGB node
        if n.type == 'RGB':
            n.outputs[0].default_value = (0.40, 0.41, 0.42, 1)
    bsdf.inputs["Base Color"].default_value = (0.40, 0.41, 0.42, 1)
    bsdf.inputs["Roughness"].default_value = 0.58
    sep = _sep_object_coords(nt)
    tri = _math(nt, 'ABSOLUTE', _math(nt, 'SUBTRACT', _math(nt, 'FRACT', _math(nt, 'DIVIDE', sep.outputs["X"], 0.25)), 0.5))
    trap = _math(nt, 'MULTIPLY', _math(nt, 'SUBTRACT', tri, 0.15), 5.0, clamp=True)      # trapezoid rib profile
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.35
    bump.inputs["Distance"].default_value = 0.03
    nt.links.new(trap, bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    wz = _world_z(nt)
    seams = _line_mask(nt, wz.outputs["Z"], 1.0, 0.012)
    _tint_base_color(nt, bsdf, seams, (0.38, 0.40, 0.41))
    # dirt gradient: full below 1.5 m above the plinth, fading out over the next 1.3 m, broken up by noise
    grad = _math(nt, 'SUBTRACT', 1.0, _math(nt, 'DIVIDE', _math(nt, 'SUBTRACT', wz.outputs["Z"], 1.5), 1.3), clamp=True)
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 0.6
    noise.inputs["Detail"].default_value = 5.0
    tc = nt.nodes.new("ShaderNodeTexCoord")
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
    dirt = _math(nt, 'MULTIPLY', grad, _math(nt, 'ADD', 0.35, _math(nt, 'MULTIPLY', noise.outputs["Fac"], 0.5)), clamp=True)
    _tint_base_color(nt, bsdf, dirt, (0.22, 0.21, 0.20))
    return m


def _mat_hazard():
    """45-degree yellow/black hazard stripes (pattern on local x+y)."""
    key = "env_hazard"
    m = bpy.data.materials.get(key)
    if m:
        return m
    m = bpy.data.materials.new(key)
    m.use_nodes = True
    nt, bsdf = _nodes(m)
    bsdf.inputs["Roughness"].default_value = 0.5
    sep = _sep_object_coords(nt)
    s = _math(nt, 'ADD', sep.outputs["X"], sep.outputs["Y"])
    mask = _math(nt, 'GREATER_THAN', _math(nt, 'FRACT', _math(nt, 'DIVIDE', s, 0.28)), 0.5)
    bsdf.inputs["Base Color"].default_value = (0.03, 0.03, 0.03, 1)
    _tint_base_color(nt, bsdf, mask, materials._srgb("#F2B400"))
    return m


def _mat_screen():
    m = bpy.data.materials.get("env_screen")
    if m:
        return m
    m = materials.get("emissive", color="#7FD0FF", strength=1.2).copy()
    m.name = "env_screen"
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.02, 0.03, 0.05, 1)
    b.inputs["Roughness"].default_value = 0.15
    return m


def _mats():
    return dict(
        floor=_mat_floor(), wall=_mat_wall(), hazard=_mat_hazard(), screen=_mat_screen(),
        concrete=materials.get("concrete"),
        red_steel=materials.get("painted", color="#8A1C1C", roughness=0.45),
        roof=materials.get("painted", color="#4A4E52", roughness=0.6),
        roof_steel=materials.get("painted", color="#6C7074", roughness=0.5),
        yellow=materials.get("safety_yellow"), red=materials.get("safety_red"),
        dark=materials.get("dark_metal"), black=materials.get("black_plastic"), rubber=materials.get("rubber"),
        galv=materials.get("galvanized"), steel=materials.get("machined_steel"), pipe=materials.get("steel_pipe"),
        brass=materials.get("brass"),
        cable=materials.get("cable_black"), glass=materials.get("glass_dark"), fence=materials.get("fence_mesh", color="#2A2A2A", pitch=0.06, wire=0.003),   # dark mesh, yellow frames (X-Guard style)
        wood=materials.get("painted", color="#8A6A42", roughness=0.8, coat=0.0),
        cabinet=materials.get("painted", color="#B9BCBE", roughness=0.5),
        cabinet_dark=materials.get("painted", color="#5A5D60", roughness=0.5),
        welder_red=materials.get("painted", color="#B01818", roughness=0.35, coat=0.25),
        machine_blue=materials.get("painted", color="#1F4E8C", roughness=0.42),
        machine_green=materials.get("painted", color="#2E6B3A", roughness=0.45),
        drum_grey=materials.get("painted", color="#9A9C9E", roughness=0.5),
        green_btn=materials.get("painted", color="#2ECC40", roughness=0.3, coat=0.4),
        white_btn=materials.get("painted", color="#E8E8E8", roughness=0.3, coat=0.4),
        lamp=materials.get("emissive", color="#FFE9C8", strength=30.0),
        skylight=materials.get("emissive", color="#DDE9FF", strength=3.0),
        window=materials.get("emissive", color="#E6EEFF", strength=2.5),
        red_led=materials.get("emissive", color="#FF2010", strength=12.0),
        green_led=materials.get("emissive", color="#30FF40", strength=8.0),
        curtain=materials.get("painted", color="#7A1C1C", roughness=0.7, coat=0.0),
    )


# ============================================================================ hall
def _hall(b, M):
    # floor + roof deck
    _mesh(b, "env_floor", [(-HX, -HY, 0), (HX, -HY, 0), (HX, HY, 0), (-HX, HY, 0)], [(0, 1, 2, 3)], M["floor"])
    _mesh(b, "env_roof_deck", [(-HX, -HY, ROOF_Z), (-HX, HY, ROOF_Z), (HX, HY, ROOF_Z), (HX, -HY, ROOF_Z)], [(0, 1, 2, 3)], M["roof"])
    # walls: quad in local XZ, normal -Y, rotated to face inward
    plinth = 1.2
    walls = [("env_wall_n", HALL_X, (-HX, HY, plinth), 0.0),
             ("env_wall_s", HALL_X, (HX, -HY, plinth), math.pi),
             ("env_wall_e", HALL_Y, (HX, HY, plinth), -math.pi / 2),
             ("env_wall_w", HALL_Y, (-HX, -HY, plinth), math.pi / 2)]
    for name, length, loc, rz in walls:
        _plane(b, name, length, HALL_H - plinth, M["wall"], loc, (0, 0, rz))
    # vertical panel ribs (geometry, so the corrugation still reads from far away)
    rib_h = HALL_H - plinth
    k = 0
    for x in [i * 0.5 for i in range(int(-HX / 0.5) + 1, int(HX / 0.5))]:
        for ys in (-1, 1):
            _box(b, f"env_rib_{k}", (0.07, 0.03, rib_h), (x, ys * (HY - 0.015), plinth + rib_h / 2), M["wall"]); k += 1
    for y in [i * 0.5 for i in range(int(-HY / 0.5) + 1, int(HY / 0.5))]:
        for xs in (-1, 1):
            _box(b, f"env_rib_{k}", (0.03, 0.07, rib_h), (xs * (HX - 0.015), y, plinth + rib_h / 2), M["wall"]); k += 1
    # concrete plinth ring + flashing strip on top
    for name, size, c in (("env_plinth_n", (HALL_X, 0.25, plinth), (0, HY - 0.125, plinth / 2)),
                          ("env_plinth_s", (HALL_X, 0.25, plinth), (0, -HY + 0.125, plinth / 2)),
                          ("env_plinth_e", (0.25, HALL_Y, plinth), (HX - 0.125, 0, plinth / 2)),
                          ("env_plinth_w", (0.25, HALL_Y, plinth), (-HX + 0.125, 0, plinth / 2))):
        _box(b, name, size, c, M["concrete"])
        fl = (size[0] + 0.02, 0.28, 0.03) if size[0] > size[1] else (0.28, size[1] + 0.02, 0.03)
        _box(b, name + "_cap", fl, (c[0], c[1], plinth + 0.015), M["roof_steel"])

    # steel columns (HEB 400 style) along both long walls and the end walls
    col_poly = _ibeam_poly(0.40, 0.30, 0.025, 0.015)
    col_h = ROOF_Z - 0.3
    k = 0
    for x in range(int(-HX), int(HX) + 1, int(L.COLUMN_SPACING)):
        for ys in (-1, 1):
            y = ys * (HY - 0.45)
            c = _prism(b, f"env_column_{k}", col_poly, col_h, M["red_steel"], (x, y, col_h / 2), (0, 0, math.pi / 2))
            _box(b, f"env_colbase_{k}", (0.5, 0.5, 0.04), (x, y, 0.02), M["dark"])
            _box(b, f"env_colbracket_{k}", (0.32, 0.55, 0.45), (x, y - ys * 0.45, CRANE_RAIL_Z - 0.85), M["red_steel"])
            k += 1
    for y in range(int(-HY) + 6, int(HY) - 5, int(L.COLUMN_SPACING)):
        for xs in (-1, 1):
            x = xs * (HX - 0.45)
            _prism(b, f"env_column_{k}", col_poly, col_h, M["red_steel"], (x, y, col_h / 2))
            _box(b, f"env_colbase_{k}", (0.5, 0.5, 0.04), (x, y, 0.02), M["dark"])
            k += 1

    # roof trusses spanning Y on each column line, purlins along X
    span = HALL_Y - 1.0
    for i, x in enumerate(range(int(-HX), int(HX) + 1, int(L.COLUMN_SPACING))):
        _prism(b, f"env_truss_top_{i}", _channel_poly(0.16, 0.10, 0.012), span, M["roof_steel"], (x, 0, TRUSS_TOP_Z), axis='Y')
        _prism(b, f"env_truss_bot_{i}", _channel_poly(0.14, 0.10, 0.012), span, M["roof_steel"], (x, 0, TRUSS_BOT_Z), axis='Y')
        npan = 20
        for j in range(npan + 1):
            y = -span / 2 + span * j / npan
            _box(b, f"env_truss_v_{i}_{j}", (0.06, 0.06, TRUSS_TOP_Z - TRUSS_BOT_Z), (x, y, (TRUSS_TOP_Z + TRUSS_BOT_Z) / 2), M["roof_steel"])
            if j < npan:
                y2 = -span / 2 + span * (j + 1) / npan
                p0 = (x, y, TRUSS_BOT_Z) if j % 2 == 0 else (x, y, TRUSS_TOP_Z)
                p1 = (x, y2, TRUSS_TOP_Z) if j % 2 == 0 else (x, y2, TRUSS_BOT_Z)
                _bar(b, f"env_truss_d_{i}_{j}", p0, p1, 0.05, M["roof_steel"])
    for i, y in enumerate(range(-13, 14, 3)):
        _prism(b, f"env_purlin_{i}", _channel_poly(0.18, 0.07, 0.008), HALL_X - 0.6, M["roof_steel"], (0, y, TRUSS_TOP_Z + 0.17), axis='X')

    # skylight strips + high-level window bands (emissive so they read as daylight)
    for i, y in enumerate((-7.5, 7.5)):
        _box(b, f"env_skylight_{i}", (HALL_X - 4.0, 1.8, 0.02), (0, y, ROOF_Z - 0.02), M["skylight"])
        for j, x in enumerate(range(int(-HX) + 2, int(HX) - 1, 2)):
            _box(b, f"env_skyframe_{i}_{j}", (0.05, 1.8, 0.05), (x, y, ROOF_Z - 0.03), M["roof_steel"])
    nb = int(HALL_X // L.COLUMN_SPACING)
    for i in range(nb):
        x0 = -HX + i * L.COLUMN_SPACING + 0.6
        x1 = x0 + L.COLUMN_SPACING - 1.2
        for ys in (-1, 1):
            y = ys * (HY - 0.03)
            _box(b, f"env_window_{i}_{ys}", (x1 - x0, 0.02, 1.6), ((x0 + x1) / 2, y, 6.0), M["window"])
            for j in range(5):
                _box(b, f"env_mullion_{i}_{ys}_{j}", (0.06, 0.06, 1.62), (x0 + (x1 - x0) * j / 4, y - ys * 0.02, 6.0), M["roof_steel"])
            _box(b, f"env_sill_{i}_{ys}", (x1 - x0 + 0.1, 0.08, 0.06), ((x0 + x1) / 2, y - ys * 0.03, 5.17), M["roof_steel"])
            _box(b, f"env_head_{i}_{ys}", (x1 - x0 + 0.1, 0.08, 0.06), ((x0 + x1) / 2, y - ys * 0.03, 6.83), M["roof_steel"])
    # big roller shutter + personnel door on the east wall, another shutter on the west wall
    for i, (x, y, rz) in enumerate(((HX - 0.04, -8.0, 0), (-HX + 0.04, 8.0, 0))):
        _box(b, f"env_shutter_{i}", (0.06, 5.0, 5.0), (x, y, 2.5 + 0.02), M["roof_steel"])
        for j in range(12):
            _box(b, f"env_shutter_rib_{i}_{j}", (0.02, 4.9, 0.03), (x - (0.04 if x > 0 else -0.04), y, 0.25 + 0.4 * j), M["cabinet_dark"])
        _box(b, f"env_shutter_frame_{i}", (0.12, 5.3, 0.15), (x, y, 5.1), M["yellow"])
    _box(b, "env_pdoor", (0.05, 1.0, 2.1), (HX - 0.05, -3.5, 1.05), M["machine_blue"])
    _box(b, "env_pdoor_sign", (0.02, 0.35, 0.18), (HX - 0.1, -3.5, 2.35), M["green_led"])


def _crane(b, M):
    """Runway I-beams along X on the column brackets + one bridge spanning Y with a hoist trolley."""
    beam_poly = _ibeam_poly(0.55, 0.30, 0.022, 0.013)
    for i, ys in enumerate((-1, 1)):
        y = ys * (HY - 0.45) - ys * 0.45
        _prism(b, f"env_runway_{i}", beam_poly, HALL_X - 0.6, M["yellow"], (0, y, CRANE_RAIL_Z - 0.375), axis='X')
        _box(b, f"env_runway_rail_{i}", (HALL_X - 0.6, 0.06, 0.10), (0, y, CRANE_RAIL_Z - 0.05), M["dark"])
    xb = 7.5
    yr = HY - 0.9
    span = 2 * yr
    for i, dx in enumerate((-0.6, 0.6)):
        _box(b, f"env_bridge_girder_{i}", (0.45, span - 1.2, 0.9), (xb + dx, 0, CRANE_RAIL_Z + 0.75), M["yellow"])
    for i, ys in enumerate((-1, 1)):
        _box(b, f"env_bridge_endcar_{i}", (2.4, 0.5, 0.42), (xb, ys * (yr - 0.05), CRANE_RAIL_Z + 0.21), M["yellow"])
        for j, dx in enumerate((-0.9, 0.9)):
            _rod(b, f"env_bridge_wheel_{i}_{j}", 0.18, 0.12, M["dark"], (xb + dx, ys * yr, CRANE_RAIL_Z + 0.16), (math.pi / 2, 0, 0), 20)
    # walkway railing along one girder
    for j, y in enumerate(range(int(-yr) + 2, int(yr) - 1, 2)):
        _box(b, f"env_bridge_post_{j}", (0.04, 0.04, 1.0), (xb - 1.0, y, CRANE_RAIL_Z + 1.7), M["yellow"])
    for j, z in enumerate((CRANE_RAIL_Z + 1.7, CRANE_RAIL_Z + 2.2)):
        _box(b, f"env_bridge_rail_{j}", (0.04, span - 1.6, 0.04), (xb - 1.0, 0, z), M["yellow"])
    # hoist trolley + hook block on ropes
    yt = -4.5
    _box(b, "env_trolley", (1.9, 1.3, 0.55), (xb, yt, CRANE_RAIL_Z + 1.45), M["yellow"])
    _rod(b, "env_trolley_drum", 0.18, 0.9, M["dark"], (xb, yt, CRANE_RAIL_Z + 1.9), (0, math.pi / 2, 0), 20)
    _box(b, "env_trolley_motor", (0.35, 0.3, 0.3), (xb + 0.75, yt, CRANE_RAIL_Z + 1.9), M["cabinet_dark"])
    z_hook = 5.4
    for j, dx in enumerate((-0.12, 0.12)):
        _rod(b, f"env_hoist_rope_{j}", 0.008, CRANE_RAIL_Z + 1.2 - z_hook, M["dark"], (xb + dx, yt, (CRANE_RAIL_Z + 1.2 + z_hook) / 2), segs=8)
    _box(b, "env_hook_block", (0.45, 0.22, 0.55), (xb, yt, z_hook), M["yellow"])
    _rod(b, "env_hook_shank", 0.035, 0.3, M["steel"], (xb, yt, z_hook - 0.4), segs=12)
    hook = G.torus_sweep("env_hook", 0.13, math.pi, lambda t: [0.035], seg_bend=16, seg_tube=12, collection=b.col)
    hook.location = (xb + 0.13, yt, z_hook - 0.55)
    hook.rotation_euler = (0, math.pi, 0)
    b.reg(hook, M["steel"])


def _lamps(b, M):
    """High-bay fixtures hanging from the roof (the area lights are added in build_lighting)."""
    for i, (x, y) in enumerate(LAMP_GRID):
        _rod(b, f"env_lamp_rod_{i}", 0.015, ROOF_Z - LAMP_Z - 0.45, M["dark"], (x, y, (ROOF_Z + LAMP_Z + 0.45) / 2), segs=8)
        _rod(b, f"env_lamp_body_{i}", 0.13, 0.28, M["cabinet_dark"], (x, y, LAMP_Z + 0.3), segs=20)
        _rev(b, f"env_lamp_reflector_{i}", [(0.12, 0.0), (0.34, -0.22), (0.34, -0.24), (0.115, -0.05)], M["galv"],
             (x, y, LAMP_Z + 0.24), segs=28)
        _rev(b, f"env_lamp_disc_{i}", [(0, 0), (0.33, 0), (0.33, 0.02), (0, 0.02)], M["lamp"], (x, y, LAMP_Z), segs=28, smooth=False)


# ============================================================================ fence
def _fence_bay(b, M, name, p0, p1, z0=0.12, z1=None):
    """Mesh panel with rail frame between two post centres (p0, p1 are (x, y))."""
    z1 = z1 or L.FENCE_H - 0.06
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    w = math.hypot(dx, dy)
    yaw = math.atan2(dy, dx)
    cx, cy = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
    # panel plane in local XY (so fence_mesh's object-space brick pattern is (along, up)), stood upright
    v = [(0, 0, 0), (w - 0.07, 0, 0), (w - 0.07, z1 - z0, 0), (0, z1 - z0, 0)]
    off = mathutils.Vector((0.035, 0, 0))
    loc = mathutils.Matrix.Rotation(yaw, 4, 'Z') @ off
    _mesh(b, name + "_mesh", v, [(0, 1, 2, 3)], M["fence"], (p0[0] + loc.x, p0[1] + loc.y, z0), (math.pi / 2, 0, yaw))
    for k, z in enumerate((z0, z1)):
        _box(b, f"{name}_rail{k}", (w - 0.07, 0.03, 0.03), (cx, cy, z), M["yellow"], (0, 0, yaw))
    for k, (x, y) in enumerate((p0, p1)):
        _box(b, f"{name}_stile{k}", (0.03, 0.03, z1 - z0), (x + (0.05 if k == 0 else -0.05) * math.cos(yaw),
                                                           y + (0.05 if k == 0 else -0.05) * math.sin(yaw), (z0 + z1) / 2), M["yellow"], (0, 0, yaw))


def _post(b, M, name, x, y, h=None):
    h = h or L.FENCE_H
    _box(b, name, (0.06, 0.06, h), (x, y, h / 2), M["yellow"])
    _box(b, name + "_cap", (0.07, 0.07, 0.01), (x, y, h + 0.005), M["black"])
    _box(b, name + "_plate", (0.16, 0.16, 0.012), (x, y, 0.006), M["dark"])


def _fence(b, M):
    x0, x1 = L.FENCE_X
    y0, y1 = L.FENCE_Y
    door_x0 = L.FENCE_DOOR["center_x"] - L.FENCE_DOOR["width"] / 2
    door_x1 = L.FENCE_DOOR["center_x"] + L.FENCE_DOOR["width"] / 2
    open_y0, open_y1 = -L.LOAD_OPENING["width"] / 2, L.LOAD_OPENING["width"] / 2

    def split(a, c, pitch=1.5):
        n = max(1, round((c - a) / pitch))
        return [a + (c - a) * i / n for i in range(n + 1)]

    runs = [  # (name, list of (x, y) post positions along a run, bays are between consecutive posts)
        ("n", [(x, y1) for x in split(x0, x1)]),
        ("e", [(x1, y) for y in split(y0, y1)]),
        ("s1", [(x, y0) for x in split(x0, door_x0)]),
        ("s2", [(x, y0) for x in split(door_x1, x1)]),
        ("w1", [(x0, y) for y in split(y0, open_y0)]),
        ("w2", [(x0, y) for y in split(open_y1, y1)]),
    ]
    seen = set()
    for rn, pts in runs:
        for i, p in enumerate(pts):
            key = (round(p[0], 3), round(p[1], 3))
            if key not in seen:
                seen.add(key)
                _post(b, M, f"env_fpost_{rn}_{i}", *p)
            if i + 1 < len(pts):
                _fence_bay(b, M, f"env_fbay_{rn}_{i}", p, pts[i + 1])

    # sliding door on the -Y side: yellow frame + mesh, hangs on a top rail outside the fence line
    yd = y0 - 0.10
    dw = L.FENCE_DOOR["width"]
    _box(b, "env_door_rail", (dw * 2 + 0.3, 0.06, 0.08), ((door_x0 + door_x1) / 2 - dw / 2, yd, L.FENCE_H + 0.06), M["yellow"])
    _box(b, "env_door_rail_bkt0", (0.06, 0.14, 0.06), (door_x0 - dw - 0.1, y0 - 0.05, L.FENCE_H + 0.03), M["dark"])
    _box(b, "env_door_rail_bkt1", (0.06, 0.14, 0.06), (door_x1 + 0.1, y0 - 0.05, L.FENCE_H + 0.03), M["dark"])
    for k, x in enumerate((door_x0 + 0.03, door_x1 - 0.03)):
        _box(b, f"env_door_stile_{k}", (0.06, 0.06, L.FENCE_H - 0.1), (x, yd, (L.FENCE_H - 0.1) / 2 + 0.05), M["yellow"])
    for k, z in enumerate((0.08, L.FENCE_H - 0.08)):
        _box(b, f"env_door_rail_{k}", (dw, 0.06, 0.06), (L.FENCE_DOOR["center_x"], yd, z), M["yellow"])
    for k, x in enumerate((door_x0 + 0.25, door_x1 - 0.25)):
        _rod(b, f"env_door_roller_{k}", 0.05, 0.03, M["dark"], (x, yd, L.FENCE_H + 0.06), (math.pi / 2, 0, 0), 16)
    v = [(0, 0, 0), (dw - 0.12, 0, 0), (dw - 0.12, L.FENCE_H - 0.22, 0), (0, L.FENCE_H - 0.22, 0)]
    _mesh(b, "env_door_mesh", v, [(0, 1, 2, 3)], M["fence"], (door_x0 + 0.06, yd, 0.11), (math.pi / 2, 0, 0))
    _box(b, "env_door_handle", (0.03, 0.05, 0.3), (door_x1 - 0.12, yd - 0.06, 1.05), M["dark"])
    # interlock switch box + actuator on the post beside the door, danger sign next to it
    _box(b, "env_interlock", (0.09, 0.12, 0.2), (door_x1 + 0.08, y0 - 0.08, 1.2), M["yellow"])
    _box(b, "env_interlock_act", (0.06, 0.04, 0.05), (door_x1 - 0.05, yd - 0.02, 1.2), M["dark"])
    _box(b, "env_interlock_led", (0.02, 0.005, 0.02), (door_x1 + 0.08, y0 - 0.145, 1.27), M["green_led"])
    _box(b, "env_sign_plate", (0.42, 0.012, 0.30), (door_x1 + 0.9, y0 - 0.04, 1.55), M["yellow"])
    _box(b, "env_sign_band", (0.42, 0.006, 0.08), (door_x1 + 0.9, y0 - 0.05, 1.66), M["red"])
    _box(b, "env_sign_symbol", (0.16, 0.006, 0.12), (door_x1 + 0.9, y0 - 0.05, 1.50), M["black"])
    _box(b, "env_sign2_plate", (0.42, 0.012, 0.30), (x1 + 0.04, 0.0, 1.55), M["yellow"], (0, 0, math.pi / 2))
    _box(b, "env_sign2_band", (0.42, 0.006, 0.08), (x1 + 0.05, 0.0, 1.66), M["red"], (0, 0, math.pi / 2))

    # light-curtain posts flanking the load opening (-X side)
    for k, y in enumerate((open_y0, open_y1)):
        _box(b, f"env_lc_post_{k}", (0.08, 0.08, 1.85), (x0, y, 0.925 + 0.03), M["black"])
        _box(b, f"env_lc_base_{k}", (0.2, 0.2, 0.03), (x0, y, 0.015), M["dark"])
        s = 1 if k == 0 else -1
        _box(b, f"env_lc_strip_{k}", (0.012, 0.004, 1.5), (x0, y + s * 0.042, 1.1), M["red_led"])
        _box(b, f"env_lc_led_{k}", (0.02, 0.004, 0.02), (x0, y + s * 0.042, 1.92), M["green_led"])
    # yellow/black hazard stripe on the floor along the outside of the fence line
    g = 0.06
    sw = 0.12
    _box(b, "env_hz_n", (x1 - x0 + 2 * (g + sw), sw, 0.003), ((x0 + x1) / 2, y1 + g + sw / 2, 0.0015), M["hazard"])
    _box(b, "env_hz_s", (x1 - x0 + 2 * (g + sw), sw, 0.003), ((x0 + x1) / 2, y0 - g - sw / 2, 0.0015), M["hazard"])
    _box(b, "env_hz_e", (sw, y1 - y0 + 2 * g, 0.003), (x1 + g + sw / 2, 0, 0.0015), M["hazard"])
    _box(b, "env_hz_w", (sw, y1 - y0 + 2 * g, 0.003), (x0 - g - sw / 2, 0, 0.0015), M["hazard"])


# ============================================================================ equipment
def _controller(b, M):
    (px, py, _), (sx, sy, sz) = L.CONTROLLER_CABINET["pos"], L.CONTROLLER_CABINET["size"]
    zc = sz / 2
    _box(b, "env_irc5_body", (sx, sy, sz), (px, py, zc), M["cabinet"])
    _box(b, "env_irc5_base", (sx - 0.04, sy - 0.04, 0.08), (px, py, 0.04), M["cabinet_dark"])
    yf = py - sy / 2                                  # door face (-Y)
    _box(b, "env_irc5_door", (sx - 0.06, 0.015, sz - 0.2), (px, yf - 0.007, zc + 0.04), M["cabinet"])
    _box(b, "env_irc5_door_gap", (sx - 0.05, 0.004, sz - 0.19), (px, yf - 0.001, zc + 0.04), M["cabinet_dark"])
    for k in range(6):                                # ventilation louvres (lower half of the door)
        _box(b, f"env_irc5_louvre_{k}", (0.3, 0.012, 0.012), (px, yf - 0.02, 0.28 + 0.05 * k), M["cabinet_dark"])
    _box(b, "env_irc5_lock", (0.02, 0.02, 0.08), (px + sx / 2 - 0.08, yf - 0.02, zc + 0.1), M["dark"])
    # operator panel: dark plate, e-stop, mode switch, two lamp-buttons
    _box(b, "env_irc5_panel", (0.26, 0.012, 0.13), (px - 0.12, yf - 0.02, sz - 0.3), M["cabinet_dark"])
    _rod(b, "env_irc5_estop_ring", 0.03, 0.006, M["yellow"], (px - 0.2, yf - 0.03, sz - 0.3), (math.pi / 2, 0, 0), 20)
    _rod(b, "env_irc5_estop", 0.02, 0.025, M["red"], (px - 0.2, yf - 0.04, sz - 0.3), (math.pi / 2, 0, 0), 20)
    _rod(b, "env_irc5_key", 0.014, 0.02, M["dark"], (px - 0.12, yf - 0.035, sz - 0.3), (math.pi / 2, 0, 0), 12)
    _box(b, "env_irc5_keytab", (0.006, 0.02, 0.03), (px - 0.12, yf - 0.05, sz - 0.3), M["steel"])
    _rod(b, "env_irc5_btn_g", 0.011, 0.02, M["green_btn"], (px - 0.05, yf - 0.035, sz - 0.28), (math.pi / 2, 0, 0), 12)
    _rod(b, "env_irc5_btn_w", 0.011, 0.02, M["white_btn"], (px - 0.05, yf - 0.035, sz - 0.33), (math.pi / 2, 0, 0), 12)
    _box(b, "env_irc5_led", (0.02, 0.004, 0.01), (px - 0.02, yf - 0.03, sz - 0.27), M["green_led"])
    # FlexPendant holder + pendant on the -X side, cable to the cabinet
    xf = px - sx / 2
    _box(b, "env_fp_holder", (0.03, 0.22, 0.06), (xf - 0.03, py - 0.1, 1.0), M["cabinet_dark"])
    _box(b, "env_fp_body", (0.05, 0.22, 0.30), (xf - 0.045, py - 0.1, 1.12), M["cabinet"])
    _box(b, "env_fp_screen", (0.006, 0.15, 0.11), (xf - 0.073, py - 0.1, 1.15), M["screen"])
    _rod(b, "env_fp_estop", 0.018, 0.02, M["red"], (xf - 0.075, py - 0.1, 1.24), (0, math.pi / 2, 0), 16)
    _rod(b, "env_fp_joystick", 0.007, 0.03, M["dark"], (xf - 0.08, py - 0.1, 1.03), (0, math.pi / 2, 0), 8)
    _bar(b, "env_fp_cable", (xf - 0.045, py - 0.1, 0.97), (xf - 0.02, py - 0.05, 0.4), 0, M["cable"], radius=0.006, segs=8)
    _bar(b, "env_fp_cable2", (xf - 0.02, py - 0.05, 0.4), (xf + 0.05, py - 0.05, 0.1), 0, M["cable"], radius=0.006, segs=8)
    # roof-mounted heat exchanger box + ABB-style orange stripe
    _box(b, "env_irc5_top", (sx - 0.1, sy - 0.1, 0.06), (px, py, sz + 0.03), M["cabinet_dark"])
    _box(b, "env_irc5_stripe", (sx - 0.05, 0.004, 0.03), (px, yf - 0.014, 0.16), materials.get("abb_orange"))


def _power_source(b, M):
    (px, py, _), (sx, sy, sz) = L.WELD_POWER_SOURCE["pos"], L.WELD_POWER_SOURCE["size"]
    # trolley: base plate on four castors, gas cylinder shelf at the back (+Y)
    _box(b, "env_ps_trolley", (sx + 0.06, sy + 0.06, 0.04), (px, py, 0.12), M["dark"])
    for k, (dx, dy) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1))):
        _rod(b, f"env_ps_castor_{k}", 0.045, 0.03, M["rubber"], (px + dx * (sx / 2 - 0.02), py + dy * (sy / 2 - 0.02), 0.045),
             (0, math.pi / 2, 0), 14)
    zb = 0.14
    body_h = sz - 0.3
    _box(b, "env_ps_body", (sx, 0.42, body_h), (px, py - 0.10, zb + body_h / 2), M["welder_red"])
    _box(b, "env_ps_body_low", (sx + 0.01, 0.43, 0.16), (px, py - 0.10, zb + 0.08), M["black"])
    xf = px - sx / 2                                  # front face (-X, toward the track)
    _box(b, "env_ps_front", (0.012, 0.36, body_h - 0.2), (xf - 0.006, py - 0.10, zb + body_h / 2), M["black"])
    for k in range(4):                                # cable sockets (dinse) + gas connector
        _rod(b, f"env_ps_socket_{k}", 0.02, 0.03, M["brass"] if k < 2 else M["dark"], (xf - 0.02, py - 0.24 + 0.09 * k, zb + 0.1),
             (0, math.pi / 2, 0), 14)
    _box(b, "env_ps_display", (0.006, 0.16, 0.07), (xf - 0.016, py - 0.10, zb + body_h - 0.16), M["screen"])
    _rod(b, "env_ps_knob", 0.02, 0.02, M["black"], (xf - 0.02, py + 0.04, zb + body_h - 0.28), (0, math.pi / 2, 0), 16)
    _box(b, "env_ps_led", (0.006, 0.02, 0.01), (xf - 0.016, py - 0.22, zb + body_h - 0.16), M["green_led"])
    # wire-feeder style top unit
    ft = zb + body_h
    _box(b, "env_ps_feeder", (sx - 0.05, 0.36, 0.22), (px, py - 0.10, ft + 0.11), M["welder_red"])
    _box(b, "env_ps_feeder_lid", (sx - 0.09, 0.30, 0.02), (px, py - 0.10, ft + 0.23), M["black"])
    _box(b, "env_ps_handle", (0.02, 0.28, 0.02), (px, py - 0.10, ft + 0.31), M["black"])
    for k, dy in enumerate((-0.12, 0.12)):
        _box(b, f"env_ps_handle_leg_{k}", (0.02, 0.02, 0.08), (px, py - 0.10 + dy, ft + 0.27), M["black"])
    _rod(b, "env_ps_feeder_conn", 0.018, 0.05, M["brass"], (xf - 0.005, py - 0.10, ft + 0.1), (0, math.pi / 2, 0), 14)
    # argon cylinder strapped at the back
    cyl_r = 0.10
    cy = py + sy / 2 - cyl_r - 0.02
    _rev(b, "env_ps_gas", [(0, 0), (cyl_r, 0), (cyl_r, 1.25), (0.06, 1.38), (0.04, 1.40), (0, 1.40)], M["machine_green"],
         (px, cy, 0.14), segs=28)
    _rod(b, "env_ps_gas_valve", 0.015, 0.1, M["brass"], (px, cy, 1.57), segs=10)
    _box(b, "env_ps_gas_reg", (0.06, 0.05, 0.05), (px + 0.03, cy, 1.60), M["brass"])
    _rod(b, "env_ps_gas_gauge", 0.025, 0.02, M["steel"], (px + 0.03, cy, 1.66), segs=14)
    _box(b, "env_ps_gas_chain", (sx, 0.008, 0.008), (px, cy - cyl_r, 1.0), M["steel"])
    _box(b, "env_ps_gas_bracket", (sx, 0.03, 0.06), (px, py + sy / 2 - 0.015, 1.0), M["dark"])
    _bar(b, "env_ps_gas_hose", (px, cy - 0.05, 1.60), (px, py - 0.10, ft + 0.2), 0, M["cable"], radius=0.006, segs=8)


def _wire_drum(b, M):
    (px, py, _), r, h = L.WIRE_DRUM["pos"], L.WIRE_DRUM["radius"], L.WIRE_DRUM["height"]
    _rev(b, "env_drum", [(0, 0), (r, 0), (r, h * 0.3), (r + 0.012, h * 0.3), (r + 0.012, h * 0.33), (r, h * 0.33),
                         (r, h * 0.66), (r + 0.012, h * 0.66), (r + 0.012, h * 0.69), (r, h * 0.69), (r, h), (0, h)],
         M["drum_grey"], (px, py, 0.0), segs=40)
    _box(b, "env_drum_pallet", (0.7, 0.7, 0.10), (px, py, 0.05), M["wood"])
    _rev(b, "env_drum_hood", [(0, 0), (r + 0.02, 0), (r + 0.02, 0.05), (0.12, 0.2), (0.05, 0.22), (0, 0.22)], M["black"],
         (px, py, h), segs=40)
    # wire conduit: up from the hood, over a low bend and down into the cable tray (no tall hook / lamp post)
    zc = h + 0.22
    conduit = M["cable"]
    rb = 0.22
    zt = zc + 0.45                                    # top of the vertical run (~1.5 m)
    _rod(b, "env_drum_conduit_up", 0.02, zt - zc, conduit, (px, py, (zc + zt) / 2), segs=12)
    bend = G.torus_sweep("env_drum_conduit_bend", rb, math.pi / 2, lambda t: [0.02], seg_bend=12, seg_tube=12, collection=b.col)
    bend.location = (px, py, zt)
    bend.rotation_euler = (0, 0, math.pi)             # bend toward -X
    b.reg(bend, conduit)
    xd = 2.85                                         # descent into the cable tray centre line (see _cable_management)
    run = px - rb - (xd + rb)
    _rod(b, "env_drum_conduit_out", 0.02, run, conduit, (px - rb - run / 2, py, zt + rb), (0, math.pi / 2, 0), 12)
    # second bend traversed backwards: unrotated sweep from (xd, zt) heading +Z ends at (xd + rb, zt + rb) heading +X
    bend2 = G.torus_sweep("env_drum_conduit_bend2", rb, math.pi / 2, lambda t: [0.02], seg_bend=12, seg_tube=12, collection=b.col)
    bend2.location = (xd, py, zt)
    b.reg(bend2, conduit)
    _rod(b, "env_drum_conduit_down", 0.02, zt - 0.08, conduit, (xd, py, (zt + 0.08) / 2), segs=12)
    _box(b, "env_drum_conduit_clamp", (0.05, 0.05, 0.05), (px, py, zc + 0.02), M["dark"])


def _torch_cleaner(b, M):
    """Compact torch-cleaning station: grey reamer box on a short pedestal, red wire-cutter guard, anti-spatter
    spray can in a holder, pneumatic hose down to the cable tray.  Top of the box = TORCH_CLEANER height."""
    (px, py, _), h = L.TORCH_CLEANER["pos"], L.TORCH_CLEANER["height"]
    bx, by, bz = 0.40, 0.30, 0.40                     # reamer box
    zp = h - bz                                       # pedestal top
    grey = M["cabinet"]
    _box(b, "env_tc_base", (0.30, 0.30, 0.02), (px, py, 0.01), M["dark"])
    _box(b, "env_tc_pedestal", (0.16, 0.16, zp - 0.02), (px, py, (zp - 0.02) / 2 + 0.02), M["cabinet_dark"])
    _box(b, "env_tc_box", (bx, by, bz), (px, py, zp + bz / 2), grey)
    _box(b, "env_tc_box_lid", (bx - 0.02, by - 0.02, 0.012), (px, py, h + 0.006), M["cabinet_dark"])
    # reamer opening (dark disc in the lid) with the reamer bit and V-clamp jaws either side of it
    rx, ry = px + 0.08, py
    _rev(b, "env_tc_ream_hole", [(0, 0), (0.045, 0), (0.045, 0.002), (0, 0.002)], M["black"], (rx, ry, h + 0.012), segs=20, smooth=False)
    _rev(b, "env_tc_reamer", [(0, 0), (0.010, 0), (0.010, 0.03), (0.005, 0.045), (0, 0.045)], M["steel"], (rx, ry, h + 0.012), segs=12)
    for k, s_ in enumerate((-1, 1)):
        _box(b, f"env_tc_jaw_{k}", (0.05, 0.03, 0.03), (rx, ry + s_ * 0.06, h + 0.027), M["steel"])
    # wire cutter: red guard block with a horizontal blade slot on the -Y face of the lid
    cx, cy = px - 0.10, py - 0.05
    _box(b, "env_tc_cutter", (0.12, 0.10, 0.06), (cx, cy, h + 0.042), M["red"])
    _box(b, "env_tc_cutter_slot", (0.045, 0.02, 0.006), (cx, cy - 0.05, h + 0.045), M["black"])
    _rod(b, "env_tc_cutter_cyl", 0.02, 0.06, M["cabinet_dark"], (cx, cy + 0.08, h + 0.042), (math.pi / 2, 0, 0), 14)   # pneumatic cylinder
    # anti-spatter spray can in a clip on the +Y side of the box, spray nozzle block next to the reamer
    sx_, sy_ = px + 0.10, py + by / 2 + 0.04
    _box(b, "env_tc_can_clip", (0.09, 0.04, 0.05), (sx_, py + by / 2 + 0.02, zp + 0.28), M["cabinet_dark"])
    _rev(b, "env_tc_can", [(0, 0), (0.033, 0), (0.033, 0.14), (0.028, 0.15), (0.012, 0.15), (0.012, 0.165), (0, 0.165)], M["white_btn"],
         (sx_, sy_, zp + 0.14), segs=20)
    _box(b, "env_tc_can_label", (0.068, 0.068, 0.07), (sx_, sy_, zp + 0.21), M["red"])
    _box(b, "env_tc_spray", (0.04, 0.03, 0.03), (rx - 0.08, ry + 0.03, h + 0.027), M["black"])
    # controls: two push buttons + a status LED on the -X face (facing the robot side), hose to the tray
    xf = px - bx / 2
    _rod(b, "env_tc_btn_g", 0.011, 0.02, M["green_btn"], (xf - 0.005, py - 0.06, zp + 0.30), (0, math.pi / 2, 0), 12)
    _rod(b, "env_tc_btn_r", 0.011, 0.02, M["red"], (xf - 0.005, py - 0.10, zp + 0.30), (0, math.pi / 2, 0), 12)
    _box(b, "env_tc_led", (0.006, 0.02, 0.01), (xf - 0.003, py + 0.06, zp + 0.32), M["green_led"])
    _box(b, "env_tc_gland", (0.03, 0.03, 0.03), (px, py + by / 2 + 0.015, zp + 0.06), M["black"])
    _bar(b, "env_tc_hose", (px, py + by / 2 + 0.03, zp + 0.06), (px + 0.35, py - 0.05, 0.05), 0, M["cable"], radius=0.008, segs=8)
    _bar(b, "env_tc_hose2", (px + 0.35, py - 0.05, 0.05), (2.8, py + 0.1, 0.05), 0, M["cable"], radius=0.008, segs=8)


def _fume_hood(b, M):
    (px, py, pz), (sx, sy, sz) = L.FUME_HOOD["pos"], L.FUME_HOOD["size"]
    # truncated pyramid hood (open underneath), rim, duct, fan unit, hanger rods
    top = 0.45
    v = [(-sx / 2, -sy / 2, 0), (sx / 2, -sy / 2, 0), (sx / 2, sy / 2, 0), (-sx / 2, sy / 2, 0),
         (-top / 2, -top / 2, sz), (top / 2, -top / 2, sz), (top / 2, top / 2, sz), (-top / 2, top / 2, sz)]
    f = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    _mesh(b, "env_hood", v, f, M["galv"], (px, py, pz))
    for k, (dx, dy, w, d) in enumerate(((0, -sy / 2, sx, 0.04), (0, sy / 2, sx, 0.04), (-sx / 2, 0, 0.04, sy), (sx / 2, 0, 0.04, sy))):
        _box(b, f"env_hood_rim_{k}", (w, d, 0.06), (px + dx, py + dy, pz + 0.03), M["galv"])
    _box(b, "env_hood_flange", (0.6, 0.6, 0.04), (px, py, pz + sz + 0.02), M["galv"])
    zf = TRUSS_BOT_Z - 0.55                             # fan unit hung under the trusses
    _rod(b, "env_duct_up", 0.16, zf - (pz + sz), M["galv"], (px, py, (zf + pz + sz) / 2), segs=24)
    for k, z in enumerate((pz + sz + 0.4, pz + sz + 1.8, zf - 0.6)):
        _pipe(b, f"env_duct_ring_{k}", 0.175, 0.16, 0.05, M["galv"], (px, py, z), segs=24)
    _box(b, "env_fan_box", (0.7, 0.7, 0.6), (px, py, zf + 0.3), M["galv"])
    _rod(b, "env_fan_motor", 0.12, 0.3, M["cabinet_dark"], (px - 0.5, py, zf + 0.3), (0, math.pi / 2, 0), 20)
    _box(b, "env_fan_frame", (0.9, 0.9, 0.05), (px, py, zf + 0.62), M["red_steel"])
    for k, (dx, dy) in enumerate(((-0.4, -0.4), (0.4, -0.4), (-0.4, 0.4), (0.4, 0.4))):
        _rod(b, f"env_fan_hanger_{k}", 0.012, TRUSS_BOT_Z - zf - 0.62, M["dark"], (px + dx, py + dy, (TRUSS_BOT_Z + zf + 0.62) / 2), segs=8)
        _rod(b, f"env_hood_hanger_{k}", 0.006, zf - pz - 0.1, M["dark"], (px + dx * 1.6, py + dy * 1.6, (zf + pz) / 2), segs=6)
    # horizontal exhaust duct from the fan to the north wall
    ylen = HY - py - 0.35
    _rod(b, "env_duct_out", 0.16, ylen, M["galv"], (px + 0.5, py + ylen / 2 + 0.35, zf + 0.3), (math.pi / 2, 0, 0), 24)
    bend = G.torus_sweep("env_duct_bend", 0.3, math.pi / 2, lambda t: [0.16], seg_bend=10, seg_tube=24, collection=b.col)
    bend.location = (px + 0.35, py, zf + 0.3)
    bend.rotation_euler = (math.pi / 2, 0, math.pi / 2)
    b.reg(bend, M["galv"])


def _cable_management(b, M):
    """Cable tray along the track (outside the corridor) + cable runs from the cabinets."""
    xt = 2.85
    y0, y1 = L.TRACK_Y_MIN - 0.5, L.TRACK_Y_MAX + 0.5
    ln = y1 - y0
    _prism(b, "env_tray", _u_poly(0.08, 0.25, 0.006), ln, M["galv"], (xt, (y0 + y1) / 2, 0.04), axis='Y')
    # rungs across the tray (ladder tray look)
    for k in range(int(ln / 0.3) + 1):
        _box(b, f"env_tray_rung_{k}", (0.24, 0.02, 0.02), (xt, y0 + 0.3 * k, 0.02), M["galv"])
    # corrugated flexible duct + cables lying in the tray
    _rod(b, "env_flex_duct", 0.035, ln - 0.2, M["cable"], (xt - 0.05, (y0 + y1) / 2, 0.055), (math.pi / 2, 0, 0), 12)
    _rod(b, "env_tray_cable_a", 0.014, ln - 0.4, M["cable"], (xt + 0.04, (y0 + y1) / 2 + 0.1, 0.035), (math.pi / 2, 0, 0), 8)
    _rod(b, "env_tray_cable_b", 0.010, ln - 0.6, M["welder_red"], (xt + 0.075, (y0 + y1) / 2 - 0.1, 0.03), (math.pi / 2, 0, 0), 8)
    cab = L.CONTROLLER_CABINET["pos"]
    ps = L.WELD_POWER_SOURCE["pos"]
    dr = L.WIRE_DRUM["pos"]
    for k, (p0, p1, r) in enumerate((
            ((cab[0] - 0.3, cab[1] - 0.1, 0.05), (xt + 0.05, cab[1] - 0.1, 0.05), 0.016),
            ((cab[0] - 0.3, cab[1] - 0.2, 0.04), (xt + 0.05, cab[1] - 0.2, 0.04), 0.012),
            ((ps[0] - 0.25, ps[1] - 0.2, 0.25), (xt + 0.05, ps[1] - 0.25, 0.05), 0.014),
            ((ps[0] - 0.25, ps[1] - 0.1, 0.25), (xt + 0.05, ps[1] - 0.05, 0.05), 0.010),
            ((ps[0] - 0.1, ps[1] - 0.35, 0.9), (dr[0] - 0.1, dr[1] + 0.32, L.WIRE_DRUM["height"] + 0.15), 0.008),
            ((cab[0] - 0.36, cab[1] + 0.1, 0.9), (cab[0] - 0.36, ps[1] + 0.35, 0.9), 0.008))):
        _bar(b, f"env_cable_{k}", p0, p1, 0, M["cable"], radius=r, segs=8)
    # floor cable bridge (yellow ramp) where two cables leave the tray toward a socket box at the fence line;
    # centred between the east fence posts (y = -0.68 / -2.04) and clear of the tray
    bxc, byc = 3.45, -1.36
    _box(b, "env_cable_bridge", (0.5, 0.9, 0.05), (bxc, byc, 0.025), M["yellow"])
    _box(b, "env_cable_bridge_dark", (0.16, 0.9, 0.052), (bxc, byc, 0.026), M["black"])
    fx = L.FENCE_X[1]
    _box(b, "env_socket_box", (0.16, 0.22, 0.14), (fx - 0.16, byc, 0.07), M["cabinet_dark"])
    _box(b, "env_socket_box_lid", (0.14, 0.20, 0.01), (fx - 0.16, byc, 0.145), M["cabinet"])
    for k, (dy, r) in enumerate(((-0.08, 0.012), (0.08, 0.009))):
        _bar(b, f"env_bridge_cable_{k}", (xt + 0.06, byc + dy, 0.02), (fx - 0.25, byc + dy, 0.02), 0, M["cable"], radius=r, segs=8)


def _flange(b, M, name, ro, ri, rf, thk, hub, center, rotation=(0, 0, 0), segs=40, holes=8):
    """Weld-neck flange: axis local +Z, back face at local z = 0; bolt holes faked as dark discs."""
    prof = [(ri, 0), (rf, 0), (rf, thk), (rf - (rf - ro) * 0.3, thk), (ro, hub), (ri, hub)]
    ob = _rev(b, name, prof, M["pipe"], center, rotation, segs)
    pcd = (rf + ro) / 2 + (rf - ro) * 0.15
    hr = (rf - ro) * 0.14
    for k in range(holes):
        a = 2 * math.pi * k / holes
        h = _rev(b, f"{name}_hole_{k}", [(0, -0.002), (hr, -0.002), (hr, 0.002), (0, 0.002)], M["dark"], (pcd * math.cos(a), pcd * math.sin(a), 0),
                 segs=10, smooth=False)
        G.set_parent(h, ob, keep_world=False)
    return ob


def _sleeper(b, M, name, length, center, rotation=(0, 0, 0)):
    return _box(b, name, (0.10, length, 0.10), center, M["wood"], rotation)


def _parts_pallet(b, M):
    """Euro-ish pallet with two DN250 LR elbows and a spare WN flange (shifted -X so it stays out of the keep-clear box)."""
    px, py, _ = L.PARTS_PALLET["pos"]
    px -= 0.25
    py += 0.05
    pw, pl = 0.8, 1.6
    for k, dx in enumerate((-0.35, 0, 0.35)):
        _box(b, f"env_pallet_skid_{k}", (0.1, pl, 0.022), (px + dx, py, 0.011), M["wood"])
        for j, dy in enumerate((-0.7, 0, 0.7)):
            _box(b, f"env_pallet_block_{k}_{j}", (0.1, 0.14, 0.078), (px + dx, py + dy, 0.061), M["wood"])
    for k, dy in enumerate((-0.7, 0, 0.7)):
        _box(b, f"env_pallet_strut_{k}", (pw, 0.14, 0.022), (px, py + dy, 0.111), M["wood"])
    for k in range(7):
        _box(b, f"env_pallet_board_{k}", (pw, 0.1 if k % 3 else 0.14, 0.022), (px, py - pl / 2 + 0.08 + k * (pl - 0.16) / 6, 0.133), M["wood"])
    zt = 0.144
    ro, ri = L.PIPE_OD / 2, L.PIPE_OD / 2 - L.PIPE_WALL
    R = L.ELBOW_R

    def elbow(name, loc, rz):
        e = G.torus_sweep(name, R, math.pi / 2, lambda t: [ro, ri], seg_bend=24, seg_tube=36, collection=b.col)
        e.location = loc
        e.rotation_euler = (math.pi / 2, 0, rz)   # arc plane horizontal, tube axis extent along Z
        return b.reg(e, M["pipe"])

    # elbow A: arc from its origin toward (+X, -Y) after the rotation; elbow B mirrored (rz = pi)
    elbow("env_spare_elbow_a", (px - pw / 2 + ro + 0.02, py + pl / 2 - ro - 0.03, zt + ro), 0.0)
    elbow("env_spare_elbow_b", (px + pw / 2 - ro - 0.02, py - pl / 2 + ro + 0.03, zt + ro), math.pi)
    # spare flange standing on its rim in the gap between the elbows, leaning slightly
    rf = L.FLANGE_OD / 2
    _flange(b, M, "env_spare_flange", ro, ri, rf, L.FLANGE_THK, L.FLANGE_HUB_LEN, (px + 0.05, py + 0.04, zt + rf * math.cos(0.3) + 0.02),
            (0.3, math.pi / 2, 0), holes=L.FLANGE_HOLES)
    _box(b, "env_pallet_wedge", (0.12, 0.08, 0.06), (px + 0.05, py - 0.1, zt + 0.03), M["wood"])


def _finished_rack(b, M):
    """Finished small-bore spools (DN100/150/200 with flanges) on two wooden sleepers."""
    px, py, _ = L.FINISHED_RACK["pos"]
    px -= 0.3
    for k, dy in enumerate((-0.7, 0.7)):
        _sleeper(b, M, f"env_rack_sleeper_{k}", 0.9, (px - 0.03, py + dy, 0.05), (0, 0, math.pi / 2))
    spools = [  # (pipe OD, wall, flange OD, flange thk, length, x offset, y offset)
        (0.219, 0.008, 0.340, 0.024, 2.0, -0.32, 0.0),
        (0.168, 0.007, 0.285, 0.022, 1.8, -0.02, -0.08),
        (0.114, 0.006, 0.220, 0.018, 1.6, 0.24, 0.10),
    ]
    for k, (od, wall, fod, fthk, ln, dx, dy) in enumerate(spools):
        ro, ri = od / 2, od / 2 - wall
        zc = 0.10 + fod / 2
        cx, cy = px + dx, py + dy
        _pipe(b, f"env_rack_pipe_{k}", ro, ri, ln - 0.2, M["pipe"], (cx, cy, zc), (math.pi / 2, 0, 0), 32)
        for j, s in enumerate((-1, 1)):
            _flange(b, M, f"env_rack_flange_{k}_{j}", ro, ri, fod / 2, fthk, 0.08, (cx, cy + s * ln / 2, zc),
                    (math.pi / 2 if s > 0 else -math.pi / 2, 0, 0), segs=32, holes=8)
        # weld ring where hub meets pipe
        for j, s in enumerate((-1, 1)):
            _pipe(b, f"env_rack_weld_{k}_{j}", ro + 0.004, ro - 0.001, 0.014, M["steel"], (cx, cy + s * (ln / 2 - 0.09), zc), (math.pi / 2, 0, 0), 24)
    _box(b, "env_rack_tag", (0.08, 0.001, 0.05), (px - 0.32, py - 1.0 - 0.02, 0.28), M["white_btn"])


# ============================================================================ workshop dressing (outside the fence)
def _turning_rolls(b, M, name, center, length, pipe_r, mat_frame):
    """Two roller stands with a large pipe section resting on them (axis along X)."""
    cx, cy, _ = center
    zc = 0.35 + pipe_r * 0.92
    for k, dx in enumerate((-length * 0.3, length * 0.3)):
        _box(b, f"{name}_stand_{k}", (0.6, 1.5, 0.3), (cx + dx, cy, 0.15), mat_frame)
        for j, dy in enumerate((-0.55, 0.55)):
            _box(b, f"{name}_bkt_{k}_{j}", (0.3, 0.12, 0.35), (cx + dx, cy + dy, 0.45), mat_frame)
            _rod(b, f"{name}_roll_{k}_{j}", 0.18, 0.24, M["rubber"], (cx + dx, cy + dy, 0.5), (0, math.pi / 2, 0), 20)
        _box(b, f"{name}_motor_{k}", (0.25, 0.25, 0.25), (cx + dx, cy - 0.95, 0.4), M["cabinet_dark"])
    _pipe(b, f"{name}_pipe", pipe_r, pipe_r - 0.014, length, M["pipe"], (cx, cy, zc), (0, math.pi / 2, 0), 40)


def _dressing(b, M):
    blue = M["machine_blue"]
    # big pipe sections on turning rolls (NE quadrant, seen from the wide camera)
    _turning_rolls(b, M, "env_rolls_a", (10.0, 8.5, 0), 6.0, 0.30, blue)
    _turning_rolls(b, M, "env_rolls_b", (11.0, 11.5, 0), 5.0, 0.22, blue)
    # column-and-boom welding manipulator (NW quadrant) with a pipe on rolls under the boom
    cx, cy = -9.0, 10.0
    _box(b, "env_cb_rail_0", (5.0, 0.12, 0.10), (cx, cy - 0.7, 0.05), M["dark"])
    _box(b, "env_cb_rail_1", (5.0, 0.12, 0.10), (cx, cy + 0.7, 0.05), M["dark"])
    _box(b, "env_cb_carriage", (2.2, 1.8, 0.45), (cx, cy, 0.32), blue)
    _box(b, "env_cb_column", (0.6, 0.6, 5.0), (cx, cy, 3.0), blue)
    _box(b, "env_cb_column_rail", (0.08, 0.66, 4.6), (cx - 0.3, cy, 2.9), M["steel"])
    _box(b, "env_cb_head", (0.9, 0.9, 0.9), (cx - 0.5, cy, 3.8), blue)
    _box(b, "env_cb_boom", (0.5, 4.5, 0.5), (cx - 0.5, cy - 2.6, 3.8), blue)
    _box(b, "env_cb_boom_rail", (0.55, 4.0, 0.06), (cx - 0.5, cy - 2.6, 3.5), M["steel"])
    _box(b, "env_cb_wirebox", (0.4, 0.5, 0.4), (cx - 0.5, cy - 1.8, 4.25), M["cabinet_dark"])
    _box(b, "env_cb_headunit", (0.3, 0.3, 0.6), (cx - 0.5, cy - 4.6, 3.2), M["cabinet_dark"])
    _bar(b, "env_cb_torch", (cx - 0.5, cy - 4.6, 2.9), (cx - 0.5, cy - 4.7, 2.2), 0, M["dark"], radius=0.03, segs=10)
    _box(b, "env_cb_counterweight", (0.8, 0.7, 0.7), (cx + 0.7, cy, 5.3), M["dark"])
    _box(b, "env_cb_ladder", (0.05, 0.4, 5.0), (cx + 0.32, cy, 2.6), M["yellow"])
    _turning_rolls(b, M, "env_rolls_c", (cx - 0.5, cy - 4.7, 0), 5.5, 0.26, M["machine_green"])
    # large elbows and a pipe on sleepers (SW quadrant), like the reference photos
    ro = 0.30
    for k, (x, y, rz) in enumerate(((-10.0, -9.0, 0.3), (-7.5, -10.5, 2.4))):
        e = G.torus_sweep(f"env_big_elbow_{k}", 0.9, math.pi / 2, lambda t: [ro, ro - 0.012], seg_bend=20, seg_tube=32, collection=b.col)
        e.location = (x, y, 0.1 + ro)
        e.rotation_euler = (math.pi / 2, 0, rz)
        b.reg(e, M["pipe"])
        for j in range(2):
            _sleeper(b, M, f"env_big_elbow_slp_{k}_{j}", 1.4, (x + 0.7 * math.cos(rz) * (1 if j else 0) + 0.3 * j, y - 0.6 * j - 0.3, 0.05), (0, 0, rz))
    _pipe(b, "env_floor_pipe", 0.30, 0.288, 7.0, M["pipe"], (-10.0, -12.5, 0.4), (0, math.pi / 2, 0), 40)
    for k, x in enumerate((-12.5, -7.5)):
        _sleeper(b, M, f"env_floor_pipe_slp_{k}", 1.4, (x, -12.5, 0.05))
    # gas-cylinder rack + workbench with vice (SE quadrant, seen from the reverse camera)
    gx, gy = 9.0, -9.0
    _box(b, "env_gas_rack_base", (1.5, 0.6, 0.06), (gx, gy, 0.03), M["yellow"])
    for k, dx in enumerate((-0.7, 0.7)):
        _box(b, f"env_gas_rack_post_{k}", (0.05, 0.05, 1.6), (gx + dx, gy + 0.25, 0.8), M["yellow"])
    _box(b, "env_gas_rack_bar", (1.5, 0.03, 0.03), (gx, gy + 0.25, 1.1), M["yellow"])
    for k, dx in enumerate((-0.5, -0.17, 0.17, 0.5)):
        mat = (M["machine_green"], M["drum_grey"], M["machine_blue"], M["cabinet_dark"])[k]
        _rev(b, f"env_gas_cyl_{k}", [(0, 0), (0.115, 0), (0.115, 1.3), (0.07, 1.42), (0.04, 1.45), (0, 1.45)], mat, (gx + dx, gy, 0.06), segs=24)
        _rev(b, f"env_gas_cap_{k}", [(0.07, 0), (0.075, 0), (0.075, 0.18), (0.055, 0.2), (0, 0.2), (0, 0.18), (0.06, 0.18), (0.06, 0.02), (0.07, 0.01)],
             M["steel"], (gx + dx, gy, 1.5), segs=20)
    wx, wy = 6.5, -10.0
    _box(b, "env_bench_top", (2.0, 0.8, 0.05), (wx, wy, 0.9), M["steel"])
    _box(b, "env_bench_shelf", (1.9, 0.7, 0.03), (wx, wy, 0.25), M["cabinet_dark"])
    for k, (dx, dy) in enumerate(((-0.95, -0.35), (0.95, -0.35), (-0.95, 0.35), (0.95, 0.35))):
        _box(b, f"env_bench_leg_{k}", (0.06, 0.06, 0.88), (wx + dx, wy + dy, 0.44), M["cabinet_dark"])
    _box(b, "env_bench_vice_base", (0.25, 0.2, 0.1), (wx - 0.6, wy - 0.25, 0.975), M["dark"])
    _box(b, "env_bench_vice_jaw", (0.18, 0.08, 0.12), (wx - 0.6, wy - 0.28, 1.06), M["dark"])
    _rod(b, "env_bench_vice_spindle", 0.012, 0.3, M["steel"], (wx - 0.6, wy - 0.4, 1.02), (math.pi / 2, 0, 0), 10)
    _box(b, "env_bench_grinder", (0.35, 0.15, 0.1), (wx + 0.4, wy + 0.1, 0.975), M["machine_green"])
    _box(b, "env_bench_board", (2.0, 0.03, 1.0), (wx, wy + 0.42, 1.6), M["cabinet_dark"])
    # welding curtain screens on frames (dark red) between the cell and the bench area
    for k, (x, y, rz) in enumerate(((6.0, -6.0, 0.0), (8.2, -6.0, 0.0), (-6.0, 6.5, math.pi / 2))):
        _box(b, f"env_curtain_frame_{k}", (2.0, 0.04, 0.04), (x, y, 1.95), M["dark"], (0, 0, rz))
        for j, dx in enumerate((-1.0, 1.0)):
            _box(b, f"env_curtain_leg_{k}_{j}", (0.04, 0.04, 1.95), (x + dx * math.cos(rz), y + dx * math.sin(rz), 0.975), M["dark"])
            _box(b, f"env_curtain_foot_{k}_{j}", (0.05, 0.6, 0.03), (x + dx * math.cos(rz), y + dx * math.sin(rz), 0.015), M["dark"], (0, 0, rz))
        _box(b, f"env_curtain_{k}", (1.9, 0.01, 1.75), (x, y, 1.05), M["curtain"], (0, 0, rz))
    # stacked euro pallets + a steel scrap bin against the south wall
    for k in range(4):
        _box(b, f"env_pallet_stack_{k}", (1.2, 0.8, 0.144), (4.0, -HY + 1.2, 0.072 + 0.15 * k), M["wood"])
    _box(b, "env_scrap_bin", (1.2, 1.0, 0.8), (6.0, -HY + 1.3, 0.45), M["machine_blue"])


# ============================================================================ public API
def build(collection=None):
    """Build all static hall / fence / equipment geometry.  Returns dict(collection, objects)."""
    col = collection or bpy.data.collections.new("Environment")
    if collection is None:
        bpy.context.scene.collection.children.link(col)
    b = _B(col)
    M = _mats()
    _hall(b, M)
    _crane(b, M)
    _lamps(b, M)
    _fence(b, M)
    _controller(b, M)
    _power_source(b, M)
    _wire_drum(b, M)
    _torch_cleaner(b, M)
    _fume_hood(b, M)
    _cable_management(b, M)
    _parts_pallet(b, M)
    _finished_rack(b, M)
    _dressing(b, M)
    return dict(collection=col, objects=b.objs)


def _area_light(col, name, location, target, energy, size, color, shape='SQUARE', spread=math.radians(150),
                shadow=True, shadow_res=0.01):
    """Area light aimed at `target`.  shadow_res = coarsest shadow-map texel (m): cheap on CPU/software GL."""
    ld = bpy.data.lights.new(name, 'AREA')
    ld.energy = energy
    ld.color = color
    ld.shape = shape
    ld.size = size
    ld.spread = spread
    ld.use_shadow = shadow
    ld.shadow_maximum_resolution = shadow_res
    ld.shadow_soft_size = size * 0.5
    ob = bpy.data.objects.new(name, ld)
    col.objects.link(ob)
    ob.location = location
    d = mathutils.Vector(target) - mathutils.Vector(location)
    ob.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    return ob


def build_lighting(scene):
    """World background + high-bay area lights + key/fill for the cell.  Returns dict(world, lights)."""
    col = bpy.data.collections.get("EnvironmentLights") or bpy.data.collections.new("EnvironmentLights")
    if col.name not in scene.collection.children:
        scene.collection.children.link(col)
    world = bpy.data.worlds.get("HallWorld") or bpy.data.worlds.new("HallWorld")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.30, 0.31, 0.33, 1.0)     # neutral grey, low: lets the blacks stay black
    bg.inputs[1].default_value = 0.05
    scene.world = world

    # Only env_light_key (and the arc) cast shadows in the final render (build.setup_render): the bays, fill and rim
    # are shadowless ambience, so they are kept weak and the key does the modelling under the machines.
    lights = []
    warm = (1.0, 0.80, 0.62)                                  # ~3500 K high-pressure sodium / old HID look
    for i, (x, y) in enumerate(LAMP_GRID):
        lights.append(_area_light(col, f"env_light_bay_{i}", (x, y, LAMP_Z - 0.03), (x, y, 0), 750.0, 0.6, warm, 'DISK', math.radians(160), shadow_res=0.02))
    kx, ky, kz = KEY_LIGHT_POS
    lights.append(_area_light(col, "env_light_key", (kx, ky, kz), (0.4, 0.0, 1.0), 650.0, 2.2, (1.0, 0.95, 0.90), 'SQUARE', math.radians(120), shadow_res=0.006))
    lights.append(_area_light(col, "env_light_fill", (-5.5, -7.0, 3.5), (0.5, 0.0, 1.0), 90.0, 6.0, (0.85, 0.9, 1.0), 'SQUARE', math.radians(150), shadow=False))
    lights.append(_area_light(col, "env_light_rim", (5.0, 5.0, 3.5), (0.5, 0.0, 1.2), 220.0, 4.0, (0.78, 0.86, 1.0), 'SQUARE', math.radians(150), shadow=False))

    # EEVEE Next settings: shadows + screen-space raytracing on, volumetrics off
    ev = scene.eevee
    ev.use_shadows = True
    ev.shadow_ray_count = 2
    ev.shadow_step_count = 4
    ev.use_raytracing = True
    ev.ray_tracing_method = 'SCREEN'
    rt = ev.ray_tracing_options
    rt.resolution_scale = '2'
    rt.screen_trace_quality = 0.3
    rt.trace_max_roughness = 0.6
    rt.use_denoise = True
    ev.use_fast_gi = True
    ev.fast_gi_ray_count = 2
    ev.fast_gi_step_count = 6
    ev.use_volumetric_shadows = False
    ev.light_threshold = 0.02
    ev.shadow_pool_size = '512'
    ev.shadow_resolution_scale = 0.5          # ~2x faster on CPU, no visible loss at 1080p
    return dict(world=world, lights=lights)
