"""Kit pallet (steel cassette with flange and elbow nests) and the pipe buffer rack (V-cradles) of the logistics zone.

Everything is placed from layout2: the pallet `KIT_PALLET`, the cassette base plate top `KIT_CASSETTE_Z`, the part
frames `KIT_FLANGES` / `KIT_ELBOWS` and the pipe slots `pipe_buffer_frame(i)`.  Part frames are spool frames
(kin.planar_frame(origin, x_dir)); a part placed with its root at such a frame sits in its nest:

* flange nest: machined ring under the flange back face (the raised face and the bolt holes stay free inside the
  ring), 3 short centring pins just outside the flange rim (their tops stay below the flange top face);
* elbow nest: cup under the lower (flange-side) end with a polymer lining, plus a narrow V-rest under the upper end
  (a 22 mm plate at local x = 0.357..0.379, flanks touching the tube 35 deg off its bottom);
* pipe buffer: two V-cradle beams (at the pipe ends, outside the jaw zone at the pipe middle), an end stop with a
  rubber strip at the +Y pipe end, and one presence sensor (M18, through the end stop) per slot with an LED.

Public API (DESIGN.md section 3, cassette.py):
    build(collection=None) -> dict(collection, pallet, nests, pipe_buffer)
        pallet      = dict(root=<empty 'kit_pallet_root' at the pallet centre on the floor>, objects=[...])
                      every pallet / cassette / nest object is parented to root (move the root to move the kit)
        nests       = dict(flange=[T0, T1], elbow=[T0, T1])   part-root frames, numpy 4x4 (world, at build time)
        pipe_buffer = dict(root=<empty 'kit_buf_root'>, objects=[...], slots=[T0, T1], leds=[obj0, obj1])
                      one slot / LED per layout2.PIPE_BUFFER_X entry (slot i = planar_frame(*pipe_buffer_frame(i)))
    set_presence(cas, i, on, frame=None)   buffer slot i presence LED (object property 'led_on', CONSTANT key)
    key_constant(ob, path, frame)          helper: key + CONSTANT interpolation (shared with storage.py)
    obstacles() -> list[dict(name, center, size, yaw)]   conservative world AABBs of the static geometry (no bpy use)
"""
import math

import numpy as np
import bpy
import mathutils

import tools  # noqa: F401  (demo_video on sys.path)
from cell import geom as G
from cell import materials
from cell import environment as E
from cell import positioner as P1
import layout2 as L2
import kin

NAME = "KitCassette"
GAP = 0.0003                       # design clearance between a part and its support (visually touching)

# ------------------------------------------------------------------ derived dimensions (world unless noted)
PAL_C = L2.KIT_PALLET["center"]
PAL_S = L2.KIT_PALLET["size"]
PAL_TOP = L2.KIT_PALLET["top_z"]                  # 0.144
CAS_Z = L2.KIT_CASSETTE_Z                         # 0.20 top of the cassette base plate
CAS_PLATE_T = 0.016
CAS_FRAME_Z = (PAL_TOP, CAS_Z - CAS_PLATE_T)      # RHS base frame under the plate
CAS_SIZE = (PAL_S[0] - 0.04, PAL_S[1] - 0.04)     # plate footprint (x, y)
CAS_LIP_H = 0.06
CAS_POST_H = 0.36                                 # corner (stacking) posts above the plate

FL_RING = (0.193, 0.222)          # flange seat ring inner / outer radius (inner > raised face 0.176 and bolt holes)
FL_PIN_R = 0.2125                 # pin circle (flange rim 0.2025 + pin radius + 2 mm)
FL_PIN = (0.008, 0.019)           # pin radius, height above the seat
FL_PIN_ANG = (90.0, 210.0, 330.0)
FL_PLATE = 0.46                   # square adapter plate under the ring

EL_CUP_R = (0.150, 0.168, 0.19)   # pocket radius, cup outer radius, base flange radius
EL_CUP_TOP = 0.140                # cup rim, nest-local z (elbow lower end sits on local z = SEAM_A_Z)
EL_LINING_R = 0.143               # polymer lining inner radius
EL_VREST_X = (0.357, 0.379)       # V-rest plate (nest-local x), under the elbow's upper (pipe-side) end
EL_VREST_BETA = math.radians(35.0)
EL_VREST_W = 0.100                # half width of the V pad

BUF_C = L2.PIPE_BUFFER["center"]
BUF_AXIS_Z = L2.PIPE_BUFFER["axis_z"]
BUF_X = L2.PIPE_BUFFER_X
BUF_BEAM_Y = (BUF_C[1] - 0.22, BUF_C[1] + 0.22)  # cradle beams near the pipe ends; jaws grip the middle +-0.13
BUF_BEAM_TOP = 0.69
BUF_BEAM = (0.08, 0.10)                            # beam section (y, z)
BUF_X_RANGE = (L2.PIPE_BUFFER["center"][0] - L2.PIPE_BUFFER["size"][0] / 2 + 0.03,
               L2.PIPE_BUFFER["center"][0] + L2.PIPE_BUFFER["size"][0] / 2 - 0.03)
BUF_LEG_X = (BUF_X_RANGE[0] + 0.045, BUF_X_RANGE[1] - 0.045)
BUF_V_W = 0.13                                     # V-block half width (x)
BUF_V_T = 0.06                                     # V-block thickness (y)
BUF_PIPE_END_Y = BUF_C[1] + L2.L.PIPE_LEN / 2      # +Y pipe end (pipe local x = SEAM_B_X)
BUF_STOP_Y = (BUF_PIPE_END_Y + 0.0015, BUF_PIPE_END_Y + 0.0215)   # rubber strip face .. stop plate back face
BUF_STOP_Z = (0.72, 0.99)


# ============================================================================ small helpers
def _mats():
    return dict(
        blue=materials.get("painted", color="#1F4E8C", roughness=0.42),
        frame=materials.get("painted", color="#2B2F36", roughness=0.5),
        yellow=materials.get("safety_yellow"),
        dark=materials.get("dark_metal"),
        steel=materials.get("machined_steel"),
        black=materials.get("black_plastic"),
        rubber=materials.get("rubber"),
        cable=materials.get("cable_black"),
        wood=materials.get("painted", color="#8A6A42", roughness=0.8, coat=0.0),
        wood2=materials.get("painted", color="#7A5C38", roughness=0.85, coat=0.0),
        uhmw=materials.get("painted", color="#C9C3AF", roughness=0.6, coat=0.0),
        led=_state_mat("kit_led_green", (0.19, 1.0, 0.25), 8.0, "led_on"),
    )


def _state_mat(key, color, strength, prop):
    """Emissive indicator whose strength is object property `prop` (0..1) x strength (read by an Attribute node)."""
    m = bpy.data.materials.get(key)
    if m is not None:
        return m
    m = bpy.data.materials.new(key)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (color[0] * 0.08, color[1] * 0.08, color[2] * 0.08, 1.0)
    b.inputs["Roughness"].default_value = 0.25
    b.inputs["Emission Color"].default_value = (*color, 1.0)
    att = nt.nodes.new("ShaderNodeAttribute")
    att.attribute_type = 'OBJECT'
    att.attribute_name = prop
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = 'MULTIPLY'
    mul.inputs[1].default_value = strength
    nt.links.new(att.outputs["Fac"], mul.inputs[0])
    nt.links.new(mul.outputs[0], b.inputs["Emission Strength"])
    return m


def _bevel(ob, w, segs=2):
    m = ob.modifiers.new("bevel", 'BEVEL')
    m.width = w
    m.segments = segs
    m.limit_method = 'ANGLE'
    return ob


def _bx(b, name, size, center, mat, rot=(0, 0, 0), bev=0.0):
    ob = E._box(b, name, size, center, mat, rot)
    return _bevel(ob, bev) if bev > 0 else ob


def _hex(b, name, r, h, mat, center, rot=(0, 0, 0)):
    """Hex bolt head / nut: 6-sided solid, base at center, axis local +Z."""
    return E._rev(b, name, [(0, 0), (r, 0), (r, h), (0, h)], mat, center, rot, segs=6, smooth=False)


def _to_frame(objs, F):
    """Objects built in a local frame (loc/rot relative to it) -> world by the numpy frame F."""
    Fm = G.M(F)
    for ob in objs:
        ob.matrix_world = Fm @ ob.matrix_basis


def _parent_all(objs, root):
    """Parent objects to root keeping their world transform (root.matrix_world must be final)."""
    inv = root.matrix_world.inverted()
    for ob in objs:
        if ob.parent is None and ob is not root:
            ob.parent = root
            ob.matrix_parent_inverse = inv


_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _label_mat(key, texts, cell=(320, 128), fg=(20, 20, 20), bg=(242, 180, 0)):
    """Material with a vertical atlas of text cells (PIL); a quad maps cell k via _label()."""
    m = bpy.data.materials.get(key)
    if m is not None:
        return m
    from PIL import Image, ImageDraw, ImageFont
    W, H = cell
    img = Image.new("RGB", (W, H * len(texts)), bg)
    d = ImageDraw.Draw(img)
    for k, t in enumerate(texts):
        size = int(H * 0.60)
        font = ImageFont.truetype(_FONT, size)
        while font.getlength(t) > W * 0.84 and size > 10:
            size -= 2
            font = ImageFont.truetype(_FONT, size)
        y0 = k * H
        d.rectangle((5, y0 + 5, W - 6, y0 + H - 6), outline=fg, width=4)
        d.text((W / 2, y0 + H / 2), t, font=font, fill=fg, anchor="mm")
    arr = np.asarray(img.convert("RGBA"), dtype=np.float32)[::-1] / 255.0
    bi = bpy.data.images.new(key, img.width, img.height, alpha=True)
    bi.colorspace_settings.name = 'sRGB'
    bi.pixels.foreach_set(arr.ravel())
    bi.pack()
    m = bpy.data.materials.new(key)
    m.use_nodes = True
    nt = m.node_tree
    bs = nt.nodes["Principled BSDF"]
    bs.inputs["Roughness"].default_value = 0.45
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bi
    tex.interpolation = 'Linear'
    uv = nt.nodes.new("ShaderNodeUVMap")
    nt.links.new(uv.outputs["UV"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], bs.inputs["Base Color"])
    m["n_cells"] = len(texts)
    return m


def _label(b, name, mat, k, size, center, rot, backing=None):
    """Label quad (local XY, normal +Z, text up = local +Y) showing atlas cell k; optional dark backing plate."""
    w, h = size
    n = int(mat["n_cells"])
    ob = E._mesh(b, name, [(-w / 2, -h / 2, 0), (w / 2, -h / 2, 0), (w / 2, h / 2, 0), (-w / 2, h / 2, 0)],
                 [(0, 1, 2, 3)], mat, center, rot)
    v0, v1 = 1.0 - (k + 1) / n, 1.0 - k / n
    uvl = ob.data.uv_layers.new(name="UVMap")
    uvs = [(0, v0), (1, v0), (1, v1), (0, v1)]
    for li, lp in enumerate(ob.data.loops):
        uvl.data[li].uv = uvs[lp.vertex_index]
    if backing is not None:
        R = mathutils.Euler(rot).to_matrix()
        c = mathutils.Vector(center) - R @ mathutils.Vector((0, 0, 0.0025))
        E._box(b, name + "_plate", (w + 0.012, h + 0.012, 0.004), tuple(c), backing, rot)
    return ob


# ============================================================================ elbow V-rest (numeric touch plane)
def _elbow_surface(x0, x1, n_th=240, n_a=360):
    """Outer surface points of the stage-1 elbow (spool-local) whose x lies in [x0, x1]."""
    R = L2.L.ELBOW_R
    r = L2.PIPE_R
    th = np.linspace(math.radians(60.0), math.pi / 2, n_th)[:, None]
    a = np.linspace(0, 2 * math.pi, n_a, endpoint=False)[None, :]
    cx, cz = R * (1 - np.cos(th)), L2.SEAM_A_Z + R * np.sin(th)
    px = cx + r * np.cos(a) * (-np.cos(th))
    py = r * np.sin(a) * np.ones_like(th)
    pz = cz + r * np.cos(a) * np.sin(th)
    m = (px >= x0) & (px <= x1)
    return np.stack([px[m], py[m], pz[m]], axis=1)


def _elbow_vrest_apex():
    """z (spool-local) of the V apex so that both 35-deg flanks touch the elbow (+GAP) over the plate thickness."""
    pts = _elbow_surface(*EL_VREST_X)
    sb, cb = math.sin(EL_VREST_BETA), math.cos(EL_VREST_BETA)
    c = min(float(np.min(-sb * pts[:, 1] + cb * pts[:, 2])), float(np.min(sb * pts[:, 1] + cb * pts[:, 2])))
    return (c - GAP) / cb


def _v_poly(z_bot, z_apex, w, flat, tan_b):
    """CCW (u, v) outline of a V-block: flat bottom at z_bot, flanks z = z_apex + tan_b |u| truncated at |u| = flat."""
    return [(-w, z_bot), (w, z_bot), (w, z_apex + tan_b * w), (flat, z_apex + tan_b * flat),
            (-flat, z_apex + tan_b * flat), (-w, z_apex + tan_b * w)]


def _v_block(b, name, z_bot, z_apex, w, flat, beta, length, center, axis, body_mat, liner_mat, t=0.008, span=None):
    """Steel V-block + two polymer liners on its flanks.  The liner top surfaces are the design V (apex z_apex,
    flank angle beta from horizontal); extruded `length` along `axis` ('X': outline in (y, z), 'Y': in (x, z))."""
    tb = math.tan(beta)
    zab = z_apex - t / math.cos(beta)
    objs = [E._prism(b, name, _v_poly(z_bot, zab, w, flat, tb), length, body_mat, center, axis=axis)]
    ua, ub = span or (flat + 0.012, w)
    for k, s in enumerate((1, -1)):
        q = [(ua, zab + tb * ua), (ub, zab + tb * ub), (ub, z_apex + tb * ub), (ua, z_apex + tb * ua)]
        if s < 0:
            q = [(-u, v) for u, v in q][::-1]
        objs.append(E._prism(b, f"{name}_liner{k}", q, length, liner_mat, center, axis=axis))
    return objs


# ============================================================================ pallet + cassette
def _pallet(b, M):
    cx, cy = PAL_C
    sx, sy = PAL_S
    xs = (-0.525, 0.0, 0.525)
    ys = (-0.72, 0.0, 0.72)
    for k, dx in enumerate(xs):
        _bx(b, f"kit_pal_skid{k}", (0.145, sy, 0.022), (cx + dx, cy, 0.011), M["wood2"], bev=0.002)
        for j, dy in enumerate(ys):
            _bx(b, f"kit_pal_block{k}{j}", (0.145, 0.16, 0.078), (cx + dx, cy + dy, 0.061), M["wood2"], bev=0.003)
    for j, dy in enumerate(ys):
        _bx(b, f"kit_pal_stringer{j}", (sx, 0.16, 0.022), (cx, cy + dy, 0.111), M["wood"], bev=0.002)
    widths = (0.145, 0.10, 0.10, 0.145, 0.10, 0.10, 0.145)
    for k, w in enumerate(widths):
        x = cx - 0.5275 + 1.055 * k / 6
        _bx(b, f"kit_pal_deck{k}", (w, sy, 0.022), (x, cy, PAL_TOP - 0.011), M["wood"], bev=0.002)


def _cassette(b, M):
    cx, cy = PAL_C
    wx, wy = CAS_SIZE
    z0, z1 = CAS_FRAME_Z
    zc = (z0 + z1) / 2
    fh = z1 - z0
    # RHS base frame (dark) + cross members
    for k, s in enumerate((-1, 1)):
        _bx(b, f"kit_cas_rail_y{k}", (0.06, wy, fh), (cx + s * (wx / 2 - 0.03), cy, zc), M["frame"], bev=0.003)
    for k, dy in enumerate((-(wy / 2 - 0.03), 0.0, wy / 2 - 0.03)):
        _bx(b, f"kit_cas_rail_x{k}", (wx - 0.12, 0.06, fh), (cx, cy + dy, zc), M["frame"], bev=0.003)
    # base plate (machine blue) + perimeter lip
    _bx(b, "kit_cas_plate", (wx, wy, CAS_PLATE_T), (cx, cy, CAS_Z - CAS_PLATE_T / 2), M["blue"], bev=0.004)
    t = 0.008
    zl = CAS_Z + CAS_LIP_H / 2
    _bx(b, "kit_cas_lip_front", (wx, t, CAS_LIP_H), (cx, cy - wy / 2 + t / 2, zl), M["blue"], bev=0.002)
    _bx(b, "kit_cas_lip_back", (wx, t, CAS_LIP_H), (cx, cy + wy / 2 - t / 2, zl), M["blue"], bev=0.002)
    for k, s in enumerate((-1, 1)):
        _bx(b, f"kit_cas_lip_side{k}", (t, wy - 2 * t, CAS_LIP_H), (cx + s * (wx / 2 - t / 2), cy, zl), M["blue"], bev=0.002)
    # corner stacking posts (yellow) with black caps and steel stacking cones
    for k, (sx, sy) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
        px, py = cx + sx * (wx / 2 - 0.038), cy + sy * (wy / 2 - 0.038)
        _bx(b, f"kit_cas_post{k}", (0.06, 0.06, CAS_POST_H), (px, py, CAS_Z + CAS_POST_H / 2), M["yellow"], bev=0.004)
        _bx(b, f"kit_cas_post{k}_cap", (0.07, 0.07, 0.012), (px, py, CAS_Z + CAS_POST_H + 0.006), M["black"], bev=0.002)
        E._rev(b, f"kit_cas_post{k}_cone", [(0, 0), (0.022, 0), (0.012, 0.035), (0, 0.04)], M["steel"],
               (px, py, CAS_Z + CAS_POST_H + 0.012), segs=16)
        # gusset foot plate
        _bx(b, f"kit_cas_post{k}_foot", (0.10, 0.10, 0.008), (px - sx * 0.01, py - sy * 0.01, CAS_Z + 0.004), M["frame"])
    # kit ID plate + RFID tag on the front lip (toward the handler track)
    lm = _label_mat("kit_label", ["KIT 02 · SPL-02", "F1", "F2", "E1", "E2"] + [f"P{i + 1}" for i in range(len(BUF_X))])
    yl = cy - wy / 2 - 0.0012
    _label(b, "kit_cas_id", lm, 0, (0.22, 0.05), (cx - 0.18, yl, CAS_Z + CAS_LIP_H / 2 + 0.003), (math.pi / 2, 0, 0))
    _bx(b, "kit_cas_rfid", (0.07, 0.012, 0.035), (cx + 0.12, yl - 0.004, CAS_Z + CAS_LIP_H / 2 + 0.003), M["black"], bev=0.002)
    _bx(b, "kit_cas_rfid_led", (0.008, 0.003, 0.008), (cx + 0.145, yl - 0.011, CAS_Z + CAS_LIP_H / 2 + 0.012),
        materials.get("emissive", color="#30FF40", strength=8.0))
    return lm


def _flange_nest(b, M, i, lm):
    """Flange nest in the part frame: adapter plate + seat ring (top = back face - GAP) + 3 pins."""
    o, xd = L2.KIT_FLANGES[i]
    F = kin.planar_frame(o, xd)
    zb = CAS_Z - o[2]                              # plate top in the nest frame (negative)
    pre = f"kit_fnest{i}"
    objs = []
    objs.append(_bx(b, pre + "_plate", (FL_PLATE, FL_PLATE, 0.010), (0, 0, zb + 0.005), M["frame"], bev=0.003))
    z_ring0 = zb + 0.010
    for k, (sx, sy) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
        objs.append(_hex(b, f"{pre}_bolt{k}", 0.011, 0.008, M["dark"], (sx * 0.195, sy * 0.195, z_ring0)))
    ri, ro = FL_RING
    top = -GAP
    objs.append(E._rev(b, pre + "_ring", [(ri, z_ring0), (ro, z_ring0), (ro, top - 0.002), (ro - 0.002, top),
                                          (ri + 0.002, top), (ri, top - 0.002)], M["steel"], segs=64))
    for k, ang in enumerate(FL_PIN_ANG):
        a = math.radians(ang)
        h = FL_PIN[1]
        objs.append(E._rev(b, f"{pre}_pin{k}", [(0, 0), (FL_PIN[0], 0), (FL_PIN[0], h - 0.006), (FL_PIN[0] * 0.55, h), (0, h)],
                           M["steel"], (FL_PIN_R * math.cos(a), FL_PIN_R * math.sin(a), top), segs=12))
    # label lying on the plate in front of the nest (toward the track), text readable from -Y
    objs.append(_label(b, pre + "_label", lm, 1 + i, (0.07, 0.04), (0.29, 0.0, zb + 0.0012), (0, 0, math.pi / 2)))
    _to_frame(objs, F)
    return F


def _elbow_nest(b, M, i, lm):
    """Elbow nest in the part frame: cup under the lower end (floor = end face - GAP) + V-rest under the upper end."""
    o, xd = L2.KIT_ELBOWS[i]
    F = kin.planar_frame(o, xd)
    zb = CAS_Z - o[2]                              # plate top in the nest frame (0 for the layout numbers)
    pre = f"kit_enest{i}"
    objs = []
    rp, rc, rb = EL_CUP_R
    floor = L2.SEAM_A_Z - GAP
    objs.append(E._rev(b, pre + "_cup", [(0, zb), (rb, zb), (rb, zb + 0.012), (rc, zb + 0.012), (rc, EL_CUP_TOP - 0.004),
                                         (rc - 0.004, EL_CUP_TOP), (rp, EL_CUP_TOP), (rp, floor), (0, floor)],
                       M["blue"], segs=64))
    objs.append(E._rev(b, pre + "_lining", [(EL_LINING_R, floor), (rp, floor), (rp, EL_CUP_TOP + 0.001),
                                            (EL_LINING_R, EL_CUP_TOP + 0.001)], M["uhmw"], segs=64))
    for k in range(4):
        a = math.radians(45 + 90 * k)
        objs.append(_hex(b, f"{pre}_bolt{k}", 0.010, 0.008, M["dark"], ((rb - 0.012) * math.cos(a), (rb - 0.012) * math.sin(a), zb + 0.012)))
    # V-rest: foot plate, post, gussets, steel V with polymer liners (extruded along nest-local x)
    x0, x1 = EL_VREST_X
    xc, tx = (x0 + x1) / 2, x1 - x0
    z_apex = _elbow_vrest_apex()
    z_pad = z_apex - 0.024
    objs.append(_bx(b, pre + "_vfoot", (0.10, 0.16, 0.012), (xc, 0, zb + 0.006), M["frame"], bev=0.002))
    for k, s in enumerate((-1, 1)):
        objs.append(_hex(b, f"{pre}_vfoot_bolt{k}", 0.008, 0.007, M["dark"], (xc + s * 0.032, s * 0.058, zb + 0.012)))
    z_post0, z_post1 = zb + 0.012, z_pad
    objs.append(_bx(b, pre + "_vpost", (tx, 0.07, z_post1 - z_post0), (xc, 0, (z_post0 + z_post1) / 2), M["blue"], bev=0.002))
    gus = [(0.0, 0.0), (0.045, 0.0), (0.0, 0.14)]
    for k, s in enumerate((-1, 1)):
        poly = gus if s > 0 else [(-u, v) for u, v in gus][::-1]
        objs.append(E._prism(b, f"{pre}_vgusset{k}", poly, 0.010, M["blue"], (xc + s * tx / 2, 0, z_post0), axis='Y'))
    objs += _v_block(b, pre + "_vpad", z_pad, z_apex, EL_VREST_W, 0.022, EL_VREST_BETA, tx, (xc, 0, 0), 'X',
                     M["frame"], M["uhmw"], span=(0.034, EL_VREST_W))
    objs.append(_label(b, pre + "_label", lm, 3 + i, (0.07, 0.04), (-0.235, 0.0, zb + 0.0012), (0, 0, math.pi / 2)))
    _to_frame(objs, F)
    return F


# ============================================================================ pipe buffer
def _pipe_buffer(b, M, lm):
    x0, x1 = BUF_X_RANGE
    xm = (x0 + x1) / 2
    top = BUF_BEAM_TOP
    by, bz = BUF_BEAM
    leg_top = top - bz
    for j, yb in enumerate(BUF_BEAM_Y):
        _bx(b, f"kit_buf_beam{j}", (x1 - x0, by, bz), (xm, yb, top - bz / 2), M["blue"], bev=0.004)
        _bx(b, f"kit_buf_beam{j}_cap", (0.008, by, bz), (x0 - 0.004, yb, top - bz / 2), M["frame"])
        _bx(b, f"kit_buf_beam{j}_cap2", (0.008, by, bz), (x1 + 0.004, yb, top - bz / 2), M["frame"])
        for k, xl in enumerate(BUF_LEG_X):
            _bx(b, f"kit_buf_leg{j}{k}", (0.08, 0.08, leg_top - 0.012), (xl, yb, (leg_top + 0.012) / 2), M["blue"], bev=0.003)
            _bx(b, f"kit_buf_foot{j}{k}", (0.13, 0.13, 0.012), (xl, yb, 0.006), M["frame"], bev=0.002)
            for n, (sx, sy) in enumerate(((-1, -1), (1, 1))):
                _hex(b, f"kit_buf_anchor{j}{k}{n}", 0.010, 0.012, M["dark"], (xl + sx * 0.048, yb + sy * 0.048, 0.012))
    # ties along Y between the two frames (low and mid height), a diagonal brace per end
    y0, y1 = BUF_BEAM_Y
    for k, xl in enumerate(BUF_LEG_X):
        for n, z in enumerate((0.12, 0.40)):
            _bx(b, f"kit_buf_tie{k}{n}", (0.05, y1 - y0 - 0.08, 0.05), (xl, (y0 + y1) / 2, z), M["blue"], bev=0.003)
        E._bar(b, f"kit_buf_brace{k}", (xl, y0 + 0.04, 0.145), (xl, y1 - 0.04, 0.375), 0.035, M["blue"])
    # V-blocks: saddle plate + steel V with polymer liners (45 deg flanks touching the pipe at +-45 deg) per slot and beam
    r = L2.PIPE_R
    z_apex = BUF_AXIS_Z - (r + GAP) * math.sqrt(2.0)
    for j, yb in enumerate(BUF_BEAM_Y):
        for i, xs in enumerate(BUF_X):
            _bx(b, f"kit_buf_vbase{j}{i}", (2 * BUF_V_W + 0.024, BUF_V_T + 0.01, 0.010), (xs, yb, top + 0.005), M["frame"], bev=0.002)
            _v_block(b, f"kit_buf_v{j}{i}", top + 0.010, z_apex, BUF_V_W, 0.020, math.radians(45.0), BUF_V_T, (xs, yb, 0), 'Y',
                     M["frame"], M["uhmw"], span=(0.045, BUF_V_W))
            for n, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
                _hex(b, f"kit_buf_vbolt{j}{i}{n}", 0.007, 0.006, M["dark"], (xs + sx * (BUF_V_W + 0.004), yb + sy * 0.022, top + 0.010))
    # end stop at the +Y pipe end: plate on two brackets between the pipes, rubber strip at the axis height
    ys0, ys1 = BUF_STOP_Y
    yp0 = ys0 + 0.004                                  # plate front face (rubber strip in front of it)
    zs0, zs1 = BUF_STOP_Z
    _bx(b, "kit_buf_stop", (x1 - x0, ys1 - yp0, zs1 - zs0), (xm, (yp0 + ys1) / 2, (zs0 + zs1) / 2), M["frame"], bev=0.003)
    _bx(b, "kit_buf_stop_rubber", (x1 - x0 - 0.04, 0.004, 0.06), (xm, ys0 + 0.002, BUF_AXIS_Z), M["rubber"], bev=0.001)
    _bx(b, "kit_buf_stop_hz", (x1 - x0 + 0.002, ys1 - yp0 + 0.002, 0.04), (xm, (yp0 + ys1) / 2, zs1 - 0.02), E._mat_hazard())
    yb1 = BUF_BEAM_Y[1] + by / 2
    for k in range(len(BUF_X) - 1):
        xb = (BUF_X[k] + BUF_X[k + 1]) / 2
        _bx(b, f"kit_buf_stop_bkt{k}", (0.04, yp0 - yb1, zs1 - 0.02 - (top - bz)), (xb, (yb1 + yp0) / 2, (zs1 - 0.02 + top - bz) / 2),
            M["frame"], bev=0.003)
    for k, xr in enumerate((x0 + 0.006, x1 - 0.006)):
        E._prism(b, f"kit_buf_stop_rib{k}", [(0.0, zs0), (0.06, zs0), (0.0, zs1 - 0.04)], 0.012, M["frame"], (xr, ys1, 0), axis='X')
    # presence sensors (M18 barrels through the stop plate, aimed at the pipe wall below the axis) + LEDs + cables
    leds = []
    zsen = BUF_AXIS_Z - (L2.PIPE_R - L2.L.PIPE_WALL / 2)
    cm = M["cable"]
    ln = 0.075
    for i, xs in enumerate(BUF_X):
        E._rod(b, f"kit_buf_sensor{i}", 0.009, ln, M["steel"], (xs, yp0 + ln / 2, zsen), (math.pi / 2, 0, 0), segs=16)
        E._rod(b, f"kit_buf_sensor{i}_face", 0.0085, 0.002, M["black"], (xs, yp0 + 0.0005, zsen), (math.pi / 2, 0, 0), segs=16)
        _hex(b, f"kit_buf_sensor{i}_nut", 0.014, 0.005, M["steel"], (xs, ys1, zsen), (-math.pi / 2, 0, 0))
        led = E._rod(b, f"kit_buf_led{i}", 0.0065, 0.006, M["led"], (xs, yp0 + ln + 0.003, zsen), (math.pi / 2, 0, 0), segs=12)
        led["led_on"] = 1.0
        leds.append(led)
        c = P1._cable(f"kit_buf_cable{i}", [(xs, yp0 + ln + 0.006, zsen), (xs + 0.015, yp0 + ln + 0.05, zsen - 0.04),
                                            (xs + 0.04, ys1 + 0.05, top - bz - 0.02), (x0 + 0.16, ys1 + 0.03, top - bz - 0.05)],
                      b.col, cm, radius=0.0035)
        b.objs.append(c)
    # junction box behind the rear left leg + sensor cable trunk down to the floor
    xl = BUF_LEG_X[0]
    _bx(b, "kit_buf_jbox", (0.12, 0.05, 0.10), (x0 + 0.13, ys1 + 0.03, top - bz - 0.09), M["black"], bev=0.004)
    _bx(b, "kit_buf_jbox_bkt", (0.04, ys1 + 0.005 - yb1, 0.04), (x0 + 0.10, (ys1 + 0.005 + yb1) / 2, top - bz - 0.09), M["frame"])
    c = P1._cable("kit_buf_trunk", [(x0 + 0.13, ys1 + 0.03, top - bz - 0.14), (xl + 0.05, ys1 + 0.02, 0.30),
                                    (xl + 0.04, ys1 + 0.06, 0.02)], b.col, cm, radius=0.007)
    b.objs.append(c)
    # slot labels on the front beam (-Y face, readable from the track)
    for i, xs in enumerate(BUF_X):
        _label(b, f"kit_buf_label{i}", lm, 5 + i, (0.08, 0.045), (xs, BUF_BEAM_Y[0] - by / 2 - 0.0015, top - bz / 2),
               (math.pi / 2, 0, 0))
    return leds


# ============================================================================ public API
def build(collection=None):
    col = collection or bpy.data.collections.new(NAME)
    if collection is None:
        bpy.context.scene.collection.children.link(col)
    M = _mats()
    # ---- pallet + cassette + nests
    bp = E._B(col)
    _pallet(bp, M)
    lm = _cassette(bp, M)
    nests = dict(flange=[_flange_nest(bp, M, i, lm) for i in range(len(L2.KIT_FLANGES))],
                 elbow=[_elbow_nest(bp, M, i, lm) for i in range(len(L2.KIT_ELBOWS))])
    proot = G.empty("kit_pallet_root", collection=col, size=0.3)
    proot.matrix_world = G.M(kin.tr(PAL_C[0], PAL_C[1], 0.0))
    _parent_all(bp.objs, proot)
    # ---- pipe buffer
    bb = E._B(col)
    leds = _pipe_buffer(bb, M, lm)
    broot = G.empty("kit_buf_root", collection=col, size=0.3)
    broot.matrix_world = G.M(kin.tr(BUF_C[0], BUF_C[1], 0.0))
    _parent_all(bb.objs, broot)
    slots = [kin.planar_frame(*L2.pipe_buffer_frame(i)) for i in range(len(BUF_X))]
    return dict(collection=col,
                pallet=dict(root=proot, objects=bp.objs),
                nests=nests,
                pipe_buffer=dict(root=broot, objects=bb.objs, slots=slots, leds=leds))


def set_presence(cas, i, on, frame=None):
    """Presence LED of buffer slot i: on (bool / 0..1).  Keyed with CONSTANT interpolation when frame is given."""
    led = cas["pipe_buffer"]["leds"][i]
    led["led_on"] = float(on)
    if frame is not None:
        key_constant(led, '["led_on"]', frame)


def key_constant(ob, path, frame):
    """Insert a key on ob.<path> at frame and make that key CONSTANT (state channels: LEDs, beacons)."""
    ob.keyframe_insert(data_path=path, frame=frame)
    ad = ob.animation_data
    try:
        from bpy_extras import anim_utils
        fcurves = anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot).fcurves
    except Exception:
        fcurves = ad.action.fcurves
    for fc in fcurves:
        if fc.data_path == path:
            for kp in fc.keyframe_points:
                if abs(kp.co[0] - frame) < 1e-6:
                    kp.interpolation = 'CONSTANT'


def obstacles():
    """Conservative world AABBs (dict(name, center, size, yaw=0)) of the static kit / buffer geometry."""
    out = []

    def add(name, lo, hi):
        out.append(dict(name=name, center=tuple((lo[k] + hi[k]) / 2 for k in range(3)),
                        size=tuple(hi[k] - lo[k] for k in range(3)), yaw=0.0))

    cx, cy = PAL_C
    sx, sy = PAL_S
    wx, wy = CAS_SIZE
    add("kit_pallet", (cx - sx / 2, cy - sy / 2, 0.0), (cx + sx / 2, cy + sy / 2, CAS_Z + 0.001))      # pallet + plate
    t = 0.009
    zl = CAS_Z + CAS_LIP_H + 0.001
    add("kit_lip_front", (cx - wx / 2, cy - wy / 2 - 0.016, CAS_Z), (cx + wx / 2, cy - wy / 2 + t, zl))    # + ID plate, RFID
    add("kit_lip_back", (cx - wx / 2, cy + wy / 2 - t, CAS_Z), (cx + wx / 2, cy + wy / 2, zl))
    add("kit_lip_left", (cx - wx / 2, cy - wy / 2, CAS_Z), (cx - wx / 2 + t, cy + wy / 2, zl))
    add("kit_lip_right", (cx + wx / 2 - t, cy - wy / 2, CAS_Z), (cx + wx / 2, cy + wy / 2, zl))
    for k, (px, py) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
        x, y = cx + px * (wx / 2 - 0.038), cy + py * (wy / 2 - 0.038)
        add(f"kit_post{k}", (x - 0.065, y - 0.065, CAS_Z), (x + 0.065, y + 0.065, CAS_Z + CAS_POST_H + 0.055))
    for i, (o, xd) in enumerate(L2.KIT_FLANGES):
        F = kin.planar_frame(o, xd)
        h = FL_PLATE / 2 + 0.002
        add(f"kit_fnest{i}", (o[0] - h, o[1] - h, CAS_Z), (o[0] + h, o[1] + h, o[2] - GAP))       # plate + seat ring
        for k, ang in enumerate(FL_PIN_ANG):
            p = F @ np.array([FL_PIN_R * math.cos(math.radians(ang)), FL_PIN_R * math.sin(math.radians(ang)), 0.0, 1.0])
            r = FL_PIN[0] + 0.001
            add(f"kit_fnest{i}_pin{k}", (p[0] - r, p[1] - r, o[2] - GAP), (p[0] + r, p[1] + r, o[2] + FL_PIN[1]))
    for i, (o, xd) in enumerate(L2.KIT_ELBOWS):
        F = kin.planar_frame(o, xd)
        rb = EL_CUP_R[2] + 0.002
        add(f"kit_enest{i}_cup", (o[0] - rb, o[1] - rb, CAS_Z), (o[0] + rb, o[1] + rb, o[2] + EL_CUP_TOP + 0.002))
        z_top = _elbow_vrest_apex() + math.tan(EL_VREST_BETA) * EL_VREST_W
        corners = [F @ np.array([x, y, 0.0, 1.0]) for x in (EL_VREST_X[0] - 0.047, EL_VREST_X[1] + 0.047)
                   for y in (-EL_VREST_W - 0.01, EL_VREST_W + 0.01)]
        lo = [min(c[k] for c in corners) for k in range(2)]
        hi = [max(c[k] for c in corners) for k in range(2)]
        add(f"kit_enest{i}_vrest", (lo[0], lo[1], CAS_Z), (hi[0], hi[1], o[2] + z_top + 0.002))
    x0, x1 = BUF_X_RANGE
    y0, y1 = BUF_BEAM_Y
    by, bz = BUF_BEAM
    add("kit_buf_frame", (x0 - 0.02, y0 - by / 2 - 0.03, 0.0), (x1 + 0.02, y1 + by / 2 + 0.03, BUF_BEAM_TOP + 0.01))
    z_top = BUF_AXIS_Z - (L2.PIPE_R + GAP) * math.sqrt(2.0) + BUF_V_W
    for j, yb in enumerate(BUF_BEAM_Y):
        add(f"kit_buf_cradle{j}", (min(BUF_X) - BUF_V_W - 0.012, yb - by / 2 - 0.01, BUF_BEAM_TOP),
            (max(BUF_X) + BUF_V_W + 0.012, yb + by / 2 + 0.01, z_top + 0.002))
    ys0, ys1 = BUF_STOP_Y
    add("kit_buf_stop", (x0 - 0.01, ys0, BUF_STOP_Z[0] - 0.002), (x1 + 0.01, ys1 + 0.075, BUF_STOP_Z[1] + 0.005))
    for k in range(len(BUF_X) - 1):
        xb = (BUF_X[k] + BUF_X[k + 1]) / 2
        add(f"kit_buf_stop_bkt{k}", (xb - 0.022, y1 + by / 2 - 0.001, BUF_BEAM_TOP - bz - 0.001), (xb + 0.022, ys1, BUF_STOP_Z[1]))
    add("kit_buf_jbox", (x0 + 0.06, y1 + by / 2, BUF_BEAM_TOP - bz - 0.15), (x0 + 0.20, ys1 + 0.06, BUF_BEAM_TOP - bz - 0.03))
    return out
