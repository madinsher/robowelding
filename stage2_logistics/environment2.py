"""Surroundings of the stage-2 logistics zone: safety fence with light-curtain openings and a personnel door, floor
markings, zone controller cabinet + HMI pedestal, zone signs, muting lamps at the cell loading opening, extra lights.

Everything is placed from layout2 and built with the stage-1 helpers (cell.environment._B / _mats / _fence_bay / _post /
_box / _rod / _rev / _prism ...), so the new fence is indistinguishable from the stage-1 cell fence:

* fence: yellow posts + dark mesh bays around x in [LOG_FENCE_X0, CELL_FENCE_X0], y = LOG_FENCE_Y; the +-Y runs start at
  the stage-1 corner posts (-2.4, +-3.4) (those posts are NOT rebuilt).  Openings LOG_FENCE_OPENINGS get stage-1 style
  light-curtain posts (black column, red LED strip facing the opening); PERSONNEL_DOOR is a sliding door on a top
  rail outside the fence (as the stage-1 cell door, sliding toward -X) with an interlock.  A post stands under the
  stage-1 door-rail bracket at x = -2.9 (it hung in the air in stage 1); posts carrying rail brackets have no cap;
* floor: stage-1 hazard stripe continued around the outside of the new fence (same pattern phase as env_hz_n/_s),
  yellow pedestrian walkway along the -Y side (gap at the personnel door), yellow AGV lane at the -X side with white
  corner marks at the AGV park spot, yellow loading-bay marks outside the +Y openings, yellow/black hatching of the
  transfer zone in front of the cell loading opening (both sides of the handler track end), thin white outlines of
  the kit pallet bay, the storage rack and the output conveyor (outside their footprints);
* zone controller cabinet (RAL 7035 like the stage-1 IRC5) with main switch, pilot lamps and a signal tower, outside
  the -Y fence near the cell corner; HMI pedestal next to it; cable trunking + tray on the fence top to a junction box;
* zone signs (yellow plates with a black / red header band, warning triangle and text) on the outside of the fence, as
  a PIL-drawn texture atlas; the HMI screen shows a zone overview drawn the same way;
* edition (editions.py): every text drawn into a texture (signs, HMI, cabinet name plate) is looked up in i18n2 while
  build() draws it (SIGNS, HMI_BOXES etc. hold the Russian keys), each shrunk to fit its box (the boxes are public:
  sign_text_boxes, hmi_text_boxes, checked by tests/t_i18n2.py); the logo of branding2's current brand sits on a white
  plate at the right end of the HMI footer bar and as a plate on the cabinet door (CAB_LOGO_*).  Edition-dependent
  texture / material names carry the language (and the brand): env2_signs_<lang>, env2_hmi_screen_<lang>_<brand>,
  env2_label_cab_<lang>, logo_<brand>_plate;
* muting: two slim black posts just outside the stage-1 light-curtain posts of the cell loading opening
  (MUTE_POS, y = +-MUTE_POS[1]) with crossed muting sensors and a warm-white / amber lamp (object property
  'muting_on', emission keyed by set_muting).

Public API (DESIGN.md section 3, environment2.py):
    build(collection=None) -> dict(collection, objects, muting)
        objects = every object created (all prefixed env2_); muting = [lamp_s, lamp_n] (lens objects, 'muting_on')
    build_lighting(scene) -> dict(lights=[env2_light_key, env2_light_fill])
        env2_light_key: area light per layout2.LOG_KEY_LIGHT (aimed at KEY_TARGET, see there), shadowed (the only
        new shadow caster);
        env2_light_fill: weak shadowless fill of the logistics zone.  Stage-1 lights / world are not touched.
    set_muting(env, on, frame=None)        lamp property 'muting_on' (0/1), CONSTANT key when frame is given
    obstacles() -> list[dict(name, center, size, yaw)]   conservative world AABBs of the static geometry (no bpy use)
"""
import math

import numpy as np
import bpy

import tools  # noqa: F401  (demo_video on sys.path)
from cell import layout as L
from cell import materials
from cell import environment as E
import layout2 as L2
import kin
import cassette as C
import i18n2
import branding2

NAME = "Environment2"
LIGHTS_NAME = "Environment2Lights"

# ============================================================================ geometry constants
X0 = L2.LOG_FENCE_X0                  # -11.4  -X fence line
X1 = L2.CELL_FENCE_X0                 # -2.4   shared fence line with the cell (stage-1 posts)
Y0, Y1 = L2.LOG_FENCE_Y               # -3.4 / 3.4
FH = L2.LOG_FENCE_H                   # 2.1
PITCH = 1.5                           # post pitch (as stage 1)
HZ_G, HZ_W, HZ_T = 0.06, 0.12, 0.003  # stage-1 hazard stripe: gap to the fence line, width, thickness
HZ_ORIGIN_X = (L.FENCE_X[0] + L.FENCE_X[1]) / 2   # stage-1 env_hz_n/_s origin x (keeps the stripe phase continuous)
HZ_X0 = L.FENCE_X[0] - HZ_G - HZ_W   # -2.58: where the stage-1 stripes end on the -X side
PAINT_Z = 0.002                       # floor paint (single-sided quads)
WALK_W = 0.075                        # walkway / lane line width
OUTLINE_W = 0.05                      # zone outline width
DOOR = L2.PERSONNEL_DOOR
DOOR_X = (DOOR["center_x"] - DOOR["width"] / 2, DOOR["center_x"] + DOOR["width"] / 2)
DOOR_YD = Y0 - 0.10                   # door leaf plane (outside the fence, as stage 1)
DOOR_RAIL_X = (X0 - 0.05, DOOR_X[1] + 0.05)       # rail from the corner post over the door (leaf slides toward -X)
# the stage-1 cell door rail ends in a bracket at this x (door_x0 - width - 0.1), outside the cell fence
S1_DOOR_BKT_X = L.FENCE_DOOR["center_x"] - L.FENCE_DOOR["width"] / 2 - L.FENCE_DOOR["width"] - 0.1
# muting posts: just outside the stage-1 light-curtain posts (x = -2.4, y = +-1.2, base 0.2) and the stage-1 hazard
# stripe (x in [-2.58, -2.46])
MUTE_POS = (-2.68, 1.37)
MUTE_H = 1.16                         # post top; lamp above it (top 1.30: the handler gripper passes over the -Y post)
MUTE_BASE = 0.11
# transfer-zone hatching in front of the cell loading opening: both sides of the handler track end (|y| < 0.75)
TRANSFER_X = (-3.10, -2.60)
TRANSFER_Y = (0.80, 1.30)
# walkway along the -Y side (outside), AGV lane at the -X side (outside)
WALK_Y = (Y0 - 1.00, Y0 - 2.20)       # inner / outer line centres: -4.40 / -5.60
WALK_X = (X0 - 0.40, HZ_X0 - 0.04)    # -11.80 .. -2.62
AGV_LANE_X = (L2.AGV["lane_x"] - 0.70, L2.AGV["lane_x"] + 0.70)
AGV_LANE_Y = (-8.0, L2.AGV["park"][1] + 0.95)
AGV_FOOT = (0.84, 1.30)               # parked AGV footprint (x, y) at yaw 90 deg (storage.AGV_SIZE swapped)
# zone controller cabinet + HMI pedestal (outside the -Y fence, near the cell corner)
CAB_C = (-3.20, Y0 - 0.20 - 0.20)     # centre (x, y): back face 0.20 outside the fence line (clear of the stripe)
CAB_S = (0.80, 0.40, 1.80)            # body (x, y, z) on a 0.10 plinth
CAB_PLINTH = 0.10
HMI_C = (-4.30, Y0 - 0.46)            # pedestal axis
HMI_TILT = math.radians(20.0)
HMI_PX = (640, 420)                   # HMI screen texture; footer bar (status line + logo plate) = bottom HMI_FOOT px
HMI_FOOT = 58
HMI_HEAD = 50                         # header bar height (px): title left, status dot right
HMI_BOX = (160, 100)                  # material-flow box (px, 3 px outline): label 38 px under its top, status dot below
HMI_BOXES = (("КАССЕТА", 30, 80, "g"), ("СТЕНД", 240, 80, "g"), ("ЯЧЕЙКА", 450, 80, "a"),      # (Russian key, left,
             ("СКЛАД", 30, 230, "g"), ("КОНТРОЛЬ", 240, 230, "g"), ("КОНВЕЙЕР", 450, 230, "g"))  # top px, dot g/a)
HMI_LOGO_PAD = 0.12                   # logo plate margin (fraction of its height, branding2.logo_plate)
# brand plate on the cabinet door: in the column of the name plate (x = cx - 0.08), top 5.5 cm under the main-switch
# plate (bottom z = pl + sz - 0.475), a quad 1.5 mm proud of the door face like the name plate (rounded corners from
# the texture alpha, so no rectangular backing box).  The 0.11 m between the name plate and the door top are too low
# for a 0.32 m plate (ПИГРУПП: 0.16 m high)
CAB_LOGO_W = 0.32
CAB_LOGO_DX = -0.08
CAB_LOGO_TOP = CAB_PLINTH + CAB_S[2] - 0.53
CAB_LOGO_PROUD = 0.0155               # from the door box's inner face yf (door 14 mm thick)
# cable tray on the -Y fence top (outside)
TRAY_Y = Y0 - 0.20
TRAY_Z = 2.26                         # tray bottom
TRAY_W, TRAY_H = 0.20, 0.06
TRAY_X = (-7.42, -3.30)
# zone outlines: storage rack front (storage.FRONT_RAIL_X + rail-foot half width) and half length: the rack stays
# within the width of the fence opening storage_back (girder / rail ends and end frames reach |y| = y1 - 0.01), so the
# outline runs just outside that width (+2 cm for the rack's obstacle margin)
STORAGE_FRONT_X = L2.STORAGE["x"][0] + 0.86 + 0.08
_STO_OPEN = [o for o in L2.LOG_FENCE_OPENINGS if o["name"] == "storage_back"]
STORAGE_HALF_Y = (_STO_OPEN[0]["y1"] if _STO_OPEN else max(abs(y) for y in L2.STORAGE["bay_y"][0]) + 0.29) + 0.02
# lights (build_lighting).  The key uses LOG_KEY_LIGHT as it is (target 2.5 m further from the cell than the zone
# centre): EEVEE ignores the area-light spread, and aimed straight down the 900 W key brightened the cell floor near
# the loading opening by ~11 % (EEVEE, stage-1 look); tilted it is ~2-5 % and the logistics floor is more even
# (storage end brighter, zone ~ as bright as the cell).  Measured with tests/t_environment2.py --exposure.
KEY_TARGET = tuple(L2.LOG_KEY_LIGHT["target"])
KEY_SPREAD_DEG = 150.0
FILL = dict(pos=(-7.4, -6.6, 3.6), target=(-7.0, 0.3, 0.6), energy=110.0, size=7.0, size_y=3.0)
# fence signs (atlas cells)
SIGN_Z = 1.55
SIGN_SIZE = (0.42, 0.30)
SIGNS = [
    dict(text=("ОСТОРОЖНО", "ЗОНА ЛОГИСТИКИ", "ВХОД ТОЛЬКО", "ПРИ ОСТАНОВКЕ"), bg="warn"),
    dict(text=("ОПАСНО", "РАБОТАЕТ РОБОТ-", "ПОГРУЗЧИК", ""), bg="danger"),
    dict(text=("ВНИМАНИЕ", "ЗАГРУЗКА", "КАССЕТ", ""), bg="warn"),
    dict(text=("ВНИМАНИЕ", "ЗАГРУЗКА ТРУБ", "СВЕТОВАЯ", "ЗАВЕСА"), bg="warn"),
    dict(text=("ОСТОРОЖНО", "ВЫДАЧА ГОТОВЫХ", "ДВИЖЕНИЕ AGV", ""), bg="warn"),
    dict(text=("ОПАСНО", "АВТОМАТИЧЕСКИЙ", "РЕЖИМ", ""), bg="danger"),
]
SIGN_PX = (640, 457)                  # one sign of the atlas (px); black rim 10 px wide, 6 px inside the cell edge
SIGN_BAND = (16, 116)                 # header band: y range in the cell (px), x inside the rim
SIGN_TRI = (118, 290, 170)            # warning triangle: centre x, y in the cell, size (px)
SIGN_TEXT_X = 222                     # text lines: left edge (px), right of the triangle; width up to 28 px before W


# ============================================================================ fence plan (pure python)
def _gaps(side):
    g = []
    for o in L2.LOG_FENCE_OPENINGS:
        if o["side"] == side:
            g.append((o["x0"], o["x1"]) if side in ('+Y', '-Y') else (o["y0"], o["y1"]))
    if DOOR["side"] == side:
        g.append(DOOR_X)
    return sorted(g)


def _segments(a, b, gaps):
    segs, cur = [], a
    for g0, g1 in gaps:
        if g0 > cur + 1e-6:
            segs.append((cur, g0))
        cur = max(cur, g1)
    if cur < b - 1e-6:
        segs.append((cur, b))
    return segs


def _split(a, b, forced=(), avoid=()):
    """Post coordinates from a to b at ~PITCH, through the forced ones; intermediate posts inside an avoid interval are
    pushed to its nearer end (keeps post feet out of equipment next to the fence)."""
    pts = [a] + sorted(f for f in forced if a + 0.2 < f < b - 0.2) + [b]
    out = [pts[0]]
    for p, q in zip(pts[:-1], pts[1:]):
        n = max(1, round((q - p) / PITCH))
        out += [p + (q - p) * i / n for i in range(1, n + 1)]
    for i in range(1, len(out) - 1):
        for u, v in avoid:
            if u < out[i] < v:
                out[i] = u if out[i] - u < v - out[i] else v
    return out


def _avoid(side):
    """Intervals along a fence side where no intermediate post may stand: the QC arch posts are 0.15 m from the -Y
    fence line."""
    if side == '-Y':
        return [(L2.QC_ARCH["x"] - 0.45, L2.QC_ARCH["x"] + 0.45)]
    return []


def fence_runs():
    """[(name, [(x, y), ...])]: post positions of every straight fence run; bays lie between consecutive posts."""
    forced = [S1_DOOR_BKT_X] if X0 < S1_DOOR_BKT_X < X1 else []
    runs = []
    for side, y, tag in (('+Y', Y1, 'n'), ('-Y', Y0, 's')):
        for k, (a, b) in enumerate(_segments(X0, X1, _gaps(side))):
            runs.append((f"{tag}{k}", [(x, y) for x in _split(a, b, forced if side == '-Y' else (), _avoid(side))]))
    for k, (a, b) in enumerate(_segments(Y0, Y1, _gaps('-X'))):
        runs.append((f"w{k}", [(X0, y) for y in _split(a, b)]))
    return runs


def _is_stage1_post(p):
    return abs(p[0] - X1) < 1e-6 and (abs(p[1] - Y0) < 1e-6 or abs(p[1] - Y1) < 1e-6)


def fence_posts():
    """Unique new post positions (the stage-1 corner posts at (X1, +-3.4) excluded)."""
    seen, out = set(), []
    for _, pts in fence_runs():
        for p in pts:
            key = (round(p[0], 4), round(p[1], 4))
            if key not in seen and not _is_stage1_post(p):
                seen.add(key)
                out.append(p)
    return out


def curtain_posts():
    """[(name, (x, y), (fx, fy))]: light-curtain columns at the opening edges, (fx, fy) = unit vector into the opening."""
    out = []
    for o in L2.LOG_FENCE_OPENINGS:
        if not o.get("curtain"):
            continue
        if o["side"] in ('+Y', '-Y'):
            y = Y1 if o["side"] == '+Y' else Y0
            out += [(f"{o['name']}_0", (o["x0"], y), (1.0, 0.0)), (f"{o['name']}_1", (o["x1"], y), (-1.0, 0.0))]
        else:
            x = X0 if o["side"] == '-X' else X1
            out += [(f"{o['name']}_0", (x, o["y0"]), (0.0, 1.0)), (f"{o['name']}_1", (x, o["y1"]), (0.0, -1.0))]
    return out


def _capless():
    """Posts whose top carries a rail bracket (bracket = cap): the stage-1 door-rail bracket and the new door rail."""
    pts = [(S1_DOOR_BKT_X, Y0)]
    pts += [(x, Y0) for x in (X0, DOOR_X[0], DOOR_X[1])]
    return pts


def _tray_posts():
    return [p for p in fence_posts() if abs(p[1] - Y0) < 1e-6 and TRAY_X[0] + 0.05 < p[0] < TRAY_X[1] - 0.05]


def _sign_spots():
    """[(sign index, (x, y), facing)] sign centres on bay mid-points (outside face), facing = outward normal."""
    runs = dict(fence_runs())

    def bay_mid(run, x_near=None, y_near=None):
        pts = runs[run]
        mids = [((p[0] + q[0]) / 2, (p[1] + q[1]) / 2) for p, q in zip(pts[:-1], pts[1:])]
        key = (lambda m: abs(m[0] - x_near)) if x_near is not None else (lambda m: abs(m[1] - y_near))
        return min(mids, key=key)

    s_runs = sorted([r for r in runs if r.startswith('s')], key=lambda r: runs[r][0][0])
    n_runs = sorted([r for r in runs if r.startswith('n')], key=lambda r: runs[r][0][0])
    w_runs = sorted([r for r in runs if r.startswith('w')], key=lambda r: runs[r][0][1])
    out = []
    # door sign: first bay +X of the door on the -Y side; robot sign mid -Y; kit / pipe signs next to the openings
    out.append((0, bay_mid(s_runs[-1], x_near=DOOR_X[1] + 0.9), '-Y'))
    out.append((1, bay_mid(s_runs[-1], x_near=-6.0), '-Y'))
    kit = [o for o in L2.LOG_FENCE_OPENINGS if o["name"] == "kit_pallet_bay"]
    pipe = [o for o in L2.LOG_FENCE_OPENINGS if o["name"] == "pipe_buffer_bay"]
    if kit:
        out.append((2, bay_mid(n_runs[0], x_near=kit[0]["x0"] - 0.8), '+Y'))
    if pipe:
        out.append((3, bay_mid(n_runs[-1], x_near=pipe[0]["x1"] + 0.6), '+Y'))
    out.append((4, bay_mid(w_runs[0], y_near=Y0 + 0.6), '-X'))
    out.append((5, bay_mid(n_runs[-1], x_near=X1 - 0.6), '+Y'))
    return out


# ============================================================================ materials
def _paint_mat(key, color, rough=0.62):
    """Floor paint: flat colour with worn patches (world-position noise) toward the dusty floor colour + fine scuffs."""
    m = bpy.data.materials.get(key)
    if m is not None:
        return m
    m = bpy.data.materials.new(key)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Specular IOR Level"].default_value = 0.3
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    for scale, lo, gain, amount, tint in ((2.5, 0.52, 4.0, 0.55, (0.12, 0.11, 0.095)),
                                          (22.0, 0.58, 6.0, 0.35, (0.10, 0.095, 0.085))):
        n = nt.nodes.new("ShaderNodeTexNoise")
        n.inputs["Scale"].default_value = scale
        n.inputs["Detail"].default_value = 5.0
        nt.links.new(geo.outputs["Position"], n.inputs["Vector"])
        mask = E._math(nt, 'MULTIPLY', E._math(nt, 'MULTIPLY', E._math(nt, 'SUBTRACT', n.outputs["Fac"], lo), gain, clamp=True), amount)
        E._tint_base_color(nt, b, mask, tint)
    return m


def _lamp_mat():
    """Muting lamp lens: warm-white / amber translucent plastic, emission = property 'muting_on' x 7."""
    key = "env2_muting_lamp"
    m = bpy.data.materials.get(key)
    if m is not None:
        return m
    m = C._state_mat(key, (1.0, 0.56, 0.16), 7.0, "muting_on")
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.55, 0.40, 0.20, 1.0)
    b.inputs["Roughness"].default_value = 0.22
    b.inputs["Coat Weight"].default_value = 0.4
    return m


def _image_mat(key, img, emission=0.0, rough=0.45):
    """Material from a PIL RGB image (packed); emission > 0 makes it a lit screen."""
    m = bpy.data.materials.get(key)
    if m is not None:
        return m
    arr = np.asarray(img.convert("RGBA"), dtype=np.float32)[::-1] / 255.0
    bi = bpy.data.images.new(key, img.width, img.height, alpha=True)
    bi.colorspace_settings.name = 'sRGB'
    bi.pixels.foreach_set(arr.ravel())
    bi.pack()
    m = bpy.data.materials.new(key)
    m.use_nodes = True
    nt = m.node_tree
    bs = nt.nodes["Principled BSDF"]
    bs.inputs["Roughness"].default_value = rough
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bi
    tex.interpolation = 'Linear'
    uv = nt.nodes.new("ShaderNodeUVMap")
    nt.links.new(uv.outputs["UV"], tex.inputs["Vector"])
    if emission > 0:
        bs.inputs["Base Color"].default_value = (0.01, 0.012, 0.02, 1.0)
        nt.links.new(tex.outputs["Color"], bs.inputs["Emission Color"])
        bs.inputs["Emission Strength"].default_value = emission
    else:
        nt.links.new(tex.outputs["Color"], bs.inputs["Base Color"])
    return m


def _font(size):
    from PIL import ImageFont
    return ImageFont.truetype(C._FONT, size)


def _fit(d, text, w, size):
    f = _font(size)
    while f.getlength(text) > w and size > 10:
        size -= 2
        f = _font(size)
    return f


def _sign_atlas_mat():
    """Vertical atlas of warning signs (SIGNS): yellow plate, black rim, red (or black) band with a white header,
    black warning triangle, black text lines (current language, i18n2.tr_lines).  Cell k is mapped by C._label()."""
    key = f"env2_signs_{i18n2.get_lang()}"
    m = bpy.data.materials.get(key)
    if m is not None:
        return m
    from PIL import Image, ImageDraw
    W, H = SIGN_PX
    img = Image.new("RGB", (W, H * len(SIGNS)), (242, 180, 0))
    d = ImageDraw.Draw(img)
    for k, s in enumerate(SIGNS):
        y0 = k * H
        text = i18n2.tr_lines(s["text"])
        d.rectangle((6, y0 + 6, W - 7, y0 + H - 7), outline=(20, 20, 20), width=10)
        band = (200, 16, 46) if s["bg"] == "danger" else (20, 20, 20)
        d.rectangle((16, y0 + SIGN_BAND[0], W - 17, y0 + SIGN_BAND[1]), fill=band)
        f = _fit(d, text[0], W - 80, 72)
        d.text((W / 2, y0 + sum(SIGN_BAND) / 2), text[0], font=f, fill=(255, 255, 255), anchor="mm")
        # warning triangle with an exclamation mark
        tx, ty, ts = SIGN_TRI[0], y0 + SIGN_TRI[1], SIGN_TRI[2]
        tri = [(tx, ty - ts * 0.58), (tx - ts / 2, ty + ts * 0.29), (tx + ts / 2, ty + ts * 0.29)]
        d.polygon(tri, fill=(20, 20, 20))
        inner = [(tx, ty - ts * 0.40), (tx - ts * 0.34, ty + ts * 0.20), (tx + ts * 0.34, ty + ts * 0.20)]
        d.polygon(inner, fill=(242, 180, 0))
        d.rectangle((tx - 8, ty - 38, tx + 8, ty + 6), fill=(20, 20, 20))
        d.rectangle((tx - 8, ty + 16, tx + 8, ty + 30), fill=(20, 20, 20))
        lines = [t for t in text[1:] if t]
        size = 58
        fonts = [_fit(d, t, W - SIGN_TEXT_X - 28, size) for t in lines]
        size = min(f.size for f in fonts) if fonts else size
        f = _font(size)
        n = len(lines)
        for j, t in enumerate(lines):
            yy = ty + (j - (n - 1) / 2) * size * 1.18
            d.text((SIGN_TEXT_X, yy), t, font=f, fill=(20, 20, 20), anchor="lm")
    m = _image_mat(key, img, rough=0.5)
    m["n_cells"] = len(SIGNS)
    return m


def sign_text_boxes(k=0):
    """Boxes (x0, y0, x1, y1) px the texts of sign k stay inside (tests/t_i18n2.py checks every drawn text): the
    header band and the body right of the warning triangle, both inside the black rim."""
    W, H = SIGN_PX
    y0 = k * H
    tx, _, ts = SIGN_TRI
    return [(16, y0 + SIGN_BAND[0], W - 16, y0 + SIGN_BAND[1]), (tx + ts / 2, y0 + SIGN_BAND[1], W - 16, y0 + H - 16)]


def hmi_text_boxes():
    """Boxes (x0, y0, x1, y1) px the texts of the HMI screen stay inside (tests/t_i18n2.py checks every drawn text):
    header bar left of the status dot, the label band of every flow box (inside its outline, above the status dot),
    footer bar left of the logo plate."""
    W, H = HMI_PX
    bw = HMI_BOX[0]
    return ([(0, 0, W - 46, HMI_HEAD)] + [(x + 3, y + 3, x + bw - 3, y + 62) for _, x, y, _ in HMI_BOXES]
            + [(0, H - HMI_FOOT, hmi_logo_box()[0], H)])


def hmi_logo_box():
    """(x0, y0, x1, y1) px of the logo plate on the HMI texture: right end of the footer bar, 8 px inside it."""
    W, H = HMI_PX
    ph = HMI_FOOT - 16
    pw = ph * branding2.logo_aspect(pad=HMI_LOGO_PAD)
    return (W - 12 - pw, H - 8 - ph, W - 12, H - 8)


def _hmi_mat():
    """HMI screen: zone overview (material flow boxes with status dots), header and status line (current language,
    i18n2), brand logo plate at the right end of the footer bar; emissive."""
    key = f"env2_hmi_screen_{i18n2.get_lang()}_{branding2.get_brand()}"
    m = bpy.data.materials.get(key)
    if m is not None:
        return m
    from PIL import Image, ImageDraw
    tr = i18n2.tr
    W, H = HMI_PX
    img = Image.new("RGB", (W, H), (12, 20, 34))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, W, HMI_HEAD), fill=(26, 72, 140))
    t = tr("ЗОНА ЛОГИСТИКИ · АВТО")
    d.text((16, HMI_HEAD / 2), t, font=_fit(d, t, W - 76, 26), fill=(240, 244, 250), anchor="lm")   # clear of the dot
    d.ellipse((W - 46, 13, W - 20, 39), fill=(46, 204, 64))
    bw, bh = HMI_BOX
    for t, x, y, st in HMI_BOXES:
        t = tr(t)
        d.rectangle((x, y, x + bw, y + bh), fill=(20, 40, 70), outline=(90, 170, 240), width=3)
        d.text((x + bw / 2, y + 38), t, font=_fit(d, t, bw - 20, 24), fill=(220, 232, 245), anchor="mm")
        c = (46, 204, 64) if st == "g" else (255, 170, 20)
        d.ellipse((x + 68, y + 62, x + 92, y + 86), fill=c)
    for x0, y0, x1, y1 in ((190, 130, 240, 130), (400, 130, 450, 130), (530, 180, 530, 230), (450, 280, 400, 280),
                           (240, 280, 190, 280)):
        d.line((x0, y0, x1, y1), fill=(240, 244, 250), width=5)
        ang = math.atan2(y1 - y0, x1 - x0)
        a1, a2 = ang + math.radians(150), ang - math.radians(150)
        d.polygon([(x1, y1), (x1 + 16 * math.cos(a1), y1 + 16 * math.sin(a1)), (x1 + 16 * math.cos(a2), y1 + 16 * math.sin(a2))],
                  fill=(240, 244, 250))
    d.rectangle((0, H - HMI_FOOT, W, H), fill=(22, 30, 46))
    lb = hmi_logo_box()
    t = tr("SPL-02   ТАКТ 06:40   ЗАВЕСА: MUTING")
    d.text((16, H - 29), t, font=_fit(d, t, lb[0] - 32, 22), fill=(200, 212, 228), anchor="lm")    # 16 px before the logo
    branding2.paste_logo(img, lb, plate=True, pad=HMI_LOGO_PAD)
    return _image_mat(key, img, emission=1.3, rough=0.15)


def _mats():
    M = E._mats()
    M.update(
        paint_yellow=_paint_mat("env2_paint_yellow", materials._srgb("#E2A800")),
        paint_white=_paint_mat("env2_paint_white", (0.50, 0.50, 0.48)),
        lamp=_lamp_mat(),
        signs=_sign_atlas_mat(),
        hmi=_hmi_mat(),
        amber=materials.get("painted", color="#C87800", roughness=0.3, coat=0.3),
        red_lens=materials.get("painted", color="#7A0E0E", roughness=0.3, coat=0.3),
        frame=materials.get("painted", color="#2B2F36", roughness=0.5),
        label_cab=C._label_mat(f"env2_label_cab_{i18n2.get_lang()}", [i18n2.tr("ШУ ЗОНЫ ЛОГИСТИКИ  =LZ1+CP01")], cell=(900, 96),
                               fg=(235, 235, 235), bg=(40, 42, 46)),
        label_mute=C._label_mat("env2_label_mute", ["MUTING"], cell=(320, 96)),
        label_jb=C._label_mat("env2_label_jb", ["=LZ1+JB02"], cell=(320, 96), fg=(235, 235, 235), bg=(40, 42, 46)),
    )
    return M


# ============================================================================ fence
def _fence(b, M):
    capless = _capless()
    for rn, pts in fence_runs():
        for i in range(len(pts) - 1):
            E._fence_bay(b, M, f"env2_fbay_{rn}_{i}", pts[i], pts[i + 1])
    for k, (x, y) in enumerate(fence_posts()):
        name = f"env2_fpost_{k}"
        if any(abs(x - cx) < 1e-6 and abs(y - cy) < 1e-6 for cx, cy in capless):
            E._box(b, name, (0.06, 0.06, FH), (x, y, FH / 2), M["yellow"])
            E._box(b, name + "_plate", (0.16, 0.16, 0.012), (x, y, 0.006), M["dark"])
        else:
            E._post(b, M, name, x, y)
    # light-curtain columns at the openings (stage-1 env_lc_* look: black column, base, red LED strip toward the opening)
    for name, (x, y), (fx, fy) in curtain_posts():
        n = f"env2_lc_{name}"
        E._box(b, n + "_post", (0.08, 0.08, 1.85), (x, y, 0.925 + 0.03), M["black"])
        E._box(b, n + "_base", (0.2, 0.2, 0.03), (x, y, 0.015), M["dark"])
        strip = (0.004, 0.012, 1.5) if abs(fx) > 0.5 else (0.012, 0.004, 1.5)
        E._box(b, n + "_strip", strip, (x + fx * 0.042, y + fy * 0.042, 1.1), M["red_led"])
        led = (0.004, 0.02, 0.02) if abs(fx) > 0.5 else (0.02, 0.004, 0.02)
        E._box(b, n + "_led", led, (x + fx * 0.042, y + fy * 0.042, 1.86), M["green_led"])
        E._box(b, n + "_cap", (0.085, 0.085, 0.012), (x, y, 1.886), M["dark"])
    _door(b, M)
    _hazard_stripes(b, M)


def _door(b, M):
    """Sliding personnel door on a top rail outside the -Y fence (stage-1 cell door style), leaf slides toward -X."""
    dx0, dx1 = DOOR_X
    dw = dx1 - dx0
    yd = DOOR_YD
    ra, rb = DOOR_RAIL_X
    E._box(b, "env2_door_rail", (rb - ra, 0.06, 0.08), ((ra + rb) / 2, yd, FH + 0.06), M["yellow"])
    for k, x in enumerate((X0, dx0, dx1)):
        E._box(b, f"env2_door_rail_bkt{k}", (0.06, 0.14, 0.06), (x, Y0 - 0.05, FH + 0.03), M["dark"])
    for k, x in enumerate((dx0 + 0.03, dx1 - 0.03)):
        E._box(b, f"env2_door_stile_{k}", (0.06, 0.06, FH - 0.1), (x, yd, (FH - 0.1) / 2 + 0.05), M["yellow"])
    for k, z in enumerate((0.08, FH - 0.08)):
        E._box(b, f"env2_door_rail_{k}", (dw, 0.06, 0.06), ((dx0 + dx1) / 2, yd, z), M["yellow"])
    for k, x in enumerate((dx0 + 0.25, dx1 - 0.25)):
        E._rod(b, f"env2_door_roller_{k}", 0.05, 0.03, M["dark"], (x, yd, FH + 0.06), (math.pi / 2, 0, 0), 16)
    v = [(0, 0, 0), (dw - 0.12, 0, 0), (dw - 0.12, FH - 0.22, 0), (0, FH - 0.22, 0)]
    E._mesh(b, "env2_door_mesh", v, [(0, 1, 2, 3)], M["fence"], (dx0 + 0.06, yd, 0.11), (math.pi / 2, 0, 0))
    E._box(b, "env2_door_handle", (0.03, 0.05, 0.3), (dx1 - 0.12, yd - 0.06, 1.05), M["dark"])
    for k, z in enumerate((0.93, 1.17)):
        E._box(b, f"env2_door_handle_leg{k}", (0.03, 0.03, 0.03), (dx1 - 0.12, yd - 0.035, z), M["dark"])
    # interlock switch on the door-edge post (+X side), actuator on the leaf, status LED
    E._box(b, "env2_interlock", (0.09, 0.12, 0.2), (dx1 + 0.08, Y0 - 0.08, 1.2), M["yellow"])
    E._box(b, "env2_interlock_act", (0.06, 0.04, 0.05), (dx1 - 0.05, yd - 0.02, 1.2), M["dark"])
    E._box(b, "env2_interlock_led", (0.02, 0.005, 0.02), (dx1 + 0.08, Y0 - 0.145, 1.27), M["green_led"])
    # request-to-enter push button box beside the interlock
    E._box(b, "env2_door_reqbox", (0.08, 0.06, 0.12), (dx1 + 0.08, Y0 - 0.05, 0.98), M["yellow"])
    E._rod(b, "env2_door_reqbtn", 0.014, 0.02, M["green_btn"], (dx1 + 0.08, Y0 - 0.085, 0.99), (math.pi / 2, 0, 0), 12)


def _hazard_stripes(b, M):
    """Stage-1 yellow/black floor stripe continued around the outside of the new fence (3 mm boxes as env_hz_*)."""
    ya = Y1 + HZ_G + HZ_W / 2
    xa, xb = X0 - HZ_G - HZ_W, HZ_X0
    for name, yc in (("env2_hz_n", ya), ("env2_hz_s", -ya)):
        # same object origin x as the stage-1 stripes so the 45-degree pattern continues without a phase jump
        ox = HZ_ORIGIN_X
        v = [(xa - ox, -HZ_W / 2, -HZ_T / 2), (xb - ox, -HZ_W / 2, -HZ_T / 2), (xb - ox, HZ_W / 2, -HZ_T / 2), (xa - ox, HZ_W / 2, -HZ_T / 2),
             (xa - ox, -HZ_W / 2, HZ_T / 2), (xb - ox, -HZ_W / 2, HZ_T / 2), (xb - ox, HZ_W / 2, HZ_T / 2), (xa - ox, HZ_W / 2, HZ_T / 2)]
        f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        E._mesh(b, name, v, f, M["hazard"], (ox, yc, HZ_T / 2))
    E._box(b, "env2_hz_w", (HZ_W, Y1 - Y0 + 2 * HZ_G, HZ_T), (X0 - HZ_G - HZ_W / 2, 0.0, HZ_T / 2), M["hazard"])


# ============================================================================ floor markings
def _rect(x0, y0, x1, y1):
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _hline(x0, x1, y, w):
    return _rect(x0, y - w / 2, x1, y + w / 2)


def _vline(y0, y1, x, w):
    return _rect(x - w / 2, y0, x + w / 2, y1)


def paint_groups():
    """{object name: (material key, [rects (x0, y0, x1, y1)])} of the floor paint (pure python, also for obstacles)."""
    w, o = WALK_W, OUTLINE_W
    g = {}
    # ---- pedestrian walkway along the -Y side (outside): inner line broken at the personnel door, closed -X end
    yi, yo = WALK_Y
    xa, xb = WALK_X
    walk = [_hline(xa, DOOR_X[0] - 0.10, yi, w), _hline(DOOR_X[1] + 0.10, xb, yi, w), _hline(xa, xb, yo, w),
            _vline(yo - w / 2, yi + w / 2, xa + w / 2, w)]
    # short door approach lines from the walkway to the stripe
    for x in (DOOR_X[0] - 0.10, DOOR_X[1] + 0.10):
        walk.append(_vline(Y0 - HZ_G - HZ_W - 0.04, yi - w / 2, x, w))
    g["env2_paint_walkway"] = ("paint_yellow", walk)
    # ---- AGV lane at the -X side (outside) with a stop line beyond the park spot
    lx0, lx1 = AGV_LANE_X
    ly0, ly1 = AGV_LANE_Y
    g["env2_paint_agv_lane"] = ("paint_yellow", [_vline(ly0, ly1, lx0, w), _vline(ly0, ly1, lx1, w),
                                                 _hline(lx0 - w / 2, lx1 + w / 2, ly1 - w / 2, w)])
    # AGV park spot: white corner marks just outside the parked footprint (+ 3 cm margin of storage.obstacles)
    px, py = L2.AGV["park"]
    hx, hy = AGV_FOOT[0] / 2 + 0.05 + o, AGV_FOOT[1] / 2 + 0.05 + o
    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            cx, cy = px + sx * hx, py + sy * hy
            corners.append(_rect(cx, cy, cx - sx * 0.28, cy - sy * o))
            corners.append(_rect(cx, cy - sy * o, cx - sx * o, cy - sy * 0.28))
    g["env2_paint_agv_park"] = ("paint_white", corners)
    # ---- loading bays outside the +Y openings (U open toward the fence)
    bays = []
    yb0, yb1 = Y1 + HZ_G + HZ_W + 0.04, Y1 + 0.75
    for op in L2.LOG_FENCE_OPENINGS:
        if op["side"] != '+Y':
            continue
        xa_, xb_ = op["x0"] + 0.06, op["x1"] - 0.06
        bays += [_vline(yb0, yb1, xa_, w), _vline(yb0, yb1, xb_, w), _hline(xa_ - w / 2, xb_ + w / 2, yb1, w)]
    g["env2_paint_bays"] = ("paint_yellow", bays)
    # ---- zone outlines inside the fence (white, outside the equipment footprints)
    zones = []
    kc, ks = L2.KIT_PALLET["center"], L2.KIT_PALLET["size"]
    kx0, kx1 = kc[0] - ks[0] / 2 - 0.09, kc[0] + ks[0] / 2 + 0.09
    ky0, ky1 = kc[1] - ks[1] / 2 - 0.09, Y1 - 0.14
    zones += [_vline(ky0, ky1, kx0, o), _vline(ky0, ky1, kx1, o), _hline(kx0 - o / 2, kx1 + o / 2, ky0, o)]
    sx0, sx1 = X0 + 0.12, STORAGE_FRONT_X + 0.08      # -X end clear of the storage_back light-curtain base plates
    sy = STORAGE_HALF_Y + 0.03 + o / 2
    zones += [_hline(sx0, sx1, sy, o), _hline(sx0, sx1, -sy, o), _vline(-sy - o / 2, sy + o / 2, sx1, o)]
    cv = L2.CONVEYOR
    cx0, cx1 = cv["x0"] - 0.15, cv["x1"] + 0.15
    cy_in = cv["y"] + cv["width"] / 2 + 0.13
    cy_out = Y0 + 0.14
    qx = L2.QC_ARCH["x"]
    zones += [_hline(cx0, qx - 0.45, cy_in, o), _hline(qx + 0.45, cx1, cy_in, o),
              _vline(cy_out, cy_in + o / 2, cx0, o), _vline(cy_out, cy_in + o / 2, cx1, o)]
    g["env2_paint_zones"] = ("paint_white", zones)
    return g


def hatch_rects():
    """Transfer-zone hatching pads (x0, y0, x1, y1) either side of the handler track end."""
    (xa, xb), (ya, yb) = TRANSFER_X, TRANSFER_Y
    return [(xa, ya, xb, yb), (xa, -yb, xb, -ya)]


def _paint_mesh(b, name, rects, mat, z=PAINT_Z, origin=None):
    ox, oy = origin or (sum((r[0] + r[2]) / 2 for r in rects) / len(rects), sum((r[1] + r[3]) / 2 for r in rects) / len(rects))
    verts, faces = [], []
    for x0, y0, x1, y1 in rects:
        k = len(verts)
        verts += [(x0 - ox, y0 - oy, 0.0), (x1 - ox, y0 - oy, 0.0), (x1 - ox, y1 - oy, 0.0), (x0 - ox, y1 - oy, 0.0)]
        faces.append((k, k + 1, k + 2, k + 3))
    return E._mesh(b, name, verts, faces, mat, (ox, oy, z))


def _floor(b, M):
    for name, (mk, rects) in paint_groups().items():
        if rects:
            _paint_mesh(b, name, rects, M[mk])
    hr = hatch_rects()
    _paint_mesh(b, "env2_hatch_transfer", hr, M["hazard"], z=PAINT_Z + 0.0005)
    # yellow border around each hatch pad
    w = 0.04
    border = []
    for x0, y0, x1, y1 in hr:
        border += [_rect(x0, y0, x1, y0 + w), _rect(x0, y1 - w, x1, y1), _rect(x0, y0 + w, x0 + w, y1 - w), _rect(x1 - w, y0 + w, x1, y1 - w)]
    _paint_mesh(b, "env2_hatch_border", border, M["paint_yellow"], z=PAINT_Z + 0.001)


# ============================================================================ muting lamps
def _muting(b, M):
    lamps = []
    mx, my = MUTE_POS
    for k, s in enumerate((-1, 1)):
        y = s * my
        p = f"env2_mute{k}"
        C._bx(b, p + "_base", (MUTE_BASE, MUTE_BASE, 0.012), (mx, y, 0.006), M["dark"], bev=0.002)
        for j, (dx, dy) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
            C._hex(b, f"{p}_anchor{j}", 0.008, 0.008, M["steel"], (mx + dx * 0.04, y + dy * 0.04, 0.012))
        C._bx(b, p + "_post", (0.05, 0.05, MUTE_H - 0.012), (mx, y, 0.012 + (MUTE_H - 0.012) / 2), M["black"], bev=0.004)
        # two muting sensors crossing in front of the curtain (angled +-20 deg toward the track axis), with lenses
        for j, (z, a) in enumerate(((0.42, 0.35), (0.62, -0.35))):
            yaw = -s * math.pi / 2 + a            # face toward the opening (-s * y)
            fx, fy = math.cos(yaw), math.sin(yaw)
            c = (mx + fx * 0.045, y + fy * 0.045, z)
            E._box(b, f"{p}_sensor{j}_bkt", (0.05, 0.056, 0.012), (mx + fx * 0.012, y + fy * 0.012, z - 0.034), M["dark"], (0, 0, yaw))
            C._bx(b, f"{p}_sensor{j}", (0.04, 0.03, 0.055), c, M["black"], (0, 0, yaw), bev=0.003)
            E._rod(b, f"{p}_sensor{j}_lens", 0.009, 0.004, M["glass"], (c[0] + fx * 0.021, c[1] + fy * 0.021, z + 0.008),
                   (0, math.pi / 2, yaw), 12)
            E._box(b, f"{p}_sensor{j}_led", (0.004, 0.008, 0.006), (c[0] + fx * 0.021, c[1] + fy * 0.021, z - 0.016),
                   M["green_led"], (0, 0, yaw))
        # label plate facing -X (toward the logistics zone)
        C._label(b, p + "_label", M["label_mute"], 0, (0.09, 0.027), (mx - 0.0265, y, MUTE_H - 0.22), (math.pi / 2, 0, -math.pi / 2),
                 backing=M["black"])
        # lamp: black base, warm-white/amber lens (keyed), black cap
        z = MUTE_H
        E._rev(b, p + "_lamp_base", [(0, 0), (0.034, 0), (0.034, 0.028), (0.030, 0.034), (0, 0.034)], M["black"], (mx, y, z), segs=24)
        lens = E._rev(b, p + "_lamp", [(0, 0), (0.031, 0), (0.031, 0.085), (0, 0.085)], M["lamp"], (mx, y, z + 0.034), segs=24)
        E._rev(b, p + "_lamp_cap", [(0, 0), (0.033, 0), (0.033, 0.010), (0.025, 0.016), (0, 0.018)], M["black"], (mx, y, z + 0.119), segs=24)
        lens["muting_on"] = 0.0
        lamps.append(lens)
    return lamps


# ============================================================================ zone controller cabinet, HMI, cable tray
def _cabinet(b, M):
    (cx, cy), (sx, sy, sz), pl = CAB_C, CAB_S, CAB_PLINTH
    grey, dark = M["cabinet"], M["cabinet_dark"]
    yf = cy - sy / 2                      # door face (-Y)
    C._bx(b, "env2_cab_plinth", (sx, sy, pl), (cx, cy, pl / 2), M["black"], bev=0.004)
    C._bx(b, "env2_cab_body", (sx, sy, sz), (cx, cy, pl + sz / 2), grey, bev=0.006)
    C._bx(b, "env2_cab_roof", (sx + 0.02, sy + 0.02, 0.025), (cx, cy, pl + sz + 0.0125), dark, bev=0.004)
    C._bx(b, "env2_cab_door", (sx - 0.05, 0.014, sz - 0.07), (cx, yf - 0.007, pl + sz / 2), grey, bev=0.004)
    E._box(b, "env2_cab_door_gap", (sx - 0.04, 0.004, sz - 0.06), (cx, yf - 0.001, pl + sz / 2), dark)
    for k, z in enumerate((pl + 0.35, pl + sz - 0.35)):     # hinges on the -X edge
        E._rod(b, f"env2_cab_hinge{k}", 0.009, 0.07, M["dark"], (cx - sx / 2 + 0.03, yf - 0.016, z), segs=10)
    # swing handle + lock, main switch (yellow plate + red rotary handle), pilot lamps, name plate
    xh = cx + sx / 2 - 0.07
    C._bx(b, "env2_cab_handle_base", (0.04, 0.012, 0.18), (xh, yf - 0.020, pl + sz / 2), M["dark"], bev=0.003)
    C._bx(b, "env2_cab_handle", (0.024, 0.03, 0.12), (xh, yf - 0.040, pl + sz / 2 - 0.01), M["black"], bev=0.004)
    xm, zm = cx + 0.14, pl + sz - 0.42
    E._box(b, "env2_cab_mainsw_plate", (0.11, 0.006, 0.11), (xm, yf - 0.017, zm), M["yellow"])
    E._rod(b, "env2_cab_mainsw_hub", 0.028, 0.02, M["red"], (xm, yf - 0.03, zm), (math.pi / 2, 0, 0), 20)
    C._bx(b, "env2_cab_mainsw_lever", (0.024, 0.022, 0.10), (xm, yf - 0.045, zm), M["red"], (0, 0.35, 0), bev=0.004)
    for k, (dx, mat) in enumerate(((-0.20, M["green_led"]), (-0.12, M["white_btn"]), (-0.04, M["red_lens"]))):
        E._rod(b, f"env2_cab_pilot{k}_ring", 0.018, 0.008, M["dark"], (cx + dx, yf - 0.018, pl + sz - 0.30), (math.pi / 2, 0, 0), 16)
        E._rod(b, f"env2_cab_pilot{k}", 0.013, 0.016, mat, (cx + dx, yf - 0.026, pl + sz - 0.30), (math.pi / 2, 0, 0), 16)
    C._label(b, "env2_cab_label", M["label_cab"], 0, (0.40, 0.043), (cx - 0.08, yf - 0.0155, pl + sz - 0.17), (math.pi / 2, 0, 0),
             backing=None)
    # brand plate (branding2's current brand) facing -Y (local +Z of the quad -> world -Y: rotx(+90 deg))
    lh = CAB_LOGO_W / branding2.logo_aspect()
    lg = branding2.logo_plate_object("env2_cab_logo", CAB_LOGO_W,
                                     kin.tr(cx + CAB_LOGO_DX, yf - CAB_LOGO_PROUD, CAB_LOGO_TOP - lh / 2) @ kin.rotx(math.pi / 2),
                                     collection=b.col)
    b.objs += [lg] + list(lg.children)
    # ventilation: fan-filter grilles on the door (bottom) and the +X side (top)
    for k, (c, rot, size) in enumerate((((cx - 0.12, yf - 0.02, pl + 0.28), (0, 0, 0), (0.20, 0.012, 0.20)),
                                         ((cx + sx / 2 + 0.006, cy, pl + sz - 0.35), (0, 0, math.pi / 2), (0.20, 0.012, 0.20)))):
        C._bx(b, f"env2_cab_fan{k}", size, c, dark, rot, bev=0.003)
        for j in range(5):
            off = -0.07 + 0.035 * j
            if k == 0:
                E._box(b, f"env2_cab_fan{k}_slat{j}", (0.17, 0.006, 0.012), (c[0], c[1] - 0.008, c[2] + off), M["black"])
            else:
                E._box(b, f"env2_cab_fan{k}_slat{j}", (0.006, 0.17, 0.012), (c[0] + 0.008, c[1], c[2] + off), M["black"])
    # signal tower on the roof: base, green (lit) / amber / red segments, cap
    tx, ty, tz = cx + sx / 2 - 0.10, cy + 0.06, pl + sz + 0.025
    E._rev(b, "env2_cab_tower_base", [(0, 0), (0.032, 0), (0.032, 0.03), (0.02, 0.04), (0.012, 0.04), (0.012, 0.12), (0, 0.12)],
           M["black"], (tx, ty, tz), segs=20)
    for k, mat in enumerate((M["green_led"], M["amber"], M["red_lens"])):
        E._rev(b, f"env2_cab_tower_seg{k}", [(0, 0), (0.030, 0), (0.030, 0.058), (0, 0.058)], mat, (tx, ty, tz + 0.12 + 0.060 * k), segs=20)
    E._rev(b, "env2_cab_tower_cap", [(0, 0), (0.031, 0), (0.031, 0.012), (0.02, 0.02), (0, 0.022)], M["black"], (tx, ty, tz + 0.30), segs=20)
    # cable trunking from the roof up into the fence-top tray
    zt = pl + sz + 0.025
    C._bx(b, "env2_cab_trunk", (0.14, 0.08, TRAY_Z + 0.03 - zt), (TRAY_X[1] - 0.10, cy + sy / 2 - 0.06, (TRAY_Z + 0.03 + zt) / 2),
          M["galv"], bev=0.004)


def _hmi(b, M):
    x, y = HMI_C
    C._bx(b, "env2_hmi_foot", (0.34, 0.30, 0.016), (x, y, 0.008), M["dark"], bev=0.003)
    for j, (dx, dy) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
        C._hex(b, f"env2_hmi_anchor{j}", 0.01, 0.01, M["steel"], (x + dx * 0.13, y + dy * 0.11, 0.016))
    zc = 1.02
    E._rod(b, "env2_hmi_column", 0.045, zc - 0.016, M["cabinet_dark"], (x, y, (zc + 0.016) / 2), segs=20)
    E._rod(b, "env2_hmi_collar", 0.055, 0.04, M["cabinet_dark"], (x, y, zc), segs=20)
    # tilted operator panel (screen faces -Y, tilted back HMI_TILT), built in its local frame and placed with a matrix
    lb = E._B(b.col)
    ew, ed, eh = 0.50, 0.09, 0.40
    C._bx(lb, "env2_hmi_case", (ew, ed, eh), (0, 0, 0), M["cabinet"], bev=0.008)
    C._bx(lb, "env2_hmi_bezel", (0.42, 0.006, 0.28), (0, -ed / 2 - 0.003, 0.04), M["black"], bev=0.002)
    scr = E._mesh(lb, "env2_hmi_screen", [(-0.19, 0, -0.12), (0.19, 0, -0.12), (0.19, 0, 0.12), (-0.19, 0, 0.12)], [(0, 1, 2, 3)],
                  M["hmi"], (0, -ed / 2 - 0.0065, 0.045))
    uvl = scr.data.uv_layers.new(name="UVMap")
    for li, lp in enumerate(scr.data.loops):
        uvl.data[li].uv = [(0, 0), (1, 0), (1, 1), (0, 1)][lp.vertex_index]
    # buttons row under the screen: e-stop (yellow collar + red mushroom), key switch, green / white buttons
    zb = -0.15
    E._rod(lb, "env2_hmi_estop_ring", 0.034, 0.008, M["yellow"], (0.16, -ed / 2 - 0.004, zb), (math.pi / 2, 0, 0), 20)
    E._rev(lb, "env2_hmi_estop", [(0, 0), (0.012, 0), (0.012, 0.012), (0.026, 0.016), (0.026, 0.028), (0, 0.032)], M["red"],
           (0.16, -ed / 2 - 0.008, zb), (math.pi / 2, 0, 0), segs=20)
    E._rod(lb, "env2_hmi_key", 0.014, 0.016, M["dark"], (0.07, -ed / 2 - 0.008, zb), (math.pi / 2, 0, 0), 12)
    E._box(lb, "env2_hmi_keytab", (0.005, 0.012, 0.024), (0.07, -ed / 2 - 0.018, zb), M["steel"])
    for k, (dx, mat) in enumerate(((-0.02, M["green_btn"]), (-0.08, M["white_btn"]), (-0.14, M["amber"]))):
        E._rod(lb, f"env2_hmi_btn{k}", 0.012, 0.016, mat, (dx, -ed / 2 - 0.008, zb), (math.pi / 2, 0, 0), 12)
    # rain hood over the screen
    C._bx(lb, "env2_hmi_hood", (ew + 0.02, 0.08, 0.012), (0, -ed / 2 - 0.03, eh / 2 + 0.004), M["cabinet_dark"], bev=0.003)
    F = kin.tr(x, y, zc + 0.03 + eh / 2 * math.cos(HMI_TILT)) @ kin.rotx(-HMI_TILT)
    C._to_frame(lb.objs, F)
    b.objs.extend(lb.objs)
    # yoke between the collar and the case
    C._bx(b, "env2_hmi_yoke", (0.14, 0.06, 0.08), (x, y + 0.02, zc + 0.05), M["cabinet_dark"], bev=0.004)
    # cable from the pedestal foot to the cabinet in a yellow/black floor cover
    cx, cy = CAB_C
    xa, xb = x + 0.17, cx - CAB_S[0] / 2
    C._bx(b, "env2_hmi_cover", (xb - xa, 0.16, 0.025), ((xa + xb) / 2, y, 0.0125), M["hazard"], bev=0.006)


def _tray(b, M):
    """Galvanised cable tray on the -Y fence top (outside), from the cabinet trunking to a junction box."""
    xa, xb = TRAY_X
    ln = xb - xa
    E._prism(b, "env2_tray", E._u_poly(TRAY_H, TRAY_W, 0.004), ln, M["galv"], ((xa + xb) / 2, TRAY_Y, TRAY_Z + TRAY_H / 2), axis='X')
    E._rod(b, "env2_tray_cable_a", 0.018, ln - 0.1, M["cable"], ((xa + xb) / 2, TRAY_Y - 0.04, TRAY_Z + 0.022), (0, math.pi / 2, 0), 10)
    E._rod(b, "env2_tray_cable_b", 0.012, ln - 0.2, M["cable"], ((xa + xb) / 2 + 0.05, TRAY_Y + 0.03, TRAY_Z + 0.016), (0, math.pi / 2, 0), 8)
    E._rod(b, "env2_tray_cable_c", 0.010, ln - 0.3, M["cabinet_dark"], ((xa + xb) / 2 - 0.05, TRAY_Y + 0.065, TRAY_Z + 0.014), (0, math.pi / 2, 0), 8)
    for k, (px, py) in enumerate(_tray_posts()):
        # stand-off on the post top + cantilever arm under the tray
        E._box(b, f"env2_tray_standoff{k}", (0.04, 0.04, TRAY_Z - 0.03 - FH - 0.01), (px, py, (TRAY_Z - 0.03 + FH + 0.01) / 2), M["dark"])
        E._box(b, f"env2_tray_arm{k}", (0.04, TRAY_W + 0.10, 0.03), (px, TRAY_Y + 0.03, TRAY_Z - 0.015), M["dark"])
    # junction box on the outside of the post at the -X tray end, conduit down from the tray end
    jx = min(_tray_posts(), key=lambda p: p[0])[0] if _tray_posts() else xa + 0.1
    # back face on the outer face of the post (0.06 post: y = Y0 - 0.03)
    C._bx(b, "env2_jbox", (0.22, 0.10, 0.26), (jx, Y0 - 0.08, 1.80), M["cabinet"], bev=0.006)
    C._label(b, "env2_jbox_label", M["label_jb"], 0, (0.12, 0.036), (jx, Y0 - 0.1315, 1.87), (math.pi / 2, 0, 0))
    E._bar(b, "env2_jbox_conduit", (xa + 0.06, TRAY_Y - 0.02, TRAY_Z + 0.01), (jx - 0.05, Y0 - 0.10, 1.93), 0, M["cable"], radius=0.014, segs=10)


# ============================================================================ signs
def _signs(b, M):
    w, h = SIGN_SIZE
    rot = {'-Y': (math.pi / 2, 0, 0), '+Y': (math.pi / 2, 0, math.pi), '-X': (math.pi / 2, 0, -math.pi / 2)}
    nrm = {'-Y': (0.0, -1.0), '+Y': (0.0, 1.0), '-X': (-1.0, 0.0)}
    for k, (idx, (x, y), face) in enumerate(_sign_spots()):
        nx, ny = nrm[face]
        plate = (w, 0.012, h) if face in ('-Y', '+Y') else (0.012, w, h)
        E._box(b, f"env2_sign{k}_plate", plate, (x + nx * 0.04, y + ny * 0.04, SIGN_Z), M["yellow"])
        C._label(b, f"env2_sign{k}", M["signs"], idx, (w - 0.01, h - 0.01), (x + nx * 0.0465, y + ny * 0.0465, SIGN_Z), rot[face])
        # two clamps onto the top rail region of the mesh
        for j, s in enumerate((-1, 1)):
            off = s * (w / 2 - 0.05)
            c = (x + off, y + ny * 0.02, SIGN_Z + h / 2 - 0.03) if face in ('-Y', '+Y') else (x + nx * 0.02, y + off, SIGN_Z + h / 2 - 0.03)
            E._box(b, f"env2_sign{k}_clip{j}", (0.02, 0.05, 0.02) if face in ('-Y', '+Y') else (0.05, 0.02, 0.02), c, M["dark"])


# ============================================================================ public API
def build(collection=None):
    col = collection or bpy.data.collections.new(NAME)
    if collection is None:
        bpy.context.scene.collection.children.link(col)
    b = E._B(col)
    M = _mats()
    _fence(b, M)
    _floor(b, M)
    muting = _muting(b, M)
    _cabinet(b, M)
    _hmi(b, M)
    _tray(b, M)
    _signs(b, M)
    env = dict(collection=col, objects=b.objs, muting=muting)
    set_muting(env, 0.0)
    return env


def build_lighting(scene):
    """Logistics-zone key (LOG_KEY_LIGHT, shadowed) + a weak shadowless fill; the stage-1 lights / world stay as they are."""
    col = bpy.data.collections.get(LIGHTS_NAME) or bpy.data.collections.new(LIGHTS_NAME)
    if col.name not in scene.collection.children:
        scene.collection.children.link(col)
    k = L2.LOG_KEY_LIGHT
    key = E._area_light(col, "env2_light_key", k["pos"], KEY_TARGET, k["energy"], k["size"], (1.0, 0.95, 0.90), 'SQUARE',
                        math.radians(KEY_SPREAD_DEG), shadow=True, shadow_res=0.01)
    fill = E._area_light(col, "env2_light_fill", FILL["pos"], FILL["target"], FILL["energy"], FILL["size"], (0.85, 0.9, 1.0),
                         'RECTANGLE', math.radians(150), shadow=False)
    fill.data.size_y = FILL["size_y"]
    return dict(lights=[key, fill])


def set_muting(env, on, frame=None):
    """Muting lamps on/off (object property 'muting_on' on every lamp lens); CONSTANT keys when frame is given."""
    for lamp in env["muting"]:
        lamp["muting_on"] = float(on)
        if frame is not None:
            C.key_constant(lamp, '["muting_on"]', frame)


def obstacles():
    """Conservative world AABBs (dict(name, center, size, yaw=0)) of the static geometry: fence runs (posts, bays,
    light-curtain columns, signs), personnel door, muting posts, cabinet + HMI + tray, floor paint (thin boxes)."""
    out = []

    def add(name, lo, hi):
        out.append(dict(name=name, center=tuple((lo[k] + hi[k]) / 2 for k in range(3)),
                        size=tuple(hi[k] - lo[k] for k in range(3)), yaw=0.0))

    def along(p, q, cross, z0, z1, lo_ext=0.0):
        """Box along the fence segment p-q (axis-aligned runs), +-cross across it, z0..z1."""
        x0, x1 = sorted((p[0], q[0]))
        y0, y1 = sorted((p[1], q[1]))
        if abs(p[1] - q[1]) < 1e-9:
            return (x0 - lo_ext, y0 - cross, z0), (x1 + lo_ext, y1 + cross, z1)
        return (x0 - cross, y0 - lo_ext, z0), (x1 + cross, y1 + lo_ext, z1)

    for rn, pts in fence_runs():                    # bays: mesh, rails and stiles within +-0.016 of the fence line
        for i in range(len(pts) - 1):
            add(f"env2_fbay_{rn}_{i}", *along(pts[i], pts[i + 1], 0.02, 0.10, FH - 0.06 + 0.02))
    for k, (x, y) in enumerate(fence_posts()):
        add(f"env2_fpost_{k}_plate", (x - 0.081, y - 0.081, 0.0), (x + 0.081, y + 0.081, 0.013))
        add(f"env2_fpost_{k}", (x - 0.036, y - 0.036, 0.0), (x + 0.036, y + 0.036, FH + 0.011))
    for name, (x, y), _ in curtain_posts():
        add(f"env2_lc_{name}_base", (x - 0.101, y - 0.101, 0.0), (x + 0.101, y + 0.101, 0.031))
        add(f"env2_lc_{name}", (x - 0.047, y - 0.047, 0.0), (x + 0.047, y + 0.047, 1.893))
    nrm = {'-Y': (0.0, -1.0), '+Y': (0.0, 1.0), '-X': (-1.0, 0.0)}
    for k, (_, (x, y), face) in enumerate(_sign_spots()):   # plate, label and clips on the outside face
        nx, ny = nrm[face]
        a = (x + nx * -0.006, y + ny * -0.006)
        b_ = (x + nx * 0.054, y + ny * 0.054)
        hw = SIGN_SIZE[0] / 2 + 0.005
        lo_ = (min(a[0], b_[0]) - (hw if face != '-X' else 0), min(a[1], b_[1]) - (hw if face == '-X' else 0), SIGN_Z - SIGN_SIZE[1] / 2 - 0.005)
        hi_ = (max(a[0], b_[0]) + (hw if face != '-X' else 0), max(a[1], b_[1]) + (hw if face == '-X' else 0), SIGN_Z + SIGN_SIZE[1] / 2 + 0.005)
        add(f"env2_sign{k}", lo_, hi_)
    # personnel door: leaf, rail, brackets, handle, interlock (outside the -Y fence line)
    add("env2_door", (DOOR_RAIL_X[0] - 0.01, Y0 - 0.20, 0.0), (DOOR_RAIL_X[1] + 0.12, Y0 + 0.03, FH + 0.11))
    mx, my = MUTE_POS
    for k, s in enumerate((-1, 1)):
        add(f"env2_mute{k}", (mx - 0.075, s * my - 0.075, 0.0), (mx + 0.075, s * my + 0.075, MUTE_H + 0.14))
    (cx, cy), (sx, sy, sz) = CAB_C, CAB_S
    add("env2_cabinet", (cx - sx / 2 - 0.03, cy - sy / 2 - 0.07, 0.0), (cx + sx / 2 + 0.03, cy + sy / 2 + 0.02, CAB_PLINTH + sz + 0.35))
    x, y = HMI_C
    add("env2_hmi", (x - 0.28, y - 0.30, 0.0), (x + 0.28, y + 0.20, 1.55))
    add("env2_hmi_cover", (x + 0.16, y - 0.09, 0.0), (cx - sx / 2 + 0.01, y + 0.09, 0.03))
    add("env2_tray", (TRAY_X[0] - 0.02, TRAY_Y - TRAY_W / 2 - 0.02, FH), (TRAY_X[1] + 0.02, Y0 + 0.03, TRAY_Z + TRAY_H + 0.02))
    add("env2_cab_trunk", (TRAY_X[1] - 0.18, cy + sy / 2 - 0.11, CAB_PLINTH + sz), (TRAY_X[1] - 0.02, cy + sy / 2 - 0.01, TRAY_Z + 0.04))
    jp = _tray_posts()
    if jp:
        jx = min(p[0] for p in jp)
        add("env2_jbox", (min(jx - 0.12, TRAY_X[0] - 0.02), Y0 - 0.16, 1.66), (jx + 0.12, Y0 - 0.02, TRAY_Z + 0.04))
    # floor paint and stripes (<= 4 mm): one thin box per painted rectangle
    groups = dict(paint_groups())
    groups["env2_hatch"] = (None, hatch_rects())
    hz = X0 - HZ_G - HZ_W
    groups["env2_hz"] = (None, [(hz, Y1 + HZ_G, HZ_X0, Y1 + HZ_G + HZ_W), (hz, Y0 - HZ_G - HZ_W, HZ_X0, Y0 - HZ_G),
                                (hz, Y0 - HZ_G, X0 - HZ_G, Y1 + HZ_G)])
    for name, (_, rects) in groups.items():
        for k, r in enumerate(rects):
            add(f"{name}_{k}", (r[0] - 0.001, r[1] - 0.001, 0.0), (r[2] + 0.001, r[3] + 0.001, 0.004))
    return out
