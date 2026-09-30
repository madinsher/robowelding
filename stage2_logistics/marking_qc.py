"""Laser marking + weld-profile QC arch (portal) over the output conveyor, and the laser-marked ID decal on the pipe.

Placed from layout2 `QC_ARCH` (posts at y0 / y1, beam at beam_z, x = CONV_QC_X), `QC_MARK_LOCAL`, `QC_SCAN_LOCAL`.
Every object of the arch is under ONE root empty 'qc_root' (world (QC_ARCH.x, 0, 0), identity rotation), so the
stage-1 `vfx.laser_line` skips the whole arch when it ray-casts the scanner line onto the spool.

* portal: two square-tube posts (base plates, anchors, gussets), crossbeam (centre line at beam_z) with two linear
  rails underneath, a rack on its +X face, rubber end stops, cable tray + energy chain on top (fixed end at the +Y
  post; the loop and both runs follow the carriage through drivers on the carriage's location[1]), stack light on
  the +Y end of the beam, laser source / controller cabinet at the +Y post, HMI panel on the -Y post (facing +Y);
* carriage 'qc_carriage' (channel location[1] = world y of the heads' working plane): plate on four guide blocks,
  servo motor + pinion on its +X side, energy-chain bracket on its -X side, and two heads side by side along X, both
  aimed at the pipe top line x = QC_ARCH.x (so both work at the carriage y):
    - fiber-laser marking head (galvo box + f-theta lens, tilted toward +X, fiber cable, emission LED) on the -X side;
      the beam 'qc_beam' (emissive core + glow sheath, object property 'on') is a Stretch-To line from the lens to the
      writing spot; build_mark() re-targets it to the moving front of the decal;
    - weld-profile scanner (laser-triangulation sensor, tilted toward -X) on the +X side; 'qc_sensor' is the empty for
      `vfx.laser_line`: its +Z looks straight down, its Y (line direction) runs along the pipe axis (world Y), and the
      sensor housing's laser window sits exactly at vfx.LASER_WINDOW_FALLBACK in the sensor frame;
  the lowest head part is >= 0.2 m above the spool top (pipe crest seat_z + 0.618) wherever the carriage is.

Decal ('qc_mark', child of the pipe root, spool-local geometry): laser-cleaned bright patch with a dark DataMatrix-like
code and "SPL-02 · WPS-07" / "QC OK" (PIL image) wrapped on the pipe top around QC_MARK_LOCAL (x 0.415 .. 0.545, between
the seam-B heat tint and the stage-1 paint marking).  Object properties read by its shader: 'reveal' (0..1, the mark
appears left to right = along the pipe axis, local +X) and 'hot' (0..1, glow: strong at the writing front, a dim red
afterglow on the rest).  An empty 'qc_mark_front' follows the writing front (drivers on 'reveal', with a small raster
wobble across the text height); the marker beam and the spot light aim at it.

HMI screen ('qc_display', atlas of the 4 states): texts in the current language of i18n2 (looked up while build() draws
the texture), each shrunk to fit its box; the logo of branding2's current brand on a white plate at the right end of
the footer bar.  Texture / material names carry both (qc_display_img_<lang>_<brand>, qc_display_<lang>_<brand>).
The label atlas of the arch / cabinet (qc_label_<lang>) has its laser warning in the current language too; the text
boxes of the screen atlas and of the decal are public (display_text_boxes, mark_text_box: tests/t_i18n2.py).

Public API (DESIGN.md section 3, marking_qc.py):
    build(collection=None) -> dict(collection, root, carriage, marker, beam, scanner, sensor, tower, display,
                                   spot, led, objects, carriage_range)
        tower = dict(red=obj, amber=obj, green=obj)   (object property 'on')
        display = HMI screen object (object property 'state' 0 idle, 1 marking, 2 scanning, 3 QC OK)
    build_mark(pipe_root) -> decal object 'qc_mark' (properties 'reveal', 'hot')
    set_carriage(qc, y, frame=None)       carriage world y (clamped to carriage_range); keys location[1] only
    set_marker(qc, on, frame=None)        laser on/off: beam + emission LED property 'on', spot light energy (CONSTANT)
    set_mark(decal, reveal, hot, frame=None)   keys the decal's 'reveal' and 'hot'
    set_tower(qc, state, frame=None)      "off" | "amber" | "green" | "red" (CONSTANT keys on the three lamps)
    set_display(qc, state, frame=None)    0..3 or "idle" | "marking" | "scanning" | "ok" (CONSTANT)
    scan_line(qc, intervals, collection=None)  vfx.laser_line(qc["sensor"], intervals) with the stage-1 torch lens
                                          hidden from its window lookup (in a full scene the torch's 'torch_laser_lens'
                                          would otherwise become the emitter of the QC line)
    obstacles() -> list[dict(name, center, size, yaw)]   conservative world AABBs (carriage: its whole travel)
"""
import math

import numpy as np
import bpy
import mathutils

import tools  # noqa: F401  (demo_video on sys.path)
from cell import geom as G
from cell import materials
from cell import environment as E
from cell import vfx
import layout2 as L2
import kin
import cassette as C

NAME = "MarkingQC"
A = L2.QC_ARCH
QX = A["x"]
Y0, Y1 = A["y0"], A["y1"]
BZ = A["beam_z"]

# ------------------------------------------------------------------ portal (root-local x = world x - QX; y, z world)
POST = 0.12
BEAM_W, BEAM_H = 0.16, 0.20
BEAM_BOT, BEAM_TOP = BZ - BEAM_H / 2, BZ + BEAM_H / 2          # 2.00 / 2.20
BEAM_Y = (Y0 - 0.07, Y1 + 0.07)
RAIL_X, RAIL_W, RAIL_H = 0.05, 0.02, 0.022
BLOCK = (0.034, 0.07, 0.028)
PLATE_Z = (BEAM_BOT - RAIL_H - BLOCK[2] - 0.02, BEAM_BOT - RAIL_H - BLOCK[2])   # carriage plate (1.93, 1.95)
PLATE_X, PLATE_HY = (-0.20, 0.19), 0.13                          # carriage plate x range / half width
POST_CAP = 0.08                                                  # half size (y) of the post top plates under the beam
RAIL_Y = (Y0 + POST_CAP + 0.004, Y1 - POST_CAP - 0.004)          # linear rails between the post top plates
CAR_RANGE = (Y0 + 0.23, Y1 - 0.23)                               # carriage travel (world y): the guide-block seals
                                                                 # stop 8 mm short of the rubber end stops
SPOOL_TOP = L2.CARRIER["seat_z"] + L2.SPOOL_TOP_Z                # pipe crest on a carrier (1.418)
# marking head: lens exit 0.22 above the crest, 0.07 toward -X, optical axis through the crest line
LENS = (-0.07, SPOOL_TOP + 0.22)
# scanner: laser window 0.25 above the crest; the sensor empty sits 0.126 below it (vfx.LASER_WINDOW_FALLBACK)
WIN_OFF = tuple(vfx.LASER_WINDOW_FALLBACK)                         # (0.078, 0, -0.126)
WIN = (WIN_OFF[0], SPOOL_TOP + 0.25)
SENSOR_Z = WIN[1] + WIN_OFF[2]
# energy chain on the beam top: fixed end at y = CHAIN_YF, loop at (CHAIN_YF + y - CHAIN_C) / 2
CHAIN_X, CHAIN_W, CHAIN_H, CHAIN_R = -0.035, 0.07, 0.035, 0.06
CHAIN_YF = Y1 - 0.10
CHAIN_C = CHAIN_YF - CAR_RANGE[0] + 0.08
CHAIN_Z = BEAM_TOP + 0.005 + CHAIN_H / 2                         # lower-run centre line
TOWER_Y = Y1 + 0.02
CAB = dict(x=(-0.25, 0.25), y=(Y1 + 0.08, Y1 + 0.38), h=1.10)    # laser source / controller cabinet
HMI = dict(x=(-0.12, 0.12), z=(1.12, 1.30), t=0.035)              # on the -Y post's +Y face
SPOT_POWER = 3.0

# ------------------------------------------------------------------ decal (spool-local)
MARK_X = (0.415, 0.545)
MARK_W = 0.046
MARK_R = L2.PIPE_R + 0.0006
MARK_IMG = (1300, 460)
MARK_DM = (50, 20)                  # decal DataMatrix: left edge, module pitch (px); the text lines start 50 px after it
MARK_WOBBLE = (0.012, 90.0)        # raster wobble of the writing spot across the text: amplitude (m), frequency


def _mats():
    return dict(
        blue=materials.get("painted", color="#1F4E8C", roughness=0.42),
        blue2=materials.get("painted", color="#26558F", roughness=0.42),
        frame=materials.get("painted", color="#2B2F36", roughness=0.5),
        charcoal=materials.get("painted", color="#3A3E44", roughness=0.45, coat=0.1),
        alu=materials.get("painted", color="#A9ADB2", roughness=0.35, coat=0.0),
        cabinet=materials.get("painted", color="#B9BCBE", roughness=0.5),
        cabinet_dark=materials.get("painted", color="#5A5D60", roughness=0.5),
        yellow=materials.get("safety_yellow"),
        red=materials.get("safety_red"),
        galv=materials.get("galvanized"),
        steel=materials.get("machined_steel"),
        dark=materials.get("dark_metal"),
        black=materials.get("black_plastic"),
        rubber=materials.get("rubber"),
        cable=materials.get("cable_black"),
        glass=materials.get("glass_dark"),
        hazard=E._mat_hazard(),
        red_glass=_red_glass(),
        green_led=materials.get("emissive", color="#30FF40", strength=6.0),
        blue_led=materials.get("emissive", color="#39B8FF", strength=5.0),
        laser_led=C._state_mat("qc_led_laser", (1.0, 0.12, 0.04), 14.0, "on"),
    )


def _red_glass():
    m = bpy.data.materials.get("qc_red_glass")
    if m is None:
        m = materials.get("glass_dark").copy()
        m.name = "qc_red_glass"
        b = m.node_tree.nodes["Principled BSDF"]
        b.inputs["Base Color"].default_value = (0.35, 0.02, 0.02, 1.0)
        b.inputs["Roughness"].default_value = 0.08
    return m


def _lamp_mat(key, color, strength):
    """Stack-light segment: tinted translucent-looking plastic, emission = object property 'on' x strength."""
    m = bpy.data.materials.get(key)
    if m is not None:
        return m
    m = C._state_mat(key, color, strength, "on")
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (color[0] * 0.30, color[1] * 0.30, color[2] * 0.30, 1.0)
    b.inputs["Roughness"].default_value = 0.3
    b.inputs["Coat Weight"].default_value = 0.6
    return m


def _beam_mat(key, color, strength, alpha):
    """Laser beam: emission mixed with transparency by the object property 'on' (x alpha)."""
    m = bpy.data.materials.get(key)
    if m is not None:
        return m
    m = bpy.data.materials.new(key)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*color, 1.0)
    em.inputs["Strength"].default_value = strength
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    att = nt.nodes.new("ShaderNodeAttribute")
    att.attribute_type = 'OBJECT'
    att.attribute_name = "on"
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = 'MULTIPLY'
    mul.use_clamp = True
    mul.inputs[1].default_value = alpha
    nt.links.new(att.outputs["Fac"], mul.inputs[0])
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(mul.outputs[0], mix.inputs[0])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(em.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    m.surface_render_method = 'BLENDED'
    m.use_transparent_shadow = True
    return m


def _chain_mat(key, space_obj=None):
    """Energy chain plastic: black with link grooves every 35 mm along y of `space_obj` (None: the object itself)."""
    m = bpy.data.materials.get(key)
    if m is not None:
        return m
    m = materials.get("black_plastic").copy()
    m.name = key
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.03, 0.03, 0.032, 1.0)
    b.inputs["Roughness"].default_value = 0.5
    tc = nt.nodes.new("ShaderNodeTexCoord")
    if space_obj is not None:
        tc.object = space_obj
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs[0])
    mask = E._line_mask(nt, sep.outputs["Y"], 0.035, 0.004)
    E._tint_base_color(nt, b, mask, (0.004, 0.004, 0.004))
    return m


# ============================================================================ PIL images
import fonts2  # noqa: E402
import i18n2  # noqa: E402
import branding2  # noqa: E402
_FONT_B = fonts2.SANS_BOLD
_FONT = fonts2.SANS


def _font(path, size):
    from PIL import ImageFont
    return ImageFont.truetype(path, int(size))


def _fit(text, path, size, width):
    f = _font(path, size)
    while f.getlength(text) > width and size > 8:
        size -= 2
        f = _font(path, size)
    return f


def _to_bpy_image(name, img):
    arr = np.asarray(img.convert("RGBA"), dtype=np.float32)[::-1] / 255.0
    bi = bpy.data.images.get(name)
    if bi is None:
        bi = bpy.data.images.new(name, img.width, img.height, alpha=True)
    bi.colorspace_settings.name = 'sRGB'
    bi.alpha_mode = 'STRAIGHT'
    bi.pixels.foreach_set(arr.ravel())
    bi.pack()
    return bi


def datamatrix_modules(n=18, seed=2402):
    """n x n boolean module grid shaped like an ECC200 DataMatrix: solid L finder (left column, bottom row), alternating
    clock track (top row, right column), pseudo-random data inside (fixed seed)."""
    rng = np.random.default_rng(seed)
    g = rng.random((n, n)) < 0.47
    g[:, 0] = True
    g[n - 1, :] = True
    g[0, :] = (np.arange(n) % 2) == 0
    g[:, n - 1] = (np.arange(n) % 2) == 1
    g[n - 1, n - 1] = True
    return g


def _mark_image():
    """RGBA decal image: laser-cleaned bright rectangle (thin oxide rim) with dark annealed code + two text lines."""
    name = "qc_mark_img"
    bi = bpy.data.images.get(name)
    if bi is not None:
        return bi
    from PIL import Image, ImageDraw, ImageFilter
    W, H = MARK_IMG
    rng = np.random.default_rng(5)
    base = np.zeros((H, W, 4), np.float32)
    # brushed / ablated metal: bright grey with fine streaks along u
    streak = rng.normal(0.0, 1.0, (H, 1)) * 5.0 + rng.normal(0.0, 1.0, (H, W)) * 3.0
    base[..., 0] = 168 + streak
    base[..., 1] = 171 + streak
    base[..., 2] = 176 + streak
    base[..., 3] = 255
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGBA")
    d = ImageDraw.Draw(img)
    # oxide rim (straw -> blue) around the cleaned patch
    for k, col in enumerate(((150, 120, 70), (95, 100, 140))):
        d.rounded_rectangle((k * 3, k * 3, W - 1 - k * 3, H - 1 - k * 3), radius=18, outline=(*col, 255), width=3)
    ink = (22, 21, 24, 255)
    # DataMatrix: 18 x 18 modules, 20 px each, left, vertically centred
    g = datamatrix_modules()
    n = g.shape[0]
    x0, mod = MARK_DM
    y0 = (H - n * mod) // 2
    for r in range(n):
        for c in range(n):
            if g[r, c]:
                d.rectangle((x0 + c * mod, y0 + r * mod, x0 + (c + 1) * mod - 2, y0 + (r + 1) * mod - 2), fill=ink)
    tx = x0 + n * mod + 50
    tw = W - tx - 45
    f1 = _fit("SPL-02 · WPS-07", _FONT_B, 150, tw)
    d.text((tx, H * 0.30), "SPL-02 · WPS-07", font=f1, fill=ink, anchor="lm")
    f2 = _font(_FONT_B, f1.size)
    d.text((tx, H * 0.72), "QC OK", font=f2, fill=ink, anchor="lm")
    f3 = _fit("DN250  0417", _FONT, f1.size * 0.55, tw - f2.getlength("QC OK ") - 20)
    d.text((W - 45, H * 0.74), "DN250  0417", font=f3, fill=ink, anchor="rm")
    # rounded corners of the patch (alpha), slightly soft edge
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, W - 1, H - 1), radius=18, fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(1.2))
    img.putalpha(mask)
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    return _to_bpy_image(name, img)


def mark_text_box():
    """(x0, y0, x1, y1) px the text lines of the decal stay inside (tests/t_i18n2.py): the cleaned patch right of the
    DataMatrix (from the middle of the 50 px gap), inside the 2 x 3 px oxide rim."""
    W, H = MARK_IMG
    x0, mod = MARK_DM
    return (x0 + datamatrix_modules().shape[0] * mod + 25, 6, W - 6, H - 6)


DISPLAY_STATES = ("idle", "marking", "scanning", "ok")
DISPLAY_CELL = (640, 400)              # one screen of the atlas (px)
DISPLAY_HEAD = 56                      # header bar height (px): title left, DISPLAY_TAG right
DISPLAY_TAG = "SPL-02"
DISPLAY_FOOT = 44                      # footer bar height (px): job line left, logo plate right
DISPLAY_LOGO_PAD = 0.12                # logo plate margin (fraction of the plate height, branding2.logo_plate)


def _edition_key(base):
    """Image / material name of an edition-dependent texture: language + brand, so a scene built for one edition never
    picks up the other one's cached texture."""
    return f"{base}_{i18n2.get_lang()}_{branding2.get_brand()}"


def display_logo_box(k=0):
    """(x0, y0, x1, y1) px of the logo plate in atlas cell k: right end of the footer bar, 5 px inside it."""
    W, H = DISPLAY_CELL
    ph = DISPLAY_FOOT - 10
    pw = ph * branding2.logo_aspect(pad=DISPLAY_LOGO_PAD)
    y1 = k * H + H - 5
    return (W - 10 - pw, y1 - ph, W - 10, y1)


def _display_tag_x():
    """Left edge (px) of the header tag (DISPLAY_TAG, 30 px bold, right-aligned 18 px from the right edge)."""
    return DISPLAY_CELL[0] - 18 - _font(_FONT_B, 30).getlength(DISPLAY_TAG)


def display_text_boxes(k=0):
    """Boxes (x0, y0, x1, y1) px the texts of atlas cell k stay inside (tests/t_i18n2.py checks every drawn text):
    header title left of the tag (up to the middle of their 12 px gap), the tag, the body between the two bars, the
    footer line left of the logo plate."""
    W, H = DISPLAY_CELL
    y0 = k * H
    xs = _display_tag_x() - 6
    yb = y0 + H - DISPLAY_FOOT
    return [(0, y0, xs, y0 + DISPLAY_HEAD), (xs, y0, W, y0 + DISPLAY_HEAD), (0, y0 + DISPLAY_HEAD, W, yb),
            (0, yb, display_logo_box(k)[0], y0 + H)]


def _display_image():
    """Vertical atlas of the 4 HMI screens (cell k = DISPLAY_STATES[k], top to bottom).  Texts in the current
    language (i18n2, looked up here at draw time), each shrunk to fit its box; the brand's logo (branding2) on a white
    plate at the right end of every footer bar."""
    name = _edition_key("qc_display_img")
    bi = bpy.data.images.get(name)
    if bi is not None:
        return bi
    from PIL import Image, ImageDraw
    tr = i18n2.tr
    W, H = DISPLAY_CELL
    img = Image.new("RGB", (W, H * 4), (12, 16, 22))
    d = ImageDraw.Draw(img)
    fh = _font(_FONT_B, 30)
    fs = _font(_FONT, 28)
    fok = _font(_FONT_B, 30)
    # header title from x = 18 to 12 px before the tag (the Russian title at 30 px ran into the tag)
    title = tr("QC-01  МАРКИРОВКА / КОНТРОЛЬ")
    ft = _fit(title, _FONT_B, 30, _display_tag_x() - 12 - 18)
    big = lambda t: _fit(t, _FONT_B, 64, W - 60)            # noqa: E731  state title (SCANNING WELD B: 64 -> 54 px)
    small = lambda t, w=W - 60: _fit(t, _FONT, 28, w)       # noqa: E731
    foot = "WPS-07 · DN250 · C-01"
    for k, st in enumerate(DISPLAY_STATES):
        y0 = k * H
        d.rectangle((0, y0, W, y0 + DISPLAY_HEAD), fill=(31, 78, 140))
        d.text((18, y0 + DISPLAY_HEAD / 2), title, font=ft, fill=(235, 240, 245), anchor="lm")
        d.text((W - 18, y0 + DISPLAY_HEAD / 2), DISPLAY_TAG, font=fh, fill=(242, 180, 0), anchor="rm")
        d.rectangle((0, y0 + H - DISPLAY_FOOT, W, y0 + H), fill=(26, 32, 42))
        d.text((18, y0 + H - 22), foot, font=small(foot, display_logo_box(k)[0] - 36), fill=(150, 160, 175), anchor="lm")
        if st == "idle":
            t1, t2 = tr("ОЖИДАНИЕ"), tr("паллета не на позиции")
            d.text((W / 2, y0 + 170), t1, font=big(t1), fill=(150, 160, 175), anchor="mm")
            d.text((W / 2, y0 + 250), t2, font=small(t2), fill=(110, 120, 135), anchor="mm")
        elif st == "marking":
            t1, t2 = tr("МАРКИРОВКА"), tr("лазер 50 Вт · DataMatrix")
            d.text((W / 2, y0 + 140), t1, font=big(t1), fill=(255, 150, 40), anchor="mm")
            d.text((W / 2, y0 + 205), t2, font=small(t2), fill=(200, 205, 215), anchor="mm")
            d.rectangle((60, y0 + 250, W - 60, y0 + 290), outline=(200, 205, 215), width=3)
            d.rectangle((66, y0 + 256, 66 + (W - 132) * 0.6, y0 + 284), fill=(255, 150, 40))
        elif st == "scanning":
            t1 = tr("СКАН ШВА B")
            d.text((W / 2, y0 + 115), t1, font=big(t1), fill=(90, 200, 255), anchor="mm")
            # bead profile plot: baseline with a reinforcement bump
            gx0, gx1, gy = 70, W - 70, y0 + 290
            d.rectangle((gx0 - 10, y0 + 170, gx1 + 10, y0 + 320), outline=(60, 70, 85), width=2)
            pts = []
            for i in range(121):
                u = i / 120
                bump = math.exp(-((u - 0.5) / 0.09) ** 2) * 70 + math.sin(u * 60) * 2
                pts.append((gx0 + (gx1 - gx0) * u, gy - bump))
            d.line(pts, fill=(255, 60, 40), width=4)
            d.line((gx0, gy - 55, gx1, gy - 55), fill=(90, 200, 255), width=1)
        else:
            t1 = tr("ГОДЕН · QC OK")
            d.text((W / 2, y0 + 125), t1, font=big(t1), fill=(60, 230, 90), anchor="mm")
            for i, t in enumerate(("Шов A", "Шов B", "Маркировка")):
                t = tr(t)
                yy = y0 + 200 + i * 42
                # item left at 150, "OK" right-aligned at W - 150: the item may use the width up to 20 px before "OK"
                d.text((150, yy), t, font=small(t, W - 320 - fok.getlength("OK")), fill=(210, 215, 225), anchor="lm")
                d.text((W - 150, yy), "OK", font=fok, fill=(60, 230, 90), anchor="rm")
    for k in range(len(DISPLAY_STATES)):          # after the text: paste writes into img, the ImageDraw stays valid
        branding2.paste_logo(img, display_logo_box(k), plate=True, pad=DISPLAY_LOGO_PAD)
    return _to_bpy_image(name, img)


def _display_mat():
    key = _edition_key("qc_display")
    m = bpy.data.materials.get(key)
    if m is not None:
        return m
    bi = _display_image()
    n = len(DISPLAY_STATES)
    m = bpy.data.materials.new(key)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Roughness"].default_value = 0.12
    b.inputs["Coat Weight"].default_value = 0.8
    uv = nt.nodes.new("ShaderNodeUVMap")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(uv.outputs["UV"], sep.inputs[0])
    att = nt.nodes.new("ShaderNodeAttribute")
    att.attribute_type = 'OBJECT'
    att.attribute_name = "state"
    # v' = (v + (n - 1 - state)) / n  (cell 0 at the top of the atlas)
    off = E._math(nt, 'SUBTRACT', float(n - 1), att.outputs["Fac"])
    v2 = E._math(nt, 'DIVIDE', E._math(nt, 'ADD', sep.outputs["Y"], off), float(n))
    comb = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(sep.outputs["X"], comb.inputs["X"])
    nt.links.new(v2, comb.inputs["Y"])
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bi
    tex.interpolation = 'Linear'
    tex.extension = 'EXTEND'
    nt.links.new(comb.outputs[0], tex.inputs["Vector"])
    dim = nt.nodes.new("ShaderNodeMix")
    dim.data_type = 'RGBA'
    dim.blend_type = 'MULTIPLY'
    dim.inputs[0].default_value = 1.0
    dim.inputs[7].default_value = (0.15, 0.15, 0.15, 1.0)
    nt.links.new(tex.outputs["Color"], dim.inputs[6])
    nt.links.new(dim.outputs[2], b.inputs["Base Color"])
    nt.links.new(tex.outputs["Color"], b.inputs["Emission Color"])
    b.inputs["Emission Strength"].default_value = 2.2
    return m


def _mark_mat():
    """Decal shader: image colour / alpha, wipe mask u < reveal, glow = hot x (afterglow + writing front)."""
    key = "qc_mark"
    m = bpy.data.materials.get(key)
    if m is not None:
        return m
    bi = _mark_image()
    m = bpy.data.materials.new(key)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Metallic"].default_value = 0.85
    uv = nt.nodes.new("ShaderNodeUVMap")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(uv.outputs["UV"], sep.inputs[0])
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bi
    tex.interpolation = 'Linear'
    tex.extension = 'CLIP'
    nt.links.new(uv.outputs["UV"], tex.inputs["Vector"])
    ar = nt.nodes.new("ShaderNodeAttribute")
    ar.attribute_type = 'OBJECT'
    ar.attribute_name = "reveal"
    ah = nt.nodes.new("ShaderNodeAttribute")
    ah.attribute_type = 'OBJECT'
    ah.attribute_name = "hot"
    dlt = E._math(nt, 'SUBTRACT', ar.outputs["Fac"], sep.outputs["X"])        # reveal - u (> 0: already marked)

    def mrange(v, a, b_, c=0.0, d_=1.0):
        n = nt.nodes.new("ShaderNodeMapRange")
        n.clamp = True
        n.inputs["From Min"].default_value = a
        n.inputs["From Max"].default_value = b_
        n.inputs["To Min"].default_value = c
        n.inputs["To Max"].default_value = d_
        nt.links.new(v, n.inputs["Value"])
        return n.outputs["Result"]

    shown = mrange(dlt, -0.003, 0.003)
    alpha = E._math(nt, 'MULTIPLY', tex.outputs["Alpha"], shown)
    nt.links.new(alpha, b.inputs["Alpha"])
    nt.links.new(tex.outputs["Color"], b.inputs["Base Color"])
    bw = nt.nodes.new("ShaderNodeRGBToBW")
    nt.links.new(tex.outputs["Color"], bw.inputs[0])
    nt.links.new(mrange(bw.outputs[0], 0.02, 0.4, 0.62, 0.28), b.inputs["Roughness"])
    # glow: afterglow 0.02 + front^3 (front = 1 at the writing edge, 0 at 7 % behind), x hot, x alpha
    front = mrange(dlt, 0.07, 0.0)
    f3 = E._math(nt, 'MULTIPLY', E._math(nt, 'MULTIPLY', front, front), front)
    glow = E._math(nt, 'MULTIPLY', E._math(nt, 'ADD', f3, 0.02), ah.outputs["Fac"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.elements[0].position = 0.0
    cr.elements[0].color = (0.55, 0.04, 0.0, 1.0)
    e = cr.elements.new(0.45)
    e.color = (1.0, 0.33, 0.04, 1.0)
    cr.elements[-1].position = 1.0
    cr.elements[-1].color = (1.0, 0.82, 0.55, 1.0)
    nt.links.new(glow, ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], b.inputs["Emission Color"])
    nt.links.new(E._math(nt, 'MULTIPLY', E._math(nt, 'MULTIPLY', glow, 20.0), alpha), b.inputs["Emission Strength"])
    m.surface_render_method = 'BLENDED'
    m.use_transparency_overlap = False
    m.use_transparent_shadow = True
    m.use_backface_culling = True
    return m


# ============================================================================ portal (static)
def _gusset(b, name, x_sign, y, z_top, M, size=(0.12, 0.14), t=0.012, down=True):
    """Triangular plate in the X-Z plane at y: leg `size[0]` along +-X at z_top, `size[1]` down the post."""
    w, h = size
    poly = [(0.0, 0.0), (x_sign * w, 0.0), (0.0, -h if down else h)]
    if x_sign * (-h if down else h) < 0:
        poly = poly[::-1]
    return E._prism(b, name, poly, t, M["blue"], (x_sign * POST / 2, y, z_top), axis='Y')


def _posts(b, M):
    for k, y in enumerate((Y0, Y1)):
        p = f"qc_post{k}"
        C._bx(b, p, (POST, POST, BEAM_BOT - 0.015 - 0.015), (0.0, y, (BEAM_BOT - 0.015 + 0.015) / 2), M["blue"], bev=0.004)
        C._bx(b, p + "_base", (0.26, 0.18, 0.015), (0.0, y, 0.0075), M["frame"], bev=0.002)
        C._bx(b, p + "_top", (0.18, 2 * POST_CAP, 0.015), (0.0, y, BEAM_BOT - 0.0075), M["frame"], bev=0.002)
        for n, (dx, dy) in enumerate(((-0.1, -0.06), (0.1, -0.06), (0.1, 0.06), (-0.1, 0.06))):
            C._hex(b, f"{p}_anchor{n}", 0.011, 0.012, M["dark"], (dx, y + dy, 0.015))
            E._rod(b, f"{p}_anchor{n}_stud", 0.007, 0.02, M["steel"], (dx, y + dy, 0.037), segs=8)
        for n, s in enumerate((-1, 1)):
            _gusset(b, f"{p}_gbase{n}", s, y, 0.015, M, size=(0.06, 0.10), down=False)
            _gusset(b, f"{p}_gtop{n}", s, y, BEAM_BOT - 0.015, M, size=(0.09, 0.14), down=True)
        for n, (dx, dy) in enumerate(((-0.07, -0.055), (0.07, -0.055), (0.07, 0.055), (-0.07, 0.055))):
            C._hex(b, f"{p}_topbolt{n}", 0.009, 0.008, M["dark"], (dx, y + dy, BEAM_BOT - 0.015), (math.pi, 0, 0))


def _beam(b, M, lm):
    ya, yb = BEAM_Y
    ym, L = (ya + yb) / 2, yb - ya
    C._bx(b, "qc_beam_girder", (BEAM_W, L, BEAM_H), (0.0, ym, BZ), M["blue"], bev=0.005)
    for k, y in enumerate((ya, yb)):
        C._bx(b, f"qc_beam_cap{k}", (BEAM_W + 0.01, 0.008, BEAM_H + 0.01), (0.0, y + (0.004 if k == 0 else -0.004), BZ), M["frame"])
    ra, rb = RAIL_Y
    for k, s in enumerate((-1, 1)):
        C._bx(b, f"qc_rail{k}", (RAIL_W, rb - ra, RAIL_H), (s * RAIL_X, (ra + rb) / 2, BEAM_BOT - RAIL_H / 2), M["steel"], bev=0.001)
        for j, y in enumerate((ra + 0.012, rb - 0.012)):
            C._bx(b, f"qc_endstop{k}{j}", (0.03, 0.02, 0.03), (s * RAIL_X, y, BEAM_BOT - 0.015), M["rubber"], bev=0.003)
    # rack on the +X face (teeth as a fine ridge pattern would alias: plain ground bar) + its clamp bolts
    C._bx(b, "qc_rack", (0.012, rb - ra, 0.03), (BEAM_W / 2 + 0.006, (ra + rb) / 2, BEAM_BOT + 0.03), M["steel"])
    n = int((rb - ra) / 0.25)
    for k in range(n + 1):
        C._hex(b, f"qc_rack_bolt{k}", 0.006, 0.004, M["dark"], (BEAM_W / 2 + 0.012, ra + (rb - ra) * k / n, BEAM_BOT + 0.03), (0, math.pi / 2, 0))
    # cable tray on the top, fixed end of the energy chain
    tray = E._u_poly(0.05, 0.08, 0.003)
    E._prism(b, "qc_tray", tray, CHAIN_YF + 0.06 - Y0, M["galv"], (CHAIN_X, (CHAIN_YF + 0.06 + Y0) / 2, BEAM_TOP + 0.025), axis='Y')
    C._bx(b, "qc_chain_fixed", (0.08, 0.05, 0.03), (CHAIN_X, CHAIN_YF + 0.025, CHAIN_Z), M["frame"], bev=0.002)
    # labels on the +X face: arch ID and laser warning
    C._label(b, "qc_label_id", lm, 0, (0.22, 0.088), (BEAM_W / 2 + 0.0015, (Y0 + Y1) / 2 + 0.35, BZ + 0.03), (math.pi / 2, 0, math.pi / 2),
             backing=M["frame"])
    C._label(b, "qc_label_laser", lm, 1, (0.22, 0.088), (BEAM_W / 2 + 0.0015, (Y0 + Y1) / 2 - 0.35, BZ + 0.03), (math.pi / 2, 0, math.pi / 2),
             backing=M["frame"])


def _tower(b, M):
    """Stack light on the +Y end of the beam: base, pole, green / amber / red segments, cap."""
    y = TOWER_Y
    z = BEAM_TOP
    E._rev(b, "qc_tower_base", [(0, 0), (0.035, 0), (0.035, 0.022), (0.028, 0.03), (0, 0.03)], M["black"], (0.0, y, z), segs=24)
    E._rod(b, "qc_tower_pole", 0.0125, 0.06, M["steel"], (0.0, y, z + 0.06), segs=16)
    z += 0.09
    lamps = {}
    h = 0.075
    for k, (key, col) in enumerate((("green", (0.12, 1.0, 0.22)), ("amber", (1.0, 0.55, 0.02)), ("red", (1.0, 0.06, 0.03)))):
        E._rev(b, f"qc_tower_ring{k}", [(0, 0), (0.034, 0), (0.034, 0.006), (0, 0.006)], M["black"], (0.0, y, z), segs=24)
        ob = E._rev(b, f"qc_tower_{key}", [(0, 0), (0.031, 0), (0.033, 0.006), (0.033, h - 0.006), (0.031, h), (0, h)],
                    _lamp_mat(f"qc_lamp_{key}", col, 18.0), (0.0, y, z + 0.006), segs=28)
        ob["on"] = 0.0
        lamps[key] = ob
        z += h + 0.006
    E._rev(b, "qc_tower_cap", [(0, 0), (0.034, 0), (0.034, 0.012), (0.02, 0.02), (0, 0.02)], M["black"], (0.0, y, z), segs=24)
    return lamps


def _cabinet(b, M, lm):
    (xa, xb), (ya, yb), h = CAB["x"], CAB["y"], CAB["h"]
    xc, yc = (xa + xb) / 2, (ya + yb) / 2
    C._bx(b, "qc_cab_body", (xb - xa, yb - ya, h - 0.08), (xc, yc, 0.08 + (h - 0.08) / 2), M["cabinet"], bev=0.006)
    C._bx(b, "qc_cab_plinth", (xb - xa - 0.02, yb - ya - 0.02, 0.08), (xc, yc, 0.04), M["cabinet_dark"])
    # door on the +X face (toward the loading side), louvres, handle, e-stop, key switch, status lamps
    xf = xb
    C._bx(b, "qc_cab_door_gap", (0.004, yb - ya - 0.04, h - 0.16), (xf + 0.001, yc, 0.08 + (h - 0.08) / 2), M["cabinet_dark"])
    C._bx(b, "qc_cab_door", (0.012, yb - ya - 0.05, h - 0.17), (xf + 0.007, yc, 0.08 + (h - 0.08) / 2), M["cabinet"], bev=0.003)
    for k in range(5):
        C._bx(b, f"qc_cab_louvre{k}", (0.008, 0.16, 0.012), (xf + 0.016, yc, 0.20 + 0.04 * k), M["cabinet_dark"])
    C._bx(b, "qc_cab_handle", (0.02, 0.02, 0.10), (xf + 0.024, yb - 0.06, 0.62), M["dark"], bev=0.004)
    E._rod(b, "qc_cab_estop_ring", 0.03, 0.006, M["yellow"], (xf + 0.016, ya + 0.08, 0.95), (0, math.pi / 2, 0), 20)
    E._rod(b, "qc_cab_estop", 0.02, 0.022, M["red"], (xf + 0.025, ya + 0.08, 0.95), (0, math.pi / 2, 0), 20)
    E._rod(b, "qc_cab_key", 0.012, 0.018, M["dark"], (xf + 0.022, ya + 0.16, 0.95), (0, math.pi / 2, 0), 12)
    C._bx(b, "qc_cab_led_power", (0.004, 0.018, 0.018), (xf + 0.015, ya + 0.08, 1.01), M["green_led"])
    C._bx(b, "qc_cab_led_laser", (0.004, 0.018, 0.018), (xf + 0.015, ya + 0.12, 1.01), M["laser_led"])
    C._label(b, "qc_cab_label", lm, 2, (0.18, 0.072), (xf + 0.0135, yc, 0.80), (math.pi / 2, 0, math.pi / 2))
    # conduit: cabinet top -> +Y post (+X face) -> beam top at the chain's fixed end
    xc_, yc_ = 0.11, Y1 + 0.03
    pts = [(xc_, yc - 0.05, h), (xc_, yc - 0.05, h + 0.08), (xc_, yc_, h + 0.25), (xc_, yc_, BEAM_TOP + 0.03), (xc_, CHAIN_YF + 0.04, BEAM_TOP + 0.03),
           (CHAIN_X, CHAIN_YF + 0.04, BEAM_TOP + 0.03)]
    for k in range(len(pts) - 1):
        E._bar(b, f"qc_conduit{k}", pts[k], pts[k + 1], 0, M["cable"], radius=0.016, segs=10)
        if k:
            E._rod(b, f"qc_conduit_bend{k}", 0.0175, 0.035, M["cable"], pts[k], segs=10)
    for k in range(3):
        z = h + 0.40 + 0.18 * k
        C._bx(b, f"qc_conduit_clip{k}", (0.065, 0.04, 0.02), (POST / 2 + 0.0325, yc_, z), M["frame"])


def _hmi(b, M, lm):
    """HMI panel on the -Y post's +Y face (facing +Y), screen property 'state'."""
    (xa, xb), (za, zb), t = HMI["x"], HMI["z"], HMI["t"]
    yb = Y0 + POST / 2
    C._bx(b, "qc_hmi_body", (xb - xa, t, zb - za), ((xa + xb) / 2, yb + t / 2, (za + zb) / 2), M["charcoal"], bev=0.004)
    w, h = (xb - xa) - 0.03, (zb - za) - 0.045
    ob = E._mesh(b, "qc_display", [(-w / 2, 0, -h / 2), (w / 2, 0, -h / 2), (w / 2, 0, h / 2), (-w / 2, 0, h / 2)], [(0, 1, 2, 3)],
                 _display_mat(), ((xa + xb) / 2, yb + t + 0.0008, (za + zb) / 2 + 0.01))
    # quad in local XZ, facing -Y by winding; mirror so the normal faces +Y and the image reads from +Y
    ob.rotation_euler = (0.0, 0.0, math.pi)
    uvl = ob.data.uv_layers.new(name="UVMap")
    uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
    for li, lp in enumerate(ob.data.loops):
        uvl.data[li].uv = uvs[lp.vertex_index]
    ob["state"] = 0.0
    for k, x in enumerate((xa + 0.03, xa + 0.06)):
        E._rod(b, f"qc_hmi_btn{k}", 0.008, 0.008, M["black"] if k else M["red"], (x, yb + t + 0.004, za + 0.016), (math.pi / 2, 0, 0), 12)
    C._bx(b, "qc_hmi_bracket", (0.05, 0.02, 0.10), ((xa + xb) / 2, yb + 0.004, za - 0.03), M["frame"])
    C._label(b, "qc_hmi_label", lm, 1, (0.10, 0.04), (0.0, yb + 0.0015, zb + 0.10), (math.pi / 2, 0, math.pi), backing=M["frame"])
    # cable down the post face into a gland on the base plate
    E._bar(b, "qc_hmi_cable", ((xa + xb) / 2 + 0.05, yb + 0.012, za), ((xa + xb) / 2 + 0.05, yb + 0.012, 0.03), 0, M["cable"], radius=0.005,
           segs=8)
    E._rev(b, "qc_hmi_gland", [(0, 0), (0.011, 0), (0.011, 0.012), (0.008, 0.018), (0, 0.018)], M["dark"],
           ((xa + xb) / 2 + 0.05, yb + 0.012, 0.015), segs=12)
    return ob


# ============================================================================ carriage + heads (built at carriage y = 0)
def _carriage(b, M):
    za, zb = PLATE_Z
    C._bx(b, "qc_car_plate", (PLATE_X[1] - PLATE_X[0], 2 * PLATE_HY, zb - za), (sum(PLATE_X) / 2, 0.0, (za + zb) / 2), M["alu"], bev=0.003)
    for k, (sx, sy) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
        C._bx(b, f"qc_car_block{k}", BLOCK, (sx * RAIL_X, sy * 0.075, zb + BLOCK[2] / 2), M["dark"], bev=0.003)
        C._bx(b, f"qc_car_seal{k}", (BLOCK[0] + 0.004, 0.006, BLOCK[2] - 0.006), (sx * RAIL_X, sy * (0.075 + BLOCK[1] / 2 + 0.003),
              zb + BLOCK[2] / 2 - 0.002), M["red"])
    for k, (sx, sy) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
        C._hex(b, f"qc_car_bolt{k}", 0.008, 0.005, M["dark"], (sx * 0.05, sy * 0.105, za), (math.pi, 0, 0))
    # servo drive on the +X side: riser plate, motor plate, pinion on the rack, servo motor + gearbox, connector
    C._bx(b, "qc_car_riser", (0.015, 0.14, 0.15), (0.1775, 0.0, zb + 0.07), M["alu"], bev=0.002)
    C._bx(b, "qc_car_riser_foot", (0.07, 0.14, 0.012), (0.15, 0.0, zb + 0.006), M["alu"], bev=0.002)
    C._bx(b, "qc_car_motorplate", (0.10, 0.14, 0.015), (0.135, 0.0, 2.0775), M["alu"], bev=0.002)
    E._rev(b, "qc_pinion", [(0, 0), (0.03, 0), (0.03, 0.03), (0, 0.03)], M["steel"], (0.125, 0.0, BEAM_BOT + 0.015), segs=24, smooth=False)
    E._rod(b, "qc_pinion_shaft", 0.01, 0.04, M["steel"], (0.125, 0.0, 2.05), segs=12)
    E._rev(b, "qc_servo_gear", [(0, 0), (0.035, 0), (0.035, 0.05), (0, 0.05)], M["charcoal"], (0.125, 0.0, 2.085), segs=24)
    C._bx(b, "qc_servo", (0.07, 0.07, 0.13), (0.125, 0.0, 2.135 + 0.065), M["black"], bev=0.006)
    C._bx(b, "qc_servo_cap", (0.074, 0.074, 0.012), (0.125, 0.0, 2.271), M["charcoal"], bev=0.003)
    E._rod(b, "qc_servo_conn", 0.009, 0.03, M["steel"], (0.125, 0.036 + 0.015, 2.24), (math.pi / 2, 0, 0), 12)
    # energy-chain bracket on the -X side: riser and top arm over the beam to the chain's moving end
    C._bx(b, "qc_car_chainpost", (0.02, 0.05, 2.37 - zb), (-0.105, 0.0, (2.37 + zb) / 2), M["alu"], bev=0.002)
    C._bx(b, "qc_car_chainarm", (0.10, 0.07, 0.01), (-0.065, 0.0, CHAIN_Z + 2 * CHAIN_R + CHAIN_H / 2 + 0.005), M["alu"], bev=0.002)


def _marker_frame():
    """Marking head frame (carriage coords): origin at the lens exit, +Z up the optical axis, X in the X-Z plane."""
    o = np.array([LENS[0], 0.0, LENS[1]])
    target = np.array([0.0, 0.0, SPOOL_TOP])
    return kin.frame((1.0, 0.0, 0.0), o - target, o)


def _scanner_frame():
    o = np.array([WIN[0], 0.0, WIN[1]])
    target = np.array([0.0, 0.0, SPOOL_TOP])
    return kin.frame((1.0, 0.0, 0.0), o - target, o)


def _marker_head(col, M, lm):
    """Galvo marking head in its own frame; returns (objects, beam, led)."""
    b = E._B(col)
    E._rev(b, "qc_mk_lens", [(0, 0.004), (0.028, 0.004), (0.03, 0.0), (0.036, 0.0), (0.036, 0.05), (0, 0.05)], M["black"], segs=32)
    E._rev(b, "qc_mk_lens_ring", [(0.03, -0.0005), (0.037, -0.0005), (0.037, 0.008), (0.03, 0.008)], M["steel"], segs=32)
    E._rev(b, "qc_mk_lens_glass", [(0, 0.0035), (0.028, 0.0035), (0.028, 0.005), (0, 0.005)], M["glass"], segs=32)
    C._bx(b, "qc_mk_galvo", (0.13, 0.12, 0.11), (-0.01, 0.0, 0.05 + 0.055), M["charcoal"], bev=0.006)
    C._bx(b, "qc_mk_galvo_band", (0.132, 0.122, 0.012), (-0.01, 0.0, 0.07), M["blue"], bev=0.002)
    E._rod(b, "qc_mk_expander", 0.022, 0.13, M["alu"], (-0.075 - 0.065, 0.0, 0.11), (0, math.pi / 2, 0), 24)
    E._rod(b, "qc_mk_collimator", 0.016, 0.04, M["black"], (-0.205 - 0.02, 0.0, 0.11), (0, math.pi / 2, 0), 16)
    E._rod(b, "qc_mk_fiber_conn", 0.01, 0.025, M["steel"], (-0.245 - 0.0125, 0.0, 0.11), (0, math.pi / 2, 0), 12)
    led = C._bx(b, "qc_mk_led", (0.02, 0.012, 0.012), (0.03, 0.0, 0.162), M["laser_led"])
    led["on"] = 0.0
    C._label(b, "qc_mk_label", lm, 1, (0.08, 0.032), (-0.01, 0.0605, 0.10), (math.pi / 2, 0, math.pi), backing=None)
    C._bx(b, "qc_mk_adapter", (0.06, 0.10, 0.02), (-0.03, 0.0, 0.17), M["alu"], bev=0.002)
    # beam: unit line along +Y (Stretch To aims +Y at the focus and scales it to the distance)
    made = []
    for name, r, nv, mat in (("qc_beam", 0.0008, 8, _beam_mat("qc_beam_core", (1.0, 0.72, 0.42), 60.0, 1.0)),
                             ("qc_beam_glow", 0.0035, 12, _beam_mat("qc_beam_halo", (1.0, 0.45, 0.15), 6.0, 0.35))):
        bpy.ops.object.select_all(action='DESELECT')
        ob = G.cylinder(name, r, 1.0, location=(0, 0.5, 0), rotation=(math.pi / 2, 0, 0), vertices=nv, collection=col, smooth=False)
        bpy.context.view_layer.objects.active = ob
        ob.select_set(True)
        bpy.ops.object.transform_apply(location=True, rotation=True)
        ob.data.materials.append(mat)
        ob["on"] = 0.0
        ob.visible_shadow = False
        b.objs.append(ob)
        made.append(ob)
    core, glow = made
    glow.parent = core
    return b.objs, core, glow, led


def _scanner_head(col, M):
    """Laser-triangulation profile sensor in its own frame (origin = laser window, +Z up the laser axis)."""
    b = E._B(col)
    C._bx(b, "qc_sc_body", (0.09, 0.06, 0.12), (0.02, 0.0, 0.0605), M["blue2"], bev=0.006)
    C._bx(b, "qc_sc_face", (0.092, 0.062, 0.012), (0.02, 0.0, 0.006), M["black"], bev=0.002)
    C._bx(b, "qc_sc_laserwin", (0.024, 0.036, 0.002), (0.0, 0.0, -0.0005), M["red_glass"])
    C._bx(b, "qc_sc_camwin", (0.03, 0.036, 0.002), (0.047, 0.0, -0.0005), M["glass"])
    C._bx(b, "qc_sc_led", (0.01, 0.003, 0.006), (0.05, 0.0315, 0.10), M["green_led"])
    C._bx(b, "qc_sc_led2", (0.01, 0.003, 0.006), (0.035, 0.0315, 0.10), M["blue_led"])
    E._rod(b, "qc_sc_conn", 0.008, 0.025, M["steel"], (0.045, 0.0, 0.133), segs=12)
    C._bx(b, "qc_sc_mount", (0.04, 0.07, 0.03), (-0.005, 0.0, 0.135), M["alu"], bev=0.002)
    return b.objs


def _curve_tube(col, name, pts, radius, mat):
    cu = bpy.data.curves.new(name, 'CURVE')
    cu.dimensions = '3D'
    cu.bevel_depth = radius
    cu.bevel_resolution = 2
    cu.resolution_u = 8
    sp = cu.splines.new('BEZIER')
    sp.bezier_points.add(len(pts) - 1)
    for bp, p in zip(sp.bezier_points, pts):
        bp.co = p
        bp.handle_left_type = bp.handle_right_type = 'AUTO'
    ob = bpy.data.objects.new(name, cu)
    col.objects.link(ob)
    ob.data.materials.append(mat)
    return ob


def _chain(col, carriage, root):
    """Energy chain: lower run (fixed end at CHAIN_YF), U-loop and upper run to the carriage, all driven by the carriage y."""
    w, h, R = CHAIN_W, CHAIN_H, CHAIN_R
    # link grooves: the lower run lies still (its length changes through its y scale, so the grooves are mapped in the
    # unscaled root space); the upper run travels with the carriage
    mat_s = _chain_mat("qc_chain_static", root)
    mat_m = _chain_mat("qc_chain_moving", carriage)

    def unit_box(name, sign, z, mat):
        v = [(x, y, zz) for y in (0.0, sign * 1.0) for x in (-w / 2, w / 2) for zz in (-h / 2, h / 2)]
        f = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        ob = G.new_object(name, v, f, col, smooth=False)
        E._recalc(ob)
        ob.data.materials.append(mat)
        ob.location = (CHAIN_X, CHAIN_YF, z)
        return ob

    lower = unit_box("qc_chain_lower", -1, CHAIN_Z, mat_s)
    upper = unit_box("qc_chain_upper", 1, CHAIN_Z + 2 * R, mat_m)
    # half annulus (rectangular section) in the Y-Z plane, bulging toward -Y
    n = 16
    v, f = [], []
    for i in range(n + 1):
        a = math.pi / 2 + math.pi * i / n
        for rr in (R - h / 2, R + h / 2):
            for x in (-w / 2, w / 2):
                v.append((x, rr * math.cos(a), rr * math.sin(a)))
    for i in range(n):
        a0, a1 = 4 * i, 4 * (i + 1)
        f += [(a0, a0 + 1, a1 + 1, a1), (a0 + 2, a1 + 2, a1 + 3, a0 + 3), (a0, a1, a1 + 2, a0 + 2), (a0 + 1, a0 + 3, a1 + 3, a1 + 1)]
    loop = G.new_object("qc_chain_loop", v, f, col, smooth=False)
    E._recalc(loop)
    loop.data.materials.append(mat_m)
    loop.location = (CHAIN_X, CHAIN_YF, CHAIN_Z + R)
    for ob in (lower, upper, loop):
        ob.parent = root
        ob.matrix_parent_inverse = mathutils.Matrix.Identity(4)
    yl = f"({CHAIN_YF:.4f} + y - {CHAIN_C:.4f}) / 2"
    _driver(lower, "scale", 1, f"({CHAIN_YF:.4f} - y + {CHAIN_C:.4f}) / 2", carriage)
    _driver(loop, "location", 1, yl, carriage)
    _driver(upper, "location", 1, yl, carriage)
    _driver(upper, "scale", 1, f"(y - {CHAIN_YF:.4f} + {CHAIN_C:.4f}) / 2", carriage)
    return [lower, upper, loop]


def _driver(ob, path, index, expr, src, prop="location[1]", var="y"):
    fc = ob.driver_add(path, index)
    d = fc.driver
    d.type = 'SCRIPTED'
    for v in list(d.variables):
        d.variables.remove(v)
    v = d.variables.new()
    v.name = var
    v.type = 'SINGLE_PROP'
    v.targets[0].id_type = 'OBJECT'
    v.targets[0].id = src
    v.targets[0].data_path = prop
    d.expression = expr
    return fc


# ============================================================================ build
def _root_tr():
    return kin.tr(QX, 0.0, 0.0)


def build(collection=None):
    col = collection or bpy.data.collections.new(NAME)
    if collection is None:
        bpy.context.scene.collection.children.link(col)
    M = _mats()
    # arch / cabinet labels; cell 1 (laser warning) in the current language (i18n2.EN_LATIN), hence the name
    lm = C._label_mat(f"qc_label_{i18n2.get_lang()}", ["QC-01", i18n2.tr("LASER KL.4"), "LASER 50W"])
    Rm = G.M(_root_tr())
    # ---- static portal: built in root-local coordinates, then moved under the root
    b = E._B(col)
    _posts(b, M)
    _beam(b, M, lm)
    lamps = _tower(b, M)
    _cabinet(b, M, lm)
    display = _hmi(b, M, lm)
    root = G.empty("qc_root", collection=col, size=0.5)
    for ob in b.objs:
        ob.parent = root
        ob.matrix_parent_inverse = mathutils.Matrix.Identity(4)
    root.matrix_world = Rm
    # ---- carriage (identity under the root; its location[1] = world y)
    car = G.empty("qc_carriage", collection=col, size=0.2, display='ARROWS')
    car.parent = root
    bc = E._B(col)
    _carriage(bc, M)
    mk_objs, beam, glow, led = _marker_head(col, M, lm)
    Fm = _marker_frame()
    head_objs = [o for o in mk_objs if o is not glow]
    C._to_frame(head_objs, Fm)
    sc_objs = _scanner_head(col, M)
    Fs = _scanner_frame()
    C._to_frame(sc_objs, Fs)
    # hangers: marking head Z-slide (with hand wheel) and scanner bracket, both from the carriage plate
    top = PLATE_Z[0]
    mk_top = (Fm @ np.array([-0.03, 0.0, 0.18, 1.0]))[:3]
    C._bx(bc, "qc_mk_slide", (0.03, 0.09, top - mk_top[2] + 0.01), (mk_top[0], 0.0, (top + mk_top[2] - 0.01) / 2), M["frame"], bev=0.002)
    C._bx(bc, "qc_mk_slide_rail", (0.012, 0.05, top - mk_top[2] - 0.02), (mk_top[0] - 0.021, 0.0, (top + mk_top[2]) / 2), M["steel"])
    E._rod(bc, "qc_mk_wheel", 0.028, 0.01, M["black"], (mk_top[0], 0.05, top - 0.03), (math.pi / 2, 0, 0), 20)
    E._rod(bc, "qc_mk_wheel_hub", 0.006, 0.03, M["steel"], (mk_top[0], 0.045, top - 0.03), (math.pi / 2, 0, 0), 10)
    sc_top = (Fs @ np.array([-0.005, 0.0, 0.15, 1.0]))[:3]
    C._bx(bc, "qc_sc_hanger", (0.03, 0.07, top - sc_top[2] + 0.005), (sc_top[0], 0.0, (top + sc_top[2] - 0.005) / 2), M["frame"], bev=0.002)
    # cables: fiber (armoured, yellow) from the collimator to the chain bracket, scanner cable, servo cable
    fib0 = (Fm @ np.array([-0.27, 0.0, 0.11, 1.0]))[:3]
    fiber = _curve_tube(col, "qc_fiber", [tuple(fib0), (fib0[0] - 0.03, 0.0, fib0[2] + 0.10), (-0.30, 0.0, 1.96), (-0.20, 0.0, 2.06),
                                          (-0.125, 0.0, 2.20), (-0.118, 0.0, 2.33)], 0.0065, M["yellow"])
    scc0 = (Fs @ np.array([0.045, 0.0, 0.145, 1.0]))[:3]
    scab = _curve_tube(col, "qc_sc_cable", [tuple(scc0), (scc0[0] + 0.01, 0.05, scc0[2] + 0.05), (0.10, 0.12, top - 0.02), (0.0, 0.155, top - 0.005),
                                            (-0.09, 0.15, top + 0.05), (-0.10, 0.05, 2.20), (-0.10, 0.015, 2.33)], 0.004, M["cable"])
    scab2 = _curve_tube(col, "qc_servo_cable", [(0.125, 0.066, 2.24), (0.10, 0.10, 2.30), (0.0, 0.11, 2.38), (-0.09, 0.03, 2.37)], 0.005, M["cable"])
    car_objs = bc.objs + head_objs + sc_objs + [fiber, scab, scab2]
    for ob in car_objs:
        ob.parent = car
        ob.matrix_parent_inverse = mathutils.Matrix.Identity(4)
    glow.parent = beam
    # ---- scanner sensor empty: +Z down, Y along the pipe axis (world -Y), window at LASER_WINDOW_FALLBACK
    sensor = G.empty("qc_sensor", collection=col, size=0.06, display='ARROWS')
    sensor.parent = car
    sensor.matrix_basis = G.M(kin.tr(0.0, 0.0, SENSOR_Z) @ kin.rotx(math.pi))
    # ---- beam focus (default target: the crest under the lens axis), spot light
    focus = G.empty("qc_beam_focus", collection=col, size=0.03)
    focus.parent = car
    focus.location = (0.0, 0.0, SPOOL_TOP)
    con = beam.constraints.new('STRETCH_TO')
    con.target = focus
    con.rest_length = 1.0
    con.volume = 'NO_VOLUME'
    con.keep_axis = 'SWING_Y'
    ld = bpy.data.lights.new("qc_mark_spot", 'POINT')
    ld.energy = 0.0
    ld.color = (1.0, 0.72, 0.45)
    ld.shadow_soft_size = 0.003
    spot = bpy.data.objects.new("qc_mark_spot", ld)
    col.objects.link(spot)
    spot.parent = focus
    spot.location = (0.0, 0.0, 0.012)
    # ---- energy chain (drivers on the carriage y)
    chain = _chain(col, car, root)
    qc = dict(collection=col, root=root, carriage=car, marker=dict(objects=head_objs, frame=Fm), beam=beam, beam_glow=glow,
              scanner=dict(objects=sc_objs, frame=Fs), sensor=sensor, tower=lamps, display=display, spot=spot, led=led,
              focus=focus, chain=chain, objects=b.objs + car_objs, carriage_range=CAR_RANGE)
    set_carriage(qc, Y0 + 0.25)
    bpy.context.view_layer.update()
    return qc


# ============================================================================ decal
def _mark_collection():
    col = bpy.data.collections.get(NAME)
    if col is None:
        col = bpy.data.collections.new(NAME)
        bpy.context.scene.collection.children.link(col)
    return col


def build_mark(pipe_root):
    """Laser-mark decal wrapped on the pipe top around QC_MARK_LOCAL, parented to pipe_root (its frame = spool frame).
    Also re-targets the marker beam and spot light (if the arch exists) to the decal's writing front."""
    col = _mark_collection()
    xa, xb = MARK_X
    zc = L2.PIPE_AXIS_Z
    span = MARK_W / MARK_R
    nu, nv = 26, 8
    v, uv, f = [], [], []
    for i in range(nu + 1):
        x = xa + (xb - xa) * i / nu
        for j in range(nv + 1):
            a = -span / 2 + span * j / nv
            v.append((x, MARK_R * math.sin(a), zc + MARK_R * math.cos(a)))
            uv.append((i / nu, j / nv))
    for i in range(nu):
        for j in range(nv):
            a = i * (nv + 1) + j
            f.append((a, a + nv + 1, a + nv + 2, a + 1))
    ob = G.new_object("qc_mark", v, f, col, smooth=True)
    me = ob.data
    if me.polygons[0].normal.z < 0:
        me.flip_normals()
    uvl = me.uv_layers.new(name="UVMap")
    for poly in me.polygons:
        for li in poly.loop_indices:
            uvl.data[li].uv = uv[me.loops[li].vertex_index]
    me.materials.append(_mark_mat())
    ob.visible_shadow = False
    ob["reveal"] = 0.0
    ob["hot"] = 0.0
    ob.parent = pipe_root
    ob.matrix_parent_inverse = mathutils.Matrix.Identity(4)
    ob.matrix_basis = mathutils.Matrix.Identity(4)
    # writing front: x follows 'reveal', y a raster wobble across the text height
    front = G.empty("qc_mark_front", collection=col, size=0.02)
    front.parent = ob
    front.location = (xa, 0.0, zc + MARK_R)
    amp, frq = MARK_WOBBLE
    _driver(front, "location", 0, f"{xa:.4f} + {xb - xa:.4f} * r", ob, prop='["reveal"]', var="r")
    _driver(front, "location", 1, f"{amp:.4f} * sin({frq:.1f} * r)", ob, prop='["reveal"]', var="r")
    ob["front"] = front
    beam = bpy.data.objects.get("qc_beam")
    spot = bpy.data.objects.get("qc_mark_spot")
    if beam is not None:
        for c in beam.constraints:
            if c.type == 'STRETCH_TO':
                c.target = front
    if spot is not None:
        spot.parent = front
        spot.matrix_parent_inverse = mathutils.Matrix.Identity(4)
        spot.location = (0.0, 0.0, 0.012)
    return ob


# ============================================================================ setters
def _key_const(idb, path, frame):
    C.key_constant(idb, path, frame)


def set_carriage(qc, y, frame=None):
    """Carriage (both heads' working plane) at world y, clamped to the travel range.  Keys location[1] only."""
    car = qc["carriage"]
    lo, hi = qc.get("carriage_range", CAR_RANGE)
    car.location[1] = min(max(float(y), lo), hi)
    if frame is not None:
        car.keyframe_insert(data_path="location", index=1, frame=frame)


def set_marker(qc, on, frame=None):
    """Marking laser on / off: beam + glow + emission LED property 'on', spot light energy (CONSTANT keys)."""
    v = float(on)
    for ob in (qc["beam"], qc["beam_glow"], qc["led"]):
        ob["on"] = v
        ob.update_tag()
        if frame is not None:
            _key_const(ob, '["on"]', frame)
    ld = qc["spot"].data
    ld.energy = SPOT_POWER * v
    if frame is not None:
        _key_const(ld, "energy", frame)


def set_mark(decal, reveal, hot, frame=None):
    """Decal wipe 'reveal' (0..1, left to right) and glow 'hot' (0..1)."""
    decal["reveal"] = min(max(float(reveal), 0.0), 1.0)
    decal["hot"] = min(max(float(hot), 0.0), 1.0)
    decal.update_tag()                  # ID-property edits do not tag the depsgraph (the writing-front drivers read 'reveal')
    if frame is not None:
        decal.keyframe_insert(data_path='["reveal"]', frame=frame)
        decal.keyframe_insert(data_path='["hot"]', frame=frame)


TOWER_STATES = ("off", "amber", "green", "red")


def set_tower(qc, state, frame=None):
    """Stack light: "off" | "amber" | "green" | "red" (one lamp on).  CONSTANT keys on the three lamps."""
    if state not in TOWER_STATES:
        raise ValueError(f"tower state {state!r} not in {TOWER_STATES}")
    for key, ob in qc["tower"].items():
        ob["on"] = 1.0 if key == state else 0.0
        ob.update_tag()
        if frame is not None:
            _key_const(ob, '["on"]', frame)


def set_display(qc, state, frame=None):
    """HMI screen: 0 idle, 1 marking, 2 scanning, 3 QC OK (or the names in DISPLAY_STATES).  CONSTANT key."""
    k = DISPLAY_STATES.index(state) if isinstance(state, str) else int(state)
    if not 0 <= k < len(DISPLAY_STATES):
        raise ValueError(f"display state {state!r}")
    ob = qc["display"]
    ob["state"] = float(k)
    ob.update_tag()
    if frame is not None:
        _key_const(ob, '["state"]', frame)


def scan_line(qc, intervals, collection=None):
    """vfx.laser_line for the QC scanner.  vfx takes the emitter window from an object named 'torch_laser_lens' when one
    exists (the stage-1 welding torch in the full scene); it is renamed for the duration of the call so the QC line
    starts at the scanner's own window (vfx.LASER_WINDOW_FALLBACK in the sensor frame)."""
    lens = bpy.data.objects.get("torch_laser_lens")
    if lens is not None:
        lens.name = "torch_laser_lens__qc_tmp"
    try:
        return vfx.laser_line(qc["sensor"], intervals, collection)
    finally:
        if lens is not None:
            lens.name = "torch_laser_lens"


# ============================================================================ obstacles (pure python)
def obstacles():
    """Conservative world AABBs (dict(name, center, size, yaw=0)).  The carriage box covers its whole travel."""
    out = []

    def add(name, lo, hi):
        out.append(dict(name=name, center=tuple((lo[k] + hi[k]) / 2 for k in range(3)),
                        size=tuple(hi[k] - lo[k] for k in range(3)), yaw=0.0))

    x = QX
    add("qc_post_m", (x - 0.16, Y0 - 0.095, 0.0), (x + 0.16, Y0 + POST / 2 + HMI["t"] + 0.012, BEAM_BOT))
    add("qc_post_p", (x - 0.16, Y1 - 0.095, 0.0), (x + 0.16, Y1 + 0.095, BEAM_BOT))
    add("qc_beam", (x - BEAM_W / 2 - 0.03, BEAM_Y[0] - 0.01, BEAM_BOT - 0.035), (x + BEAM_W / 2 + 0.03, BEAM_Y[1] + 0.01, BEAM_TOP + 0.01))
    add("qc_top", (x - 0.09, BEAM_Y[0], BEAM_TOP), (x + 0.13, BEAM_Y[1], BEAM_TOP + 0.37))         # tray, chain, stack light, conduit
    add("qc_carriage", (x - 0.42, CAR_RANGE[0] - PLATE_HY - 0.03, SPOOL_TOP + 0.19),
        (x + 0.21, CAR_RANGE[1] + PLATE_HY + 0.03, 2.40))
    (xa, xb), (_, yb), h = CAB["x"], CAB["y"], CAB["h"]
    add("qc_cabinet", (x + xa - 0.01, Y1 + POST / 2, 0.0), (x + xb + 0.04, yb + 0.01, h + 0.30))
    return out
