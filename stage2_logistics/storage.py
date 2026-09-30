"""Stepped storage rack for finished spools (2 tiers, staggered bays) at the -X end of the logistics zone + an AGV.

Layout from layout2 `STORAGE` / `storage_frame(tier, bay)` / `AGV`:

* tier 0 (front, low): seat girder at x = STORAGE.x[0] (flange seats at seat_z[0]), V-posts under the pipe legs at
  spool-local x = 0.86 standing on a floor rail;
* tier 1 (back, high): seat girder at x = STORAGE.x[1] on columns; its V-posts stand on the tier-0 girder in the gaps
  between the tier-0 spools (the tier-1 bays are staggered half a pitch) and are tied back to the tier-1 girder below
  the tier-1 seats;
* every bay: machined seat ring under the flange back face (+3 centring pins) and a V-head (steel V + polymer liners,
  45 deg flanks) touching the pipe bottom near local x = 0.86; bay number plates A1..A6 (tier 0), B1..B5 (tier 1);
* back side (x = BACK_X = -11.24): end columns, header with a sign and a mesh transom above the tier-1 spools, a hazard
  bumper at the floor - it closes the fence opening `storage_back` (the light curtain itself is environment2's); the
  sign (both faces) reads [logo | text]: branding2's current brand on a white plate, the text in i18n2's current
  language (texture / material sto_sign_<lang>_<brand>, drawn by build(); boxes: sign_logo_box, sign_text_box);
* AGV (outside the fence, parked at AGV["park"]): chassis, drive wheels + casters, 2 lidars, bumpers, LED band,
  E-stops, lifting table (`sto_agv_lift`), amber beacon (`sto_agv_beacon`, object property 'beacon_on').

Public API (DESIGN.md section 3, storage.py):
    build(collection=None) -> dict(collection, rack, bays, agv)
        rack = dict(root=<empty 'sto_rack_root'>, objects=[...], labels={(tier, bay): obj})
        bays = {(tier, bay): T}    spool frame (numpy 4x4) = kin.planar_frame(*layout2.storage_frame(tier, bay))
        agv  = dict(root=<empty 'sto_agv_root'>, lift=<empty 'sto_agv_lift'>, beacon=<mesh 'sto_agv_beacon'>,
                    objects=[...]);  root: location (x, y, 0), rotation_euler z = yaw, AGV forward = local +X
    set_agv(st, x, y, yaw, frame=None)     keys root location[0], location[1], rotation_euler[2] only
    set_agv_lift(st, h, frame=None)        lifting-table stroke h in [0, AGV_LIFT_MAX] m (lift.location[2])
    set_beacon(st, on, frame=None)         beacon on/off (property 'beacon_on', CONSTANT key)
    obstacles() -> list[dict(name, center, size, yaw)]   conservative world AABBs of the rack and the parked AGV
"""
import math

import numpy as np
import bpy

import tools  # noqa: F401  (demo_video on sys.path)
from cell import geom as G
from cell import materials
from cell import environment as E
from cell import positioner as P1
import layout2 as L2
import kin
import cassette as C
import fonts2
import i18n2
import branding2

NAME = "Storage"
GAP = C.GAP

ST = L2.STORAGE
TIER_X = ST["x"]                         # (-10.10, -10.90) flange centre x per tier
SEAT_Z = ST["seat_z"]                    # (0.35, 1.00)
BAY_Y = ST["bay_y"]
VPOST_LX = 0.86                          # spool-local x of the V-post under the pipe
RING = (0.193, 0.222)                    # seat ring inner / outer radius (as the kit nests)
RING_BASE_R = 0.245
RING_H = 0.030                           # ring body height; base disc 0.010 below it
GIRDER_W = 0.30
GIRDER_TOP = tuple(z - RING_H - 0.010 for z in SEAT_Z)     # 0.31 / 0.96
GIRDER_H = (0.12, 0.14)
OPEN_Y = L2.LOG_FENCE_OPENINGS[2]["y1"]  # 2.1: storage_back opening half width (the rack stays inside it)
# end frames at y = +-RACK_Y: just outside the seat base discs of the outermost tier-0 bays (2.05 for the layout numbers)
RACK_Y = max(abs(y) for y in ST["bay_y"][0]) + RING_BASE_R + 0.005
RACK_Y_EXT = min(RACK_Y + 0.04, OPEN_Y - 0.01)             # girder / rail ends (2.09)
END_Y = RACK_Y_EXT - 0.05                # end legs / columns / back columns (0.10 wide) flush with the girder ends
# back columns / header / transom.  The fence line is L2.LOG_FENCE_X0 = -11.4; environment2's light-curtain columns
# stand on 0.2 m base plates at (LOG_FENCE_X0, +-OPEN_Y): the back rail (0.10 wide) keeps 1 cm off those plates
BACK_X = L2.LOG_FENCE_X0 + 0.10 + 0.05 + 0.01               # -11.24
BUMPER_HALF_Y = OPEN_Y - 0.10 - 0.01     # hazard bumper at the fence line, between the light-curtain base plates
BACK_H = L2.LOG_FENCE_H
TRANSOM_Z = 1.72                         # mesh infill between the tier-1 spools' tops and the header
FRONT_RAIL_X = TIER_X[0] + VPOST_LX      # tier-0 V-post line
POST = 0.07                              # V-post section
V_W = 0.12                               # V-head half width (y)
V_BOT = 0.04                             # V-head body height below the apex
RAIL = (0.10, 0.08)                      # floor rail section (width, height)
TIE_Z = (GIRDER_TOP[1] - GIRDER_H[1] + 0.003, GIRDER_TOP[1] - GIRDER_H[1] + 0.043)   # tier-1 post tie (z range)
AGV_SIZE = (1.30, 0.84)                  # chassis footprint (x forward, y)
AGV_DECK_Z = 0.30                        # top of the chassis; lift plate rests on it at h = 0
AGV_LIFT_MAX = 0.10
LEGS_Y = (-END_Y,) + tuple(BAY_Y[1]) + (END_Y,)           # tier-0 girder legs (under the tier-1 posts + ends)
COLS_Y = (-END_Y + 0.01, -1.08, 0.0, 1.08, END_Y - 0.01)    # tier-1 girder columns (0.12 wide)
SIGN_PX = (1800, 118)                    # header sign texture (quad 1.30 x 0.085 m)
SIGN_LOGO_PAD = 0.12                     # logo plate margin (fraction of its height, branding2.logo_plate)


def _mats():
    return dict(
        blue=materials.get("painted", color="#1F4E8C", roughness=0.42),
        frame=materials.get("painted", color="#2B2F36", roughness=0.5),
        charcoal=materials.get("painted", color="#3A3E44", roughness=0.45, coat=0.1),
        yellow=materials.get("safety_yellow"),
        red=materials.get("safety_red"),
        dark=materials.get("dark_metal"),
        steel=materials.get("machined_steel"),
        black=materials.get("black_plastic"),
        rubber=materials.get("rubber"),
        glass=materials.get("glass_dark"),
        uhmw=materials.get("painted", color="#C9C3AF", roughness=0.6, coat=0.0),
        led_blue=materials.get("emissive", color="#39B8FF", strength=6.0),
        hazard=E._mat_hazard(),
        fence=materials.get("fence_mesh", color="#2A2A2A", pitch=0.08, wire=0.005),
        screen=E._mat_screen(),
        beacon=_beacon_mat(),
    )


def _beacon_mat():
    """Amber dome: orange plastic, emission = property 'beacon_on' x 5 (brighter washes out to white under AgX)."""
    m = C._state_mat("sto_beacon", (1.0, 0.30, 0.01), 5.0, "beacon_on")
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.55, 0.22, 0.02, 1.0)
    b.inputs["Roughness"].default_value = 0.2
    b.inputs["Coat Weight"].default_value = 0.4
    return m


def sign_logo_box():
    """(x0, y0, x1, y1) px of the logo plate on the header sign texture: left end, 16 px inside the edges."""
    W, H = SIGN_PX
    ph = H - 32
    return (22, 16, 22 + ph * branding2.logo_aspect(pad=SIGN_LOGO_PAD), 16 + ph)


def _sign_divider_x():
    """Centre x (px) of the 5 px divider between the logo plate and the text, 22 px right of the plate."""
    return sign_logo_box()[2] + 22


def sign_text_box():
    """(x0, y0, x1, y1) px the sign text stays inside (tests/t_i18n2.py): right of the divider, inside the 4 px black
    rim (5..9 px from the edges)."""
    W, H = SIGN_PX
    return (_sign_divider_x() + 3, 9, W - 9, H - 9)


def _sign_mat():
    """Header sign, one C._label atlas cell in the C._label_mat look (yellow, 4 px black rim, bold black text):
    [logo plate | divider | text centred in the rest, shrunk to 92 % of it]; text in the current language."""
    key = f"sto_sign_{i18n2.get_lang()}_{branding2.get_brand()}"
    m = bpy.data.materials.get(key)
    if m is not None:
        return m
    from PIL import Image, ImageDraw
    W, H = SIGN_PX
    fg, bg = (20, 20, 20), (242, 180, 0)
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    d.rectangle((5, 5, W - 6, H - 6), outline=fg, width=4)
    lb = sign_logo_box()
    xd = _sign_divider_x()
    d.rectangle((xd - 2, 16, xd + 2, H - 17), fill=fg)
    t = i18n2.tr("СКЛАД ГОТОВЫХ СПУЛОВ · FINISHED SPOOLS")
    xa, xb = xd + 24, W - 24
    size = int(H * 0.60)
    font = fonts2.truetype(C._FONT, size)
    while font.getlength(t) > (xb - xa) * 0.92 and size > 10:
        size -= 2
        font = fonts2.truetype(C._FONT, size)
    d.text(((xa + xb) / 2, H / 2), t, font=font, fill=fg, anchor="mm")
    branding2.paste_logo(img, lb, plate=True, pad=SIGN_LOGO_PAD)
    arr = np.asarray(img.convert("RGBA"), dtype=np.float32)[::-1] / 255.0
    bi = bpy.data.images.new(key, W, H, alpha=True)
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
    m["n_cells"] = 1
    return m


def bay_frame(tier, bay):
    return kin.planar_frame(*L2.storage_frame(tier, bay))


def _v_apex(tier):
    return SEAT_Z[tier] + L2.PIPE_AXIS_Z - (L2.PIPE_R + GAP) * math.sqrt(2.0)


def _v_bottom(tier):
    return _v_apex(tier) - V_BOT


# ============================================================================ rack
def _seat(b, M, tier, bay):
    """Seat ring + base disc + 3 pins on the tier girder, world coords of the bay frame."""
    x, y, z = TIER_X[tier], BAY_Y[tier][bay], SEAT_Z[tier]
    pre = f"sto_seat{tier}{bay}"
    zg = GIRDER_TOP[tier]
    E._rev(b, pre + "_base", [(0, zg), (RING_BASE_R, zg), (RING_BASE_R, zg + 0.010), (0, zg + 0.010)], M["frame"], (x, y, 0), segs=48)
    ri, ro = RING
    top = z - GAP
    E._rev(b, pre + "_ring", [(ri, zg + 0.010), (ro, zg + 0.010), (ro, top - 0.002), (ro - 0.002, top), (ri + 0.002, top), (ri, top - 0.002)],
           M["steel"], (x, y, 0), segs=64)
    for k in range(4):
        a = math.radians(45 + 90 * k)
        C._hex(b, f"{pre}_bolt{k}", 0.009, 0.008, M["dark"], (x + 0.2335 * math.cos(a), y + 0.2335 * math.sin(a), zg + 0.010))
    for k, ang in enumerate((90.0, 210.0, 330.0)):
        a = math.radians(ang)
        h = C.FL_PIN[1]
        E._rev(b, f"{pre}_pin{k}", [(0, 0), (C.FL_PIN[0], 0), (C.FL_PIN[0], h - 0.006), (C.FL_PIN[0] * 0.55, h), (0, h)], M["steel"],
               (x + C.FL_PIN_R * math.cos(a), y + C.FL_PIN_R * math.sin(a), top), segs=12)


def _vhead(b, M, tier, bay):
    """V-head (saddle + steel V + liners) under the pipe at local x = VPOST_LX; returns the bottom z of the saddle."""
    x = TIER_X[tier] + VPOST_LX
    y = BAY_Y[tier][bay]
    pre = f"sto_vhead{tier}{bay}"
    za = _v_apex(tier)
    zb = za - V_BOT
    C._v_block(b, pre, zb, za, V_W, 0.02, math.radians(45.0), POST + 0.01, (x, y, 0), 'X', M["frame"], M["uhmw"], span=(0.045, V_W))
    C._bx(b, pre + "_saddle", (POST + 0.03, 2 * V_W + 0.02, 0.012), (x, y, zb - 0.006), M["frame"], bev=0.002)
    for k, s in enumerate((-1, 1)):
        C._hex(b, f"{pre}_bolt{k}", 0.008, 0.006, M["dark"], (x, y + s * (V_W + 0.001), zb))
    return zb - 0.012


def _rack(b, M, lm):
    x0, x1 = TIER_X
    ye = RACK_Y_EXT
    rw, rh = RAIL
    # ---- floor rails along Y (dark) + cross rails along X at the ends and the middle
    for k, xr in enumerate((FRONT_RAIL_X, x0, x1, BACK_X)):
        C._bx(b, f"sto_rail_y{k}", (rw, 2 * ye, rh), (xr, 0.0, rh / 2), M["frame"], bev=0.004)
    for k, yr in enumerate((-RACK_Y, 0.0, RACK_Y)):
        xa, xb = BACK_X + rw / 2, FRONT_RAIL_X - rw / 2
        C._bx(b, f"sto_rail_x{k}", (xb - xa, rw, rh * 0.8), ((xa + xb) / 2, yr, rh * 0.4), M["frame"], bev=0.004)
    for k, xr in enumerate((FRONT_RAIL_X, x0, x1, BACK_X)):
        for j, yr in enumerate((-RACK_Y_EXT + 0.08, RACK_Y_EXT - 0.08)):
            C._bx(b, f"sto_foot{k}{j}", (0.16, 0.16, 0.012), (xr, yr, 0.006 + rh), M["frame"])
            for n, s in enumerate((-1, 1)):
                C._hex(b, f"sto_anchor{k}{j}{n}", 0.010, 0.010, M["dark"], (xr + s * 0.062, yr - s * 0.062, 0.012 + rh))
    # ---- tier-0 girder on short legs
    g0, h0 = GIRDER_TOP[0], GIRDER_H[0]
    C._bx(b, "sto_girder0", (GIRDER_W, 2 * ye, h0), (x0, 0.0, g0 - h0 / 2), M["blue"], bev=0.006)
    for k, yl in enumerate(LEGS_Y):
        C._bx(b, f"sto_leg0_{k}", (0.10, 0.10, g0 - h0 - rh), (x0, yl, (g0 - h0 + rh) / 2), M["blue"], bev=0.004)
    # ---- tier-1 girder on columns
    g1, h1 = GIRDER_TOP[1], GIRDER_H[1]
    C._bx(b, "sto_girder1", (GIRDER_W, 2 * ye, h1), (x1, 0.0, g1 - h1 / 2), M["blue"], bev=0.006)
    for k, yc in enumerate(COLS_Y):
        C._bx(b, f"sto_col1_{k}", (0.12, 0.12, g1 - h1 - rh), (x1, yc, (g1 - h1 + rh) / 2), M["blue"], bev=0.004)
    # ---- seats
    for t in (0, 1):
        for i in range(len(BAY_Y[t])):
            _seat(b, M, t, i)
    # ---- tier-0 V-posts on the front rail
    for i, y in enumerate(BAY_Y[0]):
        zs = _vhead(b, M, 0, i)
        C._bx(b, f"sto_vpost0{i}", (POST, POST, zs - rh), (FRONT_RAIL_X, y, (zs + rh) / 2), M["blue"], bev=0.003)
        C._bx(b, f"sto_vpost0{i}_foot", (0.14, 0.14, 0.012), (FRONT_RAIL_X, y, rh + 0.006), M["frame"])
    # ---- tier-1 V-posts on the tier-0 girder (between tier-0 spools), tied back to the tier-1 girder
    xp = x1 + VPOST_LX
    for i, y in enumerate(BAY_Y[1]):
        zs = _vhead(b, M, 1, i)
        C._bx(b, f"sto_vpost1{i}", (POST, POST, zs - g0), (xp, y, (zs + g0) / 2), M["blue"], bev=0.003)
        C._bx(b, f"sto_vpost1{i}_foot", (0.09, 0.09, 0.012), (xp, y, g0 + 0.006), M["frame"])
        xa, xb = x1 + GIRDER_W / 2, xp - POST / 2
        C._bx(b, f"sto_tie1{i}", (xb - xa, 0.05, TIE_Z[1] - TIE_Z[0]), ((xa + xb) / 2, y, sum(TIE_Z) / 2), M["frame"], bev=0.003)
        E._bar(b, f"sto_brace1{i}", (xp - POST / 2, y, g0 + 0.15), (x0 - GIRDER_W / 2 + 0.03, y, g0 + 0.012), 0.035, M["frame"])
    # ---- end frames at y = +-RACK_Y: posts, ties, braces
    for j, s in enumerate((-1, 1)):
        ye_ = s * RACK_Y
        C._bx(b, f"sto_end{j}_front", (0.08, 0.08, 0.45 - rh), (FRONT_RAIL_X, ye_, (0.45 + rh) / 2), M["blue"], bev=0.003)
        C._bx(b, f"sto_end{j}_tie_low", (FRONT_RAIL_X - BACK_X, 0.06, 0.06), ((FRONT_RAIL_X + BACK_X) / 2, ye_, 0.45 - 0.03),
              M["blue"], bev=0.003)
        C._bx(b, f"sto_end{j}_tie_mid", (x1 - BACK_X, 0.06, 0.06), ((x1 + BACK_X) / 2, ye_, g1 - h1 / 2), M["blue"], bev=0.003)
        E._bar(b, f"sto_end{j}_brace0", (FRONT_RAIL_X - 0.04, ye_, rh + 0.02), (x0 + 0.06, ye_, 0.45 - 0.06), 0.04, M["blue"])
        E._bar(b, f"sto_end{j}_brace1", (x1 - 0.07, ye_, rh + 0.02), (BACK_X + 0.06, ye_, g1 - h1 - 0.02), 0.04, M["blue"])
        E._bar(b, f"sto_end{j}_brace2", (x1 + 0.07, ye_, rh + 0.02), (x0 - 0.06, ye_, 0.45 - 0.06), 0.04, M["blue"])
        # yellow bollard at the tier-0 end of the front rail
        C._bx(b, f"sto_end{j}_guard", (0.04, 0.04, 0.30), (FRONT_RAIL_X + 0.06, ye_ + s * 0.02, rh + 0.15), M["yellow"], bev=0.004)
    # ---- back side: end columns, header, mesh transom, sign, bumper
    for j, s in enumerate((-1, 1)):
        C._bx(b, f"sto_back_col{j}", (0.10, 0.10, BACK_H - rh), (BACK_X, s * END_Y, (BACK_H + rh) / 2), M["yellow"], bev=0.004)
        C._bx(b, f"sto_back_col{j}_cap", (0.11, 0.11, 0.01), (BACK_X, s * END_Y, BACK_H + 0.005), M["black"])
    C._bx(b, "sto_back_header", (0.10, 2 * END_Y + 0.10, 0.10), (BACK_X, 0.0, BACK_H - 0.05), M["yellow"], bev=0.004)
    C._bx(b, "sto_back_transom", (0.06, 2 * END_Y - 0.10, 0.05), (BACK_X, 0.0, TRANSOM_Z), M["yellow"], bev=0.003)
    zt0, zt1 = TRANSOM_Z + 0.025, BACK_H - 0.10
    w = 2 * END_Y - 0.10
    # panel in local XY (fence_mesh reads object-space x/y, as the stage-1 fence bays), stood upright along world Y
    E._mesh(b, "sto_back_mesh", [(0, 0, 0), (w, 0, 0), (w, zt1 - zt0, 0), (0, zt1 - zt0, 0)], [(0, 1, 2, 3)],
            M["fence"], (BACK_X, -w / 2, zt0), (math.pi / 2, 0, math.pi / 2))
    sm = _sign_mat()
    C._label(b, "sto_sign_in", sm, 0, (1.30, 0.085), (BACK_X + 0.0515, 0.0, BACK_H - 0.05), (math.pi / 2, 0, math.pi / 2))
    C._label(b, "sto_sign_out", sm, 0, (1.30, 0.085), (BACK_X - 0.0515, 0.0, BACK_H - 0.05), (math.pi / 2, 0, -math.pi / 2))
    C._bx(b, "sto_back_bumper", (0.06, 2 * BUMPER_HALF_Y, 0.12), (L2.LOG_FENCE_X0 + 0.035, 0.0, 0.06), M["hazard"], bev=0.006)
    # ---- bay number plates on the girder fronts (+X faces, readable from the track), tier 0 = A, tier 1 = B
    labels = {}
    for t in (0, 1):
        xf = TIER_X[t] + GIRDER_W / 2 + 0.0035
        zl = GIRDER_TOP[t] - GIRDER_H[t] / 2 if t == 0 else GIRDER_TOP[t] - 0.035
        for i, y in enumerate(BAY_Y[t]):
            k = i if t == 0 else len(BAY_Y[0]) + i
            labels[(t, i)] = C._label(b, f"sto_label{t}{i}", lm, k, (0.10, 0.05), (xf, y, zl), (math.pi / 2, 0, math.pi / 2),
                                      backing=M["black"])
    # ---- junction box on the back column
    C._bx(b, "sto_jbox", (0.06, 0.16, 0.20), (BACK_X + 0.08, -END_Y + 0.04, 1.10), M["charcoal"], bev=0.005)
    return labels


# ============================================================================ AGV
def _agv(col, M, lm):
    b = E._B(col)
    L, W = AGV_SIZE
    zd = AGV_DECK_Z
    C._bx(b, "sto_agv_body", (L - 0.04, W - 0.03, zd - 0.08), (0, 0, 0.06 + (zd - 0.08) / 2), M["charcoal"], bev=0.02)
    C._bx(b, "sto_agv_deck", (L, W, 0.022), (0, 0, zd - 0.011), M["frame"], bev=0.006)
    C._bx(b, "sto_agv_skirt", (L - 0.10, W - 0.08, 0.05), (0, 0, 0.045), M["black"], bev=0.004)
    for k, s in enumerate((1, -1)):
        C._bx(b, f"sto_agv_bumper{k}", (0.035, W - 0.06, 0.09), (s * (L / 2 - 0.005), 0, 0.115), M["hazard"], bev=0.008)
        C._bx(b, f"sto_agv_ledband{k}", (L - 0.14, 0.006, 0.014), (0, s * (W / 2 - 0.013), 0.20), M["led_blue"])
        C._bx(b, f"sto_agv_ledfront{k}", (0.006, W - 0.20, 0.014), (s * (L / 2 - 0.018), 0, 0.20), M["led_blue"])
    # drive wheels in side pockets + casters
    for k, s in enumerate((1, -1)):
        E._rod(b, f"sto_agv_wheel{k}", 0.095, 0.05, M["rubber"], (0, s * (W / 2 - 0.02), 0.095), (math.pi / 2, 0, 0), segs=32)
        E._rod(b, f"sto_agv_hub{k}", 0.05, 0.054, M["steel"], (0, s * (W / 2 - 0.02), 0.095), (math.pi / 2, 0, 0), segs=16)
        C._bx(b, f"sto_agv_pocket{k}", (0.26, 0.004, 0.11), (0, s * (W / 2 - 0.013), 0.13), M["black"])
    for k, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        E._rod(b, f"sto_agv_caster{k}", 0.04, 0.03, M["rubber"], (sx * (L / 2 - 0.16), sy * (W / 2 - 0.12), 0.04), (math.pi / 2, 0, 0), segs=16)
    # lidars on two diagonal corners
    for k, (sx, sy) in enumerate(((1, 1), (-1, -1))):
        p = (sx * (L / 2 - 0.035), sy * (W / 2 - 0.035), 0.0)
        E._rev(b, f"sto_agv_lidar{k}", [(0, 0.14), (0.048, 0.14), (0.048, 0.23), (0, 0.23)], M["black"], p, segs=24)
        E._rev(b, f"sto_agv_lidar{k}_win", [(0.0485, 0.165), (0.0495, 0.165), (0.0495, 0.205), (0.0485, 0.205)], M["glass"], p, segs=24)
    # E-stops (red mushroom on yellow housing) at the front corners, HMI screen at the back, name plates
    for k, s in enumerate((1, -1)):
        x, y = L / 2 - 0.07, s * (W / 2 - 0.07)
        C._bx(b, f"sto_agv_estop{k}_base", (0.07, 0.07, 0.04), (x, y, zd + 0.02), M["yellow"], bev=0.005)
        E._rev(b, f"sto_agv_estop{k}", [(0, 0), (0.012, 0), (0.012, 0.012), (0.026, 0.016), (0.026, 0.026), (0, 0.03)], M["red"],
               (x, y, zd + 0.04), segs=20)
    C._bx(b, "sto_agv_hmi", (0.012, 0.20, 0.11), (-L / 2 + 0.002, 0.12, 0.21), M["frame"], bev=0.003)
    C._bx(b, "sto_agv_hmi_screen", (0.004, 0.17, 0.085), (-L / 2 - 0.005, 0.12, 0.21), M["screen"])
    for k, s in enumerate((1, -1)):
        C._label(b, f"sto_agv_name{k}", lm, len(BAY_Y[0]) + len(BAY_Y[1]), (0.20, 0.06), (0.28, s * ((W - 0.03) / 2 + 0.0015), 0.14),
                 (math.pi / 2, 0, 0 if s < 0 else math.pi))
    # beacon on a short mast at the rear-right corner
    bx, by = -(L / 2 - 0.06), -(W / 2 - 0.06)
    E._rod(b, "sto_agv_beacon_mast", 0.014, 0.10, M["dark"], (bx, by, zd + 0.05), segs=12)
    E._rev(b, "sto_agv_beacon_base", [(0, 0), (0.034, 0), (0.034, 0.025), (0, 0.025)], M["black"], (bx, by, zd + 0.10), segs=24)
    beacon = E._rev(b, "sto_agv_beacon", [(0, 0), (0.030, 0), (0.030, 0.045), (0.022, 0.065), (0, 0.072)], M["beacon"],
                    (bx, by, zd + 0.125), segs=24)
    beacon["beacon_on"] = 0.0
    root = G.empty("sto_agv_root", collection=col, size=0.4)
    for ob in b.objs:
        ob.parent = root
    # lifting table: plate with rubber strips on 4 spindles (the spindles hide in the chassis at h = 0)
    bl = E._B(col)
    C._bx(bl, "sto_agv_lift_plate", (1.04, 0.64, 0.03), (0, 0, 0.004 + 0.015), M["frame"], bev=0.004)
    for k, s in enumerate((-1, 1)):
        C._bx(bl, f"sto_agv_lift_pad{k}", (0.96, 0.06, 0.012), (0, s * 0.24, 0.034 + 0.006), M["rubber"], bev=0.002)
    for k, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        E._rod(bl, f"sto_agv_spindle{k}", 0.025, 0.13, M["steel"], (sx * 0.40, sy * 0.22, 0.004 - 0.065), segs=16)
    lift = G.empty("sto_agv_lift", collection=col, size=0.3)
    lift.parent = root
    lift.location = (0.0, 0.0, zd)
    for ob in bl.objs:
        ob.parent = lift
    return dict(root=root, lift=lift, beacon=beacon, objects=b.objs + bl.objs)


# ============================================================================ public API
def build(collection=None):
    col = collection or bpy.data.collections.new(NAME)
    if collection is None:
        bpy.context.scene.collection.children.link(col)
    M = _mats()
    lm = C._label_mat("sto_label", [f"A{i + 1}" for i in range(len(BAY_Y[0]))] + [f"B{i + 1}" for i in range(len(BAY_Y[1]))] + ["AGV 01"])
    b = E._B(col)
    labels = _rack(b, M, lm)
    root = G.empty("sto_rack_root", collection=col, size=0.4)
    root.matrix_world = G.M(kin.tr(TIER_X[1], 0.0, 0.0))
    C._parent_all(b.objs, root)
    agv = _agv(col, M, lm)
    st = dict(collection=col, rack=dict(root=root, objects=b.objs, labels=labels),
              bays={(t, i): bay_frame(t, i) for t in (0, 1) for i in range(len(BAY_Y[t]))}, agv=agv)
    set_agv(st, L2.AGV["park"][0], L2.AGV["park"][1], math.pi / 2)
    set_agv_lift(st, 0.0)
    return st


def set_agv(st, x, y, yaw, frame=None):
    """AGV pose on the floor: centre (x, y), heading yaw (rad, local +X forward).  Keys only x, y and yaw."""
    r = st["agv"]["root"]
    r.location[0] = x
    r.location[1] = y
    r.rotation_euler[2] = yaw
    if frame is not None:
        r.keyframe_insert(data_path="location", index=0, frame=frame)
        r.keyframe_insert(data_path="location", index=1, frame=frame)
        r.keyframe_insert(data_path="rotation_euler", index=2, frame=frame)


def set_agv_lift(st, h, frame=None):
    """Lifting-table stroke h (m, clamped to [0, AGV_LIFT_MAX]) above the chassis deck."""
    lf = st["agv"]["lift"]
    lf.location[2] = AGV_DECK_Z + min(max(float(h), 0.0), AGV_LIFT_MAX)
    if frame is not None:
        lf.keyframe_insert(data_path="location", index=2, frame=frame)


def set_beacon(st, on, frame=None):
    """Amber beacon on/off (object property 'beacon_on'); keyed CONSTANT when frame is given."""
    bc = st["agv"]["beacon"]
    bc["beacon_on"] = float(on)
    if frame is not None:
        C.key_constant(bc, '["beacon_on"]', frame)


def obstacles():
    """Conservative world AABBs (dict(name, center, size, yaw=0)) of the rack and of the AGV at its park position."""
    out = []

    def add(name, lo, hi):
        out.append(dict(name=name, center=tuple((lo[k] + hi[k]) / 2 for k in range(3)),
                        size=tuple(hi[k] - lo[k] for k in range(3)), yaw=0.0))

    x0, x1 = TIER_X
    ye = RACK_Y_EXT
    rw, rh = RAIL
    add("sto_floor", (BACK_X - 0.09, -ye - 0.03, 0.0), (FRONT_RAIL_X + 0.09, ye + 0.03, rh + 0.025))
    for t in (0, 1):
        # girder + seat discs / rings (up to the flange back face) ...
        add(f"sto_girder{t}", (TIER_X[t] - RING_BASE_R - 0.002, -ye - 0.01, 0.0 if t == 0 else GIRDER_TOP[1] - GIRDER_H[1]),
            (TIER_X[t] + RING_BASE_R + 0.002, ye + 0.01, SEAT_Z[t] - GAP))
        for i, y in enumerate(BAY_Y[t]):
            # ... pins (3 per seat) ...
            for k, ang in enumerate((90.0, 210.0, 330.0)):
                a = math.radians(ang)
                px, py = TIER_X[t] + C.FL_PIN_R * math.cos(a), y + C.FL_PIN_R * math.sin(a)
                r = C.FL_PIN[0] + 0.001
                add(f"sto_pin{t}{i}{k}", (px - r, py - r, SEAT_Z[t] - GAP), (px + r, py + r, SEAT_Z[t] + C.FL_PIN[1]))
            # ... V-post + V-head
            xp = TIER_X[t] + VPOST_LX
            z_top = _v_apex(t) + V_W + 0.012
            add(f"sto_vhead{t}{i}", (xp - POST / 2 - 0.02, y - V_W - 0.012, _v_bottom(t) - 0.02), (xp + POST / 2 + 0.02, y + V_W + 0.012, z_top))
            add(f"sto_vpost{t}{i}", (xp - 0.07, y - 0.07, rh if t == 0 else GIRDER_TOP[0]), (xp + 0.07, y + 0.07, _v_bottom(t) - 0.018))
    add("sto_tier1_cols", (x1 - 0.08, -ye - 0.01, 0.0), (x1 + 0.08, ye + 0.01, GIRDER_TOP[1]))
    for i, y in enumerate(BAY_Y[1]):
        add(f"sto_tie1{i}", (x1 + GIRDER_W / 2 - 0.01, y - 0.03, TIE_Z[0] - 0.005), (x1 + VPOST_LX, y + 0.03, TIE_Z[1] + 0.005))
        add(f"sto_brace1{i}", (x0 - GIRDER_W / 2, y - 0.03, GIRDER_TOP[0]), (x1 + VPOST_LX, y + 0.03, GIRDER_TOP[0] + 0.18))
    for j, s in enumerate((-1, 1)):
        # end-frame members (0.08 posts, 0.06 ties, 0.04 braces, guard) at y = +-RACK_Y, outside the outer bays' flanges
        ya, yb = sorted((s * (RACK_Y - 0.043), s * (RACK_Y + 0.045)))
        add(f"sto_end{j}", (BACK_X - 0.06, ya, 0.0), (FRONT_RAIL_X + 0.09, yb, 0.46))
        add(f"sto_end{j}_high", (BACK_X - 0.06, ya, 0.0), (x1 + 0.08, yb, GIRDER_TOP[1]))
        yc = s * END_Y
        add(f"sto_back_col{j}", (BACK_X - 0.06, yc - 0.06, 0.0), (BACK_X + 0.06, yc + 0.06, BACK_H + 0.01))
    add("sto_back_top", (BACK_X - 0.06, -END_Y - 0.06, TRANSOM_Z - 0.03), (BACK_X + 0.06, END_Y + 0.06, BACK_H + 0.01))
    add("sto_back_bumper", (L2.LOG_FENCE_X0, -BUMPER_HALF_Y - 0.005, 0.0), (L2.LOG_FENCE_X0 + 0.07, BUMPER_HALF_Y + 0.005, 0.125))
    yj = -END_Y + 0.04
    add("sto_jbox", (BACK_X + 0.045, yj - 0.09, 0.99), (BACK_X + 0.115, yj + 0.09, 1.21))
    # AGV parked (yaw = 90 deg: long axis along Y); the lift table raised to its maximum
    px, py = L2.AGV["park"]
    L, W = AGV_SIZE
    add("sto_agv_park", (px - W / 2 - 0.03, py - L / 2 - 0.03, 0.0), (px + W / 2 + 0.03, py + L / 2 + 0.03, AGV_DECK_Z + AGV_LIFT_MAX + 0.20))
    return out
