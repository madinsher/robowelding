"""Stage-2 overlays: title, captions, corner label, end card and the logos of the edition's brand.

    ovs = render_all2(sb, out_dir)    # list of overlays.Overlay (Overlay2) in compositing order

Look ("open type", STYLE2): no caption boxes.  Type is set directly on the footage in a light humanist face (Fira
Sans, shipped in assets/fonts, SIL OFL) over a soft scrim anchored in the lower-left corner; the scrim darkens the
picture towards a deep tone of the brand colour (``sb["brand"]["color"]``), so each edition takes its tint from its
logo.  Rubrics are tracked capitals after a short rule in the brand colour.  The end card is a summary frame: logo,
heading, the 7 steps as a thin timeline, three key facts.

Every Overlay carries OUTPUT frame numbers (start/end inclusive, like stage 1).  ``Overlay2`` adds per-edge fades and
``whole``:
  * corner label: one overlay per stage-2 range (``sb["corner_ranges"]``) that fades in/out (12 frames) only at the
    very start/end of the video and cuts hard at the splices to the stage-1 video segments (render mode: one range
    over the whole video);
  * logos of ``sb["brand"]`` (branding2: the customer's logo on a white rounded plate, ``sb["logos"]`` placement,
    opacity and timing): title logo (large, top left, title timing and fades), corner logo (small, top left - the
    corner label is top right - ``whole=True``: compose2 draws it over the concatenated video, stage-1 video segments
    included), end-card logo (top left, part of the end-card image).
Every text drawn comes from the storyboard (no language defaults here): the English edition draws no Cyrillic.
The stage-1 module (demo_video/post/overlays.py) supplies only the Overlay class and the font / wrap helpers.
"""
import copy
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
S2_ROOT = HERE.parent
DEMO = HERE.parents[1] / "demo_video"
for _p in (str(DEMO), str(S2_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from post import overlays                  # noqa: E402  (stage-1 module: demo_video/post/overlays.py)
import branding2                           # noqa: E402  (stage2_logistics/branding2.py; bpy only in its Blender helpers)

W, H = overlays.W, overlays.H
Overlay = overlays.Overlay


@dataclass
class Overlay2(overlays.Overlay):
    """stage-1 Overlay + optional per-edge fade lengths (None -> ``fade``) + ``whole``: drawn over the concatenated
    video (compose2), not inside one stage-2 block - it may run over the stage-1 video segments."""
    fade_in: Optional[int] = None
    fade_out: Optional[int] = None
    whole: bool = False


def fades(ov: Overlay):
    """(fade_in, fade_out) in frames for any Overlay."""
    fi = getattr(ov, "fade_in", None)
    fo = getattr(ov, "fade_out", None)
    return (ov.fade if fi is None else fi), (ov.fade if fo is None else fo)


# ----------------------------------------------------------------------------- fonts (Linux + Windows)
_FONT_DIRS = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "fonts"),   # shipped
    "/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu", "/usr/share/fonts/TTF",
    "/usr/local/share/fonts", os.path.expanduser("~/.fonts"),
    os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"),
    "/Library/Fonts", "/System/Library/Fonts/Supplemental",
] + ([os.path.join(os.environ["LOCALAPPDATA"], "Microsoft", "Windows", "Fonts")] if os.environ.get("LOCALAPPDATA")
     else [])
_FALLBACK = {False: ["DejaVuSans.ttf", "arial.ttf", "Arial.ttf", "segoeui.ttf"],
             True: ["DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf", "segoeuib.ttf"]}


def resolve_font(path: str, bold: bool) -> str:
    """``path`` if it exists, else the same file name / a Cyrillic-capable fallback (DejaVu, Arial, Segoe UI) from
    the usual font folders or matplotlib's bundled DejaVu fonts.  The storyboard keeps bare file names of the fonts
    shipped in assets/fonts."""
    names = ([os.path.basename(path)] if path else []) + _FALLBACK[bold]
    shipped = os.path.join(_FONT_DIRS[0], names[0])
    if os.path.isfile(shipped):          # the repository's copy first: identical glyphs on every OS
        return shipped
    if path and os.path.isfile(path):
        return path
    dirs = list(_FONT_DIRS)
    try:
        import matplotlib
        dirs.append(os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf"))
    except Exception:
        pass
    for name in names:
        for d in dirs:
            p = os.path.join(d, name)
            if d and os.path.isfile(p):
                return p
    raise FileNotFoundError(f"font {path!r} not found and no Cyrillic fallback (install DejaVu Sans)")


def with_fonts(sb: dict) -> dict:
    """Copy of the storyboard with font paths valid on this machine (font = regular, font_bold = the medium weight
    of the rubrics, font_light = the light weight of headings and captions; a storyboard without font_light: font)."""
    sb = copy.deepcopy(sb)
    sb["font"] = resolve_font(sb.get("font", ""), False)
    sb["font_bold"] = resolve_font(sb.get("font_bold", ""), True)
    sb["font_light"] = resolve_font(sb.get("font_light") or sb["font"], False)
    return sb


# ----------------------------------------------------------------------------- style
STYLE2 = {
    "text": (255, 255, 255, 255),
    "text_dim": (255, 255, 255, 200),       # sub-lines, fact labels
    "text_faint": (255, 255, 255, 130),     # footer
    "rubric": (255, 255, 255, 225),         # tracked capitals of the caption rubric
    "margin_x": 96,                         # left safe margin of the type
    "margin_y": 92,                         # baseline of the last line above the bottom edge
    "accent_lift": 0.12,                    # accent = brand colour mixed 12 % towards white (reads on dark footage)
    "scrim_tint": 0.80,                     # scrim colour = brand colour mixed 80 % towards black
    "scrim_strength": 0.90,                 # scrim alpha in the corner
    "halo": 120,                            # alpha of the soft dark halo under type set on the footage
    "veil": 0.78,                           # end card: alpha of the full-frame veil ...
    "veil_tint": 0.90,                      # ... brand colour mixed 90 % towards black
    "label_opacity": 0.67,                  # corner label
}
FALLBACK_BRAND = (255, 106, 19)             # a storyboard without a brand: the stage-1 orange


def _mix(c, other, t):
    return tuple(int(round(c[i] * (1 - t) + other[i] * t)) for i in range(3))


def style2(sb: dict) -> dict:
    """STYLE2 + ``sb["style2"]`` overrides + the colours derived from the brand: accent, scrim, veil_color."""
    st = dict(STYLE2)
    for k, v in (sb.get("style2") or {}).items():
        st[k] = tuple(v) if isinstance(v, list) else v
    b = sb.get("brand") or {}
    color = tuple(b.get("color") or FALLBACK_BRAND) if isinstance(b, dict) else FALLBACK_BRAND
    st["brand"] = color
    st["accent"] = _mix(color, (255, 255, 255), st["accent_lift"]) + (255,)
    st["scrim"] = _mix(color, (0, 0, 0), st["scrim_tint"])
    st["veil_color"] = _mix(color, (0, 0, 0), st["veil_tint"])
    return st


def _f(sb: dict, weight: str, size: int):
    """Font of the storyboard: 'light' (headings, captions), 'regular', 'medium' (rubrics, numbers)."""
    key = {"light": "font_light", "regular": "font", "medium": "font_bold"}[weight]
    return overlays._font(sb.get(key) or sb["font"], size)


def tracked_width(text: str, font, spacing: float) -> float:
    return overlays._text_width(font, text, spacing)


def draw_tracked(d: ImageDraw.ImageDraw, xy, text: str, font, fill, spacing: float) -> float:
    """Text with letter spacing, baseline-left anchored; returns the x after the last glyph."""
    overlays._draw_spaced(d, xy, text, font, fill, spacing)
    return xy[0] + tracked_width(text, font, spacing)


def _greedy(text: str, font, max_w: float) -> List[str]:
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if font.getlength(t) <= max_w or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def wrap_even(text: str, font, max_w: float) -> List[str]:
    """Greedy wrap, then the narrowest width that keeps the same number of lines: lines of even length instead of
    a long line with a short tail."""
    n = len(_greedy(text, font, max_w))
    if n <= 1:
        return _greedy(text, font, max_w)
    lo, hi = max(font.getlength(w) for w in text.split()), max_w
    for _ in range(20):
        mid = (lo + hi) / 2
        if len(_greedy(text, font, mid)) <= n:
            hi = mid
        else:
            lo = mid
    return _greedy(text, font, hi)


def _label_lines(text: str, font, max_w: float) -> Optional[List[str]]:
    """One line if it fits, else the best two-line split at a space (never next to a ' · ' separator, which is kept
    inside its line).  None if it cannot fit in two lines."""
    if font.getlength(text) <= max_w:
        return [text]
    words = text.split(" ")
    best = None
    for k in range(1, len(words)):
        a, b = " ".join(words[:k]), " ".join(words[k:])
        if words[k - 1] == "·" or words[k] == "·":
            continue
        wmax = max(font.getlength(a), font.getlength(b))
        if wmax <= max_w and (best is None or wmax < best[0]):
            best = (wmax, [a, b])
    return best[1] if best else None


def corner_scrim(size: Tuple[int, int], color, strength: float, power: float = 0.7) -> Image.Image:
    """RGBA image of ``size``: ``color`` with an alpha that is ``strength`` in the lower-left corner and falls to 0
    on the ellipse through the right and the top edge (the darkening sits under the type and leaves the action free)."""
    w, h = size
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt((xs / w) ** 2 + ((h - 1 - ys) / h) ** 2)
    a = strength * np.clip(1.0 - r, 0.0, 1.0) ** power
    arr = np.empty((h, w, 4), np.uint8)
    arr[..., :3] = np.array(color, np.uint8)
    arr[..., 3] = (a * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def with_halo(base: Image.Image, type_layer: Image.Image, alpha: int, radius: float = 8.0) -> Image.Image:
    """``type_layer`` (text on transparent) composited over ``base`` with a soft dark halo under the glyphs: the type
    keeps its contrast where the scrim is already thin."""
    if alpha > 0:
        a = type_layer.getchannel("A").filter(ImageFilter.GaussianBlur(radius)).point(lambda v: min(255, v * 2))
        halo = Image.new("RGBA", type_layer.size, (0, 0, 0, 0))
        halo.putalpha(a.point(lambda v: int(v * alpha / 255)))
        base = Image.alpha_composite(base, halo)
    return Image.alpha_composite(base, type_layer)


# ----------------------------------------------------------------------------- title, captions, corner label
def render_title2(sb: dict, out_dir: Path) -> Overlay2:
    """Opening title: brand-tinted scrim from the lower-left corner, light heading, tracked sub-line, accent rule."""
    st = style2(sb)
    t = sb["title"]
    img = corner_scrim((W, H), st["scrim"], st["scrim_strength"] + 0.02, power=0.8)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    f_head, f_sub = _f(sb, "light", 88), _f(sb, "regular", 28)
    mx = st["margin_x"]
    y = H - st["margin_y"]
    sub = (t.get("sub") or "").strip()
    if sub:
        draw_tracked(d, (mx + 2, y), sub, f_sub, st["text_dim"], 1.2)
        y -= 64
    for line in reversed(wrap_even(t["heading"], f_head, W - 2 * mx - 160)):
        d.text((mx - 4, y), line, font=f_head, fill=st["text"], anchor="ls")
        y -= 102
    d.rectangle([mx, y - 6, mx + 88, y - 1], fill=st["accent"])
    path = Path(out_dir) / "ov_title.png"
    with_halo(img, layer, st["halo"]).save(path)
    return Overlay2("title", str(path), 0, 0, t["start"], t["end"], fade=12)


CAPTION_BOX = (1900, 620)          # scrim area of a caption, anchored in the lower-left corner of the frame
CAPTION_WRAP = 1120                # max text width (px): the right half of the frame stays free for the action


def render_caption2(sb: dict, cap: dict, idx: int, out_dir: Path) -> Overlay2:
    """Caption: rubric (accent rule + tracked capitals) and one or two lines of light type on the corner scrim."""
    st = style2(sb)
    cw, ch = CAPTION_BOX
    img = corner_scrim((cw, ch), st["scrim"], st["scrim_strength"])
    layer = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    f_tag, f_txt = _f(sb, "medium", 23), _f(sb, "light", 48)
    mx, lh = st["margin_x"], 60
    lines = wrap_even(cap["text"], f_txt, CAPTION_WRAP)
    y0 = ch - st["margin_y"] - lh * (len(lines) - 1)
    for i, line in enumerate(lines):
        d.text((mx, y0 + i * lh), line, font=f_txt, fill=st["text"], anchor="ls")
    tag = cap.get("tag", "").upper()
    if tag:
        ty = y0 - 72
        d.rectangle([mx, ty - 11, mx + 56, ty - 7], fill=st["accent"])
        draw_tracked(d, (mx + 76, ty), tag, f_tag, st["rubric"], 4.5)
    path = Path(out_dir) / f"ov_cap{idx:02d}.png"
    with_halo(img, layer, st["halo"]).save(path)
    return Overlay2(f"caption{idx:02d}", str(path), 0, H - ch, cap["start"], cap["end"], fade=12)


def render_corner_label2(sb: dict, out_dir: Path) -> Overlay2:
    """Top-right label ("DEMO · SIMULATION · SPED UP"): small tracked capitals, no box."""
    st = style2(sb)
    text = sb["corner_label"]
    f = _f(sb, "medium", 17)
    spacing, pad = 3.2, 14
    tw = int(tracked_width(text, f, spacing))
    layer = Image.new("RGBA", (tw + 2 * pad, 22 + 2 * pad), (0, 0, 0, 0))
    draw_tracked(ImageDraw.Draw(layer), (pad, pad + 17), text, f, st["text"], spacing)
    img = with_halo(Image.new("RGBA", layer.size, (0, 0, 0, 0)), layer, st["halo"], radius=5)
    path = Path(out_dir) / "ov_corner.png"
    img.save(path)
    return Overlay2("corner", str(path), W - 80 - tw - pad, 61 - pad, 1, sb["frames"], fade=12,
                    opacity=st["label_opacity"])


# ----------------------------------------------------------------------------- end card
END_HEAD_Y = 352                   # heading baseline
END_FLOW_Y = 492                   # timeline
END_FACTS_Y = 806                  # baseline of the fact values
END_FOOT_Y = 1036                  # footer baseline


def brand_key(sb: dict) -> Optional[str]:
    """Brand of the storyboard (``sb["brand"]["key"]``; a storyboard without brand: none -> no logos)."""
    b = sb.get("brand") or {}
    key = b.get("key") if isinstance(b, dict) else b
    if key and key not in branding2.BRANDS:
        raise ValueError(f"storyboard brand {key!r} not in branding2.BRANDS {list(branding2.BRANDS)}")
    return key


def logo_size(sb: dict, height: float) -> Tuple[int, int]:
    """(w, h) in px of the brand's logo plate of that height (branding2.logo_plate size, without rendering it)."""
    return int(round(height * branding2.logo_aspect(brand_key(sb), plate=True))), int(round(height))


def _logo_plate(sb: dict, L: dict) -> Image.Image:
    """The brand's logo plate of ``L["height"]`` at ``L["opacity"]`` (default 1)."""
    plate = branding2.logo_plate(brand_key(sb), height=L["height"])
    op = float(L.get("opacity", 1.0))
    if op < 1.0:
        plate.putalpha(plate.getchannel("A").point(lambda v: int(v * op)))
    return plate


def end_card_layout(sb: dict, st: dict = None) -> dict:
    """Geometry of the end card (checked here, drawn by render_end_card2): logo box, the timeline nodes with their
    wrapped labels, the fact columns.  Raises ValueError when a text does not fit its column (shorten it)."""
    st = st or style2(sb)
    e = sb["end_card"]
    mx = st["margin_x"]
    L = (sb.get("logos") or {}).get("end_card") if brand_key(sb) else None
    logo_box = None
    if L:
        lw, lh = logo_size(sb, L["height"])
        x0, y0 = int(L.get("x", mx - 20)), int(L.get("y", 60))
        logo_box = (x0, y0, x0 + lw, y0 + lh)
        if y0 + lh > END_HEAD_Y - 110:
            raise ValueError(f"end-card logo ({lh} px at y {y0}) runs into the heading; lower logos.end_card.height")
    f_head = _f(sb, "light", 78)
    if f_head.getlength(e["heading"]) > W - 2 * mx:
        raise ValueError(f"end-card heading {e['heading']!r} is wider than the frame; shorten it")
    flow = list(e.get("flow") or [])
    f_lab = _f(sb, "regular", 28)
    nodes = []
    if flow:
        step = (W - 2 * mx - 190) / max(1, len(flow) - 1)
        for i, s in enumerate(flow):
            lines = _label_lines(s, f_lab, step - 26)
            if not lines:
                raise ValueError(f"end-card flow label {s!r} does not fit {step - 26:.0f} px in two lines; shorten it")
            nodes.append((mx + 10 + i * step, lines))
    facts = [tuple(f) for f in (e.get("facts") or [])]
    f_val, f_fl = _f(sb, "light", 66), _f(sb, "regular", 27)
    cols = []
    if facts:
        cw = (W - 2 * mx) / len(facts)
        for i, (value, label) in enumerate(facts):
            if f_val.getlength(value) > cw - 40:
                raise ValueError(f"end-card fact {value!r} is wider than its column; shorten it")
            lines = wrap_even(label, f_fl, cw - 60)
            if len(lines) > 2:
                raise ValueError(f"end-card fact label {label!r} needs {len(lines)} lines; shorten it")
            cols.append((mx + i * cw, value, lines))
    end = END_FACTS_Y + 46 + 36 * max([len(c[2]) for c in cols] or [0])
    limit = END_FOOT_Y - 60
    if end > limit:
        raise ValueError(f"end-card facts end at y={end}, the footer needs y <= {limit}")
    return {"logo_box": logo_box, "logo_h": (logo_box[3] - logo_box[1]) if logo_box else 0, "nodes": nodes,
            "cols": cols, "y0": logo_box[1] if logo_box else END_HEAD_Y - 80, "end": end, "limit": limit,
            "spacing": 1.0}


def render_end_card2(sb: dict, out_dir: Path) -> Overlay2:
    """Summary frame: a brand-tinted veil over the last shot, the logo, a light heading under an accent rule, the
    steps as a thin timeline with numbered nodes, the key facts as large light values with a short label each."""
    st = style2(sb)
    e = sb["end_card"]
    lay = end_card_layout(sb, st)
    mx = st["margin_x"]
    img = Image.new("RGBA", (W, H), st["veil_color"] + (int(255 * st["veil"]),))
    if lay["logo_box"]:
        img.alpha_composite(_logo_plate(sb, sb["logos"]["end_card"]), lay["logo_box"][:2])
    d = ImageDraw.Draw(img)
    d.rectangle([mx, END_HEAD_Y - 92, mx + 88, END_HEAD_Y - 87], fill=st["accent"])
    d.text((mx - 4, END_HEAD_Y), e["heading"], font=_f(sb, "light", 78), fill=st["text"], anchor="ls")

    if lay["nodes"]:
        f_num, f_lab = _f(sb, "medium", 21), _f(sb, "regular", 28)
        yl = END_FLOW_Y
        d.line([lay["nodes"][0][0], yl, W - mx, yl], fill=(255, 255, 255, 80), width=2)
        for i, (x, lines) in enumerate(lay["nodes"]):
            d.ellipse([x - 10, yl - 10, x + 10, yl + 10], fill=st["accent"])
            draw_tracked(d, (x - 10, yl - 28), f"{i + 1:02d}", f_num, st["text_faint"], 2.0)
            for j, line in enumerate(lines):
                d.text((x - 10, yl + 54 + j * 36), line, font=f_lab, fill=st["text"], anchor="ls")

    f_val, f_fl = _f(sb, "light", 66), _f(sb, "regular", 27)
    for x, value, lines in lay["cols"]:
        d.rectangle([x, END_FACTS_Y - 86, x + 40, END_FACTS_Y - 82], fill=st["accent"])
        d.text((x - 3, END_FACTS_Y), value, font=f_val, fill=st["text"], anchor="ls")
        for j, line in enumerate(lines):
            d.text((x, END_FACTS_Y + 46 + j * 36), line, font=f_fl, fill=st["text_dim"], anchor="ls")

    d.text((mx, END_FOOT_Y), sb["footer"], font=_f(sb, "regular", 21), fill=st["text_faint"], anchor="ls")
    path = Path(out_dir) / "ov_end2.png"
    img.save(path)
    return Overlay2("end_card", str(path), 0, 0, e["start"], e["end"], fade=14)


# ----------------------------------------------------------------------------- logos
def logo_overlays(sb: dict, out_dir: Path) -> List[Overlay2]:
    """Title logo (large plate, top left, title timing and fades) and corner logo (small plate, top left, whole
    video: ``whole=True``, fades only at its start/end) of ``sb["brand"]``; none without a brand.  The opacity of a
    logo (``sb["logos"][name]["opacity"]``) is the overlay's opacity: compose2 applies it to the whole plate."""
    brand = brand_key(sb)
    lg = sb.get("logos") or {}
    out: List[Overlay2] = []
    if not brand:
        return out
    for name, whole in (("title", False), ("corner", True)):
        L = lg.get(name)
        if not L:
            continue
        plate = branding2.logo_plate(brand, height=L["height"])
        path = Path(out_dir) / f"ov_logo_{name}.png"
        plate.save(path)
        fade = int(L.get("fade", 12))
        out.append(Overlay2(f"logo_{name}", str(path), int(L["x"]), int(L["y"]), int(L["start"]), int(L["end"]),
                            fade=fade, opacity=float(L.get("opacity", 1.0)), fade_in=fade, fade_out=fade,
                            whole=whole))
    return out


def logo_boxes(sb: dict) -> dict:
    """{"title" | "corner" | "end_card": (x0, y0, x1, y1)} pixel boxes of the logo plates of the storyboard (checks:
    compose2.verify_splices masks the corner logo, the tests look for the brand colour there)."""
    if not brand_key(sb):
        return {}
    lg = sb.get("logos") or {}
    out = {}
    for name in ("title", "corner"):
        L = lg.get(name)
        if L:
            w, h = logo_size(sb, L["height"])
            out[name] = (int(L["x"]), int(L["y"]), int(L["x"]) + w, int(L["y"]) + h)
    if lg.get("end_card"):
        box = end_card_layout(with_fonts(sb))["logo_box"]
        if box:
            out["end_card"] = box
    return out


# ----------------------------------------------------------------------------- all overlays
def corner_overlays(sb: dict, out_dir: Path) -> List[Overlay2]:
    """One corner-label overlay per stage-2 range: 12-frame fades only at the start/end of the whole video, hard
    cuts at the splices to the stage-1 segments."""
    base = render_corner_label2(sb, out_dir)
    out = []
    ranges = sb.get("corner_ranges") or [[1, sb["frames"]]]
    for k, (a, b) in enumerate(ranges):
        fi = base.fade if a == 1 else 0
        fo = base.fade if b == sb["frames"] else 0
        out.append(Overlay2(f"corner{k}", base.path, base.x, base.y, a, b, fade=base.fade, opacity=base.opacity,
                            fade_in=fi, fade_out=fo))
    return out


REQUIRED = ("title", "end_card", "footer", "corner_label", "captions")


def render_all2(sb: dict, out_dir) -> List[Overlay]:
    """Render every stage-2 overlay into ``out_dir``; order = compositing order (title, title logo, captions, end card,
    corner labels, corner logo).  Every text is taken from the storyboard: a missing key raises instead of falling
    back to a (Russian) default."""
    missing = [k for k in REQUIRED if k not in sb]
    if missing:
        raise KeyError(f"storyboard lacks {missing} (regenerate it: python3 stage2_logistics/edl.py)")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sb = with_fonts(sb)
    logos = logo_overlays(sb, out_dir)
    ovs: List[Overlay] = [render_title2(sb, out_dir)]
    ovs += [o for o in logos if not o.whole]
    for i, cap in enumerate(sb.get("captions", [])):
        ovs.append(render_caption2(sb, cap, i, out_dir))
    ovs.append(render_end_card2(sb, out_dir))
    ovs.extend(corner_overlays(sb, out_dir))
    ovs += [o for o in logos if o.whole]
    return ovs


def preview_frame2(background: Image.Image, ovs: Sequence[Overlay], frame: int) -> Image.Image:
    """PIL-only composite of the overlays visible at output ``frame`` (per-edge fades honoured)."""
    bg = background.convert("RGBA").resize((W, H))
    for ov in ovs:
        if not (ov.start <= frame <= ov.end):
            continue
        fi, fo = fades(ov)
        a = 1.0
        if fi > 0:
            a = min(a, (frame - ov.start + 1) / fi)
        if fo > 0:
            a = min(a, (ov.end - frame + 1) / fo)
        a *= ov.opacity
        im = Image.open(ov.path).convert("RGBA")
        if a < 1.0:
            im.putalpha(im.getchannel("A").point(lambda v, a=a: int(v * a)))
        bg.alpha_composite(im, (ov.x, ov.y))
    return bg.convert("RGB")


__all__ = ["Overlay", "Overlay2", "fades", "render_all2", "render_title2", "render_caption2", "render_corner_label2",
           "render_end_card2", "end_card_layout", "corner_overlays", "logo_overlays", "logo_boxes", "brand_key",
           "preview_frame2", "resolve_font", "with_fonts", "style2", "wrap_even"]
