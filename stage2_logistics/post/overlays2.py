"""Stage-2 overlays: the stage-1 title / captions / corner label (demo_video/post/overlays.py, reused unchanged),
the new stage-2 end card with a horizontal flow diagram of the 7 steps, and the logos of the edition's brand.

    ovs = render_all2(sb, out_dir)    # list of overlays.Overlay (Overlay2) in compositing order

Every Overlay carries OUTPUT frame numbers (start/end inclusive, like stage 1).  ``Overlay2`` adds per-edge fades and
``whole``:
  * corner label: one overlay per stage-2 range (``sb["corner_ranges"]``) that fades in/out (12 frames) only at the
    very start/end of the video and cuts hard at the splices, where the stage-1 video segments continue with the same
    label burned in at the same place and opacity (render mode: one range over the whole video);
  * logos of ``sb["brand"]`` (branding2: the customer's logo on a white rounded plate, ``sb["logos"]`` placement and
    timing): title logo (large, top left, title timing and fades), corner logo (small, top left - the corner label
    is top right - ``whole=True``: compose2 draws it over the concatenated video, stage-1 video segments included),
    end-card logo (centred above the heading, part of the end-card image).
Every text drawn comes from the storyboard (no language defaults here): the English edition draws no Cyrillic.
"""
import copy
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

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
    the usual font folders or matplotlib's bundled DejaVu fonts.  The storyboard keeps the stage-1 Linux paths."""
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
    """Copy of the storyboard with font paths valid on this machine."""
    sb = copy.deepcopy(sb)
    sb["font"] = resolve_font(sb.get("font", ""), False)
    sb["font_bold"] = resolve_font(sb.get("font_bold", ""), True)
    return sb


# ----------------------------------------------------------------------------- end card
def wrap_bullet(text: str, font, max_w: float) -> List[str]:
    """Like overlays.wrap_text, but a two-line bullet prefers to break after a clause (':' ',' ';') when both halves
    fit - "Каждый спул прослеживается по ID: / режимы швов, ..." instead of a mid-phrase split."""
    if font.getlength(text) <= max_w:
        return [text]
    words = text.split(" ")
    best = None
    for k in range(1, len(words)):
        a, b = " ".join(words[:k]), " ".join(words[k:])
        if a[-1] not in ":,;" or max(font.getlength(a), font.getlength(b)) > max_w:
            continue
        score = abs(font.getlength(a) - font.getlength(b))
        if best is None or score < best[0]:
            best = (score, [a, b])
    return best[1] if best else overlays.wrap_text(text, font, max_w)


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


def _flow_layout(sb: dict, steps: Sequence[str], st: dict):
    """Pick box size + label font so all step labels fit in <= 2 lines.  Returns a dict of geometry."""
    mx = st["margin_x"]
    n = len(steps)
    gap = 50                                          # arrow gap between boxes
    box_w = int((W - 2 * mx - (n - 1) * gap) / n)
    pad = 10
    for size in (26, 25, 24, 23, 22, 21, 20):
        f_lab = overlays._font(sb["font"], size)
        lines = [_label_lines(s, f_lab, box_w - 2 * pad) for s in steps]
        if all(lines):
            break
    else:
        raise ValueError("flow step labels do not fit the end-card boxes; shorten them")
    x0 = int((W - (n * box_w + (n - 1) * gap)) / 2)
    return {"gap": gap, "box_w": box_w, "box_h": 132, "font": f_lab, "line_h": int(size * 1.3), "lines": lines,
            "x0": x0}


def _draw_flow(img: Image.Image, sb: dict, st: dict, geo: dict, y: int) -> None:
    """Rounded step boxes (dark caption-panel fill, thin light outline) with an orange numbered badge and a one- or
    two-line label; orange arrows between the boxes."""
    d = ImageDraw.Draw(img)
    f_num = overlays._font(sb["font_bold"], 22)
    bw, bh, gap = geo["box_w"], geo["box_h"], geo["gap"]
    badge_r = 17
    badge_cy = y + 18 + badge_r
    for i, lines in enumerate(geo["lines"]):
        bx = geo["x0"] + i * (bw + gap)
        # box drawn on its own layer so the translucent fill composites over the veil (not replaces it)
        layer = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
        ImageDraw.Draw(layer).rounded_rectangle([0, 0, bw - 1, bh - 1], radius=14, fill=st["panel"],
                                                outline=(255, 255, 255, 70), width=2)
        img.alpha_composite(layer, (bx, y))
        cx = bx + bw / 2
        d.ellipse([cx - badge_r, badge_cy - badge_r, cx + badge_r, badge_cy + badge_r], fill=st["accent"])
        d.text((cx, badge_cy + 1), str(i + 1), font=f_num, fill=st["chip_text"], anchor="mm")
        # label block centred in the area below the badge
        area_top = badge_cy + badge_r + 8
        area_h = y + bh - 10 - area_top
        lh = geo["line_h"]
        ty = area_top + (area_h - lh * len(lines)) / 2 + lh / 2
        for line in lines:
            d.text((cx, ty), line, font=geo["font"], fill=st["text"], anchor="mm")
            ty += lh
        # arrow to the next box: shaft + triangular head, vertically centred on the box
        if i < len(geo["lines"]) - 1:
            ax0 = bx + bw + 9
            ax1 = bx + bw + gap - 9
            ay = y + bh / 2
            head = 13
            d.rectangle([ax0, ay - 2, ax1 - head + 1, ay + 2], fill=st["accent"])
            d.polygon([(ax1 - head, ay - 9), (ax1, ay), (ax1 - head, ay + 9)], fill=st["accent"])


END_TOP_MIN = 36                   # the end-card content (logo plate) never starts closer to the top edge
SPACING_STEPS = (1.0, 0.9, 0.8, 0.7, 0.6)        # gap scales tried when the content does not fit above the footer


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


def end_card_layout(sb: dict, st: dict = None) -> dict:
    """Vertical layout of the end card: logo plate (``sb["logos"]["end_card"]["height"]``, if the storyboard has a
    brand), heading + accent rule, flow diagram, KPI bullets, cta chip - centred with the stage-1 upward bias (-30 px)
    and ending at least 40 px above the footer.  If the content does not fit, the gaps shrink (SPACING_STEPS), then
    the logo (down to 80 % of its height); still too tall -> ValueError (shorten the KPI lines)."""
    st = st or overlays._style(sb)
    e = sb["end_card"]
    f_line = overlays._font(sb["font"], 36)
    f_cta = overlays._font(sb["font_bold"], 24)
    cta = (e.get("cta") or "").strip()
    cta_h = (overlays._ascent(f_cta) + 2 * 12 + 4) if cta else 0
    kpis = e.get("kpis") or e.get("lines") or []
    flow = e.get("flow") or []
    geo = _flow_layout(sb, flow, st) if flow else None
    lines: List[List[str]] = [wrap_bullet(l, f_line, 1600) for l in kpis]
    n_lines = sum(len(l) for l in lines)
    L = (sb.get("logos") or {}).get("end_card") if brand_key(sb) else None
    h0 = int(L["height"]) if L else 0
    limit = H - st["margin_y"] - 40                   # content ends above the footer (baseline H - margin_y)
    for lh in ([h0 - k * 10 for k in range(0, 3) if h0 - k * 10 >= 0.8 * h0] if h0 else [0]):
        for s in SPACING_STEPS:
            g = {"logo": round(34 * s) if lh else 0, "rule": round(40 * s), "flow": round(54 * s),
                 "cta": round(34 * s)}
            total = (lh + g["logo"] + 64 + 16 + g["rule"]
                     + ((geo["box_h"] + g["flow"]) if geo else 0)
                     + 56 * n_lines + 14 * len(lines)
                     + ((g["cta"] + cta_h) if cta else 0))
            y0 = max(END_TOP_MIN, (H - total) // 2 - 30)
            if y0 + total <= limit:
                lw = logo_size(sb, lh)[0] if lh else 0
                return {"y0": y0, "total": total, "end": y0 + total, "limit": limit, "gaps": g, "spacing": s,
                        "logo_h": lh, "logo_box": ((W - lw) // 2, y0, (W - lw) // 2 + lw, y0 + lh) if lh else None,
                        "geo": geo, "lines": lines, "cta": cta, "cta_h": cta_h}
    raise ValueError(f"end card content ({n_lines} bullet lines, logo {h0} px) does not fit above the footer "
                     f"(y <= {limit}) even with the gaps at {SPACING_STEPS[-1]:.0%}; shorten the KPI lines")


def render_end_card2(sb: dict, out_dir: Path) -> Overlay2:
    """Stage-2 end card: the stage-1 veil and typography (heading 64 bold, accent rule, 36 px bullets with orange
    squares, orange cta chip, dim footer) with the brand logo plate centred above the heading and the 7-step flow
    diagram between the heading and the KPI bullets (layout: end_card_layout)."""
    st = overlays._style(sb)
    e = sb["end_card"]
    lay = end_card_layout(sb, st)
    img = Image.new("RGBA", (W, H), (0, 0, 0, st["endcard_dim"]))
    d = ImageDraw.Draw(img)
    f_head = overlays._font(sb["font_bold"], 64)
    f_line = overlays._font(sb["font"], 36)
    f_cta = overlays._font(sb["font_bold"], 24)
    f_foot = overlays._font(sb["font"], 24)
    cta, cta_h, cta_spacing = lay["cta"], lay["cta_h"], 1.5
    g, geo, lines = lay["gaps"], lay["geo"], lay["lines"]
    y = lay["y0"]

    if lay["logo_box"]:
        # the plate is opaque over the veil (its rounded corners stay transparent)
        plate = branding2.logo_plate(brand_key(sb), height=lay["logo_h"])
        img.alpha_composite(plate, lay["logo_box"][:2])
        y += lay["logo_h"] + g["logo"]

    d.text((W / 2, y + overlays._ascent(f_head)), e["heading"], font=f_head, fill=st["text"], anchor="ms")
    y += 64 + 16
    d.rectangle([W / 2 - 70, y, W / 2 + 70, y + 5], fill=st["accent"])
    y += g["rule"]

    if geo:
        _draw_flow(img, sb, st, geo, y)
        y += geo["box_h"] + g["flow"]
        d = ImageDraw.Draw(img)

    if lines:
        max_w = max(f_line.getlength(l[0]) for l in lines)
        x0 = int((W - max_w) / 2)
        for wrapped in lines:
            for k, line in enumerate(wrapped):
                if k == 0:
                    d.rectangle([x0 - 34, y + 20, x0 - 22, y + 32], fill=st["accent"])
                d.text((x0, y + overlays._ascent(f_line)), line, font=f_line, fill=st["text"], anchor="ls")
                y += 56
            y += 14

    if cta:
        # call-to-action: centred orange chip (stage-1 spacing), separated from the bullets by a gap
        y += g["cta"]
        pad_x, pad_y = 22, 12
        tw = int(overlays._text_width(f_cta, cta, cta_spacing))
        cw = tw + 2 * pad_x
        cx0 = (W - cw) // 2
        d.rounded_rectangle([cx0, y, cx0 + cw, y + cta_h], radius=10, fill=st["accent"])
        overlays._draw_spaced(d, (cx0 + pad_x, y + pad_y + overlays._ascent(f_cta)), cta, f_cta, st["chip_text"],
                              cta_spacing)
        y += cta_h

    d.text((W / 2, H - st["margin_y"]), sb["footer"], font=f_foot, fill=st["text_dim"], anchor="ms")
    if y != lay["end"] or y > lay["limit"]:
        raise ValueError(f"end card content reaches y={y} (layout {lay['end']}, limit {lay['limit']}): it overlaps "
                         f"the footer; shorten the KPI lines")

    path = Path(out_dir) / "ov_end2.png"
    img.save(path)
    return Overlay2("end_card", str(path), 0, 0, e["start"], e["end"], fade=14)


# ----------------------------------------------------------------------------- logos
def logo_overlays(sb: dict, out_dir: Path) -> List[Overlay2]:
    """Title logo (large plate, top left, title timing and fades) and corner logo (small plate, top left, whole
    video: ``whole=True``, fades only at its start/end) of ``sb["brand"]``; none without a brand."""
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
    """One corner-label overlay per stage-2 range: stage-1 look (60 % opacity, top right), 12-frame fades only at
    the start/end of the whole video, hard cuts at the splices to the stage-1 segments."""
    base = overlays.render_corner_label(sb, out_dir)
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
    corner labels, corner logo).  Title and captions come from the stage-1 renderer unchanged.  Every text is taken
    from the storyboard: a missing key raises instead of falling back to a (Russian) default of the stage-1 module."""
    missing = [k for k in REQUIRED if k not in sb]
    if missing:
        raise KeyError(f"storyboard lacks {missing} (regenerate it: python3 stage2_logistics/edl.py)")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sb = with_fonts(sb)
    logos = logo_overlays(sb, out_dir)
    ovs: List[Overlay] = [overlays.render_title(sb, out_dir)]
    ovs += [o for o in logos if not o.whole]
    for i, cap in enumerate(sb.get("captions", [])):
        ovs.append(overlays.render_caption(sb, cap, i, out_dir))
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


__all__ = ["Overlay", "Overlay2", "fades", "render_all2", "render_end_card2", "end_card_layout", "corner_overlays",
           "logo_overlays", "logo_boxes", "brand_key", "preview_frame2", "resolve_font", "with_fonts"]
