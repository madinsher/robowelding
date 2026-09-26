"""Render the title card, lower-third captions, end card and corner label as RGBA PNGs (PIL).

Every public function returns an ``Overlay`` describing the PNG and where it goes on the
1920x1080 frame; ``compose.py`` turns the list into ffmpeg ``overlay`` filters with time-based
``enable`` expressions and alpha fades.  All sizes are in output pixels (1920x1080 layout).

The storyboard (post/storyboard.json) supplies the texts and fonts; the optional ``style`` key
can override the colours below.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080

# Default palette — may be overridden per key by storyboard["style"].
STYLE = {
    "accent": (255, 106, 19, 255),          # #FF6A13 orange chip / accents
    "chip_text": (18, 16, 14, 255),         # near-black text on the orange chip
    "text": (255, 255, 255, 255),
    "text_dim": (255, 255, 255, 178),       # ~70 % white (subtitles, footer)
    "panel": (12, 14, 18, 190),             # dark translucent caption panel (~75 %)
    "gradient": (8, 10, 14),                # title bar gradient colour (alpha ramps 0 -> gradient_alpha)
    "gradient_alpha": 215,
    "endcard_dim": 140,                     # alpha of the black veil on the end card (~55 % -> frame at ~45 %)
    "margin_x": 80,                         # left/right safe margin
    "margin_y": 72,                         # bottom safe margin
}


@dataclass
class Overlay:
    """One overlay image plus its placement and timing (storyboard frame numbers, inclusive)."""
    name: str
    path: str            # RGBA PNG on disk
    x: int               # top-left corner on the 1920x1080 frame
    y: int
    start: int           # first storyboard frame (1-based) the overlay is visible
    end: int             # last visible frame
    fade: int = 12       # fade in/out length in frames (0 = hard cut)
    opacity: float = 1.0 # constant multiplier on the PNG alpha


# ----------------------------------------------------------------------------- helpers
def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def _text_width(font: ImageFont.FreeTypeFont, text: str, spacing: float = 0.0) -> float:
    if spacing == 0.0:
        return font.getlength(text)
    return sum(font.getlength(ch) for ch in text) + spacing * max(0, len(text) - 1)


def _draw_spaced(draw: ImageDraw.ImageDraw, xy: Tuple[float, float], text: str,
                 font: ImageFont.FreeTypeFont, fill, spacing: float, anchor: str = "ls") -> None:
    """Draw text with extra letter spacing (PIL has no tracking option)."""
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill, anchor=anchor)
        x += font.getlength(ch) + spacing


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: float) -> List[str]:
    """Greedy word wrap on spaces; never breaks inside a word."""
    words = text.split()
    lines: List[str] = []
    cur = ""
    for w in words:
        trial = (cur + " " + w) if cur else w
        if font.getlength(trial) <= max_width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _style(storyboard: dict) -> dict:
    st = dict(STYLE)
    for k, v in (storyboard.get("style") or {}).items():
        st[k] = tuple(v) if isinstance(v, list) else v
    return st


def _ascent(font: ImageFont.FreeTypeFont) -> int:
    return font.getmetrics()[0]


# ----------------------------------------------------------------------------- overlays
def render_title(sb: dict, out_dir: Path) -> Overlay:
    """Opening title: dark gradient bar over the bottom third with heading + sub-line."""
    st = _style(sb)
    t = sb["title"]
    bar_h = 430
    img = Image.new("RGBA", (W, bar_h), (0, 0, 0, 0))
    # Vertical gradient: transparent at the top, ~85 % at the bottom (smoothstep for a soft edge).
    grad = Image.new("L", (1, bar_h))
    px = []
    for i in range(bar_h):
        u = i / (bar_h - 1)
        s = u * u * (3 - 2 * u)
        px.append(int(st["gradient_alpha"] * s))
    grad.putdata(px)
    layer = Image.new("RGBA", (W, bar_h), st["gradient"] + (255,))
    layer.putalpha(grad.resize((W, bar_h)))
    img.alpha_composite(layer)

    d = ImageDraw.Draw(img)
    f_head = _font(sb["font_bold"], 60)
    f_sub = _font(sb["font"], 30)
    mx, my = st["margin_x"], st["margin_y"]
    max_w = W - 2 * mx
    head_lines = wrap_text(t["heading"], f_head, max_w)
    sub_lines = wrap_text(t["sub"], f_sub, max_w)

    # Lay out from the bottom up: sub lines, gap, heading lines, accent rule.
    y = bar_h - my
    for line in reversed(sub_lines):
        d.text((mx, y), line, font=f_sub, fill=st["text_dim"], anchor="ls")
        y -= 40
    y -= 18
    for line in reversed(head_lines):
        d.text((mx, y), line, font=f_head, fill=st["text"], anchor="ls")
        y -= 72
    y -= 8
    d.rectangle([mx, y, mx + 140, y + 5], fill=st["accent"])

    path = out_dir / "ov_title.png"
    img.save(path)
    return Overlay("title", str(path), 0, H - bar_h, t["start"], t["end"], fade=12)


def render_caption(sb: dict, cap: dict, idx: int, out_dir: Path) -> Overlay:
    """Lower-third: orange rounded TAG chip + white caption text on a dark translucent panel."""
    st = _style(sb)
    f_tag = _font(sb["font_bold"], 22)
    f_txt = _font(sb["font"], 34)
    pad_x, pad_y = 28, 24
    chip_pad_x, chip_pad_y = 14, 7
    tag = cap.get("tag", "").upper()
    spacing = 1.5
    max_text_w = 1160
    lines = wrap_text(cap["text"], f_txt, max_text_w)
    line_h = 44

    chip_w = int(_text_width(f_tag, tag, spacing)) + 2 * chip_pad_x if tag else 0
    chip_h = _ascent(f_tag) + 2 * chip_pad_y + 4 if tag else 0
    text_w = max(int(f_txt.getlength(l)) for l in lines) if lines else 0
    panel_w = max(chip_w, text_w) + 2 * pad_x
    panel_h = pad_y + (chip_h + 16 if tag else 0) + line_h * len(lines) + pad_y - 6

    img = Image.new("RGBA", (panel_w, panel_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, panel_w - 1, panel_h - 1], radius=14, fill=st["panel"])
    # orange accent stripe on the left edge, clipped to the rounded panel outline
    d.rectangle([0, 0, 5, panel_h], fill=st["accent"])
    mask = Image.new("L", (panel_w, panel_h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, panel_w - 1, panel_h - 1], radius=14, fill=255)
    img.putalpha(Image.composite(img.getchannel("A"), Image.new("L", img.size, 0), mask))

    y = pad_y
    if tag:
        d.rounded_rectangle([pad_x, y, pad_x + chip_w, y + chip_h], radius=8, fill=st["accent"])
        _draw_spaced(d, (pad_x + chip_pad_x, y + chip_pad_y + _ascent(f_tag)), tag, f_tag,
                     st["chip_text"], spacing)
        y += chip_h + 16
    for line in lines:
        d.text((pad_x, y + _ascent(f_txt)), line, font=f_txt, fill=st["text"], anchor="ls")
        y += line_h

    path = out_dir / f"ov_cap{idx:02d}.png"
    img.save(path)
    x = st["margin_x"]
    yy = H - st["margin_y"] - panel_h
    return Overlay(f"caption{idx:02d}", str(path), x, yy, cap["start"], cap["end"], fade=12)


def render_end_card(sb: dict, out_dir: Path) -> Overlay:
    """End card: full-frame dark veil, centred heading + bullet lines, small footer."""
    st = _style(sb)
    e = sb["end_card"]
    img = Image.new("RGBA", (W, H), (0, 0, 0, st["endcard_dim"]))
    d = ImageDraw.Draw(img)
    f_head = _font(sb["font_bold"], 64)
    f_line = _font(sb["font"], 36)
    f_foot = _font(sb["font"], 24)

    lines: List[List[str]] = [wrap_text(l, f_line, 1500) for l in e["lines"]]
    n_lines = sum(len(l) for l in lines)
    block_h = 64 + 40 + 56 * n_lines + 14 * (len(lines) - 1)
    y = (H - block_h) // 2 - 30

    d.text((W / 2, y + _ascent(f_head)), e["heading"], font=f_head, fill=st["text"], anchor="ms")
    y += 64 + 16
    d.rectangle([W / 2 - 70, y, W / 2 + 70, y + 5], fill=st["accent"])
    y += 40

    # bullets: centred block, each line prefixed by a small orange square
    max_w = max(f_line.getlength(l[0]) for l in lines)
    x0 = int((W - max_w) / 2)
    for wrapped in lines:
        for k, line in enumerate(wrapped):
            if k == 0:
                d.rectangle([x0 - 34, y + 20, x0 - 22, y + 32], fill=st["accent"])
            d.text((x0, y + _ascent(f_line)), line, font=f_line, fill=st["text"], anchor="ls")
            y += 56
        y += 14

    footer = sb.get("footer", "Демонстрационная 3D-симуляция · моделирование на основе фото участка")
    d.text((W / 2, H - st["margin_y"]), footer, font=f_foot, fill=st["text_dim"], anchor="ms")

    path = out_dir / "ov_end.png"
    img.save(path)
    return Overlay("end_card", str(path), 0, 0, e["start"], e["end"], fade=14)


def render_corner_label(sb: dict, out_dir: Path, opacity: float = 0.6) -> Overlay:
    """Discreet top-right label ("ДЕМО · СИМУЛЯЦИЯ"), shown throughout at 60 % opacity."""
    st = _style(sb)
    text = sb.get("corner_label", "ДЕМО · СИМУЛЯЦИЯ")
    f = _font(sb["font_bold"], 22)
    spacing = 2.0
    pad_x, pad_y = 16, 9
    tw = int(_text_width(f, text, spacing))
    w, h = tw + 2 * pad_x, _ascent(f) + 2 * pad_y + 4
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=8, fill=(10, 12, 16, 150), outline=(255, 255, 255, 90), width=1)
    _draw_spaced(d, (pad_x, pad_y + _ascent(f)), text, f, st["text"], spacing)
    path = out_dir / "ov_corner.png"
    img.save(path)
    return Overlay("corner", str(path), W - st["margin_x"] - w, 40, 1, sb["frames"], fade=12, opacity=opacity)


def render_all(sb: dict, out_dir: Path) -> List[Overlay]:
    """Render every overlay of the storyboard into ``out_dir``; order = compositing order."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ovs: List[Overlay] = [render_title(sb, out_dir)]
    for i, cap in enumerate(sb.get("captions", [])):
        ovs.append(render_caption(sb, cap, i, out_dir))
    ovs.append(render_end_card(sb, out_dir))
    ovs.append(render_corner_label(sb, out_dir))
    return ovs


def preview_frame(background: Image.Image, overlays: Sequence[Overlay], frame: int) -> Image.Image:
    """Composite the overlays visible at ``frame`` on a background (for quick PIL-only checks;
    linear alpha fades reproduce what ffmpeg does)."""
    bg = background.convert("RGBA").resize((W, H))
    for ov in overlays:
        if not (ov.start <= frame <= ov.end):
            continue
        a = 1.0
        if ov.fade > 0:
            a = min(1.0, (frame - ov.start + 1) / ov.fade, (ov.end - frame + 1) / ov.fade)
        a *= ov.opacity
        im = Image.open(ov.path).convert("RGBA")
        if a < 1.0:
            im.putalpha(im.getchannel("A").point(lambda v, a=a: int(v * a)))
        bg.alpha_composite(im, (ov.x, ov.y))
    return bg.convert("RGB")
