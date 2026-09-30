"""Stage-2 post-production test for both editions (editions.py): storyboard checks -> overlay texts (no Cyrillic drawn
in the English edition) -> soundtrack (the same for both editions) -> synthetic stage-2 frames -> compose2.py
(--preview) -> ffprobe / frame-accurate splice checks (ru) / synthetic-content checks / logo colour checks -> stills.

    python3 stage2_logistics/tests/t_post2.py [--edition ru|en|all] [--threads 2] [--skip-compose]

Outputs (stage2_logistics/out/, git-ignored), per edition <ed>:
    post_test_frames_<ed>/       synthetic frames 640x360 with the scene frame number, the shot name and a 14-bit
                                 barcode of the frame number: ru every 6th scene frame of every stage-2 shot (the holds
                                 of a sparse preview), en every frame of the edit (2152, the re-rendered stage-1 shots
                                 included: no holds)
    post_test_soundtrack.wav     the soundtrack (audio2; one for both editions), reused by the compose runs
    post_test_preview_<ed>.mp4   the composed preview (1920x1080, 24 fps, AAC)
    post_check_sheet_<ed>.jpg    contact sheet (title, every caption, stage-1 part / splices, end card)
    post_check_<ed>_{title,caption,s1caption,endcard}.png   full-size stills: layout, logos, texts
Every edition ends with a line "RESULT <ed>: ..." and the run with "RESULT all: ...".
"""
import argparse
import colorsys
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]           # stage2_logistics
POST = ROOT / "post"
sys.path.insert(0, str(POST))
import compose2                                      # noqa: E402  (puts demo_video/ and stage2_logistics/ on sys.path)
import storyboard2                                   # noqa: E402
import overlays2                                     # noqa: E402
import audio2                                        # noqa: E402
import editions                                      # noqa: E402
import edl as edl_mod                                # noqa: E402
import branding2                                     # noqa: E402
import i18n2                                         # noqa: E402
from post import audio as audio1                     # noqa: E402  (stage-1 audio module)

OUT = ROOT / "out"
WAV = OUT / "post_test_soundtrack.wav"
STEPS = {"ru": 6, "en": 1}       # synthetic frames: every 6th scene frame (holds) / every frame of the edit
SW, SH = 640, 360
BITS = 14                        # scene frames up to 16383
BAR_Y = (120, 148)               # barcode row (640x360): below the title / corner logos, above the lower thirds
OUT_SCALE = 1920 / SW
FONT_B = overlays2.resolve_font("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", True)
FONT = overlays2.resolve_font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", False)

failures = []
counts = {}                      # edition -> [checks, failed]
_cur = ["-"]


def check(cond: bool, msg: str) -> None:
    print(("  ok    " if cond else "  FAIL  ") + msg)
    c = counts.setdefault(_cur[0], [0, 0])
    c[0] += 1
    if not cond:
        c[1] += 1
        failures.append(f"[{_cur[0]}] {msg}")


# ----------------------------------------------------------------------------- synthetic frames
def barcode_boxes(scale: float = 1.0):
    """Bit squares (x0, y0, x1, y1), MSB first; 640x360 layout scaled by ``scale`` (3 for 1080p)."""
    y0, y1 = BAR_Y
    return [tuple(int(round(v * scale)) for v in (24 + 36 * i, y0, 24 + 36 * i + 28, y1)) for i in range(BITS)]


_BG = {}
_FONTS = {}


def make_frame(f: int, shot: str, shot_idx: int) -> Image.Image:
    if shot_idx not in _BG:
        h = (shot_idx * 0.137) % 1.0
        top = np.array(colorsys.hsv_to_rgb(h, 0.55, 0.55)) * 255
        bot = np.array(colorsys.hsv_to_rgb(h, 0.70, 0.25)) * 255
        u = np.linspace(0, 1, SH)[:, None, None]
        _BG[shot_idx] = (top * (1 - u) + bot * u).repeat(SW, axis=1).astype(np.uint8)
    if not _FONTS:
        _FONTS.update(big=ImageFont.truetype(FONT_B, 110), mid=ImageFont.truetype(FONT, 30),
                      small=ImageFont.truetype(FONT, 20))
    img = Image.fromarray(_BG[shot_idx], "RGB")
    d = ImageDraw.Draw(img)
    d.rectangle([14, BAR_Y[0] - 10, 24 + 36 * BITS + 2, BAR_Y[1] + 10], fill=(20, 20, 20))
    for i, (x0, y0, x1, y1) in enumerate(barcode_boxes()):
        bit = (f >> (BITS - 1 - i)) & 1
        d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=(255, 255, 255) if bit else (0, 0, 0))
    d.text((SW / 2, 96), "scene frame", font=_FONTS["small"], fill=(220, 220, 220), anchor="mm")
    d.text((SW / 2, 225), f"{f}", font=_FONTS["big"], fill=(255, 255, 255), anchor="mm")
    d.text((SW / 2, 312), shot, font=_FONTS["mid"], fill=(255, 230, 160), anchor="mm")
    return img


def rendered(seg: dict, step: int):
    return [f for f in range(seg["f0"], seg["f1"] + 1) if (f - 1) % step == 0]


def expected_source(seg: dict, f: int, step: int) -> int:
    """Independent re-statement of the hold rule: nearest rendered <= f inside the shot, else the first after."""
    r = rendered(seg, step)
    le = [x for x in r if x <= f]
    return le[-1] if le else r[0]


def gen_frames(E: dict, folder: Path, step: int) -> list:
    folder.mkdir(parents=True, exist_ok=True)
    for p in folder.glob("frame_*.png"):
        p.unlink()
    made = []
    for k, s in enumerate(x for x in E["segments"] if x["src"] == "s2"):
        for f in rendered(s, step):
            make_frame(f, s["shot"], k).save(folder / f"frame_{f:04d}.png", compress_level=1)
            made.append(f)
    return made


def decode_barcode(rgb: np.ndarray) -> int:
    v = 0
    for x0, y0, x1, y1 in barcode_boxes(OUT_SCALE):
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        lum = rgb[cy - 15:cy + 15, cx - 15:cx + 15].mean()
        v = (v << 1) | int(lum > 128)
    return v


# ----------------------------------------------------------------------------- overlay text recording
def render_recording(sb: dict, out_dir: Path):
    """overlays2.render_all2 with every PIL ImageDraw.text call recorded: (overlays, [drawn strings])."""
    drawn = []
    orig = ImageDraw.ImageDraw.text

    def spy(self, xy, text, *args, **kwargs):
        drawn.append(text)
        return orig(self, xy, text, *args, **kwargs)

    ImageDraw.ImageDraw.text = spy
    try:
        ovs = overlays2.render_all2(sb, out_dir)
    finally:
        ImageDraw.ImageDraw.text = orig
    return ovs, drawn


def storyboard_words(sb: dict) -> set:
    """Words of every text the overlays must draw (tags upper-case, like overlays.render_caption)."""
    t, e = sb["title"], sb["end_card"]
    texts = [t["heading"], t["sub"], e["heading"], e["cta"], sb["footer"], sb["corner_label"]]
    texts += list(e["flow"]) + list(e["kpis"])
    for c in sb["captions"]:
        texts += [c["tag"].upper(), c["text"]]
    return {w for s in texts for w in s.split()}


# ----------------------------------------------------------------------------- checks
def storyboard_checks(ed: dict, E: dict):
    print(f"[storyboard {ed['name']}]")
    sb = storyboard2.load_json(ed["storyboard_json"])
    check(sb == json.loads(json.dumps(storyboard2.build(E), ensure_ascii=False)),
          f"post/{Path(ed['storyboard_json']).name} is up to date with post/{Path(ed['edl_json']).name} + texts")
    check((sb["edition"], sb["lang"], sb["brand"]["key"], sb["stage1_mode"]) ==
          (ed["name"], ed["lang"], ed["brand"], ed["stage1"]),
          f"edition {sb['edition']}, language {sb['lang']}, brand {sb['brand']['name']}, stage 1 {sb['stage1_mode']}")
    check((ROOT / sb["brand"]["logo"]).is_file() and "\\" not in sb["brand"]["logo"]
          and not Path(sb["brand"]["logo"]).is_absolute(), f"brand logo {sb['brand']['logo']} (relative, exists)")
    problems = storyboard2.check(sb, E)
    check(not problems, "storyboard2.check: " + ("no problems" if not problems else "; ".join(problems)))
    # compose2.load_storyboard: the documented relative form from the repository root (compose2 --final --verify),
    # and refusal of a storyboard without logos (written before the editions) or with another brand
    rel = f"post/{Path(ed['storyboard_json']).name}"
    cwd = os.getcwd()
    try:
        os.chdir(ROOT.parent)                        # the repository root: rel does not exist from here
        ok_rel = compose2.load_storyboard(rel, E) == sb
    except SystemExit as e:
        ok_rel = f"refused: {e}"
    finally:
        os.chdir(cwd)
    check(ok_rel is True, f"compose2.load_storyboard({rel!r}) resolves under stage2_logistics/ ({ok_rel})")
    stale = OUT / "tests" / f"post_test_stale_storyboard_{ed['name']}.json"
    stale.parent.mkdir(parents=True, exist_ok=True)
    other = next(k for k in branding2.BRANDS if k != sb["brand"]["key"])
    refused = []
    for what, bad in (("no logos", {k: v for k, v in sb.items() if k not in ("logos", "brand")}),
                      ("brand " + other, dict(sb, brand=dict(sb["brand"], key=other)))):
        stale.write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
        try:
            compose2.load_storyboard(str(stale), E)
        except SystemExit as e:
            refused.append(what if ("logos." in str(e) or "brand" in str(e)) else f"{what}?")
    stale.unlink()
    check(refused == ["no logos", "brand " + other], f"compose2.load_storyboard refuses a storyboard with {refused}")
    fps = sb["fps"]
    spans = storyboard2.overlay_spans(sb)
    gaps = [a2 - b1 - 1 for (_, _, b1, _), (_, a2, _, _) in zip(spans, spans[1:])]
    check(all(b1 < a2 for (_, _, b1, _), (_, a2, _, _) in zip(spans, spans[1:])),
          "title / captions / end card do not overlap")
    check(min(gaps) >= 4, f"gaps between them >= 4 frames (min {min(gaps)})")
    check(all(b - a + 1 > 2 * fade for _, a, b, fade in spans), "every overlay is longer than its two fades")
    check(all(c["end"] - c["start"] + 1 >= 28 for c in sb["captions"]),
          f"captions >= 28 frames (min {min(c['end'] - c['start'] + 1 for c in sb['captions'])})")
    s1v = [(s["out0"], s["out1"]) for s in E["segments"] if s["src"] == "s1"]
    check(all(b < o0 or a > o1 for _, a, b, _ in spans for o0, o1 in s1v),
          f"no title / caption / end card over the stage-1 video segments {s1v or '(none: render mode)'}")
    parts1 = storyboard2.stage1_parts(E)
    caps2 = [c for c in sb["captions"] if not c.get("s1")]
    caps1 = [c for c in sb["captions"] if c.get("s1")]
    check(all(c["end"] < o0 or c["start"] > o1 for c in caps2 for o0, o1 in parts1),
          f"stage-2 captions stay out of the stage-1 part {parts1}")
    s1_sb = storyboard2.load_json(storyboard2.S1_STORYBOARD)
    if ed["stage1"] == "video":
        burned = storyboard2.s1_burned_spans(E, s1_sb)
        track = sorted([(a, b) for _, a, b, _ in spans] + [(a, b) for _, a, b, _ in burned])
        gaps_all = [a2 - b1 - 1 for (_, b1), (a2, _) in zip(track, track[1:])]
        check(all(full for *_, full in burned) and min(gaps_all) >= 4,
              f"{len(burned)} burned-in stage-1 texts are complete and >= 4 frames from the stage-2 overlays "
              f"(min gap on the whole timeline {min(gaps_all)})")
        check(not caps1, "no stage-1 captions drawn (they are burned into the stage-1 video)")
    else:
        want = []
        for c in s1_sb["captions"]:
            for o0, o1 in edl_mod.map_stage1_intervals(E, [[c["start"], c["end"]]]):
                en = storyboard2.S1_CAPTIONS_EN[c["tag"]]
                want.append((o0, o1, en["tag"], en["text"], o1 - o0 == c["end"] - c["start"]))
        got = [(c["start"], c["end"], c["tag"], c["text"], True) for c in caps1]
        check(len(want) == 6 and sorted(got) == sorted(want),
              f"{len(caps1)} stage-1 captions, translated, complete, at the output frames of their stage-1 frames")
        for c in caps1:
            print(f"      {c['key']:<12s} {c['start']:5d}-{c['end']:5d} = stage-1 {c['s1'][0]}-{c['s1'][1]}  "
                  f"{c['tag']}")
    capd = {c["key"]: c for c in caps2}
    segs_c = [s for s in E["segments"] if s["src"] == "s2" and s.get("caption")]
    check(all(capd[s["caption"]]["start"] == s["out0"] + 3 for s in segs_c) and len(capd) == len(segs_c),
          f"one caption per captioned stage-2 shot, starting at out0 + 3 ({len(segs_c)} captions)")
    check(sb["title"]["start"] == 1 and sb["title"]["end"] == E["segments"][0]["out1"] - 10, "title 1 .. out1 - 10")
    ec = sb["end_card"]
    check(ec["end"] == E["frames"] and 96 <= ec["end"] - ec["start"] + 1 <= 110,
          f"end card = last {ec['end'] - ec['start'] + 1} frames")
    if sb["lang"] != "ru":
        cyr = [p for p, s in storyboard2.strings(sb) if i18n2.has_cyrillic(s) and p not in storyboard2.CYRILLIC_OK]
        check(not cyr, f"no Cyrillic in the {sb['lang']} storyboard texts" + (f": {cyr[:4]}" if cyr else ""))

    # overlays: rendered with every drawn string recorded
    ovs, drawn = render_recording(sb, OUT / f"post_test_overlays_{ed['name']}")
    joined = "".join(drawn)
    if sb["lang"] != "ru":
        bad = sorted({s for s in drawn if i18n2.has_cyrillic(s)})
        check(not bad, f"{len(drawn)} strings drawn by overlays2.render_all2, none with Cyrillic"
                       + (f": {bad[:6]}" if bad else ""))
    else:
        check(sb["title"]["heading"] in drawn and storyboard2.TITLE["heading"] == sb["title"]["heading"],
              "Russian title drawn unchanged")
    missing = sorted(w for w in storyboard_words(sb) if w not in joined)
    check(not missing, "every word of the storyboard texts is drawn" + (f"; missing {missing[:6]}" if missing else ""))
    blocks = storyboard2.s2_blocks(E)
    corners = [o for o in ovs if o.name.startswith("corner")]
    fds = [overlays2.fades(o) for o in corners]
    check([[o.start, o.end] for o in corners] == blocks, f"one corner label per stage-2 range {blocks}")
    check(fds[0][0] == 12 and fds[-1][1] == 12 and all(f[1] == 0 for f in fds[:-1])
          and all(f[0] == 0 for f in fds[1:]), f"corner label fades only at video start/end, hard at splices {fds}")
    if ed["stage1"] == "render":
        check(blocks == [[1, E["frames"]]], "render mode: one corner label over the whole video")
    lt = [o for o in ovs if o.name == "logo_title"]
    lc = [o for o in ovs if o.name == "logo_corner"]
    t = sb["title"]
    check(len(lt) == 1 and (lt[0].start, lt[0].end, overlays2.fades(lt[0]), lt[0].whole) ==
          (t["start"], t["end"], (12, 12), False) and (lt[0].x, lt[0].y) == (80, 64)
          and Image.open(lt[0].path).height == 150, "title logo: 150 px plate at (80, 64), title timing and fades")
    check(len(lc) == 1 and (lc[0].start, lc[0].end, overlays2.fades(lc[0]), lc[0].whole) ==
          (t["end"] + 1, ec["start"] - 1, (12, 12), True) and (lc[0].x, lc[0].y) == (80, 40)
          and Image.open(lc[0].path).height == 64,
          f"corner logo: 64 px plate at (80, 40), whole video {t['end'] + 1}..{ec['start'] - 1}, 12-frame fades")
    lay = overlays2.end_card_layout(overlays2.with_fonts(sb))
    check(Image.open([o for o in ovs if o.name == "end_card"][0].path).size == (1920, 1080)
          and lay["logo_box"] is not None and 110 <= lay["logo_h"] <= 130 and lay["end"] <= lay["limit"],
          f"end card 1920x1080: logo {lay['logo_h']} px, content {lay['y0']}..{lay['end']} above the footer "
          f"(<= {lay['limit']}, gaps at {lay['spacing']:.0%})")
    for c in sb["captions"]:
        print(f"      {c['key']:<12s} {c['start']:5d}-{c['end']:5d} ({(c['end'] - c['start'] + 1) / fps:.1f} s) "
              f"{c['tag']}")
    return sb


def audio_checks(sbs: dict) -> None:
    """The soundtrack plan is the same in both editions (built from both EDLs, whatever edition is tested);
    synthesis + level checks once."""
    _cur[0] = "audio"
    print("[audio]")
    plans = {}
    for name in editions.names():
        ed = editions.get(name)
        plans[name] = sbs[name]["audio"] if name in sbs else storyboard2.build(
            storyboard2.load_json(ed["edl_json"]))["audio"]
    ref = plans[editions.names()[0]]
    check(all(p == ref for p in plans.values()),
          f"audio plan identical in the editions {', '.join(plans)} (stage-1 weld/servo via the video segments in ru, "
          f"via the re-rendered stage-1 shots in en)")
    sb = next(iter(sbs.values()))
    t0 = time.time()
    stems = {}
    data = audio2.synthesize2(sb, stems_out=stems)
    audio1.write_wav(str(WAV), data)
    n_exp = int(round(sb["frames"] / sb["fps"] * audio1.SR))
    check(data.shape == (n_exp, 2), f"48 kHz stereo, {data.shape[0] / audio1.SR:.3f} s = frames/fps "
                                    f"({time.time() - t0:.0f} s to synthesize)")
    pk = 20 * math.log10(np.max(np.abs(data)))
    check(abs(pk - sb["audio"]["peak_db"]) < 0.05, f"peak {pk:.2f} dBFS")
    rep = {n: (r, p) for n, r, p in audio2.layer_report(stems)}
    for n, (r, p) in rep.items():
        print(f"      {n:<12s} rms {r:6.1f} dBFS in its intervals, peak {p:6.1f}")
    servo = rep["servo_s1"][0]
    new = [n for n in rep if n not in ("ambience", "servo_s1", "weld_s1", "chime")]
    check(all(rep[n][0] <= servo + 0.5 for n in new), f"new layers not louder than the stage-1 servo ({servo:.1f})")
    check("chime" in rep and servo < rep["chime"][0] <= servo + 6, "QC OK chime slightly above the servo level")
    check(rep["weld_s1"][0] > servo + 3, "stage-1 arc stays the loudest layer")
    g = 20 * math.log10(stems["_gain"])
    check(abs(rep["ambience"][0] - (sb["audio"]["ambience_db"] + g)) < 1.0,
          f"ambience {sb['audio']['ambience_db']} dBFS + normalisation {g:+.1f} dB")
    check(all(k in rep for k in ("tack_arc", "clamp", "marker", "scan", "agv")),
          "layers tack_arc, clamp, marker, scan, agv present")


def probe(mp4: Path) -> dict:
    r = subprocess.run([compose2.tool("ffprobe"), "-v", "error", "-show_entries",
                        "stream=codec_type,codec_name,width,height,r_frame_rate,pix_fmt,nb_frames,sample_rate,channels,"
                        "duration:format=duration", "-of", "json", str(mp4)], capture_output=True,
                       encoding="utf-8", errors="replace")
    return json.loads(r.stdout)


def sheet(frames: dict, cps, path: Path, cols: int = 4, tw: int = 480, th: int = 270) -> None:
    rows = math.ceil(len(cps) / cols)
    lab_h = 30
    img = Image.new("RGB", (cols * tw, rows * (th + lab_h)), (16, 16, 16))
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(FONT_B, 18)
    for k, (o, what) in enumerate(cps):
        x, y = (k % cols) * tw, (k // cols) * (th + lab_h)
        img.paste(Image.fromarray(frames[o - 1]).resize((tw, th), Image.LANCZOS), (x, y + lab_h))
        d.text((x + 8, y + 6), f"out {o} · {what}", font=f, fill=(255, 210, 60))
    img.save(path, quality=90)


def brand_fraction(rgb: np.ndarray, box, color, tol: float = 60.0) -> float:
    """Fraction of the pixels in box (x0, y0, x1, y1) within ``tol`` (RGB distance) of the brand colour."""
    x0, y0, x1, y1 = box
    reg = rgb[y0:y1, x0:x1].astype(np.float64)
    return float((np.sqrt(((reg - np.array(color, float)) ** 2).sum(axis=2)) < tol).mean())


def plate_fraction(sb: dict, height: int) -> float:
    """The same fraction measured on the logo plate image itself (what a perfect frame would give)."""
    im = np.asarray(branding2.logo_plate(sb["brand"]["key"], height=height).convert("RGB"))
    return brand_fraction(im, (0, 0, im.shape[1], im.shape[0]), sb["brand"]["color"])


# ----------------------------------------------------------------------------- one edition
def run_edition(name: str, threads: int, skip_compose: bool, sb: dict) -> None:
    _cur[0] = name
    ed = editions.get(name)
    E = storyboard2.load_json(ed["edl_json"])
    fps, n_frames = E["fps"], E["frames"]
    step = STEPS[name]
    frames_dir = OUT / f"post_test_frames_{name}"
    mp4 = OUT / f"post_test_preview_{name}.mp4"

    print(f"[compose {name}]")
    if not skip_compose:
        t0 = time.time()
        made = gen_frames(E, frames_dir, step)
        print(f"      {len(made)} synthetic frames ({SW}x{SH}, every {step}. scene frame) in {time.time() - t0:.0f} s")
        if step == 1:
            miss, _ = edl_mod.check_frames(E, frames_dir)
            check(sorted(made) == E["render_frames"] and not miss,
                  f"every frame of the edit rendered: {len(made)} = {len(E['render_frames'])} render_frames "
                  f"(edl.check_frames: {len(miss)} missing)")
        t0 = time.time()
        compose2.compose2(frames_dir, mp4, ed["edl_json"], ed["storyboard_json"], preview=True, threads=threads,
                          soundtrack=str(WAV), edition=name)
        print(f"      compose2 --preview: {time.time() - t0:.0f} s -> {mp4}")

    pr = probe(mp4)
    v = [s for s in pr["streams"] if s["codec_type"] == "video"][0]
    au = [s for s in pr["streams"] if s["codec_type"] == "audio"]
    dur = float(pr["format"]["duration"])
    check(abs(dur - n_frames / fps) <= 1 / fps + 1e-6, f"duration {dur:.3f} s == {n_frames}/{fps} = "
                                                         f"{n_frames / fps:.3f} s (±1 frame)")
    check(int(v["nb_frames"]) == n_frames == 2152, f"video frames {v['nb_frames']} == {n_frames} (2152)")
    check((v["codec_name"], v["width"], v["height"], v["r_frame_rate"], v["pix_fmt"]) ==
          ("h264", 1920, 1080, f"{fps}/1", "yuv420p"), f"video {v['codec_name']} {v['width']}x{v['height']} "
                                                      f"{v['r_frame_rate']} {v['pix_fmt']}")
    check(len(au) == 1 and au[0]["codec_name"] == "aac" and au[0]["sample_rate"] == "48000" and au[0]["channels"] == 2,
          "audio stream present: " + (f"{au[0]['codec_name']} {au[0]['sample_rate']} Hz {au[0]['channels']} ch "
                                      f"{float(au[0]['duration']):.3f} s" if au else "none"))

    segs1 = [x for x in E["segments"] if x["src"] == "s1"]
    if ed["stage1"] == "video":
        print("[splices]  output frame vs stage-1 frame (PSNR, 1920x1080, corner logo masked)")
        res = compose2.verify_splices(mp4, ed["edl_json"], exclude=compose2.mask_boxes(sb))
        check(len(res) == 2 * len(segs1) > 0, f"{len(res)} splice frames compared")
        for r in res:
            nb = ", ".join(f"{d:+d}: {p:.1f}" for d, p in r["neighbours"].items())
            check(r["ok"], f"out {r['out']:5d} = stage-1 frame {r['s1']:4d}: {r['psnr']:.1f} dB > 30, best match "
                           f"(neighbours {nb})")
    else:
        print("[splices]")
        check(not segs1 and compose2.verify_splices(mp4, ed["edl_json"]) == [],
              "render mode: no stage-1 video segments, no splices (the stage-1 shots are stage-2 renders)")

    print("[stage-2 content]  barcode of the held scene frame" + (" (incl. the stage-1 shots)" if not segs1 else ""))
    segs2 = [s for s in E["segments"] if s["src"] == "s2"]
    ec0, ec1 = sb["end_card"]["start"], sb["end_card"]["end"]
    probes = []
    for s in segs2:
        for o in sorted({s["out0"], s["out0"] + 1, (s["out0"] + s["out1"]) // 2, s["out1"]}):
            if o < ec0:
                probes.append((o, s))
    caps2 = [c for c in sb["captions"] if not c.get("s1")]
    cap_still = max(caps2, key=lambda c: len(c["text"]))
    still_cap = (cap_still["start"] + cap_still["end"]) // 2
    # stage-1 caption still: the longest stage-1 caption of the used frames, at the same output frame in both editions
    s1_sb = storyboard2.load_json(storyboard2.S1_STORYBOARD)
    s1c = max((c for c in s1_sb["captions"] if edl_mod.map_stage1_intervals(E, [[c["start"], c["end"]]])),
              key=lambda c: c["end"] - c["start"])
    o0, o1 = edl_mod.map_stage1_intervals(E, [[s1c["start"], s1c["end"]]])[0]
    still_s1 = (o0 + o1) // 2
    title_f, end_f = 50, ec1 - 30
    mid_f = n_frames // 2                                            # ru: stage-1 video, en: re-rendered S1 shot
    post_seg = [s for s in segs2 if s["out0"] > storyboard2.stage1_parts(E)[-1][1]][2]      # S2_12_carrier
    post_f = (post_seg["out0"] + post_seg["out1"]) // 2
    s1_logo_f = [(s["out0"] + s["out1"]) // 2 for s in segs1]
    ctrl_f = ec0 + 30
    cps = [(title_f, "title")]
    cps += [((c["start"] + c["end"]) // 2, f"{c['key']} · {c['tag']}") for c in sb["captions"]]
    for s in segs1:
        cps += [(s["out0"] - 1, "before splice"), (s["out0"], f"s1 frame {s['f0']}")]
    if segs1:
        cps += [(segs1[-1]["out1"], f"s1 frame {segs1[-1]['f1']}"), (segs1[-1]["out1"] + 1, "after splice")]
    else:
        p1 = storyboard2.stage1_parts(E)[0]
        cps += [(p1[0] - 1, "before stage-1 shots"), (p1[0], "S1 shots start"), (p1[1], "S1 shots end"),
                (p1[1] + 1, "after stage-1 shots")]
    cps += [(still_s1, "stage-1 caption"), (ec0 + 7, "end card: fade in"), (end_f, "end card")]
    cps = sorted(set(cps))
    want = ([o for o, _ in probes] + [o for o, _ in cps] + [still_cap, still_s1, title_f, end_f, mid_f, post_f,
                                                             ctrl_f] + s1_logo_f)
    frames = compose2.extract_frames(mp4, [o - 1 for o in want])
    bad = []
    for o, s in probes:
        f = s["f0"] + (o - s["out0"])
        exp = expected_source(s, f, step)
        got = decode_barcode(frames[o - 1])
        if got != exp:
            bad.append(f"out {o} ({s['shot']}, scene {f}): barcode {got} != {exp}")
    n_s1 = len([s for s in segs2 if s.get("s1")])
    check(not bad, f"{len(probes)} output frames in all {len(segs2)} stage-2 segments"
                   + (f" (incl. the {n_s1} re-rendered stage-1 shots)" if n_s1 else "")
                   + " show the expected synthetic frame" + ("" if not bad else ": " + "; ".join(bad[:6])))

    print(f"[logos]  {sb['brand']['name']} colour {tuple(sb['brand']['color'])} in the logo boxes of the output")
    boxes = overlays2.logo_boxes(sb)
    lg = sb["logos"]
    ref = {k: plate_fraction(sb, int(boxes[k][3] - boxes[k][1])) for k in boxes}
    col = sb["brand"]["color"]
    where = [("title", title_f, "title logo (title)"), ("corner", mid_f, "corner logo, mid-video frame"
                                                         + (" (stage-1 video)" if segs1 else " (stage-1 shot render)")),
             ("corner", post_f, f"corner logo ({post_seg['shot']})"), ("end_card", end_f, "end-card logo")]
    where += [("corner", f, f"corner logo inside the stage-1 video segment") for f in s1_logo_f]
    for k, f, what in where:
        fr = brand_fraction(frames[f - 1], boxes[k], col)
        check(fr >= 0.6 * ref[k], f"{what}: out {f}, box {boxes[k]}: {fr:.1%} brand-colour pixels "
                                  f"(plate itself {ref[k]:.1%})")
    fr = brand_fraction(frames[ctrl_f - 1], boxes["title"], col)
    check(fr < 0.002, f"no logo top left during the end card (out {ctrl_f}: {fr:.2%}; corner logo ends "
                      f"{lg['corner']['end']})")

    sheet(frames, cps, OUT / f"post_check_sheet_{name}.jpg")
    stills = {"title": title_f, "caption": still_cap, "s1caption": still_s1, "endcard": end_f}
    for k, o in stills.items():
        Image.fromarray(frames[o - 1]).save(OUT / f"post_check_{name}_{k}.png")
    print(f"      contact sheet ({len(cps)} frames): {OUT / f'post_check_sheet_{name}.jpg'}")
    print("      stills: " + ", ".join(f"post_check_{name}_{k}.png (out {o})" for k, o in stills.items()))


# ----------------------------------------------------------------------------- main
def main() -> None:
    editions.safe_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--edition", default="all", choices=editions.names() + ["all"])
    ap.add_argument("--threads", type=int, default=2, help="x264 threads (the machine is shared)")
    ap.add_argument("--skip-compose", action="store_true", help="reuse out/post_test_preview_<ed>.mp4")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    names = editions.names() if a.edition == "all" else [a.edition]
    sbs = {}
    for name in names:
        _cur[0] = name
        sbs[name] = storyboard_checks(editions.get(name), storyboard2.load_json(editions.get(name)["edl_json"]))
    audio_checks(sbs)
    for name in names:
        run_edition(name, a.threads, a.skip_compose, sbs[name])

    print()
    for k, (n, nf) in counts.items():
        print(f"RESULT {k}: {n - nf}/{n} checks passed" + (f", {nf} FAILED" if nf else ""))
    tot, totf = sum(c[0] for c in counts.values()), sum(c[1] for c in counts.values())
    print(f"RESULT all: {tot - totf}/{tot} checks passed ({', '.join(names)})" + (f", {totf} FAILED" if totf else ""))
    if failures:
        for f in failures:
            print("  -", f)
        sys.exit(1)


if __name__ == "__main__":
    main()
