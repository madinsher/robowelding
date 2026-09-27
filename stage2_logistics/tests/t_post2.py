"""Stage-2 post-production test: storyboard checks -> soundtrack checks -> synthetic stage-2 frames -> compose2.py
(--preview) -> ffprobe / frame-accurate splice checks / synthetic-content checks -> contact sheet.

    python3 stage2_logistics/tests/t_post2.py [--threads 2] [--skip-compose]

Outputs (stage2_logistics/out/, git-ignored):
    post_test_frames/          synthetic frames: every 6th scene frame (1, 7, 13, ...) of every stage-2 shot, 640x360,
                               showing the scene frame number, the shot name and a 12-bit barcode of the frame number
    post_test_soundtrack.wav   the soundtrack (audio2), reused by the compose run
    post_test_preview.mp4      the composed preview (1920x1080, 24 fps, AAC)
    post_check_sheet.jpg       contact sheet of the checkpoint frames (title, every caption, splices, end card)
    post_check_endcard.png, post_check_caption.png   full-size stills for a closer look at the layout
"""
import argparse
import colorsys
import json
import math
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
from post import audio as audio1                     # noqa: E402  (stage-1 audio module)

OUT = ROOT / "out"
FRAMES = OUT / "post_test_frames"
MP4 = OUT / "post_test_preview.mp4"
WAV = OUT / "post_test_soundtrack.wav"
SHEET = OUT / "post_check_sheet.jpg"
STEP = 6
SW, SH = 640, 360
BITS = 14                        # scene frames up to 16383
FONT_B = overlays2.resolve_font("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", True)
FONT = overlays2.resolve_font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", False)

failures = []


def check(cond: bool, msg: str) -> None:
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        failures.append(msg)


# ----------------------------------------------------------------------------- synthetic frames
def barcode_boxes(scale: float = 1.0):
    """Bit squares (x0, y0, x1, y1), MSB first; 640x360 layout scaled by ``scale`` (3 for 1080p)."""
    return [tuple(int(round(v * scale)) for v in (24 + 36 * i, 56, 24 + 36 * i + 28, 84)) for i in range(BITS)]


def make_frame(f: int, shot: str, shot_idx: int) -> Image.Image:
    h = (shot_idx * 0.137) % 1.0
    top = np.array(colorsys.hsv_to_rgb(h, 0.55, 0.55)) * 255
    bot = np.array(colorsys.hsv_to_rgb(h, 0.70, 0.25)) * 255
    u = np.linspace(0, 1, SH)[:, None, None]
    img = Image.fromarray((top * (1 - u) + bot * u).repeat(SW, axis=1).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(img)
    d.rectangle([14, 46, 24 + 36 * BITS + 2, 94], fill=(20, 20, 20))
    for i, (x0, y0, x1, y1) in enumerate(barcode_boxes()):
        bit = (f >> (BITS - 1 - i)) & 1
        d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=(255, 255, 255) if bit else (0, 0, 0))
    d.text((SW / 2, 190), f"{f}", font=ImageFont.truetype(FONT_B, 120), fill=(255, 255, 255), anchor="mm")
    d.text((SW / 2, 275), shot, font=ImageFont.truetype(FONT, 30), fill=(255, 230, 160), anchor="mm")
    d.text((SW / 2, 110), "scene frame", font=ImageFont.truetype(FONT, 20), fill=(220, 220, 220), anchor="mm")
    return img


def rendered(seg: dict):
    return [f for f in range(seg["f0"], seg["f1"] + 1) if (f - 1) % STEP == 0]


def expected_source(seg: dict, f: int) -> int:
    """Independent re-statement of the hold rule: nearest rendered <= f inside the shot, else the first after."""
    r = rendered(seg)
    le = [x for x in r if x <= f]
    return le[-1] if le else r[0]


def gen_frames(E: dict) -> int:
    FRAMES.mkdir(parents=True, exist_ok=True)
    for p in FRAMES.glob("frame_*.png"):
        p.unlink()
    n = 0
    for k, s in enumerate(x for x in E["segments"] if x["src"] == "s2"):
        for f in rendered(s):
            make_frame(f, s["shot"], k).save(FRAMES / f"frame_{f:04d}.png", compress_level=1)
            n += 1
    return n


def decode_barcode(rgb: np.ndarray) -> int:
    v = 0
    for x0, y0, x1, y1 in barcode_boxes(OUT_SCALE):
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        lum = rgb[cy - 15:cy + 15, cx - 15:cx + 15].mean()
        v = (v << 1) | int(lum > 128)
    return v


OUT_SCALE = 1920 / SW


# ----------------------------------------------------------------------------- checks
def storyboard_checks(E: dict) -> dict:
    print("[storyboard]")
    sb = storyboard2.load_json(storyboard2.SB2_JSON)
    check(sb == json.loads(json.dumps(storyboard2.build(E), ensure_ascii=False)),
          "post/storyboard2.json is up to date with edl.json + storyboard2.py texts")
    problems = storyboard2.check(sb, E)
    check(not problems, "storyboard2.check: " + ("no problems" if not problems else "; ".join(problems)))
    fps = sb["fps"]
    spans = storyboard2.overlay_spans(sb)
    ok_overlap = all(b1 < a2 for (_, _, b1, _), (_, a2, _, _) in zip(spans, spans[1:]))
    gaps = [a2 - b1 - 1 for (_, _, b1, _), (_, a2, _, _) in zip(spans, spans[1:])]
    check(ok_overlap, "title / captions / end card do not overlap")
    check(min(gaps) >= 4, f"gaps between them >= 4 frames (min {min(gaps)})")
    check(all(b - a + 1 > 2 * fade for _, a, b, fade in spans), "every overlay is longer than its two fades")
    check(all(c["end"] - c["start"] + 1 >= 28 for c in sb["captions"]),
          f"captions >= 28 frames (min {min(c['end'] - c['start'] + 1 for c in sb['captions'])})")
    s1 = [(s["out0"], s["out1"]) for s in E["segments"] if s["src"] == "s1"]
    check(all(b < o0 or a > o1 for _, a, b, _ in spans for o0, o1 in s1), "nothing over the stage-1 segments")
    burned = storyboard2.s1_burned_spans(E, storyboard2.load_json(storyboard2.S1_STORYBOARD))
    track = sorted([(a, b) for _, a, b, _ in spans] + [(a, b) for _, a, b, _ in burned])
    gaps_all = [a2 - b1 - 1 for (_, b1), (a2, _) in zip(track, track[1:])]
    check(all(full for *_, full in burned) and min(gaps_all) >= 4,
          f"{len(burned)} burned-in stage-1 texts are complete and >= 4 frames from the stage-2 overlays "
          f"(min gap on the whole timeline {min(gaps_all)})")
    capd = {c["key"]: c for c in sb["captions"]}
    segs_c = [s for s in E["segments"] if s["src"] == "s2" and s.get("caption")]
    check(all(capd[s["caption"]]["start"] == s["out0"] + 3 for s in segs_c) and len(capd) == len(segs_c),
          f"one caption per captioned shot, starting at out0 + 3 ({len(segs_c)} captions)")
    check(sb["title"]["start"] == 1 and sb["title"]["end"] == E["segments"][0]["out1"] - 10, "title 1 .. out1 - 10")
    ec = sb["end_card"]
    check(ec["end"] == E["frames"] and 96 <= ec["end"] - ec["start"] + 1 <= 110,
          f"end card = last {ec['end'] - ec['start'] + 1} frames")
    blocks = storyboard2.s2_blocks(E)
    ovs = overlays2.render_all2(sb, OUT / "post_test_overlays")
    corners = [o for o in ovs if o.name.startswith("corner")]
    check([[o.start, o.end] for o in corners] == blocks, f"one corner label per stage-2 range {blocks}")
    fades = [overlays2.fades(o) for o in corners]
    check(fades[0][0] == 12 and fades[-1][1] == 12 and all(f[1] == 0 for f in fades[:-1])
          and all(f[0] == 0 for f in fades[1:]), f"corner label fades only at video start/end, hard at splices {fades}")
    check(Image.open([o for o in ovs if o.name == "end_card"][0].path).size == (1920, 1080), "end card 1920x1080")
    for c in sb["captions"]:
        print(f"      {c['key']:<8s} {c['start']:5d}-{c['end']:5d} ({(c['end'] - c['start'] + 1) / fps:.1f} s) {c['tag']}")
    return sb


def audio_checks(sb: dict) -> None:
    print("[audio]")
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
    for key, lay in (("tack_arc", "tack_arc"), ("clamp", "clamp"), ("marker", "marker"), ("scan", "scan"),
                     ("agv", "agv")):
        check(key in rep, f"layer {key} present")


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


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2, help="x264 threads (the machine is shared)")
    ap.add_argument("--skip-compose", action="store_true", help="reuse out/post_test_preview.mp4")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    E = storyboard2.load_json(storyboard2.EDL_JSON)
    fps, n_frames = E["fps"], E["frames"]
    sb = storyboard_checks(E)
    audio_checks(sb)

    print("[compose]")
    if not a.skip_compose:
        t0 = time.time()
        n = gen_frames(E)
        print(f"      {n} synthetic frames ({SW}x{SH}, every {STEP}th scene frame) in {time.time() - t0:.0f} s")
        t0 = time.time()
        compose2.compose2(FRAMES, MP4, preview=True, threads=a.threads, soundtrack=str(WAV))
        print(f"      compose2 --preview: {time.time() - t0:.0f} s -> {MP4}")

    pr = probe(MP4)
    v = [s for s in pr["streams"] if s["codec_type"] == "video"][0]
    au = [s for s in pr["streams"] if s["codec_type"] == "audio"]
    dur = float(pr["format"]["duration"])
    check(abs(dur - n_frames / fps) <= 1 / fps + 1e-6, f"duration {dur:.3f} s == {n_frames}/{fps} = "
                                                         f"{n_frames / fps:.3f} s (±1 frame)")
    check(int(v["nb_frames"]) == n_frames, f"video frames {v['nb_frames']} == {n_frames}")
    check((v["codec_name"], v["width"], v["height"], v["r_frame_rate"], v["pix_fmt"]) ==
          ("h264", 1920, 1080, f"{fps}/1", "yuv420p"), f"video {v['codec_name']} {v['width']}x{v['height']} "
                                                      f"{v['r_frame_rate']} {v['pix_fmt']}")
    check(len(au) == 1 and au[0]["codec_name"] == "aac" and au[0]["sample_rate"] == "48000" and au[0]["channels"] == 2,
          "audio stream present: " + (f"{au[0]['codec_name']} {au[0]['sample_rate']} Hz {au[0]['channels']} ch "
                                      f"{float(au[0]['duration']):.3f} s" if au else "none"))

    print("[splices]  output frame vs stage-1 frame (PSNR, 1920x1080)")
    for r in compose2.verify_splices(MP4):
        nb = ", ".join(f"{d:+d}: {p:.1f}" for d, p in r["neighbours"].items())
        check(r["ok"], f"out {r['out']:5d} = stage-1 frame {r['s1']:4d}: {r['psnr']:.1f} dB > 30, best match "
                       f"(neighbours {nb})")

    print("[stage-2 content]  barcode of the held scene frame")
    segs2 = [s for s in E["segments"] if s["src"] == "s2"]
    ec0 = sb["end_card"]["start"]
    probes = []
    for s in segs2:
        for o in sorted({s["out0"], s["out0"] + 1, (s["out0"] + s["out1"]) // 2, s["out1"]}):
            if o < ec0:
                probes.append((o, s))
    cps = [(40, "титул")]
    cps += [((c["start"] + c["end"]) // 2, f"{c['key']} · {c['tag']}") for c in sb["captions"]]
    segs1 = [x for x in E["segments"] if x["src"] == "s1"]
    for s in segs1:
        cps += [(s["out0"] - 1, "до склейки"), (s["out0"], f"s1 кадр {s['f0']}")]
    cps += [(segs1[-1]["out1"], f"s1 кадр {segs1[-1]['f1']}"), (segs1[-1]["out1"] + 1, "после склейки"),
            (sb["end_card"]["start"] + 7, "финал: появление"), (sb["end_card"]["end"] - 30, "финал")]
    cps = sorted(set(cps))
    caption_still = max(sb["captions"], key=lambda c: len(c["text"]))
    still_o = (caption_still["start"] + caption_still["end"]) // 2
    frames = compose2.extract_frames(MP4, [o - 1 for o, _ in probes] + [o - 1 for o, _ in cps] + [still_o - 1])
    bad = []
    for o, s in probes:
        f = s["f0"] + (o - s["out0"])
        exp = expected_source(s, f)
        got = decode_barcode(frames[o - 1])
        if got != exp:
            bad.append(f"out {o} ({s['shot']}, scene {f}): barcode {got} != {exp}")
    check(not bad, f"{len(probes)} output frames in all {len(segs2)} stage-2 shots show the expected synthetic frame"
                   + ("" if not bad else ": " + "; ".join(bad[:6])))

    sheet(frames, cps, SHEET)
    Image.fromarray(frames[sb["end_card"]["end"] - 31]).save(OUT / "post_check_endcard.png")
    Image.fromarray(frames[still_o - 1]).save(OUT / "post_check_caption.png")
    print(f"      contact sheet ({len(cps)} frames): {SHEET}")
    print(f"      stills: {OUT / 'post_check_endcard.png'}, {OUT / 'post_check_caption.png'} "
          f"({caption_still['key']}, out {still_o})")

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
