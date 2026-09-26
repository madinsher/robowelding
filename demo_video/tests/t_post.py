"""Post-production smoke test: synthetic frames -> compose.py (preview + full-rate excerpt) ->
extracted check stills in out/post_check2_*.png.

    python3 tests/t_post.py            # every 8th frame (preview) + full-rate excerpt 930..1104
    python3 tests/t_post.py --full     # also a full 1..1104 full-rate run (slower)

The frame count comes from post/storyboard.json ("frames": 1104 = 46 s at 24 fps).
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from post.compose import compose, FFMPEG   # noqa: E402
from post import audio as audio_mod        # noqa: E402

OUT = ROOT / "out"
SB_PATH = ROOT / "post" / "storyboard.json"
W, H = 1920, 1080
N_FRAMES = json.load(open(SB_PATH, encoding="utf-8"))["frames"]
FPS = json.load(open(SB_PATH, encoding="utf-8"))["fps"]


def make_frame(n: int, w: int = W, h: int = H) -> Image.Image:
    """Moving diagonal gradient + orbiting box + big frame number (so timing is verifiable)."""
    u = (n - 1) / (N_FRAMES - 1)
    # gradient built with a small image and upscaled (fast)
    g = Image.new("RGB", (64, 36))
    px = g.load()
    for y in range(36):
        for x in range(64):
            v = (x / 63 + y / 35) / 2
            px[x, y] = (int(40 + 160 * ((v + u) % 1)), int(60 + 120 * v), int(120 + 100 * (1 - v)))
    img = g.resize((w, h), Image.BILINEAR)
    d = ImageDraw.Draw(img)
    cx, cy = w * (0.5 + 0.35 * math.cos(u * 6.28)), h * (0.45 + 0.25 * math.sin(u * 6.28))
    d.rectangle([cx - w * 0.06, cy - w * 0.06, cx + w * 0.06, cy + w * 0.06], fill=(240, 200, 60))
    d.text((w * 0.03, h * 0.03), f"frame {n:04d}", fill=(255, 255, 255))
    return img


def gen_frames(folder: Path, numbers, w: int = W, h: int = H) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for n in numbers:
        p = folder / f"frame_{n:04d}.png"
        if not p.exists():
            make_frame(n, w, h).save(p, compress_level=1)


def grab(mp4: Path, t: float, out_png: Path) -> None:
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(mp4),
                    "-frames:v", "1", str(out_png)], check=True)


def probe(mp4: Path) -> str:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "stream=codec_name,width,height,r_frame_rate,pix_fmt,sample_rate,channels:format=duration",
                        "-of", "compact", str(mp4)], capture_output=True, text=True)
    return r.stdout.strip()


def check_storyboard(sb: dict) -> None:
    """Assert the overlay schedule is sane: no overlaps, fades fit, end card inside the timeline."""
    spans = [("title", sb["title"]["start"], sb["title"]["end"])]
    spans += [(c["tag"], c["start"], c["end"]) for c in sb["captions"]]
    spans += [("end_card", sb["end_card"]["start"], sb["end_card"]["end"])]
    for name, a, b in spans:
        assert 1 <= a <= b <= sb["frames"], f"{name}: {a}..{b} outside 1..{sb['frames']}"
        assert b - a + 1 >= 2 * 12, f"{name}: {b - a + 1} frames is shorter than two 12-frame fades"
    for (n1, a1, b1), (n2, a2, b2) in zip(spans, spans[1:]):
        assert b1 < a2, f"overlap: {n1} ({a1}..{b1}) vs {n2} ({a2}..{b2})"
    servo_iv, weld_iv = audio_mod.plan_intervals(sb)
    for iv in (servo_iv, weld_iv):
        for (a1, b1), (a2, b2) in zip(iv, iv[1:]):
            assert b1 <= a2, f"audio intervals overlap: {a1:.2f}..{b1:.2f} vs {a2:.2f}..{b2:.2f}"
    print(f"storyboard ok: {len(sb['captions'])} captions, end card {sb['end_card']['start']}..{sb['end_card']['end']}"
          f" ({(sb['end_card']['end'] - sb['end_card']['start'] + 1) / sb['fps']:.2f} s), "
          f"{len(weld_iv)} arc intervals, {len(servo_iv)} servo intervals")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help=f"also compose the full {N_FRAMES}-frame sequence")
    a = ap.parse_args()
    sb = str(SB_PATH)
    check_storyboard(json.load(open(sb, encoding="utf-8")))

    def t(frame: int) -> float:                     # storyboard frame -> output time (centre of the frame)
        return (frame - 0.5) / FPS

    # 1) sparse preview: every 8th frame at 960x540 (as build.py --preview would give)
    sparse = OUT / "post_test_frames_sparse"
    gen_frames(sparse, range(1, N_FRAMES + 1, 8), 960, 540)
    t0 = time.time()
    mp4a = compose(str(sparse), str(OUT / "post_test_preview.mp4"), sb, audio=True, preview=True)
    print(f"preview compose: {time.time() - t0:.1f} s  {probe(Path(mp4a))}")
    grab(Path(mp4a), t(40), OUT / "post_check2_title.png")            # title fully in
    grab(Path(mp4a), t(215), OUT / "post_check2_caption_scan.png")    # АДАПТАЦИЯ (laser scan)
    grab(Path(mp4a), t(300), OUT / "post_check2_caption_A.png")       # ШОВ A caption
    grab(Path(mp4a), t(600), OUT / "post_check2_caption_B12.png")     # ШОВ B · СЕКТОРЫ 1–2
    grab(Path(mp4a), t(760), OUT / "post_check2_caption_180.png")     # ИНДЕКСАЦИЯ 180° (3 lines)
    grab(Path(mp4a), t(1050), OUT / "post_check2_endcard.png")        # end card, fully in

    # 2) full-rate excerpt: frames 930..N at 1920x1080 (ЗАВЕРШЕНИЕ caption -> end card -> hold)
    excerpt = OUT / "post_test_frames_excerpt"
    gen_frames(excerpt, range(930, N_FRAMES + 1))
    t0 = time.time()
    mp4b = compose(str(excerpt), str(OUT / "post_test_excerpt.mp4"), sb, audio=True, preview=False)
    print(f"excerpt compose: {time.time() - t0:.1f} s  {probe(Path(mp4b))}")
    grab(Path(mp4b), t(965) - t(930), OUT / "post_check2_excerpt_caption.png")   # ЗАВЕРШЕНИЕ caption
    grab(Path(mp4b), t(1005) - t(930), OUT / "post_check2_excerpt_fade.png")     # end card fading in
    grab(Path(mp4b), t(1100) - t(930), OUT / "post_check2_excerpt_end.png")      # end card, last second

    # 3) --no-audio path
    compose(str(excerpt), str(OUT / "post_test_excerpt_noaudio.mp4"), sb, audio=False, preview=True)
    print("no-audio:", probe(OUT / "post_test_excerpt_noaudio.mp4"))

    if a.full:
        full = OUT / "post_test_frames"
        gen_frames(full, range(1, N_FRAMES + 1))
        t0 = time.time()
        mp4c = compose(str(full), str(OUT / "post_test_full.mp4"), sb, audio=True, preview=False)
        print(f"full compose: {time.time() - t0:.1f} s  {probe(Path(mp4c))}")
        grab(Path(mp4c), t(215), OUT / "post_check2_full_adapt.png")   # АДАПТАЦИЯ caption


if __name__ == "__main__":
    main()
