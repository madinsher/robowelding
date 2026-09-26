"""Post-production smoke test: synthetic frames -> compose.py (preview + full-rate excerpt) ->
extracted check stills in out/post_check_*.png.

    python3 tests/t_post.py            # every 8th frame (preview) + full-rate excerpt 930..1008
    python3 tests/t_post.py --full     # also a full 1..1008 full-rate run (slower)
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from post.compose import compose, FFMPEG   # noqa: E402

OUT = ROOT / "out"
W, H = 1920, 1080


def make_frame(n: int, w: int = W, h: int = H) -> Image.Image:
    """Moving diagonal gradient + orbiting box + big frame number (so timing is verifiable)."""
    u = (n - 1) / 1007
    # gradient built with a small image and upscaled (fast)
    g = Image.new("RGB", (64, 36))
    px = g.load()
    for y in range(36):
        for x in range(64):
            v = (x / 63 + y / 35) / 2
            px[x, y] = (int(40 + 160 * ((v + u) % 1)), int(60 + 120 * v), int(120 + 100 * (1 - v)))
    img = g.resize((w, h), Image.BILINEAR)
    d = ImageDraw.Draw(img)
    import math
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="also compose the full 1008-frame sequence")
    a = ap.parse_args()
    sb = str(ROOT / "post" / "storyboard.json")

    # 1) sparse preview: every 8th frame at 960x540 (as build.py --preview would give)
    sparse = OUT / "post_test_frames_sparse"
    gen_frames(sparse, range(1, 1009, 8), 960, 540)
    t0 = time.time()
    mp4a = compose(str(sparse), str(OUT / "post_test_preview.mp4"), sb, audio=True, preview=True)
    print(f"preview compose: {time.time() - t0:.1f} s  {probe(Path(mp4a))}")
    grab(Path(mp4a), 1.5, OUT / "post_check_title.png")           # frame ~37: title fully in
    grab(Path(mp4a), 12.0, OUT / "post_check_caption_A.png")      # frame 289: ШОВ A caption
    grab(Path(mp4a), 30.0, OUT / "post_check_caption_180.png")    # frame 721: ИНДЕКСАЦИЯ 180°
    grab(Path(mp4a), 41.0, OUT / "post_check_endcard.png")        # frame 985: end card

    # 2) full-rate excerpt: frames 930..1008 at 1920x1080 (ЗАВЕРШЕНИЕ caption -> end card)
    excerpt = OUT / "post_test_frames_excerpt"
    gen_frames(excerpt, range(930, 1009))
    t0 = time.time()
    mp4b = compose(str(excerpt), str(OUT / "post_test_excerpt.mp4"), sb, audio=True, preview=False)
    print(f"excerpt compose: {time.time() - t0:.1f} s  {probe(Path(mp4b))}")
    grab(Path(mp4b), 0.5, OUT / "post_check_excerpt_caption.png")   # frame 942: ЗАВЕРШЕНИЕ caption
    grab(Path(mp4b), 1.45, OUT / "post_check_excerpt_fade.png")     # frame ~965: end card fading in
    grab(Path(mp4b), 2.5, OUT / "post_check_excerpt_end.png")       # frame 990: end card

    # 3) --no-audio path
    compose(str(excerpt), str(OUT / "post_test_excerpt_noaudio.mp4"), sb, audio=False, preview=True)
    print("no-audio:", probe(OUT / "post_test_excerpt_noaudio.mp4"))

    if a.full:
        full = OUT / "post_test_frames"
        gen_frames(full, range(1, 1009))
        t0 = time.time()
        mp4c = compose(str(full), str(OUT / "post_test_full.mp4"), sb, audio=True, preview=False)
        print(f"full compose: {time.time() - t0:.1f} s  {probe(Path(mp4c))}")
        grab(Path(mp4c), 8.0, OUT / "post_check_full_adapt.png")     # frame 193: АДАПТАЦИЯ caption


if __name__ == "__main__":
    main()
