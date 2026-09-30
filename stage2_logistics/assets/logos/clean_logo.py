"""Clean a flat 1-colour logo from a JPEG: ink coverage -> local-max normalisation (thin strokes keep full colour at
their centre) -> Lanczos upscale -> smooth threshold (crisp anti-aliased edge) -> RGBA in the brand colour."""
import sys
from PIL import Image, ImageFilter
import numpy as np


def clean(src, dst, scale, margin=0.04):
    a = np.asarray(Image.open(src).convert("RGB")).astype(np.float64)
    sat = a.max(2) - a.min(2)
    color = np.median(a[sat > 150], axis=0)
    bg = np.median(a[sat < 20], axis=0)
    ink = ((bg[1] - a[..., 1]) / (bg[1] - color[1]) + (bg[2] - a[..., 2]) / max(bg[2] - color[2], 1)) / 2
    ink = np.clip(ink, 0, 1)
    im = Image.fromarray((ink * 255).astype(np.uint8), "L")
    lmax = np.asarray(im.filter(ImageFilter.MaxFilter(7))).astype(np.float64) / 255
    ink = np.clip(ink / np.clip(lmax, 0.45, 1.0), 0, 1) * (lmax > 0.2)
    ys, xs = np.nonzero(ink > 0.25)
    h, w = ink.shape
    mx = int((xs.max() - xs.min()) * margin)
    x0, x1 = max(0, xs.min() - mx), min(w, xs.max() + mx + 1)
    y0, y1 = max(0, ys.min() - mx), min(h, ys.max() + mx + 1)
    im = Image.fromarray((ink[y0:y1, x0:x1] * 255).astype(np.uint8), "L")
    big = im.resize((im.width * scale, im.height * scale), Image.LANCZOS).filter(ImageFilter.GaussianBlur(scale * 0.35))
    b = np.asarray(big).astype(np.float64) / 255
    t = np.clip((b - 0.38) / 0.24, 0, 1)
    t = t * t * (3 - 2 * t)
    alpha = Image.fromarray((t * 255).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(0.8))
    rgba = Image.new("RGBA", alpha.size, tuple(int(round(c)) for c in color) + (255,))
    rgba.putalpha(alpha)
    rgba.save(dst, optimize=True)
    print(dst, rgba.size, "colour #%02X%02X%02X" % tuple(int(round(c)) for c in color))


if __name__ == "__main__":
    clean(sys.argv[1], sys.argv[2], int(sys.argv[3]))
