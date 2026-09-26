"""Make contact sheets (grids of frames with frame numbers) from a frame folder.  usage: python3 tests/contact_sheet.py out/preview_frames out/sheets [cols] [rows]"""
import sys, os, glob, re
from PIL import Image, ImageDraw, ImageFont
src, dst = sys.argv[1], sys.argv[2]
cols = int(sys.argv[3]) if len(sys.argv) > 3 else 4
rows = int(sys.argv[4]) if len(sys.argv) > 4 else 3
os.makedirs(dst, exist_ok=True)
files = sorted(glob.glob(os.path.join(src, "frame_*.png")))
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
per = cols * rows
for si in range(0, len(files), per):
    chunk = files[si:si + per]
    im0 = Image.open(chunk[0]); w, h = im0.size
    sheet = Image.new("RGB", (cols * w, rows * h), (20, 20, 20))
    for i, f in enumerate(chunk):
        im = Image.open(f).convert("RGB")
        n = re.search(r"frame_(\d+)", f).group(1)
        d = ImageDraw.Draw(im); d.rectangle((0, 0, 120, 30), fill=(0, 0, 0)); d.text((6, 3), f"f{int(n)}", font=font, fill=(255, 220, 0))
        sheet.paste(im, ((i % cols) * w, (i // cols) * h))
    out = os.path.join(dst, f"sheet_{si // per:02d}.jpg")
    sheet.save(out, quality=88)
    print(out)
