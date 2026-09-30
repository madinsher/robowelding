"""Check the texts and logos drawn INTO the 3D scene for both editions (bpy, no rendering).

For (lang, brand) in (ru, pigrupp), (en, atomix): factory-empty scene, i18n2.set_lang / branding2.set_brand, then every
module that draws text into a texture is built standalone (stage-1 spool paint markings, handler robot, tack robot,
fit-up station, kit cassette, conveyor, QC arch + laser-mark decal, storage rack, environment2) while PIL's
ImageDraw.text / multiline_text, branding2.paste_logo and i18n2.tr / tr_lines are recorded:

  * en: no Cyrillic in any drawn string; every i18n2 key the scene asks for (EN, EN_LINES and the Latin-script
    EN_LATIN ones) is drawn as its English entry and no EN_LATIN key is drawn (QC arch labels: "LASER CLASS 4")
  * ru: every Cyrillic string drawn went through i18n2 (so the English edition cannot miss it) and is drawn unchanged,
    the EN_LATIN keys too (QC arch labels: "LASER KL.4" as before)
  * both: every i18n2 entry is used by the scene; every text lies inside its OWN box (>= 3 px margin): the texture
    functions are wrapped and hand their box layout to the texts they draw - C._label_mat cells inside the 4 px rim,
    marking_qc.display_text_boxes / mark_text_box, environment2.sign_text_boxes / hmi_text_boxes,
    storage.sign_text_box - and a text drawn outside a known layout fails (stage-1 marker tiles: inside the image);
    no two texts of one image overlap, no text overlaps a logo plate; the brand's logo textures
    / materials exist and every edition-dependent one carries the brand key (none of the other brand); the QC screen,
    zone HMI and storage-sign textures show the brand colour and the white plate inside the pasted logo boxes (right
    end of the footer bars / left end of the sign); objects use those materials; the cabinet logo plate is 0.30-0.36 m
    wide, faces -Y, 1-2 mm proud of the door face (a backing, if any, touches the door), its material uses the
    texture alpha (rounded corners), inside the door and clear of the door fittings.

The texture atlases / screens / logo plates of both languages go to out/tests/i18n2/<lang>_<name>.png (for a look).

    cd /home/user/robowelding && python3 stage2_logistics/tests/t_i18n2.py
"""
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE2 = os.path.dirname(HERE)
sys.path.insert(0, STAGE2)
import tools  # noqa: E402,F401  (demo_video on sys.path)
import bpy  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
import i18n2  # noqa: E402
import branding2  # noqa: E402

OUT = os.path.join(STAGE2, "out", "tests", "i18n2")
EDITIONS = (("ru", "pigrupp"), ("en", "atomix"))
FAIL = []


def check(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg, flush=True)
    if not ok:
        FAIL.append(msg)
    return ok


# ============================================================================ recorders
class Rec:
    stage = ""
    texts = []          # dict(stage, text, bbox, im, size, src, boxes)
    logos = []          # dict(stage, box, im, size)
    keys = []           # Russian keys asked from i18n2.tr
    line_keys = []      # tuples asked from i18n2.tr_lines
    layout = []         # stack of (texture function, [boxes (x0, y0, x1, y1) px]) of the texture being drawn

    @classmethod
    def reset(cls):
        cls.texts, cls.logos, cls.keys, cls.line_keys, cls.layout = [], [], [], [], []


_text0 = ImageDraw.ImageDraw.text
_multi0 = ImageDraw.ImageDraw.multiline_text
_paste0 = branding2.paste_logo
_tr0 = i18n2.tr
_tr_lines0 = i18n2.tr_lines


def _record(draw, xy, text, args, kw):
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    text = str(text)
    font = kw.get("font", args[1] if len(args) > 1 else None)
    anchor = kw.get("anchor", args[2] if len(args) > 2 else None)
    bbox = None
    try:
        bbox = draw.textbbox(xy, text, font=font, anchor=anchor, stroke_width=kw.get("stroke_width", 0))
    except Exception:  # noqa: BLE001  (bitmap fonts etc.: no box, the language checks still apply)
        pass
    src, boxes = Rec.layout[-1] if Rec.layout else (None, None)
    # keep the core image alive: its id() identifies the image and must not be recycled
    Rec.texts.append(dict(stage=Rec.stage, text=text, xy=tuple(xy), bbox=bbox, im=draw.im, size=draw.im.size, src=src,
                          boxes=boxes))


def _text(self, xy, text, *args, **kw):
    _record(self, xy, text, args, kw)
    return _text0(self, xy, text, *args, **kw)


def _multi(self, xy, text, *args, **kw):
    _record(self, xy, text, args, kw)
    return _multi0(self, xy, text, *args, **kw)


def _paste(img, box, *args, **kw):
    size = _paste0(img, box, *args, **kw)
    Rec.logos.append(dict(stage=Rec.stage, box=tuple(float(v) for v in box), pasted=size, im=img.im, size=img.size))
    return size


def _tr(text):
    Rec.keys.append(text)
    return _tr0(text)


def _tr_lines(lines):
    Rec.line_keys.append(tuple(lines))
    return _tr_lines0(lines)


ImageDraw.ImageDraw.text = _text
ImageDraw.ImageDraw.multiline_text = _multi
branding2.paste_logo = _paste
i18n2.tr = _tr
i18n2.tr_lines = _tr_lines


# ---- box layouts: every texture function that draws text hands its boxes to the texts it draws (Rec.layout)
import cassette as C  # noqa: E402
import marking_qc as QC  # noqa: E402
import environment2 as E2  # noqa: E402
import storage as ST  # noqa: E402
import inspect  # noqa: E402

LABEL_RIM = 9           # C._label_mat: 4 px black rim 5 px inside every cell edge -> text area 9 px inside


def _label_boxes(key, texts, *a, **kw):
    ba = inspect.signature(_label_mat0).bind(key, texts, *a, **kw)
    ba.apply_defaults()
    W, H = ba.arguments["cell"]
    m = LABEL_RIM
    return [(m, k * H + m, W - m, (k + 1) * H - m) for k in range(len(texts))]


def _laid_out(mod, fname, boxes_fn):
    f0 = getattr(mod, fname)

    def f(*a, **kw):
        Rec.layout.append((f"{mod.__name__}.{fname}", boxes_fn(*a, **kw)))
        try:
            return f0(*a, **kw)
        finally:
            Rec.layout.pop()
    setattr(mod, fname, f)
    return f0


_label_mat0 = _laid_out(C, "_label_mat", _label_boxes)
_laid_out(QC, "_display_image", lambda: [b for k in range(len(QC.DISPLAY_STATES)) for b in QC.display_text_boxes(k)])
_laid_out(QC, "_mark_image", lambda: [QC.mark_text_box()])
_laid_out(E2, "_sign_atlas_mat", lambda: [b for k in range(len(E2.SIGNS)) for b in E2.sign_text_boxes(k)])
_laid_out(E2, "_hmi_mat", lambda: E2.hmi_text_boxes())
_laid_out(ST, "_sign_mat", lambda: [ST.sign_text_box()])
BOX_MARGIN = 3          # px between a text's ink box and the edge of its own box


# ============================================================================ scene
def build_modules():
    """The text-drawing modules standalone, in build2.build_scene's order; returns {module: build() result}."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    import fonts2
    from cell import spool
    import handler_robot, tack_robot, assembly_station, cassette, conveyor, marking_qc, storage, environment2  # noqa: E401
    res = {}
    steps = [
        ("spool", lambda: (fonts2.patch_stage1(), spool.build(name="spool"))[1]),
        ("handler_robot", handler_robot.build),
        ("tack_robot", tack_robot.build),
        ("assembly_station", assembly_station.build),
        ("cassette", cassette.build),
        ("conveyor", lambda: conveyor.build(n_carriers=1)),
        ("marking_qc", marking_qc.build),
        ("marking_qc.mark", lambda: marking_qc.build_mark(bpy.data.objects.new("t_i18n2_pipe_root", None))),
        ("storage", storage.build),
        ("environment2", environment2.build),
    ]
    for name, fn in steps:
        Rec.stage = name
        t0 = time.time()
        try:
            res[name] = fn()
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            check(False, f"{name}: build raised {type(e).__name__}: {e}")
        print(f"    built {name} ({time.time() - t0:.1f}s)", flush=True)
    Rec.stage = ""
    bpy.context.view_layer.update()
    return res


def image_rgba(bi):
    """bpy image -> uint8 RGBA array, rows top-down (PIL orientation)."""
    w, h = bi.size
    a = np.empty(w * h * 4, np.float32)
    bi.pixels.foreach_get(a)
    return (np.clip(a.reshape(h, w, 4)[::-1], 0.0, 1.0) * 255 + 0.5).astype(np.uint8)


def save_images(lang):
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):                 # no stale images of renamed textures
        if f.startswith(f"{lang}_") and f.endswith(".png"):
            os.remove(os.path.join(OUT, f))
    n = 0
    for bi in bpy.data.images:
        if bi.size[0] * bi.size[1] == 0:
            continue
        name = bi.name.replace("/", "_")
        Image.fromarray(image_rgba(bi), "RGBA").save(os.path.join(OUT, f"{lang}_{name}.png"))
        n += 1
    print(f"    saved {n} images to {os.path.relpath(OUT, STAGE2)}/{lang}_*.png", flush=True)


def _overlap(a, b, min_area=4.0):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w > 0 and h > 0 and w * h >= min_area


# ============================================================================ checks
def check_language(lang):
    drawn = [r["text"] for r in Rec.texts]
    drawn_set = set(drawn)
    cyr = sorted({t for t in drawn if i18n2.has_cyrillic(t)})
    keys = {k for k in Rec.keys if i18n2.has_cyrillic(k)}
    latin = {k for k in Rec.keys if k in i18n2.EN_LATIN}
    line_keys = {k for k in Rec.line_keys if any(i18n2.has_cyrillic(t) for t in k)}
    print(f"    {len(drawn)} strings drawn ({len(drawn_set)} distinct), {len(keys)} i18n2 keys, {len(latin)} Latin keys, "
          f"{len(line_keys)} multi-line keys")
    laser = {r["text"] for r in Rec.texts if r["src"] == "cassette._label_mat" and r["stage"] == "marking_qc" and "LASER" in r["text"]}
    if lang == "en":
        check(not cyr, f"en: no Cyrillic drawn ({cyr[:6]})")
        miss = [k for k in keys if i18n2.EN[k] not in drawn_set] + [k for k in latin if i18n2.EN_LATIN[k] not in drawn_set]
        miss += [f"{k} -> {t}" for k in line_keys for t in i18n2.EN_LINES[k] if t and t not in drawn_set]
        check(not miss, f"en: every key drawn as its English entry ({miss[:6]})")
        left = [k for k in i18n2.EN_LATIN if k in drawn_set]
        check(not left, f"en: no non-English Latin text (i18n2.EN_LATIN keys) drawn ({left})")
        check(laser == {"LASER CLASS 4", "LASER 50W"}, f"en: QC arch label atlas reads {sorted(laser)}")
    else:
        via = keys | {t for k in line_keys for t in k if t}
        rogue = [t for t in cyr if t not in via]
        check(not rogue, f"{lang}: every Cyrillic string drawn went through i18n2.tr / tr_lines ({rogue[:6]})")
        miss = [t for t in via | latin if t not in drawn_set]
        check(not miss, f"{lang}: every Russian key drawn unchanged ({miss[:6]})")
        check(laser == {"LASER KL.4", "LASER 50W"}, f"{lang}: QC arch label atlas reads {sorted(laser)} (as before)")
    unused = ([k for k in i18n2.EN if k not in keys] + [k for k in i18n2.EN_LATIN if k not in latin]
              + [" / ".join(k) for k in i18n2.EN_LINES if k not in line_keys])
    check(not unused, f"{lang}: every i18n2.EN / EN_LATIN / EN_LINES entry is used by the scene ({unused[:6]})")


def check_layout(lang):
    by_im = {}
    for r in Rec.texts:
        by_im.setdefault(id(r["im"]), []).append(r)
    logos_by_im = {}
    for g in Rec.logos:
        logos_by_im.setdefault(id(g["im"]), []).append(g)
    out, loose, clash, on_logo, tight = [], [], [], [], {}
    m = BOX_MARGIN
    for k, rs in by_im.items():
        W, H = rs[0]["size"]
        boxed = [r for r in rs if r["bbox"] is not None and r["text"].strip()]
        for r in boxed:
            x0, y0, x1, y1 = r["bbox"]
            where = f"{r['src'] or r['stage']}:{r['text']!r} {tuple(round(v) for v in r['bbox'])}"
            if r["stage"] == "spool":                      # stage-1 marker glyph tiles (read-only): glyph + jitter fill
                if x0 < 0 or x1 > W or y0 < 0 or y1 > H:   # the tile, no box layout of their own
                    out.append(f"{where} in the {W}x{H} image")
            elif r["boxes"] is None:
                loose.append(where)
            else:
                # the box holding the text's centre is its own; the text must keep BOX_MARGIN px from its edges
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                own = [b for b in r["boxes"] if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]]
                gap = min((min(x0 - b[0], b[2] - x1, y0 - b[1], b[3] - y1) for b in own), default=None)
                if gap is None or gap < m:
                    out.append(f"{where} {'in no box' if gap is None else f'{gap:.0f} px from its box edge'}")
                if gap is not None:
                    tight[r["src"]] = min(tight.get(r["src"], 1e9), gap)
            for g in logos_by_im.get(k, []):
                if _overlap(r["bbox"], g["box"], 1.0):
                    on_logo.append(f"{r['stage']}:{r['text']!r}")
        for i in range(len(boxed)):
            for j in range(i + 1, len(boxed)):
                a, b = boxed[i], boxed[j]
                same = a["text"] == b["text"] and abs(a["xy"][0] - b["xy"][0]) <= 3 and abs(a["xy"][1] - b["xy"][1]) <= 3
                if not same and _overlap(a["bbox"], b["bbox"]):
                    clash.append(f"{a['stage']}:{a['text']!r} x {b['text']!r}")
    print("    tightest text-to-box-edge gap: " + ", ".join(f"{s} {v:.0f} px" for s, v in sorted(tight.items())))
    check(not loose, f"{lang}: every text drawn by a texture function with a box layout ({loose[:4]})")
    check(not out, f"{lang}: every text inside its own box (>= {m} px margin) ({out[:4]})")
    check(not clash, f"{lang}: no two texts of one image overlap ({clash[:4]})")
    check(not on_logo, f"{lang}: no text on a logo plate ({on_logo[:4]})")


def _brand_pixels(arr, box, col, tol=48.0):
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    px = arr[y0:y1, x0:x1, :3].reshape(-1, 3).astype(np.float32)
    near = np.linalg.norm(px - np.asarray(col, np.float32), axis=1) < tol
    white = np.all(px > 235, axis=1)
    return float(near.mean()), float(white.mean())


def check_brand(lang, brand, res):
    other = [b for b in branding2.BRANDS if b != brand]
    col = branding2.BRANDS[brand]["color"]
    imgs = {bi.name for bi in bpy.data.images}
    mats = {m.name for m in bpy.data.materials}
    exp_img = [f"qc_display_img_{lang}_{brand}", f"env2_hmi_screen_{lang}_{brand}", f"sto_sign_{lang}_{brand}",
               f"logo_{brand}_plate_img"]
    exp_mat = [f"qc_display_{lang}_{brand}", f"env2_hmi_screen_{lang}_{brand}", f"sto_sign_{lang}_{brand}",
               f"logo_{brand}_plate", f"env2_signs_{lang}", f"env2_label_cab_{lang}", f"qc_label_{lang}"]
    check(all(n in imgs for n in exp_img), f"{lang}: brand images {exp_img} ({[n for n in exp_img if n not in imgs]} missing)")
    check(all(n in mats for n in exp_mat), f"{lang}: edition materials {exp_mat} ({[n for n in exp_mat if n not in mats]} missing)")
    stray = [n for n in imgs | mats if any(o in n for o in other)]
    check(not stray, f"{lang}: nothing of the other brand {other} ({stray[:4]})")
    # objects wear the edition's materials
    use = {"qc_display": f"qc_display_{lang}_{brand}", "env2_hmi_screen": f"env2_hmi_screen_{lang}_{brand}",
           "sto_sign_in": f"sto_sign_{lang}_{brand}", "sto_sign_out": f"sto_sign_{lang}_{brand}",
           "env2_cab_logo": f"logo_{brand}_plate", "env2_cab_label": f"env2_label_cab_{lang}", "env2_sign0": f"env2_signs_{lang}",
           "qc_label_laser": f"qc_label_{lang}", "qc_hmi_label": f"qc_label_{lang}", "qc_mk_label": f"qc_label_{lang}"}
    wrong = []
    for on, mn in use.items():
        ob = bpy.data.objects.get(on)
        if ob is None or not ob.data.materials or ob.data.materials[0].name != mn:
            wrong.append(f"{on}: {ob.data.materials[0].name if ob is not None and ob.data.materials else None}")
    check(not wrong, f"{lang}: objects use the edition materials ({wrong})")
    # the logo boxes of the three textures: position and pixels (brand colour + white plate)
    specs = [(f"qc_display_img_{lang}_{brand}", "marking_qc", [QC.display_logo_box(k) for k in range(len(QC.DISPLAY_STATES))]),
             (f"env2_hmi_screen_{lang}_{brand}", "environment2", [E2.hmi_logo_box()]),
             (f"sto_sign_{lang}_{brand}", "storage", [ST.sign_logo_box()])]
    for name, stage, boxes in specs:
        bi = bpy.data.images.get(name)
        if bi is None:
            continue
        arr = image_rgba(bi)
        pasted = sorted(tuple(round(v, 3) for v in g["box"]) for g in Rec.logos if g["stage"] == stage and tuple(g["size"]) == tuple(bi.size))
        check(pasted == sorted(tuple(round(v, 3) for v in b) for b in boxes), f"{lang}: {name}: logo pasted in {len(boxes)} box(es) {pasted[:1]}")
        fr = [_brand_pixels(arr, b, col) for b in boxes]
        check(all(n > 0.04 and w > 0.30 for n, w in fr),
              f"{lang}: {name}: brand colour {col} + white plate in every logo box "
              f"(brand {min(n for n, _ in fr):.2f}, white {min(w for _, w in fr):.2f})")
    bi = bpy.data.images.get(f"logo_{brand}_plate_img")
    if bi is not None:
        n, w = _brand_pixels(image_rgba(bi), (0, 0, bi.size[0], bi.size[1]), col)
        check(n > 0.08 and w > 0.40, f"{lang}: logo plate texture: brand colour {n:.2f}, white {w:.2f}")
    # positions in the textures: QC / HMI at the right end of the footer bar, storage sign at the left end
    qb = QC.display_logo_box(0)
    Wq, Hq = QC.DISPLAY_CELL
    check(qb[2] >= Wq - 12 and qb[1] >= Hq - QC.DISPLAY_FOOT and qb[3] <= Hq, f"QC logo box {tuple(round(v) for v in qb)} in the footer's right end")
    hb = E2.hmi_logo_box()
    check(hb[2] >= E2.HMI_PX[0] - 14 and hb[1] >= E2.HMI_PX[1] - E2.HMI_FOOT and hb[3] <= E2.HMI_PX[1],
          f"HMI logo box {tuple(round(v) for v in hb)} in the footer's right end")
    sb = ST.sign_logo_box()
    check(sb[0] <= 30 and sb[1] >= 10 and sb[3] <= ST.SIGN_PX[1] - 10, f"storage sign logo box {tuple(round(v) for v in sb)} at the left end")
    check_cabinet_plate(lang, brand, res)


def _wbox(ob):
    import mathutils
    pts = [ob.matrix_world @ mathutils.Vector(c) for c in ob.bound_box]
    return np.array([[p[k] for k in range(3)] for p in pts]).min(0), np.array([[p[k] for k in range(3)] for p in pts]).max(0)


def check_cabinet_plate(lang, brand, res):
    ob = bpy.data.objects.get("env2_cab_logo")
    door = bpy.data.objects.get("env2_cab_door")
    if not check(ob is not None and door is not None, f"{lang}: cabinet logo plate env2_cab_logo exists"):
        return
    env = res.get("environment2") or {}
    check(ob in env.get("objects", []) and all(c in env.get("objects", []) for c in ob.children)
          and ob.name in env["collection"].all_objects, f"{lang}: logo plate (+ backing) in environment2's objects / collection")
    lo, hi = _wbox(ob)
    dlo, dhi = _wbox(door)
    n = ob.matrix_world.to_3x3() @ __import__("mathutils").Vector((0, 0, 1))
    check(0.30 <= hi[0] - lo[0] <= 0.36, f"{lang}: plate width {hi[0] - lo[0]:.3f} m in [0.30, 0.36]")
    check(n.y < -0.999, f"{lang}: plate faces -Y (out of the door) ({tuple(round(v, 3) for v in n)})")
    proud = dlo[1] - hi[1]
    check(0.001 - 1e-6 <= proud <= 0.002 + 1e-6, f"{lang}: plate {proud * 1000:.2f} mm proud of the door face")
    for c in ob.children:                          # optional backing: between plate and door, touching the door
        blo, bhi = _wbox(c)
        check(dlo[1] - 1e-4 <= bhi[1] <= dlo[1] + 1e-6 and blo[1] > lo[1],
              f"{lang}: backing {c.name} between plate and door, touching the door ({(dlo[1] - bhi[1]) * 1000:.2f} mm)")
    mat = ob.data.materials[0] if ob.data.materials else None
    alpha = mat is not None and any(lk.to_socket.name == "Alpha" and lk.from_socket.name == "Alpha" for lk in mat.node_tree.links)
    check(alpha, f"{lang}: plate material uses the texture alpha (rounded corners, not black)")
    m = 0.01
    check(lo[0] >= dlo[0] + m and hi[0] <= dhi[0] - m and lo[2] >= dlo[2] + m and hi[2] <= dhi[2] - m,
          f"{lang}: plate inside the door ({lo[0]:.3f}..{hi[0]:.3f} x {lo[2]:.3f}..{hi[2]:.3f})")
    fittings = [o for o in bpy.data.objects if o.name.startswith(("env2_cab_mainsw", "env2_cab_pilot", "env2_cab_label", "env2_cab_handle",
                                                                    "env2_cab_fan0", "env2_cab_hinge"))]
    hit = []
    for o in fittings:
        a, b = _wbox(o)
        if a[0] < hi[0] + m and b[0] > lo[0] - m and a[2] < hi[2] + m and b[2] > lo[2] - m:
            hit.append(o.name)
    check(not hit, f"{lang}: plate clear (1 cm) of the door fittings ({hit})")
    # still within the cabinet's obstacle box (collision data unchanged)
    cab = [o for o in E2.obstacles() if o["name"] == "env2_cabinet"][0]
    c, s = np.array(cab["center"]), np.array(cab["size"])
    check(np.all(lo >= c - s / 2) and np.all(hi <= c + s / 2), f"{lang}: plate inside obstacle env2_cabinet")


# ============================================================================ main
def main():
    t_all = time.time()
    for lang, brand in EDITIONS:
        print(f"== edition {lang} / {brand}", flush=True)
        Rec.reset()
        i18n2.set_lang(lang)
        branding2.set_brand(brand)
        res = build_modules()
        save_images(lang)
        check_language(lang)
        check_layout(lang)
        check_brand(lang, brand, res)
    i18n2.set_lang("ru")
    branding2.set_brand("pigrupp")
    print(f"[t_i18n2] {time.time() - t_all:.0f}s", flush=True)
    if FAIL:
        print(f"{len(FAIL)} failure(s):")
        for f in FAIL:
            print("  -", f)
    print("RESULT:", "OK" if not FAIL else f"FAIL ({len(FAIL)})", flush=True)
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
