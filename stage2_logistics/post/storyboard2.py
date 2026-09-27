#!/usr/bin/env python3
"""Stage-2 storyboard generator: post/edl.json + the texts below -> post/storyboard2.json (OUTPUT frame space).

    python3 stage2_logistics/post/storyboard2.py [--edl post/edl.json] [--out post/storyboard2.json]

The result uses the stage-1 storyboard format (demo_video/post/storyboard.json: fps, frames, resolution, font,
font_bold, title, captions[], end_card, corner_label, footer, audio{...}) with every frame number in the OUTPUT
timeline of the stage-2 edit (1..edl["frames"]), plus:

    corner_ranges        output ranges of the stage-2 blocks (consecutive s2 segments); the corner label is drawn only
                         there - the stage-1 segments already carry the burned-in stage-1 label, title and captions
    s1_ranges            output ranges of the stage-1 segments (no overlays at all)
    edl_segments         [[src, f0, f1, out0, out1], ...] - compose2.py refuses a storyboard made for another EDL
    captions[i].key/.shot  caption key and the shot it belongs to
    end_card.flow        the 7 steps of the flow diagram;  end_card.kpis  the bullet lines (== end_card.lines)
    audio.layers         stage-2 sound layers with their intervals mapped to output frames (see audio2.py)

Timing rules (all in output frames):
  * title       frame 1 .. out1(first segment) - 10;
  * caption     start = out0 + 3 of its s2 segment; end = (last frame of the run of uncaptioned s2 segments that
                follow it, cut at the next s1 segment / before the end card) - 4;
  * end card    the last END_CARD_FRAMES frames of the last segment;
  * nothing over s1 segments.
All texts live in this file: edit them here and re-run the script (compose2.py reads the json).
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent             # stage2_logistics/post
S2_ROOT = HERE.parent                              # stage2_logistics
DEMO = S2_ROOT.parent / "demo_video"
for _p in (str(DEMO), str(S2_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import edl as edl_mod                              # noqa: E402  (stage2_logistics/edl.py, no bpy needed)

EDL_JSON = HERE / "edl.json"
SB2_JSON = HERE / "storyboard2.json"
S1_STORYBOARD = DEMO / "post" / "storyboard.json"

# ----------------------------------------------------------------------------- texts (Russian, exact)
TITLE = {
    "heading": "Участок сварки трубных спулов: полный цикл",
    "sub": "Комплектация · сборка и прихватка · загрузка · сварка · маркировка и контроль · складирование",
}

CAPTIONS: Dict[str, Dict[str, str]] = {       # caption key (edl segment "caption") -> tag, text
    "flange": {"tag": "КОМПЛЕКТАЦИЯ",
               "text": "Робот-погрузчик на линейном треке берёт детали из кассеты комплекта: фланец ложится на "
                       "стенд сборки, пневмоприжимы фиксируют его"},
    "elbow": {"tag": "СБОРКА",
              "text": "Отвод ставится на фланец по центрирующим упорам и V-призме — положение задаёт оснастка, "
                      "а не ручная разметка"},
    "pipe": {"tag": "СБОРКА · ЗАЗОР",
             "text": "Труба укладывается в ложемент до упора, датчик проверяет зазор стыка перед прихваткой"},
    "tack": {"tag": "ПРИХВАТКА",
             "text": "Малый сварочный робот ставит по три прихватки на каждый стык — сборка фиксируется без "
                     "ручного труда"},
    "carry": {"tag": "ПЕРЕНОС",
              "text": "Прихваченный спул забирается целиком: один комплект деталей — один спул"},
    "load": {"tag": "ЗАГРУЗКА ЯЧЕЙКИ",
             "text": "Сварочный робот в исходной позиции, световая завеса в режиме muting: погрузчик ставит спул "
                     "на планшайбу, прижимы фиксируют фланец"},
    "ready": {"tag": "ПЕРЕДАЧА В СВАРКУ",
              "text": "Погрузчик выходит за ограждение, позиционер разворачивает изделие в исходное положение "
                      "для сварки"},
    "done": {"tag": "СВАРКА ЗАВЕРШЕНА",
             "text": "Оба стыка сварены: сварочный робот уходит в исходную позицию, позиционер возвращает изделие "
                     "в позицию выгрузки"},
    "unload": {"tag": "ВЫГРУЗКА",
               "text": "Прижимы планшайбы освобождают фланец, погрузчик забирает спул из ячейки"},
    "carrier": {"tag": "ВЫХОДНОЙ РОЛЬГАНГ",
                "text": "Спул ставится на паллету-спутник и уходит по рольгангу на пост маркировки и контроля"},
    "mark": {"tag": "МАРКИРОВКА",
             "text": "Лазерный маркер наносит ID спула и номер WPS — прослеживаемость до каждого шва"},
    "scan": {"tag": "КОНТРОЛЬ",
             "text": "Лазерный профилометр сканирует шов, геометрия валика сравнивается с допусками; результат "
                     "сохраняется по ID спула"},
    "store": {"tag": "СКЛАДИРОВАНИЕ",
              "text": "Погрузчик укладывает готовый спул в стеллаж; на стенде сборки уже лежит фланец следующего "
                      "комплекта"},
}

END_CARD = {
    "heading": "Полный цикл участка сварки спулов",
    "flow": ["Кассета комплекта", "Сборка", "Прихватка", "Загрузка в ячейку", "Сварка",
             "Маркировка и контроль", "Стеллаж · AGV"],
    "kpis": [
        "3 робота: погрузчик на треке (класс 150 кг, вылет 3,2 м), прихваточный и сварочный на треке "
        "+ 2-осевой позиционер",
        "Клещевой захват с V-призмами под DN100–300, сменные губки",
        "Логистика идёт параллельно сварке: такт участка определяется временем сварки",
        "Каждый спул прослеживается по ID: режимы швов, результат контроля, место хранения",
    ],
    "cta": "Этап 2 · Логистика и комплектование · компоновка под данные заказчика",
}
FOOTER = "Демонстрационная 3D-симуляция · моделирование на основе фото участка"
CORNER_LABEL = "ДЕМО · СИМУЛЯЦИЯ · УСКОРЕНО"

# ----------------------------------------------------------------------------- timing (output frames)
TITLE_TAIL = 10            # title ends this many frames before the end of the first segment
CAPTION_IN = 3             # caption starts out0 + 3
CAPTION_OUT = 4            # caption ends 4 frames before its boundary frame
END_CARD_FRAMES = 100      # end card = last 100 frames of the last segment (~4.2 s)
MIN_CAPTION = 28           # frames
MIN_GAP = 4                # empty frames between two overlays of the title/caption/end-card track
FADE = 12                  # stage-1 fades: title/captions 12 frames, end card 14
FADE_END = 14

# ----------------------------------------------------------------------------- sound layers
# Scene-interval kind (edl["intervals"]) -> synthesized layer.  "db" = RMS inside the layer's own intervals before
# the final peak normalisation (stage 1: ambience -32, servo -27, arc -20).  New layers stay at or below the servo
# level; only the QC "OK" chime is allowed to be louder (~-24).  "pitch" transposes the stage-1 servo whine
# (audio.servo, 200-400 Hz) - lower for heavy axes, higher for small ones; "pan" -1 (left) .. +1 (right).
AUDIO_LAYERS = [
    {"name": "handler",     "type": "servo",     "kind": "handler",    "db": -28.0, "pitch": 0.80, "pan": -0.25},
    {"name": "tack_robot",  "type": "servo",     "kind": "tack",       "db": -30.0, "pitch": 1.45, "pan": 0.20},
    {"name": "positioner",  "type": "servo",     "kind": "positioner", "db": -29.0, "pitch": 0.70, "pan": 0.10},
    {"name": "qc_axis",     "type": "servo",     "kind": "qc_move",    "db": -33.0, "pitch": 1.80, "pan": -0.35},
    {"name": "tack_arc",    "type": "tack_arc",  "kind": "tack_arc",   "db": -27.0, "pan": 0.10},
    {"name": "clamp",       "type": "pneumatic", "kind": "clamp",      "db": -29.0, "pan": 0.15},
    {"name": "stud",        "type": "pneumatic", "kind": "stud",       "db": -29.0, "pan": 0.05},
    {"name": "gripper",     "type": "pneumatic", "kind": "gripper",    "db": -33.0, "pan": -0.20, "soft": True},
    {"name": "conveyor",    "type": "conveyor",  "kind": "conveyor",   "db": -30.0, "pan": -0.30},
    {"name": "marker",      "type": "marker",    "kind": "marker",     "db": -31.0, "pan": -0.30},
    {"name": "scan",        "type": "scan",      "kind": "scan",       "db": -32.0, "pan": -0.30},
    {"name": "agv",         "type": "agv",       "kind": "agv",        "db": -31.0, "pan": -0.40},
]
CHIME = {"event": "qc_ok", "db": -24.0, "pan": -0.30}
AUDIO_LEVELS = {"ambience_db": -32.0, "servo_db": -27.0, "weld_db": -20.0, "hum_db": -32.0, "peak_db": -3.0,
                "seed": 11}


# ----------------------------------------------------------------------------- helpers
def load_json(path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def s2_blocks(E: dict) -> List[List[int]]:
    """[[out0, out1], ...] of the runs of consecutive stage-2 segments (the corner-label ranges)."""
    out: List[List[int]] = []
    for s in E["segments"]:
        if s["src"] != "s2":
            continue
        if out and out[-1][1] + 1 == s["out0"]:
            out[-1][1] = s["out1"]
        else:
            out.append([s["out0"], s["out1"]])
    return out


def s1_ranges(E: dict) -> List[List[int]]:
    return [[s["out0"], s["out1"]] for s in E["segments"] if s["src"] == "s1"]


def map_with_flags(E: dict, intervals, src: str = "s2") -> List[List[int]]:
    """Map scene (src='s2') or stage-1 (src='s1') frame intervals to output frames with edl.map_intervals and tag
    each visible piece [out0, out1, real_start, real_end]: real_* = 1 when that edge is the real start/end of the
    source interval, 0 when the edit cuts the interval there (the audio places onset/stop sounds only at real edges
    and joins pieces that continue across a picture cut)."""
    to_out = edl_mod.scene_to_out if src == "s2" else edl_mod.s1_to_out
    out = []
    for a, b in intervals:
        for o0, o1 in edl_mod.map_intervals(E, [[a, b]], src):
            out.append([int(o0), int(o1), int(to_out(E, a) == o0), int(to_out(E, b) == o1)])
    return sorted(out)


def edl_signature(E: dict) -> List[list]:
    return [[s["src"], s["f0"], s["f1"], s["out0"], s["out1"]] for s in E["segments"]]


# ----------------------------------------------------------------------------- build
def plan_captions(E: dict, end_card_start: int) -> List[dict]:
    segs = E["segments"]
    caps = []
    for i, s in enumerate(segs):
        key = s.get("caption") if s["src"] == "s2" else None
        if not key:
            continue
        if key not in CAPTIONS:
            raise KeyError(f"segment {s.get('shot')}: caption key {key!r} has no text in storyboard2.CAPTIONS")
        j = i
        while j + 1 < len(segs) and segs[j + 1]["src"] == "s2" and not segs[j + 1].get("caption"):
            j += 1
        boundary = segs[j]["out1"]                         # last frame of the run (next: captioned / s1 / end)
        if end_card_start > s["out0"]:
            boundary = min(boundary, end_card_start - 1)   # last frame before the end card
        start, end = s["out0"] + CAPTION_IN, boundary - CAPTION_OUT
        caps.append({"start": start, "end": end, "tag": CAPTIONS[key]["tag"], "text": CAPTIONS[key]["text"],
                     "key": key, "shot": s.get("shot")})
    return caps


def plan_audio(E: dict, s1_sb: dict) -> dict:
    s1_audio = s1_sb.get("audio") or {}
    off = int(E.get("weld_offset", 0))
    s1_weld = s1_audio.get("weld_intervals") or []
    s1_servo = s1_audio.get("servo_intervals") or []
    # stage-1 layers inside the stage-1 segments ...
    weld = edl_mod.map_intervals(E, s1_weld, src="s1")
    servo = edl_mod.map_intervals(E, s1_servo, src="s1")
    # ... and the stage-1 choreography replayed in the stage-2 scene (scene frame = weld_offset + stage-1 frame),
    # visible in stage-2 shots (e.g. S2_10_done: the welding robot returns home = stage-1 servo move 931..1060)
    weld += edl_mod.map_intervals(E, [[a + off, b + off] for a, b in s1_weld], src="s2")
    servo += edl_mod.map_intervals(E, [[a + off, b + off] for a, b in s1_servo], src="s2")
    layers = []
    for spec in AUDIO_LAYERS:
        lay = dict(spec)
        lay["intervals"] = map_with_flags(E, E["intervals"].get(spec["kind"], []), src="s2")
        layers.append(lay)
    ev = E["events"].get(CHIME["event"])
    chime_frames = [o for o in [edl_mod.scene_to_out(E, ev) if ev is not None else None] if o is not None]
    audio = dict(AUDIO_LEVELS)
    audio.update({
        "weld_intervals": sorted([list(map(int, p)) for p in weld]),
        "servo_intervals": sorted([list(map(int, p)) for p in servo]),
        "layers": layers,
        "chime": dict(CHIME, frames=chime_frames),
        # stage-1 fallbacks (audio.plan_intervals) are not used: the explicit intervals above are always present
        "servo_tags": [], "servo_pad_s": 0.25, "weld_prefix": "ШОВ",
    })
    return audio


def build(E: dict, s1_sb: dict = None) -> dict:
    """Storyboard dict for the EDL (output frame space)."""
    s1_sb = s1_sb or load_json(S1_STORYBOARD)
    segs = E["segments"]
    if segs[0]["src"] != "s2" or segs[-1]["src"] != "s2":
        raise ValueError("the edit must start and end with a stage-2 segment (title / end card go there)")
    frames = int(E["frames"])
    last = segs[-1]
    ec_start = max(last["out0"], last["out1"] - END_CARD_FRAMES + 1)
    title = dict(TITLE, start=1, end=segs[0]["out1"] - TITLE_TAIL)
    end_card = {"start": ec_start, "end": last["out1"], "heading": END_CARD["heading"],
                "flow": list(END_CARD["flow"]), "kpis": list(END_CARD["kpis"]),
                "lines": list(END_CARD["kpis"]), "cta": END_CARD["cta"]}
    sb = {
        "fps": int(E.get("fps", 24)),
        "frames": frames,
        "resolution": list(s1_sb.get("resolution", [1920, 1080])),
        "font": s1_sb["font"],
        "font_bold": s1_sb["font_bold"],
        "title": title,
        "captions": plan_captions(E, ec_start),
        "end_card": end_card,
        "corner_label": CORNER_LABEL,
        "footer": FOOTER,
        "corner_ranges": s2_blocks(E),
        "s1_ranges": s1_ranges(E),
        "edl_segments": edl_signature(E),
        "audio": plan_audio(E, s1_sb),
    }
    problems = check(sb, E, s1_sb)
    if problems:
        raise ValueError("storyboard2 timing problems:\n  " + "\n  ".join(problems))
    return sb


# ----------------------------------------------------------------------------- checks
def overlay_spans(sb: dict) -> List[tuple]:
    """(name, start, end, fade) of the title/caption/end-card track in time order."""
    spans = [("title", sb["title"]["start"], sb["title"]["end"], FADE)]
    spans += [(c.get("key") or c["tag"], c["start"], c["end"], FADE) for c in sb["captions"]]
    spans += [("end_card", sb["end_card"]["start"], sb["end_card"]["end"], FADE_END)]
    return sorted(spans, key=lambda s: s[1])


def s1_burned_spans(E: dict, s1_sb: dict) -> List[tuple]:
    """Stage-1 title / captions / end card burned into the stage-1 video, mapped to output frames:
    (name, out0, out1, complete) - complete = False when the edit cuts the text in the middle."""
    spans = [("s1 title", s1_sb["title"]["start"], s1_sb["title"]["end"])]
    spans += [("s1 " + c["tag"], c["start"], c["end"]) for c in s1_sb.get("captions", [])]
    spans += [("s1 end card", s1_sb["end_card"]["start"], s1_sb["end_card"]["end"])]
    out = []
    for name, a, b in spans:
        for o0, o1 in edl_mod.map_intervals(E, [[a, b]], src="s1"):
            out.append((name, o0, o1, o1 - o0 == b - a))
    return out


def check(sb: dict, E: dict, s1_sb: dict = None) -> List[str]:
    """Return a list of problems (empty = ok): overlaps, gaps < MIN_GAP, too short, overlays over s1 segments,
    stage-1 burned-in texts cut by the edit or closer than MIN_GAP to a stage-2 overlay, corner ranges != s2
    blocks, audio intervals outside their segments."""
    P = []
    s1_sb = s1_sb or load_json(S1_STORYBOARD)
    frames = sb["frames"]
    if frames != E["frames"]:
        P.append(f"frames {frames} != edl frames {E['frames']}")
    blocks = s2_blocks(E)
    s1r = s1_ranges(E)
    spans = overlay_spans(sb)
    for name, a, b, fade in spans:
        if not 1 <= a <= b <= frames:
            P.append(f"{name}: {a}..{b} outside 1..{frames}")
        if b - a + 1 <= 2 * fade:
            P.append(f"{name}: {b - a + 1} frames is not longer than two {fade}-frame fades")
        if not any(o0 <= a and b <= o1 for o0, o1 in blocks):
            P.append(f"{name}: {a}..{b} is not inside one stage-2 block {blocks}")
    for c in sb["captions"]:
        if c["end"] - c["start"] + 1 < MIN_CAPTION:
            P.append(f"caption {c.get('key')}: {c['end'] - c['start'] + 1} frames < {MIN_CAPTION}")
    for (n1, a1, b1, _), (n2, a2, b2, _) in zip(spans, spans[1:]):
        if a2 - b1 - 1 < MIN_GAP:
            P.append(f"{n1} ({a1}..{b1}) and {n2} ({a2}..{b2}): gap {a2 - b1 - 1} < {MIN_GAP} frames")
    burned = s1_burned_spans(E, s1_sb)
    for name, o0, o1, full in burned:
        if not full:
            P.append(f"{name}: the edit cuts this burned-in stage-1 text (output {o0}..{o1})")
    track = sorted([(n, a, b) for n, a, b, _ in spans] + [(n, a, b) for n, a, b, _ in burned], key=lambda t: t[1])
    for (n1, a1, b1), (n2, a2, b2) in zip(track, track[1:]):
        if (n1.startswith("s1 ") or n2.startswith("s1 ")) and a2 - b1 - 1 < MIN_GAP:
            P.append(f"{n1} ({a1}..{b1}) and {n2} ({a2}..{b2}): gap {a2 - b1 - 1} < {MIN_GAP} frames")
    if sb.get("corner_ranges") != blocks:
        P.append(f"corner_ranges {sb.get('corner_ranges')} != stage-2 blocks {blocks}")
    if sb.get("edl_segments") != edl_signature(E):
        P.append("edl_segments does not match the EDL")
    au = sb.get("audio") or {}

    def inside(o0, o1, ranges):
        return any(r0 <= o0 and o1 <= r1 for r0, r1 in ranges)

    for key in ("weld_intervals", "servo_intervals"):
        for iv in au.get(key, []):
            if not (inside(iv[0], iv[1], s1r) or inside(iv[0], iv[1], blocks)):
                P.append(f"audio.{key} {iv} crosses a splice or lies outside 1..{frames}")
    for lay in au.get("layers", []):
        for iv in lay["intervals"]:
            if not inside(iv[0], iv[1], blocks):
                P.append(f"audio layer {lay['name']}: {iv[:2]} is not inside a stage-2 block")
    for f in (au.get("chime") or {}).get("frames", []):
        if not inside(f, f, blocks):
            P.append(f"chime at {f} not inside a stage-2 block")
    return P


def summary(sb: dict) -> str:
    fps = sb["fps"]
    lines = [f"storyboard2: {sb['frames']} frames = {sb['frames'] / fps:.2f} s, corner ranges {sb['corner_ranges']},"
             f" s1 ranges {sb['s1_ranges']}",
             f"  title      {sb['title']['start']:5d}-{sb['title']['end']:5d}"]
    for c in sb["captions"]:
        lines.append(f"  {c['key']:<10s} {c['start']:5d}-{c['end']:5d}  {(c['end'] - c['start'] + 1) / fps:4.1f} s"
                     f"  {c['tag']}")
    e = sb["end_card"]
    lines.append(f"  end card   {e['start']:5d}-{e['end']:5d}  {(e['end'] - e['start'] + 1) / fps:4.1f} s")
    au = sb["audio"]
    lines.append(f"  audio: {len(au['weld_intervals'])} arc + {len(au['servo_intervals'])} servo (stage 1), "
                 + ", ".join(f"{l['name']} {len(l['intervals'])}" for l in au["layers"])
                 + f", chime {au['chime']['frames']}")
    return "\n".join(lines)


def write(sb: dict, path=SB2_JSON) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(sb, fh, ensure_ascii=False, indent=1)
    return str(path)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Generate post/storyboard2.json from post/edl.json.")
    ap.add_argument("--edl", default=str(EDL_JSON))
    ap.add_argument("--out", default=str(SB2_JSON))
    a = ap.parse_args(argv)
    sb = build(load_json(a.edl))
    print(summary(sb))
    print("written", write(sb, a.out))


if __name__ == "__main__":
    main()
