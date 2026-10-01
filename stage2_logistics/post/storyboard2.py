#!/usr/bin/env python3
"""Stage-2 storyboard generator: post/edl_<ed>.json + the texts below -> post/storyboard2_<ed>.json (OUTPUT frames).

    python3 stage2_logistics/post/storyboard2.py [--edition ru|en|all]          (default all)
    python3 stage2_logistics/post/storyboard2.py --edl <edl.json> --out <storyboard.json>
    (python3 stage2_logistics/edl.py writes both EDLs and both storyboards)

The result uses the stage-1 storyboard format (demo_video/post/storyboard.json: fps, frames, resolution, font,
font_bold, title, captions[], end_card, corner_label, footer, audio{...}) with every frame number in the OUTPUT
timeline of the stage-2 edit (1..edl["frames"]), plus:

    edition, lang, stage1_mode   from the EDL (editions.py; an EDL without them: "ru", "ru", "video")
    brand                {key, name, color, logo}: branding2.BRANDS entry, logo path relative to stage2_logistics/
    logos                title / corner / end_card logo placement and timing (overlays2 draws them)
    corner_ranges        output ranges of the stage-2 blocks (consecutive s2 segments); the corner label is drawn only
                         there - the stage-1 video segments already carry the burned-in stage-1 label, title and
                         captions.  Render mode: one range over the whole video.
    s1_ranges            output ranges of the stage-1 video segments (no overlays but the corner logo)
    edl_segments         [[src, f0, f1, out0, out1], ...] - compose2.py refuses a storyboard made for another EDL
    captions[i].key/.shot  caption key and the shot it belongs to; stage-1 captions (render mode) also carry
                         "s1": [stage-1 start, end] and a key "s1_*"
    end_card.flow        the 7 steps of the timeline;  end_card.facts  [[value, label], ...] key facts
    font, font_bold, font_light   file names of the overlay fonts (assets/fonts: Fira Sans regular / medium / light)
    logos[*].opacity     the logo plates are slightly transparent
    audio.layers         stage-2 sound layers with their intervals mapped to output frames (see audio2.py)

Editions (EDL "stage1_mode"):
    video   (no edition now) the welding part is spliced from the stage-1 video: its title, captions and corner label
            are burned in, nothing is drawn over it but the corner logo, the corner label cuts hard at the splices.
    render  (ru, en) the welding part is rendered from the stage-2 scene with the stage-1 cameras: the stage-1 captions
            that the stage-1 video shows in the used frames are drawn in the edition's language at the same output
            frames (edl.map_stage1_intervals) with the stage-2 look and short texts (S1_CAPTIONS_RU / _EN); one
            corner label over the whole video.
    The soundtrack is the same in both editions (plan_audio).

Timing rules (all in output frames):
  * title       frame 1 .. out1(first segment) - 10;
  * caption     start = out0 + 3 of its s2 segment; end = (last frame of the run of uncaptioned s2 segments that
                follow it, cut at the next stage-1 part - s1 video segment or re-rendered stage-1 shot - / before the
                end card) - 4;
  * end card    the last END_CARD_FRAMES frames of the last segment;
  * logos       title logo = title timing; corner logo title end + 1 .. end card start - 1 (whole video, stage-1
                video segments included); end-card logo = part of the end card;
  * nothing over s1 video segments but the corner logo.
  * reading     every caption text <= READ_CPS characters per second it is fully visible (check() refuses more).
All texts live in this file (TEXTS: Russian, English): edit them here and re-run the script
(compose2.py reads the json).
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
import editions                                    # noqa: E402
import i18n2                                       # noqa: E402  (has_cyrillic only)

EDL_JSON = Path(editions.get()["edl_json"])        # default edition; editions.get(ed)["edl_json"] per edition
SB2_JSON = Path(editions.get()["storyboard_json"])
S1_STORYBOARD = DEMO / "post" / "storyboard.json"

# ----------------------------------------------------------------------------- texts (Russian)
# House style of the Russian texts: "е" instead of "ё", a hyphen instead of dashes (the customer's editing
# pass over the texts); keep it when a text is changed.
# A caption is read while the action runs: one short phrase per shot.  check() enforces READ_CPS (characters of the
# caption text per second it is fully visible), so a text that cannot be read within its shot is refused - shorten it.
TITLE = {
    "heading": "Сварка трубных спулов: полный цикл",
    "sub": "От комплекта деталей до склада",
}

CAPTIONS: Dict[str, Dict[str, str]] = {       # caption key (edl segment "caption") -> tag, text
    "flange": {"tag": "ПОДАЧА ДЕТАЛЕЙ", "text": "Робот ставит фланец из кассеты на стенд"},
    "elbow": {"tag": "СБОРКА", "text": "Отвод встает на фланец по упорам оснастки"},
    "pipe": {"tag": "КОНТРОЛЬ ЗАЗОРА", "text": "Труба уложена до упора, датчик проверяет зазор"},
    "tack": {"tag": "ПРИХВАТКА", "text": "Робот ставит по три прихватки на каждый стык"},
    "carry": {"tag": "ПЕРЕНОС", "text": "Погрузчик забирает собранный спул целиком"},
    "load": {"tag": "ЗАГРУЗКА В ЯЧЕЙКУ", "text": "Спул на планшайбе, прижимы держат фланец"},
    "ready": {"tag": "ГОТОВНОСТЬ", "text": "Ячейка свободна, начинается сварка"},
    "done": {"tag": "СВАРКА ЗАВЕРШЕНА", "text": "Оба стыка сварены, робот отходит"},
    "unload": {"tag": "ВЫГРУЗКА", "text": "Прижимы разжаты, погрузчик забирает спул"},
    "carrier": {"tag": "РОЛЬГАНГ", "text": "Спул едет на пост контроля"},
    "mark": {"tag": "МАРКИРОВКА", "text": "Лазер наносит номер спула"},
    "scan": {"tag": "КОНТРОЛЬ ШВА", "text": "Лазерный сканер сверяет форму шва с допусками"},
    "store": {"tag": "СКЛАД", "text": "Готовый спул уложен на стеллаж, на стенде уже следующий комплект"},
}

END_CARD = {
    "heading": "Полный цикл в одном потоке",
    "flow": ["Кассета комплекта", "Сборка", "Прихватка", "Загрузка в ячейку", "Сварка",
             "Маркировка и контроль", "Стеллаж · AGV"],
    "facts": [                                 # [value, label]: the value is read, the label explains it
        ["3 робота", "погрузчик, прихватка, сварка"],
        ["DN100-300", "один захват, сменные губки"],
        ["ID спула", "режимы шва, контроль, место на складе"],
    ],
}
FOOTER = "3D-симуляция по фотографиям участка"
CORNER_LABEL = "ДЕМО · СИМУЛЯЦИЯ · УСКОРЕНО"

# ----------------------------------------------------------------------------- texts (English)
# Same keys and structure as the Russian ones.  Vocabulary (the same word for the same machine in the scene texts,
# i18n2): pipe spool, loading robot, kit, fixture, tack welds, gap, welding robot, positioner, faceplate, clamps,
# conveyor, laser scanner, storage rack.
TITLE_EN = {
    "heading": "Pipe spool welding: the full cycle",
    "sub": "From a kit of parts to the storage rack",
}

CAPTIONS_EN: Dict[str, Dict[str, str]] = {
    "flange": {"tag": "PARTS FEED", "text": "The robot places the flange on the fixture"},
    "elbow": {"tag": "FIT-UP", "text": "The elbow seats on the flange against stops"},
    "pipe": {"tag": "GAP CHECK", "text": "Pipe laid against the stop, a sensor checks the gap"},
    "tack": {"tag": "TACK WELDING", "text": "The robot places three tack welds on each joint"},
    "carry": {"tag": "TRANSFER", "text": "The loading robot lifts the tacked spool"},
    "load": {"tag": "CELL LOADING", "text": "Spool on the faceplate, clamps hold it"},
    "ready": {"tag": "READY TO WELD", "text": "The cell is clear, welding starts"},
    "done": {"tag": "WELDING COMPLETE", "text": "Both welds done, robot retracts"},
    "unload": {"tag": "UNLOADING", "text": "Clamps release, the robot takes the spool"},
    "carrier": {"tag": "CONVEYOR", "text": "On its way to inspection"},
    "mark": {"tag": "MARKING", "text": "Laser marks the spool ID"},
    "scan": {"tag": "WELD INSPECTION", "text": "A laser scanner checks the bead against limits"},
    "store": {"tag": "STORAGE", "text": "The finished spool goes to the rack; the next kit is already on the fixture"},
}

END_CARD_EN = {
    "heading": "The full cycle in one flow",
    "flow": ["Kit cassette", "Fit-up", "Tack welding", "Cell loading", "Welding", "Marking and QC", "Rack · AGV"],
    "facts": [
        ["3 robots", "loading, tacking, welding"],
        ["DN100–300", "one gripper, interchangeable jaws"],
        ["Spool ID", "weld data, inspection, storage slot"],
    ],
}
FOOTER_EN = "3D simulation based on shop-floor photos"
CORNER_LABEL_EN = "DEMO · SIMULATION · SPED UP"

# Stage-1 captions (demo_video/post/storyboard.json, burned into the stage-1 video) for the render mode, keyed by the
# exact Russian tag of the stage-1 storyboard (note the Latin A / B and the en dash of the stage-1 tags).  The texts
# are short versions for the stage-2 look (the stage-1 video shows longer ones in its own caption boxes); the edit
# shows the 6 inside the stage-1 frames it uses.  "key" names the caption in the storyboard.
S1_CAPTIONS_RU: Dict[str, Dict[str, str]] = {
    "ПОДГОТОВКА": {"key": "s1_setup", "tag": "ПОДГОТОВКА",
                   "text": "Позиционер выставляет стык A, робот подъезжает"},
    "АДАПТАЦИЯ": {"key": "s1_tracking", "tag": "ПОИСК СТЫКА", "text": "Лазер сканирует стык"},
    "ШОВ A · ВРАЩЕНИЕ ИЗДЕЛИЯ": {"key": "s1_weld_a", "tag": "ШОВ A · ВРАЩЕНИЕ ИЗДЕЛИЯ",
                                "text": "Фланец с отводом: изделие вращается, горелка стоит в верхней точке"},
    "ПЕРЕМЕЩЕНИЕ ПО ТРЕКУ": {"key": "s1_track", "tag": "ПЕРЕЕЗД ПО ТРЕКУ",
                             "text": "Робот переезжает по треку к стыку B"},
    "ШОВ B · СЕКТОРЫ 1–2": {"key": "s1_weld_b12", "tag": "ШОВ B · СЕКТОРЫ 1-2",
                           "text": "Отвод с трубой: горелку ведет робот, два сектора по 90°"},
    "ИНДЕКСАЦИЯ 180°": {"key": "s1_index", "tag": "ПОВОРОТ НА 180°",
                        "text": "Позиционер переворачивает изделие, робот переезжает"},
    "ШОВ B · СЕКТОРЫ 3–4": {"key": "s1_weld_b34", "tag": "ШОВ B · СЕКТОРЫ 3-4",
                           "text": "Вторая половина шва B: снова сверху, без потолочной сварки"},
    "ЗАВЕРШЕНИЕ": {"key": "s1_done", "tag": "ЦИКЛ ЗАВЕРШЕН",
                   "text": "Оба стыка сварены, режимы записаны по номеру спула"},
}
S1_CAPTIONS_EN: Dict[str, Dict[str, str]] = {
    "ПОДГОТОВКА": {"key": "s1_setup", "tag": "PREPARATION",
                   "text": "Positioner presents joint A, the robot moves in"},
    "АДАПТАЦИЯ": {"key": "s1_tracking", "tag": "SEAM FINDING", "text": "Laser finds the seam"},
    "ШОВ A · ВРАЩЕНИЕ ИЗДЕЛИЯ": {"key": "s1_weld_a", "tag": "WELD A · PART ROTATES",
                                "text": "Flange to elbow: the part rotates, the torch stays on top"},
    "ПЕРЕМЕЩЕНИЕ ПО ТРЕКУ": {"key": "s1_track", "tag": "TRACK TRAVEL",
                             "text": "The robot travels along the track to joint B"},
    "ШОВ B · СЕКТОРЫ 1–2": {"key": "s1_weld_b12", "tag": "WELD B · SECTORS 1–2",
                           "text": "Elbow to pipe: the robot moves the torch, two 90° sectors"},
    "ИНДЕКСАЦИЯ 180°": {"key": "s1_index", "tag": "180° TURN",
                        "text": "The positioner flips the part, the robot relocates"},
    "ШОВ B · СЕКТОРЫ 3–4": {"key": "s1_weld_b34", "tag": "WELD B · SECTORS 3–4",
                           "text": "Second half of weld B: on top again, never overhead"},
    "ЗАВЕРШЕНИЕ": {"key": "s1_done", "tag": "CYCLE COMPLETE",
                   "text": "Both joints welded, parameters saved to the spool ID"},
}

TEXTS = {
    "ru": dict(title=TITLE, captions=CAPTIONS, end_card=END_CARD, footer=FOOTER, corner_label=CORNER_LABEL,
               s1_captions=S1_CAPTIONS_RU),
    "en": dict(title=TITLE_EN, captions=CAPTIONS_EN, end_card=END_CARD_EN, footer=FOOTER_EN,
               corner_label=CORNER_LABEL_EN, s1_captions=S1_CAPTIONS_EN),
}

# fonts of the overlays: bare file names, resolved in stage2_logistics/assets/fonts (Fira Sans, SIL OFL)
FONTS = {"font": "FiraSans-Regular.ttf", "font_bold": "FiraSans-Medium.ttf", "font_light": "FiraSans-Light.ttf"}

# ----------------------------------------------------------------------------- timing (output frames)
TITLE_TAIL = 10            # title ends this many frames before the end of the first segment
CAPTION_IN = 3             # caption starts out0 + 3
CAPTION_OUT = 4            # caption ends 4 frames before its boundary frame
END_CARD_FRAMES = 110      # end card = last 110 frames of the last segment (~4.6 s)
READ_CPS = 15.0            # reading speed limit: characters of a text per second it is fully visible
MIN_CAPTION = 28           # frames
MIN_GAP = 4                # empty frames between two overlays of the title/caption/end-card track
FADE = 12                  # stage-1 fades: title/captions 12 frames, end card 14
FADE_END = 14

# logos of sb["brand"] (overlays2 draws them from branding2): the logo on a white rounded plate, height in px.
#   title     large, top left (the title text is the bottom-third bar), title timing and fades
#   corner    small, top left (the stage-1 corner label is top RIGHT, same y), title end + 1 .. end card start - 1,
#             drawn over the whole video (stage-1 video segments included), fades only at its own start / end
#   end_card  top left above the heading, part of the end-card image (overlays2.render_end_card2)
# "opacity": the plate is slightly transparent, the footage shows through it.
LOGO_TITLE = {"x": 72, "y": 64, "height": 180, "opacity": 0.88}
LOGO_CORNER = {"x": 80, "y": 36, "height": 77, "opacity": 0.88}
LOGO_END_CARD = {"x": 76, "y": 60, "height": 144, "opacity": 0.92}

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


def edition_of(E: dict) -> dict:
    """Edition fields of an EDL (an EDL written before the editions: Russian, stage-1 video spliced)."""
    ed = E.get("edition") or editions.DEFAULT
    mode = E.get("stage1_mode") or "video"
    if mode not in ("video", "render"):
        raise ValueError(f"EDL stage1_mode {mode!r}: expected 'video' or 'render'")
    info = editions.get(ed)
    return {"edition": ed, "lang": E.get("lang") or info["lang"], "brand": E.get("brand") or info["brand"],
            "stage1_mode": mode}


def texts(lang: str) -> dict:
    if lang not in TEXTS:
        raise ValueError(f"no storyboard texts for language {lang!r} (TEXTS: {list(TEXTS)})")
    return TEXTS[lang]


def brand_entry(key: str) -> dict:
    """sb["brand"]: branding2.BRANDS entry; the logo path is relative to stage2_logistics/ (posix, works on Windows)."""
    import branding2
    b = branding2.info(key)
    return {"key": key, "name": b["name"], "color": list(b["color"]),
            "logo": Path(b["logo"]).resolve().relative_to(S2_ROOT).as_posix()}


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
    """Output ranges of the stage-1 VIDEO segments (none in render mode)."""
    return [[s["out0"], s["out1"]] for s in E["segments"] if s["src"] == "s1"]


def stage1_parts(E: dict) -> List[List[int]]:
    """Output ranges showing stage-1 frames: runs of consecutive video segments (src "s1") or re-rendered stage-1
    shots (s2 segments with an "s1" range), merged across their cuts."""
    out: List[List[int]] = []
    for s in E["segments"]:
        if not (s["src"] == "s1" or s.get("s1")):
            continue
        if out and out[-1][1] + 1 == s["out0"]:
            out[-1][1] = s["out1"]
        else:
            out.append([s["out0"], s["out1"]])
    return out


def s2_only(E: dict) -> dict:
    """The EDL restricted to the stage-2 shots proper (no s1 video segments, no re-rendered stage-1 shots): the
    stage-2 sound layers and the stage-1 choreography seen from stage-2 shots are mapped through it, so both editions
    get the same soundtrack (the stage-1 part sounds the stage-1 layers only, like the spliced video)."""
    return dict(E, segments=[s for s in E["segments"] if s["src"] == "s2" and not s.get("s1")])


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


def shot_at(E: dict, o: int):
    for s in E["segments"]:
        if s["out0"] <= o <= s["out1"]:
            return s.get("shot") or "stage-1 video"
    return None


# ----------------------------------------------------------------------------- build
def plan_captions(E: dict, end_card_start: int, captions: Dict[str, Dict[str, str]] = None) -> List[dict]:
    """Stage-2 captions (``captions``: key -> tag/text of the language, default Russian).  The run of uncaptioned
    segments a caption extends over stops at every stage-1 part (s1 video segment or re-rendered stage-1 shot)."""
    captions = CAPTIONS if captions is None else captions
    segs = E["segments"]
    caps = []
    for i, s in enumerate(segs):
        key = s.get("caption") if s["src"] == "s2" else None
        if not key:
            continue
        if key not in captions:
            raise KeyError(f"segment {s.get('shot')}: caption key {key!r} has no text in storyboard2.TEXTS")
        j = i
        while (j + 1 < len(segs) and segs[j + 1]["src"] == "s2" and not segs[j + 1].get("caption")
               and not segs[j + 1].get("s1")):
            j += 1
        boundary = segs[j]["out1"]                         # last frame of the run (next: captioned / stage-1 / end)
        if end_card_start > s["out0"]:
            boundary = min(boundary, end_card_start - 1)   # last frame before the end card
        start, end = s["out0"] + CAPTION_IN, boundary - CAPTION_OUT
        caps.append({"start": start, "end": end, "tag": captions[key]["tag"], "text": captions[key]["text"],
                     "key": key, "shot": s.get("shot")})
    return caps


def s1_caption_text(c: dict, table) -> dict:
    """tag / text / key of stage-1 caption ``c`` in the storyboard language (table None: Russian as it is)."""
    en = S1_CAPTIONS_EN.get(c["tag"])
    if en is None:
        raise KeyError(f"stage-1 caption {c['tag']!r} has no entry in storyboard2.S1_CAPTIONS_EN")
    src = c if table is None else table[c["tag"]]
    return {"tag": src["tag"], "text": src["text"], "key": en["key"]}


def plan_s1_captions(E: dict, s1_sb: dict, table) -> List[dict]:
    """Render mode: the stage-1 captions shown in the stage-1 frames the edit uses, at the output frames of those
    stage-1 frames (edl.map_stage1_intervals), with the stage-1 caption look and fade.  A caption the edit cuts is
    kept (check() reports it); one entirely in dropped frames (track move 457-540, completion 935-995) is not shown."""
    caps = []
    for c in s1_sb.get("captions", []):
        pieces = edl_mod.map_stage1_intervals(E, [[c["start"], c["end"]]])
        if not pieces:
            continue
        t = s1_caption_text(c, table)
        for o0, o1 in pieces:
            caps.append(dict(start=int(o0), end=int(o1), tag=t["tag"], text=t["text"], key=t["key"],
                             shot=shot_at(E, o0), s1=[int(c["start"]), int(c["end"])]))
    return caps


def plan_audio(E: dict, s1_sb: dict) -> dict:
    """Soundtrack plan; the same for both editions: the stage-1 layers of the stage-1 part (video segments or the
    re-rendered stage-1 shots, never both) + the stage-2 layers / the stage-1 choreography seen from the stage-2 shots
    proper (s2_only)."""
    s1_audio = s1_sb.get("audio") or {}
    off = int(E.get("weld_offset", 0))
    s1_weld = s1_audio.get("weld_intervals") or []
    s1_servo = s1_audio.get("servo_intervals") or []
    E2 = s2_only(E)
    # stage-1 layers inside the stage-1 part (pieces running across a camera cut of the re-rendered shots merged,
    # so the arc does not re-ignite at a cut) ...
    weld = edl_mod.map_stage1_intervals(E, s1_weld)
    servo = edl_mod.map_stage1_intervals(E, s1_servo)
    # ... and the stage-1 choreography replayed in the stage-2 scene (scene frame = weld_offset + stage-1 frame),
    # visible in stage-2 shots (e.g. S2_10_done: the welding robot returns home = stage-1 servo move 931..1060)
    weld += edl_mod.map_intervals(E2, [[a + off, b + off] for a, b in s1_weld], src="s2")
    servo += edl_mod.map_intervals(E2, [[a + off, b + off] for a, b in s1_servo], src="s2")
    layers = []
    for spec in AUDIO_LAYERS:
        lay = dict(spec)
        lay["intervals"] = map_with_flags(E2, E["intervals"].get(spec["kind"], []), src="s2")
        layers.append(lay)
    ev = E["events"].get(CHIME["event"])
    chime_frames = [o for o in [edl_mod.scene_to_out(E2, ev) if ev is not None else None] if o is not None]
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


def plan_logos(title: dict, end_card: dict) -> dict:
    return {
        "title": dict(LOGO_TITLE, start=title["start"], end=title["end"], fade=FADE),
        "corner": dict(LOGO_CORNER, start=title["end"] + 1, end=end_card["start"] - 1, fade=FADE),
        "end_card": dict(LOGO_END_CARD, start=end_card["start"], end=end_card["end"], fade=FADE_END),
    }


def build(E: dict, s1_sb: dict = None) -> dict:
    """Storyboard dict for the EDL (output frame space); language, brand and stage-1 mode from the EDL."""
    s1_sb = s1_sb or load_json(S1_STORYBOARD)
    ed = edition_of(E)
    T = texts(ed["lang"])
    segs = E["segments"]
    if segs[0]["src"] != "s2" or segs[-1]["src"] != "s2":
        raise ValueError("the edit must start and end with a stage-2 segment (title / end card go there)")
    frames = int(E["frames"])
    last = segs[-1]
    ec_start = max(last["out0"], last["out1"] - END_CARD_FRAMES + 1)
    title = dict(T["title"], start=1, end=segs[0]["out1"] - TITLE_TAIL)
    end_card = {"start": ec_start, "end": last["out1"], "heading": T["end_card"]["heading"],
                "flow": list(T["end_card"]["flow"]), "facts": [list(f) for f in T["end_card"]["facts"]]}
    captions = plan_captions(E, ec_start, T["captions"])
    if ed["stage1_mode"] == "render":
        captions = sorted(captions + plan_s1_captions(E, s1_sb, T["s1_captions"]), key=lambda c: c["start"])
    sb = {
        "edition": ed["edition"],
        "lang": ed["lang"],
        "stage1_mode": ed["stage1_mode"],
        "brand": brand_entry(ed["brand"]),
        "fps": int(E.get("fps", 24)),
        "frames": frames,
        "resolution": list(s1_sb.get("resolution", [1920, 1080])),
        "font": FONTS["font"],
        "font_bold": FONTS["font_bold"],
        "font_light": FONTS["font_light"],
        "title": title,
        "captions": captions,
        "end_card": end_card,
        "logos": plan_logos(title, end_card),
        "corner_label": T["corner_label"],
        "footer": T["footer"],
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
    (name, out0, out1, complete) - complete = False when the edit cuts the text in the middle.  Video mode only
    (render mode draws the stage-1 captions itself: they are in sb["captions"])."""
    spans = [("s1 title", s1_sb["title"]["start"], s1_sb["title"]["end"])]
    spans += [("s1 " + c["tag"], c["start"], c["end"]) for c in s1_sb.get("captions", [])]
    spans += [("s1 end card", s1_sb["end_card"]["start"], s1_sb["end_card"]["end"])]
    out = []
    for name, a, b in spans:
        for o0, o1 in edl_mod.map_intervals(E, [[a, b]], src="s1"):
            out.append((name, o0, o1, o1 - o0 == b - a))
    return out


def strings(obj, path="sb"):
    """(path, string) of every string in a nested dict / list (the no-Cyrillic check of the English storyboard)."""
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from strings(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from strings(v, f"{path}[{i}]")


# strings of a non-Russian storyboard that may keep Cyrillic: not drawn (audio fallback tag of the stage-1 module)
CYRILLIC_OK = {"sb.audio.weld_prefix"}


def visible_seconds(a: int, b: int, fade: int, fps: int) -> float:
    """Time an overlay shown on output frames a..b is fully visible: its length minus one fade."""
    return (b - a + 1 - fade) / fps


def reading(sb: dict) -> List[tuple]:
    """(name, characters, visible seconds) of every text that has to be read in its time: the title heading, every
    caption text, the end card (heading + fact values; the labels and the flow diagram are for a second look)."""
    fps = sb["fps"]
    t, e = sb["title"], sb["end_card"]
    out = [("title", len(t["heading"]), visible_seconds(t["start"], t["end"], FADE, fps))]
    out += [("caption " + str(c.get("key") or c["tag"]), len(c["text"]),
             visible_seconds(c["start"], c["end"], FADE, fps)) for c in sb["captions"]]
    n = len(e["heading"]) + sum(len(v) for v, _ in e.get("facts") or [])
    out.append(("end card", n, visible_seconds(e["start"], e["end"], FADE_END, fps)))
    return out


def check(sb: dict, E: dict, s1_sb: dict = None) -> List[str]:
    """Return a list of problems (empty = ok): overlaps, gaps < MIN_GAP, too short, overlays over s1 segments,
    stage-1 burned-in texts cut by the edit or closer than MIN_GAP to a stage-2 overlay, stage-2 captions over a
    stage-1 part, render-mode stage-1 captions not at the output frames of their stage-1 frames, corner ranges != s2
    blocks, logo timing, Cyrillic in a non-Russian storyboard, audio intervals outside their segments."""
    P = []
    s1_sb = s1_sb or load_json(S1_STORYBOARD)
    ed = edition_of(E)
    for k in ("edition", "lang", "stage1_mode"):
        if sb.get(k, ed[k]) != ed[k]:
            P.append(f"{k} {sb.get(k)!r} != EDL {ed[k]!r}")
    if (sb.get("brand") or {}).get("key", ed["brand"]) != ed["brand"]:
        P.append(f"brand {sb['brand'].get('key')!r} != EDL {ed['brand']!r}")
    frames = sb["frames"]
    if frames != E["frames"]:
        P.append(f"frames {frames} != edl frames {E['frames']}")
    blocks = s2_blocks(E)
    s1r = s1_ranges(E)
    parts1 = stage1_parts(E)
    spans = overlay_spans(sb)
    for name, a, b, fade in spans:
        if not 1 <= a <= b <= frames:
            P.append(f"{name}: {a}..{b} outside 1..{frames}")
        if b - a + 1 <= 2 * fade:
            P.append(f"{name}: {b - a + 1} frames is not longer than two {fade}-frame fades")
        if not any(o0 <= a and b <= o1 for o0, o1 in blocks):
            P.append(f"{name}: {a}..{b} is not inside one stage-2 block {blocks}")
    for name, n, sec in reading(sb):
        if n > READ_CPS * sec + 1e-6:
            P.append(f"{name}: {n} characters in {sec:.1f} s = {n / sec:.1f} per second > {READ_CPS:.0f}: "
                     f"shorten the text to {int(READ_CPS * sec)} characters")
    for c in sb["captions"]:
        if c["end"] - c["start"] + 1 < MIN_CAPTION:
            P.append(f"caption {c.get('key')}: {c['end'] - c['start'] + 1} frames < {MIN_CAPTION}")
        in_s1 = any(o0 <= c["start"] and c["end"] <= o1 for o0, o1 in parts1)
        touches_s1 = any(c["start"] <= o1 and o0 <= c["end"] for o0, o1 in parts1)
        if c.get("s1") and not in_s1:
            P.append(f"stage-1 caption {c.get('key')} ({c['start']}..{c['end']}) is not inside the stage-1 part")
        if not c.get("s1") and touches_s1:
            P.append(f"stage-2 caption {c.get('key')} ({c['start']}..{c['end']}) runs into the stage-1 part {parts1}")
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
    # render mode: every stage-1 caption of the used frames drawn, complete, at the output frames of its stage-1 frames
    s1caps = [c for c in sb["captions"] if c.get("s1")]
    if ed["stage1_mode"] == "render":
        want = []
        for c in s1_sb.get("captions", []):
            for o0, o1 in edl_mod.map_stage1_intervals(E, [[c["start"], c["end"]]]):
                want.append((o0, o1, c["start"], c["end"]))
                if o1 - o0 != c["end"] - c["start"]:
                    P.append(f"stage-1 caption {c['tag']}: the edit cuts it (output {o0}..{o1})")
        got = [(c["start"], c["end"], c["s1"][0], c["s1"][1]) for c in s1caps]
        if sorted(got) != sorted(want):
            P.append(f"stage-1 captions {sorted(got)} != their stage-1 frames mapped to the output {sorted(want)}")
    elif s1caps:
        P.append(f"video mode: stage-1 captions are burned into the video, none may be drawn ({len(s1caps)})")
    if sb.get("corner_ranges") != blocks:
        P.append(f"corner_ranges {sb.get('corner_ranges')} != stage-2 blocks {blocks}")
    if sb.get("edl_segments") != edl_signature(E):
        P.append("edl_segments does not match the EDL")
    lg = sb.get("logos") or {}
    t, e = sb["title"], sb["end_card"]
    for name, want in (("title", (t["start"], t["end"])), ("end_card", (e["start"], e["end"])),
                       ("corner", (t["end"] + 1, e["start"] - 1))):
        L = lg.get(name)
        if L is None:
            P.append(f"logos.{name} missing")
        elif (L["start"], L["end"]) != want:
            P.append(f"logos.{name} {L['start']}..{L['end']} != {want[0]}..{want[1]}")
        elif L["end"] - L["start"] + 1 <= 2 * L.get("fade", FADE):
            P.append(f"logos.{name}: {L['end'] - L['start'] + 1} frames is not longer than two fades")
    b = sb.get("brand") or {}
    if not b.get("logo") or not (S2_ROOT / b["logo"]).is_file():
        P.append(f"brand logo {b.get('logo')!r} not found under {S2_ROOT}")
    if sb.get("lang", "ru") != "ru":
        cyr = [f"{p} = {s!r}" for p, s in strings(sb) if i18n2.has_cyrillic(s) and p not in CYRILLIC_OK]
        if cyr:
            P.append(f"Cyrillic text in the {sb['lang']} storyboard: " + "; ".join(cyr[:5]))
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
    b = sb.get("brand") or {}
    lines = [f"storyboard2 [{sb.get('edition', 'ru')}: {sb.get('lang', 'ru')}, {b.get('name', '?')}, stage 1 "
             f"{sb.get('stage1_mode', 'video')}]: {sb['frames']} frames = {sb['frames'] / fps:.2f} s, corner ranges "
             f"{sb['corner_ranges']}, s1 ranges {sb['s1_ranges']}",
             f"  title      {sb['title']['start']:5d}-{sb['title']['end']:5d}"]
    for c in sb["captions"]:
        sec = visible_seconds(c["start"], c["end"], FADE, fps)
        lines.append(f"  {c['key']:<11s}{c['start']:5d}-{c['end']:5d}  {(c['end'] - c['start'] + 1) / fps:4.1f} s"
                     f"  {len(c['text']) / sec:4.1f} ch/s  {c['tag']}"
                     + (f"  (stage-1 {c['s1'][0]}-{c['s1'][1]})" if c.get("s1") else ""))
    e = sb["end_card"]
    lines.append(f"  end card   {e['start']:5d}-{e['end']:5d}  {(e['end'] - e['start'] + 1) / fps:4.1f} s")
    lg = sb.get("logos") or {}
    if lg:
        lines.append("  logos: " + ", ".join(f"{k} {v['start']}-{v['end']} ({v['height']} px)" for k, v in lg.items()))
    au = sb["audio"]
    lines.append(f"  audio: {len(au['weld_intervals'])} arc + {len(au['servo_intervals'])} servo (stage 1), "
                 + ", ".join(f"{l['name']} {len(l['intervals'])}" for l in au["layers"])
                 + f", chime {au['chime']['frames']}")
    return "\n".join(lines)


def write(sb: dict, path=None) -> str:
    """Write the storyboard (path None -> the storyboard_json of its edition)."""
    path = path or editions.get(sb.get("edition"))["storyboard_json"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(sb, fh, ensure_ascii=False, indent=1)
    return str(path)


def main(argv=None) -> None:
    editions.safe_console()
    ap = argparse.ArgumentParser(description="Generate post/storyboard2_<ed>.json from post/edl_<ed>.json.")
    ap.add_argument("--edition", default="all", choices=editions.names() + ["all"],
                    help="edition (editions.py); default all")
    ap.add_argument("--edl", default=None, help="EDL json (default: the edition's post/edl_<ed>.json)")
    ap.add_argument("--out", default=None, help="storyboard json (default: the edition's post/storyboard2_<ed>.json)")
    a = ap.parse_args(argv)
    if (a.edl or a.out) and a.edition == "all":
        if not a.edl:
            raise SystemExit("--out needs --edl or --edition ru|en")
        eds = [None]                      # an explicit EDL: its own edition fields
    else:
        eds = editions.names() if a.edition == "all" else [a.edition]
    audios = {}
    for name in eds:
        edl_path = a.edl or editions.get(name)["edl_json"]
        E = load_json(edl_path)
        if name and E.get("edition", editions.DEFAULT) != name:
            raise SystemExit(f"{edl_path} is the EDL of edition {E.get('edition')!r}, not {name!r}")
        sb = build(E)
        audios[sb["edition"]] = sb["audio"]
        print(summary(sb))
        print("written", write(sb, a.out))
    if len(audios) > 1:
        same = all(v == next(iter(audios.values())) for v in audios.values())
        print("soundtrack plan identical in " + ", ".join(audios) + ":", "yes" if same else "NO")
        if not same:
            raise SystemExit("the editions must have the same soundtrack (plan_audio)")


if __name__ == "__main__":
    main()
