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
    end_card.flow        the 7 steps of the flow diagram;  end_card.kpis  the bullet lines (== end_card.lines)
    audio.layers         stage-2 sound layers with their intervals mapped to output frames (see audio2.py)

Editions (EDL "stage1_mode"):
    video   (ru) the welding part is spliced from the stage-1 video: its title, captions and corner label are burned
            in, nothing is drawn over it but the corner logo, the corner label cuts hard at the splices.
    render  (en) the welding part is rendered from the stage-2 scene with the stage-1 cameras: the stage-1 captions
            that the stage-1 video shows in the used frames are drawn (translated, S1_CAPTIONS_EN) at the same output
            frames (edl.map_stage1_intervals) with the stage-1 look and fade; one corner label over the whole video.
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
All texts live in this file (TEXTS: Russian = the keys, exact; English): edit them here and re-run the script
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

# ----------------------------------------------------------------------------- texts (English)
# Same keys and structure as the Russian ones; lengths kept close to the Russian so the layout (caption wrap 1160 px,
# end-card boxes / bullets) stays the same.  Vocabulary (the same word for the same machine in the scene texts,
# i18n2): pipe spool, loading robot, kit cassette, fit-up station, tack welds, root gap, welding robot, positioner,
# faceplate, light curtain muting, carrier pallet, roller conveyor, laser marker, WPS, laser profilometer, QC, storage
# rack; British spelling (centring, modelled, labour).
TITLE_EN = {
    "heading": "Pipe spool welding line: the full cycle",
    "sub": "Kitting · fit-up and tack welding · cell loading · welding · marking and inspection · storage",
}

CAPTIONS_EN: Dict[str, Dict[str, str]] = {
    "flange": {"tag": "KITTING",
               "text": "A loading robot on a linear track picks the parts from the kit cassette: the flange goes "
                       "onto the fit-up station, where pneumatic clamps lock it"},
    "elbow": {"tag": "FIT-UP",
              "text": "The elbow is set on the flange against centring stops and a V-block — the fixture sets the "
                      "position, not manual marking-out"},
    "pipe": {"tag": "FIT-UP · ROOT GAP",
             "text": "The pipe is laid in the cradle against the end stop; a sensor checks the root gap before "
                     "tacking"},
    "tack": {"tag": "TACK WELDING",
             "text": "A compact welding robot places three tack welds on each joint — the assembly is secured "
                     "without manual labour"},
    "carry": {"tag": "TRANSFER",
              "text": "The tacked spool is picked up in one piece: one kit of parts — one spool"},
    "load": {"tag": "CELL LOADING",
             "text": "Welding robot in its home position, light curtain muted: the loading robot sets the spool on "
                     "the faceplate and the clamps lock the flange"},
    "ready": {"tag": "HANDOVER TO WELDING",
              "text": "The loading robot leaves the guarded area and the positioner turns the workpiece to the "
                      "welding start position"},
    "done": {"tag": "WELDING COMPLETE",
             "text": "Both joints are welded: the welding robot returns home and the positioner brings the "
                     "workpiece back to the unloading position"},
    "unload": {"tag": "UNLOADING",
               "text": "The faceplate clamps release the flange and the loading robot takes the spool out of the "
                       "cell"},
    "carrier": {"tag": "OUTFEED CONVEYOR",
                "text": "The spool is set on a carrier pallet and travels along the roller conveyor to the marking "
                        "and QC station"},
    "mark": {"tag": "MARKING",
             "text": "A laser marker applies the spool ID and the WPS number — traceability down to every weld"},
    "scan": {"tag": "INSPECTION",
             "text": "A laser profilometer scans the weld and checks the bead geometry against tolerances; the "
                     "result is stored under the spool ID"},
    "store": {"tag": "STORAGE",
              "text": "The loading robot places the finished spool in the storage rack; the flange of the next kit "
                      "is already on the fit-up station"},
}

END_CARD_EN = {
    "heading": "Full cycle of the spool welding line",
    "flow": ["Kit cassette", "Fit-up", "Tack welding", "Cell loading", "Welding", "Marking and QC", "Rack · AGV"],
    "kpis": [
        "3 robots: track-mounted loading robot (150 kg class, 3.2 m reach), tack welding robot, track-mounted "
        "welding robot + 2-axis positioner",
        "Clamping gripper with V-blocks for DN100–300, interchangeable jaws",
        "Logistics runs in parallel with welding: the welding time sets the cycle time of the line",
        "Every spool is traceable by its ID: weld parameters, QC result, storage location",
    ],
    "cta": "Stage 2 · Logistics and kitting · layout to be tailored to customer data",
}
FOOTER_EN = "Demonstration 3D simulation · modelled on photos of the shop floor"
CORNER_LABEL_EN = "DEMO · SIMULATION · SPED UP"

# Stage-1 captions (demo_video/post/storyboard.json, burned into the stage-1 video) for the render mode, keyed by the
# exact Russian tag (note the Latin A / B and the en dash of the stage-1 tags).  All 8 are translated; the edit shows
# the 6 inside the stage-1 frames it uses.  "key" names the caption in the storyboard.
S1_CAPTIONS_EN: Dict[str, Dict[str, str]] = {
    "ПОДГОТОВКА": {"key": "s1_setup", "tag": "PREPARATION",
                   "text": "The positioner tilts the workpiece to bring joint A into the flat position. The robot "
                           "approaches along the track"},
    "АДАПТАЦИЯ": {"key": "s1_tracking", "tag": "PATH ADAPTATION",
                  "text": "A laser sensor on the torch scans the joint: the path is corrected to match the actual "
                          "fit-up geometry"},
    "ШОВ A · ВРАЩЕНИЕ ИЗДЕЛИЯ": {"key": "s1_weld_a", "tag": "WELD A · WORKPIECE ROTATION",
                                "text": "Weld A: flange to elbow. The workpiece rotates 360° while the torch stays at "
                                        "12 o'clock, weaving across the joint"},
    "ПЕРЕМЕЩЕНИЕ ПО ТРЕКУ": {"key": "s1_track", "tag": "TRACK TRAVEL",
                             "text": "After weld A, the pipe lies along the track: the robot travels on its 7th "
                                     "axis to joint B"},
    "ШОВ B · СЕКТОРЫ 1–2": {"key": "s1_weld_b12", "tag": "WELD B · SECTORS 1–2",
                           "text": "Weld B: elbow to pipe. Here the robot moves the torch: two 90° sectors, from "
                                   "12 to 3 and from 12 to 9 o'clock"},
    "ИНДЕКСАЦИЯ 180°": {"key": "s1_index", "tag": "180° INDEXING",
                        "text": "The positioner flips the workpiece 180° over the top and the robot moves to the "
                                "other side: the unwelded half of joint B is on top again"},
    "ШОВ B · СЕКТОРЫ 3–4": {"key": "s1_weld_b34", "tag": "WELD B · SECTORS 3–4",
                           "text": "Sectors 3–4: after indexing, the whole seam is welded between the flat and "
                                   "vertical positions, never overhead"},
    "ЗАВЕРШЕНИЕ": {"key": "s1_done", "tag": "CYCLE COMPLETE",
                   "text": "Cycle complete: both joints welded, the robot and the positioner return to the loading "
                           "position. The parameters of each weld are stored under the spool ID"},
}

TEXTS = {
    "ru": dict(title=TITLE, captions=CAPTIONS, end_card=END_CARD, footer=FOOTER, corner_label=CORNER_LABEL,
               s1_captions=None),                  # None: the stage-1 captions as they are (Russian)
    "en": dict(title=TITLE_EN, captions=CAPTIONS_EN, end_card=END_CARD_EN, footer=FOOTER_EN,
               corner_label=CORNER_LABEL_EN, s1_captions=S1_CAPTIONS_EN),
}

# ----------------------------------------------------------------------------- timing (output frames)
TITLE_TAIL = 10            # title ends this many frames before the end of the first segment
CAPTION_IN = 3             # caption starts out0 + 3
CAPTION_OUT = 4            # caption ends 4 frames before its boundary frame
END_CARD_FRAMES = 100      # end card = last 100 frames of the last segment (~4.2 s)
MIN_CAPTION = 28           # frames
MIN_GAP = 4                # empty frames between two overlays of the title/caption/end-card track
FADE = 12                  # stage-1 fades: title/captions 12 frames, end card 14
FADE_END = 14

# logos of sb["brand"] (overlays2 draws them from branding2): the logo on a white rounded plate, height in px.
#   title     large, top left (the title text is the bottom-third bar), title timing and fades
#   corner    small, top left (the stage-1 corner label is top RIGHT, same y), title end + 1 .. end card start - 1,
#             drawn over the whole video (stage-1 video segments included), fades only at its own start / end
#   end_card  centred above the heading, part of the end-card image (overlays2.render_end_card2)
LOGO_TITLE = {"x": 80, "y": 64, "height": 150}
LOGO_CORNER = {"x": 80, "y": 40, "height": 64, "opacity": 1.0}
LOGO_END_CARD = {"height": 120}

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
                "flow": list(T["end_card"]["flow"]), "kpis": list(T["end_card"]["kpis"]),
                "lines": list(T["end_card"]["kpis"]), "cta": T["end_card"]["cta"]}
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
        "font": s1_sb["font"],
        "font_bold": s1_sb["font_bold"],
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
        lines.append(f"  {c['key']:<11s}{c['start']:5d}-{c['end']:5d}  {(c['end'] - c['start'] + 1) / fps:4.1f} s"
                     f"  {c['tag']}" + (f"  (stage-1 {c['s1'][0]}-{c['s1'][1]})" if c.get("s1") else ""))
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
