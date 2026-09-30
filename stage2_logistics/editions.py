"""Editions of the stage-2 video: one table, every script (build2.py, edl.py, post/*) takes ``--edition``.

    ru   Russian, ПИГРУПП logo.   The welding part is spliced from the finished stage-1 video
         (demo_video/deliverables/demo_welding_cell_1080p.mp4, frames 97-456 and 541-930): it is NOT re-rendered.
    en   English, ATOMIX logo.     The stage-1 video has Russian titles burned in, so its welding part is rendered
         again from the stage-2 scene with the stage-1 cameras (cameras2.S1_SHOTS; the stage-1 choreography is
         embedded there at WELD_OFFSET, so the shots are the same) and gets English captions in the montage.

Fields:
    lang         in-scene texts (i18n2) and overlay texts (post/storyboard2)
    brand        logo (branding2.BRANDS): title, corner, end card, in-scene plates and screens
    stage1       "video"  - splice the stage-1 video segments (edl src "s1")
                 "render" - render the stage-1 shots from the stage-2 scene (edl src "s2", shot "S1_...")
    edl_json / storyboard_json          written by edl.py, read by build2.py and post/compose2.py
    frames_dir / preview_dir / stills_dir  build2.py output folders (frames keep scene frame numbers)
    video / compact                     post/compose2.py outputs
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
POST = os.path.join(HERE, "post")
DELIV = os.path.join(HERE, "deliverables")

EDITIONS = {
    "ru": dict(name="ru", lang="ru", brand="pigrupp", stage1="video",
               label="Russian, PIGRUPP logo, welding part spliced from the stage-1 video"),
    "en": dict(name="en", lang="en", brand="atomix", stage1="render",
               label="English version, ATOMIX logo, welding shots rendered from the stage-2 scene"),
}
DEFAULT = "ru"

for _k, _e in EDITIONS.items():
    _e.setdefault("edl_json", os.path.join(POST, f"edl_{_k}.json"))
    _e.setdefault("storyboard_json", os.path.join(POST, f"storyboard2_{_k}.json"))
    _e.setdefault("frames_dir", os.path.join(OUT, f"frames_{_k}"))
    _e.setdefault("preview_dir", os.path.join(OUT, f"preview_{_k}"))
    _e.setdefault("stills_dir", os.path.join(OUT, f"stills_{_k}"))
    _e.setdefault("preview_video", os.path.join(OUT, f"preview_{_k}.mp4"))
    _e.setdefault("video", os.path.join(DELIV, f"demo_full_cycle_{_k}_{_e['brand']}_1080p.mp4"))
    _e.setdefault("compact", os.path.join(DELIV, f"demo_full_cycle_{_k}_{_e['brand']}_1080p_compact.mp4"))


def get(name=None):
    """The edition dict (a copy); ``name`` None -> DEFAULT.  Unknown names raise with the list of editions."""
    key = (name or DEFAULT).lower()
    if key not in EDITIONS:
        raise SystemExit(f"unknown edition {name!r}; editions: {', '.join(EDITIONS)}")
    return dict(EDITIONS[key])


def names():
    return list(EDITIONS)


def safe_console():
    """Print Cyrillic (Russian captions in summaries, the ru storyboard) without UnicodeEncodeError when stdout /
    stderr is redirected on a Windows console with a legacy code page: unencodable characters are escaped."""
    import sys
    for st in (sys.stdout, sys.stderr):
        enc = (getattr(st, "encoding", None) or "").lower().replace("-", "")
        if enc != "utf8" and hasattr(st, "reconfigure"):
            try:
                st.reconfigure(errors="backslashreplace")
            except (ValueError, OSError):
                pass


def add_argument(ap):
    """``--edition`` for an argparse parser (every CLI of the pipeline uses the same option; also makes the console
    output safe on Windows code pages, see safe_console)."""
    safe_console()
    ap.add_argument("--edition", default=DEFAULT, choices=names(),
                    help="ru: Russian + PIGRUPP logo (stage-1 video spliced); en: English + ATOMIX logo (all shots rendered)")
    return ap
