"""Fonts shipped with the repository (stage2_logistics/assets/fonts) so labels, decals, markings and titles look the
same on Linux, Windows and macOS (the stage-1 code expects Debian font paths)."""
import os

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "fonts")
SANS = os.path.join(DIR, "DejaVuSans.ttf")
SANS_BOLD = os.path.join(DIR, "DejaVuSans-Bold.ttf")
MARKER = os.path.join(DIR, "FreeSansBoldOblique.ttf")      # stage-1 paint-marker lettering (cell/spool.py _FONTS[0])


def truetype(path, size):
    from PIL import ImageFont
    return ImageFont.truetype(path, int(size))


def patch_stage1():
    """Make the stage-1 spool markings use the shipped fonts (spool._FONTS holds Debian paths and would fall back to
    PIL's bitmap font elsewhere, so the paint markings would not match the stage-1 video)."""
    import tools  # noqa: F401
    from cell import spool
    for p in (SANS_BOLD, MARKER):
        if p not in spool._FONTS:
            spool._FONTS.insert(0, p)
