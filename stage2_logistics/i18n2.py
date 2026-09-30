"""Language of the texts drawn INTO the 3D scene (signs, HMI screens, labels) - the overlay texts of the montage live
in post/storyboard2.py.

    import i18n2
    i18n2.set_lang("en")            # build2.build_scene does this from the edition, before any texture is drawn
    d.text(xy, i18n2.tr("СКАН ШВА B"), ...)

The Russian string is the key: in the Russian edition ``tr`` returns it unchanged, in the English one it returns the
entry of EN - a missing entry raises, so no Russian text can slip into the English video unnoticed
(tests/t_i18n2.py also records every string PIL draws while the English scene is built).  Latin-script texts pass
through, except the few in EN_LATIN that are Latin but not English (Russian-style abbreviations).
"""
import re

LANGS = ("ru", "en")
_LANG = "ru"

# Russian (key) -> English, single strings.  Every entry is drawn into a texture box of a fixed size; the drawing code
# shrinks a text to fit its box, so shorter wording keeps the English as large as the Russian (HMI boxes: one word).
EN = {
    # ---- logistics-zone HMI (environment2._hmi_mat): header, material-flow boxes (140 px), status line
    "ЗОНА ЛОГИСТИКИ · АВТО": "LOGISTICS ZONE · AUTO",
    "КАССЕТА": "CASSETTE",
    "СТЕНД": "FIT-UP",
    "ЯЧЕЙКА": "WELD CELL",
    "СКЛАД": "STORAGE",
    "КОНТРОЛЬ": "QC",
    "КОНВЕЙЕР": "CONVEYOR",
    "SPL-02   ТАКТ 06:40   ЗАВЕСА: MUTING": "SPL-02   CYCLE 06:40   CURTAIN: MUTING",
    # ---- labels: zone control cabinet name plate (environment2), storage rack header sign (storage._sign_mat)
    "ШУ ЗОНЫ ЛОГИСТИКИ  =LZ1+CP01": "LOGISTICS CONTROL PANEL  =LZ1+CP01",
    "СКЛАД ГОТОВЫХ СПУЛОВ · FINISHED SPOOLS": "FINISHED SPOOL STORAGE",
    # ---- marking / QC screen (marking_qc._display_image): header, state titles (64 px bold), detail lines
    "QC-01  МАРКИРОВКА / КОНТРОЛЬ": "QC-01  MARKING / INSPECTION",
    "ОЖИДАНИЕ": "STANDBY",
    "паллета не на позиции": "no pallet in position",
    "МАРКИРОВКА": "MARKING",
    "лазер 50 Вт · DataMatrix": "50 W laser · DataMatrix",
    "СКАН ШВА B": "SCANNING WELD B",
    "ГОДЕН · QC OK": "PASS · QC OK",
    "Шов A": "Weld A",
    "Шов B": "Weld B",
    "Маркировка": "Marking",
}

# Multi-line texts (fence signs: header + 3 lines, environment2.SIGNS) are translated as a whole - the line breaks of
# the two languages fall at different words.  Signage style: ANSI-like headers (DANGER / WARNING / CAUTION), caps.
EN_LINES = {
    ("ОСТОРОЖНО", "ЗОНА ЛОГИСТИКИ", "ВХОД ТОЛЬКО", "ПРИ ОСТАНОВКЕ"): ("CAUTION", "LOGISTICS ZONE", "ENTRY ONLY", "WHEN STOPPED"),
    ("ОПАСНО", "РАБОТАЕТ РОБОТ-", "ПОГРУЗЧИК", ""): ("DANGER", "LOADING ROBOT", "IN OPERATION", ""),
    ("ВНИМАНИЕ", "ЗАГРУЗКА", "КАССЕТ", ""): ("WARNING", "CASSETTE", "LOADING", ""),
    ("ВНИМАНИЕ", "ЗАГРУЗКА ТРУБ", "СВЕТОВАЯ", "ЗАВЕСА"): ("WARNING", "PIPE LOADING", "LIGHT", "CURTAIN"),
    ("ОСТОРОЖНО", "ВЫДАЧА ГОТОВЫХ", "ДВИЖЕНИЕ AGV", ""): ("CAUTION", "SPOOL DISPATCH", "AGV TRAFFIC", ""),
    ("ОПАСНО", "АВТОМАТИЧЕСКИЙ", "РЕЖИМ", ""): ("DANGER", "AUTOMATIC", "OPERATION", ""),
}

# Latin-script texts of the Russian scene that are not English (the Russian edition keeps them as they are): looked up
# by tr() before its Latin pass-through.  Laser warning label of the QC arch (marking_qc qc_label atlas; "KL." is the
# Russian / German abbreviation of "class", IEC 60825-1 English: "CLASS 4").
EN_LATIN = {
    "LASER KL.4": "LASER CLASS 4",
}

_CYR = re.compile("[Ѐ-ӿ]")


def set_lang(lang):
    global _LANG
    if lang not in LANGS:
        raise ValueError(f"language {lang!r} not in {LANGS}")
    _LANG = lang


def get_lang():
    return _LANG


def has_cyrillic(text):
    return bool(_CYR.search(text or ""))


def tr(text):
    """``text`` (Russian) in the current language.  Strings without Cyrillic (IDs, units, 'QC OK') pass through,
    except the EN_LATIN ones."""
    if _LANG == "ru":
        return text
    if text in EN_LATIN:
        return EN_LATIN[text]
    if not has_cyrillic(text):
        return text
    try:
        return EN[text]
    except KeyError:
        raise KeyError(f"i18n2: no English text for {text!r} - add it to i18n2.EN") from None


def tr_lines(lines):
    """A multi-line text (tuple of lines, e.g. a fence sign) in the current language, translated as a whole."""
    lines = tuple(lines)
    if _LANG == "ru" or not any(has_cyrillic(t) for t in lines):
        return lines
    try:
        return tuple(EN_LINES[lines])
    except KeyError:
        raise KeyError(f"i18n2: no English text for the lines {lines!r} - add them to i18n2.EN_LINES") from None
