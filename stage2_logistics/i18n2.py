"""Language of the texts drawn INTO the 3D scene (signs, HMI screens, labels) - the overlay texts of the montage live
in post/storyboard2.py.

    import i18n2
    i18n2.set_lang("en")            # build2.build_scene does this from the edition, before any texture is drawn
    d.text(xy, i18n2.tr("СКАН ШВА B"), ...)

The Russian string is the key: in the Russian edition ``tr`` returns it unchanged, in the English one it returns the
entry of EN - a missing entry raises, so no Russian text can slip into the English video unnoticed
(tests/t_i18n2.py also records every string PIL draws while the English scene is built).
"""
import re

LANGS = ("ru", "en")
_LANG = "ru"

# Russian (key) -> English, single strings.
EN = {
    # ---- logistics-zone HMI (environment2._hmi_mat)
    "ЗОНА ЛОГИСТИКИ · АВТО": "LOGISTICS ZONE · AUTO",
    "КАССЕТА": "KIT CASSETTE",
    "СТЕНД": "FIT-UP",
    "ЯЧЕЙКА": "WELD CELL",
    "СКЛАД": "STORAGE",
    "КОНТРОЛЬ": "QC",
    "КОНВЕЙЕР": "CONVEYOR",
    "SPL-02   ТАКТ 06:40   ЗАВЕСА: MUTING": "SPL-02   CYCLE 06:40   CURTAIN: MUTING",
    # ---- labels
    "ШУ ЗОНЫ ЛОГИСТИКИ  =LZ1+CP01": "LOGISTICS ZONE CONTROL PANEL  =LZ1+CP01",
    "СКЛАД ГОТОВЫХ СПУЛОВ · FINISHED SPOOLS": "FINISHED SPOOL STORAGE",
    # ---- marking / QC screen (marking_qc._display_image)
    "QC-01  МАРКИРОВКА / КОНТРОЛЬ": "QC-01  MARKING / INSPECTION",
    "ОЖИДАНИЕ": "STANDBY",
    "паллета не на позиции": "pallet not in position",
    "МАРКИРОВКА": "MARKING",
    "лазер 50 Вт · DataMatrix": "50 W laser · DataMatrix",
    "СКАН ШВА B": "SCANNING WELD B",
    "ГОДЕН · QC OK": "PASS · QC OK",
    "Шов A": "Weld A",
    "Шов B": "Weld B",
    "Маркировка": "Marking",
}

# Multi-line texts (fence signs: 4 lines each, environment2.SIGNS) are translated as a whole - the line breaks of the
# two languages fall at different words.
EN_LINES = {
    ("ОСТОРОЖНО", "ЗОНА ЛОГИСТИКИ", "ВХОД ТОЛЬКО", "ПРИ ОСТАНОВКЕ"): ("CAUTION", "LOGISTICS ZONE", "ENTRY ONLY", "WHEN STOPPED"),
    ("ОПАСНО", "РАБОТАЕТ РОБОТ-", "ПОГРУЗЧИК", ""): ("DANGER", "ROBOT LOADER", "IN OPERATION", ""),
    ("ВНИМАНИЕ", "ЗАГРУЗКА", "КАССЕТ", ""): ("WARNING", "KIT CASSETTE", "LOADING", ""),
    ("ВНИМАНИЕ", "ЗАГРУЗКА ТРУБ", "СВЕТОВАЯ", "ЗАВЕСА"): ("WARNING", "PIPE LOADING", "LIGHT", "CURTAIN"),
    ("ОСТОРОЖНО", "ВЫДАЧА ГОТОВЫХ", "ДВИЖЕНИЕ AGV", ""): ("CAUTION", "FINISHED GOODS", "AGV TRAFFIC", ""),
    ("ОПАСНО", "АВТОМАТИЧЕСКИЙ", "РЕЖИМ", ""): ("DANGER", "AUTOMATIC", "MODE", ""),
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
    """``text`` (Russian) in the current language.  Strings without Cyrillic (IDs, units, 'QC OK') pass through."""
    if _LANG == "ru" or not has_cyrillic(text):
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
