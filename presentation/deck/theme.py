"""Design tokens for the dissertation presentation.

Every value here was measured from the reference deck
``Sidang_Disertasi_Toto_Suharto (2).pptx`` so that generated decks are
visually indistinguishable from the approved template.
"""

from __future__ import annotations

from pptx.dml.color import RGBColor
from pptx.util import Emu, Inches, Pt


# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------
class Color:
    """Hex strings; use :func:`rgb` to convert to a python-pptx colour."""

    NAVY = "16284B"        # primary brand, headings, dark panels
    GOLD = "B8923B"        # accent: eyebrows, figures, rules, badges
    INK = "1F2A3D"         # body copy on light backgrounds
    MUTED = "6B7689"       # secondary copy, footers, captions
    LINE = "DCE2EC"        # hairline borders
    PANEL = "F4F6FA"       # card fill (light)
    PANEL_ALT = "EEF1F7"   # alternate / zebra fill
    WHITE = "FFFFFF"
    CREAM = "FBF4E4"       # rare highlight fill

    # copy on dark (navy) surfaces, in decreasing emphasis
    ON_DARK = "FFFFFF"
    ON_DARK_1 = "E4E9F2"
    ON_DARK_2 = "D8E0EE"
    ON_DARK_3 = "CDD6E6"
    ON_DARK_4 = "C8D2E6"
    ON_DARK_5 = "AAB6CC"
    GOLD_LIGHT = "E6D6A8"  # gold accent legible on navy

    GHOST = "B9C2D2"       # figure-placeholder glyph
    GREEN = "3E8E5B"       # "EMPIRIS" status badge
    RED = "C0392B"         # regression / critical value


def rgb(hex_str: str) -> RGBColor:
    return RGBColor.from_string(hex_str)


# --------------------------------------------------------------------------
# Typography
# --------------------------------------------------------------------------
class Font:
    DISPLAY = "Cambria"   # headings, figures, labels
    BODY = "Calibri"      # running copy

    # Average glyph advance as a fraction of the em, used for height
    # estimation. Calibri and Cambria are both fairly narrow.
    ADVANCE = {"Calibri": 0.472, "Cambria": 0.492}


LINE_SPACING = 1.12  # 112% — matches every body paragraph in the reference


# --------------------------------------------------------------------------
# Canvas & grid (inches)
# --------------------------------------------------------------------------
class Grid:
    SLIDE_W = 13.333
    SLIDE_H = 7.5

    MARGIN = 0.55                       # left/right text margin
    CONTENT_W = SLIDE_W - 2 * MARGIN    # 12.233 -> rounded to 12.23 below
    CONTENT_W = 12.23

    EYEBROW_Y = 0.40
    EYEBROW_H = 0.28
    TITLE_Y = 0.69
    TITLE_H = 0.74

    BAND_TOP = 1.60                     # first usable y for content blocks
    BAND_BOTTOM = 6.86                  # last usable y (above the footer)

    FOOTER_Y = 7.07
    FOOTER_H = 0.30
    PAGENO_X = 12.45
    PAGENO_W = 0.45

    GAP = 0.16                          # default vertical gap between blocks
    GUTTER = 0.22                       # default horizontal gap between columns
    PAD = 0.20                          # inner padding of a card/panel
    RADIUS = 0.09                       # corner radius of every rounded shape


class Size:
    """Type scale, in points."""

    EYEBROW = 12.0
    TITLE = 26.0
    LEAD = 13.0
    BODY = 11.5
    SMALL = 11.0
    TINY = 10.5
    MICRO = 10.0
    FOOTER = 9.0
    SOURCE = 8.5

    CARD_TITLE = 13.0
    PANEL_TITLE = 13.0
    STAT = 32.0
    STAT_SM = 20.0
    HERO = 46.0


class Stroke:
    """Line widths in EMU (1 pt = 12700)."""

    HAIRLINE = Emu(6350)
    THIN = Emu(12700)
    MEDIUM = Emu(15875)
    THICK = Emu(19050)


# Convenience re-exports so block modules need a single import.
__all__ = [
    "Color", "Font", "Grid", "Size", "Stroke",
    "rgb", "LINE_SPACING", "Emu", "Inches", "Pt",
]
