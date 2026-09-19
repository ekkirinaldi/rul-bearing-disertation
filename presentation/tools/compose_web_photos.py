"""Compose the four Wikimedia machine photographs into one manuscript figure.

The dissertation opens Bab I by naming the rotating machines that every
industry runs, and one figure carries all four. Each photograph is
centre-cropped to 4:3, scaled to a common width, and captioned (a)-(d) so the
prose can point at a panel. Author and licence stay in the figure caption
written by ``manuscript/inserts/v15/spec.yaml``; the machine-readable
record is ``assets/web/CREDITS.md`` (written by ``make web-assets``).

Usage: python3 tools/compose_web_photos.py  (or ``make montage``)
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "assets" / "web"
OUT = WEB / "mesin_rotasi_montase.png"

PANEL_W = 1200          # px per panel; the montage is two panels wide
PANEL_H = 900           # 4:3
GUTTER = 24
STRIP_H = 96            # caption strip under each panel
INK = (22, 40, 75)
PAPER = (255, 255, 255)

# (file, label) in reading order: two rows of two
PANELS: list[tuple[str, str]] = [
    ("motor_listrik_cutaway.jpg", "(a) motor listrik"),
    ("fan_sentrifugal_industri.jpg", "(b) fan sentrifugal"),
    ("pompa_cutaway.jpg", "(c) pompa"),
    ("mesin_grinding_spindle.jpg", "(d) spindle mesin grinding"),
]

# DejaVu ships with matplotlib, which the diagram tool already depends on, so
# the montage keeps the same type as the rendered charts rather than falling
# back to PIL's bitmap default.
_FONT_CANDIDATES = [
    "/opt/homebrew/lib/python3.13/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]


def _font(size: int):
    try:  # matplotlib is installed for the diagrams; reuse its DejaVu
        import matplotlib
        path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
        if path.exists():
            return ImageFont.truetype(str(path), size)
    except Exception:  # noqa: BLE001 - fall through to the candidates
        pass
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _crop_to_ratio(image: Image.Image, w: int, h: int) -> Image.Image:
    """Centre-crop to the target aspect, then resize to exactly (w, h)."""
    target = w / h
    if image.width / image.height > target:          # too wide: trim the sides
        keep = round(image.height * target)
        left = (image.width - keep) // 2
        image = image.crop((left, 0, left + keep, image.height))
    else:                                            # too tall: trim top/bottom
        keep = round(image.width / target)
        top = (image.height - keep) // 2
        image = image.crop((0, top, image.width, top + keep))
    return image.convert("RGB").resize((w, h), RESAMPLE)


def main() -> int:
    missing = [name for name, _ in PANELS if not (WEB / name).exists()]
    if missing:
        print(f"run `make web-assets` first; missing {missing}", file=sys.stderr)
        return 1

    cell_h = PANEL_H + STRIP_H
    width = 2 * PANEL_W + GUTTER
    height = 2 * cell_h + GUTTER
    sheet = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(sheet)
    font = _font(52)

    for i, (name, label) in enumerate(PANELS):
        col, row = i % 2, i // 2
        x = col * (PANEL_W + GUTTER)
        y = row * (cell_h + GUTTER)
        with Image.open(WEB / name) as source:
            sheet.paste(_crop_to_ratio(source, PANEL_W, PANEL_H), (x, y))
        box = draw.textbbox((0, 0), label, font=font)
        draw.text((x + (PANEL_W - (box[2] - box[0])) / 2,
                   y + PANEL_H + (STRIP_H - (box[3] - box[1])) / 2 - box[1]),
                  label, font=font, fill=INK)

    sheet.save(OUT, format="PNG")
    print(f"{OUT.relative_to(ROOT)}  {sheet.width}x{sheet.height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
