"""Pull the deck-worthy figures out of the V14 manuscript DOCX.

The V14 media stream is a deterministic 1:1 map from ``word/media/imageN.png``
to figure numbers (image2+3 -> Gambar I.1, image4 -> I.2, ... image58 -> VI.1),
so each figure can be extracted by index. Only figures that have no standalone
PNG under ``manuscript/assets/figures/`` are listed here.

Usage: python3 tools/extract_v14_media.py  (or ``make v14-assets``)
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "v14"

# word/media index -> target filename (named after the V14 figure number)
MAP = {
    4: "gambar_i2_fishbone.png",                # Gambar I.2  fishbone akar masalah
    16: "gambar_ii11_peta_sota.png",            # Gambar II.11 peta state of the art
    23: "gambar_iii7_akselerometer.png",        # Gambar III.7 SKF CMSS2200
    24: "gambar_iii8_rantai_akuisisi_skf.png",  # Gambar III.8 IMx-8 -> AWS -> XLSX
    25: "gambar_iii9_skf_observer.png",         # Gambar III.9 penempatan sensor OR-1/OR-2
    40: "gambar_iv11_skema_transfer_skf.png",   # Gambar IV.11 skema pengujian eksternal
    41: "gambar_v1_arsitektur_mamba_xlstm.png", # Gambar V.1  arsitektur Mamba-xLSTM-Net
    45: "gambar_v5_integrasi_sae.png",          # Gambar V.5  integrasi SAE-backbone
    46: "gambar_v6_prosedur_sae_bpfx.png",      # Gambar V.6  prosedur tiga tahap
    54: "gambar_v14_perbandingan_backbone.png", # Gambar V.14 RMSE + PHM Score
}

# These carry a caption baked into the pixels — three with a stale figure
# number from an earlier manuscript revision (D.1 / V.3 / V.4), one merely
# duplicating the slide caption. Cut the strip off at the given row, then
# trim the white margins.
CROP_BOTTOM = {40: 520, 41: 1235, 45: 860, 46: 860}


def _clean(data: bytes, bottom: int) -> bytes:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    image = image.crop((0, 0, image.width, min(bottom, image.height)))
    background = Image.new("RGB", image.size, (255, 255, 255))
    bbox = ImageChops.difference(image, background).getbbox()
    if bbox:
        pad = 8
        image = image.crop((max(bbox[0] - pad, 0), max(bbox[1] - pad, 0),
                            min(bbox[2] + pad, image.width), min(bbox[3] + pad, image.height)))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def main() -> int:
    versions = ROOT.parent / "manuscript" / "versions"
    matches = sorted(versions.glob("V14 *.docx"))
    if not matches:
        print(f"V14 manuscript not found under {versions}", file=sys.stderr)
        return 1
    docx = matches[0]
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(docx) as archive:
        for index, name in sorted(MAP.items()):
            data = archive.read(f"word/media/image{index}.png")
            if index in CROP_BOTTOM:
                data = _clean(data, CROP_BOTTOM[index])
            target = OUT / name
            target.write_bytes(data)
            print(f"image{index:<3} -> {target.relative_to(ROOT)}  ({len(data) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
