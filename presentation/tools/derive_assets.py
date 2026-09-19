"""Derive IMS/CWRU-free crops of the manuscript result figures.

IMS was dropped from the manuscript and CWRU is out of the SAE hit-rate
narrative, but the figure artwork in ``manuscript/assets`` still
carries their panels (2x2 grids). The manuscript pipeline owns those files,
so this script leaves them untouched and writes cropped copies that keep
only the PHM2012 and XJTU-SY panels into ``assets/derived/``. Re-rendering
the source figures upstream makes these crops obsolete.

Usage: python3 tools/derive_assets.py  (or ``make derived-assets``)
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT.parent / "manuscript" / "assets" / "figures" / "bab5"
OUT = ROOT / "assets" / "derived"

# source name -> (target name, crop box as fractions of (W, H))
CROPS = {
    # 2x2: PHM2012 | XJTU-SY on the top row, IMS | CWRU below -> keep top row
    "hitrate_panel.png": ("hitrate_panel_phm_xjtu.png", (0.0, 0.0, 1.0, 0.545)),
    # 2x2: PHM2012 | IMS, XJTUSY | CWRU -> keep the left column
    "sparsity_sweep.png": ("sparsity_sweep_phm_xjtu.png", (0.0, 0.0, 0.505, 1.0)),
}


def _trim(image: Image.Image, pad: int = 10) -> Image.Image:
    background = Image.new("RGB", image.size, (255, 255, 255))
    bbox = ImageChops.difference(image, background).getbbox()
    if not bbox:
        return image
    return image.crop((max(bbox[0] - pad, 0), max(bbox[1] - pad, 0),
                       min(bbox[2] + pad, image.width), min(bbox[3] + pad, image.height)))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for source_name, (target_name, (fx0, fy0, fx1, fy1)) in CROPS.items():
        source = SRC / source_name
        image = Image.open(source).convert("RGB")
        w, h = image.size
        image = image.crop((int(fx0 * w), int(fy0 * h), int(fx1 * w), int(fy1 * h)))
        image = _trim(image)
        target = OUT / target_name
        image.save(target, format="PNG")
        print(f"{source_name:22s} -> {target.relative_to(ROOT)}  {image.size}")

    # kerangka (Gambar III.1): the prognostic dataset box still lists IMS on
    # its second line (y 724-742 at 2344x893) — paint it over with the box
    # fill so the box reads "PHM2012 · XJTU-SY" only.
    source = SRC.parent / "bab3" / "kerangka_terintegrasi.png"
    image = Image.open(source).convert("RGB")
    image.paste((236, 236, 236), (80, 703, 320, 753))
    target = OUT / "kerangka_terintegrasi_no_ims.png"
    image.save(target, format="PNG")
    print(f"{'kerangka_terintegrasi':22s} -> {target.relative_to(ROOT)}  {image.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
