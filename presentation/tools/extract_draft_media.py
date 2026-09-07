"""Pull the domain-knowledge artwork out of Pak Toto's annotated draft deck.

The draft (``20260906 Sidang_Disertasi_Toto_Suharto.pptx`` plus its PDF
export, both gitignored) carries SKF training material that the reworked
deck rebuilds in the ITB design. Its pictures come in two kinds:

* ordinary PNG/JPG/GIF pictures — copied straight out of the PPTX (``blob``
  entries, keyed by slide number and media part name);
* WMF/EMF pictures and drawings made of native shapes (the vibration
  signature, the ADM hierarchy, the cost bars, the RUL curve) — cropped out
  of the PDF page rasterised at 220 dpi (``crop`` entries, keyed by slide
  number and a bounding box in inches on the 13,33 × 7,5 in canvas).

Every output is trimmed of white margins and written to ``assets/skf/``.
The PNGs are committed like ``assets/v14/``; this tool only needs to run
again when the draft changes.

Usage: python3 tools/extract_draft_media.py  (or ``make draft-assets``)
"""

from __future__ import annotations

import io
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "skf"
DRAFT_STEM = "20260906 Sidang_Disertasi_Toto_Suharto"
DPI = 220
PAGE_W_IN = 13.333

# name -> ("blob", slide, media part)  |  ("crop", slide, (x, y, w, h) in inches)
MAP: dict[str, tuple] = {
    # slide 5 · kasus ball mill
    "skf_ballmill_sketch.png": ("blob", 5, "image3.png"),
    # slide 6 · penghematan program keandalan (native bars)
    "skf_reliability_savings.png": ("crop", 6, (0.85, 1.65, 6.15, 5.05)),
    # slide 7 · kurva kegagalan fungsional
    "skf_functional_failure.png": ("blob", 7, "image5.png"),
    # slide 8-9 · Asset Diagnostic Methodology
    "skf_adm_hierarchy.png": ("crop", 9, (0.95, 1.50, 4.25, 5.55)),
    # slide 10 · peta ADM motor AC
    "skf_adm_ac_motor.png": ("crop", 10, (0.40, 1.60, 12.20, 5.40)),
    # slide 11 · vibration signature, static dan dynamic data
    "skf_vibration_signature.png": ("crop", 11, (3.05, 1.55, 8.45, 4.85)),
    # slide 12 · kurva prognostik RUL
    "skf_rul_timeline.png": ("crop", 12, (2.45, 1.70, 8.15, 5.00)),
    # slide 15 · kolase kerusakan bearing
    "skf_bearing_rusted.png": ("blob", 15, "image18.jpeg"),
    "skf_bearing_section.png": ("blob", 15, "image16.gif"),
    "skf_bearing_defect.png": ("blob", 15, "image17.gif"),
    "skf_bearing_stages.png": ("blob", 15, "image11.gif"),
    "skf_ge_spectrum.png": ("crop", 15, (5.65, 3.15, 4.40, 3.10)),
    # slide 17 · komponen, proses cutting, QA, produk
    "skf_components.png": ("crop", 17, (0.62, 1.53, 4.28, 2.41)),
    "skf_cutting_processes.png": ("crop", 17, (7.83, 1.28, 3.18, 3.06)),
    "skf_quality_assurance.png": ("crop", 17, (1.14, 4.62, 4.40, 2.06)),
    "skf_products.png": ("blob", 17, "image26.png"),
    # slide 23-24 · acceleration enveloping
    "skf_signal_overall_vs_bearing.png": ("crop", 23, (2.20, 2.40, 8.15, 3.85)),
    "skf_enveloping_steps.png": ("crop", 24, (2.44, 1.66, 7.56, 5.10)),
    # slide 25 · metode deteksi cacat
    "skf_bearing_photo.png": ("crop", 25, (0.40, 1.95, 5.30, 3.05)),
    "skf_defect_ringing.png": ("blob", 25, "image32.png"),
    "skf_defect_waveform.png": ("blob", 25, "image34.png"),
    # slide 26 · SKF acceleration enveloping (gE)
    "skf_ge_process.png": ("blob", 26, "image35.png"),
    "skf_ge_waveforms.png": ("crop", 26, (8.80, 1.40, 4.45, 5.25)),
    # slide 27 · frekuensi karakteristik bearing
    "skf_fault_freq_gauge.png": ("crop", 27, (1.32, 1.57, 4.01, 2.76)),
    # slide 31 · dari data menuju prediksi
    "skf_collect_to_prognose.png": ("blob", 31, "image40.png"),
    # slide 76 · dari tujuan pabrik ke rotasi yang andal
    "skf_plant_goals.png": ("blob", 76, "image69.png"),
}


def _trim(image: Image.Image, pad: int = 6) -> Image.Image:
    image = image.convert("RGB")
    background = Image.new("RGB", image.size, (255, 255, 255))
    bbox = ImageChops.difference(image, background).getbbox()
    if not bbox:
        return image
    return image.crop((max(bbox[0] - pad, 0), max(bbox[1] - pad, 0),
                       min(bbox[2] + pad, image.width), min(bbox[3] + pad, image.height)))


def _blob(archive: zipfile.ZipFile, slide: int, part: str) -> Image.Image:
    # The media part is global to the package; the slide number only documents
    # where the picture sits in the draft.
    data = archive.read(f"ppt/media/{part}")
    image = Image.open(io.BytesIO(data))
    image.seek(0)  # animated GIF -> first frame
    if image.mode in ("P", "RGBA", "LA"):
        rgba = image.convert("RGBA")
        flat = Image.new("RGB", rgba.size, (255, 255, 255))
        flat.paste(rgba, mask=rgba.getchannel("A"))
        return flat
    return image.convert("RGB")


def _page(pdf: Path, slide: int, cache: dict[int, Image.Image], workdir: Path) -> Image.Image:
    if slide not in cache:
        prefix = workdir / f"page{slide}"
        subprocess.run(
            ["pdftoppm", "-r", str(DPI), "-f", str(slide), "-l", str(slide), "-png",
             str(pdf), str(prefix)],
            check=True, stdout=subprocess.DEVNULL,
        )
        rendered = sorted(workdir.glob(f"page{slide}*.png"))
        cache[slide] = Image.open(rendered[0]).convert("RGB")
    return cache[slide]


def _crop(page: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    scale = page.width / PAGE_W_IN
    x, y, w, h = box
    return page.crop((round(x * scale), round(y * scale),
                      round((x + w) * scale), round((y + h) * scale)))


def main() -> int:
    pptx = ROOT / f"{DRAFT_STEM}.pptx"
    pdf = ROOT / f"{DRAFT_STEM}.pdf"
    if not pptx.exists() or not pdf.exists():
        print(f"draft deck not found under {ROOT} (need {DRAFT_STEM}.pptx and .pdf)",
              file=sys.stderr)
        return 1
    if not shutil.which("pdftoppm"):
        print("pdftoppm not found (brew install poppler)", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    cache: dict[int, Image.Image] = {}
    with zipfile.ZipFile(pptx) as archive, tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        for name, (mode, slide, ref) in MAP.items():
            if mode == "blob":
                image = _blob(archive, slide, ref)
            else:
                image = _crop(_page(pdf, slide, cache, workdir), ref)
            image = _trim(image)
            target = OUT / name
            image.save(target, format="PNG")
            print(f"slide {slide:>2} {mode:4} -> {target.relative_to(ROOT)}  "
                  f"{image.width}x{image.height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
