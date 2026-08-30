"""Every figure on a slide must be traceable to the manuscript.

The deck is a summary of the dissertation, so any number it shows has to
appear in `dissertation-docx/` (the manuscript of record). This test reads
the DOCX chapters and lampiran and asserts each headline claim is present.

It is the automated form of the manual audit that caught, among others, a
wrong bearing count for XJTU-SY, a wrong citation for the 40–50% figure, a
wrong Critical threshold, and a set of benchmark findings that were never in
the manuscript at all.

Skips cleanly when the manuscript tree is absent (e.g. a shallow checkout).
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MANUSCRIPT = ROOT.parent / "dissertation-docx"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    return "\n".join(
        "".join(node.text or "" for node in para.iter(W + "t"))
        for para in root.iter(W + "p")
    )


@pytest.fixture(scope="module")
def manuscript() -> str:
    sources = (sorted(MANUSCRIPT.glob("chapters/*.docx"))
               + sorted(MANUSCRIPT.glob("lampiran/*.docx"))
               + sorted(MANUSCRIPT.glob("frontmatter/*.docx")))
    if not sources:
        pytest.skip(f"manuscript not found under {MANUSCRIPT}")
    text = "\n".join(_docx_text(p) for p in sources)
    # The manuscript writes "17,6 %" and "1 024"; normalise so a slide that
    # writes "17,6%" still matches.
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(\d)\s+%", r"\1%", text)
    return text


# (claim, one-or-more accepted spellings) — grouped by the slide that shows it
CLAIMS: list[tuple[str, list[str]]] = [
    # s4 latar belakang
    ("40–50% kegagalan bearing", ["40–50%"]),
    ("dikutip ke Lei (2018)", ["(Lei, 2018)"]),
    ("PT SKF Cikarang", ["Cikarang"]),
    # s7–s8 metodologi & dataset
    ("split temporal CWRU", ["54/13/33"]),
    ("CWRU 48 kHz", ["48 kHz"]),
    ("PHM2012 17 bearing", ["17 bearing"]),
    ("XJTU-SY 10 bearing", ["10 bearing"]),
    ("IMS empat bearing Rexnord", ["empat bearing Rexnord"]),
    ("IMS ZA-2115", ["ZA-2115"]),
    ("IMS dikutip ke Qiu dkk. (2006)", ["Qiu dkk. (2006)"]),
    ("protokol 75 epoch", ["75 epoch"]),
    ("presisi bf16", ["bf16"]),
    ("seed 42, 43, 44", ["(42, 43, 44)"]),
    # s10 metrik diagnostik
    ("akurasi WDCNN 99,87%", ["99,87%"]),
    ("749 dari 750", ["749 dari 750"]),
    ("macro F1 0,997", ["0,997"]),
    ("split-half 0,940", ["0,940"]),
    ("severity Ball 17,6%", ["17,6%"]),
    ("severity IR 13,8%", ["13,8%"]),
    ("severity OR 8,6%", ["8,6%"]),
    ("discriminability 0,216", ["0,216"]),
    ("discriminability 0,735", ["0,735"]),
    # s12 benchmark
    ("SVM-RBF 96,4%", ["96,4%"]),
    ("LR-OvR 94,3%", ["94,3%"]),
    ("Random Forest 95,8%", ["95,8%"]),
    ("XGBoost 97,1%", ["97,1%"]),
    ("Decision Tree 92,40%", ["92,40%"]),
    ("selisih DT→RF 3,4 pp", ["3,4 poin persentas"]),
    ("XGBoost unggul 1,3 pp", ["1,3 poin persentase"]),
    ("SVM unggul 2,1 pp atas LR", ["2,1 poin persentase"]),
    ("ruang fitur HI 36-D", ["HI 36-D"]),
    # s13 FSM
    ("300 sampel background", ["300 sampel"]),
    ("500 sampel uji", ["500 sampel"]),
    ("jendela reseptif 42,67 ms", ["42,67 ms"]),
    # s15 ablasi BatchNorm
    ("varian A 99,73%", ["99,73%"]),
    ("varian A diskriminabilitas 0,315", ["0,315"]),
    # 96,13% is 99,73 - 3,60 pp; the manuscript states the delta, not the absolute
    ("varian C turun 3,60 pp", ["3,60 poin persentase"]),
    ("protokol ablasi 50 epoch", ["protokol ablasi 50 epoch"]),
    ("subset ablasi 30/50", ["30 background / 50 sampel uji"]),
    ("kenaikan 133%", ["133%", "133 %"]),
    # s16 backbone RUL
    ("Mamba-xLSTM 898K parameter", ["898"]),
    ("N-BEATS 459K parameter", ["459"]),
    ("SparseGate-TCN 249K parameter", ["249"]),
    ("PHM2012 RMSE 0,226", ["0,226"]),
    ("XJTU-SY RMSE 0,213", ["0,213"]),
    ("IMS RMSE 0,407", ["0,407"]),
    ("runner-up XJTU 0,216", ["0,216"]),
    # s17–s18 SAE-BPFx
    ("SAE k = 51", ["k = 51"]),
    ("hit-rate BPFI PHM2012 2,3%", ["2,3%"]),
    ("hit-rate BSF XJTU 1,6%", ["1,6%"]),
    ("rmax 0,447", ["0,447"]),
    ("ambang Bonferroni 0,004", ["0,004"]),
    ("IMS BPFI 1,76%", ["1,76%"]),
    ("IMS BSF 0,49%", ["0,49%"]),
    ("CWRU 5,08%", ["5,08%"]),
    ("FTF naik ke 7,81%", ["7,81%"]),
    ("sparsity sweep k = 205", ["205"]),
    # s19 validasi SKF
    ("NDE 158 akuisisi", ["158 akuisisi"]),
    ("NDE enam hari 13 jam", ["enam hari 13 jam"]),
    ("DE 78 akuisisi", ["78 akuisisi"]),
    ("DE tiga hari lima jam", ["tiga hari lima jam"]),
    ("RUL akhir NDE 10,3%", ["10,3%"]),
    ("RUL akhir DE 3,1%", ["3,1%"]),
    ("ambang kritis 20%", ["ambang kritis 20%"]),
    ("ambang peringatan 40%", ["40% dan 20%"]),
    ("EoL NDE 31 Agustus 2023 07.34", ["31 Agustus 2023 pukul 07.34"]),
    ("EoL DE 1 September 2023 07.23", ["1 September 2023 pukul 07.23"]),
    ("sisa umur NDE 34 menit", ["34 menit"]),
    ("sisa umur DE 40 menit", ["40 menit"]),
    ("gerbang fusi DE xLSTM 47% : Mamba 53%", ["xLSTM 47% berbanding Mamba 53%"]),
    ("atribusi NDE skewness 26,5%", ["26,5%"]),
    ("atribusi DE entropy 39,6%", ["39,6%"]),
    ("pipeline HI 18-D", ["HI 18-D"]),
    ("EoL ditambatkan ke kegagalan lapangan", ["ditambatkan pada peristiwa kegagalan lapangan"]),
    ("jendela 64 akuisisi", ["jendela 64 akuisisi"]),
]


@pytest.mark.parametrize("claim,spellings", CLAIMS, ids=[c for c, _ in CLAIMS])
def test_claim_appears_in_manuscript(manuscript, claim, spellings):
    assert any(s in manuscript for s in spellings), (
        f"{claim!r} is on a slide but not in dissertation-docx/ — "
        f"looked for {spellings}"
    )


def test_skf_figures_are_cited_to_lampiran_d():
    """The SKF streaming validation lives in Lampiran D.5, not Bab V."""
    spec = (ROOT / "content" / "sidang-terbuka.yaml").read_text(encoding="utf-8")
    for bad in ("Gambar V.3", "Gambar V.4", "Gambar V.5", "Gambar V.6"):
        assert bad not in spec, f"{bad} does not exist; SKF panels are Gambar D.2–D.5"
