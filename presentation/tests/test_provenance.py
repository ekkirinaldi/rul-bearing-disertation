"""Every figure on a slide must be traceable to the manuscript.

The deck is a summary of the dissertation, so any number it shows has to
appear in the V14 manuscript DOCX at the top of `presentation/` — the
manuscript of record since August 2026. The older `dissertation-docx/` tree
is deliberately NOT part of the corpus: its numbers conflict with V14 in
places (XJTU-SY 15 rekaman vs 10 bearing, IMS dropped entirely, SKF moved
from Lampiran D into Subbab IV.15 / V.5.3), and a claim passing via the
stale tree would defeat the guard.

It is the automated form of the manual audit that caught, among others, a
wrong bearing count for XJTU-SY, a wrong citation for the 40–50% figure, a
wrong Critical threshold, and a set of benchmark findings that were never in
the manuscript at all.

Skips cleanly when the V14 DOCX is absent (it is a gitignored binary).
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
    # The newest version wins: V15 adds the domain material ported from this
    # deck, so it is a superset of V14 and every claim still has to hold.
    sources = sorted(ROOT.glob("V1? *.docx"))
    if not sources:
        pytest.skip(f"manuscript DOCX not found under {ROOT}")
    text = _docx_text(sources[-1])
    # The manuscript writes "17,6 %" and "1 024"; normalise so a slide that
    # writes "17,6%" still matches.
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(\d)\s+%", r"\1%", text)
    return text


def _spec_text() -> str:
    return (ROOT / "content" / "sidang-terbuka.yaml").read_text(encoding="utf-8")


# (claim, one-or-more accepted spellings) — grouped by the slide that shows it
CLAIMS: list[tuple[str, list[str]]] = [
    # s4 latar belakang
    ("40–50% kegagalan bearing", ["40–50%", "40–50 %"]),
    ("dikutip ke Nandi dkk. (2005)", ["(Nandi dkk., 2005)"]),
    ("PT SKF Indonesia", ["PT SKF Indonesia"]),
    # s7–s8 metodologi & dataset
    ("split temporal CWRU", ["54/13/33"]),
    ("CWRU 48 kHz", ["48 kHz"]),
    ("PHM2012 17 bearing", ["17 bearing"]),
    ("XJTU-SY 15 bearing", ["15 bearing"]),
    ("protokol 75 epoch", ["75 epoch"]),
    ("presisi bf16", ["bf16"]),
    ("seed 42, 43, 44", ["(42, 43, 44)"]),
    # s12 apa yang dibaca setiap model — the input contract on every diagram
    ("HI 36-D untuk backbone RUL", ["HI 36-D"]),
    ("window 64 rekaman pada PHM2012", ["64 rekaman"]),
    ("akselerometer prognostik 25,6 kHz", ["25,6 kHz"]),
    ("20.000 hidden state untuk SAE", ["20.000"]),
    ("hidden state 128 dimensi", ["d = 128"]),
    ("WDCNN raw signal 1 × 2.048", ["1 × 2.048"]),
    # s13 dari sinyal mentah ke feature vector HI
    ("18 feature per channel", ["18 fitur per channel"]),
    ("channel fan-end CWRU", ["fan-end"]),
    ("PSD Welch untuk feature domain frekuensi", ["PSD Welch"]),
    # s10 metrik diagnostik
    ("akurasi WDCNN 99,87%", ["99,87%"]),
    ("749 dari 750", ["749/750"]),
    ("satu-satunya salah kelas Ball_014 ke IR_014", ["Ball_014 diklasifikasi sebagai IR_014"]),
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
    ("bagging 100 tree", ["100 tree"]),
    ("WDCNN ±60.710 parameter", ["60.710"]),
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
    ("Mamba-xLSTM 811K parameter pada XJTU-SY", ["811"]),
    ("N-BEATS 459K parameter", ["459"]),
    ("N-BEATS stabilitas lintas seed 0,007", ["0,007"]),
    ("SparseGate-TCN 249K parameter", ["249"]),
    # structural facts drawn into the algorithm diagrams
    ("tiga pasang blok hibrid Mamba + mLSTM", ["tiga pasang"]),
    ("SAE ekspansi delapan kali", ["delapan kali ekspansi"]),
    ("dilatasi TCN 1-2-4-8", ["1, 2, 4, dan 8"]),
    ("WDCNN kernel pertama lebar 64 stride 16", ["stride 16"]),
    ("jendela streaming 64 akuisisi", ["jendela 64 akuisisi"]),
    ("basis blok trend wear shock", ["wavelet Gabor"]),
    ("quantile head median", ["quantile head"]),
    ("PHM2012 RMSE 0,226", ["0,226"]),
    ("XJTU-SY RMSE 0,213", ["0,213"]),
    ("runner-up XJTU 0,216", ["0,216"]),
    ("PHM Score SparseGate 0,904", ["0,904"]),
    ("PHM Score Mamba XJTU 0,938", ["0,938"]),
    # s17–s18 SAE-BPFx
    ("SAE k = 51", ["k = 51"]),
    ("SAE 20.000 hidden state", ["20.000 hidden state"]),
    ("pita integrasi ±2 Hz", ["±2 Hz"]),
    ("ambang hit |r| ≥ 0,30", ["≥ 0,30"]),
    ("enam uji primer, Bonferroni 0,008", ["enam uji primer"]),
    ("ambang Bonferroni p < 0,008", ["p < 0,008"]),
    ("variasi magnituda 4–8 kali lipat", ["4–8 kali lipat"]),
    ("hit-rate BPFI PHM2012 2,3%", ["2,3%"]),
    ("hit-rate BSF XJTU 1,6%", ["1,6%"]),
    ("corr scatter n = 304", ["n = 304"]),
    ("rmax 0,447", ["0,447"]),
    ("rmax BPFO XJTU 0,468", ["0,468"]),
    ("sparsity sweep k = 205", ["k = 205"]),
    ("sweep BPFI 0,68% ke 7,13%", ["menjadi 7,13%"]),
    ("sweep BPFO 0,59% ke 8,30%", ["menjadi 8,30%"]),
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
    ("pipeline HI 18 fitur", ["18 fitur"]),
    ("geometri bearing SKF 6205", ["SKF 6205"]),
    ("akselerometer SKF CMSS2200", ["CMSS2200"]),
    ("EoL terpaut sekitar satu hari", ["terpaut sekitar satu hari"]),
    ("status demonstrasi kualitatif", ["demonstrasi kualitatif"]),
    ("pseudo-waveform dari nilai tren", ["rekonstruksi sintetis dari nilai tren"]),
    ("jendela 64 akuisisi", ["jendela 64 akuisisi", "window 64"]),
    ("interval pengambilan satu jam", ["interval pengambilan satu jam"]),
    # cadangan: mesin inferensi benchmark (V.5.4)
    ("PHM2012 demo akuisisi ke-1.344", ["ke-1.344"]),
    ("PHM2012 demo prediksi 4 jam 5 menit", ["4 jam 5 menit"]),
    ("XJTU demo 2 jam 50 menit", ["2 jam 50 menit"]),
    ("XJTU aktual 3 jam 16 menit", ["3 jam 16 menit"]),
    ("fusion gate PHM 54% xLSTM", ["54%"]),
    # cadangan: rig PRONOSTIA
    ("beban radial 4.000–5.000 N", ["4.000–5.000 N"]),
    # objek penelitian (draft-derived slides, grounded in Subbab III.2)
    ("mesin grinding OR1 dan OR2", ["OR1 dan OR2"]),
    ("Channel 15 PT SKF Indonesia", ["Channel 15"]),
    ("spindel 3.500–4.200 rpm", ["3.500–4.200 rpm"]),
    # data mana yang dipakai (V.5.3)
    ("nilai tren A/V/ENV dari SKF Observer", ["percepatan (A), kecepatan (V), dan envelope (ENV)"]),
    ("pseudo-waveform per titik tren", ["pseudo-waveform"]),
    ("window XJTU-SY 32 rekaman", ["32 rekaman"]),
    # kriteria evaluasi (III.4.2)
    ("RMSE peka terhadap kesalahan besar", ["peka terhadap kesalahan besar"]),
    ("RMSE satu-satunya metrik adil lintas dataset", ["satu-satunya metrik yang dapat dibandingkan secara adil"]),
    # mengapa tiga backbone (II.5.2, V.1)
    ("Transformer kuadratik terhadap panjang sekuens", ["kompleksitasnya kuadratik"]),
    ("xLSTM exponential gating", ["exponential gating"]),
    ("tiga prinsip desain berbeda", ["Tiga arsitektur backbone dipilih untuk mewakili tiga prinsip desain"]),
    # explainability dan mengapa SAE (I.2, II.6)
    ("RM-3: bukan sekadar korelasi statistik", ["bukan sekadar mempelajari korelasi statistik"]),
    ("hipotesis superposisi", ["hipotesis superposisi"]),
    ("Top-k kendali langsung atas sparsity", ["Top-k memberikan kendali langsung"]),
    ("XAI input-level beroperasi di tingkat masukan", ["tingkat masukan"]),
    ("SAE prognostik bearing belum pernah dilaporkan", ["belum pernah dilaporkan"]),
    # blok dasar (II.5.2, V.1) dan tabel syarat SAE (II.6.1)
    ("Mamba memilih informasi yang dipertahankan, ditolak, atau dilewatkan", ["dipertahankan, ditolak, atau dilewatkan"]),
    ("perubahan kecil jauh di awal trajektori tetap berkontribusi", ["jauh di awal trajektori degradasi tetap berkontribusi"]),
    ("exponential gating menggantikan saturasi sigmoid", ["menggantikan saturasi sigmoid"]),
    ("matrix memory menyimpan asosiasi key-value", ["asosiasi key-value"]),
    ("mLSTM konteks multi-level", ["konteks multi-level"]),
    ("basis trend polinomial Bernstein", ["polinomial Bernstein"]),
    ("basis shock wavelet Gabor", ["wavelet Gabor"]),
    ("prior struktural mempersempit ruang solusi optimizer", ["ruang solusi yang perlu dijelajahi optimizer menjadi lebih sempit"]),
    ("laju dilatasi 1, 2, 4, dan 8", ["1, 2, 4, dan 8"]),
    ("TCN menangkap periodisitas pada frekuensi rekaman tetap", ["pola periodisitas pada frekuensi rekaman yang tetap"]),
    ("SHAP tidak beroperasi pada ruang laten", ["tidak beroperasi pada ruang representasi laten"]),
    ("Templeton: jutaan fitur pada skala besar", ["jutaan fitur yang dapat diinterpretasi"]),
]

# Numbers that are on a slide but deliberately NOT from the manuscript: domain
# anecdotes from SKF training material and one published industry survey. Each
# must be credited on the slide that shows it (a `source` block or an inline
# citation naming the source), so a reader can tell them from research claims.
EXTERNAL_CLAIMS: list[tuple[str, str]] = [
    # (text on the slide, source label that must appear on the same slide)
    ("26 minggu", "SKF Group (2017)"),
    ("$26 juta", "SKF Group (2017)"),
    ("$1,5 triliun", "Siemens, 2022"),
    ("25 jam per bulan", "Siemens, 2022"),
    ("$2 juta", "Siemens, 2022"),
    ("2,3 mm/s", "SKF Group (2017)"),
    ("4.000 byte", "SKF Group (2017)"),
]


def _slide_texts() -> list[str]:
    """The spec split per slide, so a claim and its credit can be matched."""
    return re.split(r"\n  - layout:", _spec_text())


@pytest.mark.parametrize("claim,spellings", CLAIMS, ids=[c for c, _ in CLAIMS])
def test_claim_appears_in_manuscript(manuscript, claim, spellings):
    assert any(s in manuscript for s in spellings), (
        f"{claim!r} is on a slide but not in dissertation-docx/ — "
        f"looked for {spellings}"
    )


def test_cited_figures_and_tables_exist_in_v14(manuscript):
    """Every figure or table number a slide cites must exist in V14."""
    spec = _spec_text()
    cited = set(re.findall(r"(Gambar [IVX]+\.\d+|Tabel [IVX]+\.\d+)", spec))
    missing = sorted(ref for ref in cited if ref not in manuscript)
    assert missing == [], f"cited on slides but not in the V14 manuscript: {missing}"


def test_no_stale_pre_v14_references():
    """V14 moved the SKF validation from Lampiran D into Subbab IV.15 / V.5.3."""
    spec = _spec_text()
    for stale in ("Lampiran D.5", "Gambar D.2", "Gambar D.4", "Gambar D.5"):
        assert stale not in spec, (
            f"{stale} is pre-V14 numbering; the SKF material is Subbab IV.15 / "
            f"V.5.3 and its panels are Gambar V.8/V.9"
        )


def test_ims_and_cwru_out_of_the_sae_section():
    """IMS was dropped from the manuscript and CWRU from the SAE hit-rate
    narrative; the deck must not reintroduce them, including through the
    full 2x2 figure artwork (use the crops from ``make derived-assets``)."""
    spec = _spec_text()
    assert not re.search(r"\bIMS\b", spec), "IMS is no longer part of the research"
    for artwork in ("bab5/hitrate_panel.png", "negative_controls.png",
                    "bab5/sparsity_sweep.png", "bab3/kerangka_terintegrasi.png"):
        assert artwork not in spec, (
            f"{artwork} still carries IMS/CWRU panels; reference the "
            f"assets/derived/ crop instead"
        )
    assert "underpowered" not in spec, "the CWRU-underpowered caveat left with CWRU"


@pytest.mark.parametrize("claim,source", EXTERNAL_CLAIMS, ids=[c for c, _ in EXTERNAL_CLAIMS])
def test_external_claim_is_credited_on_its_slide(claim, source):
    """A number outside V14 may only appear next to its credit."""
    slides = [s for s in _slide_texts() if claim in s]
    assert slides, f"{claim!r} is listed as an external claim but is on no slide"
    for slide in slides:
        assert source in slide, (
            f"{claim!r} is shown without its credit {source!r} on the same slide"
        )


def test_every_dollar_or_anecdote_number_is_enumerated():
    """Dollar amounts never come from V14; each must be in EXTERNAL_CLAIMS."""
    spec = _spec_text()
    listed = {c for c, _ in EXTERNAL_CLAIMS}
    for amount in re.findall(r"\$[\d.,]+ ?\w*", spec):
        assert any(amount.startswith(c) for c in listed), (
            f"{amount!r} is on a slide but not enumerated in EXTERNAL_CLAIMS"
        )
