# Plan — Modifikasi dan Pengembangan Disertasi LaTeX (Fase A–H)

## Context

`dissertation-outline.md` sudah selesai diperbarui menjadi single source of truth dengan struktur 6-bab baru. `references.bib` sudah diperkaya dengan 102 entri (36 baru dari Paper/). Sekarang giliran **LaTeX manuscript** yang harus diselaraskan dengan outline tersebut.

**Masalah inti:** File LaTeX saat ini masih mengikuti struktur LAMA (Dasar Teori → Metodologi → Hasil & Pembahasan), sedangkan outline baru menuntut struktur BARU yang menggabungkan metodologi+hasil per jalur penelitian.

**Target:** Enam chapter LaTeX yang sesuai dengan outline, dapat dikompilasi penuh, dan lulus `make pre-submit`.

---

## Peta Transformasi File

| Lama (ada) | Baris | → | Baru (target) | Baris est. |
|---|---|---|---|---|
| `01-pendahuluan.tex` | 483 | → | `01-pendahuluan.tex` (rewrite) | ~550 |
| `02-tinjauan-pustaka.tex` | 846 | → | `02-tinjauan-pustaka.tex` (augment) | ~1.100 |
| `03-dasar-teori.tex` | 1.023 | → | `03-metodologi-umum.tex` (new) | ~500 |
| `04-metodologi.tex` | 1.163 | → | `04-diagnostik.tex` (new) | ~900 |
| `05-hasil-pembahasan.tex` | 680 | → | `05-prognostik.tex` (new) | ~700 |
| `06-kesimpulan.tex` | 249 | → | `06-kesimpulan.tex` (rewrite) | ~350 |
| *(tidak ada)* | — | → | `lampiran/E-svm-lr.tex` (new) | ~150 |
| *(tidak ada)* | — | → | `lampiran/F-tree.tex` (new) | ~200 |
| *(tidak ada)* | — | → | `lampiran/G-wdcnn-fsm.tex` (new) | ~250 |
| `lampiran/D-klasifikasi-industri.tex` | 137 | → | (soften, same file) | ~120 |
| `disertasi.tex` | 109 | → | (update \input references) | ~115 |

**File lama `03-dasar-teori.tex` dan `04-metodologi.tex` dan `05-hasil-pembahasan.tex` TETAP ADA** sebagai backup sampai fase terakhir — tidak dihapus dulu.

---

## Prinsip Eksekusi

1. **Satu fase = satu commit.** Setiap fase menghasilkan build yang berhasil sebelum dilanjutkan.
2. **Baca outline sebelum edit.** Selalu re-read `dissertation-outline.md` bagian relevan sebelum menulis bab.
3. **Bahasa:** Bahasa Indonesia baku. Istilah asing di-`\emph{}`. Tidak ada kata ganti orang pertama (`saya`, `kami`, `kita`) di luar kata pengantar.
4. **Sitasi diri:** gunakan `\citetitb{TotoSuharto2024Conf1SVM}` dst., tidak pernah `\cite{}`.
5. **Gambar:** Buat placeholder `\missingfigure{...}` (paket todonotes) atau `\fbox{\parbox{...}{[Figure: ...]}}` jika gambar belum tersedia. Jangan tunda penulisan karena menunggu gambar.
6. **Verifikasi:** `make build` setelah setiap fase. `make lint` wajib lulus sebelum commit. `make pre-submit` wajib lulus di akhir.
7. **Default build path: Docker.** Jalankan target Makefile melalui Docker (`danteev/texlive`) sebagai jalur referensi — distribusi TeX Live ter-pin, hasil reproducible lintas mesin. Local `make` boleh dipakai untuk iterasi cepat, tetapi sebelum commit yang dikirim ke Pak Toto **wajib** re-verify via Docker:
   ```bash
   cd writings/disertation
   docker run --rm -v "$(pwd):/workdir" -w /workdir danteev/texlive make build
   docker run --rm -v "$(pwd):/workdir" -w /workdir danteev/texlive make check
   docker run --rm -v "$(pwd):/workdir" -w /workdir danteev/texlive make pre-submit
   ```
   Verifikasi PowerShell di tiap fase dapat menggunakan local `make` jika tersedia — perintah Docker setara dengan mengganti `make X` → `docker run --rm -v "${PWD}:/workdir" -w /workdir danteev/texlive make X`.

---

## Fase A — Bab I: Pendahuluan (`01-pendahuluan.tex`)

**Sumber utama:** `dissertation-outline.md §Bab I` | `writings/SK-Toto.pdf` (sudah dibaca)

### Perubahan Struktur

**Saat ini (9 seksi):**
```
Latar Belakang | Masalah Penelitian | Tujuan Penelitian |
Lingkup & Batasan | Asumsi | Hipotesis | Pendekatan & Metodologi |
Kebaruan & Orisinalitas | Sistematika
```

**Target (7 seksi per outline §1.1–1.7):**
```
§1.1 Latar Belakang
§1.2 Rumusan Masalah (3 RQ)
§1.3 Tujuan Penelitian dan Novelti (3 Tujuan → N1–N5)
§1.4 Manfaat Penelitian (3 manfaat, dilembut)
§1.5 Batasan dan Asumsi Penelitian
§1.6 Kontribusi dan Posisi Penelitian
§1.7 Sistematika Penulisan (jelaskan 6-bab baru)
```

### Perubahan Konten Per Seksi

**§1.1 Latar Belakang (REWRITE PARTIAL):**
- Pertahankan: konteks Making Indonesia 4.0, bearings 40–50% kegagalan, evolusi PdM
- Tambahkan: dua gap kritis (signal-level XAI untuk WDCNN belum ada; model RUL belum terbukti internalisasi BPFx)
- Tambahkan: motivasi integrasi data kualitas radial clearance sebagai **motivasi industri jangka panjang** (PT~SKF IMx-8), **BUKAN sebagai RQ**
- Hapus: klaim langsung bahwa penelitian ini mengintegrasikan data kualitas (belum dikerjakan)

**§1.2 Rumusan Masalah (REWRITE — pemetaan ulang ke RQ level industri SK-Toto + sub-RQ teknis):**
- Hapus: RQ lama yang terlalu teknis di level atas
- Tulis **3 RQ level industri + 5 sub-RQ teknis** per outline §1.2:
  - **RQ1 — Integrasi Data Sensor & Arsitektur PdM untuk OEE.** Bagaimana sistem PdM berbasis vibrasi pada arsitektur multi-tier (Edge IoT → Edge Server → Cloud) menjadi fondasi untuk mengurangi *unplanned downtime* + mendukung peningkatan OEE pada produksi ball bearing, dengan integrasi data kualitas heterogen sebagai roadmap penelitian lanjutan?
    - **Sub-RQ1.1:** Pembagian beban komputasi (triage anomali / klasifikasi jenis kerusakan / estimasi RUL) ke tiga tier — keluaran tier-N dipakai tier-(N+1)?
    - *Catatan transparansi:* Integrasi penuh data kualitas *radial clearance* **tidak dijawab empiris**; hanya motivasi industri (§1.1) + future work (§VI.5). Klaim OEE = arsitektural (blueprint), bukan kuantitatif lapangan.
  - **RQ2 — Pendekatan Hybrid ML+DL untuk Deteksi & RUL.** Apakah pendekatan hybrid (kernel SVM/LR + tree DT/RF/XGB + deep WDCNN untuk diagnostik; Mamba-xLSTM / N-BEATS-xLSTM / SparseGate-TCN untuk prognostik) memberikan deteksi anomali + estimasi RUL yang konsisten lintas keluarga + lintas dataset (CWRU/PHM2012/XJTU-SY/IMS)?
    - **Sub-RQ2.1 (Diagnostik):** konsistensi akurasi kernel/tree/deep di CWRU + posisi terbaik trade-off antara akurasi dan kompleksitas untuk *edge deployment*.
    - **Sub-RQ2.2 (Prognostik):** konsistensi RUL 3 backbone lintas PHM2012/XJTU-SY/IMS dengan protokol bersama (3 seed, 75 epoch, bf16, batch 512 × window 32) + inferensi statistik (bootstrap CI + permutation test).
    - *Honest scoping:* "Hybrid" = lintas keluarga **paralel**, bukan ensembling antar keluarga ke prediktor tunggal.
  - **RQ3 — Transparansi Model melalui XAI untuk Adopsi Industri.** Bagaimana SHAP (Kernel/Tree/Deep Explainer) untuk diagnostik + Top-k SAE + pemetaan BPFx untuk prognostik dapat meningkatkan transparansi + auditability model PdM?
    - **Sub-RQ3.1 (Sinyal-level diagnostik):** apakah pola SHAP DeepExplainer WDCNN dapat dipetakan ke morfologi impuls / periodisitas BPFx + cukup stabil + diskriminatif membentuk FSM per kelas?
    - **Sub-RQ3.2 (Latent-level prognostik):** apakah Top-k SAE pada hidden states backbone RUL berkorespondensi dengan BPFO/BPFI/BSF/FTF secara statistik signifikan (bootstrap CI + permutation test + 2 negative controls Xavier-init + Gaussian noise)?

**§1.3 Tujuan Penelitian dan Novelti (REWRITE — pemetaan baru Tujuan ↔ RQ ↔ Novelti):**
- **3 Tujuan SK-Toto** dipetakan **1:1 ke 3 RQ §1.2** dan ke **5 Novelti N1–N5**:
  - **Tujuan 1 (menjawab RQ1) — Pengembangan Sistem PdM Multi-Tier.** Blueprint arsitektur 3-tier yang menyatukan diagnostik + prognostik + XAI dua-lapis. → **N5 — Kerangka Konseptual PdM Multi-Tier**. *Bukti pendukung:* tabel penempatan model per tier (parameter count, latency estimasi, tipe XAI yang sesuai).
  - **Tujuan 2 (menjawab RQ2) — Validasi Konsistensi Hybrid ML+DL Lintas Dataset.** 3 keluarga diagnostik × CWRU + 3 backbone prognostik × 3 dataset RUL + inferensi statistik. → **N4 — Universalitas + Statistical Rigor**. *Bukti:* tabel RMSE 3 backbone × 3 dataset + konvergensi BPFx-dominant lintas arsitektur + Control 1/2 hit-rate jatuh ke ~0. *Honest:* integrasi multi-modal *radial clearance* tetap di luar lingkup empiris → future work.
  - **Tujuan 3 (menjawab RQ3) — Transparansi Model melalui XAI.** SHAP per keluarga (diagnostik) + Top-k SAE + BPFx mapping (prognostik). → **N1 (FSM) + N2 (BatchNorm trade-off) + N3 (SAEBearing)** — tiga novelti empiris berlapis. *Bukti:* split-half stability 0,940; discriminability +133% pada varian tanpa BN; hit-rate BPFI 2,3% p<0,001; $r_\text{max}$ 0,507.
- **Status pembuktian eksplisit (anti-overclaim, sesuai bimbingan):**
  - **N1, N2, N3, N4 = empiris** — terukur kuantitatif dengan angka kunci di Bab IV–V.
  - **N5 = konseptual/sintesis** — *blueprint* arsitektur, **bukan** klaim hasil *deployment* lapangan. Wajib ditegaskan di teks Bab I + Bab VI agar tidak overclaim.

**§1.4 Manfaat Penelitian (NEW — tidak ada di struktur lama):**
- 3 manfaat dari SK-Toto
- Integrasi data kualitas radial clearance → "aspirasi/future work"
- Manfaat langsung: (a) kerangka XAI dua-lapis, (b) sistem PdM edge-to-cloud, (c) dasar ilmiah untuk domain non-permesinan

**§1.5 Batasan dan Asumsi (MERGE dari dua seksi lama):**
- Batasan: CWRU Load 0–3, 48 kHz, 10 kelas, segmen 2.048; PHM2012/XJTU-SY/IMS untuk prognostik; arsitektur WDCNN + 3 backbone RUL
- Asumsi: 6 item dari SK-Toto (sensor terkalibrasi, data tersedia, dll.)

**§1.6 Kontribusi dan Posisi Penelitian (REWRITE dari §Kebaruan):**
- 4 kontribusi (3 kontribusi SK-Toto + 1 kontribusi tambahan untuk multi-family hybrid), dengan **kolom "RQ yang dijawab"** eksplisit:
  - K1: Kerangka FSM (N1+N2) → menjawab RQ3 / Sub-RQ3.1 [Empiris]
  - K2: SAEBearing (N3+N4) → menjawab RQ2 / Sub-RQ2.2 + RQ3 / Sub-RQ3.2 [Empiris]
  - K3: Integrasi PdM Multi-Tier (N5) → menjawab RQ1 / Sub-RQ1.1 [Konseptual]
  - K4: Konsistensi Hybrid ML+DL Lintas Keluarga (mendukung N4) → menjawab RQ2 / Sub-RQ2.1 + Sub-RQ2.2 [Empiris]
- Tabel `tab:bab1_posisi_penelitian` (Tabel I.1): baris = pendekatan (kernel/tree/deep/RUL/SAE), kolom = (Sumber Data / Mode Operasi / Pendekatan Model / Level XAI / **FSM** / **SAE-BPFx** / **Tier PdM**)
- Diagram TikZ `fig:bab1_kontribusi`: 5 kotak N1–N5 dengan panah dependency:
  - N1↔N2 [Jalur A, Sub-RQ3.1]
  - N3↔N4 [Jalur B, Sub-RQ2.2 + Sub-RQ3.2]
  - N5 = konvergensi [PdM Multi-Tier, Sub-RQ1.1]

**§1.7 Sistematika Penulisan (REWRITE):**
- Jelaskan 6-bab baru:
  - Bab III = fondasi bersama (dataset, praproses, metrik)
  - Bab IV = metodologi + hasil diagnostik (kernel→tree→deep→FSM)
  - Bab V = metodologi + hasil prognostik (RUL+SAE-BPFx)
  - Bab VI = sintesis PdM multi-tier + interpretabilitas dua-lapis

### Verifikasi Fase A
```powershell
cd writings\disertation
make build   # harus berhasil
make lint    # tidak ada [FATAL]
Select-String -Path "chapters\01-pendahuluan.tex" -Pattern "^\\\\section"
# Harus muncul: §I.1 s.d §I.7 (7 seksi)
```

---

## Fase B — Bab II: Tinjauan Pustaka (`02-tinjauan-pustaka.tex`)

**Sumber utama:** `dissertation-outline.md §Bab II` | Paper/Conference1, Conference2, Journal1 (sudah dibaca)

### Struktur Target (per outline §2.1–2.7)

```
§2.1 Perawatan Prediktif dan Industri 4.0
§2.2 Diagnostik Bearing Berbasis Deep Learning
  §2.2.1 Evolusi Arsitektur (SVM → CNN1D → WDCNN → hybrid → Transformer)
  §2.2.2 Dataset CWRU dan Isu Temporal Leakage
  §2.2.3 [NEW] Klasifikasi Bantalan Multi-Keluarga: Kernel, Tree, Deep
§2.3 Explainable AI: Input Attribution
  §2.3.1 Taksonomi XAI (gradient/propagation/perturbation)
  §2.3.2 SHAP DeepExplainer
  §2.3.3 [NEW] SHAP Kernel/Tree/Deep Explainer — Kapan Digunakan
  §2.3.4 [NEW] Fault Signature Maps (FSM) — pengenalan
§2.4 Prognostik Bearing: Estimasi RUL
  §2.4.1 Fisika dan Data-Driven
  §2.4.2 Benchmark Dataset RUL (PHM2012 / XJTU-SY / IMS)
§2.5 Mechanistic Interpretability dan Sparse Autoencoders
  §2.5.1 Superposition dan Top-k SAE
  §2.5.2 [NEW] Gap: Belum Ada SAE untuk Bearing
§2.6 [NEW] Fisika Getaran Bearing
  (BPFO/BPFI/BSF/FTF formulas + Hilbert envelope)
§2.7 [NEW] Peta Gap Literatur
  (Tabel matriks + posisi 4 paper Pak Toto)
```

### Pekerjaan Spesifik

**§2.2.3 — BARU (±80 baris):**
- Survey ringkas 3 keluarga: Kernel (SVM/LR), Tree (DT/RF/XGB), Deep (CNN1D, WDCNN)
- Tabel `tab:bab2_klasifikasi_survey` — kolom: Algoritma / Dataset / Akurasi / SHAP Variant / FSM
- Sitasi diri: `\citetitb{TotoSuharto2024Conf1SVM}`, `\citetitb{TotoSuharto2024Conf2Tree}`, `\citetitb{TotoSuharto2025Journal1FSM}`
- Sitasi eksternal baru: `\citetitb{Guo2016ADCNN}`, `\citetitb{Wen2020ResNet50}`, `\citetitb{Neupane2020CWRUReview}`

**§2.3.3 — BARU (±50 baris):**
- Tabel perbandingan 3 explainer: KernelExplainer (model-agnostic, mahal, untuk SVM/LR), TreeExplainer (eksak Lundberg, untuk DT/RF/XGB), DeepExplainer (DeepLIFT-based, untuk WDCNN)
- Sitasi: `\citetitb{Shrikumar2017DeepLIFT}`, `\citetitb{Ribeiro2016LIME}`, `\citetitb{Mosca2022SHAPNLP}`

**§2.3.4 — BARU (±70 baris):**
- Pengenalan FSM sebagai kontribusi Journal 1
- 3 varian FSM (Signed/Absolute/Variance) dengan formula ringkas
- 3 metrik validasi (discriminability, severity monotonicity, split-half stability) — definisi informal
- Sitasi: `\citetitb{TotoSuharto2025Journal1FSM}`

**§2.4.2 — AUGMENT (tambahkan IMS dataset):**
- Paragraf pendek tentang IMS dataset (4 Rexnord bearing, 2.000 rpm)
- Update tabel benchmark dataset RUL untuk memasukkan IMS

**§2.5.2 — BARU (±60 baris):**
- Bearing sebagai test case lebih bersih dari LLM (domain terbatas, BPFx = konsep fisika well-defined)
- Gap: belum ada studi yang memetakan SAE features ke BPFx secara statistik
- Posisi Journal 2 sebagai studi pertama
- Sitasi: `\citetitb{Elhage2022Superposition}`, `\citetitb{Cunningham2023TopkSAE}`, `\citetitb{Makhzani2014kSAE}`

**§2.6 — BARU (±100 baris):**
- Geometri bearing dan formula frekuensi karakteristik:
  - BPFO = (n/2) · f_r · (1 - d·cos(α)/D)
  - BPFI = (n/2) · f_r · (1 + d·cos(α)/D)
  - BSF = (D/(2d)) · f_r · (1 - (d·cos(α)/D)²)
  - FTF = (f_r/2) · (1 - d·cos(α)/D)
- Hilbert transform + envelope spectrum (formula)
- Spectral kurtosis — pengenalan singkat
- Sitasi: `\citetitb{McFadden1984BearingModel}`, `\citetitb{Antoni2007Kurtogram}`, `\citetitb{Randall2011CondMonitoring}`

**§2.7 — BARU (±80 baris):**
- Tabel `tab:bab2_posisi_penelitian` — baris = aspek (Level XAI, Dataset, Validasi Statistik, Integrasi Diag-Prog), kolom = Conf1 / Conf2 / Journal1 / Journal2 / literatur umum
- 3 gap utama (signal-level XAI, latent-BPFx, multi-tier integrasi)
- Paragraf sintesis yang menghubungkan ke §1.2 RQ1–RQ3

**Konten Lama yang DIPINDAHKAN:**
- §"Klasifikasi Kondisi Bantalan dan Studi Kasus Industri" (current Bab II §2.5) → diintegrasikan ke §2.2.3 baru
- §"Research Gap dan Posisi Penelitian" (current akhir Bab II) → digantikan §2.7 baru

### Verifikasi Fase B
```powershell
make build
make lint
Select-String -Path "chapters\02-tinjauan-pustaka.tex" -Pattern "^\\\\subsection" | Measure-Object
# Target: ≥ 14 subsections (7 sections masing-masing ≥ 2 sub)
```

---

## Fase C — Bab III: Metodologi Umum (BUAT BARU: `03-metodologi-umum.tex`)

**Sumber materi:**
- `04-metodologi.tex §Kerangka Penelitian` → §3.1
- `04-metodologi.tex §Dataset` (PHM2012 + XJTU-SY + TAMBAH IMS + CWRU) → §3.2
- `04-metodologi.tex §Praproses + Windowing` → §3.3
- `03-dasar-teori.tex §Metrik Evaluasi` → §3.4
- `04-metodologi.tex §Protokol Pelatihan + Infrastruktur` → §3.5

### Struktur File Baru

```latex
% chapters/03-metodologi-umum.tex
\chapter{Metodologi Umum}  % Bab III

\section{Kerangka Konseptual Terintegrasi}    % §III.1
\section{Dataset}                              % §III.2
  \subsection{CWRU}                            % §III.2.1
  \subsection{PHM2012 (FEMTO-PRONOSTIA)}       % §III.2.2
  \subsection{XJTU-SY}                         % §III.2.3
  \subsection{IMS}                             % §III.2.4
\section{Praproses dan Ekstraksi Fitur HI 36-D} % §III.3
\section{Metrik Evaluasi Bersama}              % §III.4
  \subsection{Metrik Diagnostik}
  \subsection{Metrik Prognostik}
  \subsection{Metrik Interpretabilitas}
\section{Infrastruktur dan Protokol Reproduksi} % §III.5
```

### Pekerjaan Spesifik

**§3.1 — ADAPTASI dari §Kerangka Penelitian ch04 (±60 baris):**
- Diagram dua jalur paralel (Jalur A: CWRU→WDCNN→FSM; Jalur B: PHM2012/XJTU-SY/IMS→BackboneRUL→SAE-BPFx)
- `\begin{figure}...\label{fig:bab3_kerangka_terintegrasi}` — gunakan TikZ atau placeholder
- Narasi: mengapa Bab III menjadi fondasi bersama untuk Bab IV dan V

**§3.2 — ADAPTASI + TAMBAH IMS (±150 baris):**
- §3.2.1 CWRU: motor 2 HP, SKF 6205-2RS JEM, 10 kelas, 48 kHz, segmen 2.048, split temporal 54/13/33% = 1.240/310/750
  - Tabel `tab:bab3_cwru_kelas` — 10 kelas (Normal, IR/OR/Ball × 0.007/0.014/0.021)
  - Catatan: CWRU dipakai Bab IV (diagnostik) dan Bab V §5.8 (cross-dataset SAE, n=10, underpowered)
- §3.2.2 PHM2012: FEMTO-PRONOSTIA, 17 bearing, 3 kondisi beban, EOL RMS > 20 g, label linier 1→0
  - Tabel geometri bearing FEMTO
- §3.2.3 XJTU-SY: 10 bearing LDK UER204, outer-race spalling dominan
- §3.2.4 IMS: 4 Rexnord bearing, 2.000 rpm, akuisisi 10 menit, Bearing 3 outer-race + Bearing 4 rolling-element
- Sitasi: `\citetitb{Nectoux2012PRONOSTIA}`, `\citetitb{Wang2020XJTU}`, `\citetitb{Smith2015CWRU}`

**§3.3 — ADAPTASI dari ch04 §Praproses (±80 baris):**
- Pipeline bersama: segmentasi → normalisasi Z-score → windowing Hann (untuk FFT)
- Health Indicator 36-D: 9 fitur time-domain + 9 fitur freq-domain + 5 band-energy × 2 kanal
- Catatan: HI 36-D hanya untuk keluarga kernel/tree (Bab IV.2–IV.3); backbone deep beroperasi pada sinyal mentah
- Rujuk Lampiran A untuk definisi lengkap

**§3.4 — ADAPTASI dari ch03 §Metrik Evaluasi (±100 baris):**
- Diagnostik: akurasi, macro-F1, weighted-F1, confusion matrix
- FSM: discriminability index, severity monotonicity fraction, split-half stability coefficient (formula masing-masing)
- Prognostik: RMSE, MAE, PHM score (formula penalti asimetris: late worse than early)
- Interpretabilitas SAE: hit-rate H, bootstrap 95% CI, permutation p-value (formula masing-masing)

**§3.5 — ADAPTASI dari ch04 §Infrastruktur (±50 baris):**
- GPU NVIDIA A40, seed {42, 43, 44}, bf16 untuk prognostik, FP32 untuk klasifikasi
- Repositori publik: GitHub link dari Lampiran C
- Checkpoint strategi: best validation loss; early stopping patience 15 (WDCNN), fixed 75 epoch (RUL)

### Update `disertasi.tex` (Fase C)
```latex
% Ganti:
\input{chapters/03-dasar-teori.tex}
% Dengan:
\input{chapters/03-metodologi-umum.tex}
% Tambahkan graphicspath baru:
\graphicspath{...{figures/bab3/}...}
```

**PENTING:** File `03-dasar-teori.tex` JANGAN dihapus — masih berisi konten theory yang akan dipakai Fase D dan E.

### Verifikasi Fase C
```powershell
make build   # menggunakan file baru
make lint
Select-String -Path "chapters\03-metodologi-umum.tex" -Pattern "^\\\\section" | Measure-Object
# Harus muncul 5 sections
```

---

## Fase D — Bab IV: Diagnostik (BUAT BARU: `04-diagnostik.tex`)

**Sumber materi:**
- `03-dasar-teori.tex §Klasifikasi Kondisi Bantalan` → §IV metodologi klasifikasi
- `03-dasar-teori.tex §SHAP` → §IV.5 teori SHAP per keluarga
- `04-metodologi.tex §Pilar 0` → kernel/tree/deep methodology
- `05-hasil-pembahasan.tex §Hasil Studi Kasus Industri (Pilar 0)` → WDCNN results
- **NEW content:** §IV.2 kernel (Conf1), §IV.3 tree (Conf2) — tulis dari paper content
- **NEW content:** §IV.6 FSM formalisasi, §IV.10–IV.13 — tulis dari Journal 1

### Struktur File Baru

```
% chapters/04-diagnostik.tex
\chapter{Bearing Fault Detection: Metodologi dan Hasil}  % Bab IV

[METODOLOGI]
§IV.1 Pipeline Diagnostik CWRU — Overview
§IV.2 Benchmark Keluarga Kernel: SVM dan Logistic Regression
§IV.3 Benchmark Keluarga Tree: DT, RF, XGBoost
§IV.4 Backbone Deep: WDCNN
§IV.5 SHAP per Keluarga
§IV.6 Formalisasi Fault Signature Maps (FSM)

[HASIL DAN ANALISIS]
§IV.7 Hasil Benchmark Kernel
§IV.8 Hasil Benchmark Tree
§IV.9 Kinerja Klasifikasi WDCNN
§IV.10 Pola Atribusi SHAP WDCNN
§IV.11 Validasi FSM (Tiga Metrik)
§IV.12 Interpretasi Fisika: Morfologi dan Periodisitas
§IV.13 Studi Ablasi Trade-off BatchNorm
§IV.14 Sintesis Lintas Keluarga
```

### Pekerjaan Spesifik

**§IV.1 (±40 baris):**
- Overview pipeline: diagram TikZ atau placeholder `fig:bab4_pipeline_diagnostik`
- Narasi justifikasi: kernel/tree = interpretable baseline edge-deployable; WDCNN = main backbone + signal-level XAI

**§IV.2 (±80 baris — dari Conf1):**
- 9 fitur time-domain dari Appendix I Paper/Conference1
- SVM RBF + LR *one-vs-rest*, GridSearchCV (C, γ parameter ranges)
- Split: stratified 80/20, random_state fixed, 5-fold CV pada training
- Sitasi: `\citetitb{TotoSuharto2024Conf1SVM}`

**§IV.3 (±80 baris — dari Conf2):**
- DT + RF + XGBoost, hyperparameter ranges per model
- Catatan label encoding (string → integer untuk XGBoost)
- Sitasi: `\citetitb{TotoSuharto2024Conf2Tree}`

**§IV.4 (±100 baris — dari Journal1 + Zhang2017WDCNN):**
- Kernel-1: lebar 64, stride 16 → penjelasan fisik (frequency-aware)
- Blok conv 2–5: kernel 3, BatchNorm, MaxPool
- FC(100) × 2 + dropout 0.5 + softmax 10 kelas
- ~60.710 parameter total
- 3 varian ablasi: A (baseline), B (kernel k=3), C (tanpa BatchNorm)
- Sitasi: `\citetitb{Zhang2017WDCNN}`, `\citetitb{TotoSuharto2025Journal1FSM}`

**§IV.5 (±80 baris):**
- KernelExplainer: background stratified kecil, mahal, untuk SVM/LR
- TreeExplainer: eksak Lundberg 2018, untuk DT/RF/XGBoost
- DeepExplainer: DeepLIFT-based, 300 background stratified, 500 test (50/kelas), **output 10 array 500×2.048**
- Sitasi: `\citetitb{Lundberg2017SHAP}`, `\citetitb{Shrikumar2017DeepLIFT}`

**§IV.6 (±100 baris):**
- Formula Signed FSM, Absolute FSM, Variance FSM
- Formula discriminability index (jarak antar-kelas)
- Formula severity monotonicity fraction
- Formula split-half stability (Pearson antara dua subset acak)

**§IV.7–IV.8 (±120 baris gabungan):**
- Tabel akurasi/F1 SVM dan LR, 10×10 CM placeholder
- Tabel akurasi/F1 DT/RF/XGB, 10×10 CM placeholder
- Diskusi: ranking fitur per SHAP (kurtosis, crest factor, BPFI/BPFO energy konsisten)
- Angka kunci dari Paper: SVM 96.4%, LR 94.3%, RF 95.8%, XGB 97.1%

**§IV.9 (±60 baris):**
- Akurasi **99,87%** (749/750), macro F1 = 0,997, early stopping epoch 54
- Satu misklasifikasi: Ball\_014 → IR\_014
- Perbandingan dengan SVM-RBF 96,4% → reduksi error 91%
- Gambar `fig:bab4_wdcnn_training_curves` (placeholder)
- Gambar `fig:bab4_wdcnn_cm` (placeholder 10×10)

**§IV.10 (±80 baris):**
- Tiga karakteristik global: edge suppression (posisi 0–100, 1.900–2.048), modulasi ~16 sampel (stride artifact), distribusi rata pos tengah
- Profil per kelas: Normal (rendah) → Ball (difus) → IR (datar) → OR (terlokalisasi)
- OR_021: puncak |φ| ≈ 0,08 pada posisi 600–700
- Gambar `fig:bab4_shap_overlay` (placeholder)

**§IV.11 (±70 baris):**
- Split-half stability = **0,940**
- Discriminability index = **0,216** (varian A baseline)
- Severity monotonicity: Ball 17,6% > IR 13,8% > OR 8,6%
- Gambar `fig:bab4_fsm_heatmap` (placeholder)
- Gambar `fig:bab4_fsm_severity` (placeholder)

**§IV.12 (±60 baris):**
- Temuan kritis: FSM peaks **TIDAK selaras** dengan periode BPFx teoritis
- WDCNN mengklasifikasikan melalui **morfologi transien impuls** (42,67 ms), bukan periodisitas BPFx
- Implikasi: tidak butuh geometri bearing → lebih transferable

**§IV.13 (±80 baris):**
- Tabel ablasi 3 varian:

| Varian | Akurasi | Discriminability | Parameter |
|--------|---------|------------------|-----------|
| A (baseline) | 99,73% | 0,216 | ~60.710 |
| B (kernel k=3) | ↓ 0,80 pp | ↓ 17% | ÷ 7,3× |
| C (tanpa BN) | ↓ 3,60 pp | **↑ 133% (0,735)** | ~60.710 |

- **Temuan pertama dalam literatur:** BatchNorm menekan discriminability FSM
- Panduan desain: A untuk akurasi kritis; C untuk interpretabilitas maksimum
- Sitasi: `\citetitb{Ioffe2015BatchNorm}`

**§IV.14 (±60 baris):**
- Tabel pemenang per metrik: akurasi, F1 kelas minoritas, parameter, latency edge
- Konvergensi ranking fitur kernel/tree (kurtosis + crest factor + BPFI/BPFO)
- Posisi WDCNN+FSM: akurasi tertinggi + signal-level XAI unik

### Update `disertasi.tex` (Fase D)
```latex
% Ganti:
\input{chapters/04-metodologi.tex}
% Dengan:
\input{chapters/04-diagnostik.tex}
```

### Verifikasi Fase D
```powershell
make build
make lint
# Cek angka kunci ada:
Select-String -Path "chapters\04-diagnostik.tex" -Pattern "99,87|0,940|0,216|133"
# Harus muncul semua angka kunci
```

---

## Fase E — Bab V: Prognostik (BUAT BARU: `05-prognostik.tex`)

**Sumber materi:**
- `03-dasar-teori.tex §xLSTM, §Mamba, §SAE, §Analisis Sinyal Getaran` → §V metodologi
- `04-metodologi.tex §Backbone RUL, §SAE, §BPFx, §Eksperimen` → §V metodologi
- `05-hasil-pembahasan.tex §Hasil Prediksi RUL, §Hasil Interpretabilitas` → §V hasil
- Paper/Journal2_RUL_Journal.pdf (sudah dibaca) → angka kunci + referensi baru

### Struktur File Baru

```
% chapters/05-prognostik.tex
\chapter{Remaining Useful Life: Metodologi dan Hasil}  % Bab V

[METODOLOGI]
§V.1 Arsitektur Backbone RUL
  §V.1.1 Mamba-xLSTM-Net
  §V.1.2 N-BEATS-xLSTM-RUL
  §V.1.3 SparseGate-TCN-RUL
§V.2 Top-k Sparse Autoencoder
§V.3 Prosedur Pemetaan SAE → BPFx
§V.4 Inferensi Statistik dan Negative Controls

[HASIL DAN ANALISIS]
§V.5 Kinerja Backbone RUL
§V.6 Hit-Rate BPFx — Hasil Utama
§V.7 Validasi Negative Controls
§V.8 Perbandingan Lintas Arsitektur
§V.9 Sparsity Sweep
```

### Pekerjaan Spesifik

**§V.1 (±150 baris — ADAPTASI dari ch04 + ch03):**
- Mamba-xLSTM-Net: selective SSM + mLSTM matrix-memory + gated fusion, 898K params PHM / 811K XJTU
- N-BEATS-xLSTM-RUL: basis-block (trend/wear/shock) + xLSTM residual, 459K params
- SparseGate-TCN-RUL: dilated causal conv + sparse gating, 249K params (lightweight)
- Protokol training bersama: 75 epoch, bf16, batch 512 × window 32, seed {42,43,44}
- Sitasi: `\citetitb{Gu2023Mamba}`, `\citetitb{Beck2024xLSTM}`, `\citetitb{Oreshkin2020NBEATS}`, `\citetitb{TotoSuharto2025Journal2RUL}`

**§V.2 (±80 baris — ADAPTASI dari ch03 §SAE + ch04 §SAE):**
- Encoder W_enc ∈ ℝ^(d_lat × d), decoder W_dec, d_lat = 8d = 1.024, k = 51 (~5% aktif)
- Training: pool N = 20.000 hidden states (backbone frozen), AdamW lr=1e-3, 50 epoch, loss MSE
- Tujuan: mendorong monosemanticity (satu SAE feature = satu konsep fisika)
- Sitasi: `\citetitb{Bricken2023Monosemanticity}`, `\citetitb{Cunningham2023TopkSAE}`, `\citetitb{Makhzani2014kSAE}`

**§V.3 (±100 baris — dari Journal2 §Metodologi):**
- Hilbert envelope spectrum + band-pass pada resonansi dominan (spectral kurtosis)
- Korelasi Pearson r_{i,φ} antara aktivasi SAE z_{r,i} dan amplitudo BPFx A_{r,φ}
- Hit-rate H_φ = |{i: |r_{i,φ}| ≥ 0,30}| / d_lat; threshold 0,30 = korelasi moderate
- Sitasi: `\citetitb{Antoni2007Kurtogram}`, `\citetitb{McFadden1984BearingModel}`

**§V.4 (±80 baris):**
- Bootstrap B=1.000 → 95% CI untuk H
- Permutation test B=1.000 → p-value two-sided
- Control 1: model Xavier-init (uji artefak arsitektur)
- Control 2: Gaussian noise hidden states (uji artefak prosedur SAE)
- Sparsity sweep: k ∈ {10, 51, 102, 205}

**§V.5 (±80 baris — dari ch05 §Hasil Prediksi RUL, REPLACE placeholder):**
Tabel kinerja backbone:

| Dataset | Pemenang | RMSE (mean±std, 3 seed) |
|---|---|---|
| PHM2012 | SparseGate-TCN | **0,226 ± 0,030** |
| XJTU-SY | N-BEATS-xLSTM | **0,259 ± 0,003** |
| IMS | Mamba-xLSTM | **0,407 ± 0,040** |

- Gambar `fig:bab5_rul_curves` (placeholder)

**§V.6 (±100 baris — dari Journal2 §Hasil):**
Tabel hit-rate BPFx:

| Dataset | BPFI | BPFO | BSF | FTF | p-value | Konsistensi Fisika |
|---|---|---|---|---|---|---|
| PHM2012 | **2,3%** | 2,0% | 0 | 0 | < 0,001 | mixed IR+OR spalling |
| XJTU-SY | 0 | **2,2%** | 0,3% | 0 | < 0,001 | outer-race spalling |
| IMS | **1,76%** | 0 | 0,49% | 0 | 0,001 | cage+rolling fatigue |
| CWRU | **5,08%** | 0 | 0 | 0 | > 0,05 (underpowered) | IR-dominated |

- r_max = 0,507 (PHM2012, BPFI); r_max = 0,501 (XJTU-SY, BPFO)
- Gambar `fig:bab5_hitrate_panel` (placeholder)
- Gambar `fig:bab5_correlation_scatter` (placeholder)

**§V.7 (±60 baris):**
- Hit-rate turun drastis pada Control 1 + Control 2
- Inference: korespondensi SAE↔BPFx adalah emergent property representasi terlatih
- Gambar `fig:bab5_negative_controls` (placeholder)

**§V.8 (±60 baris):**
- Mamba-xLSTM unggul BPFI di PHM2012 (SSM inductive bias → temporal richer)
- XJTU-SY: Mamba BPFO dominan (4,69%)
- CWRU (cross-dataset): semua konvergen ke BPFI-dominant (distribusi data dominan)

**§V.9 (±50 baris):**
- k=51 optimal; k<10 kehilangan BPFx; k>102 noise BPFx sekunder
- Gambar `fig:bab5_sparsity_sweep` (placeholder)

### Update `disertasi.tex` (Fase E)
```latex
% Ganti:
\input{chapters/05-hasil-pembahasan.tex}
% Dengan:
\input{chapters/05-prognostik.tex}
```

### Verifikasi Fase E
```powershell
make build
make lint
Select-String -Path "chapters\05-prognostik.tex" -Pattern "0,226|0,507|2,3|underpowered"
# Semua angka kunci harus ada
```

---

## Fase F — Bab VI: Kesimpulan (UPDATE: `06-kesimpulan.tex`)

**Sumber:** `dissertation-outline.md §Bab VI` | `06-kesimpulan.tex` existing

### Perubahan Struktur

**Saat ini (5 subseksi):**
- Kesimpulan (Tujuan 1/2/3 = RUL-focused)
- Kontribusi Keilmuan + Industri
- Saran Pengembangan Lanjutan

**Target (per outline §VI.1–VI.5):**
```
§VI.1 Pembahasan Terintegrasi (dua lapis interpretabilitas)
§VI.2 Kerangka PdM Multi-Tier (3 tier: Edge IoT → Edge Server → Cloud)
§VI.3 Kesimpulan (N1–N5 terjawab kuantitatif)
§VI.4 Keterbatasan
§VI.5 Rekomendasi Penelitian Lanjutan
```

### Pekerjaan Spesifik

**§VI.1 — BARU (±80 baris):**
- FSM (Bab IV) = input attribution, sinyal level, horizon waktu ~42,67 ms
- SAE (Bab V) = latent concept, hidden state level, ratusan akuisisi degradasi
- Rantai konsistensi: kurtosis/crest factor → impuls terlokalisasi (FSM) → BPFx di latent (SAE)
- Gambar `fig:bab6_pdm_multitier` (placeholder/TikZ diagram)

**§VI.2 — BARU (±70 baris):**
Tabel PdM Multi-Tier:

| Tier | Lokasi | Model | XAI | Output |
|---|---|---|---|---|
| 1 — Edge IoT | Sensor/IMx-8 | SVM/LR (Conf 1) | SHAP KernelExplainer | Triage anomali |
| 2 — Edge Server | Gateway pabrik | WDCNN+FSM (Journal 1) | SHAP DeepExplainer+FSM | Jenis+keparahan, 99,87% |
| 3 — Cloud/GPU | AWS/on-prem | Backbone RUL+SAE-BPFx (Journal 2) | SAE-BPFx mapping | RUL estimate + audit |

**§VI.3 — REWRITE (±80 baris):**
- Ringkasan kuantitatif N1–N5:
  - N1 (FSM): split-half stability 0,940
  - N2 (BatchNorm trade-off): discriminability +133% pada varian tanpa BN
  - N3 (SAEBearing): hit-rate BPFI 2,3% (PHM2012, p < 0,001)
  - N4 (universalitas+statistik): 3 arsitektur × 4 dataset, bootstrap CI, permutation test, 2 negative controls
  - N5 (integrasi): kerangka PdM multi-tier 3 tier
- Kontribusi inti: **jembatan pertama** AI data-driven ↔ teori getaran klasik yang dapat diverifikasi statistik

**§VI.4 — REWRITE (±60 baris):**
- Kondisi operasi tunggal per dataset
- CWRU = seeded faults (EDM) → tidak sepenuhnya representatif
- SAE post-hoc (belum dalam loop pelatihan)
- CWRU underpowered untuk SAE (n=10 recording, p > 0,05)
- **Integrasi data kualitas radial clearance belum dieksekusi** (SK-Toto Manfaat #1 → future work)

**§VI.5 — REWRITE (±80 baris):**
7 rekomendasi per outline:
1. Validasi cross-rig: Paderborn + MFPT
2. SAE in-loop (joint training)
3. Normalizer-Free Networks sebagai alternatif BatchNorm
4. SHAP pada STFT/CWT (time-frequency domain)
5. Edge real-time deployment (< 100 ms pada Jetson Orin Nano)
6. Integrasi ERP/CMMS
7. **Prioritas future work:** kumpulkan data kualitas radial clearance PT~SKF + integrasikan sebagai multi-modal PdM (melengkapi Tujuan 2 SK-Toto secara penuh)

### Verifikasi Fase F
```powershell
make build
make lint
Select-String -Path "chapters\06-kesimpulan.tex" -Pattern "^\\\\section" | Measure-Object
# Harus 5 sections (VI.1–VI.5)
```

---

## Fase G — Lampiran (D soften + E/F/G baru)

### G1: Soften Lampiran D (`lampiran/D-klasifikasi-industri.tex`)

**Perubahan:**
- Hapus klaim "validasi lintas mesin penuh" — ganti dengan "sanity check eksternal terbatas"
- Tambahkan: §roadmap pengumpulan data kualitas radial clearance untuk mendukung Tujuan 2 SK-Toto
- Hapus tabel benchmark 6 algoritma jika ada klaim OOD performance — ganti dengan kalimat bersyarat

### G2: Buat Lampiran E (`lampiran/E-svm-lr.tex`)

Sumber: Paper/Conference1_Classification_SVM_LR.pdf

Konten:
- Tabel GridSearchCV: C ∈ {0.1, 1, 10, 100}, γ ∈ {scale, auto, 0.01, 0.1}, kernel RBF, class_weight balanced
- Tabel per-class precision/recall/F1 untuk SVM dan LR
- Gambar placeholder CM 10×10 SVM dan LR
- Catatan reproduksi: `Notebook/Conference1_Classification_SVM_LR.ipynb`

### G3: Buat Lampiran F (`lampiran/F-tree.tex`)

Sumber: Paper/Conference2_Classification_Tree.pdf

Konten:
- Tabel hyperparameter DT (max_depth, min_samples_leaf), RF (n_estimators=100, max_depth=20), XGBoost (n_estimators=150, lr=0.05, max_depth=6, lambda)
- Per-class F1 per model
- Gambar placeholder CM 10×10 × 3 model
- Catatan label encoding string→integer untuk XGBoost
- Catatan reproduksi: `Notebook/Conference2_Classification_Tree.ipynb`

### G4: Buat Lampiran G (`lampiran/G-wdcnn-fsm.tex`)

Sumber: Paper/Journal1_Fault Signature Maps.pdf

Konten:
- Tabel arsitektur WDCNN lengkap (layer, kernel size, stride, padding, output dim, parameter)
- Hyperparameter training: Adam lr=0,001, StepLR γ=0,5 step=20, batch 64, epoch 100 max, early stopping 15
- Gambar placeholder training curves (loss + accuracy train/val)
- Gambar placeholder FSM heatmap full-resolution (10 kelas × 2.048 posisi)
- Tabel ablasi kernel size: variasi 32/64/128 → akurasi + discriminability
- Catatan reproduksi: `Notebook/Journal1_Fault Signature Maps.ipynb`

### Update `disertasi.tex` (Fase G)
```latex
% Tambahkan setelah \input{lampiran/D-klasifikasi-industri.tex}:
\input{lampiran/E-svm-lr.tex}
\input{lampiran/F-tree.tex}
\input{lampiran/G-wdcnn-fsm.tex}
```

### Verifikasi Fase G
```powershell
make build
make lint
# Cek lampiran baru terdaftar di PDF (akan ada di daftar isi lampiran)
```

---

## Fase H — Front Matter + disertasi.tex Finalisasi

### H1: Abstrak Indonesia (`00-abstrak-id.tex`) — REWRITE

4 paragraf per outline:
1. Konteks + motivasi (Making Indonesia 4.0, bearings 40–50% kegagalan, gap XAI sinyal mentah + latent-physics)
2. Jalur diagnostik: 3 keluarga benchmark CWRU; WDCNN+FSM akurasi **99,87%**, split-half stability 0,940; BatchNorm trade-off (+133% discriminability)
3. Jalur prognostik: 3 backbone RUL; **RMSE 0,226 PHM2012**; Top-k SAE → BPFx hit-rate BPFI 2,3% (p < 0,001); r_max = 0,507; negative controls validasi
4. Sintesis: 5 novelti N1–N5; PdM multi-tier 3 tier; jembatan pertama AI data-driven ↔ teori getaran klasik
- Keywords: ≤ 7 frasa, drawn from abstract content

### H2: Abstract English (`00-abstract-en.tex`) — REWRITE

Struktur sama, terjemahan akurat, keywords sama.

### H3: Daftar Singkatan (`00-daftar-singkatan.tex`) — UPDATE

Tambahkan (alphabetical):
- BSF, CBM, CI, CV, CWRU, DT, EDM, FTF, FSM, IMx-8, IoT, IR, IMS, mLSTM, OEE, OR, PHM, RF, SAE, SHAP, sLSTM, SSM, SVM, WDCNN, XAI, XGB, XJTU

### H4: `disertasi.tex` — FINAL CHECK

Pastikan:
- Semua chapter files yang baru ada di `\input`
- Files lama (03-dasar-teori, 04-metodologi, 05-hasil-pembahasan) TIDAK ada di `\input` lagi
- `\graphicspath` mencakup `figures/bab3/`, `figures/bab4/`, `figures/bab5/`
- Metadata judul, NIM, prodi, promotor sudah benar
- `pdfkeywords` diperbarui dengan keywords baru

### Verifikasi Fase H
```powershell
make pre-submit   # compile + wordcount + lint — semua harus lulus
# Cek wordcount per bab (target: Bab IV + Bab V masing-masing ≥ 5.000 kata)
```

---

## Urutan Commit yang Disarankan

```
[Fase A] restrukturisasi Bab I Pendahuluan sesuai outline §1.1–1.7
[Fase B] augmentasi Bab II §2.2.3/§2.3.3/§2.3.4/§2.5.2/§2.6/§2.7
[Fase C] buat Bab III Metodologi Umum (03-metodologi-umum.tex)
[Fase D] buat Bab IV Diagnostik (04-diagnostik.tex) — metodologi
[Fase D.2] buat Bab IV Diagnostik — hasil §IV.7–IV.14
[Fase E] buat Bab V Prognostik (05-prognostik.tex) — metodologi
[Fase E.2] buat Bab V Prognostik — hasil §V.5–V.9
[Fase F] update Bab VI Kesimpulan §VI.1–VI.5
[Fase G] lampiran D soften + E/F/G baru
[Fase H] front matter (abstrak, singkatan) + disertasi.tex finalisasi
[FINAL] make pre-submit — clean build confirmed
```

---

## Files Kritis

| File | Peran | Aksi |
|---|---|---|
| `writings/dissertation-outline.md` | ★ Single source of truth | Baca sebelum tiap fase |
| `writings/disertation/disertasi.tex` | Master file | Update per fase C/D/E/G/H |
| `chapters/01-pendahuluan.tex` | Bab I | Rewrite Fase A |
| `chapters/02-tinjauan-pustaka.tex` | Bab II | Augment Fase B |
| `chapters/03-metodologi-umum.tex` | Bab III (NEW) | Create Fase C |
| `chapters/04-diagnostik.tex` | Bab IV (NEW) | Create Fase D |
| `chapters/05-prognostik.tex` | Bab V (NEW) | Create Fase E |
| `chapters/06-kesimpulan.tex` | Bab VI | Rewrite Fase F |
| `lampiran/D-klasifikasi-industri.tex` | Lampiran D | Soften Fase G |
| `lampiran/E-svm-lr.tex` | Lampiran E (NEW) | Create Fase G |
| `lampiran/F-tree.tex` | Lampiran F (NEW) | Create Fase G |
| `lampiran/G-wdcnn-fsm.tex` | Lampiran G (NEW) | Create Fase G |
| `chapters/00-abstrak-id.tex` | Abstrak ID | Rewrite Fase H |
| `chapters/00-abstract-en.tex` | Abstract EN | Rewrite Fase H |
| `chapters/00-daftar-singkatan.tex` | Daftar Singkatan | Update Fase H |

**Files backup (JANGAN hapus sampai Fase H selesai):**
- `chapters/03-dasar-teori.tex` — sumber konten untuk Fase C/D/E
- `chapters/04-metodologi.tex` — sumber konten untuk Fase C/D/E
- `chapters/05-hasil-pembahasan.tex` — sumber konten untuk Fase D/E

---

## Outstanding TODOs (Perlu Konfirmasi Pak Toto)

1. **Venue + tahun 4 paper** sebelum finalisasi bib entries `TotoSuharto20XXKey`
2. **NIP promotor/ko-promotor** di disertasi.tex (3 TODO(verify) sudah ada)
3. **Volume data PT~SKF** (jumlah titik, label, ground-truth) sebelum nulis Lampiran D revisi
4. **Journal 2 notebook** belum ada di Notebook/ — perlu ditambahkan untuk reproduksi Bab V
5. **Geometri bearing NSK 6804** (TODO di ch03 baris 264) — konfirmasi sebelum pindah ke §3.2
