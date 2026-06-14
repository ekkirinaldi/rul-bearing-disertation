# Outline Disertasi Final — Toto Suharto

**Judul:** *Perawatan Prediktif untuk Sistem Produksi dengan Pendekatan Analisis Big Data dan Kecerdasan Buatan Menggunakan Data Kondisi Mesin dan Informasi Kualitas yang Real Time*

**Program:** Doktor Teknik dan Manajemen Industri, Fakultas Teknologi Industri, Institut Teknologi Bandung

**NIM:** 33420002 — **Tahun:** 2025/2026

**Status:** Draft outline final — restrukturisasi mengikuti enam-bab [Outline_Disertasi_6Bab.pdf](Outline_Disertasi_6Bab.pdf), dengan Bab I diselaraskan ke [SK-Toto.pdf](SK-Toto.pdf) dan pembahasan kernel/tree/deep CWRU yang dipertahankan dari karya teknis Pak Toto (Conference 1, Conference 2, Journal 1, Journal 2).

---

## Logika Struktur Disertasi

Bab III berisi **metodologi bersama** (*shared*) yang berlaku untuk keseluruhan disertasi: kerangka konseptual, dataset, infrastruktur, dan metrik evaluasi. Bab IV dan Bab V masing-masing membuka dengan sub-bab metodologi spesifiknya sendiri sebelum masuk ke hasil — sehingga pembaca melihat dengan jelas perbedaan metode antara **studi diagnostik** (kernel → tree → deep WDCNN + SHAP + FSM) dan **studi prognostik** (Backbone RUL + Top-k SAE + BPFx mapping).

Alur: **Bab I** (konteks) → **Bab II** (teori) → **Bab III** (fondasi bersama) → **Bab IV** (metode + hasil: Diagnostik) → **Bab V** (metode + hasil: Prognostik) → **Bab VI** (sintesis + kesimpulan).

---

## Dua Jalur Paralel + Konvergensi

```
   JALUR A (DIAGNOSTIK)              JALUR B (PROGNOSTIK)
   CWRU                              PHM2012 · XJTU-SY · IMS
   ↓                                 ↓
   Benchmark Kernel/Tree/Deep        Backbone RUL
   (SVM/LR · DT/RF/XGB · WDCNN)      (Mamba-xLSTM · N-BEATS-xLSTM · SparseGate-TCN)
   ↓                                 ↓
   SHAP (Kernel/Tree/Deep            Top-k Sparse Autoencoder
   Explainer)                        ↓
   ↓                                 Pemetaan ke BPFx
   Fault Signature Maps (FSM)        (Hilbert envelope + Pearson +
                                     bootstrap CI + permutation test)
                       ↘            ↙
                    KONVERGENSI (BAB VI)
                    · Kerangka PdM Multi-Tier (Edge IoT → Edge Server → Cloud/GPU)
                    · Interpretabilitas dua-lapis:
                      input-attribution (FSM) ↔ latent-concept (SAE-BPFx)
```

| Jalur | Pertanyaan inti | Dataset | Karya sumber |
|-------|-----------------|---------|--------------|
| **A — Diagnostik** | Jenis kerusakan apa, dan mengapa model memutuskan demikian? | CWRU (48 kHz, Load 0–3, 10 kelas, segmen 2 048 titik) | Conference 1 (SVM/LR), Conference 2 (DT/RF/XGB), Journal 1 (WDCNN + FSM) |
| **B — Prognostik** | Kapan komponen akan gagal, dan apakah representasi laten model mengandung fisika BPFx? | PHM2012, XJTU-SY, IMS (validasi tambahan: CWRU) | Journal 2 (Mamba-xLSTM-Net / N-BEATS-xLSTM-RUL / SparseGate-TCN-RUL + Top-k SAE) |

**Validasi industri terbatas:** sekumpulan kecil titik data PT~SKF Indonesia digunakan sebagai *sanity check* eksternal pada Lampiran D / penutup Bab V, bukan sebagai dataset utama.

---

## Lima Novelti Penelitian (N1–N5)

Diadopsi dari Outline_6Bab; dipertahankan sebagai *anchor* kontribusi formal disertasi. Setiap novelti dilengkapi **status pembuktian** untuk menghindari *over-claim*.

1. **N1 — Fault Signature Maps (FSM).** XAI level sinyal pertama untuk WDCNN, resolusi 2 048 titik, 3 varian (Signed / Absolute / Variance), 3 metrik validasi (discriminability, severity monotonicity, split-half stability). *[Jalur A — **empiris**; bukti: split-half stability 0,940, discriminability 0,216, severity monotonicity Ball 17,6% / IR 13,8% / OR 8,6%]*
2. **N2 — Trade-off BatchNorm.** Temuan pertama bahwa BatchNorm menekan *discriminability* FSM; panduan desain arsitektur antara akurasi maksimum dan interpretabilitas maksimum. *[Jalur A — **empiris**; bukti: discriminability +133% pada varian tanpa BN, dengan trade-off akurasi −3,60 pp]*
3. **N3 — SAEBearing.** Adaptasi pertama *mechanistic interpretability* (Top-k Sparse Autoencoder, gaya LLM Bricken–Cunningham) ke domain *bearing prognostics*. *[Jalur B — **empiris**; bukti: hit-rate BPFI 2,3% (PHM2012) dan BPFO 2,2% (XJTU-SY), p < 0,001, $r_\text{max}$ 0,507]*
4. **N4 — Universalitas + Statistical Rigor.** 3 arsitektur backbone × 4 dataset (PHM2012, XJTU-SY, IMS, CWRU), dengan bootstrap CI + permutation test + dua negative controls (Xavier-init + Gaussian noise). *[Jalur B — **empiris**; bukti: konvergensi BPFx-dominant lintas arsitektur + hit-rate jatuh ke ~0 pada kedua control]*
5. **N5 — Kerangka Konseptual PdM Multi-Tier.** Sintesis FSM (input-attribution) dan SAE-BPFx (latent-concept) menjadi **blueprint** sistem PdM tiga-tier (Edge IoT → Edge Server → Cloud/GPU) yang dapat diaudit lapis demi lapis. *[Konvergensi, Bab VI — **konseptual/sintesis** (bukan empiris); bukti pendukung: penempatan model per tier berbasis parameter, latency, dan tipe XAI yang sesuai. **Bukan** klaim hasil *deployment* lapangan.]*

---

## Struktur Disertasi (6 Bab + Lampiran A–G)

---

## Bab I — Pendahuluan

**Tujuan bab:** Memaparkan motivasi industri, masalah penelitian, tujuan + novelti, manfaat, batasan + asumsi, kontribusi & posisi, dan sistematika disertasi. *Diselaraskan dengan [SK-Toto.pdf §I](SK-Toto.pdf)*

### 1.1 Latar Belakang
- **(retained dari SK-Toto §I.1)** Konteks *Making Indonesia 4.0* (April 2018): lima sektor prioritas (makanan-minuman, tekstil, kimia, otomotif, elektronika); IoT, big data, AI sebagai enabler.
- **(retained)** Dua jenis perawatan konvensional (korektif + preventif) dengan kelemahan: di satu sisi kerusakan tak terduga, di sisi lain pergantian terlalu cepat. Perawatan prediktif sebagai jalan tengah.
- **(retained dari Outline_6Bab §I.1)** Bantalan gelinding = **40–50% penyebab kegagalan rotating machinery**. Evolusi pendekatan: fisika → statistik → *deep learning*.
- **(NEW para)** Munculnya arsitektur generasi terbaru (Mamba, xLSTM) untuk *long-sequence modeling* — peluang baru untuk RUL prediksi.
- **(NEW para)** **Dua gap besar yang belum terisi:**
  1. **Gap diagnostik:** XAI level *sinyal mentah* (bukan fitur agregat) belum ada — interpretabilitas WDCNN langsung di domain waktu 2 048 titik.
  2. **Gap prognostik:** Model RUL berbasis *deep learning* belum dibuktikan menginternalisasi fisika degradasi (frekuensi karakteristik bantalan BPFO/BPFI/BSF/FTF) dalam ruang latennya secara statistik signifikan.
- **(retained)** Konteks PT~SKF Indonesia: mesin grinding OR1/OR2, akuisisi melalui SKF~IMx-8 → AWS Cloud, kebutuhan integrasi data sensor + data kualitas (*vibration checking*, *radial clearance checking*) sebagai **motivasi industri jangka panjang**. *Pelaksanaan integrasi data kualitas heterogen termasuk dalam Bab VI sebagai future work.*
- **(NEW para)** Penegasan posisi penelitian: disertasi ini fokus pada **pipeline vibrasi end-to-end** (diagnostik + prognostik + XAI), dengan validasi eksternal terbatas pada subset data PT~SKF Indonesia. Integrasi penuh data kualitas radial clearance bukan deliverable disertasi.

### 1.2 Rumusan Masalah
*Tiga RQ disusun pada **level pertanyaan industri** mengikuti [SK-Toto §I.2](SK-Toto.pdf), dengan **sub-RQ teknis** yang dapat dijawab penuh oleh papers Pak Toto + draft disertasi. Sub-RQ menjadi anchor empiris setiap bab; RQ tingkat atas menjaga kontinuitas dengan framing usulan penelitian Agustus 2025.*

- **RQ1 — Integrasi Data Sensor dan Arsitektur PdM untuk OEE.** *Bagaimana sistem perawatan prediktif berbasis sensor vibrasi yang melekat pada arsitektur multi-tier (Edge IoT → Edge Server → Cloud/GPU) dapat dirancang sebagai fondasi untuk mengurangi unplanned downtime dan mendukung peningkatan OEE pada produksi ball bearing, dengan integrasi penuh data kualitas heterogen (vibration checking + radial clearance checking) sebagai roadmap penelitian lanjutan?*
  - **Sub-RQ1.1 — Pembagian beban komputasi lintas tier.** *Bagaimana memetakan tiga jenis beban inferensi (triage anomali ringan, klasifikasi jenis kerusakan, estimasi RUL) ke tiga tier (Edge IoT, Edge Server, Cloud) sehingga keluaran setiap tier dapat dipakai oleh tier berikutnya dalam satu loop perawatan?*
  - *Backing:* [06-kesimpulan.tex](disertation/chapters/06-kesimpulan.tex) §VI.2 (Kerangka PdM Multi-Tier — Tabel + diagram); [Lampiran D](disertation/lampiran/D-klasifikasi-industri.tex) (konteks SKF~IMx-8 + AWS).
  - *Catatan transparansi:* Integrasi penuh data kualitas heterogen (radial clearance) yang dimaksud SK-Toto §I.2 #1 **tidak dijawab secara empiris** oleh disertasi ini — hanya disertakan sebagai motivasi industri di §1.1 dan future work eksplisit di §VI.5. Disertasi membatasi diri pada *pipeline vibrasi end-to-end* dengan validasi eksternal terbatas (Lampiran D). Klaim peningkatan OEE bersifat **arsitektural** (blueprint mengurangi *unplanned downtime* melalui deteksi dini + estimasi RUL), **bukan** klaim kuantitatif dari studi *deployment* lapangan.

- **RQ2 — Pendekatan Hybrid ML+DL untuk Deteksi Kerusakan dan Estimasi RUL.** *Apakah pendekatan hybrid yang menggabungkan keluarga algoritma statistik/kernel (SVM, LR), tree-based (DT, RF, XGBoost), dan deep learning (WDCNN untuk diagnostik; Mamba-xLSTM-Net, N-BEATS-xLSTM-RUL, SparseGate-TCN-RUL untuk prognostik) dapat memberikan deteksi anomali (klasifikasi jenis kerusakan) dan estimasi RUL bantalan yang konsisten lintas keluarga algoritma dan lintas dataset publik (CWRU, PHM2012, XJTU-SY, IMS)?*
  - **Sub-RQ2.1 — Konsistensi diagnostik multi-model.** *Sejauh mana benchmark klasifikasi multi-model pada CWRU (kernel/tree/deep) menghasilkan akurasi yang konsisten, dan model mana yang menempati posisi terbaik pada trade-off antara akurasi dan kompleksitas parameter untuk *edge deployment*?*
  - **Sub-RQ2.2 — Universalitas backbone RUL.** *Sejauh mana ketiga arsitektur backbone (Mamba-xLSTM, N-BEATS-xLSTM, SparseGate-TCN) menghasilkan estimasi RUL yang konsisten lintas tiga dataset (PHM2012, XJTU-SY, IMS) di bawah protokol pelatihan bersama (3 seed, 75 epoch, bf16, batch 512 × window 32) dengan inferensi statistik (bootstrap CI + permutation test)?*
  - *Backing:* Conf 1, Conf 2, Journal 1, Journal 2; [04-diagnostik.tex](disertation/chapters/04-diagnostik.tex) §4.7–4.9 + §4.14; [05-prognostik.tex](disertation/chapters/05-prognostik.tex) §5.5 + §5.8.
  - *Honest scoping:* "Hybrid" dalam disertasi ini berarti **lintas keluarga algoritma yang dievaluasi paralel** (kernel/tree/deep di Bab IV; 3 backbone DL di Bab V) — **bukan** ensembling antar keluarga ke satu prediktor tunggal. Integrasi multi-modal dengan data kualitas radial clearance tetap di luar lingkup empiris.

- **RQ3 — Transparansi Model melalui XAI untuk Adopsi Industri.** *Bagaimana metode Explainable AI — SHAP (KernelExplainer / TreeExplainer / DeepExplainer) untuk diagnostik dan Top-k Sparse Autoencoder + pemetaan ke frekuensi karakteristik bantalan (BPFx) untuk prognostik — dapat meningkatkan transparansi dan auditability model PdM sehingga layak diadopsi oleh praktisi industri manufaktur?*
  - **Sub-RQ3.1 — Atribusi level sinyal (diagnostik).** *Apakah pola atribusi SHAP DeepExplainer pada sinyal mentah 2 048 titik WDCNN dapat dipetakan ke morfologi impuls transien atau periodisitas BPFx — dan apakah pemetaan ini cukup stabil dan diskriminatif untuk membentuk fingerprint (Fault Signature Map) per kelas kerusakan?*
  - **Sub-RQ3.2 — Konsep laten (prognostik).** *Apakah Top-k Sparse Autoencoder yang dilatih pada hidden states backbone RUL dapat mengekstrak fitur laten yang berkorespondensi dengan BPFO/BPFI/BSF/FTF secara statistik signifikan — dibuktikan dengan bootstrap CI, permutation test, dan dua negative controls (Xavier init + Gaussian noise)?*
  - *Backing:* Journal 1, Journal 2; [04-diagnostik.tex](disertation/chapters/04-diagnostik.tex) §4.10–4.13; [05-prognostik.tex](disertation/chapters/05-prognostik.tex) §5.6–5.8. Angka kunci: split-half stability 0,940 (FSM); $r_\text{max} = 0{,}507$ (PHM2012, BPFI), $r_\text{max} = 0{,}501$ (XJTU-SY, BPFO).

### 1.3 Tujuan Penelitian dan Novelti
*Tiga tujuan dari [SK-Toto §I.3](SK-Toto.pdf), dipetakan eksplisit ke lima novelti N1–N5 dan ke tiga RQ §1.2. Setiap tujuan adalah anchor untuk satu RQ; novelti adalah bukti pembuktiannya.*

- **Tujuan 1 — Pengembangan Sistem PdM Multi-Tier (menjawab RQ1).** Merancang *blueprint* sistem perawatan prediktif berbasis sensor vibrasi yang mengintegrasikan blok diagnostik (klasifikasi kerusakan) dan blok prognostik (estimasi RUL) ke dalam arsitektur tiga-tier (Edge IoT → Edge Server → Cloud/GPU) yang dapat diaudit lapis demi lapis. → **Novelti N5 — Kerangka Konseptual PdM Multi-Tier** (sintesis FSM input-attribution + SAE-BPFx latent-concept). *Bukti:* tabel penempatan model per tier di Bab VI.2 (berbasis parameter count, latency estimasi, tipe XAI yang sesuai).

- **Tujuan 2 — Validasi Konsistensi Hybrid ML+DL Lintas Dataset (menjawab RQ2).** Memvalidasi konsistensi tiga keluarga algoritma diagnostik (kernel/tree/deep) dan tiga arsitektur backbone prognostik (Mamba-xLSTM, N-BEATS-xLSTM, SparseGate-TCN) lintas empat dataset publik (CWRU, PHM2012, XJTU-SY, IMS), dengan inferensi statistik *bootstrap* + *permutation test* + dua *negative controls*. → **Novelti N4 — Universalitas + Statistical Rigor**. *Bukti:* tabel RMSE 3 backbone × 3 dataset RUL + konvergensi BPFx-dominant lintas arsitektur + hit-rate jatuh ke ~0 pada Control 1/2. *Catatan honest:* Tujuan 2 SK-Toto tentang integrasi data kualitas waktu-nyata dijalankan terbatas pada *multi-kanal vibrasi* (drive-end + fan-end CWRU; horizontal + vertical PHM2012); integrasi multi-modal dengan *radial clearance checking* dijadikan future work eksplisit (§VI.5).

- **Tujuan 3 — Transparansi Model melalui XAI untuk Adopsi Industri (menjawab RQ3).** Mengintegrasikan metode *Explainable AI* berbasis SHAP (KernelExplainer / TreeExplainer / DeepExplainer) di sisi diagnostik dan *Top-k Sparse Autoencoder* + pemetaan BPFx di sisi prognostik untuk meningkatkan transparansi + auditability model bagi praktisi industri. → **Novelti N1 — Fault Signature Maps** (XAI level sinyal pertama untuk WDCNN) + **N2 — Trade-off BatchNorm** (temuan pertama BN menekan discriminability FSM) + **N3 — SAEBearing** (adaptasi pertama *mechanistic interpretability* ke domain *bearing prognostics*). *Bukti:* split-half stability 0,940; discriminability +133% pada varian tanpa BN; hit-rate BPFI 2,3% (PHM2012) p < 0,001.

**Catatan tentang status pembuktian** *(menjawab perhatian over-claim, sesuai bimbingan Pak Toto):*
- **N1, N2, N3, N4** = novelti **empiris**, terukur kuantitatif dengan angka kunci di Bab IV–V (hit-rate, p-value, split-half stability, akurasi, discriminability).
- **N5** = novelti **konseptual/sintesis** — *blueprint* arsitektur yang menggabungkan komponen-komponen empiris N1–N4 ke satu sistem multi-tier. Bukti pendukungnya adalah **penempatan model yang sesuai pada tier yang sesuai** (latency, parameter, tipe XAI), bukan eksperimen *deployment* lapangan.
- **RQ1** dijawab pada level **blueprint arsitektur** (Bab VI.2), bukan pada level *cross-machine deployment study*. Validasi industri PT~SKF terbatas pada *sanity check* eksternal (Lampiran D).

### 1.4 Manfaat Penelitian
*Adaptasi dari [SK-Toto §I.4](SK-Toto.pdf), dilembutkan: klaim multi-modal radial clearance digeser ke aspirasi/future work.*

1. **(retained)** Memberikan contoh kasus implementasi analisis *big data* dan AI dalam sistem produksi untuk meningkatkan produktivitas dan daya saing manufaktur Indonesia.
2. **(retained)** Hasil penelitian berupa pipeline *open-source* (notebook reproduksi + bobot model + skrip evaluasi statistik) yang dapat menjadi contoh integrasi diagnostik–prognostik dan layer XAI dalam sistem perawatan prediktif.
3. **(retained)** Pipeline yang dikembangkan untuk produksi ball bearing dapat menjadi dasar metodologis untuk sistem PdM pada proses produksi lain yang menggunakan permesinan rotasi.
4. **(softened)** **Aspirasi jangka panjang:** integrasi penuh data kualitas heterogen (*vibration checking* + *radial clearance checking* PT~SKF) sebagai input model multi-modal — *future work* (Bab VI.5).

### 1.5 Batasan dan Asumsi Penelitian
*Gabungan [SK-Toto §I.5](SK-Toto.pdf) + [Outline_6Bab §I.4](Outline_Disertasi_6Bab.pdf).*

**Batasan:**
1. **Sistem produksi:** proses permesinan grinding pada produksi ball bearing.
2. **Sinyal:** vibrasi mentah dari akselerometer; *frequency-domain features* dan envelope spectrum dihitung sebagai derivat.
3. **Dataset diagnostik:** CWRU (Load 0–3, 48 kHz, 10 kelas Normal + IR/OR/Ball × 0.007/0.014/0.021 inch, segmen 2 048 titik, temporal split 54/13/33%).
4. **Dataset prognostik:** PHM2012 (FEMTO-PRONOSTIA, 17 bearing, 3 kondisi beban), XJTU-SY (LDK UER204, 10 bearing), IMS (Rexnord, 4 bearing).
5. **Arsitektur:** WDCNN (diagnostik); Mamba-xLSTM-Net / N-BEATS-xLSTM-RUL / SparseGate-TCN-RUL (prognostik). Baseline tambahan: SVM-RBF, LR, DT, RF, XGBoost (kernel/tree family).
6. **XAI:** SHAP (KernelExplainer / TreeExplainer / DeepExplainer) untuk diagnostik; Top-k SAE + Hilbert envelope mapping untuk prognostik.
7. **Validasi industri:** terbatas pada subset titik data PT~SKF Indonesia (volume kecil, sanity check eksternal).
8. **Perangkat:** menggunakan *library open-source* (PyTorch, scikit-learn, XGBoost, SHAP, Mamba-SSM, xLSTM). Tidak ada pengembangan *library* baru.

**Asumsi:** (adopsi dari [SK-Toto §I.5](SK-Toto.pdf) #1–#6)
1. Sensor vibrasi/suhu/putaran sudah dipasang dan terkalibrasi pada mesin OR1/OR2 PT~SKF.
2. Data quality check (*vibration checking* + *radial clearance checking*) tersedia secara waktu-nyata bagi tim PdM SKF (digunakan sebagai *ground-truth* validasi eksternal Lampiran D).
3. Mesin beroperasi dalam jangkauan normal sesuai spesifikasi SKF.
4. Data historis perawatan tersedia untuk validasi.
5. Infrastruktur komputasi (GPU A40, on-premise + cloud) tersedia untuk pelatihan dan inferensi.
6. Proses grinding memiliki karakteristik degradasi yang dapat dimodelkan dengan ML berbasis data.

### 1.6 Kontribusi dan Posisi Penelitian
- **Kontribusi keilmuan** (adaptasi [SK-Toto §I.6](SK-Toto.pdf) + Novelti N1–N5 + pemetaan ke RQ §1.2):

  | # | Kontribusi | Novelti | RQ yang dijawab | Tipe |
  |---|------------|---------|-----------------|------|
  | 1 | **Kerangka FSM** — XAI level sinyal mentah pertama untuk WDCNN, termasuk temuan trade-off BatchNorm yang belum pernah dilaporkan dalam literatur klasifikasi bantalan. | **N1 + N2** | RQ3 / Sub-RQ3.1 | Empiris |
  | 2 | **SAEBearing** — Adaptasi pertama *mechanistic interpretability* (Top-k Sparse Autoencoder) ke domain *bearing prognostics*, dengan validasi statistical-rigorous (bootstrap CI + permutation test + 2 negative controls) lintas 4 dataset dan 3 arsitektur backbone. | **N3 + N4** | RQ2 / Sub-RQ2.2; RQ3 / Sub-RQ3.2 | Empiris |
  | 3 | **Integrasi PdM Multi-Tier** — *Blueprint* arsitektur yang menyatukan FSM (input-attribution) dan SAE-BPFx (latent-concept) ke tiga tier (Edge IoT → Edge Server → Cloud/GPU) yang dapat diaudit lapis demi lapis. | **N5** | RQ1 / Sub-RQ1.1 | Konseptual |
  | 4 | **Konsistensi Hybrid ML+DL Lintas Keluarga** — Benchmark paralel kernel/tree/deep untuk diagnostik (CWRU) dan tiga backbone DL untuk prognostik (PHM2012/XJTU-SY/IMS) sebagai bukti universalitas pendekatan hybrid pada level *family of algorithms*. | Mendukung **N4** | RQ2 / Sub-RQ2.1 + Sub-RQ2.2 | Empiris |

- **Posisi penelitian** (matriks gap, augmentasi dari [Tabel I-1 SK-Toto](SK-Toto.pdf)):
  - Tabel `tab:bab1_posisi_penelitian` dengan kolom: Sumber Data | Mode Operasi | Pendekatan Model | XAI | **FSM** | **SAE-BPFx** | Tier PdM. Posisikan empat karya Pak Toto + literatur eksternal pada matriks ini.
- **Diagram TikZ** `fig:bab1_kontribusi`: lima kotak N1–N5 dengan panah dependency: N1↔N2 (Jalur A, menjawab Sub-RQ3.1); N3↔N4 (Jalur B, menjawab Sub-RQ2.2 + Sub-RQ3.2); N5 menghubungkan keduanya ke output PdM Multi-Tier (menjawab Sub-RQ1.1).

### 1.7 Sistematika Penulisan
- **Bab II — Tinjauan Pustaka:** PdM Industri 4.0, diagnostik DL, XAI input attribution, prognostik RUL, mechanistic interpretability + SAE, fisika bearing, peta gap literatur.
- **Bab III — Metodologi Umum (Shared):** kerangka konseptual terintegrasi, spesifikasi 4 dataset (CWRU/PHM2012/XJTU-SY/IMS), praproses & ekstraksi fitur HI 36-D, metrik evaluasi bersama, infrastruktur reproduksi.
- **Bab IV — Diagnostik (Metode + Hasil):** pipeline benchmark **kernel → tree → deep WDCNN** dengan SHAP per keluarga + FSM untuk WDCNN; hasil + ablasi BatchNorm + interpretasi fisika.
- **Bab V — Prognostik (Metode + Hasil):** tiga backbone RUL + Top-k SAE + pemetaan BPFx; hit-rate + bootstrap CI + permutation test + negative controls + sparsity sweep.
- **Bab VI — Kesimpulan dan Rekomendasi:** pembahasan terintegrasi, kerangka PdM multi-tier, kesimpulan N1–N5, keterbatasan, rekomendasi penelitian lanjutan.
- **Paragraf eksplisit:** Bab IV dan Bab V disusun dengan urutan **metode → hasil per jalur** (bukan metode terpisah lalu hasil terpisah), agar pembaca melihat dengan jelas perbedaan pendekatan diagnostik dengan prognostik.

---

## Bab II — Tinjauan Pustaka

**Tujuan bab:** Memetakan *state-of-the-art*, kelemahan tiap pendekatan, dan *research gap*. Mengikuti urutan [Outline_6Bab §II.1–II.7](Outline_Disertasi_6Bab.pdf), diaugmentasi dengan materi [02-tinjauan-pustaka.tex](disertation/chapters/02-tinjauan-pustaka.tex) existing.

### 2.1 Perawatan Prediktif dan Industri 4.0
- Evolusi *Condition-Based Maintenance* (CBM) — standar **ISO 13374** (data processing) dan **ISO 13381** (prognostics).
- Ekosistem sensor dan IoT industri — peran *gateway* (SKF~IMx-8), penyimpanan cloud (AWS), dan analitik tier-wise.
- Posisi PdM dalam konteks *Making Indonesia 4.0* (lima sektor prioritas).

### 2.2 Diagnostik Bearing Berbasis Deep Learning
- **2.2.1 Evolusi Arsitektur.** SVM + fitur statistik → CNN 1D → **WDCNN** (Zhang 2017/2018, kernel lebar di lapisan pertama untuk *frequency-aware extraction*) → hybrid CNN-LSTM → Transformer.
- **2.2.2 Dataset CWRU.** Protokol *test rig* (motor 2 HP, SKF 6205-2RS, drive-end + fan-end akselerometer); jenis kesalahan EDM-seeded; isu *temporal leakage* pada random split.
- **(NEW) 2.2.3 Klasifikasi Bantalan Multi-Keluarga: Kernel, Tree, Deep.** Survey ringkas tiga keluarga dengan sitasi-diri ke karya Pak Toto:
  - `\citetitb{TotoSuharto2024Conf1SVM}` — SVM RBF + Logistic Regression + SHAP KernelExplainer pada CWRU.
  - `\citetitb{TotoSuharto2024Conf2Tree}` — Decision Tree + Random Forest + XGBoost + SHAP TreeExplainer pada CWRU.
  - `\citetitb{TotoSuharto2025Journal1FSM}` — WDCNN + SHAP DeepExplainer + *Fault Signature Maps* pada CWRU.
- **(NEW) Tabel** `tab:bab2_klasifikasi_survey`: kolom Algoritma / Dataset / Akurasi / SHAP variant / FSM. Baris untuk tiga karya Pak Toto + benchmark eksternal.

### 2.3 Explainable AI: Input Attribution
- **2.3.1 Taxonomy Metode XAI.** *Gradient-based* (Integrated Gradients), *propagation-based* (LRP, DeepLIFT), *perturbation-based* (LIME, SHAP). Catatan: hampir semua bekerja di level fitur agregat — gap yang dijawab Paper 1 (Journal 1).
- **2.3.2 SHAP DeepExplainer.** Shapley values, *rescale rule* DeepLIFT-based (Shrikumar 2017), kompatibilitas dengan input sinyal mentah.
- **(NEW) 2.3.3 SHAP Kernel/Tree/Deep Explainer — Kapan Dipakai Apa.** Tabel perbandingan: KernelExplainer (model-agnostic, mahal, butuh *background*); TreeExplainer (eksak Lundberg 2018, cepat untuk tree-based); DeepExplainer (NN-specific, DeepLIFT-based, untuk WDCNN dan sejenis).
- **(NEW) 2.3.4 Fault Signature Maps (FSM).** Pengantar konsep FSM dari Journal 1 sebagai komplemen SHAP *pointwise*: agregasi posisional dari nilai SHAP DeepExplainer yang menghasilkan *fingerprint* per kelas kerusakan. Tiga varian (Signed/Absolute/Variance), tiga metrik validasi (discriminability, severity monotonicity, split-half stability).

### 2.4 Prognostik Bearing: Estimasi RUL
- **2.4.1 Fisika dan Data-Driven.** Model Paris/Archard hingga LSTM → Transformer → Mamba / xLSTM. Tren: akurasi mendekati saturasi → fokus bergeser ke **trustworthiness/interpretability**.
- **2.4.2 Benchmark Dataset RUL.** PHM2012 (FEMTO), XJTU-SY (LDK UER204), IMS (Rexnord). Metrik: RMSE, MAE, skor PHM (penalti asimetris: terlambat > lebih awal).
- **(retained dari [02-tinjauan-pustaka.tex](disertation/chapters/02-tinjauan-pustaka.tex))** Subseksi survey deep learning untuk RUL: XGBoost, CNN-LSTM, TCN, BiLSTM-Attention, GRU, ARIMA-LSTM (sebagian dimanfaatkan sebagai baseline pelengkap).

### 2.5 Mechanistic Interpretability dan Sparse Autoencoders
- **2.5.1 Superposition dan Top-k SAE.** Elhage et al. 2022 (*superposition hypothesis*), Bricken–Cunningham 2023 (*Top-k SAE* dalam LLM). Tujuan: monosemantisitas.
- **2.5.2 Gap: Belum Ada SAE untuk Bearing.** Bearing sebagai *test case* lebih bersih dari bahasa; BPFx = himpunan konsep fisika terdefinisi yang **harus** ada di latent space model RUL yang baik. Posisi: Journal 2 (Pak Toto) sebagai studi pertama yang membawa Top-k SAE + mapping konsep fisika ke domain *bearing prognostics*.

### 2.6 Fisika Getaran Bearing
- Formula **BPFO** (*Ball Pass Frequency Outer*), **BPFI** (*Ball Pass Frequency Inner*), **BSF** (*Ball Spin Frequency*), **FTF** (*Fundamental Train Frequency*) berdasarkan geometri bantalan (jumlah bola n, diameter bola d_b, diameter pitch d_m, sudut kontak α) dan kecepatan poros f_r.
- **Envelope analysis** melalui Hilbert transform — band-pass di resonansi dominan (dipilih dengan *spectral kurtosis*), envelope amplitudo A_r(t), spektrum envelope A_r,φ(f).
- Posisi: digunakan sebagai *ground-truth fisika* untuk validasi pemetaan SAE-BPFx (Bab V).

### 2.7 Peta Gap Literatur
- Matriks: **Level Interpretabilitas** (fitur / sinyal / latent) × **Dataset** (CWRU / PHM2012 / XJTU-SY / IMS / industri) × **Validasi Statistik** (akurasi-saja / CI / permutation test) × **Integrasi Diagnostik-Prognostik** (terpisah / terpadu).
- **Gap 1 (Diagnostik level sinyal):** Hampir semua XAI bearing beroperasi di level fitur; **belum ada** XAI level sinyal mentah untuk WDCNN dengan validasi multi-metrik. → Dijawab Journal 1 / Bab IV.
- **Gap 2 (Latent interpretability prognostik):** Belum ada SAE untuk *bearing prognostics* yang memetakan latent ke BPFx dengan statistical rigor. → Dijawab Journal 2 / Bab V.
- **Gap 3 (Integrasi):** Belum ada kerangka yang menyatukan input-attribution (sinyal) + latent-concept (laten) menjadi PdM multi-tier auditable. → Dijawab Bab VI.
- **(NEW)** Tabel `tab:bab2_posisi_penelitian`: posisi Conf 1 + Conf 2 + Journal 1 + Journal 2 dibandingkan dengan literatur eksternal pada matriks tersebut.

---

## Bab III — Metodologi Umum (Shared Foundation)

**Tujuan bab:** Memuat fondasi bersama yang dipakai oleh Bab IV dan Bab V — kerangka konseptual, dataset, praproses, metrik, infrastruktur. Metodologi spesifik tiap jalur dijelaskan di awal Bab IV / Bab V.

### 3.1 Kerangka Konseptual Terintegrasi
- Dua jalur paralel:
  - **Jalur A — Diagnostik:** CWRU → WDCNN → SHAP DeepExplainer → FSM (input attribution).
  - **Jalur B — Prognostik:** PHM2012 / XJTU-SY / IMS → Backbone RUL → Top-k SAE → BPFx mapping (latent concept interpretability).
- Konvergensi di Bab VI: kerangka PdM multi-tier (Edge IoT → Edge Server → Cloud/GPU) yang memakai output kedua jalur.
- **(figure)** `fig:bab3_kerangka_terintegrasi`: diagram TikZ dua-jalur dengan blok konvergensi.

### 3.2 Dataset
*Mengikuti verbatim [Outline_6Bab §III.2](Outline_Disertasi_6Bab.pdf).*

- **3.2.1 CWRU — Bab IV.**
  - *Case Western Reserve University Bearing Data Center.*
  - Test rig: motor 2 HP, SKF 6205-2RS JEM, akselerometer drive-end + fan-end.
  - 10 kelas: Normal + (Inner Race / Outer Race / Ball Fault) × (0.007 / 0.014 / 0.021 inch).
  - Drive-end 48 kHz; Load 0–3 HP.
  - Segmen 2 048 titik; **temporal split 54/13/33%** = 1 240 / 310 / 750 sampel.
  - Normalisasi: Z-score (fit-on-train).
- **3.2.2 PHM2012 — Bab V.**
  - FEMTO-PRONOSTIA challenge, 17 bearing, 3 kondisi beban.
  - End-of-Life criterion: RMS > 20 g.
  - Split bearing-wise; label RUL linier 1 → 0.
- **3.2.3 XJTU-SY — Bab V.**
  - 10 bearing LDK UER204; *outer-race spalling* dominan.
  - Split bearing-wise.
- **3.2.4 IMS — Bab V.**
  - 4 bearing Rexnord, 2 000 rpm, akuisisi tiap 10 menit.
  - Bearing 3: outer-race fault; Bearing 4: rolling-element fatigue.
- **(validasi tambahan)** CWRU juga dipakai di Bab V untuk validasi cross-dataset SAE (catatan: n=10 recording → underpowered; ditandai eksplisit).

### 3.3 Praproses Sinyal Bersama & Ekstraksi Fitur HI 36-D
- Pipeline praproses umum: segmentasi *windowing*, normalisasi, *Hann window* untuk FFT.
- Health Indicator 36-D — 9 fitur time-domain + 9 fitur frequency-domain + 5 band-energy × 2 kanal (drive-end + fan-end). Definisi lengkap dirujuk ke [Lampiran A](disertation/lampiran/A-detail-fitur.tex).
- Catatan: HI 36-D hanya digunakan oleh keluarga **kernel** dan **tree** di Bab IV.2–IV.3; backbone deep (WDCNN, RUL) beroperasi langsung pada sinyal mentah.

### 3.4 Metrik Evaluasi Bersama
- **Diagnostik (Bab IV):** Akurasi, macro-F1, weighted-F1, *confusion matrix*. Khusus FSM: discriminability index, severity monotonicity fraction, split-half stability coefficient.
- **Prognostik (Bab V):** RMSE↓, MAE↓, **skor PHM** (penalti asimetris terlambat > lebih awal).
- **Interpretabilitas SAE (Bab V):** hit-rate H = |{i: |r_{i,φ}| ≥ 0,30}| / d_lat, bootstrap 95% CI (B=1 000), permutation p-value (B=1 000 two-sided).

### 3.5 Infrastruktur dan Protokol Reproduksi
- GPU NVIDIA A40 (on-premise + cloud).
- Seeds: {42, 43, 44} untuk multi-seed reproducibility (Bab V); seed tetap dari notebook (`RANDOM_SEED`) untuk Bab IV.
- Mixed precision bf16 untuk training prognostik; FP32 untuk klasifikasi.
- Repositori publik: [rul-bearing-disertation](https://github.com/dito-eka/rul-bearing-disertation) (notebook reproduksi Conf 1, Conf 2, Journal 1).
- Checkpoint disimpan pada loss validasi terbaik; protokol *early stopping*: patience 15 (WDCNN), tiada early-stop (RUL — 75 epoch fixed).

---

## Bab IV — Bearing Fault Detection: Metodologi dan Hasil (Diagnostik)

**Tujuan bab:** Menyajikan jalur diagnostik secara end-to-end. Diawali dengan **benchmark tiga keluarga algoritma** (kernel → tree → deep) sebagai progresi naratif "mengapa deep learning diperlukan", lalu deep-dive pada backbone utama WDCNN dengan SHAP DeepExplainer dan formalisasi *Fault Signature Maps*.

**Sumber utama:** Conference 1 (kernel), Conference 2 (tree), Journal 1 (deep + FSM); notebook [Notebook/Conference1_*.ipynb](../Notebook/Conference1_Classification_SVM_LR.ipynb), [Notebook/Conference2_*.ipynb](../Notebook/Conference2_Classification_Tree.ipynb), [Notebook/Journal1_*.ipynb](../Notebook/Journal1_Fault%20Signature%20Maps.ipynb).

### [ Metodologi Studi Diagnostik ]

#### 4.1 Pipeline Diagnostik CWRU — Overview
- *Diagram alur:* CWRU sinyal mentah → (jalur fitur HI) → kernel/tree → SHAP fitur; (jalur sinyal mentah) → WDCNN → SHAP DeepExplainer → FSM.
- Justifikasi naratif: kernel/tree sebagai *interpretable baseline* yang dapat dijalankan di Edge IoT; WDCNN sebagai backbone utama dengan akurasi tertinggi + interpretabilitas level sinyal.
- **(figure)** `fig:bab4_pipeline_diagnostik`: diagram TikZ alur tiga keluarga + SHAP variant per keluarga + FSM untuk WDCNN.

#### 4.2 Benchmark Keluarga Kernel — SVM + Logistic Regression (Conf 1)
- **Sumber notebook:** [Notebook/Conference1_Classification_SVM_LR.ipynb](../Notebook/Conference1_Classification_SVM_LR.ipynb).
- **Ekstraksi fitur:** 9 fitur time-domain (RMS, kurtosis, crest factor, peak-to-peak, skewness, std, mean, variance, MAD) + 9 fitur frequency-domain (spectral centroid, spectral entropy, energi pita BPFO/BPFI/BSF/FTF) per *window* (geometri SKF 6205).
- **Standardisasi:** Z-score *fit-on-train*.
- **Model:** SVM RBF + Logistic Regression *one-vs-rest* (10 kelas).
- **Tuning:** GridSearchCV; SVM `C ∈ {0.1, 1, 10, 100}`, `γ ∈ {scale, auto, 0.01, 0.1}`; LR `C ∈ {0.1, 1, 10}`.
- **Split:** stratified 80/20 dengan `random_state` tetap; *5-fold stratified CV* pada train set untuk tuning.

#### 4.3 Benchmark Keluarga Tree — DT + RF + XGBoost (Conf 2)
- **Sumber notebook:** [Notebook/Conference2_Classification_Tree.ipynb](../Notebook/Conference2_Classification_Tree.ipynb).
- **Fitur:** sama dengan §4.2 (9 + 9 dimensi).
- **Model:** Decision Tree, Random Forest, XGBoost — kriteria splitting Gini/entropy; *bagging* untuk RF; *gradient boosting* + regularisasi L1/L2 untuk XGBoost.
- **Tuning:** grid hyperparameter spesifik per model (detail di [Lampiran F](#lampiran-f-detail-klasifikasi-tree-based)).
- **Catatan teknis:** label encoding string → integer untuk XGBoost.

#### 4.4 Backbone Deep: WDCNN (Journal 1)
- **Sumber notebook:** [Notebook/Journal1_Fault Signature Maps.ipynb](../Notebook/Journal1_Fault%20Signature%20Maps.ipynb).
- **Arsitektur (Zhang 2017/2018):** Kernel-1 lebar **64**, stride **16** (untuk *frequency-aware feature extraction*); blok konvolusi 2–5 dengan kernel 3; dua lapisan FC(100) dengan dropout 0.5; softmax 10 kelas. Total **~60 710 parameter**.
- **Input:** sinyal mentah 2 048 sampel (tanpa ekstraksi fitur manual).
- **Tiga varian ablasi:**
  - **(A) Baseline:** WDCNN penuh (kernel 64, dengan BatchNorm).
  - **(B) Kernel sempit:** kernel-1 = 3 (bukan 64) → uji peran kernel lebar.
  - **(C) Tanpa BatchNorm:** baseline minus BN → uji peran BN pada FSM discriminability.
- **Pelatihan:** Adam lr=0.001, StepLR, CrossEntropyLoss, early stopping patience=15.

#### 4.5 SHAP per Keluarga
- **KernelExplainer untuk SVM/LR (§4.2).** *Background* berukuran kecil (stratified sampling dari training set); *test sample* terstratifikasi; perhitungan SHAP per kelas.
- **TreeExplainer untuk DT/RF/XGBoost (§4.3).** Eksak (Lundberg 2018); kompatibel multi-class.
- **DeepExplainer untuk WDCNN (§4.4).** DeepLIFT-based (Shrikumar 2017); background 300 sampel terstratifikasi (30/kelas); target eksplanasi 500 sampel uji (50/kelas); output: **10 array atribusi (500 × 2 048) = 10,24 juta nilai SHAP per model.**
- **(NEW)** Catatan kritis: ketiga *explainer* dipilih sesuai keluarga model untuk efisiensi + akurasi atribusi.

#### 4.6 Formalisasi Fault Signature Maps (FSM)
- **Definisi tiga varian:**
  - **Signed FSM:** $\text{FSM}_c^{\text{signed}}[p] = \frac{1}{|\mathcal{D}_c|} \sum_{x \in \mathcal{D}_c} \phi^{(c)}(x)[p]$ — rata-rata atribusi dengan tanda.
  - **Absolute FSM:** $\text{FSM}_c^{\text{abs}}[p] = \frac{1}{|\mathcal{D}_c|} \sum_{x \in \mathcal{D}_c} |\phi^{(c)}(x)[p]|$ — *fingerprint* utama.
  - **Variance FSM:** variansi $\phi^{(c)}(x)[p]$ antar sampel — uji stabilitas.
- **Tiga metrik validasi:**
  - (i) *Discriminability index* — sejauh mana FSM antar-kelas berbeda satu sama lain.
  - (ii) *Severity monotonicity fraction* — fraksi posisi di mana FSM meningkat monoton sejalan severity (0.007 → 0.014 → 0.021 inch).
  - (iii) *Split-half stability coefficient* — korelasi FSM antara dua subset random separuh dari data uji.

### [ Hasil dan Analisis Studi Diagnostik ]

#### 4.7 Hasil Benchmark Kernel (SVM + LR)
- **Tabel:** hyperparameter terbaik dari GridSearchCV (SVM `C=*`, `γ=*`; LR `C=*`).
- **Tabel:** Accuracy, macro-F1, weighted-F1 + *confusion matrix* 10 × 10 per model.
- **Figure:** SHAP global *bar chart* (mean |SHAP|) untuk SVM dan LR; per-class stacked bar.
- **Diskusi:** fitur teratas paling kontributif (umumnya: kurtosis, crest factor, energi BPFI/BPFO); kapan SVM mengungguli LR (boundary non-linear).

#### 4.8 Hasil Benchmark Tree (DT + RF + XGBoost)
- **Tabel:** hyperparameter terbaik per model.
- **Tabel:** Accuracy, F1, *confusion matrix* per model.
- **Figure:** SHAP *summary plot* untuk DT, RF, XGB (per-class jika ada ruang).
- **Figure:** perbandingan *feature ranking* antar tree models — umumnya XGB > RF > DT untuk akurasi.
- **Diskusi:** XGBoost diharapkan pemenang tree-family; trade-off antara akurasi dan interpretabilitas DT terhadap ensemble.

#### 4.9 Kinerja Klasifikasi WDCNN
- **Akurasi 99,87%** (749/750 sampel uji). **Macro F1 = 0,997.** Early stopping pada epoch ke-54.
- Hanya **satu misklasifikasi:** Ball_014 → IR_014.
- Perbandingan terhadap SVM-RBF 96,4% (§4.7) → **reduksi error pengujian 91%** (dari 27 ke 1 sampel).
- **(figure)** `fig:bab4_wdcnn_training_curves` — loss/akurasi train+val.
- **(figure)** `fig:bab4_wdcnn_cm` — confusion matrix 10 × 10.

#### 4.10 Pola Atribusi SHAP WDCNN
- **Tiga karakteristik global:**
  - (a) *Suppression tepi*: atribusi hampir nol di posisi 0–100 dan 1 900–2 048 (zona padding/edge).
  - (b) Modulasi ~16 sampel → **artefak stride** layer pertama (stride=16).
  - (c) Distribusi merata |φ| ≈ 0,015–0,025 pada zona tengah.
- **Profil per kelas:** Normal (rendah, merata) → Ball (difus, sedikit lebih tinggi) → IR (datar, fokus medium) → OR (terlokalisasi, puncak tajam). **OR_021** puncak |φ| ≈ 0,08 di posisi 600–700.
- **(figure)** `fig:bab4_shap_overlay` — overlay sinyal mentah + atribusi SHAP untuk satu sampel per kelas kerusakan.

#### 4.11 Validasi FSM (Tiga Metrik)
- **Split-half stability = 0,940** → FSM sangat stabil antar subset.
- **Discriminability index = 0,216** (baseline A).
- **Severity monotonicity:** Ball 17,6% > IR 13,8% > OR 8,6%. Interpretasi: severity Ball paling mudah diranking dari pola atribusi; OR paling sulit (puncak sudah terlokalisasi sejak severity rendah).
- **(figure)** `fig:bab4_fsm_heatmap` — FSM heatmap 10 kelas × 2 048 posisi (absolute variant).
- **(figure)** `fig:bab4_fsm_severity` — overlay FSM tiga severity (0.007/0.014/0.021 inch) per fault type.

#### 4.12 Interpretasi Fisika: Morfologi dan Periodisitas
- **Temuan kritis:** Puncak FSM **TIDAK selaras** dengan periode teoretis BPFO/BPFI/BSF (verifikasi melalui cross-correlation FSM ↔ delta-train di periode BPFx).
- **Inferensi:** WDCNN mengklasifikasi melalui **morfologi transien impuls** (durasi efektif ≈ 42,67 ms, sesuai kernel-1 width 64 / 1.5 kHz envelope), **bukan melalui periodisitas BPFx**.
- **Implikasi praktis:** WDCNN tidak butuh informasi geometri bearing atau kecepatan poros — **lebih transferable lintas mesin** (asalkan modus kegagalan menghasilkan impuls transien serupa).
- Catatan teoretis: konsisten dengan hipotesis bahwa CNN 1D dengan kernel lebar bertindak sebagai *envelope detector* terlokalisasi.

#### 4.13 Studi Ablasi Trade-off BatchNorm (N2 — Temuan Pertama)
| Varian | Deskripsi | Akurasi | Discriminability | Parameter |
|--------|-----------|---------|------------------|-----------|
| **A** | Baseline (kernel 64 + BN) | 99,73% | 0,216 | ~60 710 |
| **B** | Kernel sempit k=3 + BN | ↓ 0,80 pp | ↓ 17% | **÷ 7,3×** |
| **C** | Baseline tanpa BatchNorm | ↓ 3,60 pp | **↑ 133% (0,735)** | ~60 710 |

- **Insight kunci:** BatchNorm meningkatkan akurasi tapi **menekan discriminability FSM** sebesar 133%. Temuan pertama dalam literatur klasifikasi bantalan.
- **Panduan desain (Bab VI):** Varian **A** untuk akurasi kritis (Tier 2 Edge Server); Varian **C** untuk interpretabilitas maksimum (audit, riset, R&D).
- **(figure)** `fig:bab4_fsm_ablation` — overlay FSM tiga varian A, B, dan C untuk satu kelas representatif (OR_021).

#### 4.14 Cross-Family Synthesis
- **Tabel pemenang per metrik:** akurasi, F1 per kelas minoritas, parameter, latency inferensi (estimasi Edge).
- **Diskusi:**
  - WDCNN unggul akurasi tapi butuh data lebih banyak + GPU inferensi.
  - XGBoost trade-off optimal untuk *edge deployment* (Tier 2).
  - SVM/LR sebagai *interpretable baseline* yang dapat berjalan di Tier 1 (Edge IoT).
- **Konvergensi ranking fitur antar keluarga:** kurtosis, crest factor, dan energi BPFI/BPFO konsisten muncul di top-5 SHAP kernel/tree → **validasi silang diagnostik** yang memperkuat kepercayaan praktisi.
- Posisi WDCNN+FSM: melampaui keluarga kernel/tree dalam akurasi DAN menambahkan interpretabilitas level sinyal yang tidak tersedia di kernel/tree.

---

## Bab V — Remaining Useful Life: Metodologi dan Hasil (Prognostik)

**Tujuan bab:** Menyajikan jalur prognostik secara end-to-end — tiga backbone RUL → Top-k Sparse Autoencoder → pemetaan ke frekuensi karakteristik bantalan (BPFx) dengan validasi statistik *bootstrap CI* + *permutation test* + dua *negative controls*.

**Sumber utama:** Journal 2 (Mamba-xLSTM-Net + Top-k SAE + BPFx mapping); chapters [04-metodologi.tex](disertation/chapters/04-metodologi.tex) §4.5–4.10 + [05-hasil-pembahasan.tex](disertation/chapters/05-hasil-pembahasan.tex).

### [ Metodologi Studi Prognostik ]

#### 5.1 Arsitektur Backbone RUL
*Tiga arsitektur yang dipakai sebagai *substrate* untuk Top-k SAE.*

- **Mamba-xLSTM-Net.** Gabungan *selective state-space model* Mamba-3 (konteks panjang efisien) + *matrix-memory* mLSTM (presisi rekuren) + *gated fusion*. 898 K params (PHM) / 811 K (XJTU). Detail di [Lampiran B](disertation/lampiran/B-hyperparameter-baseline.tex).
- **N-BEATS-xLSTM-RUL.** Basis-block (trend / wear / shock) + xLSTM residual stack. Memanfaatkan *prior struktural* kurva degradasi. 459 K params.
- **SparseGate-TCN-RUL.** Dilated causal convolution + sparse gating. Latency rendah → baseline ringan untuk edge inferensi. 249 K params.
- **Protokol training bersama:** 75 epoch, mixed precision bf16, batch 512 × window 32, seed {42, 43, 44}, checkpoint pada loss validasi terbaik, optimizer AdamW.

#### 5.2 Top-k Sparse Autoencoder
- **Arsitektur:** encoder $W_\text{enc} \in \mathbb{R}^{d_\text{lat} \times d}$, decoder $W_\text{dec}$, bias pre-encoder $b_\text{pre}$. Latent dimension $d_\text{lat} = 8d = 1\,024$. Sparsity $k = 51$ (~5% aktif).
- **Pelatihan:** pool $N = 20\,000$ hidden states (backbone frozen), AdamW lr=1e-3, 50 epoch, loss MSE rekonstruksi.
- **Prinsip:** mendorong **monosemantisitas** — setiap fitur SAE mengkodekan satu konsep fisika, bukan campuran polysemantic (analogi LLM Bricken-Cunningham 2023).

#### 5.3 Prosedur Pemetaan SAE → BPFx
- **Hilbert envelope spectrum:** band-pass pada resonansi dominan (band dipilih melalui *spectral kurtosis*), envelope $A_r(t)$, spektrum envelope $A_{r,\varphi}(f) = \int_{[\varphi - \Delta, \varphi + \Delta]} A_r(f) df$ untuk $\varphi \in \{\text{BPFO}, \text{BPFI}, \text{BSF}, \text{FTF}\}$.
- **Korelasi Pearson** $r_{i, \varphi}$ antara profil aktivasi fitur SAE $z_{r, i}$ (untuk *recording* $r$) dan amplitudo BPFx $A_{r, \varphi}$ lintas semua *recording* dalam dataset.
- **Hit-rate** $H_\varphi = |\{i : |r_{i, \varphi}| \geq 0{,}30\}| / d_\text{lat}$. Threshold 0,30 = korelasi moderat secara praktis (rule-of-thumb domain sinyal).

#### 5.4 Inferensi Statistik dan Negative Controls
- **Bootstrap B=1 000** resample lintas *recording* → **95% CI** untuk hit-rate H.
- **Permutation test B=1 000** shuffle label *recording* terhadap amplitudo → **p-value two-sided** untuk hit-rate.
- **Control 1 — Model belum-terlatih (Xavier init):** apakah korespondensi yang terdeteksi adalah artefak arsitektur backbone?
- **Control 2 — Gaussian noise hidden states:** apakah korespondensi artefak prosedur SAE itu sendiri?
- **Sparsity sweep:** $k \in \{10, 51, 102, 205\}$ untuk uji robustness pilihan $k$.

### [ Hasil dan Analisis Studi Prognostik ]

#### 5.5 Kinerja Backbone RUL
*Setiap backbone harus viable secara prognostik sebelum representasi latennya bermakna untuk diuji secara fisika.*

| Dataset | Pemenang | RMSE (mean ± std, 3 seed) |
|---------|----------|---------------------------|
| **PHM2012** | SparseGate-TCN | **0,226 ± 0,030** |
| **XJTU-SY** | N-BEATS-xLSTM | **0,259 ± 0,003** |
| **IMS** | Mamba-xLSTM | **0,407 ± 0,040** |

- **Catatan:** Ketiga backbone *viable* lintas dataset; tidak ada pemenang universal → semua layak diperiksa makna fisik representasi latennya.
- **(figure)** `fig:bab5_rul_curves` — kurva prediksi terhadap *ground-truth* untuk satu bearing representatif per dataset.

#### 5.6 Hit-Rate BPFx — Hasil Utama (N3 + N4)
| Dataset | BPFI | BPFO | BSF | FTF | p-value | Konsistensi fisis |
|---------|------|------|-----|-----|---------|-------------------|
| **PHM2012** | **2,3%** | 2,0% | 0 | 0 | < 0,001 | Mixed inner + outer race spalling |
| **XJTU-SY** | 0 | **2,2%** | 0,3% | 0 | < 0,001 | Outer-race spalling LDK UER204 |
| **IMS** | **1,76%** | 0 | 0,49% | 0 | 0,001 | Cage + rolling-element fatigue (B3 + B4) |
| **CWRU** | **5,08%** | 0 | 0 | 0 | > 0,05 (underpowered, n=10) | BPFI dominant — konsisten distribusi IR |

- **Statistik tambahan:** korelasi maksimum $r_\text{max} = 0{,}507$ (PHM2012, BPFI); $r_\text{max} = 0{,}501$ (XJTU-SY, BPFO).
- **(figure)** `fig:bab5_hitrate_panel` — bar chart hit-rate per dataset × per BPFx, dengan annotations p-value.
- **(figure)** `fig:bab5_correlation_scatter` — scatter $z_{r,i}$ terhadap $A_{r,\varphi}$ untuk fitur SAE top-1 di tiap dataset.

#### 5.7 Negative Controls — Eliminasi Artefak
- Hit-rate jatuh drastis (mendekati 0) pada **Control 1** (model Xavier-init) dan **Control 2** (Gaussian noise hidden states).
- **Inferensi:** korespondensi SAE↔BPFx adalah **emergent property** representasi yang terlatih, **bukan artefak arsitektur** maupun **artefak prosedur SAE**.
- **(figure)** `fig:bab5_negative_controls` — bar chart komparatif hit-rate antara model terlatih, Control 1, dan Control 2 per dataset.

#### 5.8 Perbandingan Lintas Arsitektur
- **PHM2012:** Mamba-xLSTM memimpin BPFI (hit-rate tertinggi) → arsitektur lebih kuat secara temporal → alignment fisika lebih kaya.
- **XJTU-SY:** Mamba-xLSTM **BPFO-dominant (4,69%)**; N-BEATS-xLSTM dan SparseGate-TCN menyebar → **bias induktif state-space mengamplifikasi tanda outer-race**.
- **CWRU (cross-dataset validation):** Ketiga arsitektur konvergen ke **BPFI-dominant (4,69–5,08%)** → distribusi data mendominasi, bukan bias arsitektur. Tetapi p-value > 0,05 (n=10 recording — underpowered).
- **Diskusi:** universalitas hit-rate BPFx lintas arsitektur memperkuat klaim N3+N4.

#### 5.9 Sparsity Sweep
- **k = 51 (~5% aktif)** = pilihan optimal untuk semua dataset.
- **k < 10:** beberapa BPFx kehilangan signifikansi (under-allocation kapasitas).
- **k > 102:** noise BPFx sekunder meningkat (over-allocation → polysemantic).
- **(figure)** `fig:bab5_sparsity_sweep` — kurva hit-rate per dataset terhadap k.

---

## Bab VI — Kesimpulan dan Rekomendasi

**Tujuan bab:** Menutup disertasi dengan **sintesis dua-jalur** + kerangka PdM multi-tier + rekapitulasi N1–N5 + keterbatasan + rekomendasi.

### 6.1 Pembahasan Terintegrasi
- **FSM (Bab IV)** = *input attribution*, level sinyal, cakrawala waktu **~42,67 ms** (durasi transien impuls).
- **SAE (Bab V)** = *latent concept*, level hidden state, cakrawala waktu **ratusan akuisisi degradasi**.
- Dua lapisan interpretabilitas berbeda — **saling melengkapi, bukan redundan**.
- **Paradoks yang tampak:** FSM tidak sejajar dengan periode BPFx, tetapi SAE berkorelasi dengan amplitudo BPFx envelope. **Bukan kontradiksi** — mencerminkan perbedaan cakrawala waktu: WDCNN klasifikasi melalui morfologi transien (window pendek); backbone RUL belajar dari evolusi degradasi lintas ratusan akuisisi (window panjang) → menyerap fisika BPFx di latent.
- **Rantai konsistensi lintas-lapisan:**
  ```
  kurtosis / crest factor (fitur agregat, Bab IV §4.7–4.8)
      ↓
  impuls terlokalisasi (FSM/sinyal, Bab IV §4.10–4.12)
      ↓
  BPFx di latent space (SAE, Bab V §5.6–5.8)
  ```
  Ketiga lapisan mengeksploitasi tanda fisik *underlying* yang sama; tidak ada kontradiksi antar-skala.

### 6.2 Kerangka PdM Multi-Tier (N5)
| Tier | Lokasi | Model | XAI | Output |
|------|--------|-------|-----|--------|
| **Tier 1 — Edge IoT** | Sensor / IMx-8 | SVM/LR (Conf 1) | SHAP KernelExplainer fitur | Triage anomali (binary alert) |
| **Tier 2 — Edge Server** | Gateway pabrik | WDCNN + FSM (Journal 1) | SHAP DeepExplainer + FSM | Konfirmasi jenis & keparahan, akurasi 99,87% |
| **Tier 3 — Cloud / GPU** | AWS / on-prem | Backbone RUL + SAE-BPFx (Journal 2) | SAE-BPFx mapping + statistical tests | Estimasi RUL + audit latent↔physics |

- **(figure)** `fig:bab6_pdm_multitier` — arsitektur tiga-tier dengan aliran data + jenis interpretabilitas.
- **Implikasi:** integrasi dengan ERP/CMMS untuk *closed-loop maintenance scheduling*.

### 6.3 Kesimpulan
- **Lima novelti (N1–N5) dengan status pembuktian eksplisit:**
  - **N1 — empiris:** split-half stability 0,940; discriminability 0,216; severity monotonicity Ball 17,6% / IR 13,8% / OR 8,6%.
  - **N2 — empiris:** discriminability +133% pada varian tanpa BN; akurasi −3,60 pp.
  - **N3 — empiris:** hit-rate BPFI 2,3% (PHM2012, p < 0,001); BPFO 2,2% (XJTU-SY, p < 0,001).
  - **N4 — empiris:** universalitas BPFx-dominant lintas 3 arsitektur × 4 dataset; hit-rate jatuh ke ~0 pada Control 1 + Control 2.
  - **N5 — konseptual/sintesis:** blueprint PdM tiga-tier (penempatan model per tier berdasarkan parameter, latency, tipe XAI). **Bukan** klaim hasil *deployment* lapangan.
- **Kontribusi inti:** *jembatan pertama* antara data-driven AI (deep learning) dan classical vibration theory (BPFx) yang **dapat diverifikasi statistik** dengan bootstrap CI + permutation test + 2 negative controls.
- **Posisi terhadap SK-Toto §I.3 (di bawah pemetaan ulang Tujuan ↔ RQ ↔ Novelti yang baru):**
  - **Tujuan 1 (RQ1 — sistem PdM multi-tier):** dijawab pada level **blueprint arsitektur** (N5 konseptual). Validasi *deployment* lapangan = future work.
  - **Tujuan 2 (RQ2 — validasi konsistensi hybrid ML+DL):** dijawab **penuh** secara empiris (N4) lintas 4 dataset × 3 arsitektur diagnostik + 3 backbone prognostik. Integrasi multi-modal *radial clearance* tetap di luar lingkup empiris → future work (§VI.5).
  - **Tujuan 3 (RQ3 — transparansi XAI):** dijawab **penuh** secara empiris dengan tiga novelti empiris berlapis (N1 input-attribution; N2 ablation; N3 latent-concept).

### 6.4 Keterbatasan
- *Single operating condition* per dataset (PHM2012 fokus pada 3 kondisi beban tetap, XJTU-SY pada satu).
- **CWRU = seeded faults** (EDM) → tidak sepenuhnya mewakili kegagalan natural di industri.
- **SAE post-hoc** — belum dalam loop pelatihan backbone (joint training berpotensi lebih monosemantik).
- **CWRU underpowered untuk SAE** (n=10 recording → p > 0,05 meski hit-rate tinggi 5,08%).
- **Integrasi data kualitas radial clearance PT~SKF belum dieksekusi** — dijadikan future work eksplisit (SK-Toto Manfaat #1).
- Validasi industri PT~SKF: hanya *qualitative external sanity check* (subset N titik), bukan *cross-machine validation* lengkap.

### 6.5 Rekomendasi Penelitian Lanjutan
1. **Validasi cross-rig:** Paderborn (KAt-DataCenter) + MFPT — uji transferability WDCNN+FSM lintas rig + spindle speed.
2. **SAE in-loop:** integrasikan Top-k SAE dalam loop pelatihan backbone (joint loss) → uji apakah monosemantisitas meningkat.
3. **Normalizer-Free Networks (NF-Net):** alternatif BatchNorm yang potensial mempertahankan akurasi tanpa menekan discriminability FSM (testing untuk konfirmasi N2).
4. **SHAP pada STFT / CWT:** XAI di domain time-frequency untuk komplementaritas dengan FSM (sinyal mentah).
5. **Edge real-time deployment:** target inferensi WDCNN < 100 ms pada SoC (NVIDIA Jetson Orin Nano atau RK3588).
6. **Integrasi ERP / CMMS:** closed-loop maintenance scheduling berbasis hit-rate BPFx + RUL forecast.
7. **Future work prioritas (SK-Toto Tujuan 2):** **kumpulkan data kualitas radial clearance PT~SKF dalam volume signifikan dan integrasikan dengan vibrasi sebagai input multi-modal model RUL** — penyelesaian Tujuan 2 SK-Toto secara penuh.

---

## Lampiran

**Tujuan:** Detail teknis yang panjang/repetitif yang mendukung Bab III–V.

### Lampiran A — Detail Fitur Health Indicator 36-D
- (retained) Definisi 9 fitur time-domain + 9 fitur frequency-domain + 5 band-energy per kanal, dengan formula matematis dan kode referensi `hi.py`.

### Lampiran B — Hyperparameter Baseline (RUL)
- (retained) Tabel lengkap hyperparameter Mamba-xLSTM-Net (898 K params PHM, 811 K XJTU), N-BEATS-xLSTM-RUL (459 K), SparseGate-TCN-RUL (249 K). Optimizer, lr scheduler, gradient clipping, weight decay.

### Lampiran C — Kode Implementasi Kunci
- (retained, perluas) Snippet kode reproduksi untuk pelatihan + evaluasi backbone RUL. **Tambahkan** link ke tiga notebook diagnostik: Conf 1, Conf 2, Journal 1.

### Lampiran D — Klasifikasi Industri PT~SKF Indonesia (Softened)
- **(REVISED)** Konteks: mesin OR1/OR2, akuisisi melalui SKF~IMx-8 → AWS Cloud (1 paragraf).
- Volume data SKF: hanya **N titik** (perlu dikonfirmasi dengan Pak Toto saat eksekusi); insufisien untuk training/test split penuh.
- **Skema:** gunakan SKF data hanya untuk **inferensi** dengan classifier yang sudah dilatih pada CWRU; bandingkan label prediksi dengan label kondisi yang diobservasi tim PdM SKF.
- **Tidak ada klaim** *cross-machine validation* lengkap; klaim hanya *qualitative external sanity check*.
- **Roadmap data:** sub-bab kecil tentang rencana pengumpulan data radial clearance + integrasi multi-modal (mendukung SK-Toto Tujuan 2 → future work).

### Lampiran E — Detail Klasifikasi SVM/LR (NEW; sumber: Conference 1)
- Tabel hyperparameter GridSearchCV lengkap (C, γ, kernel, class_weight).
- Figure: SHAP summary plot untuk SVM dan LR (per-class breakdown).
- Tabel: per-class precision/recall/F1.
- Confusion matrix figure (10 × 10).
- Kode reproduksi training + SHAP (snippet dari [Notebook/Conference1_*.ipynb](../Notebook/Conference1_Classification_SVM_LR.ipynb)).

### Lampiran F — Detail Klasifikasi Tree-Based (NEW; sumber: Conference 2)
- Tabel hyperparameter DT, RF, XGBoost (max_depth, n_estimators, learning_rate, subsample, reg_lambda).
- Figure: SHAP summary plot per model.
- Tabel: per-class precision/recall/F1 per model.
- Confusion matrix figure × 3 model.
- Catatan tentang label encoding (string → integer) yang diperlukan XGBoost.
- Kode reproduksi (snippet dari [Notebook/Conference2_*.ipynb](../Notebook/Conference2_Classification_Tree.ipynb)).

### Lampiran G — Detail WDCNN dan FSM (NEW; sumber: Journal 1)
- Arsitektur WDCNN lengkap (diagram + tabel layer dengan kernel size, stride, padding, output dim).
- Hyperparameter training (Adam lr=0.001, StepLR γ=0.5 step=20, batch 64, epoch 100 max, early stopping patience 15).
- Figure: training curves lengkap (loss + accuracy train/val).
- Figure: FSM heatmap full-resolution (Absolute variant, 10 kelas × 2 048 posisi).
- Tabel ablasi: variasi kernel size lapisan pertama (32 / 64 / 128) → efek pada akurasi + discriminability.
- Catatan reproduksi SHAP DeepExplainer pada sinyal mentah 2 048 sampel (dari [Notebook/Journal1_*.ipynb](../Notebook/Journal1_Fault%20Signature%20Maps.ipynb)).

---

## Bagian Awal (Front Matter)

### Abstrak Bahasa Indonesia ([00-abstrak-id.tex](disertation/chapters/00-abstrak-id.tex))
**(REVISED)** Tulis ulang menjadi 4 paragraf:
- **Paragraf 1 — Konteks & motivasi.** Making Indonesia 4.0, bantalan = 40–50% kegagalan rotating; gap kepercayaan industrial (XAI level sinyal + bukti fisika di latent RUL belum ada); konteks PT~SKF lembut.
- **Paragraf 2 — Jalur Diagnostik.** Benchmark tiga keluarga (kernel/tree/deep) pada CWRU; SHAP per keluarga; **WDCNN + FSM, akurasi 99,87%, split-half stability 0,940, temuan BatchNorm trade-off**.
- **Paragraf 3 — Jalur Prognostik.** Tiga backbone RUL (Mamba-xLSTM, N-BEATS-xLSTM, SparseGate-TCN) lintas PHM2012/XJTU-SY/IMS; **Top-k SAE → BPFx mapping; hit-rate BPFI 2,3% (p<0,001), $r_\text{max} = 0{,}507$**; negative controls mengeliminasi artefak.
- **Paragraf 4 — Sintesis & kontribusi.** Lima novelti N1–N5; kerangka PdM multi-tier; jembatan pertama data-driven AI ↔ classical vibration theory yang dapat diverifikasi statistik.
- **Angka utama** per jalur disertakan (akurasi 99,87% / RMSE 0,226 / hit-rate BPFI 2,3% / $r_\text{max}$ 0,507).

### Abstract English ([00-abstract-en.tex](disertation/chapters/00-abstract-en.tex))
- Same 4-paragraph restructuring, English translation.

### Kata Pengantar ([00-kata-pengantar.tex](disertation/chapters/00-kata-pengantar.tex))
- (retained) Acknowledgments retained; sentuhan minor untuk acknowledgment baru jika perlu (data PT~SKF, kolaborator notebook).

### Daftar Singkatan ([00-daftar-singkatan.tex](disertation/chapters/00-daftar-singkatan.tex))
**(ADD)** WDCNN, FSM, BPFI, BPFO, BSF, FTF, SAE, SSM, mLSTM, sLSTM, CWRU, PHM, XJTU, IMS, DT, RF, XGB, SHAP, OEE, IoT, IMx-8, CBM, EDM, CV, CI.

---

## Referensi ([references.bib](disertation/references.bib))

**Entri baru yang perlu ditambahkan:**

```bibtex
@inproceedings{TotoSuharto2024Conf1SVM,
  title  = {Bearing Fault Classification with SVM/Logistic Regression and SHAP KernelExplainer on CWRU},
  author = {Suharto, Toto and ...},
  ...
}
@inproceedings{TotoSuharto2024Conf2Tree,
  title  = {Tree-Based Ensemble (DT/RF/XGBoost) for Bearing Fault Classification with SHAP TreeExplainer on CWRU},
  ...
}
@article{TotoSuharto2025Journal1FSM,
  title  = {Fault Signature Maps: Signal-Level Interpretability for WDCNN Bearing Diagnostics},
  ...
}
@article{TotoSuharto2025Journal2RUL,
  title  = {Bridging Deep Learning and Bearing Physics: Top-k Sparse Autoencoders for Latent Concept Interpretability in RUL Prediction},
  ...
}
@article{Zhang2018WDCNN,        title = {Wide-Deep CNN for Bearing Fault Diagnosis under Variable Working Conditions}, ...}
@inproceedings{Shrikumar2017DeepLIFT, title = {Learning Important Features Through Propagating Activation Differences}, ...}
@article{Elhage2022Superposition,    title = {Toy Models of Superposition}, ...}
@article{Bricken2023TopkSAE,         title = {Towards Monosemanticity: Decomposing Language Models with Dictionary Learning}, ...}
@misc{CWRU2014Dataset,               title = {Case Western Reserve University Bearing Data Center}, ...}
```

**Aksi eksekusi:** Verifikasi venue + tahun publikasi aktual + halaman + DOI dari empat papers Pak Toto sebelum membuat bib entry final.

---

## Catatan Eksekusi

1. **Urutan implementasi:**
   - **Fase A** — Restrukturisasi [01-pendahuluan.tex](disertation/chapters/01-pendahuluan.tex) selaras SK-Toto (1.1 → 1.7 di outline ini).
   - **Fase B** — Split [03-dasar-teori.tex](disertation/chapters/03-dasar-teori.tex) + [04-metodologi.tex](disertation/chapters/04-metodologi.tex) menjadi (a) Bab III shared, (b) Bab IV metodologi diagnostik, (c) Bab V metodologi prognostik.
   - **Fase C** — Restrukturisasi [05-hasil-pembahasan.tex](disertation/chapters/05-hasil-pembahasan.tex) menjadi (a) Bab IV hasil diagnostik dengan benchmark kernel/tree/deep, (b) Bab V hasil prognostik.
   - **Fase D** — Tambahkan Lampiran E/F/G; lembutkan Lampiran D.
   - **Fase E** — Restrukturisasi abstrak + daftar singkatan.
   - **Fase F** — Update [06-kesimpulan.tex](disertation/chapters/06-kesimpulan.tex) selaras VI.1–VI.5.
2. **Komitmen kecil-kecil:** setiap perubahan section di-commit terpisah agar mudah di-review oleh Pak Toto.
3. **Verifikasi build setelah setiap section:** `make build` di [writings/disertation/](disertation/).
4. **Bahasa:** Bahasa Indonesia. Istilah teknis Inggris ditulis dengan `\emph{...}` (italik). Hindari *Indonesianisasi* paksa terhadap istilah baku (*deep learning*, *Sparse Autoencoder*, *backpropagation*, *envelope spectrum*).
5. **Sitasi-diri:** gunakan format `\citetitb{TotoSuharto20XXKey}` mengikuti konvensi yang sudah ada di disertasi.
6. **Konsistensi penomoran:** setelah Bab IV/V direstrukturisasi (kernel/tree/deep di awal Bab IV), semua referensi `\autoref{}` ke section lain harus diperbarui.
7. **Konfirmasi data PT~SKF:** sebelum menulis Lampiran D versi softened, tanyakan ke Pak Toto: berapa titik data SKF yang tersedia? Apa labelnya? Bagaimana *ground-truth* ditetapkan?
8. **Konfirmasi venue paper:** sebelum membuat bib entry self-citation, dapatkan judul lengkap + venue + tahun + halaman + DOI dari Pak Toto.
9. **Verifikasi notebook ↔ paper alignment:** notebook Journal 2 belum ada di repositori — flag sebagai TODO untuk Pak Toto (kebutuhan reproduksi backbone RUL + Top-k SAE).
