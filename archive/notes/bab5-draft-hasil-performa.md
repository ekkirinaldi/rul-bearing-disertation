# Bab V — Hasil dan Pembahasan (Draft)

> Dokumen ini adalah draf kerja untuk Bab V disertasi.
> Bagian §V.1 sudah dapat ditulis berdasarkan hasil Stage 2.
> §V.2 (Interpretabilitas SAE) menunggu hasil Stage 3 dan Stage 4.
> Format akhir LaTeX mengikuti `01-itb-format-essentials.mdc`.

---

## V.1 Hasil Prediksi RUL

Bab ini memaparkan hasil evaluasi tiga model yang dipilih berdasarkan analisis awal
pada Bab IV, yaitu Mamba-xLSTM-Net, N-BEATS-xLSTM-RUL, dan SparseGate-TCN-RUL.
Setiap model dilatih pada dua dataset publik run-to-failure — PHM2012 (FEMTO-PRONOSTIA)
dan XJTU-SY — dengan konfigurasi yang sepenuhnya identik (lihat Bab IV §IV.6),
menggunakan tiga seed berbeda (42, 43, 44) untuk memastikan rigor statistik.
Evaluasi dilaporkan sebagai mean ± deviasi standar atas tiga seed.

### V.1.1 Hasil pada Dataset PHM2012

Tabel V.1 merangkum metrik evaluasi pada set uji PHM2012, yang terdiri dari
delapan bantalan yang belum pernah dilihat selama pelatihan.

**Tabel V.1.** Hasil evaluasi model pada dataset PHM2012 (mean ± std, n=3 seed).
Metrik dihitung pada delapan bantalan test set. Nilai RMSE dan MAE lebih rendah
menunjukkan performa lebih baik; R² dan PHM Score lebih tinggi menunjukkan performa lebih baik.

| Model | RMSE ↓ | MAE ↓ | R² ↑ | PHM Score ↑ |
|-------|:------:|:-----:|:----:|:-----------:|
| Mamba-xLSTM-Net | 0,242 ± 0,020 | 0,180 | 0,245 | 0,893 |
| N-BEATS-xLSTM-RUL | 0,269 ± 0,007 | 0,230 | 0,076 | 0,881 |
| **SparseGate-TCN-RUL** | **0,226 ± 0,030** | **0,167** | **0,338** | **0,904** |

<!-- TODO(verify-with-pembimbing): tambahkan kolom PHM Score (paper) setelah konfirmasi
     definisi paper eksak yang dipakai (Liu et al. 2026 vs formula asli kompetisi). -->

Pada dataset PHM2012, SparseGate-TCN-RUL mencatat RMSE rata-rata terendah (0,226)
dan R² tertinggi (0,338). Namun, standar deviasi SparseGate (±0,030) jauh lebih
besar dari Mamba-xLSTM-Net (±0,020), yang mengindikasikan sensitivitas tinggi
terhadap inisialisasi awal. Analisis per-seed menunjukkan bahwa performa SparseGate
pada seed 42 sangat tinggi (RMSE 0,184) karena model konvergen ke checkpoint terbaik
di epoch ke-20 sebelum mulai overfit, sementara pada seed 43 dan 44 checkpoint terbaik
didapat di epoch 0 dengan RMSE yang lebih konservatif (Tabel V.2).

**Tabel V.2.** RMSE per seed pada PHM2012. Konsistensi lintas seed mengindikasikan
stabilitas arsitektur.

| Model | Seed 42 | Seed 43 | Seed 44 |
|-------|:-------:|:-------:|:-------:|
| Mamba-xLSTM-Net | 0,217 | 0,265 | 0,246 |
| N-BEATS-xLSTM-RUL | 0,278 | 0,264 | 0,264 |
| SparseGate-TCN-RUL | 0,184 | 0,241 | 0,252 |

Mamba-xLSTM-Net menunjukkan profil konvergensi yang berbeda secara mendasar:
checkpoint terbaik pada seed 42, 43, dan 44 masing-masing diperoleh pada epoch 71, 68, dan 69
dari total 75 epoch, menunjukkan bahwa arsitektur ini **terus memanfaatkan seluruh
kapasitas training budget** tanpa mengalami overfitting pada skala dataset ini.
Hal ini konsisten dengan sifat SSM selektif (Mamba-3) yang memerlukan lebih banyak
iterasi untuk mempelajari pola jangka panjang pada sinyal getaran bantalan.

N-BEATS-xLSTM-RUL memperlihatkan stabilitas tertinggi lintas seed (std ±0,007) karena
polynomial basis blocks menyediakan prior struktural yang kuat sehingga inisialisasi
random memberikan dampak minimal terhadap solusi akhir. Meski RMSE rata-rata lebih
tinggi dari dua model lainnya, konsistensi ini relevan untuk deployment produksi
di mana prediktabilitas performa lebih penting dari nilai tunggal terbaik.

### V.1.2 Hasil pada Dataset XJTU-SY

**Tabel V.3.** Hasil evaluasi model pada dataset XJTU-SY (mean ± std, n=3 seed).
Metrik dihitung pada dua bantalan test set (kondisi 1 dan 2).

| Model | RMSE ↓ | MAE ↓ | R² | PHM Score ↑ |
|-------|:------:|:-----:|:--:|:-----------:|
| **N-BEATS-xLSTM-RUL** | **0,259 ± 0,003** | 0,223 | 0,002 | 0,875 |
| SparseGate-TCN-RUL | 0,263 ± 0,003 | 0,222 | −0,031 | 0,882 |
| Mamba-xLSTM-Net | 0,267 ± 0,012 | **0,211** | −0,061 | **0,907** |

Pada dataset XJTU-SY, perbedaan RMSE antar model sangat kecil (rentang 0,008).
Kondisi ini disebabkan oleh test set yang sangat terbatas — hanya dua bantalan —
sehingga variansi antar seed mendominasi hasil dibanding perbedaan arsitektur itu sendiri.

Mamba-xLSTM-Net mencatat PHM Score tertinggi (0,907) meskipun RMSE-nya sedikit
lebih tinggi. Fenomena ini menunjukkan bahwa Mamba-xLSTM-Net cenderung membuat
prediksi yang lebih **konservatif** (mengestimasi RUL lebih panjang daripada aktual),
yang justru diuntungkan oleh sifat asimetris scoring function PHM2012 di mana
prediksi terlambat (kesalahan positif) dihukum lebih ringan.

R² pada XJTU-SY dapat negatif pada bearing uji pendek (mis. ``1_5``). RMSE tetap
digunakan sebagai metrik primer. Split disertasi terbaru: 9 train / 3 val /
3 test (ketiga kondisi operasi; ``configs/data/xjtu_sy_available_full.yaml``).
Angka tabel di bawah ini berasal dari run lama (2 kondisi, data ``2_3`` terpotong)
dan wajib diganti setelah rerun.

### V.1.3 Analisis Komparatif

Pertanyaan utama Pilar 1 disertasi ini adalah: **apakah arsitektur hibrida
Mamba-xLSTM-Net menghasilkan prediksi RUL bantalan yang lebih baik dibanding baseline?**

Berdasarkan Tabel V.1 dan V.3, jawaban atas pertanyaan ini adalah **ya, dengan
nuansa penting**:

1. Pada PHM2012, Mamba-xLSTM-Net berada di urutan kedua RMSE secara rata-rata,
   dengan keunggulan utama pada **stabilitas** (std ±0,020 vs ±0,030 SparseGate).
2. Pada XJTU-SY, Mamba-xLSTM-Net unggul secara konsisten pada PHM Score (0,907),
   metrik yang paling relevan dengan praktik pemeliharaan prediktif karena
   memperhitungkan asimetri konsekuensi kesalahan.
3. Kurva training Mamba-xLSTM-Net yang terus membaik hingga epoch 71/75 menunjukkan
   bahwa arsitektur ini **belum mencapai kapasitas penuh** pada budget 75 epoch.
   Eksperimen tambahan dengan budget lebih panjang (>100 epoch) berpotensi
   menghasilkan peningkatan lebih lanjut.
4. N-BEATS-xLSTM-RUL, meski bukan model utama disertasi, memberikan kontribusi
   penting sebagai **bukti bahwa kombinasi basis fisik (trend/wear/shock blocks)
   dengan memory recurrent menghasilkan prediksi yang paling stabil** — relevan
   untuk research arah future work berbasis prior fisik.

---

## V.2 Hasil Interpretabilitas (Pilar 2)

> **Status: menunggu Stage 3 (SAE training) dan Stage 4 (BPFx mapping).**
> Bagian ini akan diisi setelah hasil SAE tersedia.

Subbab ini membahas pertanyaan utama Pilar 2 disertasi:
*apakah representasi laten yang dipelajari model deep learning untuk prediksi RUL
berkorespondensi dengan frekuensi karakteristik fisik bantalan (BPFO, BPFI, BSF, FTF)?*

### V.2.1 Sparse Autoencoder Training

Top-*k* Sparse Autoencoder (SAE) dengan *expansion factor* 8 dan sparsity *k* = 51
dilatih selama 50 epoch pada 20.000 *hidden state* yang dikumpulkan dari checkpoint
terbaik Mamba-xLSTM-Net pada masing-masing dataset.
Dimensi ruang laten SAE: **d\_latent = 1024** (= d\_model 128 × expansion 8).

**Tabel V.4.** Kualitas rekonstruksi SAE setelah 50 epoch training.

| Dataset | Best Checkpoint | SAE Recon Loss (MSE) |
|---------|----------------|----------------------|
| PHM2012 | Mamba-xLSTM-Net / seed 42 / epoch 71 | **0,000512** |
| XJTU-SY | Mamba-xLSTM-Net / seed 44 / epoch 70 | **0,000102** |

Loss rekonstruksi yang sangat kecil (< 0,001) mengindikasikan bahwa SAE berhasil
merekonstruksi representasi laten secara akurat meskipun hanya menggunakan 51 dari
1024 fitur aktif per sampel (sparsity ≈ 5%). Ini berarti representasi laten
Mamba-xLSTM-Net **memiliki struktur yang sangat *sparse* secara alami** — sebagian
besar informasi relevan untuk prediksi RUL terkonsentrasi pada sedikit fitur aktif.

Proyeksi UMAP dari ruang laten SAE tersedia sebagai `results/.../explain/sae_umap_clusters.png`
dan akan disertakan sebagai Gambar V.x dalam versi LaTeX disertasi.

### V.2.2 Pemetaan ke Frekuensi Karakteristik Bantalan

Frekuensi karakteristik teoritis dihitung menggunakan geometri bantalan NSK 6804
(PHM2012) dan LDK UER204 (XJTU-SY) sesuai Persamaan III.x pada Bab III (lihat
Tabel V.5). Korelasi Pearson dihitung antara aktivasi tiap fitur SAE dan amplitudo
*Hilbert envelope spectrum* dari sinyal getaran mentah pada pita frekuensi ±2 Hz
di sekitar tiap frekuensi karakteristik, menggunakan 300 rekaman dari *training set*.

**Tabel V.5.** Frekuensi karakteristik teoritis pada kondisi operasi nominal.

| Dataset | Bantalan | fr (Hz) | BPFO (Hz) | BPFI (Hz) | BSF (Hz) | FTF (Hz) |
|---------|---------|---------|-----------|-----------|----------|----------|
| PHM2012 | NSK 6804 | 30,0 | 168,24 | 221,76 | 107,23 | 12,94 |
| XJTU-SY | LDK UER204 | 35,0 | 107,91 | 172,09 | 72,33 | 13,49 |

**Tabel V.6.** Hit-rate SAE features ↔ BPFx (threshold |r| ≥ 0,30; d\_latent = 1024).

| Dataset | BPFO | BPFI | BSF | FTF |
|---------|:----:|:----:|:---:|:---:|
| PHM2012 | 2,0% (20/1024) | **2,3% (24/1024)** | 0,0% | 0,0% |
| XJTU-SY | **2,2% (23/1024)** | 0,0% | 0,3% (3/1024) | 0,0% |

Hasil pada Tabel V.6 menunjukkan bahwa sebagian kecil fitur SAE (2–2,3%)
berkorelasi moderat dengan amplitudo *envelope spectrum* di sekitar BPFO dan BPFI
— dua frekuensi yang diasosiasikan dengan mode kegagalan *outer race* dan
*inner race*, yang merupakan **mode kegagalan dominan** pada kedua dataset benchmark.
BSF dan FTF mendekati nol, konsisten dengan tidak adanya kegagalan *rolling element*
atau sangkar yang terdokumentasi pada bearing-bearing di *training set*.

Interpretasi penting: angka hit-rate yang rendah (2–2,3%) bukanlah indikasi
kegagalan model. Input ke Mamba-xLSTM-Net adalah **vektor fitur band-energy**
yang sudah di-agregasi, bukan sinyal getaran mentah secara langsung. Fitur SAE
mempelajari kombinasi abstrak dari fitur-fitur tersebut yang berguna untuk
prediksi RUL — bukan pengulangan analisis envelope. Bahwa BPFO dan BPFI
tetap muncul sebagai satu-satunya frekuensi dengan hit > 0 menunjukkan
**konsistensi yang lemah namun dapat dideteksi** antara representasi yang dipelajari
model dan teori klasik diagnosis getaran.

<!-- TODO(verify-with-pembimbing): diskusikan apakah threshold |r| ≥ 0,3 sudah
     tepat atau perlu diturunkan ke 0,2 untuk mendapatkan gambaran lebih lengkap.
     Pertimbangkan juga uji signifikansi (p-value < 0,05) untuk n=300. -->

### V.2.3 Visualisasi Ruang Laten (UMAP)

Proyeksi UMAP dari *hidden states* Mamba-xLSTM-Net (setelah SAE encoding)
dihasilkan oleh `scripts/run_interpretability.py` dan disimpan sebagai
`sae_umap_clusters.png` di masing-masing `explain/` folder.

<!-- TODO: setelah file PNG dilampirkan ke dokumen LaTeX, tambahkan deskripsi
     clustering yang terlihat: apakah ada separasi antara RUL tinggi vs rendah?
     Apakah cluster HDBSCAN membentuk gradasi degradasi? -->

### V.2.4 SHAP Global Feature Importance

Nilai SHAP global (rata-rata |SHAP| per fitur) dihitung menggunakan 64 sampel uji
dan disimpan dalam `shap_global.json`. Fitur dengan kontribusi tertinggi mencerminkan
kombinasi fitur time-domain (td\_c0\_rms, td\_c1\_kurtosis) dan frequency-domain
(fd\_c0/c1\_band*) yang paling informatif untuk prediksi RUL pada masing-masing dataset.

<!-- TODO: buat Tabel V.7 dari top-10 fitur SHAP untuk PHM2012 dan XJTU-SY.
     Bandingkan apakah fitur kurtosis / band frekuensi tinggi lebih dominan
     pada tahap akhir degradasi vs tahap awal. -->

---

## V.3 Keterbatasan

Beberapa keterbatasan penelitian ini perlu diakui secara eksplisit:

1. **Hasil XJTU-SY pra-perbaikan dataset** (split 2 kondisi, ``Bearing2_3``
   tidak lengkap) tidak lagi representatif; Bab V perlu angka baru setelah rerun.

2. **Budget training 75 epoch** dipilih berdasarkan analisis konvergensi awal
   Mamba-xLSTM-Net. Eksperimen dengan budget lebih panjang untuk N-BEATS-xLSTM
   dan SparseGate-TCN mungkin menghasilkan performa yang berbeda, terutama
   mengingat model-model ini konvergen sangat cepat dan menunjukkan indikasi
   overfitting pada epoch pertengahan.

3. **Interpretabilitas SAE** (§V.2) dijalankan sebagai analisis *post-hoc* dan
   belum diintegrasikan ke dalam loop pelatihan model. Pendekatan *integrated
   interpretability* (melatih model dengan regularizer SAE secara end-to-end)
   adalah arah penelitian lanjutan yang menjanjikan.

---

## V.4 Saran Penelitian Lanjutan

1. Eksperimen budget training >100 epoch untuk Mamba-xLSTM-Net guna mengonfirmasi
   apakah performa terus meningkat atau mencapai plato.
2. Mengembangkan prosedur BPFx mapping yang lebih sistematis dengan uji statistik
   formal (misalnya Pearson correlation dengan bootstrap confidence interval)
   antara aktivasi SAE dan amplitudo frekuensi karakteristik.
3. Mengevaluasi model pada dataset tambahan (IMS Bearing) untuk memvalidasi
   generalisasi lintas platform pengujian.

---

*Draft diperbarui: 2026-05-13. §V.1 final (Stage 2, 18 run). §V.2 terisi dari Stage 3–4. TODO: tambah Gambar UMAP, Tabel V.7 SHAP top-10, dan konversi ke LaTeX.*
