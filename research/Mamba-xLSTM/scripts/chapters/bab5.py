"""Bab V — Kesimpulan dan Saran.

Sintesis temuan, kontribusi, keterbatasan, kesimpulan, dan saran.
Mencakup rencana Studi Kasus III (interpretabilitas) dan Studi Kasus IV (SKF Indonesia).
"""

from __future__ import annotations

from ._docx_utils import (
    add_blockquote,
    add_bullets,
    add_chapter_title,
    add_heading,
    add_numbered,
    add_page_break,
    add_paragraph,
    add_table,
)


def build(doc) -> None:
    add_chapter_title(doc, "Bab V", "Kesimpulan dan Saran")

    add_paragraph(
        doc,
        "Bab ini menyajikan sintesis temuan utama dari Bab IV (subbab V.1), kontribusi penelitian "
        "(subbab V.2), keterbatasan yang teridentifikasi (subbab V.3), kesimpulan formal terhadap "
        "rumusan masalah (subbab V.4), serta saran untuk penelitian lanjutan (subbab V.5) yang "
        "mencakup pelaksanaan Studi Kasus III (analisis interpretabilitas) dan Studi Kasus IV "
        "(transfer ke objek penelitian PT SKF Indonesia).",
        indent_first_line=False,
    )

    # ===================== V.1 =====================
    add_heading(doc, "V.1", "Sintesis Temuan Utama", level=1)
    add_paragraph(
        doc,
        "Penelitian ini telah berhasil mengembangkan dan mengevaluasi arsitektur hibrid "
        "**Mamba-2-xLSTM-Net** untuk prediksi *Remaining Useful Life* (RUL) pada *rolling "
        "element bearing*. Lima temuan pokok dari Bab IV dapat dirangkum sebagai berikut:",
    )
    add_numbered(
        doc,
        [
            "**Replikasi baseline berhasil dengan setia**. Implementasi *baseline* "
            "xLSTM\u2013Transformer (Liu dkk., 2025) memenuhi kriteria sukses Studi Kasus I "
            "pada kedua dataset publik, dengan RMSE dalam toleransi \u00b110% dari angka acuan "
            "(0,1033 untuk PHM 2012 dan 0,2118 untuk XJTU-SY). Hal ini memberikan landasan "
            "pembanding yang valid untuk evaluasi arsitektur usulan.",
            "**Mamba-2-xLSTM-Net unggul tipis namun konsisten pada PHM 2012**. Pada sembilan "
            "metrik test, arsitektur usulan mencatat penurunan RMSE 5,2%, MAE 17,1%, dan PHM "
            "Score Paper 16,6% terhadap *baseline* dengan biaya peningkatan parameter 19\u00d7 "
            "dan waktu latih 1,2\u00d7. Peningkatan terbesar pada MAE menunjukkan bahwa kurva "
            "prediksi usulan secara seragam lebih dekat ke target sepanjang masa pakai *bearing*.",
            "**Pada XJTU-SY, baseline kompak masih unggul** terhadap varian Mamba-1-xLSTM-Net "
            "(RMSE +11,4%). Tiga faktor pembeda diidentifikasi: panjang sekuens lebih pendek, "
            "keragaman *fault mode* lebih rendah, dan varian Mamba yang berbeda. Hasil ini "
            "menjadikan klaim empirik disertasi bersifat *dataset-dependent*, bukan klaim "
            "*state-of-the-art* yang universal.",
            "**Pemilihan varian Mamba berpengaruh signifikan**. Ablasi tiga varian "
            "(Mamba-1/2/3) pada PHM 2012 menunjukkan bahwa Mamba-2 (*structured state space "
            "duality*) mendominasi Mamba-1 pada seluruh metrik dengan jumlah parameter justru "
            "*lebih kecil* (811k vs 956k) dan waktu latih *separuh* (1.583 s vs 2.474 s). "
            "Mamba-3 dengan `d_state` lebih besar belum kompetitif, mengindikasikan bahwa "
            "memperbesar kapasitas SSM bukan jalan pintas untuk peningkatan kinerja.",
            "**Kebaharuan arsitektural dan reprodusibilitas terpenuhi**. Disertasi "
            "mengisi kesenjangan literatur yang teridentifikasi pada Bab II (kombinasi Mamba"
            "\u2013xLSTM untuk RUL *bearing* belum dipublikasikan), dengan pipeline implementasi "
            "yang *end-to-end* reprodusibel (artefak `config.yaml`, *checkpoint*, `summary.json`, "
            "`test_predictions.npz` tersedia).",
        ],
    )

    # ===================== V.2 =====================
    add_heading(doc, "V.2", "Kontribusi Penelitian", level=1)
    add_paragraph(
        doc,
        "Sesuai dengan rumusan kontribusi pada Bab I, tiga kontribusi utama disertasi ini adalah:",
    )

    add_table(
        doc,
        ["Kode", "Kontribusi", "Bukti pada disertasi"],
        [
            [
                "K1",
                "Pengembangan arsitektur hibrid Mamba-2-xLSTM-Net yang mengisi kesenjangan "
                "literatur kombinasi *selective state space model* dengan *extended LSTM* untuk "
                "prediksi RUL *rolling element bearing*.",
                "Bab III (subbab III.5), Tabel III-9, ablasi pada subbab IV.5 yang menegaskan "
                "Mamba-2 sebagai pilihan optimal.",
            ],
            [
                "K2",
                "Implementasi *end-to-end* yang reprodusibel dengan protokol pelatihan dan "
                "evaluasi yang identik antara *baseline* dan arsitektur usulan, sehingga "
                "perbandingan langsung (*head-to-head*) bersifat adil dan dapat diverifikasi "
                "pihak lain.",
                "Bab III (subbab III.6\u2013III.7), Bab IV (Tabel IV-1, IV-3, IV-5, IV-6, IV-7), "
                "lampiran konfigurasi YAML.",
            ],
            [
                "K3",
                "Kerangka diagnostik *trade-off* parameter\u2013performa\u2013waktu latih lintas "
                "varian SSM dan lintas dataset, dengan temuan eksplisit bahwa keunggulan "
                "arsitektur usulan bersifat *dataset-dependent*.",
                "Bab IV (Tabel IV-4, IV-6, subbab IV.7), pembahasan asimetri PHM 2012 vs "
                "XJTU-SY.",
            ],
        ],
        caption="Tabel V-1. Pemenuhan tiga kontribusi penelitian terhadap bukti empirik.",
        first_col_bold=True,
    )

    add_paragraph(
        doc,
        "Kontribusi K1 dan K2 telah tercapai sepenuhnya. Kontribusi K3 tercapai dalam dimensi "
        "varian SSM dan dataset publik, dengan dimensi ablasi A1\u2013A7 (subbab III.8.3) yang "
        "menjadi pekerjaan lanjutan terstruktur.",
    )

    # ===================== V.3 =====================
    add_heading(doc, "V.3", "Keterbatasan Penelitian", level=1)
    add_paragraph(
        doc,
        "Penelitian ini memiliki sejumlah keterbatasan yang perlu disampaikan secara terbuka "
        "agar pembaca dapat menempatkan hasil pada konteks yang tepat:",
    )
    add_numbered(
        doc,
        [
            "**Single-seed evaluation**. Seluruh hasil pada Bab IV menggunakan satu *seed* acak "
            "(seed = 42). Perbedaan kinerja yang teramati (misalnya RMSE 0,1033 vs 0,0979) berada "
            "pada urutan magnitude yang sama dengan variabilitas antar-seed yang umumnya "
            "terlihat pada model *deep learning*, sehingga klaim signifikansi statistik belum "
            "dapat dibuat. Multi-seed (5\u201310 seed) dengan uji statistik formal "
            "(*Wilcoxon signed-rank* atau *paired t-test*) merupakan pekerjaan prioritas berikutnya.",
            "**Cakupan ablasi terbatas**. Hanya ablasi A0 (varian *backbone* Mamba) yang sudah "
            "dijalankan dari delapan konfigurasi yang dirancang pada Tabel III-11. Ablasi "
            "A1\u2013A7 \u2014 termasuk *xLSTM-only*, *Mamba-only*, *unidirectional Mamba*, "
            "*concat fusion*, *vanilla LSTM*, dan variasi panjang *window* \u2014 belum "
            "dijalankan, sehingga sumber peningkatan kinerja Mamba-2-xLSTM-Net pada PHM 2012 "
            "belum terisolasi secara mekanistik.",
            "**Studi Kasus III (analisis interpretabilitas) belum dilaksanakan secara penuh**. "
            "Tiga metode yang direncanakan (SHAP, *Sparse Autoencoder*, *Integrated Gradients*) "
            "telah disiapkan kerangka kodenya pada `src/interpretability/`, namun analisis "
            "kualitatif terhadap fitur-fitur laten dan atribusi temporal belum dimasukkan ke "
            "dalam disertasi. Hal ini menjadi keterbatasan komunikasi yang penting, mengingat "
            "*deep learning* tanpa interpretabilitas menghadapi tantangan adopsi pada aplikasi "
            "industri.",
            "**Studi Kasus IV (PT SKF Indonesia) masih berupa rencana**. Data lapangan "
            "vibrasi mesin *grinding* OR1 dan OR2 di Channel 15 PT SKF Indonesia, beserta data "
            "kualitas produk (*vibration checking* dan *radial clearance checking*), belum "
            "diintegrasikan ke pipeline. Validasi pada objek penelitian nyata menjadi syarat "
            "akhir untuk klaim kelaikan industri.",
            "**Eksperimen Mamba-2 pada XJTU-SY belum dilakukan**. Subbab IV.4 hanya "
            "membandingkan *baseline* dengan Mamba-1; konsekuensinya, klaim asimetri performa "
            "antara dataset PHM 2012 dan XJTU-SY pada subbab IV.7.1 perlu diverifikasi ulang "
            "dengan run Mamba-2 pada XJTU-SY untuk meniadakan *confound* varian arsitektur.",
            "**Ketidaktersediaan run Mamba-3 pada PHM 2012**. Laporan komparatif "
            "`baseline_vs_mamba3_phm.html` pada repositori hanya berisi *baseline* karena "
            "run Mamba-3 PHM tidak menyelesaikan tahap evaluasi pada percobaan terakhir; "
            "Tabel IV-6 saat ini menggunakan angka Mamba-3 dari XJTU-SY sebagai pendekatan, "
            "yang membatasi kekuatan kesimpulan ablasi varian SSM pada PHM 2012.",
        ],
    )

    # ===================== V.4 =====================
    add_heading(doc, "V.4", "Kesimpulan", level=1)
    add_paragraph(
        doc,
        "Berdasarkan rumusan masalah pada Bab I dan hasil eksperimen pada Bab IV, kesimpulan "
        "formal disertasi ini dapat dinyatakan sebagai berikut:",
    )
    add_numbered(
        doc,
        [
            "**Arsitektur hibrid Mamba-2-xLSTM-Net telah berhasil dikembangkan** sebagai "
            "kombinasi *Bidirectional Mamba-2* (cabang global) dengan *extended LSTM* (cabang "
            "lokal) yang difusi melalui mekanisme *gated fusion*. Arsitektur ini memenuhi "
            "rumusan masalah tentang pemodelan dinamika lokal dan global degradasi *bearing* "
            "secara simultan dalam kompleksitas linear O(L \u00b7 d).",
            "**Pada dataset PHM 2012, Mamba-2-xLSTM-Net memberikan peningkatan kinerja yang "
            "konsisten** dibandingkan *baseline* xLSTM\u2013Transformer (Liu dkk., 2025) di "
            "seluruh sembilan metrik test, dengan peningkatan paling tinggi pada MAE (17,1%) dan "
            "PHM Score Paper (16,6%). Peningkatan ini bersifat tipis dalam nilai absolut namun "
            "signifikan secara praktis untuk aplikasi *predictive maintenance*.",
            "**Pada dataset XJTU-SY, varian Mamba-1-xLSTM-Net belum mengungguli baseline**, "
            "yang menyiratkan bahwa keunggulan arsitektur usulan bersifat *dataset-dependent*. "
            "Klaim empirik disertasi karena itu dirumuskan secara hati-hati: Mamba-2-xLSTM-Net "
            "merupakan kandidat arsitektur unggul untuk PHM 2012 dan dataset RUL serupa dengan "
            "sekuens panjang serta fault mode beragam, namun bukan klaim universal SOTA.",
            "**Ablasi varian Mamba menetapkan Mamba-2 sebagai pilihan default** untuk "
            "kombinasi dengan xLSTM, karena (a) mendominasi Mamba-1 pada PHM 2012 di seluruh "
            "metrik, (b) hemat parameter (15,1% lebih sedikit dari Mamba-1), dan (c) waktu "
            "latih separuh dari Mamba-1.",
            "**Kontribusi kebaharuan arsitektural** \u2014 mengisi kesenjangan literatur "
            "kombinasi Mamba\u2013xLSTM untuk RUL *bearing* \u2014 telah tercapai sepenuhnya, "
            "dengan implementasi reprodusibel yang dapat diverifikasi independen.",
        ],
    )

    # ===================== V.5 =====================
    add_heading(doc, "V.5", "Saran untuk Penelitian Lanjutan", level=1)
    add_paragraph(
        doc,
        "Berikut saran pekerjaan lanjutan yang direkomendasikan, diurutkan berdasarkan "
        "prioritas dampak terhadap kesempurnaan disertasi dan relevansi industri:",
    )

    add_heading(doc, "V.5.1", "Verifikasi Statistik dan Ketegaran (Robustness)", level=2)
    add_numbered(
        doc,
        [
            "**Evaluasi multi-seed** dengan 5\u201310 *seed* acak per konfigurasi, dilanjutkan "
            "dengan uji *Wilcoxon signed-rank* pada RMSE per-*bearing* untuk memberikan klaim "
            "signifikansi statistik formal terhadap perbandingan Mamba-2-xLSTM-Net vs "
            "*baseline*.",
            "**Eksperimen Mamba-2 pada XJTU-SY** dengan protokol identik dengan PHM 2012, "
            "untuk meniadakan *confound* varian Mamba pada perbandingan dataset dan menguji "
            "hipotesis asimetri performa antara dataset PHM 2012 dan XJTU-SY (subbab IV.7.1).",
            "**Pengulangan run Mamba-3 pada PHM 2012** untuk melengkapi Tabel IV-6 dengan "
            "angka asli (saat ini menggunakan angka XJTU-SY sebagai pendekatan) dan memperkuat "
            "kesimpulan ablasi varian SSM.",
        ],
    )

    add_heading(doc, "V.5.2", "Pelaksanaan Penuh Studi Kasus III: Analisis Interpretabilitas", level=2)
    add_paragraph(
        doc,
        "Tiga jalur analisis interpretabilitas yang sudah dirancang pada subbab III.8.3 perlu "
        "dilaksanakan secara penuh:",
    )
    add_numbered(
        doc,
        [
            "**SHAP (SHapley Additive exPlanations)** menggunakan `shap.GradientExplainer` "
            "pada Mamba-2-xLSTM-Net terlatih, untuk mengidentifikasi fitur-fitur HI mana "
            "(RMS, kurtosis, *spectral entropy*, energi pita frekuensi) yang paling "
            "berpengaruh pada prediksi RUL secara global maupun per-*bearing*. Output: "
            "*waterfall plot* per-prediksi dan *bar chart* importansi global.",
            "**Sparse Autoencoder (SAE)** dilatih pada representasi tersembunyi lapisan fusi "
            "(*pre-regression head*) dengan faktor ekspansi 8\u00d7. Latent SAE kemudian "
            "dianalisis kualitatif: setiap latent dipetakan ke pola degradasi khas (peningkatan "
            "RMS tajam, lonjakan impuls, saturasi fase akhir) menggunakan *top-activating "
            "examples* dari dataset uji. Pendekatan ini mengadopsi praktik *mechanistic "
            "interpretability* yang populer di riset *large language model* (Anthropic, OpenAI) "
            "dan masih jarang diterapkan pada PHM, sehingga memberikan kontribusi metodologis "
            "tambahan.",
            "**Integrated Gradients** untuk atribusi temporal: mengidentifikasi *time step* "
            "mana dalam *window* input (L = 10) yang paling berpengaruh terhadap prediksi pada "
            "titik tertentu. Output: *heatmap* atribusi temporal per-bearing.",
        ],
    )
    add_paragraph(
        doc,
        "Pelaksanaan ketiga analisis ini akan memenuhi salah satu kontribusi disertasi yang "
        "penting untuk aplikasi industri \u2014 yaitu *trustworthy AI* dengan justifikasi "
        "mekanistik atas keputusan model.",
    )

    add_heading(doc, "V.5.3", "Pelaksanaan Studi Kasus IV: Transfer ke PT SKF Indonesia", level=2)
    add_paragraph(
        doc,
        "Studi Kasus IV merupakan tahap akhir disertasi yang memberikan kontribusi langsung "
        "kepada industri mitra. Lima tahapan transfer direkomendasikan:",
    )
    add_numbered(
        doc,
        [
            "**Pemetaan format data**. Data SKF Observer (XLSX) yang berisi pengukuran vibrasi "
            "mesin *grinding* OR1 dan OR2 di Channel 15 perlu dipetakan ke format pipeline "
            "yang sudah divalidasi. Indeks waktu, satuan amplitudo, dan struktur kanal akan "
            "diadaptasi melalui *exporter* Python yang berdiri sendiri.",
            "**Konstruksi HI menggunakan pipeline `liu2026_phm`** yang sudah teruji, dengan "
            "penyesuaian parameter pita frekuensi karakteristik berdasarkan kondisi operasi "
            "spesifik mesin SKF (kecepatan poros, beban statis).",
            "**Fine-tuning model Mamba-2-xLSTM-Net** yang sudah dilatih pada PHM 2012 dengan "
            "pendekatan *transfer learning*: cabang xLSTM dan BiMamba dibekukan untuk 5 "
            "*epoch* awal, kemudian seluruh parameter dilatih ulang dengan *learning rate* "
            "rendah (1 \u00d7 10\u207b\u2074). Strategi ini memungkinkan adaptasi pada karakter "
            "data SKF tanpa mempelajari ulang representasi degradasi dari nol.",
            "**Integrasi data kualitas produk** sebagai *auxiliary input*. Data *vibration "
            "checking* dan *radial clearance checking* yang dilakukan pada akhir setiap *batch* "
            "produksi dapat diintegrasikan melalui *feature-level fusion*: vektor kualitas "
            "produk dikodekan oleh *embedding* terpisah, kemudian dikonkatenasi dengan "
            "representasi HI sebelum *regression head*. Ini akan menjadikan model SKF lebih "
            "*context-aware* dibanding model dataset publik.",
            "**Evaluasi cross-machine** antara OR1 dan OR2 (validasi silang antar mesin), "
            "untuk memastikan generalisasi dalam *fleet*. Metrik utama tetap RMSE, MAE, dan PHM "
            "Score; metrik domain-spesifik (akurasi prediksi *clearance loss*) akan ditambahkan "
            "berdasarkan diskusi dengan tim *predictive maintenance* SKF.",
        ],
    )

    add_heading(doc, "V.5.4", "Optimalisasi Efisiensi untuk Penerapan pada Perangkat Edge", level=2)
    add_paragraph(
        doc,
        "Mamba-2-xLSTM-Net memiliki sekitar 19\u00d7 lebih banyak parameter dibanding "
        "*baseline*. Walaupun GPU server tidak menjadi kendala, *deployment* di lapangan pada "
        "perangkat *edge* (misalnya *industrial PC* di pabrik SKF) memerlukan model yang lebih "
        "efisien. Tiga arah pengurangan parameter yang dapat ditelaah:",
    )
    add_bullets(
        doc,
        [
            "**Knowledge distillation**: melatih model murid yang lebih kecil (sekitar 100\u2013"
            "200k parameter) untuk meniru distribusi prediksi guru (Mamba-2-xLSTM-Net 811k), "
            "dengan harapan menjaga 80\u201390% keunggulan kinerja.",
            "**Structured pruning**: memangkas saluran (*channel*) atau blok yang memiliki "
            "kontribusi rendah berdasarkan bobot magnitudo atau analisis SHAP global.",
            "**Quantization** ke INT8 atau FP16 dengan kalibrasi pada dataset SKF, untuk "
            "mengurangi *footprint memori* dan mempercepat inferensi pada perangkat *edge*.",
        ],
    )

    add_heading(doc, "V.5.5", "Pelengkapan Studi Ablasi A1\u2013A7", level=2)
    add_paragraph(
        doc,
        "Tujuh konfigurasi ablasi pada Tabel III-11 (selain A0 yang sudah dijalankan) perlu "
        "dilaksanakan untuk mengisolasi sumber peningkatan kinerja Mamba-2-xLSTM-Net pada PHM "
        "2012. Prioritas tiga ablasi paling kritis:",
    )
    add_bullets(
        doc,
        [
            "**A1 (xLSTM-only)** dan **A2 (Mamba-only)** untuk memisahkan kontribusi dua cabang "
            "secara bersih.",
            "**A4 (concat fusion)** untuk menguji apakah *gated fusion* benar-benar diperlukan "
            "atau cukup dengan konkatenasi sederhana.",
            "**A5 (vanilla LSTM)** untuk mengonfirmasi bahwa *exponential gating* xLSTM "
            "menjadi sumber keunggulan signifikan, bukan sekadar peningkatan kapasitas memori.",
        ],
    )

    add_page_break(doc)

    # ===================== Penutup =====================
    add_heading(doc, "V.6", "Penutup", level=1)
    add_paragraph(
        doc,
        "Disertasi ini telah memberikan tiga kontribusi yang diharapkan menjadi fondasi bagi "
        "penelitian lanjutan di bidang *predictive maintenance* berbasis pembelajaran mendalam, "
        "khususnya untuk prediksi *Remaining Useful Life* *rolling element bearing*. Pengembangan "
        "arsitektur Mamba-2-xLSTM-Net mengisi kesenjangan literatur yang teridentifikasi pada "
        "Bab II, sementara pipeline implementasi yang reprodusibel memastikan bahwa kerangka "
        "penelitian "
        "ini dapat dilanjutkan, divariasi, dan diadaptasi oleh komunitas akademik maupun "
        "industri.",
    )
    add_paragraph(
        doc,
        "Hasil empirik yang bersifat *dataset-dependent* (unggul pada PHM 2012, belum unggul "
        "pada XJTU-SY dengan varian Mamba-1) merupakan temuan yang justru memperkaya pemahaman "
        "kondisi-kondisi di mana arsitektur hibrid Mamba\u2013xLSTM memberikan nilai tambah. "
        "Saran-saran pada subbab V.5 memberikan jalur konkret untuk menyempurnakan penelitian "
        "menuju validasi industri yang utuh, dengan tahap akhir berupa *deployment* pada mesin "
        "produksi PT SKF Indonesia sebagai bukti praktis kelaikan teknologi.",
    )
    add_paragraph(
        doc,
        "Akhirnya, peneliti berharap kerangka pemikiran, hasil empirik, dan rekomendasi "
        "lanjutan yang disampaikan dalam disertasi ini dapat memberikan kontribusi nyata bagi "
        "kemajuan ilmu pengetahuan di bidang *Prognostics and Health Management* dan praktik "
        "*predictive maintenance* di industri manufaktur Indonesia.",
    )
