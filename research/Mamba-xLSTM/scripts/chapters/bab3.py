"""Bab III \u2014 Pengembangan Model.

Versi yang dilengkapi diagram arsitektur (xLSTM, Mamba, model usulan) dan
rumus matematika ber-LaTeX yang di-render via matplotlib mathtext.

Catatan:
- Seluruh diagram dibangun oleh ``chapters._diagrams`` dan tersimpan di
  ``Mamba-xLSTM/results/_chapter_assets/diagrams/``.
- Seluruh rumus matematika dibangun oleh ``chapters._equations`` (LaTeX
  mathtext) dan tersimpan di ``Mamba-xLSTM/results/_chapter_assets/equations/``.
"""

from __future__ import annotations

from chapters._diagrams import ASSET_ROOT as DIAGRAM_ROOT
from chapters._docx_utils import (
    add_blockquote,
    add_bullets,
    add_chapter_title,
    add_diagram,
    add_equation_image,
    add_heading,
    add_numbered,
    add_paragraph,
    add_table,
)


def _diag(name: str) -> str:
    """Return diagram filename path under the assets root."""
    return str(DIAGRAM_ROOT / name)


def build(doc) -> None:
    add_chapter_title(doc, "Bab III", "Pengembangan Model")

    add_paragraph(
        doc,
        "Bab ini menguraikan pengembangan model yang dilakukan dalam penelitian ini. "
        "Subbab III.1 menyajikan sintesis *state of the art* yang menjadi landasan pemilihan model. "
        "Subbab III.2 menjelaskan proses penyempitan fokus penelitian dari enam algoritma kandidat awal "
        "menjadi dua arsitektur *state of the art* setelah dilakukan kajian literatur lanjutan. "
        "Subbab III.3 memformulasikan masalah prediksi *Remaining Useful Life* (RUL) secara matematis. "
        "Subbab III.4 dan III.5 masing-masing menguraikan arsitektur model acuan (*baseline*) "
        "xLSTM\u2013Transformer dan model usulan Mamba-xLSTM-Net secara rinci, dengan menegaskan bahwa "
        "varian final yang menjadi fokus evaluasi pada Bab IV adalah **Mamba-2-xLSTM-Net**. "
        "Subbab III.6 menjelaskan protokol pelatihan, subbab III.7 metrik evaluasi, dan "
        "subbab III.8 rancangan eksperimen beserta skenario validasi silang. "
        "Subbab III.9 menutup bab ini dengan diagram alur keseluruhan penelitian.",
        indent_first_line=False,
    )

    # ===================== III.1 =====================
    add_heading(doc, "III.1", "Sintesis Model State of the Art", level=1)
    add_paragraph(
        doc,
        "Peneliti melakukan pemaparan *state of the art* (SotA) penelitian di bidang "
        "*predictive maintenance* (PdM) yang telah disajikan pada Tabel III-1 hingga Tabel III-5. "
        "Sintesis model acuan yang dipilih berdasarkan fokus penelitian ini ditampilkan pada Tabel III-6. "
        "Berdasarkan hasil pemaparan tersebut, awalnya diidentifikasi enam pendekatan utama sebagai "
        "kandidat algoritma: (1) XGBoost sebagai *baseline* *traditional machine learning*, "
        "(2) CNN-LSTM (Khorram dkk., 2021), (3) *Temporal Convolutional Network* (TCN) dengan "
        "*graph fusion* (Li dkk., 2020), (4) BiLSTM dengan *Hierarchical Improved Fusion Attention* "
        "(Huang dkk., 2025), (5) GRU *lightweight* (Bai dkk., 2025), dan (6) hibrid ARIMA-LSTM "
        "(Hamiane dkk., 2024). Keenam kandidat tersebut merepresentasikan spektrum pendekatan yang "
        "cukup lengkap, mulai dari *traditional machine learning* berbasis rekayasa fitur hingga "
        "*deep learning* *end-to-end* dengan berbagai mekanisme perhatian.",
    )
    add_paragraph(
        doc,
        "Selama pelaksanaan penelitian, dilakukan kajian literatur lanjutan yang mengungkap "
        "perkembangan signifikan di bidang pemodelan sekuens pada rentang tahun 2023\u20132025. "
        "Perkembangan tersebut bermuara pada munculnya dua arsitektur baru yang menunjukkan kinerja "
        "unggul pada berbagai tugas pemodelan deret waktu, yaitu *extended Long Short-Term Memory* "
        "(xLSTM) yang diperkenalkan oleh Beck dkk. (2024) dan *Mamba* (*selective state space model*) "
        "yang diperkenalkan oleh Gu & Dao (2023) beserta penerusnya Mamba-2 oleh Dao & Gu (2024). "
        "Temuan literatur ini, ditambah dengan publikasi Liu dkk. (2025) mengenai penerapan "
        "xLSTM\u2013Transformer untuk prediksi RUL *rolling element bearing* pada dataset XJTU-SY "
        "dan PHM 2012, mendorong peneliti untuk menyempitkan fokus penelitian dari enam kandidat "
        "algoritma menjadi dua arsitektur SotA yang akan dibandingkan secara mendalam. Proses "
        "penyempitan ini dijelaskan pada subbab berikutnya.",
    )

    # ===================== III.2 =====================
    add_heading(doc, "III.2", "Penyempitan Fokus Penelitian: Evolusi Menuju Arsitektur State of the Art", level=1)
    add_heading(doc, "III.2.1", "Rasionalisasi Penyempitan Fokus", level=2)
    add_paragraph(
        doc,
        "Pengujian komparatif enam algoritma sebagaimana direncanakan pada Tabel III-7 memiliki "
        "keunggulan dalam cakupan spektrum pendekatan, namun mengandung dua kelemahan metodologis "
        "yang mendasar. *Pertama*, keenam algoritma tersebut diusulkan pada rentang tahun 2018"
        "\u20132025 dengan acuan pembanding yang berbeda-beda, sehingga perbandingan langsung di "
        "antara keenamnya tidak memiliki titik referensi tunggal yang kuat. *Kedua*, kedalaman "
        "analisis pada masing-masing algoritma menjadi terbatas bila dibagi untuk enam varian, "
        "sehingga kontribusi ilmiah yang dihasilkan cenderung bersifat tolok ukur empiris "
        "(*empirical benchmarking*) tanpa kebaharuan arsitektural yang substantif. Kedua "
        "kelemahan ini berpotensi melemahkan posisi penelitian sebagai disertasi doktoral yang "
        "menuntut kebaharuan dan orisinalitas (*novelty and originality*) sebagaimana disyaratkan "
        "dalam Pedoman Penulisan Disertasi Doktor ITB.",
    )
    add_paragraph(
        doc,
        "Oleh karena itu, peneliti melakukan penyempitan fokus berdasarkan dua pertimbangan utama. "
        "Pertama, pemilihan arsitektur acuan (*baseline*) yang paling mutakhir dan terbukti unggul "
        "untuk prediksi RUL *bearing* berdasarkan publikasi terbaru. Kedua, pengajuan satu arsitektur "
        "usulan (*proposed*) yang mengisi kesenjangan penelitian (*research gap*) yang belum "
        "dikerjakan dalam literatur. Hasil penyempitan ini adalah perbandingan langsung "
        "(*head-to-head*) antara xLSTM\u2013Transformer sebagai *baseline* dan Mamba-2-xLSTM-Net "
        "sebagai arsitektur usulan final, dengan tiga varian Mamba (Mamba-1, Mamba-2, Mamba-3) "
        "ditelaah secara ablasi untuk mengisolasi pengaruh blok *selective state space*.",
    )

    add_heading(doc, "III.2.2", "Evolusi Pemodelan Sekuens 2014\u20132025", level=2)
    add_paragraph(
        doc,
        "Pemodelan sekuens dalam pembelajaran mendalam telah mengalami evolusi pesat dalam satu "
        "dasawarsa terakhir. Evolusi tersebut dapat dikelompokkan ke dalam empat generasi utama, "
        "sebagaimana diringkas pada Tabel III-8 dan diilustrasikan pada Gambar III.1.",
    )

    add_diagram(
        doc,
        _diag("fig_evolution_timeline.png"),
        "Gambar III.1 Garis waktu evolusi pemodelan sekuens dalam pembelajaran mendalam (2014\u20132025): "
        "dari LSTM/GRU ke Transformer, kemudian ke arsitektur linear-time (Mamba, xLSTM, Mamba-2) yang "
        "menjadi landasan disertasi ini.",
        width_inches=6.5,
    )

    add_table(
        doc,
        ["Generasi", "Periode", "Arsitektur utama", "Kompleksitas", "Keterbatasan utama"],
        [
            ["Pertama", "1997\u20132014", "LSTM, GRU", "O(L \u00b7 d\u00b2)",
             "Vanishing gradient pada sekuens panjang; memori tersembunyi skalar berkapasitas rendah"],
            ["Kedua", "2017\u20132023", "Transformer, Multi-Head Attention", "O(L\u00b2 \u00b7 d)",
             "Kompleksitas kuadratik terhadap panjang sekuens L; konteks panjang mahal secara komputasi"],
            ["Ketiga", "2023\u20132024", "xLSTM, Mamba (selective SSM)", "O(L \u00b7 d) atau O(L \u00b7 d\u00b2) linear",
             "Varian baru; interpretabilitas mekanistik masih terbatas"],
            ["Keempat", "2024\u20132025", "Mamba-2, Mamba-3, xLSTM\u2013Transformer hibrid", "O(L \u00b7 d)",
             "Integrasi dengan dataset PHM industri masih sangat terbatas"],
        ],
        caption="Tabel III-8. Evolusi pemodelan sekuens dalam pembelajaran mendalam.",
        first_col_bold=True,
    )
    add_paragraph(
        doc,
        "Keterangan: L adalah panjang sekuens input, d adalah dimensi model.",
        indent_first_line=False,
    )

    add_paragraph(
        doc,
        "LSTM generasi pertama (Hochreiter & Schmidhuber, 1997) dan varian GRU (Cho dkk., 2014) "
        "memiliki keterbatasan kapasitas memori tersembunyi yang berupa vektor skalar dan rentan "
        "terhadap *vanishing gradient*. Transformer generasi kedua (Vaswani dkk., 2017) mengatasi "
        "keterbatasan tersebut melalui mekanisme *self-attention*, namun dengan konsekuensi "
        "kompleksitas kuadratik O(L\u00b2) yang menjadi kendala ketika panjang sekuens mencapai "
        "ribuan *time step* sebagaimana umumnya terjadi pada sinyal vibrasi *bearing*. Gambar III.2 "
        "memvisualisasikan dampak praktis perbedaan kompleksitas tersebut: pada panjang sekuens "
        "khas PHM 2012 (\u22482800 akuisisi), biaya *self-attention* lebih besar tiga sampai empat "
        "orde magnitudo dibandingkan *selective scan* Mamba.",
    )

    add_diagram(
        doc,
        _diag("fig_complexity_chart.png"),
        "Gambar III.2 Skala biaya komputasi sebagai fungsi panjang sekuens L: kuadratik untuk "
        "*self-attention* Transformer vs linear untuk *selective state space model* Mamba "
        "(d = 128, skala log).",
        width_inches=5.5,
    )

    add_paragraph(
        doc,
        "Dua terobosan generasi ketiga masing-masing mengatasi kelemahan yang berbeda. xLSTM "
        "(Beck dkk., 2024) memperluas LSTM klasik dengan dua modifikasi penting: (1) "
        "*exponential gating* untuk mengatasi masalah saturasi akhir masa pakai komponen, dan "
        "(2) *matrix memory cell* (mLSTM) yang meningkatkan kapasitas memori dari skalar menjadi "
        "matriks, sehingga mampu menampung representasi fitur yang jauh lebih kaya. Mamba "
        "(Gu & Dao, 2023) mengambil jalur berbeda melalui *selective state space model* (SSM) yang "
        "mencapai kompleksitas linear O(L \u00b7 d) dengan tetap mempertahankan kemampuan menangkap "
        "*long-range dependencies*. Mamba-2 (Dao & Gu, 2024) menunjukkan bahwa Transformer dan SSM "
        "sesungguhnya merupakan dua sisi dari kerangka matematis yang sama (*structured state space "
        "duality*), sehingga kedua pendekatan tersebut saling melengkapi, bukan saling meniadakan.",
    )
    add_paragraph(
        doc,
        "Pada generasi keempat, kombinasi antar arsitektur mulai dieksplorasi. Liu dkk. (2025) "
        "mengusulkan xLSTM\u2013Transformer untuk prediksi RUL *bearing* dengan hasil yang lebih "
        "baik dibandingkan LSTM dan LSTM\u2013Transformer pada dataset XJTU-SY dan PHM 2012. "
        "Wang dkk. (2025) dan Liu F. dkk. (2025) masing-masing menunjukkan potensi Mamba pada "
        "peramalan deret waktu umum dan prediksi RUL *aero-engine*. Namun demikian, sepengetahuan "
        "peneliti pada saat penulisan, **belum terdapat publikasi yang menggabungkan Mamba dengan "
        "xLSTM khusus untuk prediksi RUL rolling element bearing**. Hal ini membentuk kesenjangan "
        "penelitian (*research gap*) yang menjadi fokus disertasi ini.",
    )

    add_heading(doc, "III.2.3", "Kesenjangan Penelitian dan Posisi Kontribusi", level=2)
    add_paragraph(
        doc,
        "Tabel III-9 merangkum hasil pemetaan literatur Mamba dan xLSTM dalam dua dimensi: "
        "arsitektur yang digunakan dan domain aplikasi.",
    )

    add_table(
        doc,
        ["No", "Peneliti (Tahun)", "Arsitektur", "Domain", "Kesenjangan dengan disertasi ini"],
        [
            ["1", "Liu dkk. (2025)", "xLSTM + Transformer Multi-Head Attention",
             "Bearing RUL (XJTU-SY, PHM 2012)",
             "Tidak menggunakan Mamba; menjadi baseline acuan disertasi ini"],
            ["2", "Wang dkk. (2025)", "Mamba (murni)", "Peramalan deret waktu umum",
             "Bukan untuk RUL bearing; tidak menggunakan xLSTM"],
            ["3", "Liu F. dkk. (2025)",
             "Enhanced Mamba + Multi-Head Attention + Learnable Scaling",
             "RUL aero-engine dan baterai", "Bukan untuk bearing; tidak menggunakan xLSTM"],
            ["4", "Mamba-SDP (2025)", "Mamba + Scaled Dot-Product Attention + FFT",
             "RUL bearing", "Tidak menggunakan xLSTM"],
            ["5", "Dao & Gu (2024)", "Mamba-2 (structured state space duality)",
             "Pemodelan bahasa", "Bukan PHM"],
            ["6", "Disertasi ini",
             "Mamba-2 + xLSTM + Gated Fusion (final); Mamba-1/Mamba-3 sebagai ablasi",
             "RUL bearing (PHM 2012, XJTU-SY)",
             "Mengisi kesenjangan kombinasi Mamba\u2013xLSTM untuk bearing RUL"],
        ],
        caption="Tabel III-9. Pemetaan literatur Mamba dan xLSTM pada domain PHM (Prognostics and Health Management).",
    )

    add_paragraph(
        doc,
        "Dengan demikian, kontribusi utama penelitian ini terletak pada pengembangan arsitektur "
        "hibrid Mamba-2-xLSTM-Net yang memadukan dua kekuatan komplementer: kemampuan "
        "*exponential gating* dan *matrix memory* dari xLSTM untuk menangkap dinamika lokal "
        "degradasi *bearing*, serta kemampuan *linear-time selective scanning* dari Mamba-2 untuk "
        "memodelkan tren global sepanjang ribuan *time step*. Arsitektur usulan ini diposisikan "
        "sebagai penyempurnaan dari *baseline* xLSTM\u2013Transformer (Liu dkk., 2025) dengan "
        "mengganti cabang Transformer berkompleksitas kuadratik menjadi cabang Mamba berkompleksitas "
        "linear; Mamba-1 (versi orisinal) dan Mamba-3 (varian dengan *d_state* lebih besar) "
        "ditelaah secara ablasi pada Bab IV untuk mengisolasi pengaruh kapasitas blok SSM.",
    )

    # ===================== III.2a (new) — Preprocessing =====================
    _BLUE = "0000FF"

    add_heading(
        doc,
        "III.2a",
        "Praproses Sinyal Getaran dan Ekstraksi Fitur",
        level=1,
        color=_BLUE,
    )

    add_heading(
        doc,
        "III.2a.1",
        "Dasar Sinyal Getaran dan Analisis Spektrum",
        level=2,
        color=_BLUE,
    )
    add_paragraph(
        doc,
        "Saat mesin beroperasi, komponen bergerak menimbulkan gaya dinamis yang menyebabkan "
        "struktur bergetar. Getaran ditransmisikan ke sensor akselerometer yang dipasang pada "
        "*bearing housing*, menghasilkan sinyal percepatan dalam satuan $g$ "
        r"(1 $g$ = 9,81 m/s$^2$).",
        color=_BLUE,
    )
    add_paragraph(
        doc,
        "Sinyal getaran dapat dianalisis dalam dua ranah. Pada *time domain*, sinyal ditampilkan "
        "sebagai amplitudo terhadap waktu; informasi yang diperoleh meliputi nilai RMS, puncak, "
        "dan pola gelombang (periodik, acak, atau impulsif). Pada *frequency domain*, sinyal "
        "diubah menjadi spektrum amplitudo terhadap frekuensi sehingga sumber getaran dapat "
        "diidentifikasi: *unbalance* memberikan puncak pada 1x kecepatan putar, *misalignment* "
        "pada 1x dan 2x, sedangkan kerusakan *bearing* menghasilkan puncak pada frekuensi "
        "karakteristik BPFO, BPFI, BSF, dan FTF.",
        color=_BLUE,
    )
    add_paragraph(
        doc,
        "Transformasi dari *time domain* ke *frequency domain* dilakukan dengan *Discrete Fourier "
        "Transform* (DFT). Algoritma *Fast Fourier Transform* (FFT) mengurangi kompleksitas "
        r"komputasi dari $O(N^2)$ menjadi $O(N \log N)$ dengan memanfaatkan simetri periodik "
        "kernel DFT, sehingga FFT menjadi standar de facto pada *predictive maintenance* berbasis "
        "getaran.",
        color=_BLUE,
    )
    add_paragraph(
        doc,
        "Getaran mesin dapat dikuantifikasi dalam tiga parameter yang saling berkaitan. "
        "*Displacement* (satuan mikrometer) mengukur perpindahan dari posisi nominal dan cocok "
        "untuk frekuensi rendah seperti *unbalance* dan *misalignment*. *Velocity* (satuan "
        "mm/s RMS) merupakan parameter yang direkomendasikan standar ISO 10816/20816 untuk "
        "evaluasi kondisi keseluruhan mesin pada frekuensi menengah. *Acceleration* "
        r"(satuan $g$ atau m/s$^2$) sensitif terhadap frekuensi tinggi dan karenanya menjadi "
        "parameter utama untuk deteksi kerusakan *bearing* dan *gear mesh*.",
        color=_BLUE,
    )

    add_heading(
        doc,
        "III.2a.2",
        "Enveloping dan Demodulasi Amplitudo",
        level=2,
        color=_BLUE,
    )
    add_paragraph(
        doc,
        "Kerusakan *bearing* (aus, *pitting*, atau *spalling* pada *race* atau *rolling element*) "
        "menimbulkan impuls berulang saat elemen bergulir melewati cacat. Impuls tersebut sering "
        "tenggelam dalam komponen dominan lain seperti *unbalance* atau getaran *gear mesh*. "
        "Teknik *enveloping* atau demodulasi amplitudo digunakan untuk mengekstraksi impuls "
        "tersembunyi tersebut sehingga BPFO, BPFI, BSF, dan FTF muncul jelas dalam spektrum. "
        "Proses *enveloping* terdiri atas langkah-langkah berikut.",
        color=_BLUE,
    )
    add_numbered(
        doc,
        [
            "**Akuisisi sinyal percepatan.** Sinyal getaran direkam menggunakan akselerometer "
            "karena parameter percepatan sensitif pada frekuensi tinggi tempat impuls *bearing* muncul.",
            "**Filter *band-pass* frekuensi tinggi.** Sinyal dilewatkan filter *band-pass* pada "
            "rentang 5--20 kHz untuk mengisolasi zona frekuensi tempat impuls termodulasi, "
            "sekaligus membuang komponen frekuensi rendah yang mendominasi.",
            "**Rektifikasi.** Sinyal hasil filter diubah ke nilai absolut sehingga komponen "
            "negatif terlipat ke atas.",
            "***Low-pass filter* atau FFT.** Filter *low-pass* atau FFT diterapkan untuk "
            "menghasilkan amplop (*envelope*) sinyal, yakni kurva yang mengikuti puncak-puncak "
            "impuls berulang.",
            "**Analisis spektrum amplop.** FFT dari amplop menghasilkan *envelope spectrum* "
            "di mana puncak pada BPFO, BPFI, BSF, FTF, dan harmoniknya menandai mode kegagalan "
            "*bearing* yang aktif.",
        ],
        color=_BLUE,
    )
    add_paragraph(
        doc,
        "Dalam konteks disertasi ini, *envelope spectrum* digunakan sebagai referensi "
        "eksperimental untuk memvalidasi interpretabilitas laten (*hit-rate* SAE terhadap BPFx) "
        "pada Bab V. Korelasi Pearson antara aktivasi fitur SAE dan amplitudo *envelope spectrum* "
        r"pada pita $\pm$2 Hz di sekitar BPFO, BPFI, BSF, dan FTF merupakan metrik kuantitatif "
        "inti Novelti N3.",
        color=_BLUE,
    )

    add_heading(
        doc,
        "III.2a.3",
        "Tahapan Praproses Data",
        level=2,
        color=_BLUE,
    )
    add_paragraph(
        doc,
        "Sebelum sinyal getaran dapat digunakan untuk melatih model *machine learning* atau "
        "*deep learning*, diperlukan serangkaian tahap praproses yang memproses dan "
        "mentransformasikan data agar dapat ditangani secara efisien oleh model. Tujuh tahapan "
        "praproses yang diterapkan dalam penelitian ini adalah sebagai berikut.",
        color=_BLUE,
    )
    add_numbered(
        doc,
        [
            "**Validasi data sensor.** Data yang terkumpul diperiksa kualitasnya untuk memastikan "
            "rekaman tidak mengandung artefak akuisisi, saturasi *amplifier*, atau kegagalan transmisi.",
            "**Sinkronisasi fitur.** Sinyal dari berbagai sumber (akselerometer, temperatur, "
            "data kualitas produk) yang direkam pada waktu berbeda diselaraskan ke basis deret "
            "waktu yang seragam sebelum pengolahan lebih lanjut.",
            "**Pembersihan data.** Nilai yang hilang (*missing values*) diinterpolasi atau dihapus; "
            "duplikat dibuang; dan *outlier* teridentifikasi melalui metode "
            r"$3\sigma$ atau IQR diproses sesuai kebijakan per-dataset.",
            "**Pengkodean dan diskritisasi.** Fitur kategorikal atau *timestamp* diproyeksikan "
            "ke ruang numerik yang dapat ditangani oleh model. Untuk sinyal getaran kontinu, "
            "tahap ini umumnya berupa konversi tipe data dan pengaturan presisi numerik.",
            "**Segmentasi.** Data panjang dibagi menjadi segmen-segmen menggunakan jendela "
            "geser, memungkinkan analisis paralel dan pemrosesan dataset berskala besar. "
            "Detail konfigurasi jendela per dataset diuraikan pada Subbab III.3.",
            "**Penskalaan fitur.** Normalisasi atau standardisasi diterapkan agar semua fitur "
            "berada pada skala yang sebanding, mencegah dominasi fitur dengan rentang nilai besar. "
            "Standardisasi Z-score berbasis *training set* diterapkan pada penelitian ini.",
            "***Noise handling*.** Komponen derau frekuensi tinggi yang tidak berkorelasi dengan "
            "degradasi *bearing* diatasi melalui kombinasi filter *band-pass* (untuk *enveloping*) "
            "dan pembobotan jendela Hann sebelum FFT pada konstruksi HI.",
        ],
        color=_BLUE,
    )

    # ===================== III.3 =====================
    add_heading(doc, "III.3", "Formulasi Masalah Prediksi Remaining Useful Life", level=1)
    add_heading(doc, "III.3.1", "Definisi Masalah", level=2)
    add_paragraph(
        doc,
        "Diberikan sinyal vibrasi mentah dari suatu *bearing* yang direkam secara periodik selama "
        "masa pakainya hingga mengalami kegagalan (*run to failure*). Pada setiap periode akuisisi "
        r"$t \in \{1, 2, \ldots, T\}$, direkam sinyal vibrasi horizontal dan vertikal dengan panjang "
        "*fixed window* pada frekuensi sampling yang ditentukan oleh dataset. Tujuan prediksi RUL "
        r"adalah menghasilkan fungsi $f_\theta$ (dengan parameter $\theta$) sehingga untuk setiap "
        r"titik waktu $t$, model memberikan prediksi RUL ternormalisasi $\hat{y}_t \in [0, 1]$ "
        "yang merepresentasikan fraksi sisa masa pakai komponen:",
    )
    add_equation_image(
        doc,
        r"\hat{y}_t \;=\; f_\theta\bigl(x_{t-L+1:\,t}\bigr)",
        label="III.1", width_inches=2.6,
    )
    add_paragraph(
        doc,
        r"dengan $x_{t-L+1:t}$ adalah jendela fitur *health indicator* (HI) sepanjang $L$ "
        r"*time step*, dan $L$ adalah panjang *window* input yang pada penelitian ini ditetapkan "
        r"$L = 10$ mengikuti konfigurasi Liu dkk. (2025).",
    )
    add_paragraph(
        doc,
        r"Nilai $\hat{y}_t = 1$ menyatakan kondisi *bearing* baru, sedangkan $\hat{y}_t = 0$ "
        "menyatakan kondisi gagal. Masalah ini dirumuskan sebagai tugas regresi deret waktu "
        "*one-step* dengan target skalar.",
    )

    add_heading(doc, "III.3.2", "Konstruksi Health Indicator", level=2)
    add_paragraph(
        doc,
        "Sinyal vibrasi mentah tidak langsung digunakan sebagai input model, melainkan terlebih "
        "dahulu dikonversi menjadi *health indicator* (HI) melalui pipeline ekstraksi fitur yang "
        "mengikuti praktik terbaik di bidang PHM. Pada penelitian ini digunakan dua pipeline HI yang "
        "berbeda tergantung karakteristik dataset:",
    )
    add_numbered(
        doc,
        [
            "**Pipeline statistik multi-domain** (untuk dataset dengan variasi *fault mode* kaya "
            "seperti PHM 2012): ekstraksi fitur domain waktu (*root mean square*, kurtosis, "
            "skewness, *crest factor*, *peak-to-peak*, *shape factor*, *impulse factor*, *margin "
            "factor*) dan fitur domain frekuensi (*spectral centroid*, *spectral entropy*, *mean "
            "frequency*, *root mean square frequency*, dan energi pada lima pita frekuensi "
            "karakteristik). Hasil ekstraksi menghasilkan vektor fitur 34-dimensi per akuisisi per "
            "sumbu, sehingga total 68 fitur untuk dua sumbu.",
            "**Pipeline Liu isomap HI** (untuk replikasi konfigurasi Liu dkk., 2025): konstruksi HI satu "
            "dimensi menggunakan *isometric mapping* (Isomap) yang memetakan fitur vibrasi "
            "berdimensi tinggi ke manifold satu dimensi yang monoton terhadap progres degradasi. "
            "Pipeline ini diberi label `liu2026_phm` dan `liu2026_xjtu` pada konfigurasi eksperimen.",
        ],
    )
    add_paragraph(
        doc,
        r"Kedua pipeline kemudian dilakukan penghalusan eksponensial (*exponential smoothing*) "
        r"dengan faktor $\alpha = 0{,}1$ untuk menekan *noise* akuisisi, serta normalisasi Min-Max "
        r"ke rentang $[0, 1]$ berdasarkan statistik dataset pelatihan.",
    )

    add_heading(doc, "III.3.3", "Skema Pelabelan RUL", level=2)
    add_paragraph(
        doc,
        "Penelitian ini menggunakan dua skema pelabelan yang keduanya dipertimbangkan dalam "
        "eksperimen:",
    )
    add_numbered(
        doc,
        [
            "**Skema linier** (*linear label scheme*): target RUL diturunkan secara linier dari 1 "
            "pada *time step* awal hingga 0 pada *time step* akhir (*end of life*). Skema ini "
            "sederhana namun memiliki kelemahan karena memaksa model belajar degradasi yang terjadi "
            "sejak awal masa pakai, padahal pada kenyataannya *bearing* berada pada kondisi sehat "
            "untuk periode yang panjang sebelum terjadi degradasi.",
            "**Skema piecewise Liu 2026** (`piecewise_liu2026`): target RUL dipertahankan konstan "
            r"pada nilai 1 hingga titik *degradation onset* $t_d$, kemudian menurun secara linier "
            r"hingga 0 pada *end of life* $t_f$.",
        ],
    )
    add_equation_image(
        doc,
        r"y_t \;=\; 1, \qquad \text{untuk } t < t_d",
        label="III.2a", width_inches=3.2,
    )
    add_equation_image(
        doc,
        r"y_t \;=\; \dfrac{t_f - t}{t_f - t_d}, \qquad \text{untuk } t_d \le t \le t_f",
        label="III.2b", width_inches=3.8,
    )

    add_diagram(
        doc,
        _diag("fig_rul_label_schemes.png"),
        "Gambar III.3 Perbandingan skema pelabelan RUL: skema linier (biru) yang menurun sejak "
        "akuisisi pertama vs skema *piecewise* `liu2026` (hijau) yang mempertahankan target = 1 "
        "selama fase sehat dan baru menurun setelah titik *degradation onset* $t_d$.",
        width_inches=5.5,
    )

    add_paragraph(
        doc,
        r"Skema *piecewise* lebih realistis secara fisik karena *bearing* memang memiliki fase "
        r"sehat yang panjang sebelum mengalami degradasi. Titik *degradation onset* $t_d$ "
        r"ditentukan secara otomatis melalui deteksi perubahan signifikan pada HI menggunakan "
        r"metode *3-sigma* pada jendela *sliding* sebagaimana diusulkan Liu dkk. (2025). Pada "
        r"penelitian ini, skema `piecewise_liu2026` dipilih sebagai skema utama karena "
        r"menghasilkan prediksi yang lebih bermakna secara fisik.",
    )

    # ===================== III.4 =====================
    add_heading(doc, "III.4", "Arsitektur Baseline: xLSTM\u2013Transformer", level=1)
    add_paragraph(
        doc,
        "Arsitektur *baseline* yang direplikasi pada penelitian ini mengikuti proposal Liu dkk. "
        "(2025). Arsitektur ini merupakan *encoder\u2013decoder* yang menggabungkan kekuatan "
        "*Multi-Head Attention* Transformer untuk pengkodean konteks global dengan kekuatan xLSTM "
        "untuk pemodelan dinamika temporal lokal. Diagram lengkap arsitektur tersaji pada "
        "Gambar III.6 di akhir subbab ini.",
    )

    add_heading(doc, "III.4.1", "Komponen sLSTM (scalar LSTM)", level=2)
    add_paragraph(
        doc,
        "Blok sLSTM (Beck dkk., 2024) memperluas LSTM klasik dengan dua modifikasi kunci. "
        "Modifikasi pertama adalah penggantian *sigmoid gating* dengan *exponential gating*, yang "
        r"memungkinkan nilai *gate* melampaui rentang $[0, 1]$ sehingga model dapat merevisi "
        "memori sel secara lebih agresif ketika dibutuhkan. Modifikasi kedua adalah stabilisasi "
        "numerik melalui *normalizer state* yang mencegah *overflow* akibat *exponential gating*. "
        "Skema blok sLSTM ditampilkan pada Gambar III.4 dan persamaan rinci diberikan pada "
        "Persamaan III.3a\u2013g.",
    )

    add_diagram(
        doc,
        _diag("fig_slstm_block.png"),
        "Gambar III.4 Blok *scalar LSTM* (sLSTM) dengan empat *gate* (input, forget, candidate, "
        "output), dua di antaranya menggunakan *exponential gating*, dan komposisi sel skalar "
        "dengan *normalizer state* $n_t$.",
        width_inches=6.0,
    )

    add_paragraph(doc, "Persamaan sLSTM pada waktu $t$ adalah sebagai berikut:")
    add_equation_image(doc, r"z_t = \tanh\!\bigl(W_z\,x_t + R_z\,h_{t-1} + b_z\bigr)",
                       label="III.3a", width_inches=3.6)
    add_equation_image(doc, r"i_t = \exp\!\bigl(W_i\,x_t + R_i\,h_{t-1} + b_i\bigr)",
                       label="III.3b", width_inches=3.6)
    add_equation_image(doc, r"f_t = \exp\!\bigl(W_f\,x_t + R_f\,h_{t-1} + b_f\bigr) \;\text{atau}\; \sigma(\cdot)",
                       label="III.3c", width_inches=4.6)
    add_equation_image(doc, r"o_t = \sigma\!\bigl(W_o\,x_t + R_o\,h_{t-1} + b_o\bigr)",
                       label="III.3d", width_inches=3.6)
    add_equation_image(doc, r"c_t = f_t \odot c_{t-1} + i_t \odot z_t",
                       label="III.3e", width_inches=3.0)
    add_equation_image(doc, r"n_t = f_t \odot n_{t-1} + i_t \quad (\text{normalizer state})",
                       label="III.3f", width_inches=4.2)
    add_equation_image(doc, r"h_t = o_t \odot \bigl(c_t / n_t\bigr)",
                       label="III.3g", width_inches=2.6)
    add_paragraph(
        doc,
        r"dengan $x_t$ input pada waktu $t$, $h_t$ *hidden state*, $c_t$ *cell state*, "
        r"$n_t$ *normalizer state*, $\odot$ perkalian *element-wise*, $\sigma$ fungsi *sigmoid*, "
        r"serta $W_*, R_*$, dan $b_*$ masing-masing matriks bobot input, matriks bobot rekuren, "
        r"dan bias. *Exponential gating* pada $i_t$ dan $f_t$ menjadi kunci kapabilitas sLSTM "
        r"untuk menangani saturasi pada fase akhir masa pakai *bearing*.",
    )

    add_heading(doc, "III.4.2", "Komponen mLSTM (matrix LSTM)", level=2)
    add_paragraph(
        doc,
        "Blok mLSTM (Beck dkk., 2024) memperluas kapasitas memori LSTM dari vektor ke matriks, "
        "terinspirasi dari struktur memori asosiatif. Memori sel "
        r"$C_t \in \mathbb{R}^{d \times d}$ menyimpan pasangan *key" "\u2013" r"value* yang memungkinkan "
        "model mengingat banyak asosiasi sekaligus. Skema blok mLSTM diilustrasikan pada "
        "Gambar III.5 dan persamaan rinci diberikan pada Persamaan III.4a\u2013i.",
    )

    add_diagram(
        doc,
        _diag("fig_mlstm_block.png"),
        "Gambar III.5 Blok *matrix LSTM* (mLSTM) dengan proyeksi *query/key/value*, "
        "*exponential gating*, dan memori matriks $C_t$ berdimensi $d \\times d$.",
        width_inches=6.0,
    )

    add_paragraph(doc, "Persamaan mLSTM adalah sebagai berikut:")
    add_equation_image(doc,
                       r"q_t = W_q x_t + b_q,\quad k_t = W_k x_t + b_k,\quad v_t = W_v x_t + b_v",
                       label="III.4a\u2013c", width_inches=5.0)
    add_equation_image(doc, r"i_t = \exp(W_i x_t + b_i),\quad f_t = \exp(W_f x_t + b_f)",
                       label="III.4d,e", width_inches=4.4)
    add_equation_image(doc, r"o_t = \sigma(W_o x_t + b_o)",
                       label="III.4f", width_inches=2.6)
    add_equation_image(doc,
                       r"C_t = f_t \cdot C_{t-1} + i_t \cdot \bigl(v_t\,k_t^{\!\top} / \sqrt{d}\bigr)",
                       label="III.4g", width_inches=4.4)
    add_equation_image(doc, r"n_t = f_t \cdot n_{t-1} + i_t \cdot k_t",
                       label="III.4h", width_inches=3.0)
    add_equation_image(doc,
                       r"h_t = o_t \odot \bigl(C_t \cdot q_t\bigr) / \max\!\bigl(|n_t^{\!\top} q_t|,\,1\bigr)",
                       label="III.4i", width_inches=4.4)
    add_paragraph(
        doc,
        r"Dibandingkan sLSTM, mLSTM tidak memiliki koneksi rekuren pada *gate* (hanya bergantung "
        r"pada $x_t$), sehingga lebih mudah diparalelkan. Kapasitas memori matriks $C_t$ "
        r"memberikan kemampuan representasi yang jauh lebih kaya, yang menjadi penting ketika "
        r"fitur vibrasi multi-dimensi perlu diingat sepanjang periode yang panjang.",
    )

    add_heading(doc, "III.4.3", "Multi-Head Self-Attention", level=2)
    add_paragraph(
        doc,
        "Komponen *Multi-Head Self-Attention* (MHSA) pada cabang Transformer mengikuti formulasi "
        r"Vaswani dkk. (2017). Untuk setiap kepala $h \in \{1, \ldots, H\}$, dihitung:",
    )
    add_equation_image(
        doc,
        r"\mathrm{Attn}_h(Q,K,V) = \mathrm{softmax}\!\Bigl(\tfrac{(Q W_h^{Q})(K W_h^{K})^{\!\top}}{\sqrt{d_k}}\Bigr)\,(V W_h^{V})",
        label="III.5", width_inches=5.4,
    )
    add_paragraph(
        doc,
        r"dengan $Q, K, V$ masing-masing *query*, *key*, dan *value* yang diproyeksikan dari input "
        r"melalui matriks $W_h^Q, W_h^K, W_h^V$. Output semua kepala kemudian digabungkan dan "
        r"diproyeksikan:",
    )
    add_equation_image(
        doc,
        r"\mathrm{MHSA}(Q,K,V) = \mathrm{Concat}\bigl(\mathrm{Attn}_1, \ldots, \mathrm{Attn}_H\bigr)\,W^{O}",
        label="III.6", width_inches=4.8,
    )
    add_paragraph(
        doc,
        r"Pada implementasi *baseline*, digunakan $H = 4$ kepala dengan $d_{\mathrm{model}} = 32$, "
        r"sehingga $d_k = 8$ per kepala. *Positional encoding* sinusoidal ditambahkan pada input "
        r"sebelum masuk ke MHSA untuk memberikan informasi urutan.",
    )

    add_heading(doc, "III.4.4", "Arsitektur Penuh Encoder\u2013Decoder", level=2)
    add_paragraph(
        doc,
        "Arsitektur xLSTM\u2013Transformer secara keseluruhan terdiri dari komponen-komponen "
        "berikut, sebagaimana digambarkan pada Gambar III.6:",
    )
    add_numbered(
        doc,
        [
            r"**Projeksi input**: lapisan linier $F \to d_{\mathrm{model}}$ yang memetakan vektor "
            "fitur HI menjadi dimensi model.",
            "**Positional encoding**: penambahan *sinusoidal positional encoding* untuk memberi "
            "model informasi urutan temporal.",
            "**Encoder Transformer**: 1 lapisan *encoder* Transformer dengan MHSA "
            r"($H = 4$) dan *feed-forward network* (FFN) dengan dimensi tersembunyi 64.",
            "**Stack xLSTM**: 2 blok mLSTM dengan 1 blok sLSTM pada posisi tengah (konfigurasi "
            "`slstm_positions = [1]`).",
            "**Decoder dengan cross-attention**: 1 lapisan *decoder* yang melakukan "
            "*cross-attention* antara output xLSTM (sebagai *query*) dengan output *encoder* "
            "Transformer (sebagai *key* dan *value*).",
            "**Regression head**: *Layer Normalization* "
            r"$\to \mathrm{Linear}(d_{\mathrm{model}} \to 32) \to$ GELU "
            r"$\to \mathrm{Dropout}(0{,}1) \to \mathrm{Linear}(32 \to 1) \to$ Sigmoid, untuk "
            r"menghasilkan prediksi RUL skalar pada rentang $[0, 1]$.",
        ],
    )

    add_diagram(
        doc,
        _diag("fig_baseline_architecture.png"),
        "Gambar III.6 Diagram arsitektur *baseline* xLSTM\u2013Transformer (Liu dkk., 2025). "
        "Cabang *encoder* Transformer (kiri) dan *stack* xLSTM (kanan) digabungkan melalui "
        "*decoder* dengan *cross-attention* sebelum melalui *regression head*.",
        width_inches=6.2,
    )

    add_paragraph(
        doc,
        "Jumlah parameter total arsitektur *baseline* adalah 43.409 untuk konfigurasi PHM 2012 "
        "dan 43.441 untuk konfigurasi XJTU-SY. Selisih kecil disebabkan oleh perbedaan jumlah "
        "fitur HI antar kedua dataset.",
    )

    # ===================== III.5 =====================
    add_heading(doc, "III.5", "Arsitektur Usulan: Mamba-2-xLSTM-Net", level=1)
    add_heading(doc, "III.5.1", "Motivasi dan Prinsip Desain Hibrid", level=2)
    add_paragraph(
        doc,
        "Arsitektur *baseline* xLSTM\u2013Transformer memiliki dua keterbatasan yang menjadi "
        "motivasi desain model usulan:",
    )
    add_numbered(
        doc,
        [
            "**Kompleksitas kuadratik dari Multi-Head Self-Attention**. MHSA memiliki kompleksitas "
            r"$O(L^2 \cdot d)$ terhadap panjang sekuens. Pada dataset PHM 2012, satu *bearing* "
            "dapat menghasilkan lebih dari 2.800 akuisisi, sehingga jika model diharapkan memproses "
            "konteks yang panjang, kompleksitas kuadratik ini menjadi hambatan nyata baik dari sisi "
            "memori maupun waktu komputasi (lihat ilustrasi Gambar III.2).",
            "**Keterbatasan representasi long-range dependencies**. Walaupun MHSA pada prinsipnya "
            "dapat menjangkau konteks panjang, dalam praktik dibatasi oleh *context window* yang "
            "dapat ditampung memori. Untuk sinyal degradasi *bearing* yang dinamikanya meliputi "
            "fase sehat panjang diikuti fase degradasi pendek, kemampuan menangkap tren global "
            "ribuan *time step* menjadi kritikal.",
        ],
    )
    add_paragraph(
        doc,
        r"*Selective state space model* (Mamba) mengatasi kedua keterbatasan tersebut secara "
        r"simultan: memiliki kompleksitas linear $O(L \cdot d)$ dan secara eksplisit didesain "
        r"untuk pemodelan *long-range dependencies*. Namun demikian, Mamba tidak memiliki "
        r"kapabilitas *exponential gating* dan *matrix memory* yang unggul pada dinamika lokal "
        r"seperti yang dimiliki xLSTM.",
    )
    add_paragraph(
        doc,
        "Prinsip desain arsitektur usulan adalah **memadukan dua kekuatan komplementer** melalui "
        "arsitektur dua cabang (*dual-branch*) dengan mekanisme fusi terpandu (*gated fusion*):",
    )
    add_bullets(
        doc,
        [
            "**Cabang A (xLSTM)** menangkap dinamika lokal degradasi dengan *exponential gating* "
            "yang mampu merespons perubahan tajam pada fase akhir masa pakai.",
            "**Cabang B (Bidirectional Mamba-2)** menangkap tren global sepanjang ribuan *time "
            "step* dengan kompleksitas linear, serta *bidirectional scan* yang memungkinkan setiap "
            "*time step* mengakses konteks masa lalu dan masa depan secara simultan.",
            "**Fusi gated** memungkinkan model belajar secara adaptif kapan mengandalkan sinyal "
            "lokal (cabang A) dan kapan mengandalkan tren global (cabang B), yang diharapkan "
            "bervariasi sepanjang masa pakai *bearing*.",
        ],
    )

    add_heading(doc, "III.5.2", "Selective State Space Model (Mamba-1)", level=2)
    add_paragraph(
        doc,
        "*State space model* (SSM) dalam bentuk kontinu didefinisikan oleh persamaan diferensial:",
    )
    add_equation_image(doc, r"\frac{dh(t)}{dt} = A\,h(t) + B\,x(t)",
                       label="III.7a", width_inches=2.6)
    add_equation_image(doc, r"y(t) = C\,h(t)",
                       label="III.7b", width_inches=1.8)
    add_paragraph(
        doc,
        r"dengan $h(t)$ *hidden state*, $x(t)$ input, $y(t)$ output, serta $A, B, C$ matriks "
        r"parameter. Untuk implementasi pada sinyal diskrit, dilakukan diskritisasi menggunakan "
        r"*zero-order hold* dengan parameter langkah waktu $\Delta$:",
    )
    add_equation_image(doc, r"\bar{A} = \exp(\Delta\,A)",
                       label="III.8a", width_inches=2.0)
    add_equation_image(
        doc,
        r"\bar{B} = (\Delta\,A)^{-1}\bigl(\exp(\Delta\,A) - I\bigr)\,\Delta\,B",
        label="III.8b", width_inches=4.4,
    )
    add_equation_image(doc, r"h_t = \bar{A}\,h_{t-1} + \bar{B}\,x_t",
                       label="III.8c", width_inches=2.6)
    add_equation_image(doc, r"y_t = C\,h_t",
                       label="III.8d", width_inches=1.6)
    add_paragraph(
        doc,
        r"Kontribusi utama Mamba-1 (Gu & Dao, 2023) adalah menjadikan parameter "
        r"$\Delta, B$, dan $C$ sebagai fungsi dari input $x_t$ (*input-dependent*, atau "
        r"dalam istilah lain *selective*):",
    )
    add_equation_image(
        doc,
        r"B_t = \mathrm{Linear}_B(x_t),\quad C_t = \mathrm{Linear}_C(x_t),\quad \Delta_t = \mathrm{softplus}\!\bigl(\mathrm{Linear}_\Delta(x_t)\bigr)",
        label="III.9", width_inches=6.0,
    )

    add_diagram(
        doc,
        _diag("fig_mamba_ssm_block.png"),
        "Gambar III.7 Diagram blok *selective state space model* (Mamba) yang menunjukkan tiga "
        r"komponen utama: parameter selektif $\Delta_t, B_t, C_t$ yang bergantung pada input, "
        "diskritisasi ZOH, dan rekurensi *state-space* dengan *parallel scan* berkompleksitas "
        r"$O(L \cdot d)$.",
        width_inches=6.2,
    )

    add_paragraph(
        doc,
        "Dengan demikian, model dapat secara selektif menentukan informasi mana yang disimpan "
        "dalam *hidden state* dan mana yang diabaikan, bergantung pada input aktual. Kombinasi "
        "selektivitas ini dengan algoritma *parallel scan* yang efisien memungkinkan Mamba-1 "
        "berjalan pada kompleksitas linear dengan tetap mempertahankan ekspresivitas yang "
        "sebanding dengan Transformer.",
    )

    add_heading(doc, "III.5.3", "Mamba-2: Structured State Space Duality", level=2)
    add_paragraph(
        doc,
        r"Mamba-2 (Dao & Gu, 2024) menyempurnakan Mamba-1 melalui kerangka *structured state "
        r"space duality* (SSD) yang mengungkap ekuivalensi matematis antara *selective SSM* "
        r"dengan kelas tertentu dari *causal attention*. Dua perubahan struktural utama "
        r"membedakan Mamba-2 dari Mamba-1: (i) matriks transisi $A$ direstriksi menjadi "
        r"*scalar-times-identity* ($A_t = a_t \cdot I$) untuk memungkinkan blok matriks SSM "
        r"dirumuskan ulang sebagai *block matrix multiplication* yang efisien pada *tensor cores*, "
        r"dan (ii) parameter $\Delta, A, B, C$ dihitung secara *projection-paralel* di awal blok, "
        r"menghilangkan ketergantungan sekuensial yang ada pada Mamba-1. Perbandingan kedua "
        r"varian dirangkum pada Gambar III.8.",
    )

    add_diagram(
        doc,
        _diag("fig_mamba1_vs_mamba2.png"),
        "Gambar III.8 Perbandingan Mamba-1 (Gu & Dao, 2023) dan Mamba-2 (Dao & Gu, 2024). "
        "Mamba-2 menerapkan empat penyederhanaan struktural yang menghasilkan throughput "
        "2\u20138\u00d7 lebih tinggi dengan kapasitas representasi yang serupa.",
        width_inches=6.2,
    )

    add_paragraph(
        doc,
        "Konsekuensi praktis dari restriksi ini adalah peningkatan *throughput* hingga 2\u20138 "
        r"kali lipat dibandingkan Mamba-1 untuk konfigurasi $d_{\mathrm{state}}$ yang sama, "
        r"sekaligus mengurangi *parameter footprint* per blok karena tidak diperlukan "
        r"parameterisasi penuh untuk $A$. Pada penelitian ini, blok Mamba-2 dikonfigurasi dengan "
        r"$d_{\mathrm{state}} = 128$, $\mathrm{headdim} = 32$, dan $\mathrm{expand} = 2$, dengan "
        r"jumlah blok bidireksional tetap 2 untuk perbandingan adil dengan varian Mamba-1.",
    )
    add_paragraph(
        doc,
        r"Varian ketiga, **Mamba-3** (eksplorasi internal), menggunakan $d_{\mathrm{state}} = 256$ "
        r"dengan *head dimension* yang lebih besar untuk menelaah pengaruh peningkatan kapasitas "
        r"SSM terhadap kinerja prediksi RUL. Hasil ablasi tiga varian (Mamba-1/2/3) dilaporkan "
        r"pada subbab IV.5.",
    )

    add_heading(doc, "III.5.4", "Blok Bidirectional Mamba", level=2)
    add_paragraph(
        doc,
        "Pada penelitian ini, Mamba digunakan dalam konfigurasi bidireksional (*bidirectional "
        "Mamba*, BiMamba) untuk memperkaya representasi dengan konteks dua arah, sebagaimana "
        "diilustrasikan pada Gambar III.9. Persamaan BiMamba:",
    )
    add_equation_image(doc, r"h^{\rightarrow}_t = \mathrm{Mamba}_{\mathrm{forward}}(x_{1:t})",
                       label="III.10a", width_inches=3.4)
    add_equation_image(doc, r"h^{\leftarrow}_t = \mathrm{Mamba}_{\mathrm{backward}}(x_{t:L})",
                       label="III.10b", width_inches=3.4)
    add_equation_image(
        doc,
        r"h^{\mathrm{BiMamba}}_t = \mathrm{Linear}\!\Bigl(\bigl[h^{\rightarrow}_t \,;\, h^{\leftarrow}_t\bigr]\Bigr)",
        label="III.10c", width_inches=4.0,
    )

    add_diagram(
        doc,
        _diag("fig_bimamba.png"),
        "Gambar III.9 Blok *Bidirectional Mamba*: konkatenasi *hidden state* dari Mamba *forward* "
        "dan *backward*, kemudian diproyeksikan kembali ke dimensi $d_{\\mathrm{model}}$.",
        width_inches=6.0,
    )

    add_paragraph(
        doc,
        r"dengan $[\;\cdot\;;\;\cdot\;]$ operasi konkatenasi, dan $\mathrm{Linear}$ adalah lapisan "
        r"linier yang memproyeksikan hasil konkatenasi berukuran $2 \times d_{\mathrm{model}}$ "
        r"kembali ke $d_{\mathrm{model}}$. Konfigurasi eksperimen menggunakan 2 blok BiMamba "
        r"dengan parameter dari subbab III.5.3, mengikuti *default* pustaka `mamba-ssm`.",
    )

    add_heading(doc, "III.5.5", "Mekanisme Gated Fusion", level=2)
    add_paragraph(
        doc,
        "Output cabang xLSTM dan cabang BiMamba digabungkan melalui mekanisme *gated fusion* yang "
        "secara adaptif menentukan bobot kontribusi masing-masing cabang pada setiap *time step*, "
        "sebagaimana digambarkan pada Gambar III.10:",
    )
    add_equation_image(
        doc,
        r"g_t = \sigma\!\Bigl(W_g\,\bigl[h^{\mathrm{xLSTM}}_t \,;\, h^{\mathrm{BiMamba}}_t\bigr] + b_g\Bigr)",
        label="III.11a", width_inches=4.6,
    )
    add_equation_image(
        doc,
        r"h^{\mathrm{fused}}_t = g_t \odot h^{\mathrm{xLSTM}}_t + (1 - g_t) \odot h^{\mathrm{BiMamba}}_t",
        label="III.11b", width_inches=5.4,
    )

    add_diagram(
        doc,
        _diag("fig_gated_fusion.png"),
        "Gambar III.10 Mekanisme *gated fusion*: gabungan adaptif representasi xLSTM (lokal) dan "
        "BiMamba-2 (global) melalui *gate* $g_t$ yang dipelajari untuk setiap *time step*.",
        width_inches=6.0,
    )

    add_paragraph(
        doc,
        r"dengan $g_t \in [0, 1]^{d_{\mathrm{model}}}$ *gate* yang dipelajari, $\sigma$ fungsi "
        r"*sigmoid*, serta $W_g$ dan $b_g$ parameter yang dapat dilatih. Secara intuitif, nilai "
        r"$g_t$ mendekati 1 berarti model lebih mengandalkan representasi lokal dari xLSTM, "
        r"sedangkan $g_t$ mendekati 0 berarti lebih mengandalkan representasi global dari "
        r"BiMamba. *Gate* ini dapat divariasi per *time step*, memungkinkan model secara dinamis "
        r"menyesuaikan strategi fusi sepanjang masa pakai *bearing*.",
    )

    add_heading(doc, "III.5.6", "Regression Head", level=2)
    add_paragraph(
        doc,
        "*Regression head* pada arsitektur usulan mengikuti desain yang kompatibel dengan "
        "*baseline* untuk memastikan perbandingan yang adil:",
    )
    add_equation_image(doc, r"\hat{z}_t = \mathrm{LayerNorm}\bigl(h^{\mathrm{fused}}_t\bigr)",
                       label="III.12a", width_inches=3.2)
    add_equation_image(doc, r"\hat{z}_t = \mathrm{GELU}\!\bigl(\mathrm{Linear}(d_{\mathrm{model}} \to 64)(\hat{z}_t)\bigr)",
                       label="III.12b", width_inches=4.8)
    add_equation_image(doc, r"\hat{z}_t = \mathrm{Dropout}(0{,}1)(\hat{z}_t)",
                       label="III.12c", width_inches=3.0)
    add_equation_image(doc, r"\hat{y}_t = \mathrm{Sigmoid}\!\bigl(\mathrm{Linear}(64 \to 1)(\hat{z}_t)\bigr)",
                       label="III.12d", width_inches=4.4)
    add_paragraph(doc, "*Dropout* ditempatkan setelah GELU untuk regularisasi.")

    add_heading(doc, "III.5.7", "Arsitektur Penuh dan Jumlah Parameter", level=2)
    add_paragraph(
        doc,
        r"Konfigurasi yang digunakan dalam eksperimen adalah: $d_{\mathrm{model}} = 128$, 3 blok "
        r"xLSTM dengan sLSTM pada posisi 1, 2 blok BiMamba (varian Mamba-2 untuk model usulan "
        r"final), `head_hidden = 64`, dan `dropout = 0,1`. Skema arsitektur lengkap disajikan "
        r"pada Gambar III.11. Jumlah parameter total Mamba-2-xLSTM-Net adalah 811.585 untuk "
        r"konfigurasi PHM 2012 dan 811.713 untuk konfigurasi XJTU-SY. Varian ablasi "
        r"Mamba-1-xLSTM-Net memiliki jumlah parameter yang lebih besar (955.921 pada PHM 2012) "
        r"karena parameterisasi penuh matriks $A$; sebaliknya, Mamba-2-xLSTM-Net memanfaatkan "
        r"restriksi $A = a \cdot I$ untuk menghemat parameter.",
    )

    add_diagram(
        doc,
        _diag("fig_proposed_architecture.png"),
        "Gambar III.11 Diagram arsitektur usulan **Mamba-2-xLSTM-Net** (final). Cabang xLSTM "
        "(kiri) dan cabang BiMamba-2 (kanan) digabungkan melalui *gated fusion* sebelum melalui "
        "*regression head*. Parameter total: 811,6 K (PHM 2012) / 811,7 K (XJTU-SY).",
        width_inches=6.4,
    )

    add_paragraph(
        doc,
        r"Peningkatan jumlah parameter dibandingkan *baseline* (sekitar 19$\times$) merupakan "
        r"konsekuensi logis dari peningkatan $d_{\mathrm{model}}$ dari 32 menjadi 128 serta "
        r"keberadaan dua cabang paralel. Implikasi parameterisasi yang berlebih "
        r"(*overparameterization*) ini dianalisis pada Bab V.",
    )

    add_table(
        doc,
        ["Aspek", "Baseline xLSTM\u2013Transformer", "Mamba-2-xLSTM-Net (final)", "Mamba-1-xLSTM-Net (ablasi)"],
        [
            ["d_model", "32", "128", "128"],
            ["Blok xLSTM", "2 mLSTM + 1 sLSTM", "2 mLSTM + 1 sLSTM", "2 mLSTM + 1 sLSTM"],
            ["Blok Transformer / Mamba", "1 encoder + 1 decoder", "2 BiMamba-2", "2 BiMamba-1"],
            ["Mekanisme fusi", "Cross-attention", "Gated fusion", "Gated fusion"],
            ["Kompleksitas attention / scan", "O(L\u00b2 \u00b7 d)", "O(L \u00b7 d)", "O(L \u00b7 d)"],
            ["Parameter (PHM 2012)", "43.409", "811.585", "955.921"],
            ["Parameter (XJTU-SY)", "43.441", "\u2014 (belum dijalankan)", "811.713"],
        ],
        caption="Tabel III-10. Perbandingan parameter dan kompleksitas tiga arsitektur (baseline, model usulan final, dan ablasi).",
        first_col_bold=True,
    )

    # ===================== III.6 =====================
    add_heading(doc, "III.6", "Protokol Pelatihan", level=1)
    add_paragraph(
        doc,
        "Untuk memastikan perbandingan yang adil antara kedua arsitektur, protokol pelatihan "
        "dibuat identik sejauh memungkinkan. Pengecualian ada pada parameter arsitektural "
        r"($d_{\mathrm{model}}$, jumlah blok) yang secara inheren berbeda. Ringkasan protokol "
        "ditampilkan pada Gambar III.12.",
    )

    add_diagram(
        doc,
        _diag("fig_training_protocol.png"),
        "Gambar III.12 Ringkasan protokol pelatihan: identik antara *baseline* xLSTM\u2013Transformer "
        "dan model usulan Mamba-2-xLSTM-Net.",
        width_inches=6.2,
    )

    add_heading(doc, "III.6.1", "Fungsi Loss", level=2)
    add_paragraph(
        doc,
        "Fungsi *loss* yang digunakan adalah *Mean Squared Error* (MSE) antara prediksi RUL dan "
        "target:",
    )
    add_equation_image(
        doc,
        r"\mathcal{L}_{\mathrm{MSE}}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \bigl(y_i - \hat{y}_i(\theta)\bigr)^2",
        label="III.13", width_inches=4.4,
    )
    add_paragraph(
        doc,
        r"dengan $N$ jumlah sampel pelatihan. MSE dipilih karena kompatibel dengan *output* "
        r"sigmoid pada *regression head*, serta memberikan penalti lebih besar pada kesalahan "
        r"besar, yang sesuai untuk prediksi RUL di mana kesalahan besar di dekat *end of life* "
        r"memiliki konsekuensi fatal.",
    )

    add_heading(doc, "III.6.2", "Optimizer dan Hyperparameter Pelatihan", level=2)
    add_paragraph(
        doc,
        "Kedua arsitektur dilatih menggunakan *optimizer* Adam (Kingma & Ba, 2015) dengan "
        "konfigurasi berikut:",
    )
    add_bullets(
        doc,
        [
            r"*Learning rate*: $1 \times 10^{-3}$",
            "*Weight decay*: 0",
            "*Scheduler*: tidak digunakan (konstan sepanjang pelatihan)",
            "*Gradient clipping*: norma maksimum 1,0",
            "*Early stopping patience*: 9999 (efektif tidak aktif; model dilatih penuh 50 *epoch*)",
            "*Monotonicity weight*: 0 (tidak menggunakan penalti monotonisitas tambahan)",
            "*Precision*: FP32",
            "*Batch size*: 32",
            "*Max epochs*: 50",
        ],
    )
    add_paragraph(
        doc,
        "*Seed* acak ditetapkan pada 42 untuk memastikan reprodusibilitas eksperimen. "
        "*Checkpoint* terbaik dipilih berdasarkan metrik `train/loss` untuk konsistensi antara "
        "kedua arsitektur, sesuai konfigurasi YAML yang dilampirkan pada Lampiran B.",
    )

    add_heading(doc, "III.6.3", "Perangkat Keras dan Perangkat Lunak", level=2)
    add_paragraph(
        doc,
        "Seluruh eksperimen dijalankan pada satu GPU NVIDIA dengan dukungan CUDA untuk memastikan "
        "kompatibilitas dengan pustaka `mamba-ssm` yang memerlukan kernel CUDA terkustom. "
        "Kerangka kerja pembelajaran mendalam yang digunakan adalah PyTorch 2.1 dengan PyTorch "
        "Lightning 2.1 untuk orkestrasi pelatihan, serta Hydra 1.3 untuk manajemen konfigurasi. "
        "Pustaka utama lainnya meliputi `xlstm` 1.0 (NX-AI), `mamba-ssm` 2.2 (mendukung Mamba-1 "
        "dan Mamba-2 backend), dan `causal-conv1d` 1.2.",
    )

    # ===================== III.7 =====================
    add_heading(doc, "III.7", "Metrik Evaluasi", level=1)
    add_paragraph(
        doc,
        "Empat metrik utama digunakan untuk mengevaluasi dan membandingkan model, masing-masing "
        "menyorot aspek kinerja yang berbeda.",
    )

    add_heading(doc, "III.7.1", "Root Mean Squared Error (RMSE)", level=2)
    add_paragraph(
        doc,
        "Metrik regresi utama yang mengukur deviasi rata-rata prediksi dari nilai sebenarnya dalam "
        "satuan yang sama dengan target:",
    )
    add_equation_image(
        doc,
        r"\mathrm{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \bigl(y_i - \hat{y}_i\bigr)^2}",
        label="III.14", width_inches=3.6,
    )
    add_paragraph(
        doc,
        "Nilai lebih kecil menunjukkan kinerja lebih baik. RMSE sensitif terhadap kesalahan besar "
        "dan menjadi metrik standar pada prediksi RUL.",
    )

    add_heading(doc, "III.7.2", "Mean Absolute Error (MAE)", level=2)
    add_paragraph(doc, "Metrik sekunder yang lebih *robust* terhadap *outlier*:")
    add_equation_image(
        doc,
        r"\mathrm{MAE} = \frac{1}{N} \sum_{i=1}^{N} \bigl|y_i - \hat{y}_i\bigr|",
        label="III.15", width_inches=3.0,
    )

    add_heading(doc, "III.7.3", "Koefisien Determinasi (R\u00b2)", level=2)
    add_paragraph(
        doc,
        "Metrik *goodness-of-fit* yang mengukur fraksi varians data yang dijelaskan oleh model:",
    )
    add_equation_image(
        doc,
        r"R^2 = 1 - \frac{\sum_{i=1}^{N} \bigl(y_i - \hat{y}_i\bigr)^2}{\sum_{i=1}^{N} \bigl(y_i - \bar{y}\bigr)^2}",
        label="III.16", width_inches=3.6,
    )
    add_paragraph(
        doc,
        r"dengan $\bar{y}$ rata-rata target. Nilai 1 menunjukkan prediksi sempurna, 0 menunjukkan "
        r"kinerja setara dengan menebak rata-rata, dan nilai negatif menunjukkan kinerja lebih "
        r"buruk daripada menebak rata-rata. Perlu dicatat bahwa pada skema `piecewise_liu2026`, "
        r"target RUL sebagian besar bernilai 1 (fase sehat panjang), sehingga varians target "
        r"kecil dan $R^2$ menjadi metrik yang kurang informatif pada konteks ini. Analisis lebih "
        r"lanjut diberikan pada Bab IV dan V.",
    )

    add_heading(doc, "III.7.4", "PHM Score", level=2)
    add_paragraph(
        doc,
        "*Scoring function* standar kompetisi IEEE PHM 2012 (Nectoux dkk., 2012) yang memberikan "
        r"penalti asimetris: prediksi yang terlalu dini (optimistik, $\mathrm{RUL}_{\mathrm{pred}} > \mathrm{RUL}_{\mathrm{true}}$) "
        r"diberi penalti lebih ringan, sedangkan prediksi yang terlambat (pesimistik, "
        r"$\mathrm{RUL}_{\mathrm{pred}} < \mathrm{RUL}_{\mathrm{true}}$) diberi penalti lebih "
        "berat, karena secara operasional lebih aman memperkirakan *bearing* akan gagal lebih "
        "cepat daripada lebih lama. Bentuk kurva penalti asimetris diilustrasikan pada "
        "Gambar III.13.",
    )
    add_equation_image(
        doc,
        r"E_{r,i} = 100 \cdot \frac{\mathrm{RUL}_{\mathrm{actual},i} - \mathrm{RUL}_{\mathrm{pred},i}}{\mathrm{RUL}_{\mathrm{actual},i}}",
        label="III.17a", width_inches=5.2,
    )
    add_equation_image(
        doc,
        r"A_i = \exp\!\bigl(-\ln(0{,}5) \cdot (E_{r,i}/5)\bigr),\quad \text{jika } E_{r,i} \le 0",
        label="III.17b", width_inches=5.4,
    )
    add_equation_image(
        doc,
        r"A_i = \exp\!\bigl(+\ln(0{,}5) \cdot (E_{r,i}/20)\bigr),\quad \text{jika } E_{r,i} > 0",
        label="III.17c", width_inches=5.6,
    )
    add_equation_image(
        doc,
        r"\mathrm{PHM\,Score} = \frac{1}{N} \sum_{i=1}^{N} A_i",
        label="III.17d", width_inches=3.2,
    )

    add_diagram(
        doc,
        _diag("fig_phm_scoring.png"),
        "Gambar III.13 Kurva *PHM Score* asimetris (IEEE PHM 2012, Nectoux dkk., 2012). "
        "Prediksi terlalu dini (sisi kiri, $E_r \\le 0$) menerima penalti lebih ringan dibanding "
        "prediksi terlambat (sisi kanan, $E_r > 0$).",
        width_inches=5.5,
    )

    add_paragraph(
        doc,
        r"Nilai lebih besar menunjukkan kinerja lebih baik, dengan nilai maksimum 1,0. Pada "
        r"penelitian ini dilaporkan dua varian PHM Score: (1) `phm_score` versi proyek yang "
        r"dihitung pada RUL ternormalisasi, dan (2) `phm_score_paper` yang dihitung sesuai "
        r"protokol Liu dkk. (2025) dalam satuan waktu fisik. Keduanya dilaporkan untuk transparansi.",
    )

    add_heading(doc, "III.7.5", "Metrik Tambahan per-Bearing", level=2)
    add_paragraph(
        doc,
        "Selain metrik agregat, dilaporkan pula metrik per-*bearing* untuk mengevaluasi "
        "konsistensi kinerja model di berbagai kondisi operasi. Metrik per-*bearing* meliputi "
        "`rmse_per_bearing`, `phm_per_bearing`, dan `phm_paper_per_bearing`.",
    )

    # ===================== III.8 =====================
    add_heading(doc, "III.8", "Rancangan Eksperimen", level=1)
    add_paragraph(
        doc,
        "Eksperimen disusun dalam empat studi kasus yang diurutkan secara inkremental dari "
        "validasi metode pada dataset publik hingga rencana transfer ke objek penelitian SKF "
        "Indonesia.",
    )

    add_heading(doc, "III.8.1", "Studi Kasus I: Replikasi Baseline pada Dataset Publik", level=2)
    add_paragraph(
        doc,
        "**Tujuan**: memverifikasi kebenaran implementasi *baseline* xLSTM\u2013Transformer "
        "melalui replikasi hasil Liu dkk. (2025) pada dataset PHM 2012 dan XJTU-SY.",
        indent_first_line=False,
    )
    add_paragraph(doc, "**Tahapan**:", indent_first_line=False)
    add_numbered(
        doc,
        [
            "Unduh dan pra-pemrosesan dataset sesuai pipeline yang dijelaskan pada Bab IV.",
            "Implementasi arsitektur sesuai subbab III.4.",
            "Pelatihan model dengan konfigurasi pada subbab III.6.",
            "Evaluasi pada *bearing* uji yang ditetapkan dengan metrik pada subbab III.7.",
            "Perbandingan hasil dengan angka yang dilaporkan Liu dkk. (2025).",
        ],
    )
    add_paragraph(
        doc,
        "**Kriteria sukses**: RMSE pada *bearing* uji berada dalam rentang \u00b110% dari angka "
        "yang dilaporkan Liu dkk. (2025) pada kedua dataset.",
        indent_first_line=False,
    )

    add_heading(doc, "III.8.2", "Studi Kasus II: Evaluasi Arsitektur Usulan Mamba-2-xLSTM-Net", level=2)
    add_paragraph(
        doc,
        "**Tujuan**: membandingkan kinerja arsitektur usulan terhadap *baseline* pada kedua "
        "dataset publik dengan protokol pelatihan dan evaluasi yang identik.",
        indent_first_line=False,
    )
    add_paragraph(doc, "**Tahapan**:", indent_first_line=False)
    add_numbered(
        doc,
        [
            "Implementasi arsitektur Mamba-2-xLSTM-Net sesuai subbab III.5.",
            "Pelatihan dengan konfigurasi yang identik dengan Studi Kasus I, *seed* 42.",
            "Evaluasi dengan metrik yang sama.",
            "Perbandingan langsung (*head-to-head*) pada setiap metrik dan setiap *bearing* uji.",
            "Analisis statistik (uji *Wilcoxon signed-rank* pada RMSE per-*bearing*) untuk menilai "
            "signifikansi perbedaan kinerja \u2014 direncanakan untuk iterasi multi-seed berikutnya.",
        ],
    )
    add_paragraph(
        doc,
        "**Kriteria sukses**: Mamba-2-xLSTM-Net menunjukkan peningkatan yang signifikan pada "
        "minimal dua dari empat metrik utama pada setidaknya satu dataset, dengan sinyal yang "
        "konsisten pada *bearing* uji yang sekuensnya panjang.",
        indent_first_line=False,
    )

    add_heading(doc, "III.8.3", "Studi Kasus III: Ablasi dan Analisis Interpretabilitas", level=2)
    add_paragraph(
        doc,
        "**Tujuan**: mengisolasi kontribusi masing-masing komponen arsitektur usulan serta "
        "memberikan interpretasi mekanistik terhadap fitur degradasi yang dipelajari model. "
        "Konfigurasi ablasi yang akan dijalankan dirangkum pada Tabel III-11.",
        indent_first_line=False,
    )

    add_table(
        doc,
        ["ID", "Konfigurasi", "Komponen yang dihapus/diganti", "Hipotesis"],
        [
            ["A0", "Mamba-1 vs Mamba-2 vs Mamba-3 backbone",
             "Variasi blok BiMamba pada cabang B",
             "Menguji pengaruh kapasitas dan struktur SSM (sudah dijalankan sebagian, lihat IV.5)"],
            ["A1", "xLSTM-only", "Hilangkan cabang BiMamba",
             "Menguji kontribusi cabang Mamba"],
            ["A2", "Mamba-only", "Hilangkan cabang xLSTM",
             "Menguji kontribusi cabang xLSTM"],
            ["A3", "Unidirectional Mamba", "BiMamba \u2192 Mamba satu arah",
             "Menguji kontribusi bidireksionalitas"],
            ["A4", "Concat fusion", "Gated fusion \u2192 konkatenasi sederhana",
             "Menguji kontribusi gating"],
            ["A5", "LSTM-biasa sebagai pengganti xLSTM",
             "xLSTM \u2192 LSTM standar",
             "Menguji kontribusi exponential gating"],
            ["A6", "Tanpa exponential smoothing HI", "\u03b1 = 0 (tanpa pemulusan)",
             "Menguji kontribusi pra-pemrosesan"],
            ["A7", "Window pendek (L = 5) vs panjang (L = 50)",
             "Variasi panjang window input",
             "Menguji kebutuhan konteks panjang"],
        ],
        caption="Tabel III-11. Konfigurasi studi ablasi (A0 sudah dijalankan sebagian).",
        first_col_bold=True,
    )

    add_paragraph(
        doc,
        "Untuk analisis interpretabilitas, akan diterapkan tiga metode komplementer:",
    )
    add_numbered(
        doc,
        [
            "**SHAP (SHapley Additive exPlanations)**: melalui `shap.GradientExplainer` pada "
            "Mamba-2-xLSTM-Net terlatih, untuk mengidentifikasi fitur HI yang paling berpengaruh "
            "pada prediksi RUL secara global dan per-*bearing*.",
            "**Sparse Autoencoder (SAE)**: dilatih pada representasi tersembunyi lapisan fusi "
            "untuk menemukan fitur-fitur laten yang dapat diinterpretasi. Setiap laten SAE "
            "kemudian dipetakan ke pola degradasi khas (misalnya peningkatan RMS tajam, lonjakan "
            "impuls, saturasi fase akhir).",
            "**Integrated Gradients**: untuk atribusi temporal, yaitu mengidentifikasi *time step* "
            "dalam *window* input yang paling berpengaruh terhadap prediksi pada titik tertentu.",
        ],
    )
    add_paragraph(
        doc,
        "Analisis ini bertujuan memberikan kepercayaan mekanistik pada model usulan, yang "
        "merupakan salah satu poin lemah arsitektur *deep learning* pada aplikasi industri.",
    )

    add_heading(
        doc,
        "III.8.4",
        "Studi Kasus IV: Rencana Transfer ke Objek PT SKF Indonesia (Tahap Akhir Penelitian)",
        level=2,
    )
    add_paragraph(
        doc,
        "**Tujuan**: mengadaptasi model Mamba-2-xLSTM-Net yang telah divalidasi pada dataset "
        "publik untuk diterapkan pada data vibrasi mesin *grinding* OR1 dan OR2 di Channel 15 "
        "PT SKF Indonesia, dengan integrasi data kualitas produk (*vibration checking* dan "
        "*radial clearance checking*) sebagai fitur tambahan.",
        indent_first_line=False,
    )
    add_paragraph(
        doc,
        "Rencana transfer ini akan dibahas secara ringkas pada bagian akhir Bab V sebagai hasil "
        "antara dan rencana kerja lanjutan, dengan tahapan sebagai berikut:",
    )
    add_numbered(
        doc,
        [
            "Pemetaan fitur HI dataset publik ke format data SKF (menggunakan *exporter* XLSX dari "
            "SKF Observer).",
            "Konstruksi HI menggunakan pipeline serupa `liu2026_phm` yang telah divalidasi.",
            "*Fine-tuning* model Mamba-2-xLSTM-Net yang telah dilatih pada dataset publik, dengan "
            "pendekatan *transfer learning*.",
            "Integrasi data kualitas produk sebagai *auxiliary input* melalui *feature-level "
            "fusion*.",
            "Evaluasi pada data OR1 dan OR2, dengan validasi silang antar mesin.",
        ],
    )
    add_paragraph(
        doc,
        "Kerangka rancangan detail untuk Studi Kasus IV akan dikembangkan lebih lanjut pada "
        "iterasi disertasi berikutnya, sesuai skema pendekatan penelitian pada Gambar I.14.",
    )

    # ===================== III.9 =====================
    add_heading(doc, "III.9", "Diagram Alur Penelitian", level=1)
    add_paragraph(
        doc,
        "Gambar III.14 menyajikan diagram alur keseluruhan penelitian, mulai dari identifikasi "
        "masalah dan kajian literatur, pengumpulan dan pra-pemrosesan data, implementasi model "
        "*baseline* dan usulan, eksperimen komparatif pada dataset publik, analisis "
        "interpretabilitas, hingga rencana transfer ke objek penelitian SKF Indonesia. Diagram "
        "ini menunjukkan dengan jelas bahwa penelitian pada tahap ini berfokus pada validasi "
        "metode (Studi Kasus I\u2013III) pada dataset publik, dengan penerapan pada objek nyata "
        "sebagai tahap akhir.",
    )

    add_diagram(
        doc,
        _diag("fig_research_workflow.png"),
        "Gambar III.14 Diagram alur penelitian: dari kajian literatur hingga rencana transfer ke "
        "PT SKF Indonesia.",
        width_inches=6.0,
    )

    add_blockquote(
        doc,
        "Catatan: pada disertasi versi penuh, diagram di atas akan diperluas dengan label timeline "
        "pelaksanaan dan ditampilkan dalam orientasi *landscape* bila diperlukan.",
    )
