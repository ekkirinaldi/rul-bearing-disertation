# Bab III Pengembangan Model

Bab ini menguraikan pengembangan model yang dilakukan dalam penelitian ini. Subbab III.1 menyajikan sintesa *state of the art* yang menjadi landasan pemilihan model. Subbab III.2 menjelaskan proses penyempitan fokus penelitian dari enam algoritma kandidat awal menjadi dua arsitektur *state of the art* setelah dilakukan kajian literatur lanjutan. Subbab III.3 memformulasikan masalah prediksi *Remaining Useful Life* (RUL) secara matematis. Subbab III.4 dan III.5 masing-masing menguraikan arsitektur model acuan (*baseline*) xLSTM–Transformer dan model usulan Mamba-xLSTM-Net secara rinci. Subbab III.6 menjelaskan protokol pelatihan, subbab III.7 metrik evaluasi, dan subbab III.8 rancangan eksperimen beserta skenario validasi silang.

## III.1   Sintesa Model *State of the Art*

Peneliti melakukan pemaparan *state of the art* (SotA) penelitian di bidang *predictive maintenance* (PdM) yang telah disajikan pada Tabel III-1 hingga Tabel III-5. Sintesa model acuan yang dipilih berdasarkan fokus penelitian ini ditampilkan pada Tabel III-6. Berdasarkan hasil pemaparan tersebut, awalnya diidentifikasi enam pendekatan utama sebagai kandidat algoritma: (1) XGBoost sebagai *baseline* *traditional machine learning*, (2) CNN-LSTM (Khorram dkk., 2021), (3) *Temporal Convolutional Network* (TCN) dengan *graph fusion* (Li dkk., 2020), (4) BiLSTM dengan *Hierarchical Improved Fusion Attention* (Huang dkk., 2025), (5) GRU *lightweight* (Bai dkk., 2025), dan (6) hibrid ARIMA-LSTM (Hamiane dkk., 2024). Keenam kandidat tersebut merepresentasikan spektrum pendekatan yang cukup lengkap, mulai dari *traditional machine learning* berbasis rekayasa fitur hingga *deep learning* *end-to-end* dengan berbagai mekanisme perhatian.

Selama pelaksanaan penelitian, dilakukan kajian literatur lanjutan yang mengungkap perkembangan signifikan di bidang pemodelan sekuens pada rentang tahun 2023–2025. Perkembangan tersebut bermuara pada munculnya dua arsitektur baru yang menunjukkan kinerja unggul pada berbagai tugas pemodelan deret waktu, yaitu *extended Long Short-Term Memory* (xLSTM) yang diperkenalkan oleh Beck dkk. (2024) dan *Mamba* (*selective state space model*) yang diperkenalkan oleh Gu & Dao (2023) beserta penerusnya Mamba-2 oleh Dao & Gu (2024). Temuan literatur ini, ditambah dengan publikasi Liu dkk. (2025) mengenai penerapan xLSTM–Transformer untuk prediksi RUL *rolling element bearing* pada dataset XJTU-SY dan PHM 2012, mendorong peneliti untuk menyempitkan fokus penelitian dari enam kandidat algoritma menjadi dua arsitektur SotA yang akan dibandingkan secara mendalam. Proses penyempitan ini dijelaskan pada subbab berikutnya.

## III.2   Penyempitan Fokus Penelitian: Evolusi Menuju Arsitektur *State of the Art*

### III.2.1   Rasional Penyempitan

Benchmarking enam algoritma sebagaimana direncanakan pada Tabel III-7 memiliki keunggulan dalam cakupan spektrum pendekatan, namun memiliki dua kelemahan metodologis yang mendasar. *Pertama*, keenam algoritma tersebut diusulkan pada tahun 2018–2025 dengan baseline komparasi yang berbeda-beda, sehingga perbandingan langsung di antara keenamnya tidak memiliki titik referensi tunggal yang kuat. *Kedua*, kedalaman analisis pada masing-masing algoritma menjadi terbatas bila dibagi untuk enam varian, sehingga kontribusi ilmiah yang dihasilkan cenderung berupa *empirical benchmarking* tanpa novelty arsitektural yang substantif. Kedua kelemahan ini berpotensi melemahkan posisi penelitian sebagai disertasi doktoral yang menuntut *novelty and originality* sebagaimana disyaratkan dalam Pedoman Penulisan Disertasi Doktor ITB.

Oleh karena itu, peneliti melakukan penyempitan fokus berdasarkan dua pertimbangan utama. Pertama, pemilihan arsitektur acuan (*baseline*) yang paling mutakhir dan terbukti unggul untuk prediksi RUL *bearing* berdasarkan publikasi terbaru. Kedua, pengajuan satu arsitektur usulan (*proposed*) yang mengisi celah teori (*gap*) yang belum dikerjakan dalam literatur. Hasil penyempitan ini adalah perbandingan head-to-head antara xLSTM–Transformer sebagai *baseline* dan Mamba-xLSTM-Net sebagai arsitektur usulan.

### III.2.2   Evolusi Pemodelan Sekuens 2014–2025

Pemodelan sekuens dalam pembelajaran mendalam telah mengalami evolusi pesat dalam satu dasawarsa terakhir. Evolusi tersebut dapat dikelompokkan ke dalam empat generasi utama, sebagaimana diringkas pada Tabel III-8.

**Tabel III-8.** Evolusi pemodelan sekuens dalam pembelajaran mendalam

| Generasi | Periode | Arsitektur utama | Kompleksitas | Keterbatasan utama |
| :--- | :--- | :--- | :--- | :--- |
| Pertama | 1997–2014 | LSTM, GRU | O(L · d²) | *Vanishing gradient* pada sekuens panjang; memori tersembunyi skalar berkapasitas rendah |
| Kedua | 2017–2023 | Transformer, Multi-Head Attention | O(L² · d) | Kompleksitas kuadratik terhadap panjang sekuens L; konteks panjang mahal secara komputasi |
| Ketiga | 2023–2024 | xLSTM, Mamba (*selective SSM*) | O(L · d) atau O(L · d²) linear | Varian baru; interpretabilitas mekanistik masih terbatas |
| Keempat | 2024–2025 | Mamba-2, xLSTM–Transformer hibrid | O(L · d) | Integrasi dengan dataset PHM industri masih sangat terbatas |

Keterangan: L adalah panjang sekuens input, d adalah dimensi model.

LSTM generasi pertama (Hochreiter & Schmidhuber, 1997) dan varian GRU (Cho dkk., 2014) memiliki keterbatasan kapasitas memori tersembunyi yang berupa vektor skalar dan rentan terhadap *vanishing gradient*. Transformer generasi kedua (Vaswani dkk., 2017) mengatasi keterbatasan tersebut melalui mekanisme *self-attention*, namun dengan konsekuensi kompleksitas kuadratik O(L²) yang menjadi kendala ketika panjang sekuens mencapai ribuan *time step* sebagaimana umumnya terjadi pada sinyal vibrasi *bearing*.

Dua terobosan generasi ketiga masing-masing mengatasi kelemahan yang berbeda. xLSTM (Beck dkk., 2024) memperluas LSTM klasik dengan dua modifikasi penting: (1) *exponential gating* untuk mengatasi masalah saturasi akhir masa pakai komponen, dan (2) *matrix memory cell* (mLSTM) yang meningkatkan kapasitas memori dari skalar menjadi matriks, sehingga mampu menampung representasi fitur yang jauh lebih kaya. Mamba (Gu & Dao, 2023) mengambil jalur berbeda melalui *selective state space model* (SSM) yang mencapai kompleksitas linear O(L · d) dengan tetap mempertahankan kemampuan menangkap *long-range dependencies*. Mamba-2 (Dao & Gu, 2024) menunjukkan bahwa Transformer dan SSM sesungguhnya merupakan dua sisi dari kerangka matematis yang sama (*structured state space duality*), sehingga kedua pendekatan tersebut saling melengkapi, bukan saling meniadakan.

Pada generasi keempat, kombinasi antar arsitektur mulai diekplorasi. Liu dkk. (2025) mengusulkan xLSTM–Transformer untuk prediksi RUL *bearing* dengan hasil yang lebih baik dibandingkan LSTM dan LSTM–Transformer pada dataset XJTU-SY dan PHM 2012. Wang dkk. (2025) dan Liu F. dkk. (2025) masing-masing menunjukkan potensi Mamba pada peramalan deret waktu umum dan prediksi RUL *aero-engine*. Namun demikian, sepengetahuan peneliti pada saat penulisan, **belum terdapat publikasi yang menggabungkan Mamba dengan xLSTM khusus untuk prediksi RUL *rolling element bearing***. Hal ini membentuk celah teori (*gap theory*) yang menjadi fokus disertasi ini.

### III.2.3   Celah Teori dan Posisi Kontribusi

Tabel III-9 merangkum hasil pemetaan literatur Mamba dan xLSTM dalam dua dimensi: arsitektur yang digunakan dan domain aplikasi.

**Tabel III-9.** Pemetaan literatur Mamba dan xLSTM pada domain PHM (*Prognostics and Health Management*)

| No | Peneliti (Tahun) | Arsitektur | Domain | Gap dengan disertasi ini |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Liu dkk. (2025) | xLSTM + Transformer *Multi-Head Attention* | *Bearing* RUL (XJTU-SY, PHM 2012) | Tidak menggunakan Mamba; menjadi *baseline* acuan disertasi ini |
| 2 | Wang dkk. (2025) | Mamba (murni) | Peramalan deret waktu umum | Bukan untuk RUL *bearing*; tidak menggunakan xLSTM |
| 3 | Liu F. dkk. (2025) | *Enhanced Mamba* + *Multi-Head Attention* + *Learnable Scaling* | RUL *aero-engine* dan baterai | Bukan untuk *bearing*; tidak menggunakan xLSTM |
| 4 | Mamba-SDP (2025) | Mamba + *Scaled Dot-Product Attention* + FFT | RUL *bearing* | Tidak menggunakan xLSTM |
| 5 | Dao & Gu (2024) | Mamba-2 (*structured state space duality*) | Pemodelan bahasa | Bukan PHM |
| 6 | **Disertasi ini** | **Mamba + xLSTM + Gated Fusion** | **RUL *bearing* (PHM 2012, XJTU-SY)** | **Mengisi celah kombinasi Mamba–xLSTM untuk *bearing* RUL** |

Dengan demikian, kontribusi utama penelitian ini terletak pada pengembangan arsitektur hibrid Mamba-xLSTM-Net yang memadukan dua kekuatan komplementer: kemampuan *exponential gating* dan *matrix memory* dari xLSTM untuk menangkap dinamika lokal degradasi *bearing*, serta kemampuan *linear-time selective scanning* dari Mamba untuk memodelkan tren global sepanjang ribuan *time step*. Arsitektur usulan ini diposisikan sebagai penyempurnaan dari *baseline* xLSTM–Transformer (Liu dkk., 2025) dengan mengganti cabang Transformer berkompleksitas kuadratik menjadi cabang Mamba berkompleksitas linear.

## III.3   Formulasi Masalah Prediksi *Remaining Useful Life*

### III.3.1   Definisi Masalah

Diberikan sinyal vibrasi mentah dari suatu *bearing* yang direkam secara periodik selama masa pakainya hingga mengalami kegagalan (*run to failure*). Pada setiap periode akuisisi t ∈ {1, 2, ..., T}, direkam sinyal vibrasi horizontal dan vertikal dengan panjang fixed window pada frekuensi sampling yang ditentukan oleh dataset. Tujuan prediksi RUL adalah menghasilkan fungsi f_θ (dengan parameter θ) sehingga untuk setiap titik waktu t, model memberikan prediksi RUL ternormalisasi ŷ_t ∈ [0, 1] yang merepresentasikan fraksi sisa masa pakai komponen:

> **ŷ_t = f_θ(x_{t-L+1:t})**   ... (III.1)

dengan x_{t-L+1:t} adalah jendela fitur *health indicator* (HI) sepanjang L *time step*, dan L adalah panjang window input yang pada penelitian ini ditetapkan L = 10 mengikuti konfigurasi Liu dkk. (2025).

Nilai ŷ_t = 1 menyatakan kondisi *bearing* baru, sedangkan ŷ_t = 0 menyatakan kondisi gagal. Masalah ini dirumuskan sebagai tugas regresi deret waktu *one-step* dengan target skalar.

### III.3.2   Konstruksi *Health Indicator*

Sinyal vibrasi mentah tidak langsung digunakan sebagai input model, melainkan terlebih dahulu dikonversi menjadi *health indicator* (HI) melalui pipeline feature extraction yang mengikuti praktik terbaik di bidang PHM. Pada penelitian ini digunakan dua pipeline HI yang berbeda tergantung karakteristik dataset:

1. **Pipeline statistik multi-domain** (untuk dataset dengan variasi fault mode kaya seperti PHM 2012): ekstraksi fitur domain waktu (*root mean square*, kurtosis, skewness, *crest factor*, *peak-to-peak*, *shape factor*, *impulse factor*, *margin factor*) dan fitur domain frekuensi (*spectral centroid*, *spectral entropy*, *mean frequency*, *root mean square frequency*, dan energi pada lima pita frekuensi karakteristik). Hasil ekstraksi menghasilkan vektor fitur 34-dimensi per akuisisi per sumbu, sehingga total 68 fitur untuk dua sumbu.
2. **Pipeline Liu *isomap* HI** (untuk replikasi setup Liu dkk., 2025): konstruksi HI satu dimensi menggunakan *isometric mapping* (Isomap) yang memetakan fitur vibrasi berdimensi tinggi ke manifold satu dimensi yang monoton terhadap progres degradasi. Pipeline ini diberi label `liu2026_phm` dan `liu2026_xjtu` pada konfigurasi eksperimen.

Kedua pipeline kemudian dilakukan penghalusan eksponensial (*exponential smoothing*) dengan faktor α = 0.1 untuk menekan *noise* akuisisi, serta normalisasi Min-Max ke rentang [0, 1] berdasarkan statistik dataset pelatihan.

### III.3.3   Skema Pelabelan RUL

Penelitian ini menggunakan dua skema pelabelan yang keduanya dipertimbangkan dalam eksperimen:

1. **Skema linier** (*linear label scheme*): target RUL diturunkan secara linier dari 1 pada *time step* awal hingga 0 pada *time step* akhir (*end of life*). Skema ini sederhana namun memiliki kelemahan karena memaksa model belajar degradasi yang terjadi sejak awal masa pakai, padahal pada kenyataannya *bearing* berada pada kondisi sehat untuk periode yang panjang sebelum terjadi degradasi.

2. **Skema *piecewise* Liu 2026** (*piecewise_liu2026*): target RUL dipertahankan konstan pada nilai 1 hingga titik *degradation onset* t_d, kemudian menurun secara linier hingga 0 pada *end of life* t_f:

> **y_t = 1**, untuk t < t_d                                 ... (III.2a)
>
> **y_t = (t_f − t) / (t_f − t_d)**, untuk t_d ≤ t ≤ t_f     ... (III.2b)

Skema *piecewise* lebih realistis secara fisik karena *bearing* memang memiliki fase sehat yang panjang sebelum mengalami degradasi. Titik *degradation onset* t_d ditentukan secara otomatis melalui deteksi perubahan signifikan pada HI menggunakan metode *3-sigma* pada jendela sliding sebagaimana diusulkan Liu dkk. (2025). Pada penelitian ini, skema *piecewise_liu2026* dipilih sebagai skema utama karena menghasilkan prediksi yang lebih bermakna secara fisik.

## III.4   Arsitektur *Baseline*: xLSTM–Transformer

Arsitektur *baseline* yang direplikasi pada penelitian ini mengikuti proposal Liu dkk. (2025). Arsitektur ini merupakan *encoder–decoder* yang menggabungkan kekuatan *Multi-Head Attention* Transformer untuk pengkodean konteks global dengan kekuatan xLSTM untuk pemodelan dinamika temporal lokal. Diagram arsitektur penuh disajikan pada Gambar III.3.

### III.4.1   Komponen *scalar* LSTM (sLSTM)

Blok sLSTM (Beck dkk., 2024) memperluas LSTM klasik dengan dua modifikasi kunci. Modifikasi pertama adalah penggantian *sigmoid gating* dengan *exponential gating*, yang memungkinkan nilai gate melampaui rentang [0, 1] sehingga model dapat merevisi memori sel secara lebih agresif ketika dibutuhkan. Modifikasi kedua adalah stabilisasi numerik melalui *normalizer state* yang mencegah *overflow* akibat *exponential gating*. Persamaan sLSTM pada waktu t adalah sebagai berikut:

> **z_t = tanh(W_z · x_t + R_z · h_{t-1} + b_z)**                   ... (III.3a)
>
> **i_t = exp(W_i · x_t + R_i · h_{t-1} + b_i)**                    ... (III.3b)
>
> **f_t = exp(W_f · x_t + R_f · h_{t-1} + b_f)** atau σ(...)        ... (III.3c)
>
> **o_t = σ(W_o · x_t + R_o · h_{t-1} + b_o)**                      ... (III.3d)
>
> **c_t = f_t ⊙ c_{t-1} + i_t ⊙ z_t**                               ... (III.3e)
>
> **n_t = f_t ⊙ n_{t-1} + i_t**    (*normalizer state*)             ... (III.3f)
>
> **h_t = o_t ⊙ (c_t / n_t)**                                       ... (III.3g)

dengan x_t input pada waktu t, h_t *hidden state*, c_t *cell state*, n_t *normalizer state*, ⊙ perkalian *element-wise*, σ fungsi *sigmoid*, serta W_*, R_*, dan b_* masing-masing matriks bobot input, matriks bobot rekurens, dan bias. *Exponential gating* pada i_t dan f_t menjadi kunci kapabilitas sLSTM untuk menangani saturasi pada fase akhir masa pakai *bearing*.

### III.4.2   Komponen *matrix* LSTM (mLSTM)

Blok mLSTM (Beck dkk., 2024) memperluas kapasitas memori LSTM dari vektor ke matriks, terinspirasi dari struktur memori asosiatif. Memori sel C_t ∈ ℝ^{d×d} menyimpan pasangan *key–value* yang memungkinkan model mengingat banyak asosiasi sekaligus. Persamaan mLSTM adalah sebagai berikut:

> **q_t = W_q · x_t + b_q**,   **k_t = W_k · x_t + b_k**,   **v_t = W_v · x_t + b_v**    ... (III.4a–c)
>
> **i_t = exp(W_i · x_t + b_i)**,   **f_t = exp(W_f · x_t + b_f)**   ... (III.4d,e)
>
> **o_t = σ(W_o · x_t + b_o)**                                       ... (III.4f)
>
> **C_t = f_t · C_{t-1} + i_t · (v_t · k_t^⊤ / √d)**                 ... (III.4g)
>
> **n_t = f_t · n_{t-1} + i_t · k_t**                                ... (III.4h)
>
> **h_t = o_t ⊙ (C_t · q_t) / max(|n_t^⊤ · q_t|, 1)**                ... (III.4i)

Dibandingkan sLSTM, mLSTM tidak memiliki koneksi rekurens pada gate (hanya bergantung pada x_t), sehingga lebih mudah diparalelkan. Kapasitas memori matriks C_t memberikan kemampuan representasi yang jauh lebih kaya, yang menjadi penting ketika fitur vibrasi multi-dimensi perlu diingat sepanjang periode yang panjang.

### III.4.3   *Multi-Head Self-Attention*

Komponen *Multi-Head Self-Attention* (MHSA) pada cabang Transformer mengikuti formulasi Vaswani dkk. (2017). Untuk setiap kepala h ∈ {1, ..., H}, dihitung:

> **Attn_h(Q, K, V) = softmax( (Q · W_h^Q) (K · W_h^K)^⊤ / √d_k ) · (V · W_h^V)**   ... (III.5)

dengan Q, K, V masing-masing *query*, *key*, dan *value* yang diproyeksikan dari input melalui matriks W_h^Q, W_h^K, W_h^V. Output semua kepala kemudian digabungkan dan diproyeksikan:

> **MHSA(Q, K, V) = Concat(Attn_1, ..., Attn_H) · W^O**              ... (III.6)

Pada implementasi *baseline*, digunakan H = 4 kepala dengan d_model = 32, sehingga d_k = 8 per kepala. *Positional encoding* sinusoidal ditambahkan pada input sebelum masuk ke MHSA untuk memberikan informasi urutan.

### III.4.4   Arsitektur Penuh *Encoder–Decoder*

Arsitektur xLSTM–Transformer secara keseluruhan terdiri dari komponen-komponen berikut:

1. **Projeksi input**: lapisan linier F → d_model yang memetakan vektor fitur HI menjadi dimensi model.
2. **Positional encoding**: penambahan *sinusoidal positional encoding* untuk memberi model informasi urutan temporal.
3. **Encoder Transformer**: 1 lapisan *encoder* Transformer dengan MHSA (H = 4) dan *feed-forward network* (FFN) dengan dimensi tersembunyi 64.
4. **Stack xLSTM**: 2 blok mLSTM dengan 1 blok sLSTM pada posisi tengah (konfigurasi `slstm_positions = [1]`).
5. **Decoder dengan cross-attention**: 1 lapisan *decoder* yang melakukan *cross-attention* antara output xLSTM (sebagai *query*) dengan output *encoder* Transformer (sebagai *key* dan *value*).
6. **Regression head**: *Layer Normalization* → Linear(d_model → 32) → GELU → Dropout(0.1) → Linear(32 → 1) → Sigmoid, untuk menghasilkan prediksi RUL skalar pada rentang [0, 1].

Jumlah parameter total arsitektur *baseline* adalah 44,529 untuk konfigurasi PHM 2012 dan 43,441 untuk konfigurasi XJTU-SY. Selisih kecil disebabkan oleh perbedaan jumlah fitur HI antar kedua dataset.

## III.5   Arsitektur Usulan: Mamba-xLSTM-Net

### III.5.1   Motivasi dan Prinsip Desain Hibrid

Arsitektur *baseline* xLSTM–Transformer memiliki dua keterbatasan yang menjadi motivasi desain model usulan:

1. **Kompleksitas kuadratik dari *Multi-Head Self-Attention***. MHSA memiliki kompleksitas O(L² · d) terhadap panjang sekuens. Pada dataset PHM 2012, satu *bearing* dapat menghasilkan lebih dari 2800 akuisisi, sehingga jika model diharapkan memproses konteks yang panjang, kompleksitas kuadratik ini menjadi hambatan nyata baik dari sisi memori maupun waktu komputasi.

2. **Keterbatasan representasi *long-range dependencies***. Walaupun MHSA pada prinsipnya dapat menjangkau konteks panjang, dalam praktik dibatasi oleh *context window* yang dapat ditampung memori. Untuk sinyal degradasi *bearing* yang dinamikanya meliputi fase sehat panjang diikuti fase degradasi pendek, kemampuan menangkap tren global ribuan *time step* menjadi kritikal.

*Selective state space model* (Mamba) mengatasi kedua keterbatasan tersebut secara simultan: memiliki kompleksitas linear O(L · d) dan secara eksplisit didesain untuk pemodelan *long-range dependencies*. Namun demikian, Mamba tidak memiliki kapabilitas *exponential gating* dan *matrix memory* yang unggul pada dinamika lokal seperti yang dimiliki xLSTM.

Prinsip desain arsitektur usulan adalah **memadukan dua kekuatan komplementer** melalui arsitektur dua cabang (*dual-branch*) dengan mekanisme fusi terpandu (*gated fusion*):

- **Cabang A (xLSTM)** menangkap dinamika lokal degradasi dengan *exponential gating* yang mampu merespons perubahan tajam pada fase akhir masa pakai.
- **Cabang B (*Bidirectional Mamba*)** menangkap tren global sepanjang ribuan *time step* dengan kompleksitas linear, serta *bidirectional scan* yang memungkinkan setiap *time step* mengakses konteks masa lalu dan masa depan secara simultan.
- **Fusi gated** memungkinkan model belajar secara adaptif kapan mengandalkan sinyal lokal (cabang A) dan kapan mengandalkan tren global (cabang B), yang diharapkan bervariasi sepanjang masa pakai *bearing*.

### III.5.2   *Selective State Space Model* (Mamba)

*State space model* (SSM) dalam bentuk kontinu didefinisikan oleh persamaan diferensial:

> **dh(t)/dt = A · h(t) + B · x(t)**                                 ... (III.7a)
>
> **y(t) = C · h(t)**                                                ... (III.7b)

dengan h(t) *hidden state*, x(t) input, y(t) output, serta A, B, C matriks parameter. Untuk implementasi pada sinyal diskrit, dilakukan diskritisasi menggunakan *zero-order hold* dengan parameter langkah waktu Δ:

> **Ā = exp(Δ · A)**                                                 ... (III.8a)
>
> **B̄ = (Δ · A)^{−1} · (exp(Δ · A) − I) · Δ · B**                   ... (III.8b)
>
> **h_t = Ā · h_{t-1} + B̄ · x_t**                                    ... (III.8c)
>
> **y_t = C · h_t**                                                  ... (III.8d)

Kontribusi utama Mamba (Gu & Dao, 2023) adalah menjadikan parameter Δ, B, dan C sebagai fungsi dari input x_t (*input-dependent*, atau dalam istilah lain *selective*):

> **B_t = Linear_B(x_t)**, **C_t = Linear_C(x_t)**, **Δ_t = softplus(Linear_Δ(x_t))**   ... (III.9)

Dengan demikian, model dapat secara selektif menentukan informasi mana yang disimpan dalam *hidden state* dan mana yang diabaikan, bergantung pada input aktual. Kombinasi selektivitas ini dengan algoritma *parallel scan* yang efisien memungkinkan Mamba berjalan pada kompleksitas linear dengan tetap mempertahankan ekspresivitas yang sebanding dengan Transformer.

### III.5.3   Blok *Bidirectional Mamba*

Pada penelitian ini, Mamba digunakan dalam konfigurasi bidireksional (*bidirectional Mamba*, BiMamba) untuk memperkaya representasi dengan konteks dua arah. Persamaan BiMamba:

> **h^→_t = Mamba_forward(x_{1:t})**                                  ... (III.10a)
>
> **h^←_t = Mamba_backward(x_{t:L})**                                 ... (III.10b)
>
> **h^BiMamba_t = Linear([h^→_t ; h^←_t])**                           ... (III.10c)

dengan [ ; ] operasi konkatenasi, dan Linear adalah lapisan linier yang memproyeksikan hasil konkatenasi berukuran 2d_model kembali ke d_model. Konfigurasi eksperimen menggunakan 2 blok BiMamba dengan `d_state = 64`, `d_conv = 4`, dan `expand = 2`, mengikuti *default* pustaka `mamba-ssm`.

### III.5.4   Mekanisme *Gated Fusion*

Output cabang xLSTM dan cabang BiMamba digabungkan melalui mekanisme *gated fusion* yang secara adaptif menentukan bobot kontribusi masing-masing cabang pada setiap *time step*:

> **g_t = σ( W_g · [h^xLSTM_t ; h^BiMamba_t] + b_g )**                ... (III.11a)
>
> **h^fused_t = g_t ⊙ h^xLSTM_t + (1 − g_t) ⊙ h^BiMamba_t**           ... (III.11b)

dengan g_t ∈ [0, 1]^{d_model} *gate* yang dipelajari, σ fungsi sigmoid, serta W_g dan b_g parameter yang dapat dilatih. Secara intuitif, nilai g_t mendekati 1 berarti model lebih mengandalkan representasi lokal dari xLSTM, sedangkan g_t mendekati 0 berarti lebih mengandalkan representasi global dari BiMamba. Gate ini dapat divariasi per *time step*, memungkinkan model secara dinamis menyesuaikan strategi fusi sepanjang masa pakai *bearing*.

### III.5.5   *Regression Head*

*Regression head* pada arsitektur usulan mengikuti desain yang kompatibel dengan *baseline* untuk memastikan perbandingan yang adil:

> **ẑ_t = LayerNorm(h^fused_t)**                                     ... (III.12a)
>
> **ẑ_t = GELU(Linear(d_model → 64)(ẑ_t))**                          ... (III.12b)
>
> **ẑ_t = Dropout(0.1)(ẑ_t)**                                         ... (III.12c)
>
> **ŷ_t = Sigmoid(Linear(64 → 1)(ẑ_t))**                             ... (III.12d)

Dropout ditempatkan setelah GELU untuk regularisasi.

### III.5.6   Arsitektur Penuh dan Jumlah Parameter

Arsitektur Mamba-xLSTM-Net secara keseluruhan disajikan pada Gambar III.4. Konfigurasi yang digunakan dalam eksperimen adalah: d_model = 128, 3 blok xLSTM dengan sLSTM pada posisi 1, 2 blok BiMamba, `head_hidden = 64`, dan `dropout = 0.1`. Jumlah parameter total adalah 955,921 untuk konfigurasi PHM 2012 dan 811,713 untuk konfigurasi XJTU-SY. Peningkatan jumlah parameter dibandingkan *baseline* (sekitar 20×) merupakan konsekuensi logis dari peningkatan d_model dari 32 menjadi 128 serta kedua cabang paralel. Implikasi *overparameterization* ini akan dianalisis pada Bab V.

**Tabel III-10.** Perbandingan parameter dan kompleksitas kedua arsitektur

| Aspek | *Baseline* xLSTM–Transformer | Mamba-xLSTM-Net (usulan) |
| :--- | :--- | :--- |
| d_model | 32 | 128 |
| Blok xLSTM | 2 mLSTM + 1 sLSTM | 2 mLSTM + 1 sLSTM |
| Blok Transformer / Mamba | 1 encoder + 1 decoder | 2 BiMamba |
| Mekanisme fusi | Cross-attention | Gated fusion |
| Kompleksitas attention / scan | O(L² · d) | O(L · d) |
| Parameter (PHM 2012) | 44,529 | 955,921 |
| Parameter (XJTU-SY) | 43,441 | 811,713 |

## III.6   Protokol Pelatihan

Untuk memastikan perbandingan yang adil antara kedua arsitektur, protokol pelatihan dibuat identik sejauh memungkinkan. Pengecualian ada pada parameter arsitektural (d_model, jumlah blok) yang secara inheren berbeda.

### III.6.1   Fungsi Loss

Fungsi *loss* yang digunakan adalah *Mean Squared Error* (MSE) antara prediksi RUL dan target:

> **L_MSE(θ) = (1 / N) · Σ_{i=1}^{N} ( y_i − ŷ_i(θ) )²**              ... (III.13)

dengan N jumlah sampel pelatihan. MSE dipilih karena kompatibel dengan output sigmoid pada *regression head*, serta memberikan penalti lebih besar pada kesalahan besar, yang sesuai untuk prediksi RUL di mana kesalahan besar di dekat *end of life* memiliki konsekuensi fatal.

### III.6.2   *Optimizer* dan *Hyperparameter* Pelatihan

Kedua arsitektur dilatih menggunakan optimizer *Adam* (Kingma & Ba, 2015) dengan konfigurasi berikut:

- *Learning rate*: 1 × 10⁻³
- *Weight decay*: 0
- *Scheduler*: tidak digunakan (konstan sepanjang pelatihan)
- *Gradient clipping*: norma maksimum 1.0
- *Early stopping patience*: 9999 (efektif tidak aktif; model dilatih penuh 50 *epoch*)
- *Monotonicity weight*: 0 (tidak menggunakan penalti monotonisitas tambahan)
- *Precision*: FP32
- *Batch size*: 32
- *Max epochs*: 50

Seed acak ditetapkan pada 42 untuk memastikan reprodusibilitas eksperimen. *Checkpoint* terbaik dipilih berdasarkan metrik `train/loss` untuk kedua arsitektur pada Mamba-xLSTM-Net agar konsisten, sesuai konfigurasi YAML yang dilampirkan pada Lampiran B.

### III.6.3   Perangkat Keras dan Perangkat Lunak

Seluruh eksperimen dijalankan pada satu GPU NVIDIA dengan dukungan CUDA untuk memastikan kompatibilitas dengan pustaka `mamba-ssm` yang memerlukan kernel CUDA terkustom. Kerangka kerja pembelajaran mendalam yang digunakan adalah PyTorch 2.1 dengan PyTorch Lightning 2.1 untuk orkestrasi pelatihan, serta Hydra 1.3 untuk manajemen konfigurasi. Pustaka utama lainnya meliputi `xlstm` 1.0 (NX-AI), `mamba-ssm` 2.2, dan `causal-conv1d` 1.2.

## III.7   Metrik Evaluasi

Empat metrik utama digunakan untuk mengevaluasi dan membandingkan model, masing-masing menyorot aspek kinerja yang berbeda.

### III.7.1   *Root Mean Squared Error* (RMSE)

Metrik regresi utama yang mengukur deviasi rata-rata prediksi dari nilai sebenarnya dalam satuan yang sama dengan target:

> **RMSE = √( (1 / N) · Σ_{i=1}^{N} ( y_i − ŷ_i )² )**                ... (III.14)

Nilai lebih kecil menunjukkan kinerja lebih baik. RMSE sensitif terhadap kesalahan besar dan menjadi metrik standar pada prediksi RUL.

### III.7.2   *Mean Absolute Error* (MAE)

Metrik sekunder yang lebih *robust* terhadap *outlier*:

> **MAE = (1 / N) · Σ_{i=1}^{N} | y_i − ŷ_i |**                       ... (III.15)

### III.7.3   Koefisien Determinasi (R²)

Metrik *goodness-of-fit* yang mengukur fraksi varians data yang dijelaskan oleh model:

> **R² = 1 − ( Σ ( y_i − ŷ_i )² ) / ( Σ ( y_i − ȳ )² )**              ... (III.16)

dengan ȳ rata-rata target. Nilai 1 menunjukkan prediksi sempurna, 0 menunjukkan kinerja setara dengan menebak rata-rata, dan nilai negatif menunjukkan kinerja lebih buruk daripada menebak rata-rata. Perlu dicatat bahwa pada skema *piecewise_liu2026*, target RUL sebagian besar bernilai 1 (fase sehat panjang), sehingga varians target kecil dan R² menjadi metrik yang kurang informatif pada konteks ini. Analisis lebih lanjut diberikan pada Bab V.

### III.7.4   PHM Score

*Scoring function* standar kompetisi IEEE PHM 2012 (Nectoux dkk., 2012) yang memberikan penalti asimetris: prediksi yang terlalu dini (optimistik, RUL_pred > RUL_true) diberi penalti lebih ringan, sedangkan prediksi yang terlambat (pesimistik, RUL_pred < RUL_true) diberi penalti lebih berat, karena secara operasional lebih aman memperkirakan *bearing* akan gagal lebih cepat daripada lebih lama:

> **Er_i = 100 · ( RUL_actual_i − RUL_pred_i ) / RUL_actual_i**       ... (III.17a)
>
> **A_i = exp( −ln(0.5) · (Er_i / 5) )**, jika Er_i ≤ 0              ... (III.17b)
>
> **A_i = exp( +ln(0.5) · (Er_i / 20) )**, jika Er_i > 0             ... (III.17c)
>
> **PHM Score = (1 / N) · Σ_{i=1}^{N} A_i**                           ... (III.17d)

Nilai lebih besar menunjukkan kinerja lebih baik, dengan nilai maksimum 1.0. Pada penelitian ini dilaporkan dua varian PHM Score: (1) `phm_score` versi proyek yang dihitung pada RUL ternormalisasi, dan (2) `phm_score_paper` yang dihitung sesuai protokol Liu dkk. (2025) dalam satuan waktu fisik. Keduanya dilaporkan untuk transparansi.

### III.7.5   Metrik Tambahan per-*Bearing*

Selain metrik agregat, dilaporkan pula metrik per-*bearing* untuk mengevaluasi konsistensi kinerja model di berbagai kondisi operasi. Metrik per-*bearing* meliputi `rmse_per_bearing`, `phm_per_bearing`, dan `phm_paper_per_bearing`.

## III.8   Rancangan Eksperimen

Eksperimen disusun dalam empat studi kasus yang diurutkan secara inkremental dari validasi metode pada dataset publik hingga rencana transfer ke objek penelitian SKF Indonesia.

### III.8.1   Studi Kasus I: Replikasi *Baseline* pada Dataset Publik

**Tujuan**: memverifikasi kebenaran implementasi *baseline* xLSTM–Transformer melalui replikasi hasil Liu dkk. (2025) pada dataset PHM 2012 dan XJTU-SY.

**Tahapan**:
1. Unduh dan pra-pemrosesan dataset sesuai pipeline yang dijelaskan pada Bab IV.
2. Implementasi arsitektur sesuai subbab III.4.
3. Pelatihan model dengan konfigurasi pada subbab III.6.
4. Evaluasi pada *bearing* uji yang ditetapkan dengan metrik pada subbab III.7.
5. Perbandingan hasil dengan angka yang dilaporkan Liu dkk. (2025).

**Kriteria sukses**: RMSE pada *bearing* uji berada dalam rentang ±10% dari angka yang dilaporkan Liu dkk. (2025) pada kedua dataset.

### III.8.2   Studi Kasus II: Evaluasi Arsitektur Usulan Mamba-xLSTM-Net

**Tujuan**: membandingkan kinerja arsitektur usulan terhadap *baseline* pada kedua dataset publik dengan protokol pelatihan dan evaluasi yang identik.

**Tahapan**:
1. Implementasi arsitektur Mamba-xLSTM-Net sesuai subbab III.5.
2. Pelatihan dengan konfigurasi yang identik dengan Studi Kasus I, *seed* 42.
3. Evaluasi dengan metrik yang sama.
4. Perbandingan *head-to-head* pada setiap metrik dan setiap *bearing* uji.
5. Analisis statistik (uji *Wilcoxon signed-rank* pada RMSE per-*bearing*) untuk menilai signifikansi perbedaan kinerja.

**Kriteria sukses**: Mamba-xLSTM-Net menunjukkan peningkatan yang signifikan pada minimal dua dari empat metrik utama pada setidaknya satu dataset, dengan sinyal yang konsisten pada *bearing* uji yang sekuensnya panjang.

### III.8.3   Studi Kasus III: Ablasi dan Analisis Interpretabilitas

**Tujuan**: mengisolasi kontribusi masing-masing komponen arsitektur usulan serta memberikan interpretasi mekanistik terhadap fitur degradasi yang dipelajari model.

Konfigurasi ablasi yang akan dijalankan dirangkum pada Tabel III-11.

**Tabel III-11.** Konfigurasi studi ablasi

| ID | Konfigurasi | Komponen yang dihapus/diganti | Hipotesis |
| :--- | :--- | :--- | :--- |
| A1 | xLSTM-only | Hilangkan cabang BiMamba | Menguji kontribusi cabang Mamba |
| A2 | Mamba-only | Hilangkan cabang xLSTM | Menguji kontribusi cabang xLSTM |
| A3 | *Unidirectional* Mamba | BiMamba → Mamba satu arah | Menguji kontribusi bidireksionalitas |
| A4 | *Concat fusion* | *Gated fusion* → konkatenasi sederhana | Menguji kontribusi *gating* |
| A5 | LSTM-biasa sebagai pengganti xLSTM | xLSTM → LSTM standar | Menguji kontribusi *exponential gating* |
| A6 | Tanpa *exponential smoothing* HI | α = 0 (tanpa pemulusan) | Menguji kontribusi pra-pemrosesan |
| A7 | Window pendek (L = 5) vs panjang (L = 50) | Variasi panjang *window* input | Menguji kebutuhan konteks panjang |

Untuk analisis interpretabilitas, akan diterapkan tiga metode komplementer:

1. **SHAP (SHapley Additive exPlanations)**: melalui `shap.GradientExplainer` pada Mamba-xLSTM-Net terlatih, untuk mengidentifikasi fitur HI yang paling berpengaruh pada prediksi RUL secara global dan per-*bearing*.
2. **Sparse Autoencoder (SAE)**: dilatih pada representasi tersembunyi lapisan fusi untuk menemukan fitur-fitur laten yang dapat diinterpretasi. Setiap laten SAE kemudian dipetakan ke pola degradasi khas (misalnya peningkatan RMS tajam, lonjakan impuls, saturasi fase akhir).
3. **Integrated Gradients**: untuk atribusi temporal, yaitu mengidentifikasi *time step* dalam *window* input yang paling berpengaruh terhadap prediksi pada titik tertentu.

Analisis ini bertujuan memberikan kepercayaan mekanistik pada model usulan, yang merupakan salah satu poin lemah arsitektur *deep learning* pada aplikasi industri.

### III.8.4   Studi Kasus IV: Rancangan Transfer ke Objek SKF Indonesia (Rencana Tahap Akhir)

**Tujuan**: mengadaptasi model Mamba-xLSTM-Net yang telah divalidasi pada dataset publik untuk diterapkan pada data vibrasi mesin *grinding* OR1 dan OR2 di Channel 15 PT SKF Indonesia, dengan integrasi data kualitas produk (*vibration checking* dan *radial clearance checking*) sebagai fitur tambahan.

Rancangan transfer akan dibahas secara ringkas pada bagian akhir Bab V sebagai hasil antara dan rencana kerja lanjutan, dengan tahapan sebagai berikut:

1. Pemetaan fitur HI dataset publik ke format data SKF (menggunakan *exporter* XLSX dari SKF Observer).
2. Konstruksi HI menggunakan pipeline serupa `liu2026_phm` yang telah divalidasi.
3. *Fine-tuning* model Mamba-xLSTM-Net yang telah dilatih pada dataset publik, dengan pendekatan *transfer learning*.
4. Integrasi data kualitas produk sebagai *auxiliary input* melalui *feature-level fusion*.
5. Evaluasi pada data OR1 dan OR2, dengan validasi silang antar mesin.

Kerangka rancangan detail untuk Studi Kasus IV akan dikembangkan lebih lanjut pada iterasi disertasi berikutnya, sesuai skema pendekatan penelitian pada Gambar I.14.

## III.9   Diagram Alur Penelitian

Gambar III.5 menyajikan diagram alur keseluruhan penelitian, mulai dari identifikasi masalah dan kajian literatur, pengumpulan dan pra-pemrosesan data, implementasi model *baseline* dan usulan, eksperimen komparatif pada dataset publik, analisis interpretabilitas, hingga rencana transfer ke objek penelitian SKF Indonesia. Diagram ini menunjukkan dengan jelas bahwa penelitian pada tahap ini berfokus pada validasi metode (Studi Kasus I–III) pada dataset publik, dengan penerapan pada objek nyata sebagai tahap akhir.

[Placeholder untuk Gambar III.5 — akan dibuat sebagai visualisasi terpisah]
