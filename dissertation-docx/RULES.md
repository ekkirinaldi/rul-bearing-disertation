# RULES.md — Aturan Penulisan Disertasi dalam Format DOCX

Dokumen DOCX di folder ini adalah **naskah utama** disertasi. Naskah disunting
langsung di Word; tidak ada format sumber untuk membangunnya kembali, karena
pohon LaTeX asal porting sudah dihapus pada September 2026 dan hanya dapat
diambil kembali dari riwayat git. Semua suntingan naskah dilakukan pada file
`.docx` di sini, mengikuti aturan dalam dokumen ini. Aturan format bersumber dari *Pedoman Penulisan Disertasi Doktor
ITB* (SPs, April 2016) dan *Juknis Disertasi Mei 2019*; template resmi:
`assets/template.docx` (salinan `disertasi/template-disertasi_Mei2019.docx`).

---

## 1. Peta Gaya (Word Styles)

Gunakan **hanya** gaya berikut. Jangan membuat gaya baru atau memformat manual
(font/ukuran/spasi langsung pada teks).

| Elemen | Style Word | Catatan |
|---|---|---|
| Judul bab ("Bab VI Kesimpulan…") | **Heading 1** | Penomoran otomatis `Bab %1` (Roman) via numbering numId 2. **BUKAN** style `JudulBab` (tidak membawa penomoran). |
| Judul subbab ("VI.1 …") | **Heading 2** | `%1.%2`, rata kiri tanpa indentasi. |
| Judul anak subbab ("VI.1.1 …") | **Heading 3** | `%1.%2.%3`. |
| Paragraf isi | **Paragraf** | TNR 12 pt, rata kiri-kanan, 1,5 spasi, tanpa indentasi baris pertama. |
| Caption gambar | **JudulGambar** | Di **bawah** gambar. |
| Caption tabel | **judulTabel** | Di **atas** tabel. |
| Isi sel tabel | **IsiTabel** | TNR 12 pt, spasi tunggal (gaya tambahan pipeline). |
| Paragraf berisi gambar | **Gambar** | Rata tengah (gaya tambahan pipeline). |
| Entri daftar pustaka | **Daftarpustaka** | Hanging indent 1,27 cm, spasi tunggal. |
| Judul lampiran | **Lampiran** / **Lampiransub1** | Penomoran **literal** dalam teks ("Lampiran A Judul", "A.1 Judul") — gaya template tidak membawa penomoran otomatis. Caption di lampiran memakai prefiks literal "A." + field `SEQ GambarLampA` (sekuens terpisah per lampiran agar aman terhadap F9 setelah merge). |

**Pemisah paragraf:** satu paragraf kosong ber-style `Paragraf` di antara dua
blok isi (konvensi dokumen V5; ekuivalen "satu baris kosong 1,5 spasi" pada
Pedoman). Tidak ada paragraf kosong antara paragraf-gambar dan caption-nya
(`Gambar` → `JudulGambar`). **Antara caption tabel dan tabelnya wajib ada satu
paragraf kosong** (`judulTabel` → `Paragraf` kosong → tabel), sesuai Juknis
Mei 2019.

## 2. Penomoran Bab dan Field

- Penomoran bab/subbab adalah **otomatis** dari Heading 1–3 (numbering numId 2
  template). File per-bab memakai `lvlOverride startOverride` agar bab
  berdiri sendiri menampilkan nomor babnya yang benar.
- **Nomor halaman** per-bab diset via `pgNumType start` dari nilai `.aux`
  LaTeX; setelah merge final menjadi penomoran kontinu.

## 3. Caption Gambar/Tabel (resep field)

Format caption: `Gambar VI.2 Judul gambar` / `Tabel VI.1 Judul tabel`
(sentence case, tanpa titik akhir). Struktur internal:

```
run("Gambar ") + bookmarkStart + run("VI.") + { SEQ Gambar \* ARABIC \s 1 } + bookmarkEnd + run(" ") + judul
```

- **Prefiks bab Roman ditulis literal** ("VI."), bukan field STYLEREF —
  LibreOffice merusak STYLEREF, dan angka bab tidak berubah selama bab tidak
  ditata ulang. Jika urutan bab berubah, prefiks literal harus diganti manual
  (cari-ganti per bab).
- Field SEQ menyimpan **cached result** sehingga dokumen tampil benar tanpa
  F9; F9 di Word tetap aman (SEQ menghitung ulang dalam bab).
- Bookmark = label LaTeX asal yang disanitasi (`fig:bab6_framework` →
  `fig_bab6_framework`; non-alfanumerik → `_`, maks 40 karakter). Caption
  baru: tambahkan bookmark serupa agar bisa dirujuk.

## 4. Rujukan Silang

- **Gambar/Tabel/Persamaan:** field `REF nama_bookmark \h` dengan cached
  result. Di Word: Insert → Cross-reference. Jangan pernah mengetik nomor
  literal "Gambar VI.2" di luar field.
- **Bab/subbab/lampiran:** nomor **literal** dalam teks ("Bab III",
  "subbab V.2") — struktur bab dianggap stabil. Bila menambah/menghapus
  subbab, periksa rujukan literal dengan cari-ganti.
- Kapitalisasi wajib: `Gambar`, `Tabel`, `Bab`, `Lampiran`, `Persamaan`
  selalu kapital bila diikuti nomor.

## 5. Persamaan

- Persamaan matematis: Word equation (OMML), satu baris sendiri, rata tengah
  via tab-stop tengah; nomor `(VI.1)` rata kanan via tab-stop kanan, dengan
  struktur `(` + literal Roman + `.` + field `SEQ Persamaan` + `)` yang
  di-bookmark.
- **Skalar sederhana bukan equation.** Ekspresi seperti *r* = 0,507 ditulis
  sebagai teks biasa (huruf variabel miring) — koma desimal di dalam OMML
  dirender dengan spasi ("0 , 507"). Equation hanya untuk rumus sungguhan.

## 6. Sitasi dan Daftar Pustaka

- Sitasi adalah **teks jadi** (baked) hasil pandoc citeproc + CSL resmi ITB-SPs
  (`assets/itb-sps.csl`, locale `id-ID`): format `(Penulis, tahun)`,
  ≥3 penulis → `dkk.` (bukan `et al.`), penghubung `dan` (bukan `and`/`&`).
- **Sitasi baru ditulis tangan mengikuti format yang sama**, dan entri
  Daftar Pustaka disisipkan langsung pada posisi alfabetisnya mengikuti
  format ITB:
  `Nama, I. (tahun): Judul kalimat, *Nama Jurnal*, **volume**, hal–hal.`
- Daftar Pustaka tunggal di akhir dokumen master (style `Daftarpustaka`),
  alfabetis, tanpa nomor; setiap entri harus dirujuk di badan teks dan
  sebaliknya.

## 7. Angka, Satuan, Tabel, Gambar

- **Desimal pakai koma** (`25,5`); **ribuan pakai titik** (`1.024`). Hindari
  desimal 3 digit agar tidak ambigu dengan pemisah ribuan.
- Bilangan < 10 ditulis huruf (`enam model`); ≥ 10 pakai angka.
- Jangan memulai kalimat dengan angka, simbol, atau rumus.
- Tabel: layout **fixed** dengan lebar kolom eksplisit (jangan autofit);
  caption di atas; header diulang antar-halaman (`tblHeader`) untuk tabel
  panjang; **grid border ITB** (`tblBorders`: `single`, sz=4, color=auto pada
  semua sisi) diterapkan otomatis oleh `restyle.py` (kecuali tabel layout
  gambar berdampingan). Setiap tabel/gambar **wajib dirujuk** dalam teks.
- Gambar: lebar mengikuti `assets/figure-map.tsv` (fraksi dari lebar teks
  ±14 cm); resolusi raster minimal 300 dpi.

## 8. Aturan Prosa (wajib — ditegakkan `tools/lint_docx.sh`)

Diturunkan utuh dari aturan ITB di `docs/writing-guide.md`; ringkasan operasional:

**Bahasa:**
- Bahasa Indonesia baku (KBBI/PUEBI). Tanpa kata ganti orang pertama
  (`saya`, `kami`, `kita`) — gunakan kalimat pasif (kecuali Kata Pengantar).
- Tanpa `di mana` sebagai kata ganti relatif; gunakan `yang`/`tempat`/
  `pada saat`.
- Jangan memulai kalimat dengan `maka`, `sedangkan`, `sehingga`.
- `dkk.` bukan `et al.`; `dan` bukan `&`.
- Istilah asing perlu dimiringkan (*deep learning*, *envelope spectrum*).
  Istilah domain bearing tetap bahasa Inggris: `bearing`, `rolling element`
  (bukan `bantalan`/`gelinding`).
- Tanpa kata hubung asing (`vs`, `via`, `i.e.`, `e.g.`, `etc.`, `cf.`) —
  gunakan padanan Indonesia.
- Ejaan baku: `objek`, `analisis`, `aktivitas`, `praktik`, `risiko`,
  `frekuensi`, `sistem`, `manajemen`, `hipotesis`, dst.

**Gaya prosa organik (anti-ciri tulisan mesin):**
- **Tanpa em dash (—)** dalam prosa: ganti koma/titik koma/titik dua, atau
  pecah kalimat.
- Frasa terlarang: `tidak dapat dipungkiri`, `memainkan peran penting`,
  `penting untuk dicatat`, `perlu dicatat bahwa`, `dalam era modern/digital`,
  `tentunya`, `secara komprehensif/holistik`, `merupakan hal yang (sangat)
  penting/krusial`, `perlu dipahami/diperhatikan bahwa`, `pada
  dasarnya/intinya`, `dapat dikatakan bahwa`, `berbagai macam`,
  `sangat/amat penting`, label `keluarga` untuk kelompok algoritma.
- Variasi pembuka paragraf (jangan dua paragraf berurutan dibuka frasa sama);
  paragraf 3–5 kalimat, satu gagasan utama; selingi kalimat pendek dan
  panjang; angka hasil ditulis spesifik (`12,55 piksel`, bukan "cukup
  besar"); setiap pilihan desain disertai alasan eksplisit.

## 9. Alur Kerja dan Lint

```bash
cd dissertation-docx
make bab6           # regenerasi satu bab dari LaTeX (hanya selama porting)
make verify-bab6    # XML valid + render PDF + REF resolve + volume isi + sitasi
make lint           # lint prosa + cek artefak docx pada semua bab
```

- `tools/lint_docx.sh` mengekstrak teks via pandoc dan memeriksa: token `@@`
  tersisa, field REF tak ter-resolve, artefak perintah LaTeX, caption tanpa
  rujukan, dan seluruh keluarga aturan prosa di §8. **Perbaiki [FATAL]
  sebelum commit; tinjau setiap [WARN].**
- Setelah penyuntingan manual di Word: simpan, jalankan `make lint`, dan
  pastikan pembaruan field (Ctrl+A, F9) tidak mengubah nomor (cached result
  harus sudah benar).
- Catatan known-issue: cek desimal-titik pada lint mengecualikan grup tepat
  3 digit (dianggap pemisah ribuan); desimal-titik 3 digit sungguhan tidak
  terdeteksi otomatis.

## 10. Menyisipkan Materi ke Naskah yang Sudah Disunting Manual

Target `make babN` **meregenerasi** bab dari LaTeX sehingga suntingan manual di
Word hilang. Untuk menambah materi ke naskah yang sudah disunting tangan,
gunakan `tools/insert_docx.py`:

```bash
make insert-dry SPEC=inserts/v15/spec.yaml   # resolusi anchor saja
make insert     SPEC=inserts/v15/spec.yaml   # tulis dokumen keluaran
```

Spesifikasi YAML (lihat `inserts/v15/spec.yaml`) menautkan setiap sisipan pada
paragraf yang sudah ada (`text_prefix`, `heading`, atau `label`) dan menyusun
blok ber-style sesuai aturan di atas: `Paragraf`, `Heading3`, `Gambar` +
`JudulGambar`, `judulTabel` + tabel, serta entri `Daftarpustaka`. Penanda dalam
teks: `*miring*`, `**tebal**`, dan `[ref:label]` untuk field REF.

- Nomor gambar/tabel dihitung ulang setelah penyisipan: cached result pada
  field SEQ, cached result pada setiap field REF, **dan** nomor yang diketik
  literal di dalam teks (naskah V14 memuat 41 rujukan semacam itu).
- Rujukan silang yang cached result-nya kosong dibiarkan apa adanya dan
  dilaporkan, karena mengisinya akan memunculkan teks yang sebelumnya tidak
  tampak.
- Alat menolak menimpa dokumen sumber, dan menolak sisipan yang anchor-nya
  tidak unik.
- Setelah dibuka di Word: Ctrl+A, F9 untuk memperbarui Daftar Isi, Daftar
  Gambar, dan Daftar Tabel.
- Catatan render: LibreOffice mengabaikan sakelar `\s 1` pada field SEQ
  sehingga PDF hasil `soffice --convert-to pdf` menomori gambar secara
  berurutan lintas bab. Word menghormatinya dan menampilkan nomor yang benar.
  Perilaku ini sudah ada pada naskah sebelum penyisipan.
