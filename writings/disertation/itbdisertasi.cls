%% itbdisertasi.cls
%% Document class for ITB Doctoral Dissertation (Disertasi Doktor)
%% Complies with:
%%   - Pedoman Penulisan Disertasi Doktor ITB
%%   - Juknis Penulisan Disertasi (Mei 2019)
%%
%% Engine preferred: LuaLaTeX (UTF-8 native + system fonts)
%% Engine fallback : pdfLaTeX (with newtxtext for Times-like font)

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{itbdisertasi}[2026/05/03 v0.1 ITB Doctoral Dissertation Class]

% =====================================================================
% Options
% =====================================================================
\newif\if@itbdraft       \@itbdraftfalse
\newif\if@itbfinal       \@itbfinaltrue
\newif\if@itblua         \@itbluatrue   % default to LuaLaTeX

\DeclareOption{draft}{\@itbdrafttrue\@itbfinalfalse}
\DeclareOption{final}{\@itbfinaltrue\@itbdraftfalse}
\DeclareOption{pdftex}{\@itbluafalse}
\DeclareOption{lualatex}{\@itbluatrue}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ProcessOptions\relax

% Base class: report, twoside for mirror margins, openright for chapters on odd pages
\LoadClass[12pt,a4paper,twoside,openright]{report}

% Layout tokens (margins, vertical rhythm, sampul gaps) — single file.
% Must live beside this .cls; compile from |writings/disertation/|.
\input{itbdisertasi-layout.tex}

% =====================================================================
% Encoding & Fonts
% =====================================================================
\RequirePackage{iftex}

\ifluatex
  \RequirePackage{fontspec}
  % Times New Roman if available; fallback to TeX Gyre Termes
  \IfFontExistsTF{Times New Roman}{
    \setmainfont{Times New Roman}
  }{
    \setmainfont{TeX Gyre Termes}
  }
\else
  \RequirePackage[utf8]{inputenc}
  \RequirePackage[T1]{fontenc}
  \RequirePackage{newtxtext}
  \RequirePackage{newtxmath}
\fi

% =====================================================================
% Language (Bahasa Indonesia primary, English secondary)
% =====================================================================
\RequirePackage[main=bahasa,english]{babel}

% =====================================================================
% Geometry — ITB mirror margins (tokens: |itbdisertasi-layout.tex|)
%   Odd pages (right-hand): left=4cm, right=3cm, top=3cm, bottom=3cm
%   Even pages (left-hand): right=4cm, left=3cm, others=3cm
% =====================================================================
\RequirePackage{geometry}
\ITBApplyPageGeometry

% =====================================================================
% Line spacing: 1.5 body; 1.0 for blocks
% =====================================================================
\RequirePackage{setspace}
\onehalfspacing  % approximately 1.5

% Slightly more permissive justification to avoid frequent tiny overfull
% boxes in Indonesian text (which has many long compound words).
% This trades a small amount of inter-word spacing variation for cleaner
% margins -- no body text ever bleeds past the printable area.
\emergencystretch=3em
\hbadness=10000
\hfuzz=2pt

% =====================================================================
% Paragraph: no indent, blank line between paragraphs
% =====================================================================
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.6\baselineskip plus 0.1\baselineskip minus 0.1\baselineskip}

% =====================================================================
% Headings — Roman chapter numbering, Indonesian "BAB"
% =====================================================================
\renewcommand{\thechapter}{\Roman{chapter}}
\renewcommand{\thesection}{\thechapter.\arabic{section}}
\renewcommand{\thesubsection}{\thesection.\arabic{subsection}}
\renewcommand{\thesubsubsection}{\thesubsection.\arabic{subsubsection}}

% Counters reset per chapter
\RequirePackage{amsmath}
\numberwithin{equation}{chapter}
\numberwithin{figure}{chapter}
\numberwithin{table}{chapter}
\renewcommand{\theequation}{\thechapter.\arabic{equation}}
\renewcommand{\thefigure}{\thechapter.\arabic{figure}}
\renewcommand{\thetable}{\thechapter.\arabic{table}}

% Chapter title format: "Bab I   Pendahuluan" pada SATU baris, sesuai
% diagram juknis SPs ITB Mei 2019 ("Bab II   Tinjauan Pustaka").
% Spasi vertikal: |\ITBLayoutChapterTitleAbove| dsb.\ di
% |itbdisertasi-layout.tex|.
\RequirePackage[explicit]{titlesec}
\titleformat{\chapter}[block]
  {\normalfont\fontsize{14pt}{17pt}\selectfont\bfseries\centering}
  {\chaptertitlename\ \thechapter}
  {1.5em}
  {#1}
\titlespacing*{\chapter}{0pt}{\ITBLayoutChapterTitleAbove}{\ITBLayoutChapterTitleBelow}

\titleformat{\section}
  {\normalfont\fontsize{12}{14}\bfseries}
  {\thesection}{1em}{#1}
\titlespacing*{\section}{0pt}{\ITBLayoutSectionTitleAbove}{\ITBLayoutSectionTitleBelow}

\titleformat{\subsection}
  {\normalfont\fontsize{12}{14}\bfseries}
  {\thesubsection}{1em}{#1}
\titlespacing*{\subsection}{0pt}{\ITBLayoutSubsectionTitleAbove}{\ITBLayoutSubsectionTitleBelow}

\titleformat{\subsubsection}
  {\normalfont\fontsize{12}{14}\bfseries\itshape}
  {\thesubsubsection}{1em}{#1}
\titlespacing*{\subsubsection}{0pt}{\ITBLayoutSubsubsectionTitleAbove}{\ITBLayoutSubsubsectionTitleBelow}

% Localize chapter name
\addto\captionsbahasa{%
  \renewcommand{\chaptername}{Bab}%
  \renewcommand{\contentsname}{DAFTAR ISI}%
  \renewcommand{\listfigurename}{DAFTAR GAMBAR}%
  \renewcommand{\listtablename}{DAFTAR TABEL}%
  \renewcommand{\bibname}{DAFTAR PUSTAKA}%
  \renewcommand{\appendixname}{LAMPIRAN}%
  \renewcommand{\figurename}{Gambar}%
  \renewcommand{\tablename}{Tabel}%
}

% =====================================================================
% Headers / Footers
% =====================================================================
\RequirePackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0pt}
\fancypagestyle{plain}{%
  \fancyhf{}%
  \fancyfoot[C]{\thepage}%
  \renewcommand{\headrulewidth}{0pt}%
}

% =====================================================================
% Captions — single-spaced, justified with hanging indent
% =====================================================================
\RequirePackage[labelfont=bf,labelsep=period,font=small,justification=justified,singlelinecheck=true]{caption}

% =====================================================================
% Daftar Isi / Daftar Gambar / Daftar Tabel
%   Spasi vertikal: |\ITBApplyTocloftLayout| (|itbdisertasi-layout.tex|).
% =====================================================================
\RequirePackage{tocloft}
\ITBApplyTocloftLayout
% Judul "DAFTAR ISI"/"DAFTAR GAMBAR"/"DAFTAR TABEL" rata-tengah,
% bold, ukuran 14pt.
\renewcommand{\cfttoctitlefont}{\hfill\fontsize{14pt}{17pt}\selectfont\bfseries}
\renewcommand{\cftaftertoctitle}{\hfill}
\renewcommand{\cftloftitlefont}{\hfill\fontsize{14pt}{17pt}\selectfont\bfseries}
\renewcommand{\cftafterloftitle}{\hfill}
\renewcommand{\cftlottitlefont}{\hfill\fontsize{14pt}{17pt}\selectfont\bfseries}
\renewcommand{\cftafterlottitle}{\hfill}

% =====================================================================
% Bibliography (biblatex + biber, Indonesian localization)
% =====================================================================
\RequirePackage[backend=biber,
                style=authoryear,
                citestyle=authoryear,
                maxcitenames=2,
                mincitenames=1,
                maxbibnames=99,
                uniquelist=false,
                uniquename=false,
                dashed=false,
                sorting=nyt,
                giveninits=true,
                useprefix=true,
                natbib=true,
                isbn=false,
                doi=true,
                url=false,
                eprint=false,
                date=year]{biblatex}

% Map babel's `bahasa` (which has no .lbx file in biblatex) to english.lbx
% so that all bibliography string overrides below are actually applied.
% Without this, biblatex emits literal keys like "andothers" because the
% bahasa language file is missing and fallback never triggers correctly.
\DeclareLanguageMapping{bahasa}{english}
\DeclareLanguageMapping{indonesian}{english}

% Indonesian localization for "and", "et al.", "in", etc.
% (Applied to english.lbx because of the mapping above.)
\DefineBibliographyStrings{english}{%
  andothers     = {dkk\adddot},
  andmore       = {dkk\adddot},
  and           = {dan},
  in            = {pada},
  editor        = {ed\adddot},
  editors       = {ed\adddot},
  bytranslator  = {diterjemahkan oleh},
  byeditor      = {disunting oleh},
  pages         = {hlm\adddot},
  page          = {hlm\adddot},
  volume        = {vol\adddot},
  number        = {no\adddot},
  edition       = {edisi},
  chapter       = {bab},
  techreport    = {laporan teknis},
  mathesis      = {tesis magister},
  phdthesis     = {disertasi doktor},
  url           = {URL},
  urlseen       = {diakses pada},
  bibliography  = {DAFTAR PUSTAKA},
  references    = {Daftar Pustaka}
}

% Bibliography format: hanging indent 1.27cm, single spacing
\setlength{\bibhang}{1.27cm}
\renewcommand*{\bibfont}{\small\singlespacing}
\setlength{\bibitemsep}{0.5\baselineskip}

% Year in parens after author: "Hill (2007)" / "(Hill, 2007)"
\renewbibmacro*{begentry}{}
% Use comma between author and year per ITB style
\renewcommand*{\nameyeardelim}{\addcomma\space}

% Custom citation macros mirroring ITB conventions
% \citetitb{key}      -> (Hill, 2007); (Gao dan Zhao, 2009); (Kramer dkk., 2005)
% \citenameitb{key}   -> Hill (2007); Gao dan Zhao (2009); Kramer dkk. (2005)
\newcommand{\citetitb}[2][]{\parencite[#1]{#2}}
\newcommand{\citenameitb}[2][]{\textcite[#1]{#2}}

% =====================================================================
% Math, tables, figures
% =====================================================================
\RequirePackage{amssymb,amsthm,mathtools}
\RequirePackage{tikz}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric,fit,calc,backgrounds}
\RequirePackage{booktabs}
\RequirePackage{array}
\RequirePackage{longtable}
\RequirePackage{multirow}
\RequirePackage{tabularx}
\RequirePackage{graphicx}
\RequirePackage{float}
\RequirePackage{subcaption}
\RequirePackage{xcolor}
\RequirePackage{enumitem}

% Indonesian decimal marker for siunitx
\RequirePackage{siunitx}
\sisetup{
  output-decimal-marker={,},
  group-separator={.},
  group-minimum-digits=4,
  list-final-separator={ dan },
  range-phrase={ sampai }
}

% Algorithms
\RequirePackage{algorithm}
\RequirePackage{algpseudocode}
\renewcommand{\listalgorithmname}{DAFTAR ALGORITMA}
\floatname{algorithm}{Algoritma}
\renewcommand{\thealgorithm}{\thechapter.\arabic{algorithm}}

% =====================================================================
% Hyperref (last among packages by convention)
% =====================================================================
\RequirePackage[hidelinks,
                bookmarks=true,
                bookmarksnumbered=true,
                pdfdisplaydoctitle=true,
                unicode=true]{hyperref}
\RequirePackage[noabbrev,capitalise,nameinlink]{cleveref}

% Indonesian cleveref names
\crefname{chapter}{Bab}{Bab}
\Crefname{chapter}{Bab}{Bab}
\crefname{section}{Subbab}{Subbab}
\Crefname{section}{Subbab}{Subbab}
\crefname{equation}{Pers.}{Pers.}
\Crefname{equation}{Persamaan}{Persamaan}
\crefname{figure}{Gambar}{Gambar}
\Crefname{figure}{Gambar}{Gambar}
\crefname{table}{Tabel}{Tabel}
\Crefname{table}{Tabel}{Tabel}
\crefname{algorithm}{Algoritma}{Algoritma}
\Crefname{algorithm}{Algoritma}{Algoritma}

% =====================================================================
% Disertasi metadata variables
% =====================================================================
\newcommand{\@nim}{}
\newcommand{\nim}[1]{\renewcommand{\@nim}{#1}}
\newcommand{\@prodi}{}
\newcommand{\prodi}[1]{\renewcommand{\@prodi}{#1}}
\newcommand{\@kodeprodi}{}
\newcommand{\kodeprodi}[1]{\renewcommand{\@kodeprodi}{#1}}
\newcommand{\@fakultas}{}
\newcommand{\fakultas}[1]{\renewcommand{\@fakultas}{#1}}
\newcommand{\@bulan}{}
\newcommand{\bulan}[1]{\renewcommand{\@bulan}{#1}}
\newcommand{\@tahun}{}
\newcommand{\tahun}[1]{\renewcommand{\@tahun}{#1}}
\newcommand{\@promotor}{}
\newcommand{\promotor}[1]{\renewcommand{\@promotor}{#1}}
\newcommand{\@nipPromotor}{}
\newcommand{\nipPromotor}[1]{\renewcommand{\@nipPromotor}{#1}}
% Anggota 1 (alias kopromotor untuk backward-compat)
\newcommand{\@anggotasatu}{}
\newcommand{\anggotasatu}[1]{\renewcommand{\@anggotasatu}{#1}}
\newcommand{\kopromotor}[1]{\renewcommand{\@anggotasatu}{#1}}% backward-compat alias
\newcommand{\@nipAnggotaSatu}{}
\newcommand{\nipAnggotaSatu}[1]{\renewcommand{\@nipAnggotaSatu}{#1}}
\newcommand{\nipKopromotor}[1]{\renewcommand{\@nipAnggotaSatu}{#1}}% backward-compat alias
% Anggota 2
\newcommand{\@anggotadua}{}
\newcommand{\anggotadua}[1]{\renewcommand{\@anggotadua}{#1}}
\newcommand{\@nipAnggotaDua}{}
\newcommand{\nipAnggotaDua}[1]{\renewcommand{\@nipAnggotaDua}{#1}}
\newcommand{\@logoITB}{figures/logo-itb.png}% TODO: ganti dengan file vektor logo-itb.pdf resmi ITB
\newcommand{\logoITB}[1]{\renewcommand{\@logoITB}{#1}}

% =====================================================================
% Sampul / Halaman Judul
%   Mengikuti diagram tata letak resmi pada Juknis SPs ITB (Mei 2019,
%   halaman 2). Urutan dari atas ke bawah:
%     1. Judul (TNR 14, bold, kapital, spasi 1)
%     2. "DISERTASI" (TNR 14, bold)
%     3. "Karya tulis ... Institut Teknologi Bandung" (TNR 12)
%     4. "Oleh"
%     5. NAMA MAHASISWA (TNR 12, bold, kapital)
%     6. NIM
%     7. (Program Studi Doktor ...)
%     8. Logo Ganesha (tinggi ~3,5 cm) -- DI BAWAH identitas mahasiswa
%     9. INSTITUT TEKNOLOGI BANDUNG
%    10. Bulan Tahun
% =====================================================================
\newcommand{\sampul}{%
  \begin{titlepage}
  \thispagestyle{empty}
  \begin{center}
    \vspace*{\ITBSampulTopGap}
    {\fontsize{14pt}{17pt}\selectfont\bfseries\MakeUppercase{\@title}\par}
    \vspace{\ITBSampulAfterTitle}
    {\fontsize{14pt}{17pt}\selectfont\bfseries DISERTASI\par}
    \vspace{\ITBSampulAfterDisertasi}
    {Karya tulis sebagai salah satu syarat\\
     untuk memperoleh gelar Doktor dari\\
     Institut Teknologi Bandung\par}
    \vspace{\ITBSampulAfterKaryaTulis}
    {Oleh\par}
    \vspace{\ITBSampulAfterOleh}
    {\bfseries\MakeUppercase{\@author}\par}
    {NIM: \@nim\par}
    \vspace{\ITBSampulAfterNimBlock}
    {(Program Studi Doktor \@prodi)\par}
    \vspace{\ITBSampulAfterProdi}
    \includegraphics[height=\ITBSampulLogoHeight]{\@logoITB}\par
    \vfill
    {\fontsize{14pt}{17pt}\selectfont\bfseries INSTITUT TEKNOLOGI BANDUNG\par}
    \vspace{\ITBSampulAfterInstitusi}
    {\bfseries\@bulan{} \@tahun\par}
  \end{center}
  \end{titlepage}
  % Pastikan halaman verso (kosong) di belakang sampul tidak menampilkan
  % nomor halaman arab sebelum \itbfrontmatter sempat mengaktifkan
  % penomoran roman.
  \thispagestyle{empty}
  \mbox{}\newpage
}

% =====================================================================
% Halaman Pengesahan (mendukung 1 Ketua + 2 Anggota)
%   Mengikuti diagram tata letak resmi pada Juknis SPs ITB (Mei 2019,
%   halaman 5). Seluruh konten dipusatkan (centered); TIDAK ada blok
%   tanda tangan penulis --- hanya tanda tangan Tim Pembimbing.
%   Nama pembimbing ditulis di antara tanda kurung sesuai contoh
%   pada juknis.
% =====================================================================
\newcommand{\itbsignaturerule}{\rule{6cm}{0.4pt}}
\newcommand{\halamanpengesahan}{%
  \cleardoublepage
  \thispagestyle{empty}
  \addcontentsline{toc}{chapter}{HALAMAN PENGESAHAN}
  \begin{singlespace}
  \begin{center}
    {\fontsize{14pt}{17pt}\selectfont\bfseries HALAMAN PENGESAHAN\par}
    \vspace{\ITBPengesahanAfterHeading}
    {\fontsize{14pt}{17pt}\selectfont\bfseries\MakeUppercase{\@title}\par}
    \vspace{\ITBPengesahanAfterTitle}
    Oleh\\[2pt]
    \textbf{\@author}\\
    \textbf{NIM: \@nim}\\
    \textbf{(Program Studi Doktor \@prodi)}\\
    Institut Teknologi Bandung\par
    \vspace{\ITBPengesahanAfterAuthorBlock}
    Menyetujui\\
    Tim Pembimbing\par
    \vspace{\ITBPengesahanAfterMenyetujui}
    Tanggal \makebox[3.5cm]{\dotfill}\par
    \vspace{\ITBPengesahanAfterTanggal}
    Ketua\par
    \vspace{\ITBPengesahanAfterKetuaLabel}
    \itbsignaturerule\\
    (\@promotor)\par
    \vspace{\ITBPengesahanAfterKetuaRule}
    \begin{minipage}[t]{0.48\textwidth}
      \centering
      Anggota\\[\ITBPengesahanAnggotaRuleDrop]
      \itbsignaturerule\\
      (\@anggotasatu)
    \end{minipage}\hfill
    \begin{minipage}[t]{0.48\textwidth}
      \centering
      Anggota\\[\ITBPengesahanAnggotaRuleDrop]
      \itbsignaturerule\\
      (\@anggotadua)
    \end{minipage}
  \end{center}
  \end{singlespace}
  \cleardoublepage
}

% =====================================================================
% Pedoman Penggunaan Disertasi
% =====================================================================
\newcommand{\pedomanpenggunaan}{%
  \chapter*{PEDOMAN PENGGUNAAN DISERTASI}
  \addcontentsline{toc}{chapter}{Pedoman Penggunaan Disertasi}
  \begin{singlespace}
  Disertasi Doktor yang tidak dipublikasikan terdaftar dan tersedia di
  Perpustakaan Institut Teknologi Bandung, dan terbuka untuk umum dengan
  ketentuan bahwa hak cipta ada pada pengarang dengan mengikuti aturan
  HaKI yang berlaku di Institut Teknologi Bandung. Referensi kepustakaan
  diperkenankan dicatat, tetapi pengutipan atau peringkasan hanya dapat
  dilakukan seizin pengarang dan harus disertai dengan kebiasaan ilmiah
  untuk menyebutkan sumbernya.

  \vspace{\ITBPedomanBetweenParagraphs}
  Memperbanyak atau menerbitkan sebagian atau seluruh disertasi haruslah
  seizin Dekan Sekolah Pascasarjana, Institut Teknologi Bandung.
  \end{singlespace}
  \cleardoublepage
}

% =====================================================================
% Abstrak / Abstract environment
% =====================================================================
\newenvironment{abstrak}{%
  \chapter*{ABSTRAK}
  \addcontentsline{toc}{chapter}{ABSTRAK}
  \begin{singlespace}
  \noindent
  \textbf{\MakeUppercase{\@title}}\\[6pt]
  \textbf{Oleh}\\
  \textbf{\@author}\\
  \textbf{NIM: \@nim}\\
  \textbf{(Program Studi Doktor \@prodi)}\\[12pt]
}{%
  \end{singlespace}
  \cleardoublepage
}

\newenvironment{abstract*}{%
  \chapter*{ABSTRACT}
  \addcontentsline{toc}{chapter}{ABSTRACT}
  \selectlanguage{english}
  \begin{singlespace}
  \noindent
  \textbf{\MakeUppercase{\@title}}\\[6pt]
  \textbf{By}\\
  \textbf{\@author}\\
  \textbf{NIM: \@nim}\\
  \textbf{(\@prodi{} Doctoral Program)}\\[12pt]
}{%
  \end{singlespace}
  \selectlanguage{bahasa}
  \cleardoublepage
}

% Kata Pengantar environment
\newenvironment{katapengantar}{%
  \chapter*{KATA PENGANTAR}
  \addcontentsline{toc}{chapter}{KATA PENGANTAR}
  \begin{singlespace}
}{%
  \end{singlespace}
  \cleardoublepage
}

% Alias environments for convenience
\newenvironment{abstractid}{\begin{abstrak}}{\end{abstrak}}
\newenvironment{abstracten}{\begin{abstract*}}{\end{abstract*}}

% Front matter / main matter helpers (renamed to avoid clash with report)
\newcommand{\itbfrontmatter}{%
  \pagenumbering{roman}
  \pagestyle{fancy}
  \fancyhf{}
  \fancyfoot[C]{\thepage}
}
\newcommand{\itbmainmatter}{%
  \cleardoublepage
  \pagenumbering{arabic}
  \setcounter{page}{1}
}

\endinput
