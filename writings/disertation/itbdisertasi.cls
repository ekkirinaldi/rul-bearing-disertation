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
% Geometry — ITB mirror margins
%   Odd pages (right-hand): left=4cm, right=3cm, top=3cm, bottom=3cm
%   Even pages (left-hand): right=4cm, left=3cm, others=3cm
% =====================================================================
\RequirePackage{geometry}
\geometry{
  a4paper,
  inner=4cm,        % left on odd, right on even
  outer=3cm,        % right on odd, left on even
  top=3cm,
  bottom=3cm,
  headheight=15pt,
  headsep=12pt,
  footskip=30pt,
  twoside
}

% =====================================================================
% Line spacing: 1.5 body; 1.0 for blocks
% =====================================================================
\RequirePackage{setspace}
\onehalfspacing  % approximately 1.5

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

% Chapter title format: "Bab I" on first line, title on next
\RequirePackage[explicit]{titlesec}
\titleformat{\chapter}[display]
  {\normalfont\fontsize{14}{18}\bfseries\centering}
  {\MakeUppercase{\chaptertitlename}\ \thechapter}
  {12pt}
  {\MakeUppercase{#1}}
\titlespacing*{\chapter}{0pt}{20pt}{30pt}

\titleformat{\section}
  {\normalfont\fontsize{12}{14}\bfseries}
  {\thesection}{1em}{#1}
\titlespacing*{\section}{0pt}{18pt}{6pt}

\titleformat{\subsection}
  {\normalfont\fontsize{12}{14}\bfseries}
  {\thesubsection}{1em}{#1}
\titlespacing*{\subsection}{0pt}{14pt}{4pt}

\titleformat{\subsubsection}
  {\normalfont\fontsize{12}{14}\bfseries\itshape}
  {\thesubsubsection}{1em}{#1}
\titlespacing*{\subsubsection}{0pt}{12pt}{4pt}

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

% Indonesian localization for "and", "et al.", "in", etc.
\DefineBibliographyStrings{english}{%
  andothers = {dkk\adddot},
  and       = {dan},
  in        = {pada},
  editor    = {ed\adddot},
  editors   = {eds\adddot},
  pages     = {hlm\adddot},
  page      = {hlm\adddot},
  volume    = {vol\adddot},
  number    = {no\adddot},
  edition   = {edisi},
  chapter   = {bab},
  bibliography = {DAFTAR PUSTAKA}
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
% =====================================================================
\newcommand{\sampul}{%
  \begin{titlepage}
  \thispagestyle{empty}
  \begin{center}
    \vspace*{0.5cm}
    \includegraphics[width=3cm]{\@logoITB}\par
    \vspace{1.2cm}
    {\fontsize{14pt}{17pt}\selectfont\bfseries\MakeUppercase{\@title}\par}
    \vspace{1.5cm}
    {\bfseries DISERTASI DOKTOR\par}
    \vspace{0.8cm}
    {Karya tulis sebagai salah satu syarat\\
     untuk memperoleh gelar Doktor dari\\
     Institut Teknologi Bandung\par}
    \vspace{1.5cm}
    {Oleh\par}
    \vspace{0.4cm}
    {\bfseries\@author\par}
    {NIM: \@nim\par}
    \vspace{0.4cm}
    {(Program Studi Doktor \@prodi)\par}
    \vfill
    \includegraphics[width=2.5cm]{\@logoITB}\par
    \vspace{0.4cm}
    {\bfseries INSTITUT TEKNOLOGI BANDUNG\par}
    {\bfseries\@bulan{} \@tahun\par}
  \end{center}
  \end{titlepage}
  \cleardoublepage
}

% =====================================================================
% Halaman Pengesahan (mendukung 1 Ketua + 2 Anggota)
% =====================================================================
\newcommand{\halamanpengesahan}{%
  \chapter*{PENGESAHAN}
  \thispagestyle{empty}
  \addcontentsline{toc}{chapter}{Pengesahan}
  \begin{singlespace}
  \begin{center}
  {\bfseries\MakeUppercase{\@title}\par}
  \end{center}
  \vspace{0.5cm}

  \noindent
  Oleh \textbf{\@author}\\
  NIM: \@nim\\
  (Program Studi Doktor \@prodi)\\
  Institut Teknologi Bandung

  \vspace{0.8cm}
  Adalah benar dibuat oleh saya sendiri dan belum pernah dibuat dan
  dipublikasikan oleh siapapun baik di dalam maupun di luar Institut
  Teknologi Bandung kecuali bagian-bagian tertentu yang telah dinyatakan
  sumbernya dengan jelas.

  \vspace{1.5cm}
  \begin{flushright}
  Bandung, \@bulan{} \@tahun\\[2.5cm]
  \underline{\@author}\\
  NIM: \@nim
  \end{flushright}

  \vspace{1.0cm}
  Menyetujui Tim Pembimbing

  \vspace{0.3cm}
  \noindent Ketua

  \vspace{2.0cm}
  \noindent\underline{\@promotor}\\
  NIP. \@nipPromotor

  \vspace{0.8cm}
  \noindent
  \begin{tabular}{p{7cm}p{7cm}}
  Anggota, & Anggota, \\[2.0cm]
  \underline{\@anggotasatu} & \underline{\@anggotadua} \\
  NIP. \@nipAnggotaSatu & NIP. \@nipAnggotaDua
  \end{tabular}
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

  \vspace{0.5cm}
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
