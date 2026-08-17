%%
%% mdpi.cls -- Faithful MDPI journal class shim
%% Approximates the MDPI journal layout for LaTeX submission.
%% All required packages confirmed present in TeX Live 2021/2022.
%% Created 2025-07-20 for paper1_MDPI.tex
%%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{mdpi}[2025/07/20 MDPI Journal Class Shim v1.0]

%% =====================================================================
%%  BASE CLASS: article, A4, 11pt, single column
%% =====================================================================
\LoadClass[a4paper,10pt]{article}

%% =====================================================================
%%  FONTS: Times Roman via mathptmx (text + math)
%% =====================================================================
\RequirePackage[T1]{fontenc}
\RequirePackage{mathptmx}

%% =====================================================================
%%  PAGE GEOMETRY: A4, 2 cm margins all around, no header/footer area
%% =====================================================================
\RequirePackage[a4paper,
  top    = 2cm,
  bottom = 2cm,
  left   = 2cm,
  right  = 2cm,
  nohead,
  nofoot
]{geometry}

\pagestyle{plain}

%% =====================================================================
%%  MICROTYPE: optical margin alignment + character protrusion
%% =====================================================================
\RequirePackage{microtype}

%% =====================================================================
%%  PARAGRAPH STYLE: no indent, small inter-paragraph skip (MDPI style)
%% =====================================================================
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt plus 1pt minus 1pt}

%% =====================================================================
%%  SECTION FORMATTING
%%  MDPI uses bold, Arabic-numbered headings:
%%    "1.", "1.1.", "1.1.1."   (NOT Roman numerals)
%% =====================================================================
\RequirePackage{titlesec}

\titleformat{\section}
  {\normalfont\large\bfseries}     % font spec
  {\thesection.}                   % label  e.g. "1."
  {0.5em}                          % sep between label and title text
  {}                               % before-code (none)

\titleformat{\subsection}
  {\normalfont\normalsize\bfseries}
  {\thesubsection.}
  {0.5em}{}

\titleformat{\subsubsection}
  {\normalfont\normalsize\bfseries\itshape}
  {\thesubsubsection.}
  {0.5em}{}

\titlespacing*{\section}
  {0pt}{14pt plus 2pt minus 1pt}{6pt plus 1pt minus 1pt}
\titlespacing*{\subsection}
  {0pt}{10pt plus 1pt minus 1pt}{4pt plus 1pt minus 1pt}
\titlespacing*{\subsubsection}
  {0pt}{8pt plus 1pt}{3pt plus 1pt}

%% =====================================================================
%%  CAPTION STYLE
%%    Tables:  caption ABOVE  -- bold "Table N." prefix, small font
%%    Figures: caption BELOW  -- bold "Figure N." prefix, small font
%% =====================================================================
\RequirePackage{caption}
\captionsetup{
  labelsep  = period,    % "Figure 1. Caption text"
  labelfont = bf,        % bold label
  font      = small,     % small caption text
  skip      = 4pt
}
\captionsetup[table] {position = above, skip = 4pt}
\captionsetup[figure]{position = below, skip = 4pt}

%% =====================================================================
%%  TITLE / AUTHOR / AFFILIATION / ABSTRACT / KEYWORD MACHINERY
%%
%%  Usage in document:
%%    \title{Paper title}
%%    \author{A. Author$^{1}$, B. Author$^{2}$}
%%    \affil{1}{First Institution Name}
%%    \affil{2}{Second Institution Name}
%%    \mdpiabstract{Full abstract text...}
%%    \keyword{keyword1; keyword2; keyword3}
%%    \maketitle
%% =====================================================================

%% Internal storage (@ is letter in .cls scope -- safe to use)
\def\mdpi@title{}
\def\mdpi@authors{}
\def\mdpi@affiliations{}
\def\mdpi@abstracttext{}
\def\mdpi@keywords{}

%% \title{text}
\renewcommand{\title}[1]{\gdef\mdpi@title{#1}}

%% \author{formatted string -- include superscript refs manually}
\renewcommand{\author}[1]{\gdef\mdpi@authors{#1}}

%% \affil{N}{Institution text}
%% Call once per affiliation; they accumulate in order called.
\newcommand{\affil}[2]{%
  \g@addto@macro\mdpi@affiliations{%
    \noindent\textsuperscript{#1}\,#2\par\vspace{1pt}%
  }%
}

%% \mdpiabstract{text}
\newcommand{\mdpiabstract}[1]{\gdef\mdpi@abstracttext{#1}}

%% \keyword{semicolon-separated list}
\newcommand{\keyword}[1]{\gdef\mdpi@keywords{#1}}

%% =====================================================================
%%  \maketitle  --  typesets the full MDPI title block:
%%    Title (LARGE, bold, flush-left)
%%    Authors (large, bold, flush-left)
%%    Affiliations (small, numbered)
%%    Horizontal rule
%%    Abstract: <text>
%%    Keywords: <italic list>
%%    Horizontal rule
%% =====================================================================
\renewcommand{\maketitle}{%
  \begin{flushleft}
    {\LARGE\bfseries\mdpi@title\par}%
    \vspace{10pt}%
    {\large\bfseries\mdpi@authors\par}%
    \vspace{5pt}%
    {\small\mdpi@affiliations}%
  \end{flushleft}%
  \vspace{4pt}%
  \noindent\rule{\linewidth}{0.5pt}\par
  \vspace{3pt}%
  {\small\noindent\textbf{Abstract:} \mdpi@abstracttext\par}%
  \vspace{5pt}%
  {\small\noindent\textbf{Keywords:} \textit{\mdpi@keywords}\par}%
  \vspace{3pt}%
  \noindent\rule{\linewidth}{0.5pt}\par
  \vspace{14pt}%
}

%% =====================================================================
%%  SUPPRESS unused article-class \maketitle side-effects
%% =====================================================================
\let\@maketitle\relax

%% End of mdpi.cls
