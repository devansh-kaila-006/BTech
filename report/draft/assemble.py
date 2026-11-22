import os
import sys
import re

# Import all draft sections
from front_matter import FRONT_MATTER_TEX
from chapter1 import CHAPTER1_TEX
from chapter2 import CHAPTER2_TEX
from chapter3 import CHAPTER3_TEX
from chapter4 import CHAPTER4_TEX
from chapter5 import CHAPTER5_TEX
from chapter6 import CHAPTER6_TEX
from chapter7 import CHAPTER7_TEX
from chapter8 import CHAPTER8_TEX
from references import REFERENCES_TEX

PREAMBLE_TEX = r"""% =====================================================================
%  PROJECT REPORT - Dept. of CSE, Amrita School of Computing, Bengaluru
%  LaTeX Project Report for B.Tech Capstone Project (22AIE498)
%  Team ID: 23UG109 | Panel No: 12
%
%  COMPATIBILITY: Works with BOTH pdfLaTeX, XeLaTeX, and LuaLaTeX!
%    Overleaf: Works by default on pdfLaTeX or Menu -> Compiler -> XeLaTeX
%    Locally:  pdflatex report_template_AIE.tex  OR  xelatex report_template_AIE.tex
% =====================================================================
\documentclass[12pt]{report}

% ---------------- Page setup (same as the Word file) -----------------
\usepackage[letterpaper,
            top=1.38in, bottom=1.18in, left=1.58in, right=1.18in,
            headsep=0.38in, footskip=0.5in]{geometry}

% ---------------- Multi-Engine Font Handling (pdfTeX / XeTeX / LuaTeX) -
\usepackage{iftex}
\ifpdftex
  \usepackage[utf8]{inputenc}
  \usepackage[T1]{fontenc}
  \usepackage{mathptmx}  % Times Roman clone for pdfTeX
  \newcommand{\hffont}{\sffamily}
\else
  \usepackage{fontspec}
  % Body: Times New Roman (falls back to a Times clone if not installed)
  \IfFontExistsTF{Times New Roman}{\setmainfont{Times New Roman}}{%
    \IfFontExistsTF{TeX Gyre Termes}{\setmainfont{TeX Gyre Termes}}{%
      \setmainfont{Liberation Serif}}}
  % Header / footer: Calibri (falls back to Carlito, its metric clone)
  \IfFontExistsTF{Calibri}{\newfontfamily\hffont{Calibri}}{%
    \IfFontExistsTF{Carlito}{\newfontfamily\hffont{Carlito}}{%
      \newfontfamily\hffont{Liberation Sans}}}
\fi

% ---------------- Packages -------------------------------------------
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{setspace}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{titletoc}
\usepackage{enumitem}
\usepackage{array}
\usepackage{booktabs}
\usepackage[labelsep=colon]{caption}
\usepackage[numbers,square,sort&compress]{natbib}
\usepackage[hidelinks]{hyperref}   % clickable TOC entries, no coloured boxes
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{framed}               % for safe listings inside minipage
\usepackage{microtype}            % character protrusion and font expansion to prevent overfull hboxes

\graphicspath{{figures/}{./}}

\definecolor{wordred}{HTML}{FF0000}
\newcommand{\red}[1]{\textcolor{wordred}{#1}}

\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
\emergencystretch=3em
\tolerance=2000
\hbadness=2000
\sloppy

% =====================================================================
%  METADATA
% =====================================================================
\newcommand{\projecttitle}{Hierarchical-Action, Streaming-Ingestion Predictive-RL Autoscaling for Heterogeneous GPU Clusters}
\newcommand{\monthyear}{December 2026}       % header (right) and title page
\newcommand{\guidename}{Dr. Beena B. M., Associate Professor}

% ---------------- Headers and footers --------------------------------
% Front matter: centred roman page number only
\fancypagestyle{front}{%
  \fancyhf{}
  \renewcommand{\headrulewidth}{0pt}
  \fancyfoot[C]{\thepage}}

% Main chapters: CURRENT CHAPTER NAME (left, automatic) | Month, year (right)
% Footer: Dept. of CSE, ASC, Bengaluru (left) | Page | n (right)
\fancypagestyle{main}{%
  \fancyhf{}
  \renewcommand{\headrulewidth}{0pt}
  \fancyhead[L]{\hffont\small\leftmark}
  \fancyhead[R]{\hffont\small\monthyear}
  \fancyfoot[L]{\hffont\small\vspace*{\baselineskip}Dept. of CSE, ASC, Bengaluru}
  \fancyfoot[R]{\hffont\small Page | \thepage\vspace*{\baselineskip}}}
\fancypagestyle{plain}{}          % chapter pages keep the same header

\renewcommand{\chaptermark}[1]{\markboth{#1}{}}   % header = chapter title
\renewcommand{\sectionmark}[1]{}

% \chapter does NOT start a new page by itself (so notes can sit above
% the heading). Put \clearpage before every \chapter.
\makeatletter
\renewcommand{\chapter}{\global\@topnum\z@\@afterindentfalse
  \secdef\@chapter\@schapter}
\makeatother

% ---------------- Headings -------------------------------------------
% Chapter:  "CHAPTER - n"  and the title below, centred, bold, 14 pt
\titleformat{\chapter}[display]
  {\normalfont\fontsize{14}{17}\bfseries\centering}
  {CHAPTER - \thechapter}{6pt}{\MakeUppercase}
\titlespacing*{\chapter}{0pt}{0pt}{18pt}
% Section 1.1 (bold 12 pt), Subsection 1.1.1 (bold 12 pt)
\titleformat{\section}{\normalfont\fontsize{12}{14}\bfseries}{\thesection}{0.6em}{}
\titlespacing*{\section}{0pt}{12pt}{6pt}
\titleformat{\subsection}{\normalfont\fontsize{12}{14}\bfseries}{\thesubsection}{0.6em}{}
\titlespacing*{\subsection}{0pt}{10pt}{4pt}

\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{2}          % TOC shows chapters, sections, subsections

% ---------------- Table of Contents style ----------------------------
\contentsmargin{0.5in}
\newcommand{\tocdots}{\titlerule*[0.6pc]{.}}
% Chapters: "CHAPTER 1 – INTRODUCTION ........ 1"
\titlecontents{chapter}[0pt]{\addvspace{3pt}}
  {CHAPTER \thecontentslabel\ -- \MakeUppercase}
  {\hspace*{0.25in}\MakeUppercase}            % un-numbered entries
  {\tocdots\contentspage}
% Sections: "1.1 INTRODUCTION TO ....... 2"
\titlecontents{section}[0.95in]{}
  {\contentslabel{0.45in}\MakeUppercase}{\MakeUppercase}
  {\tocdots\contentspage}
% Subsections: "5.2.1 ..........  4"
\titlecontents{subsection}[1.55in]{}
  {\contentslabel{0.6in}\MakeUppercase}{\MakeUppercase}
  {\tocdots\contentspage}
% Figures and tables: "1.1   Role hierarchy ........ 5"
\titlecontents{figure}[0.55in]{\addvspace{2pt}}
  {\contentslabel{0.55in}}{}{\tocdots\contentspage}
\titlecontents{table}[0.55in]{\addvspace{2pt}}
  {\contentslabel{0.55in}}{}{\tocdots\contentspage}

\renewcommand{\contentsname}{TABLE OF CONTENTS}
\renewcommand{\listfigurename}{List Of Figures}
\renewcommand{\listtablename}{List Of Tables}
\renewcommand{\bibname}{REFERENCES}

% ---------------- Captions -------------------------------------------
% Figures: "Fig. 1.1: Title" below the figure; Tables: "Table 1.1: Title" above
\captionsetup[figure]{name=Fig., font=bf, labelfont=bf, skip=6pt}
\captionsetup[table]{name=Table, font=bf, labelfont=bf, position=top, skip=6pt}

% ---------------- References (IEEE style numbering "[1].") -----------
\renewcommand{\bibnumfmt}[1]{[#1].}
\setlength{\bibsep}{2pt}
\renewcommand{\bibpreamble}{Follow IEEE format and if websites are there
  put it at the end and also books.\par\vspace{6pt}}

% Heading for un-numbered front-matter pages, also added to the TOC
\newcommand{\frontchapter}[1]{%
  \clearpage\phantomsection
  \addcontentsline{toc}{chapter}{#1}%
  \chapter*{#1}}

\begin{document}

% "Page No." heading on the right of the Table of Contents
\addtocontents{toc}{\protect\hfill\textbf{Page No.}\protect\par}
"""

POSTAMBLE_TEX = r"""
\end{document}
"""

def convert_markdown_formatting(text):
    """
    Finds markdown code spans `code_span` and bold spans **bold** outside of verbatim environments,
    escapes underscores and ampersands in code spans, wraps in \texttt{...} and \textbf{...}.
    """
    parts = re.split(r'(\\begin\{verbatim\}.*?\\end\{verbatim\})', text, flags=re.DOTALL)
    out = []
    bt = chr(96)
    pattern_code = bt + r'([^' + bt + r'\n]+?)' + bt
    for p in parts:
        if p.startswith(r'\begin{verbatim}'):
            out.append(p)
        else:
            # Convert bold **text** to \textbf{text}
            p_conv = re.sub(r'\*\*([^*]+?)\*\*', r'\\textbf{\1}', p)
            # Convert code `text` to \texttt{text}
            def repl(m):
                code = m.group(1)
                code_esc = code.replace('_', r'\_').replace('&', r'\&')
                # Allow line breaks at dots, slashes, and hyphens so long monospace spans never overflow the margin
                code_esc = code_esc.replace('.', r'.\allowbreak ').replace('/', r'/\allowbreak ').replace('-', r'-\allowbreak ')
                return r'\texttt{' + code_esc + '}'
            p_conv = re.sub(pattern_code, repl, p_conv)
            out.append(p_conv)
    return ''.join(out)

def assemble_report():
    raw_body = (
        FRONT_MATTER_TEX +
        CHAPTER1_TEX +
        CHAPTER2_TEX +
        CHAPTER3_TEX +
        CHAPTER4_TEX +
        CHAPTER5_TEX +
        CHAPTER6_TEX +
        CHAPTER7_TEX +
        CHAPTER8_TEX +
        REFERENCES_TEX
    )
    
    # Process markdown spans and formatting
    processed_body = convert_markdown_formatting(raw_body)
    
    full_doc = PREAMBLE_TEX + processed_body + POSTAMBLE_TEX
    
    # Replace literal unicode en-dash and em-dash with standard LaTeX equivalents
    full_doc = full_doc.replace('–', '--').replace('—', '---')
    
    return full_doc

if __name__ == "__main__":
    report_text = assemble_report()
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "report_template_AIE.tex"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Assembled report written to {out_path} ({len(report_text)} bytes, {len(report_text.splitlines())} lines).")
