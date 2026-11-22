# Front matter for Amrita B.Tech Report

FRONT_MATTER_TEX = r"""
% =====================================================================
%  TITLE PAGE
% =====================================================================
\begin{titlepage}
\centering
\vspace*{0.6in}
{\fontsize{14}{17}\selectfont\bfseries \projecttitle\par}
\vspace{10pt}
A PROJECT REPORT\par
\vspace{10pt}
{\fontsize{14}{17}\selectfont\bfseries\itshape Submitted by\par}
\vspace{14pt}
\begin{tabular}{@{}l@{\hspace{0.45in}}l@{}}
BL.EN.U4AIE23004 & Devansh Kaila\\[16pt]
BL.EN.U4AIE23072 & Akshay Damera\\
\end{tabular}\par
\vspace{18pt}
{\fontsize{14}{17}\selectfont\bfseries\itshape in partial fulfillment for the award of the degree of\par}
\vspace{18pt}
{\fontsize{14}{17}\selectfont\bfseries BACHELOR OF TECHNOLOGY\par}
\vspace{16pt}
{\small IN\par}
\vspace{16pt}
COMPUTER SCIENCE AND ENGINEERING\par
(Artificial Intelligence) \\
\vspace{12pt}
\includegraphics[width=1.75in]{amrita_logo.png}\par
\vspace{12pt}
AMRITA SCHOOL OF COMPUTING, BENGALURU\par
\vspace{12pt}
{\fontsize{14}{17}\selectfont AMRITA VISHWA VIDYAPEETHAM\par}
\vspace{12pt}
BENGALURU 560 035\par
\vspace{12pt}
\MakeUppercase{\monthyear}\par
\end{titlepage}

% =====================================================================
%  FRONT MATTER  (roman page numbers)
% =====================================================================
\pagestyle{front}
\pagenumbering{roman}
\setcounter{page}{1}

% ---------------- ACKNOWLEDGEMENTS -----------------------------------
\frontchapter{ACKNOWLEDGEMENTS}
\begin{spacing}{1.5}
\setlength{\parindent}{0.5in}
\indent The satisfaction that accompanies the successful completion of any task would be incomplete without mentioning the people who made it possible, and whose constant encouragement and guidance inspired us throughout this project work.

We offer our sincere pranams at the lotus feet of \textbf{``AMMA'', MATA AMRITANANDAMAYI DEVI}, who showered her blessings upon us throughout the course of this project.

We owe our deep gratitude to \textbf{Prof. Manoj P.}, Director, Amrita Vishwa Vidyapeetham Bengaluru Campus, for providing an academic setting that lets students pursue technical research. We place our heartfelt gratitude to \textbf{Dr. Gopalakrishnan E.A.}, Principal, Amrita School of Computing and Amrita School of Artificial Intelligence, Bengaluru, for his consistent institutional support and encouragement.

It's a genuine pleasure to express our gratitude to our project guide, \textbf{\guidename}, Department of Computer Science and Engineering, Amrita School of Computing, Bengaluru. Her feedback, patience, and technical pointers kept this work on track whenever our training loops diverged or the simulator hit memory bottlenecks.

We thank our project panel members for their critical questions during reviews, and all faculty members for their teaching and assistance. Finally, we're forever grateful to our parents, who supported and encouraged us every day.
\end{spacing}

% ---------------- ABSTRACT -------------------------------------------
\frontchapter{ABSTRACT}
\begin{spacing}{1.5}
Kubernetes Horizontal Pod Autoscaler reacts only after resource saturation happens. That's too late. When sudden traffic bursts hit a cluster, physical nodes take anywhere from 30 to 120 seconds to boot up and join the pool. During that startup window, incoming container pods sit trapped in pending queues, and request latency spikes past contract thresholds. 

This project tackles that lag. We built an end-to-end predictive reinforcement learning autoscaler designed specifically for heterogeneous GPU clusters. 

First, we scrapped batch dataframe loading. Reading giant trace files with Pandas ate 6.71~MB per chunk and bogged down execution; our streaming engine uses PyArrow batch iterators with an $O(1)$ ring buffer, cutting memory use by 84\% down to 1.04~MB and boosting ingestion throughput by 4.6x (4,064 rows per second vs 880 rows per second). Second, we trained an ensemble of predictive oracles—a multi-horizon Time-Series Transformer forecasting workloads at $t+1$, $t+5$, and $t+15$, an XGBoost regressor for scheduling delays, and a PyTorch neural net for server wattage. Third, we fused these predictions into a continuous Proximal Policy Optimization agent that manages standard 16-core CPU nodes and 8-GPU accelerators separately. To stop exploratory actions from starving live containers, we put a runtime safe action shield in front of the cluster; it clamped unsafe scale-down commands and wiped out 1,158 baseline SLA violations entirely. Over a 100,000-step evaluation on unseen Alibaba traces, the predictive policy cut SLA delay breaches by 32.8\% while raising cumulative reward by 4.7\%.
\end{spacing}

% ---------------- AUTOMATIC LISTS ------------------------------------
\clearpage
{\linespread{0.95}\selectfont\tableofcontents}

\clearpage\phantomsection
\addcontentsline{toc}{chapter}{LIST OF FIGURES}
\listoffigures

\clearpage\phantomsection
\addcontentsline{toc}{chapter}{LIST OF TABLES}
\listoftables
"""
