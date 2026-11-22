# Chapter 3: System Requirements and Analysis

CHAPTER3_TEX = r"""
\clearpage
\chapter{System Requirements and Analysis}

Building an autoscaler that makes sense for production clusters starts with real data. Synthetic benchmarks like Poisson arrival queues look neat in mathematical proofs, but they don't capture how messy production clusters actually get. Production workloads arrive in irregular bursts, feature long-tailed execution durations, and run on hardware pools that don't match incoming requests neatly. 

\section{Workload Dataset: Alibaba Production GPU Cluster Trace}

We grounded our entire study on the Alibaba OpenB GPU Cluster Trace (\texttt{cluster\allowbreak-trace\allowbreak-gpu\allowbreak-v2023}). This dataset captures resource traces across an enterprise production cluster serving co-located web services, batch analytics, and deep learning training jobs. 

Table~\ref{tab:dataset_spec} outlines the core parameters of the raw and cleaned trace files.

\begin{table}[htbp]
\centering
\caption{Alibaba GPU Cluster Trace Specifications}
\label{tab:dataset_spec}
\resizebox{0.88\linewidth}{!}{%
\begin{tabular}{ll}
\toprule
\textbf{Trace Attribute} & \textbf{Observed Dataset Value} \\
\midrule
Total Container Pods Processed & 156,383 \\
Heterogeneous Compute Nodes & 1,800+ physical servers \\
GPU-Accelerated Workload Fraction & 52.1\% \\
CPU-Intensive Workload Fraction & 47.9\% \\
Dominant GPU Architectures & NVIDIA T4, P100, V100, MISC \\
Trace Time Duration & 14 continuous days (1.2M seconds) \\
Cleaned Parquet Trace Size & 1.18~MB (compressed) \\
\bottomrule
\end{tabular}%
}
\end{table}

\section{Exploratory Data Analysis and Empirical Findings}

Before training neural networks or formulating reinforcement learning rewards, we conducted a 14-point exploratory data analysis across the Alibaba trace. The findings directly shaped our model architectures.

\subsection{Node Capacities and Density}

Figure~\ref{fig:node_caps} illustrates the distribution of compute cores and system memory across the physical nodes in the cluster.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\linewidth]{01_node_capacities.png}
\caption{Distribution of CPU and memory capacities across cluster worker nodes}
\label{fig:node_caps}
\end{figure}

The physical hosts aren't all built the same way. Most standard compute machines carry 96 cores and over 400~GB of RAM. The GPU nodes follow a strict bimodal split, packing either 2 or 8 accelerator cards. Packing tiny single-core CPU tasks onto an 8-GPU machine causes severe hardware fragmentation.

\subsection{Pod Resource Requests and Density Flavors}

Figure~\ref{fig:pod_demands} shows the requested CPU and memory distributions for container pods, while Figure~\ref{fig:hexbin} shows bivariate density across both axes on a logarithmic scale.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\linewidth]{02_pod_resource_demands.png}
\caption{Distribution of requested CPU cores and memory allocations across pods}
\label{fig:pod_demands}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\linewidth]{03_pod_cpu_mem_hexbin.png}
\caption{Bivariate density distribution of CPU and memory demand on log scale}
\label{fig:hexbin}
\end{figure}

Standard CPU pods ask for a median of roughly 10 cores. GPU jobs vary wildly: some demand almost zero CPU while requesting full GPUs, while others pull 120 cores alongside memory allocations over 200~GB. The hexbin density reveals distinct "flavors" where thousands of pods request identical $10^1$ milli-core and $10^3$~MiB profiles.

\subsection{GPU Architecture Preferences}

Figure~\ref{fig:gpu_specs} details the specific GPU models requested by accelerated pods.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\linewidth]{03b_gpu_specs_requested.png}
\caption{Breakdown of GPU architectures requested by container workloads}
\label{fig:gpu_specs}
\end{figure}

NVIDIA T4 cards dominate the cluster, making up over 60\% of GPU requests, with older P100 and specialized accelerators handling the remainder.

\subsection{Temporal Dynamics, Autocorrelation, and Burstiness}

Figure~\ref{fig:burstiness} plots arrival intensity over time, and Figure~\ref{fig:autocorr} plots the arrival autocorrelation function.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\linewidth]{04_workload_burstiness.png}
\caption{Arrival rate intensity over time showing pronounced traffic spikes}
\label{fig:burstiness}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\linewidth]{13_arrival_autocorrelation.png}
\caption{Autocorrelation function of pod arrival rates across time lags}
\label{fig:autocorr}
\end{figure}

Workloads arrive in sharp waves. Within a single hour, over 4,200 pods arrive together, followed by sharp lulls. The autocorrelation curve shows clear repeating diurnal cycles. 

When demand exhibits strong autocorrelation, sequence predictors like LSTMs and Transformers can spot incoming waves well before reactive thresholds trip.

\subsection{Duration and Scheduling Delay Distributions}

Figures~\ref{fig:dur_cdf} and \ref{fig:delay_cdf} display cumulative distribution functions (CDFs) for task durations and scheduling delays.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\linewidth]{05_duration_cdf.png}
\caption{Cumulative distribution function of pod execution durations}
\label{fig:dur_cdf}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\linewidth]{06_scheduling_delay_cdf.png}
\caption{Cumulative distribution function of pod scheduling delays}
\label{fig:delay_cdf}
\end{figure}

Nearly 90\% of container jobs finish within 2 hours. But the top 1\% run for days. The median scheduling delay sits at 2 seconds during quiet periods; during burst periods, 99th-percentile delays jump past 14 minutes.

\subsection{QoS Classes, Pod Phases, and Failure Paradox}

Figures~\ref{fig:qos_dist}, \ref{fig:phase_dist}, and \ref{fig:failure_cpu} analyze task priorities and failure rates.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\linewidth]{07_qos_distribution.png}
\caption{Proportion of Latency-Sensitive (LS) vs Best-Effort (BE) pods}
\label{fig:qos_dist}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\linewidth]{08_pod_phase_distribution.png}
\caption{Final execution phase breakdown across all trace pods}
\label{fig:phase_dist}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\linewidth]{09_failure_vs_cpu_demand.png}
\caption{Relationship between pod failure rates and requested CPU core count}
\label{fig:failure_cpu}
\end{figure}

Over 37,120 pods failed during the trace window. Interestingly, we discovered an inverse relationship between CPU request size and failure probability: smaller pods fail far more often than large, multi-node training jobs.

\subsection{Concurrency, Batching, and Feature Correlations}

Figures~\ref{fig:concurrency}, \ref{fig:batch_sizes}, \ref{fig:corr_matrix}, and \ref{fig:survival} capture global concurrency, bag-of-tasks batching, inter-feature correlations, and survival probabilities.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\linewidth]{11_global_concurrency.png}
\caption{Concurrent pod occupancy count integrated across the full trace}
\label{fig:concurrency}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\linewidth]{12_job_batch_sizes.png}
\caption{Histogram of simultaneous pod arrival batch sizes}
\label{fig:batch_sizes}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\linewidth]{10_correlation_matrix.png}
\caption{Correlation matrix across pod attributes, resource demands, and delays}
\label{fig:corr_matrix}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\linewidth]{14_survival_curve.png}
\caption{Non-parametric survival curve showing probability of ongoing execution}
\label{fig:survival}
\end{figure}

Concurrency spikes past 20,400 active containers at peak load. Large distributed training jobs submit up to 480 pods in the exact same second, creating sudden step-function demands on cluster schedulers.

\section{Functional Requirements}

The autoscaling system must fulfill six core functional capabilities:
\begin{enumerate}[label=\textbf{FR-\arabic*.}]
\item \textbf{Continuous Batch Streaming Ingestion:} Stream pre-processed Parquet records tick-by-tick without loading the full 156k-row dataset into memory.
\item \textbf{Predictive State Assembly:} Maintain an $O(1)$ rolling-window buffer and query the oracle ensemble to construct an observation tensor at each step.
\item \textbf{Decoupled Dual-Pool Actuation:} Output independent scaling decisions for CPU-optimized and GPU-accelerated server pools.
\item \textbf{Runtime Action Interception:} Evaluate the proposed scaling action against current cluster demand and override dangerous scale-down actions.
\item \textbf{SLA Violation Tracking:} Monitor cluster utilization and record contract breaches whenever node utilization exceeds 80\%.
\item \textbf{Prometheus Metric Export:} Expose operational metrics and scaling recommendations over an HTTP interface formatted for standard monitoring tools.
\end{enumerate}

\section{Non-Functional Requirements and Real-Time Bounds}

Production cloud systems impose strict operational constraints:
\begin{itemize}
\item \textbf{Decision Latency:} The complete inference loop (including feature extraction, oracle predictions, PPO forward pass, and shield evaluation) must execute in under 15 milliseconds.
\item \textbf{Ingestion Throughput:} The streaming engine must sustain throughput exceeding 3,000 events per second to support fast-forward trace replay.
\item \textbf{Process Footprint:} Peak memory utilization must stay under 1,024~MB so the autoscaler can run inside a lightweight Kubernetes sidecar.
\item \textbf{Safety Invariance:} The system must guarantee zero SLA breaches caused by controller-driven under-provisioning.
\end{itemize}

\section{Hardware and Software Requirements}

Development and evaluation used the following specifications:
\begin{itemize}
\item \textbf{Hardware Testbed:}
  \begin{itemize}
  \item Host CPU: Multi-core x86\_64 processor (8+ physical cores)
  \item System RAM: Minimum 16~GB DDR4/DDR5
  \item Accelerator: NVIDIA GPU with CUDA 11.8+ support (used for offline training)
  \item Storage: 20~GB available SSD storage for traces and model weights
  \end{itemize}
\item \textbf{Software Stack:}
  \begin{itemize}
  \item Operating System: Windows 11 / Ubuntu Linux 22.04 LTS
  \item Language Runtime: Python 3.10+
  \item Deep Learning: PyTorch 2.0+, Stable-Baselines3 2.0+
  \item Machine Learning: XGBoost 2.0+, Scikit-Learn 1.2+, Optuna 3.0+
  \item Data Ingestion: PyArrow 14.0+, NumPy 1.24+, Pandas 2.0+
  \item Microservice: FastAPI, Uvicorn, Docker, Docker Compose
  \end{itemize}
\end{itemize}

\section{Mathematical Problem Formulation}

We model cluster autoscaling as a Constrained Markov Decision Process (CMDP) defined by the tuple $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \mathcal{C}, \gamma)$:
\begin{itemize}
\item \textbf{State Space ($\mathcal{S} \subseteq \mathbb{R}^{11}$):} At time step $t$, the state vector $\mathbf{s}_t$ includes normalized pod requests, cluster pool sizes, current utilization levels, and oracle forecasts:
\begin{equation}
\mathbf{s}_t = \left[ c_t^{\text{norm}}, m_t^{\text{norm}}, w_t, N_t^{\text{cpu}}, N_t^{\text{gpu}}, U_t^{\text{cpu}}, U_t^{\text{gpu}}, \hat{c}_{t+1}, \hat{m}_{t+1}, \hat{d}_{t+1}, \hat{p}_{t} \right]
\end{equation}
\item \textbf{Action Space ($\mathcal{A} = [-1.0, 1.0]^2$):} The agent outputs continuous scaling velocities $\mathbf{a}_t = [a_t^{\text{cpu}}, a_t^{\text{gpu}}]$.
\item \textbf{Reward Function ($\mathcal{R}$):} Reflects operational costs:
\begin{equation}
R_t = - \left( C_{\text{power}}(N_t^{\text{cpu}}, N_t^{\text{gpu}}) + C_{\text{SLA}}(U_t^{\text{cpu}}, U_t^{\text{gpu}}) \right)
\end{equation}
where power cost scales linearly with active servers and SLA cost penalizes utilization exceeding 80\%:
\begin{equation}
C_{\text{SLA}} = 12.0 \cdot \max(0, U_t^{\text{cpu}} - 0.8) + 30.0 \cdot \max(0, U_t^{\text{gpu}} - 0.8)
\end{equation}
\item \textbf{Constraint Cost ($\mathcal{C}$):} To formalize safety without manual weight tweaking, we bound the expected SLA violation rate:
\begin{equation}
\min_{\pi_\theta} \; \mathbb{E}_{\pi} \left[ \sum_{t=0}^{T} \gamma^t C_{\text{power}}(t) \right] \quad \text{subject to} \quad \mathbb{E}_{\pi} \left[ \sum_{t=0}^{T} \gamma^t \mathbb{I}(U_t > 0.8) \right] \le \epsilon_{\text{SLA}}
\end{equation}
where $\epsilon_{\text{SLA}} = 0.05$ represents an upper bound of 5\% allowable breach timesteps.
\end{itemize}
"""
