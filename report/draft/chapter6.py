# Chapter 6: System Testing & Performance Profiling

CHAPTER6_TEX = r"""
\clearpage
\chapter{System Testing and Performance Profiling}

Software that controls live cloud infrastructure needs rigorous validation. A small bug in a ring buffer or a single unhandled NaN in a neural network prediction can cause an autoscaler to crash, leaving nodes unmanaged while traffic surges continue to arrive. 

We validated our system through a two-tiered testing strategy: first, an automated unit and integration test suite executed through `pytest`; second, a comprehensive real-time systems microbenchmark that measured latency, throughput, and memory consumption across the entire control plane.

\section{Automated QA Test Suite Overview}

The automated test suite in `tests/` covers every core component of the system. Table~\ref{tab:test_suite} summarizes the test modules, targets, and verification outcomes.

\begin{table}[htbp]
\centering
\caption{Automated Unit and Integration Test Suite Verification}
\label{tab:test_suite}
\resizebox{\linewidth}{!}{%
\begin{tabular}{llcc}
\toprule
\textbf{Test Module} & \textbf{Component Under Test} & \textbf{Cases} & \textbf{Result} \\
\midrule
`test_streaming_engine.py` & Ingestion Engine \& Rolling Buffer & 3 & Passed \\
`test_oracles.py` & LSTM, Transformer, XGBoost, DNN & 2 & Passed \\
`test_environments.py` & Single \& Heterogeneous Gym Envs & 2 & Passed \\
`test_safe_shield.py` & Action Interception \& Safe Boundary & 1 & Passed \\
`test_industry_baselines.py` & Kubernetes HPA v2 \& PID Controller & 3 & Passed \\
`test_api_server.py` & FastAPI Endpoints \& Prometheus Metrics & 3 & Passed \\
\midrule
\textbf{Total Test Suite} & \textbf{Complete System Pipeline} & \textbf{14} & \textbf{100\% Passed} \\
\bottomrule
\end{tabular}%
}
\end{table}

All 14 tests completed and passed in 20.00 seconds on Python 3.13.

\section{Unit Testing: Data Streaming and Memory Invariance}

In `test_streaming_engine.py`, we verified three critical properties:
\begin{itemize}
\item **Continuous Tick Ingestion:** Confirmed that \texttt{Streaming\-Ingester.\allowbreak step()} retrieves sequential dictionary records without dropping fields or raising unexpected `StopIteration` exceptions.
\item **Rolling Window Eviction:** Confirmed that `RollingWindowBuffer` maintains a strict maximum length of 64 entries. When 100 consecutive records are pushed, exactly the 64 most recent events remain in the buffer.
\item **Zero-Padding on Reset:** Verified that querying the buffer before 64 records have arrived returns a zero-padded tensor of exact shape $(64, 3)$, preventing shape mismatch errors in downstream neural networks.
\end{itemize}

\section{Oracle Numerical Stability and Output Shape Validation}

In `test_oracles.py`, we evaluated model inference under edge conditions:
\begin{itemize}
\item **Multi-Horizon Output Dimensions:** Confirmed that passing a batch of shape $(B, 64, 3)$ through the multi-horizon Transformer returns an output tensor of exact shape $(B, 3, 2)$ corresponding to the 3 lookahead horizons and 2 resource dimensions.
\item **Absence of Numerical Anomalies:** Passed zero-filled, extreme, and negative inputs through the models to ensure that neither the PyTorch models nor the XGBoost booster produce `NaN` or `Inf` outputs.
\end{itemize}

\section{Gymnasium MDP Contract and Action Space Verification}

In `test_environments.py`, we verified that `K8sScalingEnv` and `HeterogeneousK8sEnv` strictly adhere to the Gymnasium API contracts:
\begin{itemize}
\item Verified that `env.reset()` returns an observation within the declared observation space bounds ($[-1000.0, 1000.0]^{11}$) and an empty metadata dictionary.
\item Confirmed that calling `env.step(action)` with extreme boundary actions ($[-1.0, -1.0]$ and $[1.0, 1.0]$) produces valid next-state vectors, finite scalar rewards, and boolean completion flags without crashing.
\end{itemize}

\section{Safe Shield Clamping Invariance Proof}

In `test_safe_shield.py`, we tested the safety barrier under adversarial conditions:
\begin{itemize}
\item We initialized the cluster with a high simulated CPU demand of 80 cores. Under this demand, safe minimum capacity requires:
\begin{equation}
N_{\text{safe}} = \frac{80.0}{6.4} = 12.5 \text{ nodes}
\end{equation}
\item We forced the policy to issue a maximal scale-down command ($a = -1.0$), attempting to drop capacity down to 5.0 nodes.
\item The `SafeActionShield` successfully intercepted the command, clamped the action to the exact boundary $N_{\text{safe}}$, and injected the $-50.0$ penalty into the reward.
\end{itemize}
This confirmed that the safety barrier holds even when the RL policy makes worst-case decisions.

\section{Baseline Emulation and Accounting Verification}

In `test_industry_baselines.py`, we validated the baseline controllers:
\begin{itemize}
\item **Kubernetes HPA Deadband:** Verified that when utilization sits at 78\% (within $\pm 10\%$ of the 80\% target), HPA outputs an action of 0.0 (no scaling).
\item **PID Anti-Windup Clamping:** Verified that sustained over-utilization does not cause the integrator term to grow unbounded, ensuring rapid recovery when demand subsides.
\item **Financial Ledger Accounting:** Confirmed that \texttt{Cloud\-Financial\-And\-Carbon\-Ledger} accurately tracks dollar spend and carbon emissions across evaluation steps.
\end{itemize}

\section{Real-Time Systems Microbenchmarking (Stage 15)}

Simulating an autoscaler in Python is one thing; proving that it can run inside a real-time Kubernetes control loop is another. In Stage 15, we profiled the complete control plane end-to-end. Figure~\ref{fig:microbench} presents the microbenchmark results across all four system layers.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.9\linewidth]{stage15_systems_microbenchmark.png}
\caption{Comprehensive systems profiling across inference latency, streaming throughput, and memory footprint}
\label{fig:microbench}
\end{figure}

Table~\ref{tab:microbenchmark_results} summarizes the quantitative findings from the profiling run.

\begin{table}[htbp]
\centering
\caption{System Microbenchmark Performance Metrics}
\label{tab:microbenchmark_results}
\resizebox{\linewidth}{!}{%
\begin{tabular}{llcc}
\toprule
\textbf{Component} & \textbf{Performance Metric} & \textbf{Measured Value} & \textbf{Compliance} \\
\midrule
PyArrow Ingestion & Peak Streaming Throughput & 44,615 rows/sec & Target $>10\text{k}$ \\
Record Deserializer & Per-Event Ingestion Latency & 22.41 $\mu$s / record & Sub-50 $\mu$s \\
Oracle State Fusion & Median Latency ($p_{50}$) & 8.41 ms & Target $<10$ ms \\
Oracle Tail Latency & 99th-Percentile Latency ($p_{99}$) & 13.36 ms & Target $<25$ ms \\
PPO Policy Forward & Median Forward Pass & 0.15 ms & Sub-millisecond \\
\textbf{End-to-End Decision} & \textbf{Median Control Loop ($p_{50}$)} & \textbf{8.57 ms} & \textbf{Sub-10ms} \\
\textbf{End-to-End Tail} & \textbf{99th-Percentile Loop ($p_{99}$)} & \textbf{13.57 ms} & \textbf{Bounded Tail} \\
Scheduling Throughput & Maximum Decision Rate & 117 actions/sec & High Frequency \\
Process Memory & Resident Set Size (RSS) & 637.9 MB & $<1,024$ MB \\
\bottomrule
\end{tabular}%
}
\end{table}

These microbenchmarks confirm three practical systems advantages:
\begin{enumerate}
\item **Sub-10ms Control Loop:** The median end-to-end decision executes in 8.57 milliseconds, with the 99th-percentile capped at 13.57 milliseconds. The autoscaler can issue up to 117 scaling decisions per second, far exceeding standard Kubernetes sync intervals (15 to 30 seconds).
\item **Zero-Copy Streaming:** Apache Arrow memory mapping eliminates Python deserialization bottlenecks, achieving 44,615 records per second at 22.41 microseconds per event.
\item **Lightweight Footprint:** The entire runtime (including PyTorch models, the XGBoost booster, Stable-Baselines3 weights, and ring buffers) consumes only 637.9~MB of RAM, making it practical to run as a standard Kubernetes sidecar.
\end{enumerate}
"""
