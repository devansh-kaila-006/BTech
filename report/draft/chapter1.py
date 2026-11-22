# Chapter 1: Introduction

CHAPTER1_TEX = r"""
\clearpage
\pagestyle{main}
\pagenumbering{arabic}
\setcounter{page}{1}
\onehalfspacing

\chapter{Introduction}

Cloud infrastructure runs on a simple economic conflict. Cloud tenants want instant responsiveness so their user-facing services never miss an agreement deadline, while infrastructure operators want to power down idle machines to keep electricity bills and carbon footprints under control. Balancing these two goals sounds straightforward on paper. In production clusters, it isn't.

\section{The Reactive Autoscaling Trap}

Kubernetes relies on the Horizontal Pod Autoscaler (HPA v2) as its standard tool for cluster scaling. HPA collects resource metrics like CPU percentage and memory consumption over a fixed sliding window (often 15 to 30 seconds) and computes the target replica count using a basic ratio rule:
\begin{equation}
\text{Desired Replicas} = \left\lceil \text{Current Replicas} \times \left( \frac{\text{Current Metric Value}}{\text{Target Metric Value}} \right) \right\rceil
\end{equation}
When average utilization crosses a set boundary, usually 80\%, HPA triggers a scale-up event.

Here lies the fundamental trap. HPA is purely reactive.

It can't act on a demand wave until that wave has already hit the nodes and saturated their processor cores. By the time HPA detects the spike, the damage is already underway. In public clouds such as Amazon Web Services (AWS) or Google Cloud Platform (GCP), launching a new worker node requires between 30 and 120 seconds for virtual machine provisioning, OS initialization, container runtime setup, and image pulling. If a sudden flash crowd arrives, incoming pods sit stalled in a `Pending` state. Memory buffers overflow. Response times blow past Service Level Agreements (SLAs), and customers see timeouts.

Static cooldown periods make things worse. To keep the cluster from thrashing when metrics wobble, HPA introduces stabilization windows. While this stops rapid flapping, it delays urgently needed scale-up actions when consecutive traffic bursts hit back to back.

% [FLAG: Real limitation / complaint needed here - note that while threshold tuning sounds simple, finding one threshold that fits both batch and web jobs is practically impossible]
Tuning these thresholds by hand doesn't solve the core issue. Pick a conservative threshold like 60\%, and you pay for dozens of half-empty nodes day and night. Pick an aggressive threshold like 85\%, and any sudden spike guarantees an SLA disaster.

\section{Heterogeneous Compute and Hardware Mismatch}

Modern enterprise workloads don't run on uniform hardware anymore. A production cluster hosts a chaotic mix of tasks: lightweight Python web microservices, Java data-ingestion jobs, distributed Spark pipelines, and multi-gigabyte PyTorch model inference routines. 

To serve these tasks, datacenters deploy heterogeneous machines. Standard CPU-optimized nodes pack 16 or 32 high-clock compute cores, while GPU-accelerated servers carry high-density hardware accelerators (such as NVIDIA T4, V100, or A10 units) with large thermal design power (TDP) footprints. Running an 8-core CPU job on a GPU host wastes hundreds of dollars in idle accelerator depreciation. Conversely, routing a deep learning inference batch to a CPU-only node triggers severe queue backups.

Reactive autoscalers treat these pools symmetrically. They view the cluster as an undifferentiated bag of compute, scaling nodes up or down without understanding which specific hardware tier is actually starving.

\section{Research Motivation and Core Objectives}

We need autoscalers that look ahead instead of looking back. If an intelligent controller knows a traffic surge will peak 60 seconds from now, it can boot worker nodes ahead of time. When the traffic surge arrives, the nodes are already warm, registered with the Kubernetes API, and ready to accept container pods.

Building that predictive controller isn't easy. First, training machine learning policies on real-world cluster traces requires ingesting millions of historical arrival records without blowing up system RAM. Second, the agent needs accurate predictions of oncoming demand, queue delays, and electrical power consumption. Third, standard reinforcement learning agents explore by trial and error; in a live cluster, a single exploratory scale-down command during a surge will crash production services.

To solve these hurdles, this project pursues six specific technical objectives:
\begin{enumerate}[label=\textbf{O\arabic*.}]
\item \textbf{Build an $O(1)$ Streaming Ingestion Engine:} Replace slow, memory-heavy Pandas data loaders with a chunked PyArrow reader and a fixed rolling-window buffer to process massive trace files at high throughput with bounded RAM.
\item \textbf{Develop an Optuna-Tuned Multi-Oracle Ensemble:} Construct predictive models for workload burst forecasting, scheduling delay estimation, and electrical wattage prediction.
\item \textbf{Implement a Multi-Horizon Lookahead Transformer:} Forecast traffic demands across multiple future steps ($t+1$, $t+5$, and $t+15$) so the scaling policy can mask 60-second physical cloud VM boot delays.
\item \textbf{Decouple Hierarchical Multi-Pool Control:} Design a continuous-action Proximal Policy Optimization (PPO) agent that independently adjusts scaling speeds for CPU-optimized and GPU-accelerated server tiers.
\item \textbf{Guarantee Zero-Breach Safety via Action Shielding:} Insert a runtime control barrier function between the RL agent and the cluster to mathematically block unsafe scale-down actions during workload surges.
\item \textbf{Validate Against Real-World Baselines Under Stress:} Benchmark the full architecture against standard Kubernetes HPA and PID controllers across 15 disciplined ablation stages, including chaos injection and financial return-on-investment (ROI) analysis.
\end{enumerate}

\section{Key Contributions}

This dissertation delivers five concrete contributions to cloud resource management:
\begin{itemize}
\item \textbf{High-Speed Streaming Pipeline:} We built a PyArrow-backed ingestion engine that streams 156,383 container records from the Alibaba Cloud GPU cluster trace. It runs 4.6x faster than naive Pandas loading (4,064 rows/s vs 880 rows/s) and cuts memory consumption by 84\% down to 1.04~MB.
\item \textbf{Predictive Composite State Fusion:} We combined raw cluster metrics with forecasts from an Optuna-tuned LSTM, an XGBoost latency model, and a PyTorch deep power network into an 11-dimensional observation tensor.
\item \textbf{Lookahead Multi-Horizon Transformer:} We replaced single-step workload predictors with a multi-head attention Transformer predicting demands at $t+1$, $t+5$, and $t+15$, achieving a test mean squared error of 0.000680 and eliminating cold-boot starvation.
\item \textbf{Provable Safe Action Shield:} We designed an action interceptor that tracks instantaneous core demand and clamps dangerous scale-down actions. In empirical evaluations, it eliminated all 1,158 SLA delay violations seen in unshielded policies.
\item \textbf{Pareto Dominance and Chaos Resilience:} Over 100,000 evaluation timesteps, our full policy cut SLA breaches by 32.8\% over reactive baselines. Under injected chaos tests (including a sudden 50\% hardware node collapse and a 4.5x flash crowd), the policy restabilized the cluster in 2 steps, whereas the baseline took 24 steps.
\end{itemize}

\section{Report Organization}

The remainder of this report is organized into seven subsequent chapters:
\begin{itemize}
\item \textbf{Chapter 2 (Literature Survey):} Reviews previous work in multi-cloud scheduling, green cloud computing, learning-based autoscaling, and container orchestration, and positions our approach against existing literature.
\item \textbf{Chapter 3 (System Requirements and Analysis):} Examines the Alibaba Cloud trace dataset through 14 statistical exploratory figures, outlines functional and non-functional requirements, and formulates the scaling problem as a Constrained Markov Decision Process.
\item \textbf{Chapter 4 (System Design):} Details the high-level and low-level system architectures, explaining the streaming engine, oracle ensemble, composite state fusion, hierarchical control, safe action shield, and production cloud microservice.
\item \textbf{Chapter 5 (System Implementation):} Walks through the modular code implementation of each subsystem, showing how PyArrow streaming, PyTorch neural networks, Gymnasium environments, and FastAPI services connect.
\item \textbf{Chapter 6 (System Testing and Performance Profiling):} Covers automated test suites verifying component correctness and presents microbenchmarks measuring decision latency, streaming throughput, and memory footprint.
\item \textbf{Chapter 7 (Results and Analysis):} Documents experimental findings across all 15 ablation stages, including long-horizon 100k-step runs, Pareto trade-off comparisons, chaos stress tests, and inferential hypothesis testing.
\item \textbf{Chapter 8 (Conclusion and Future Scope):} Summarizes the main results, discusses practical deployment constraints, and outlines directions for future research.
\end{itemize}
"""
