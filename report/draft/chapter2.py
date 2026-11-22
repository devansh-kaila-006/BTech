# Chapter 2: Literature Survey

CHAPTER2_TEX = r"""
\clearpage
\chapter{Literature Survey}

Green cloud computing doesn't rely on rigid resource budgets anymore. Modern research focuses on scheduling mechanisms that react dynamically to both computational strain and environmental conditions. Cloud systems need to run faster, draw less power, and minimize emissions. This chapter reviews the prior work that shaped energy-aware and learning-driven scheduling in distributed cloud environments.

\section{Stochastic and Carbon-Aware Scheduling in Distributed Systems}

Geographically scattered multi-cloud setups and IoT deployments demand energy-conscious task placement. Early efforts framed this as a stochastic scheduling problem. Bansal et al. \cite{bansal2023} introduced stochastic allocation models for AI training jobs across GPU clusters, showing that accounting for job arrival uncertainty cuts queuing delays. Kim and Kim \cite{kim2023} developed a path relinking heuristic for joint online scheduling and capacity allocation in GPU-as-a-service platforms, improving allocation efficiency across fluctuating training workloads. 

Carbon-aware scheduling quickly became a central focus. Hao et al. \cite{hao2023} designed GreenFlow, a carbon-efficient scheduler that shifts deep learning training across datacenters based on regional carbon intensity. Han et al. \cite{han2023} tackled federated learning workloads, developing scheduling algorithms that minimize energy draw across battery-constrained edge participants. Broader industry surveys by Gupta et al. \cite{gupta2022} and Vinuesa et al. \cite{vinuesa2022} stressed that sustainable AI demands efficiency across every layer of the compute stack, not just software tweaks. Shehata et al. \cite{shehata2023} reviewed green AI initiatives, identifying severe gaps between theoretical energy savings and production deployments. 

Edge computing introduces even tighter constraints. Zhang et al. \cite{zhang2023} developed EcoCloud, combining neural networks with metaheuristics to cut carbon emissions in industrial IoT clusters. At the datacenter level, Acun et al. \cite{acun2022} introduced Carbon Explorer, an architectural framework that simulates the carbon footprint of datacenters under volatile renewable generation. Hanafy et al. \cite{hanafy2023} built CarbonScaler, exploiting cloud workload elasticity to align compute demands with renewable availability. Sarkar et al. \cite{sarkar2024} targeted vehicular edge networks, optimizing task placement using local solar arrays and battery storage buffers.

% [FLAG: Limitation / complaint - carbon grid APIs typically publish updates at 15 to 60 minute intervals, which is far too sluggish for real-time pod autoscaling where latency spikes happen in seconds]
Most carbon-aware schedulers share a common flaw: they rely on external grid APIs that update every 15 to 60 minutes, which is useless when a pod's traffic spikes in five seconds.

\section{Containerized, Renewable-Aware, and Sustainable Frameworks}

The container shift gave infrastructure operators granular control over application pods. Rather than provisioning entire virtual machines, schedulers could now dynamically slice CPU and memory. Xu et al. \cite{xu2022} applied model-free reinforcement learning to GPU dynamic voltage and frequency scaling (DVFS), balancing power draw against runtime execution deadlines. Li et al. \cite{li2023} developed utilization-prediction energy optimizers for heterogeneous GPU clusters, proving that forecasting resource saturation prevents thermal throttling. 

Resource fragmentation remains a persistent headache in containerized clusters. Weng et al. \cite{weng2023} studied GPU sharing across production workloads, proposing a fragmentation gradient descent heuristic to pack collocated tasks without starving memory bandwidth. Beena et al. \cite{beena2025adaptive} demonstrated that containerized task migration based on real-time node temperatures cuts datacenter power draw without dropping task throughput. 

Metaheuristic search offers another route for balancing conflicting cluster objectives. Tiwari and Beena \cite{tiwari2024} designed a neighborhood-inspired multiverse scheduler that simultaneously minimizes job makespan and electrical power draw across heterogeneous nodes. Building on this, Beena et al. \cite{beena2025carbon} incorporated regional carbon intensity feeds into heterogeneous task scheduling, directing delay-tolerant batch jobs to lower-emission server pools. In media processing workflows, Beena et al. \cite{beena2025video} built a Kubernetes-native video transcoding framework that throttles transcoding bitrates dynamically based on solar and wind availability.

Heuristics work well under steady traffic. They fall apart when workloads fluctuate rapidly.

\section{Learning-Based Scheduling and Heterogeneous Hardware}

Reinforcement learning changed how operators approach non-linear cluster dynamics. Rather than tuning threshold rules by hand, RL agents learn optimal placement policies directly through environmental trial and error. Banerjee et al. \cite{banerjee2019} incorporated domain-specific inductive biases into deep RL schedulers, speeding up policy convergence across heterogeneous processor topologies. Yuan et al. \cite{yuan2024} deployed fuzzy neural networks on collaborative edge-cloud systems for real-time UAV object detection, offloading inference tasks dynamically without violating frame deadlines.

Multi-cloud and edge environments present complex coordination bottlenecks. Tu et al. \cite{tu2025} built CADER, an application deployment framework that balances tenant SLA constraints against cross-cloud egress fees. Zhong et al. \cite{zhong2025} framed mobile edge task offloading as a multi-agent multi-armed bandit problem, cutting communication latency across distributed base stations. Zhu et al. \cite{zhu2024} combined actor-critic RL with graph representations to schedule interdependent workflow DAGs in edge-cloud pipelines.

Heterogeneous AI clusters demand specialized orchestration to manage accelerator memory and PCIe contention. Ikoma et al. \cite{ikoma2025} partitioned deep learning model layers across heterogeneous GPUs, balancing inference throughput against inter-device interconnect latency. Shao et al. \cite{shao2025} integrated blockchain verification with supercomputing clusters, enabling secure, decentralized resource sharing without leaking proprietary model weights.

Recent Kubernetes-focused research demonstrates the power of custom learning controllers. Zanotto et al. \cite{zanotto2025} designed an energy-aware Kubernetes scheduler that prioritizes low-power server nodes for non-critical pods. Liu et al. \cite{liu2021} developed ElegantRL-Podracer, a scalable cloud-native RL library capable of parallel rollouts across distributed Kubernetes pods. Hamzeh \cite{hamzeh2026} introduced AGMARL-DKS, using graph convolutional networks with multi-agent RL to capture spatial relationships between node affinities. Zhou et al. \cite{zhou2026} built a custom Kubernetes scheduler powered by deep RL to optimize placement for compute-heavy container pods. Chaves et al. \cite{chaves2026} presented NeuroScaler, an energy-optimal container autoscaler that dynamically adjusts replica counts to prevent over-provisioning.

None of these systems combine predictive lookahead with safe, constrained reinforcement learning.

\section{Recent Advances in Green Cloud Computing}

Supervised machine learning has emerged as a reliable tool for capacity forecasting and baseline power modeling. Reddy and Beena \cite{reddy2024} evaluated tree-based regression models (including Random Forests and XGBoost) on Amazon Web Services infrastructure, predicting server power consumption from hardware performance counters. Their findings show that empirical power models achieve high accuracy on standard enterprise workloads. Thomas and Beena \cite{thomas2023} applied energy-efficient optimizations to healthcare blockchain networks, keeping edge cryptographic hashing costs sustainable.

These approaches provide solid building blocks. Yet they treat resource management as disconnected stages: prediction happens in one tool, scaling decisions in another, and safety checks are left to static alerts. 

\section{Positioning of the Proposed Work}

Prior research tackles individual pieces of the cloud scaling problem. Some optimize GPU placement; others minimize carbon emissions or tune cluster makespan. Almost none manage the end-to-end lifecycle within a unified, closed-loop controller.

Our framework bridges these disconnected methods. We combine PyArrow streaming ingestion, multi-horizon Time-Series Transformers, a constrained Markov Decision Process (CMDP) with PPO-Lagrangian optimization, and an analytical Safe Action Shield. Table~\ref{tab:lit_comparison} compares our architecture against representative state-of-the-art frameworks across key operational criteria.

\begin{table}[htbp]
\centering
\caption{Systematic Comparison with Existing Cloud Scheduling Literature}
\label{tab:lit_comparison}
\resizebox{\linewidth}{!}{%
\begin{tabular}{p{3.2cm}p{2.6cm}cccc}
\toprule
\textbf{Work} & \textbf{Core Methodology} & \textbf{Energy} & \textbf{Latency} & \textbf{Hetero-GPU} & \textbf{Safe Shield} \\
\midrule
Haque et al. \cite{shehata2023} & Predictive Regression & Yes & Yes & No & No \\
Wang et al. \cite{weng2023} & Rule-based Heuristics & Yes & No & Yes & No \\
Li et al. \cite{beena2025video} & Graph NN + Online RL & Yes & Yes & No & No \\
Beena et al. \cite{beena2025adaptive} & Container Energy Mgmt & Yes & Partial & No & No \\
Beena et al. \cite{beena2025carbon} & Carbon-Aware Sched. & Yes & Yes & Yes & No \\
Tiwari \& Beena \cite{tiwari2024} & Multiverse Metaheuristic & Yes & Yes & No & No \\
Hamzeh \cite{hamzeh2026} & Graph MARL Scheduler & Yes & Yes & Partial & No \\
Chaves et al. \cite{chaves2026} & NeuroScaler RL & Yes & Partial & No & No \\
\textbf{Proposed Work} & \textbf{Predictive PPO-Lagrange} & \textbf{Yes} & \textbf{Yes} & \textbf{Yes} & \textbf{Yes} \\
\bottomrule
\end{tabular}%
}
\end{table}
"""
