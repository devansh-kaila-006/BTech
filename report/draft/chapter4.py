# Chapter 4: System Design

CHAPTER4_TEX = r"""
\clearpage
\chapter{System Design}

A production autoscaler can't rely on simple heuristics or unconstrained neural networks. If an agent has complete freedom to explore arbitrary actions, it will eventually issue a scale-down command right before a traffic peak, dropping cluster capacity and violating agreements. Our architecture pairs predictive machine learning with deterministic safety barriers.

\section{High-Level Architectural Overview}

The architecture splits into six interacting components arranged in a closed feedback pipeline:
\begin{enumerate}
\item \textbf{Data Streaming Engine:} Ingests parquet records tick-by-tick using PyArrow zero-copy iterators and feeds a fixed-size rolling window.
\item \textbf{Predictive Oracles Ensemble:} Runs an ensemble of three specialized models (Transformer, XGBoost, and a deep neural net) to forecast demand, latency, and wattage.
\item \textbf{Composite State Fusion Layer:} Merges instantaneous cluster telemetry with oracle forecasts into an 11-dimensional state representation.
\item \textbf{Hierarchical PPO Policy:} Decides scaling velocities independently for the CPU pool and the GPU pool.
\item \textbf{Safe RL Action Shield:} Evaluates the agent's proposed action against a mathematical capacity invariant, overriding commands that would drop capacity below safe limits.
\item \textbf{Simulated Cluster \& Telemetry Feedback:} Applies safe scaling commands to the simulated cluster, computes multi-objective rewards, and feeds updated states back to the agent.
\end{enumerate}

Figure~\ref{fig:arch_flowchart} shows how data moves through this closed control loop.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.95\linewidth]{00_system_architecture_flowchart.png}
\caption{End-to-end predictive reinforcement learning autoscaling system architecture and closed-loop control flow}
\label{fig:arch_flowchart}
\end{figure}

\section{Streaming Ingestion and Rolling-Window Memory}

Loading entire datasets into memory using Pandas creates severe bottlenecks during reinforcement learning episodes. A single episode may reset dozens of times, and re-reading millions of rows from disk stalls policy iteration.

We solved this with `StreamingIngester` and `RollingWindowBuffer`. 

The streaming ingester uses PyArrow batch iterators (`iter_batches`) with a chunk size of 5,000 records. Only the current chunk stays resident in RAM. As records stream through, the ingester pushes them into a `collections.deque` buffer with a maximum length of $W=64$ entries. Appends and evictions take $O(1)$ amortized time. Memory consumption remains strictly constant regardless of how many steps the episode runs.

Figure~\ref{fig:streaming_deque} illustrates the zero-copy batch streaming process and rolling-window deque eviction.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.95\linewidth]{00_streaming_deque_architecture.png}
\caption{PyArrow zero-copy ingestion and $O(1)$ rolling-window deque buffer architecture with auto zero-padding}
\label{fig:streaming_deque}
\end{figure}

\section{Predictive Oracles Ensemble}

The agent needs to know what's coming before it happens. We built three specialized oracles:

\subsection{Workload Forecasting Oracle}
We developed two alternative sequence models:
\begin{itemize}
\item \textbf{LSTM Oracle:} A 3-layer Long Short-Term Memory network ($d_{\text{hidden}} = 32$) with dropout of 0.2. It takes historical sequences from the rolling buffer and outputs projected CPU and memory demand:
\begin{equation}
[\hat{c}_{t+1}, \hat{m}_{t+1}] = f_{\text{LSTM}}(\mathbf{X}_{t-W:t})
\end{equation}
\item \textbf{Time-Series Transformer:} A 2-layer Transformer encoder ($d_{\text{model}} = 64$, 4 attention heads, feedforward dimension 128) with sinusoidal positional encodings. In Stage 7 ablations, it outperformed the LSTM by reducing test mean squared error to 0.000679.
\end{itemize}

Figure~\ref{fig:multi_horizon_arch} details the internal attention layers and multi-horizon projection heads of our Time-Series Transformer.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.95\linewidth]{00_multi_horizon_transformer_arch.png}
\caption{Multi-horizon lookahead Transformer architecture with decoupled forecasting heads for $t+1$, $t+5$, and $t+15$}
\label{fig:multi_horizon_arch}
\end{figure}

\subsection{Latency Forecasting Oracle}
Scheduling delays don't follow clean linear curves. When a cluster approaches saturation, queuing delays explode non-linearly. We used an Extreme Gradient Boosting (XGBoost) regressor tuned with Optuna (`n_estimators=100`, `max_depth=6`, `learning_rate=0.1`). It maps current pod requests and node capacities to projected scheduling delays in seconds.

\subsection{Power Forecasting Oracle}
Server power draw depends on dynamic processor utilization, memory activity, and thermal design power (TDP) ceilings. We built a deep neural network with 3 hidden layers (32 units each), batch normalization, and ReLU activations to map node utilization features to continuous electrical wattage.

\section{Composite State Fusion Architecture}

The agent needs a clear view of both current conditions and oracle projections. The state fusion layer merges these signals:
\begin{equation}
\mathbf{s}_t^{\text{composite}} = \left[ \mathbf{s}_t^{\text{raw}} \,\|\, \hat{\mathbf{y}}_t^{\text{workload}} \,\|\, \hat{y}_t^{\text{latency}} \,\|\, \hat{y}_t^{\text{power}} \right] \in \mathbb{R}^7
\end{equation}
For single-pool environments, this produces a 7-dimensional vector. For heterogeneous environments, we append cluster fleet metrics (active CPU nodes, active GPU nodes, CPU utilization, and GPU utilization), yielding an 11-dimensional observation vector.

In Stage 8, we also evaluated a dynamic Multi-Head Attention Fusion layer. While self-attention across state tokens is theoretically appealing, our empirical tests showed that flat concatenation provides a much more stable target during early PPO training.

\section{Hierarchical Heterogeneous Control}

Real enterprise datacenters don't run single-node-type clusters. They divide infrastructure into distinct compute tiers. Our hierarchical controller decouples actuation into two independent scaling channels:
\begin{itemize}
\item \textbf{CPU-Optimized Pool ($N^{\text{cpu}} \in [1, 100]$):} Standard 16-core servers ($c6i.4xlarge$ equivalent) providing bulk CPU throughput.
\item \textbf{GPU-Accelerated Pool ($N^{\text{gpu}} \in [1, 40]$):} High-density accelerator instances carrying 8 GPUs and 96 CPU cores ($g4dn.12xlarge$ equivalent).
\end{itemize}

Figure~\ref{fig:hierarchical_control_arch} depicts the decoupled multi-channel scaling policy.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.95\linewidth]{00_hierarchical_control_diagram.png}
\caption{Decoupled hierarchical control architecture managing CPU-optimized and GPU-accelerated server tiers}
\label{fig:hierarchical_control_arch}
\end{figure}

The policy network outputs a continuous 2D action vector:
\begin{equation}
\mathbf{a}_t = [a_t^{\text{cpu}}, a_t^{\text{gpu}}] \in [-1.0, 1.0]^2
\end{equation}
Scaling velocity applies as:
\begin{align}
N_{t+1}^{\text{cpu}} &= \text{clip}\left( N_t^{\text{cpu}} + 5.0 \cdot a_t^{\text{cpu}}, \; 1.0, \; 100.0 \right) \\
N_{t+1}^{\text{gpu}} &= \text{clip}\left( N_t^{\text{gpu}} + 2.0 \cdot a_t^{\text{gpu}}, \; 1.0, \; 40.0 \right)
\end{align}
This decoupling stops the agent from making expensive mistakes, like scaling up power-hungry GPU instances when only lightweight CPU microservices are spiking.

\section{Safe RL Action Shield}

Standard reinforcement learning agents learn by making mistakes. In production cloud autoscaling, you can't afford mistakes that take down services.

We placed a runtime `SafeActionShield` between the policy and the cluster environment. The shield enforces a mathematical control barrier function:
\begin{equation}
N_{\text{safe}} = \frac{\text{CPU Demand}_t}{8.0 \times 0.8} = \frac{\text{CPU Demand}_t}{6.4}
\end{equation}
If the agent's proposed action would push capacity below $N_{\text{safe}}$, the shield intercepts the command:
\begin{equation}
a_t^* = \frac{N_{\text{safe}} - N_t}{5.0}
\end{equation}
The shield executes the safe action $a_t^*$ instead of the agent's unsafe command. It also injects a $-50.0$ penalty into the reward signal, teaching the policy to respect the safety boundary naturally over time.

Figure~\ref{fig:safe_shield_flowchart} diagrams the control barrier evaluation and clamping workflow.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.88\linewidth]{00_safe_shield_flowchart.png}
\caption{Safe RL Action Shield control barrier verification and clamping decision logic}
\label{fig:safe_shield_flowchart}
\end{figure}

\section{Constrained Policy Optimization via PPO-Lagrangian}

Balancing power savings against SLA violations using fixed reward weights is notoriously fragile. If the SLA penalty weight is set too low, the agent under-provisions to save energy. If it's set too high, the agent keeps dozens of idle servers running around the clock.

We formulated the problem as a Constrained MDP and solved it using PPO-Lagrangian. The dual update rule adjusts the Lagrange multiplier $\lambda_k$ at the end of each rollout:
\begin{equation}
\lambda_{k+1} = \max\left(0, \; \lambda_k + \alpha_\lambda \left( \hat{C}_{\text{SLA}}(\pi_k) - \epsilon_{\text{SLA}} \right)\right)
\end{equation}
where $\alpha_\lambda = 0.05$ is the dual learning rate and $\epsilon_{\text{SLA}} = 0.05$ is the target breach budget. When breaches exceed 5\%, $\lambda$ climbs automatically, pushing the policy back into compliance. Once compliant, $\lambda$ decays, letting the agent prioritize power reduction.

Figure~\ref{fig:cmdp_flowchart} illustrates this two-timescale dual gradient ascent optimization process.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.95\linewidth]{00_cmdp_lagrangian_flowchart.png}
\caption{PPO-Lagrangian constrained optimization dynamics and dual gradient ascent update loop}
\label{fig:cmdp_flowchart}
\end{figure}

\section{Cloud-Native Production Deployment Architecture}

To make the system deployable in real Kubernetes clusters, we wrapped the policy in a production microservice:
\begin{itemize}
\item \textbf{FastAPI REST Server:} Exposes `/v1/scaling-decision` with sub-10ms response times and `/health` diagnostic endpoints.
\item \textbf{Prometheus Exporter:} Serves `/metrics` in OpenMetrics format, exporting recommended node counts, instantaneous wattage, and cumulative SLA breaches.
\item \textbf{Kubernetes Custom Metrics \& KEDA:} Allows standard Kubernetes Horizontal Pod Autoscalers or KEDA external scalers to consume our recommendations directly.
\end{itemize}

Figure~\ref{fig:cloud_deployment_topo} details the Kubernetes control plane integration and custom metrics topology.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.95\linewidth]{00_production_k8s_deployment.png}
\caption{Production Kubernetes cloud deployment topology with KEDA external scaler and Prometheus observability}
\label{fig:cloud_deployment_topo}
\end{figure}
"""
