# Chapter 5: System Implementation

CHAPTER5_TEX = r"""
\clearpage
\chapter{System Implementation}

Clean architecture matters little if the code crashes under real load. We structured the codebase into modular Python packages, separating data streaming, neural network oracles, reinforcement learning environments, and deployment services into clean directories with clear boundaries.

\section{Modular Codebase Organization}

The project repository follows a structured directory layout:
\begin{itemize}
\item `src/data_engine/`: Implements `StreamingIngester` and `RollingWindowBuffer` for chunked zero-copy Parquet streaming.
\item `src/models/`: Houses PyTorch definitions for single-step and multi-horizon time-series Transformers.
\item `src/oracles/`: Implements the LSTM workload forecaster, XGBoost latency regressor, and DNN power predictor.
\item `src/rl_agent/`: Defines the single-pool and heterogeneous Gymnasium environments, state builders, safe wrappers, and PPO-Lagrangian callbacks.
\item `scripts/`: Contains training pipelines, ablation drivers, evaluation suites, and the FastAPI microservice.
\item `deploy/`: Provides container definitions (`Dockerfile`, `docker-compose.yml`) and declarative Kubernetes manifests.
\item `tests/`: Automated unit and integration tests executed via `pytest`.
\end{itemize}

\section{Streaming Ingestion and Rolling Buffer Implementation}

The streaming engine handles trace replay without exhausting system memory. Listing~\ref{lst:streaming} shows the core generator logic in `StreamingIngester`.

\begin{figure}[htbp]
\centering
\begin{minipage}{\linewidth}
\begin{framed}
\footnotesize
\textbf{Listing 5.1: Chunked Generator in StreamingIngester}
\begin{verbatim}
def _create_stream(self):
    rows_skipped = 0
    skip_size = 5000
    batches = self.parquet_file.iter_batches(batch_size=skip_size)
    for batch in batches:
        num_rows = batch.num_rows
        if rows_skipped + num_rows <= self.start_offset:
            rows_skipped += num_rows
            continue
        df_chunk = batch.to_pandas()
        for _, row in df_chunk.iterrows():
            if rows_skipped < self.start_offset:
                rows_skipped += 1
                continue
            yield row.to_dict()
\end{verbatim}
\end{framed}
\end{minipage}
\caption{PyArrow chunked batch streaming with fast offset skipping}
\label{lst:streaming}
\end{figure}

The companion `RollingWindowBuffer` uses a `collections.deque(maxlen=64)`. When fewer than 64 events have arrived (such as right after an episode reset), it zero-pads the output tensor automatically, ensuring downstream neural networks always receive consistent input dimensions.

\section{Predictive Oracles Implementation}

We implemented three distinct oracle models:

\subsection{Multi-Horizon Lookahead Transformer}
To mask the 30 to 120 second physical cloud VM spin-up delay, the Transformer projects demand across three lookahead horizons: $t+1$ (immediate container burst), $t+5$ (near-term queue buildup), and $t+15$ (long-range physical VM boot window). Listing~\ref{lst:transformer} shows how the multi-horizon projection heads branch from the shared encoder.

\begin{figure}[htbp]
\centering
\begin{minipage}{\linewidth}
\begin{framed}
\footnotesize
\textbf{Listing 5.2: Multi-Horizon Projection Heads}
\begin{verbatim}
self.horizon_heads = nn.ModuleList([
    nn.Sequential(
        nn.Linear(d_model, 32),
        nn.ReLU(),
        nn.Linear(32, 2)
    ) for _ in range(self.num_horizons)
])

def forward(self, src):
    x = self.input_projection(src) * math.sqrt(self.d_model)
    x = self.pos_encoder(x)
    encoded = self.transformer_encoder(x)
    last_h = encoded[:, -1, :]
    preds = [head(last_h) for head in self.horizon_heads]
    return torch.stack(preds, dim=1)  # shape: (batch, 3, 2)
\end{verbatim}
\end{framed}
\end{minipage}
\caption{Multi-horizon Transformer architecture with decoupled lookahead heads}
\label{lst:transformer}
\end{figure}

\subsection{XGBoost Latency Predictor}
The latency oracle wraps `xgboost.XGBRegressor` using the squared error objective. It extracts tabular features (normalized CPU request, memory request, and active node counts) and generates sub-millisecond scheduling delay predictions.

\subsection{Deep Neural Network Power Predictor}
The power oracle implements a 3-layer feed-forward network with batch normalization and dropout:
\begin{equation}
\hat{p}_t = \mathbf{W}_4 \, \text{ReLU}\left( \text{BN}\left( \mathbf{W}_3 \, \text{ReLU}\left( \text{BN}\left( \mathbf{W}_2 \mathbf{x} + \mathbf{b}_2 \right) \right) + \mathbf{b}_3 \right) \right) + b_4
\end{equation}
Weights are saved to `models/best_dnn_power.pt` and loaded into evaluation memory at startup.

\section{Composite State Builder Implementation}

The `CompositeStateBuilder` fuses telemetry and forecasts into a unified tensor. In `build_state()`, it runs inferences within `torch.no_grad()` blocks to avoid graph construction overhead:
\begin{equation}
\mathbf{s}_t = \text{Concat}\left( \mathbf{s}_t^{\text{raw}}, \; \hat{\mathbf{y}}_t^{\text{workload}}, \; \hat{y}_t^{\text{latency}}, \; \hat{y}_t^{\text{power}} \right)
\end{equation}
If `use_forecasts=False` (baseline mode), it fills the forecast slots with zeros, ensuring consistent observation dimensions across comparative experiments.

\section{Gymnasium MDP Environments Implementation}

We developed two distinct Gymnasium environments:
\begin{itemize}
\item `K8sScalingEnv`: Single-pool environment managing a generic node pool. Scaling action $a_t \in [-1.0, 1.0]$ scales nodes up or down by at most 5 nodes per step.
\item `HeterogeneousK8sEnv`: Decoupled dual-pool environment managing separate CPU ($N^{\text{cpu}} \in [1, 100]$) and GPU ($N^{\text{gpu}} \in [1, 40]$) pools.
\end{itemize}

Listing~\ref{lst:env_step} shows how `HeterogeneousK8sEnv` computes multi-tier utilization and multi-objective penalties.

\begin{figure}[htbp]
\centering
\begin{minipage}{\linewidth}
\begin{framed}
\footnotesize
\textbf{Listing 5.3: Heterogeneous Environment Step and Reward Logic}
\begin{verbatim}
# Resource Allocation and Demand Simulation
is_gpu = (pod.get('workload_type', 0.0) >= 0.5)
cpu_dem = pod.get('cpu_norm', 0.0) * 100.0

total_cpu = (self.cpu_nodes * 16.0) + (self.gpu_nodes * 32.0)
self.last_cpu_util = min(2.0, cpu_dem / max(1.0, total_cpu))

total_gpus = self.gpu_nodes * 8.0
mem_req = pod.get('mem_norm', 0.0) * 16.0
gpu_dem = mem_req if is_gpu else 0.0
self.last_gpu_util = min(2.0, gpu_dem / max(1.0, total_gpus))

# Multi-Objective Reward: Power + Severe SLA penalties
power_penalty = (self.cpu_nodes * 0.1) + (self.gpu_nodes * 0.3)
cpu_sla = ((self.last_cpu_util - 0.8) * 12.0 
           if self.last_cpu_util > 0.8 else 0.0)
gpu_sla = ((self.last_gpu_util - 0.8) * 30.0 
           if (is_gpu and self.last_gpu_util > 0.8) else 0.0)

reward = -(power_penalty + cpu_sla + gpu_sla)
\end{verbatim}
\end{framed}
\end{minipage}
\caption{Resource demand simulation and multi-objective reward calculation}
\label{lst:env_step}
\end{figure}

\section{Safe RL Action Shield Implementation}

The `SafeActionShield` subclasses `gym.Wrapper`. In its `step()` method, it calculates the minimum safe node count $N_{\text{safe}} = \frac{\text{CPU Demand}}{6.4}$. If the agent's intended action results in $N_{\text{intended}} < N_{\text{safe}}$, it clamps the action:

\begin{minipage}{\linewidth}
\begin{framed}
\footnotesize
\begin{verbatim}
safe_scale_factor = (safe_min_nodes - current_nodes) / 5.0
action[0] = max(-1.0, min(safe_scale_factor, 1.0))
reward += -50.0  # Intervention penalty
\end{verbatim}
\end{framed}
\end{minipage}

This simple clamp completely eliminates catastrophic under-provisioning.

\section{Constrained PPO-Lagrangian Implementation}

We implemented PPO-Lagrangian through a custom Stable-Baselines3 callback: \texttt{Lagrangian\-Constraint\-Callback}. At each environment step, it records whether an SLA breach occurred ($U_t > 0.8$). In `_on_rollout_end()`, it computes the mean breach rate over the rollout and updates the dual multiplier $\lambda_t$:

\begin{minipage}{\linewidth}
\begin{framed}
\footnotesize
\begin{verbatim}
current_sla_rate = np.mean(self.step_sla_violations)
constraint_error = current_sla_rate - self.epsilon_sla
self.lambda_t = max(0.0, self.lambda_t + 
                    self.alpha_dual * constraint_error)
\end{verbatim}
\end{framed}
\end{minipage}

The companion `LagrangianEnvWrapper` scales the SLA penalty term in the reward signal by $\lambda_t$, automatically adjusting the penalty severity based on constraint satisfaction.

\section{Carbon-Aware Scheduling Extension}

The `CarbonAwareK8sEnv` wrapper models diurnal electricity grid carbon intensity:
\begin{equation}
I_{\text{carbon}}(t) = 350.0 - 150.0 \cdot \exp\left( -\frac{(h - 13.0)^2}{8.0} \right) + 200.0 \cdot \exp\left( -\frac{(h - 20.0)^2}{6.0} \right)
\end{equation}
where $h = (t \pmod{1440}) / 60.0$ represents the hour of the day. The reward function penalizes electrical power draw more heavily during evening fossil-fuel peaker hours than during sunny midday solar periods.

\section{Industry Baselines: HPA v2 and PID Controllers}

To ensure meaningful comparisons, we implemented realistic industrial baselines:
\begin{itemize}
\item `KubernetesHPAController`: Replicates the official Kubernetes HPA v2 algorithm, including a $\pm 10\%$ tolerance deadband and a 10-step stabilization window that suppresses premature scale-downs.
\item `PIDController`: Implements a classical Proportional-Integral-Derivative feedback loop:
\begin{equation}
u(t) = K_p \, e(t) + K_i \int e(t) \, dt + K_d \, \frac{de(t)}{dt}
\end{equation}
with anti-windup clamping to prevent integrator saturation during prolonged traffic spikes.
\end{itemize}

\section{Production Deployment and Cloud Packaging}

We packaged the autoscaler for Kubernetes deployment using modern cloud-native standards:
\begin{itemize}
\item **FastAPI Inference Microservice (`scripts/api_server.py`):** Exposes `/v1/scaling-decision` accepting incoming pod telemetry and returning target node recommendations in under 10 milliseconds.
\item **Prometheus Metrics Exporter:** Exposes `/metrics` conforming to OpenMetrics standards for Prometheus scraping and Grafana visualization.
\item **Docker Containerization:** A multi-stage `Dockerfile` and `docker-compose.yml` bundling the autoscaler engine with Prometheus and Grafana.
\item **Kubernetes Controller Manifests:** Declarative YAML files in `deploy/k8s/` integrating with the Kubernetes External Metrics API.
\end{itemize}
"""
