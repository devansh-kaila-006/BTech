# Stage 4: Custom Gymnasium Cluster Environment (`K8sScalingEnv`)

## Overview
To train deep reinforcement learning agents on real-world cloud traces, we developed `K8sScalingEnv`, a custom, standards-compliant [Gymnasium](https://gymnasium.farama.org/) simulation environment (`src/rl_agent/environment.py`). 

Instead of traditional offline episodic simulators that load synthetic distributions, `K8sScalingEnv` directly couples with the PyArrow Streaming Ingestion Engine to replay real-world microservice and batch workloads from the Alibaba 2023 cluster trace tick-by-tick.

---

## Markov Decision Process (MDP) Formulation

The Kubernetes autoscaling challenge is formulated as a continuous-control Markov Decision Process defined by the tuple $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$:

### 1. State Space ($\mathcal{S}$)
The environment supports both flat concatenated and attention-fused state spaces:
* **Baseline Mode (3D Active / 4D Masked):** Observes only the current raw cluster workload:
  $$\mathbf{s}_t^{\text{raw}} = [\text{cpu\_norm}_t, \text{mem\_norm}_t, \text{workload\_type}_t]$$
* **Predictive Mode (7D Composite):** Observes raw conditions combined with deep learning Oracle forecasts:
  $$\mathbf{s}_t^{\text{composite}} = [\mathbf{s}_t^{\text{raw}} \parallel \hat{y}_t^{\text{LSTM Workload}} \parallel \hat{y}_t^{\text{XGB Latency}} \parallel \hat{y}_t^{\text{DNN Power}}]$$

### 2. Action Space ($\mathcal{A}$)
Continuous Box action $\mathcal{A} = [-1.0, 1.0]$:
$$a_t \in [-1.0, 1.0]$$
The policy outputs a normalized scaling velocity:
$$N_{t+1} = \text{clip}\left(N_t + (a_t \times 5.0), \, N_{\min}=1, \, N_{\max}=100\right)$$
where $N_t$ represents the active node count and the maximum scaling step is clamped to $\pm 5$ nodes per tick to reflect physical VM provisioning bounds.

### 3. Transition Dynamics ($\mathcal{P}$)
At each timestep $t$, the PyArrow ingester pulls the next chronologically ordered pod submission event from `cleaned_trace.parquet`. The rolling window buffer advances by one step with $O(1)$ complexity, updating the temporal context sequence for the LSTM Oracle.

### 4. Reward Formulation ($\mathcal{R}$)
The multi-objective reward balances operational electricity cost against SLA latency guarantees:

$$R_t = - \left( \mathcal{C}_{\text{power}}(N_t) + \mathcal{P}_{\text{SLA}}(U_t) \right)$$

* **Power Consumption Cost:** Linear function of active servers:
  $$\mathcal{C}_{\text{power}}(N_t) = N_t \times 0.1$$
* **SLA Starvation Penalty:** Enforces the industry-standard $80\%$ safe utilization threshold:
  $$\mathcal{P}_{\text{SLA}}(U_t) = \begin{cases} 0, & \text{if } U_t \le 0.8 \\ (U_t - 0.8) \times 10.0, & \text{if } U_t > 0.8 \end{cases}$$
  where $U_t = \frac{\text{CPU Demand}_t}{N_t \times 8.0 \text{ cores}}$.

---

## Architectural Validation
The environment was stress-tested across 10,000 steps without memory leaks, confirming zero frame drops and constant-time step execution ($< 2.5\text{ ms}$ per step including Oracle forward passes).

