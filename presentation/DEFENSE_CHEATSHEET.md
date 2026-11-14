# B.Tech Capstone Project Defense Cheatsheet & Rapid-Fire Q&A
**Course:** 22AIE498 — Capstone Project | **Team ID:** `23UG109` | **Panel:** `12`  
**Researchers:** Devansh Kaila, Akshay Damera | **Project Guide:** Dr. Beena  
**Project Title:** *Hierarchical-Action, Streaming-Ingestion Predictive-RL Autoscaling for Heterogeneous GPU Clusters*

---

## 🎯 Section 1: Reinforcement Learning & Control Theory

### Q1: Why did you choose Proximal Policy Optimization (PPO) over off-policy algorithms like SAC or TD3?
**Answer:**
1. **On-Policy Stability:** Cloud autoscaling is safety-critical. Off-policy algorithms like SAC and TD3 use replay buffers containing historical transitions generated under outdated policies. In a non-stationary workload environment (diurnal traffic shifts), stale transitions cause value function overestimation and policy divergence.
2. **Clipped Surrogate Objective:** PPO's clipping mechanism ($r_t(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} \in [1-\epsilon, 1+\epsilon]$) strictly bounds policy updates, preventing catastrophic scale-down decisions during policy iteration.
3. **Continuous Control Synergy:** PPO handles continuous action spaces $[\Delta\text{CPU}, \Delta\text{GPU}] \in [-1.0, 1.0]^2$ reliably with Gaussian action distribution outputs.

---

### Q2: Why did the 3D Baseline outperform the 7D Predictive Agent at 5,000 steps, but lose decisively at 100,000 steps?
**Answer:**
This is the **State-Space Complexity Hypothesis**:
* In low-sample regimes ($T=5,000$), the 3D baseline's policy network only needs to learn a simple mapping across a low-dimensional manifold ($\text{dim}(\mathcal{S})=3$), allowing rapid early convergence.
* The 7D predictive agent must learn to correlate 4 deep neural network predictions with future reward signals. At 5,000 steps, the agent treats predictions as high-variance noise.
* By 100,000 steps, the predictive policy decodes the temporal causal link between forecast peaks and SLA penalties, cutting SLA breaches by **32.8%** and surpassing the baseline's cumulative reward by **+4.7%**.

---

### Q3: Why not just use a tuned industrial PID controller?
**Answer:**
* **Phase Lag & Delay Inability:** A PID controller is inherently reactive. The error $e(t) = \text{util}(t) - \text{util}_{\text{target}}$ is strictly backward-looking. In physical clouds where VM provisioning takes 30–60 seconds, a PID controller cannot act until *after* the cluster is saturated, guaranteeing SLA breaches.
* **Control Thrashing:** The derivative term ($K_d \frac{de}{dt}$) causes severe oscillations during noisy, non-differentiable pod burst spikes, leading to excessive node churn and power wastage.
* **Non-Linear Dynamics:** Kubernetes scaling involves non-linear step costs, bin-packing constraints, and heterogeneous resource couplings that linear feedback controllers cannot model.

---

### Q4: How does the Safe RL Action Shield work mathematically?
**Answer:**
The shield acts as a **runtime control barrier function**:
1. It computes the minimum safe capacity $N_{\text{safe}}$ required to satisfy the $80\%$ SLA utilization bound:
   $$N_{\text{safe}} = \frac{\text{CPU Demand}_t}{8.0 \times 0.8} = \frac{\text{CPU Demand}_t}{6.4}$$
2. Given the agent's proposed action $a_t \in [-1.0, 1.0]$, the tentative node count is $N_{\text{tentative}} = N_t + 5.0 a_t$.
3. If $N_{\text{tentative}} < N_{\text{safe}}$, the shield overrides the action:
   $$a_t^* = \frac{N_{\text{safe}} - N_t}{5.0}$$
4. An intervention penalty ($-50.0$) is added to the reward signal, guiding the policy to learn the safety boundary without human intervention. This guaranteed **0 SLA breaches** in Stage 9.

---

### Q5: What is the mathematical justification for PPO-Lagrangian (Stage 12)?
**Answer:**
Standard RL uses scalar reward weighting: $R_t = -(\text{Power} + \beta \cdot \text{Delay})$. Hand-tuning $\beta$ is fragile—a high $\beta$ causes extreme over-provisioning, while a low $\beta$ violates SLAs.
We formulate the problem as a **Constrained Markov Decision Process (CMDP)**:
$$\min_{\pi_\theta} \; \mathbb{E}[C_{\text{power}}] \quad \text{s.t.} \quad \mathbb{E}[C_{\text{SLA}}] \le \epsilon_{\text{SLA}} = 0.05$$
The Lagrangian dual function is:
$$\mathcal{L}(\theta, \lambda) = J(\theta) - \lambda \left( \mathbb{E}[C_{\text{SLA}}] - \epsilon_{\text{SLA}} \right)$$
Using two-timescale dual gradient ascent:
$$\theta_{k+1} = \theta_k + \alpha_\theta \nabla_\theta \mathcal{L}(\theta_k, \lambda_k)$$
$$\lambda_{k+1} = \max\left(0, \; \lambda_k + \alpha_\lambda (\hat{C}_{\text{SLA}}(\pi_k) - \epsilon_{\text{SLA}})\right)$$
As long as the SLA violation exceeds $5\%$, $\lambda$ automatically increases, forcing the policy to prioritize compliance. Once compliant, $\lambda$ decays to minimize power cost.

---

## ⚡ Section 2: Systems, Data Engineering & Kubernetes

### Q6: How does your PyArrow streaming engine achieve constant memory?
**Answer:**
* Conventional Pandas loads the entire tabular dataset into RAM via memory-mapped C arrays, consuming 7.2 GB+ and crashing edge simulation nodes.
* Our `StreamingIngester` uses `pyarrow.parquet.ParquetFile.iter_batches(batch_size=1024)`. Only the active chunk resides in memory.
* As rows arrive, they are pushed into a `collections.deque(maxlen=64)` which maintains $O(1)$ amortized insertion and eviction time complexity.
* Result: Memory consumption dropped by **84%** (from 6.71 MB down to 1.04 MB in rolling state), with throughput increasing **4.6x** (4,064 rows/sec).

---

### Q7: How does this system deploy into an actual Kubernetes cluster?
**Answer:**
We architected the system to follow the **Kubernetes Custom Metrics API & KEDA architecture**:
1. Our FastAPI service (`scripts/api_server.py`) exposes `/v1/scaling-decision` and `/metrics`.
2. A lightweight Kubernetes Operator or KEDA external scaler polls this endpoint every 5 seconds.
3. The response provides target node counts for both `NodePool-CPU` (via Cluster Autoscaler API) and `Deployment` pod replica counts (via standard Kubernetes client API).
4. Prometheus scrapes `/metrics` via standard Kubernetes service annotations (`prometheus.io/scrape: "true"`).

---

### Q8: How does your multi-horizon transformer solve physical VM cold-boot delays?
**Answer:**
* Spinning up a physical EC2 instance or GKE node takes **30 to 120 seconds** (container runtime init, kernel boot, CNI network attachment).
* A single-step ($t+1$) predictor only gives a 2–4 second warning—useless for physical node boot.
* Our **Multi-Horizon Transformer** simultaneously forecasts three horizons: $[t+1, t+5, t+15]$.
* Horizon $t+15$ corresponds to $\approx 60\text{ seconds}$ into the future. When $t+15$ detects a sustained surge, the cluster autoscaler triggers VM provisioning early so the node enters the `Ready` state exactly as the traffic peak arrives.

---

### Q9: Why decouple CPU and GPU pools instead of scaling total cluster capacity?
**Answer:**
Modern AI workloads exhibit extreme resource heterogeneity:
* **Preprocessing/ETL Pods:** Require 16 CPU cores and 32 GB RAM, zero GPUs.
* **LLM/Vision Inference Pods:** Require high-density NVIDIA A10/T4 GPUs.
* If a cluster autoscaler treats capacity monolithically, a burst of CPU preprocessing pods would trigger the scale-up of GPU nodes ($3.91/hour each), wasting thousands of dollars in idle GPU silicon.
* Decoupling into Pool 0 ($N_{\text{cpu}}$) and Pool 1 ($N_{\text{gpu}}$) with continuous 2D control $[\Delta\text{CPU}, \Delta\text{GPU}]$ allows surgical capacity matching.

---

## 🔬 Section 3: Machine Learning, Oracles & Ablations

### Q10: Why did Multihead Attention underperform Flat Concatenation in Stage 8?
**Answer:**
1. **Sample Inefficiency:** Self-attention computes $\mathcal{O}(N^2)$ dot-product interactions across tokens with dynamic projection matrices ($W_Q, W_K, W_V$). Training these parameters concurrently with an RL policy introduces non-stationarity in the input manifold.
2. **Fixed Dimensionality:** Our state space consists of structured, heterogeneous scalars (utilization, active nodes, forecasts). Unlike natural language, these features have fixed semantic positions. A flat concatenation provides a stationary, linear inductive bias that PPO's MLP policy networks learn in 5x fewer steps.

---

### Q11: What happens if an Oracle produces a false-positive burst prediction?
**Answer:**
1. **Safety Filter:** The RL policy does not blindly scale up based on one feature; the critic network evaluates the value function over the joint state $\mathcal{S}_t \in \mathbb{R}^7$.
2. **Bounded Downside:** If the agent scales up unnecessarily, the power penalty is small ($0.1 \times \Delta N$). However, the Safe RL Shield prevents false scale-downs.
3. **Multi-Model Consensus:** The state vector incorporates three independent oracles (Transformer for workload, XGBoost for delay, DNN for power). A single noisy prediction is regularized by the remaining features.

---

### Q12: What is the computational latency of the state builder and oracles?
**Answer:**
* PyTorch Multi-Horizon Transformer inference: **0.82 ms** (CPU)
* XGBoost Latency Oracle inference: **0.18 ms**
* PyTorch DNN Power Oracle inference: **0.34 ms**
* State concatenation and normalization: **0.12 ms**
* **Total End-to-End Decision Latency:** **< 2.5 ms**, well within the 1,000–5,000 ms control loop period of production cloud schedulers.

---

## 📊 Section 4: Statistical Rigor & Scientific Validity

### Q13: How did you prove that the predictive agent's 32.8% SLA reduction wasn't random seed noise?
**Answer:**
We performed **Paired Inferential Hypothesis Testing** across 6 independent, unseen trace segments (6,000 steps total):
* **Paired Student's t-test:** $p = 0.000412 < 0.001$, decisively rejecting the null hypothesis $H_0$.
* **Wilcoxon Signed-Rank Test:** Confirmed non-parametric rank significance ($p < 0.001$).
* The predictive agent produced superior SLA compliance across **100% of tested trace segments**, proving that proactive foresight provides an invariant structural advantage.

---

### Q14: How does the Carbon-Aware Extension work?
**Answer:**
* We model the regional power grid's diurnal carbon intensity $I(t) \in [180, 650] \text{ gCO}_2/\text{kWh}$ (based on California CAISO / European ENTSO-E grid profiles).
* The reward function integrates dynamic emissions:
  $$R_{\text{carbon}} = - \left( \text{Power}_{\text{kW}} \times I(t) \times \gamma_{\text{carbon}} \right)$$
* During peak renewable generation hours (noon solar peak, low $I(t)$), the agent scales up aggressively to pre-process queues. During fossil-heavy evening peaks, it throttles non-urgent Best-Effort jobs to minimize carbon footprint while strictly honoring Latency-Sensitive SLAs.

---

## 🏆 Summary Punchline for the Viva Panel
> *"We have transformed Kubernetes autoscaling from a backward-looking reactive feedback loop into an anticipatory, constrained-optimal control system. By proving our architecture across 156,000 Alibaba production trace events, eliminating SLA violations via safe action shielding, and wrapping it in an open-source Kubernetes-native microservice, we deliver a publication-grade blueprint for next-generation cloud infrastructure."*

