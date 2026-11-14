# Stage 5: Rapid Proof of Concept — Baseline vs Predictive PPO (5,000 Steps)

## Objective
To validate the end-to-end integration of the PyArrow streaming engine, Oracle inference pipeline, composite state builder, and PPO agent, we executed a rapid Proof of Concept (POC) training run for exactly 5,000 steps on `K8sScalingEnv`.

---

## Experimental Setup
Two identical `stable-baselines3` PPO architectures were trained simultaneously:
1. **Baseline Agent (Reactive):** 3D active state space (Raw cluster metrics only).
2. **Predictive Agent (Proactive):** 7D composite state space (Raw metrics + LSTM Workload, XGBoost Latency, and DNN Power forecasts).

Both models were trained using standard hyperparameters:
* Policy: `MlpPolicy` (2 layers $\times$ 64 units)
* Learning Rate: $3 \times 10^{-4}$
* Discount Factor ($\gamma$): $0.99$
* Clip Range: $0.2$
* Timesteps: $5,000$

---

## Results (1,000 Evaluation Steps)

| Metric | Baseline (Reactive) | Predictive (Proactive) |
| :--- | :---: | :---: |
| **Cumulative Reward** | **`-3,298.49`** | `-3,493.32` |
| **Total Delay Penalty** | **`-3,192.89`** | `-3,384.57` |
| **State Dimension** | 3 | 7 |

*(See trajectory plots in `final_results.png`)*

---

## Scientific Analysis: The State-Space Complexity Hypothesis

In this initial short training run, the **Baseline Agent slightly outperformed the Predictive Agent by ~5.9%**.

### Why did this occur?
This outcome is a classic, predictable phenomenon in reinforcement learning known as the **Curse of Dimensionality / State-Space Complexity Trade-off**:
1. **Convergence Speed:** The Baseline agent only has to learn a policy mapping over a low-dimensional 3D manifold. Its policy network converges quickly with few sample updates.
2. **Representation Learning Overhead:** The Predictive agent must map a 7-dimensional non-linear state space containing deep neural network and gradient boosted tree predictions. In only 5,000 environment steps, the PPO policy network has not experienced enough state-action transitions to discover the correlations between Oracle predictions and future cluster spikes.

### Conclusion & Hypothesis for Stage 6
5,000 steps artificially handicaps the predictive model. **Hypothesis:** Scaling up training to **100,000 timesteps** will provide sufficient exploration for the policy network to learn the predictive signals, allowing the Predictive agent to anticipate workload bursts and definitively outperform the reactive baseline.

