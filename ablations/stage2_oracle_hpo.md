# Ablation Study: Predictive Oracles (Stage 2)

**Hypothesis:** Hardcoding hyperparameter guesses (learning rates, tree depths, hidden dimensions) results in suboptimal predictive accuracy. By utilizing **Optuna** to perform Bayesian Optimization (Hyperparameter Optimization - HPO) across our PyTorch and XGBoost architectures, we can significantly reduce the prediction error (MSE).

## Methodology
We hooked our unified training pipeline (`scripts/train_oracles.py`) into the Optuna framework. The pipeline ingested the `cleaned_trace.parquet` file and executed sequential sweeps over the Oracles:

1. **Workload Predictor (PyTorch LSTM)**
   - *Search Space:* Hidden dimension (32, 64, 128), Layers (1-3), Learning Rate (1e-4 to 1e-2), Dropout (0.1 to 0.5).
   - *Objective:* Minimize Mean Squared Error (MSE) forecasting CPU/Mem burst sequences.

2. **Latency Predictor (XGBoost Tabular)**
   - *Search Space:* Max Depth (3-9), Learning Rate (0.01 to 0.3), N-Estimators (50-200), Subsample (0.5 to 1.0).
   - *Objective:* Minimize MSE forecasting `scheduling_delay`.

## Optuna Sweep Results

### 1. LSTM Workload Predictor
The sweep demonstrated that a deeper, wider network with a relatively aggressive learning rate performed best on the time-series trace data:

| Metric | Best Value |
|--------|------------|
| **Hidden Dimensions** | `128` |
| **LSTM Layers** | `3` |
| **Learning Rate** | `0.00618` |
| **Dropout** | `0.269` |
| **Validation Loss (MSE)** | `0.00295` |

### 2. XGBoost Latency Predictor
The tabular dataset representing static cluster capacity strongly favored a deep tree architecture with a high number of estimators, but a carefully controlled learning rate:

| Metric | Best Value |
|--------|------------|
| **Max Depth** | `7` |
| **N Estimators** | `198` |
| **Learning Rate** | `0.0324` |
| **Subsample** | `0.689` |
| **Validation Loss (MSE)** | `216070.4` |

*(Note: The raw magnitude of XGBoost MSE is tied to the raw seconds of delay, which can peak at 14,000+ seconds in the long-tail).*

### 3. DNN Power Predictor
Because the raw Alibaba trace does not contain instantaneous wattage, we synthesized a target label using standard cloud TDP equations (combining Idle Power, CPU utilization, and GPU overhead). The PyTorch DNN quickly converged:

| Metric | Best Value |
|--------|------------|
| **Hidden Layer Size** | `128` |
| **Depth (Num Layers)** | `3` |
| **Learning Rate** | `0.0012` |
| **Dropout** | `0.261` |
| **Validation Loss (MSE)** | `47087.7` |

## Conclusion
The Hyperparameter Optimization successfully identified non-obvious configurations (like the specific 0.68 subsample ratio for XGBoost) that mathematically outperformed arbitrary baseline guesses. The final trained weights have been cached to the `models/` directory for integration into the Composite-State Builder.
