# Architecture Document: Composite-State Fusion (Stage 3)

**Goal:** The final RL agent (PPO) cannot achieve "predictive" autoscaling if it only observes the raw, real-time cluster metrics. We must fuse the raw metrics with the forecasts from our pre-trained Oracles (from Stage 2) to create a **Composite State Tensor**.

## Implementation Details

We built the `CompositeStateBuilder` in `src/rl_agent/state_builder.py`. 

### The Loading Mechanism
During initialization, the builder loads the trained weights from the `models/` directory onto the GPU (or CPU) and sets the models to `.eval()` mode:
1. `LSTMOracle` (Workload prediction)
2. `XGBoostOracle` (Latency baseline)
3. `DNNPowerOracle` (Wattage baseline)

### The `build_state()` Pipeline
Every time the RL environment steps forward in time, the `CompositeStateBuilder` triggers:
1. **Raw State Extraction**: Pulls the current normalized CPU, Memory, and Workload Type (`dim=3`).
2. **Oracle Inference**:
   - The `LSTM` ingests the historical sequence tensor and outputs `[pred_cpu, pred_mem]`.
   - The `XGBoost` model ingests the raw state and outputs `[pred_delay]`.
   - The `DNN` model ingests the raw state and outputs `[pred_power]`.
3. **Concatenation**: Fuses all outputs into a single, massive 1D float32 tensor of shape `(7,)`.

### Feature Flag (Ablation Control)
Crucially, the `CompositeStateBuilder` includes a `use_forecasts` boolean flag. 
* If `True` (Predictive-RL): The agent sees the full 7-dimensional composite state.
* If `False` (Standard Baseline RL): The oracle outputs are masked to `[0.0, 0.0, 0.0, 0.0]`. 
This guarantees that during final testing, any performance gain observed in the Predictive-RL model can be mathematically attributed to the Oracle inputs, completely isolating our core engineering contribution.
