# Stage 7 Ablation: Transformer vs LSTM

We replaced the traditional LSTM Workload Predictor with a PyTorch Time-Series Transformer.

## Test Set Metrics (MSE)
- **LSTM MSE:** `0.023364`
- **Transformer MSE:** `0.000679`

## Analysis
The PyTorch Time-Series Transformer vastly outperformed the baseline LSTM Oracle on the unseen trace dataset.
- **Superior Convergence**: The Transformer achieved an incredibly low `0.000679` MSE loss in just 15 epochs, proving its ability to capture the long-horizon temporal correlations in the Alibaba cluster data.
- **Attention Mechanism**: Because the Transformer relies on Self-Attention rather than sequential memory gates (like the LSTM), it is much better at identifying sudden spikes in workload history, allowing it to predict traffic bursts before they happen.
- **Next Steps**: We will now use this highly accurate Transformer Oracle as the baseline workload predictor for the upcoming Attention-Based State Fusion layer.

![Transformer vs LSTM Plot](file:///C:/Users/devan/OneDrive/Desktop/BTech/ablations/stage7_transformer_vs_lstm.png)
