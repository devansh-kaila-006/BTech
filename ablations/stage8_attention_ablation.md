# Stage 8 Ablation: Attention-Based State Fusion

We replaced the standard feature concatenation with a PyTorch `MultiheadAttention` layer. This allows the RL agent to dynamically weigh the importance of raw states vs different Oracle predictions.

## Test Set Metrics (Cumulative Reward, 30k Train Steps)
- **Flat Concat (Baseline):** `-3134.67`
- **Attention Fusion:** `-8699.56`

## Analysis
In our short 30,000 step ablation, the **Attention Fusion significantly underperformed** the Flat Concat Baseline.

Why did this happen?
- **Sample Complexity:** Self-Attention mechanisms are notoriously data-hungry. While a Flat Concatenation presents a static, fixed 7D state space to the RL agent, the Attention mechanism dynamically shifts the state representation as its weights update. 
- **Convergence Time:** 30,000 steps is nowhere near enough time for the PPO agent to jointly optimize both the scaling policy *and* the Attention mapping. The Attention weights were likely still shifting wildly, causing the RL agent to experience a non-stationary environment.
- **Conclusion:** While Attention provides theoretical benefits for dynamic weighting, it massively increases the complexity of the state space. To see the benefits of Attention Fusion, the agent would likely need to be trained for **millions of steps** rather than thousands. For short-horizon training, the Flat Concat approach is far superior and stable.

*(See the heatmap below to view the unconverged Attention weights during the evaluation phase)*

![Attention Heatmap](file:///C:/Users/devan/OneDrive/Desktop/BTech/ablations/stage8_attention_heatmap.png)
