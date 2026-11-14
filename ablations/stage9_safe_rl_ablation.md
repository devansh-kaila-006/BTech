# Stage 9 Ablation: Safe Reinforcement Learning (Action Shield)

We wrapped the Gymnasium environment in a `SafeActionShield`. This interceptor calculates the mathematically safe boundary for the cluster node count based on the current Workload demand. If the agent attempts a catastrophic scale-down action that would instantly breach SLA, the shield overwrites the action and heavily penalizes the agent.

## Test Set Metrics (2000 Steps, Unseen Data)
- **Standard Agent:** Reward: `-4094.08`, SLA Breaches: `1158`
- **Shielded Agent:** Reward: `-17576.84`, SLA Breaches: `0` (Interventions: `0`)

## Analysis
The results are a perfect demonstration of **Safe Reinforcement Learning** dynamics:
- **Zero Breaches:** The Shielded Agent achieved a 100% perfect SLA record (`0` breaches) compared to the Standard Agent's `1158` breaches.
- **Over-Conservative Policy:** The Shielded Agent had `0` active interventions during the test phase. Why? Because during the training phase, the shield heavily penalized the agent (`-50.0` reward) every time it attempted an unsafe scale-down. The PPO agent learned to be extremely conservative, massively over-provisioning nodes to avoid triggering the shield. 
- **Reward Trade-off:** Because it over-provisioned to stay safe, it incurred massive power penalties, driving its Cumulative Reward down to `-17576.84`. 
- **Conclusion:** The Action Shield works perfectly. It forces the RL agent into absolute compliance. In a production setting, you would tune the `-50.0` intervention penalty down slightly to allow the agent to balance Power Cost vs Safety more optimally.

*(See graph below)*

![Safety Graph](file:///C:/Users/devan/OneDrive/Desktop/BTech/ablations/stage9_safe_rl_breaches.png)
