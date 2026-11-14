import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
from src.rl_agent.environment import K8sScalingEnv

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_trace.parquet")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
ABLATION_DIR = os.path.join(PROJECT_ROOT, "ablations")

def train_agent(use_attention, timesteps=30000):
    print(f"\n--- Training PPO (use_attention={use_attention}) ---")
    # Use offset 0 for training
    env = K8sScalingEnv(DATA_PATH, MODEL_DIR, use_forecasts=True, use_attention=use_attention, max_steps=5000, start_offset=0)
    
    model = PPO("MlpPolicy", env, verbose=0, n_steps=2048, batch_size=64, n_epochs=10, ent_coef=0.01)
    
    # We will manually track rewards by running evaluate every 10k steps
    # But for a simple ablation, we can just train and then evaluate
    model.learn(total_timesteps=timesteps)
    return model, env

def evaluate_agent(model, env, steps=1000):
    obs, _ = env.reset()
    cumulative_reward = 0
    attention_weights_history = []
    
    for _ in range(steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        cumulative_reward += reward
        
        if env.use_attention and hasattr(env.state_builder, 'last_attention_weights') and env.state_builder.last_attention_weights is not None:
            # last_attention_weights shape is (batch, num_heads, seq_len, seq_len) -> (1, 4, 4, 4)
            # Or it might be (batch, target_len, source_len) depending on PyTorch version default return
            weights = env.state_builder.last_attention_weights
            if len(weights.shape) == 3: # (batch, target, source)
                mean_weights = weights[0].mean(axis=0) # average across targets -> (4,)
            else:
                # If MultiHeadAttention returns averaged over heads: (batch, target, source)
                mean_weights = weights[0].mean(axis=0)
            attention_weights_history.append(mean_weights)
            
        if done:
            break
            
    return cumulative_reward, attention_weights_history

if __name__ == "__main__":
    # 1. Train Flat Concat
    flat_model, flat_env = train_agent(use_attention=False, timesteps=30000)
    
    # 2. Train Attention Fusion
    attn_model, attn_env = train_agent(use_attention=True, timesteps=30000)
    
    print("\n--- Evaluating Both Agents (Unseen Test Data) ---")
    test_flat_env = K8sScalingEnv(DATA_PATH, MODEL_DIR, use_forecasts=True, use_attention=False, max_steps=1000, start_offset=100000)
    test_attn_env = K8sScalingEnv(DATA_PATH, MODEL_DIR, use_forecasts=True, use_attention=True, max_steps=1000, start_offset=100000)
    
    flat_reward, _ = evaluate_agent(flat_model, test_flat_env, steps=1000)
    attn_reward, attn_history = evaluate_agent(attn_model, test_attn_env, steps=1000)
    
    print(f"Flat Concat Reward: {flat_reward:.2f}")
    print(f"Attention Fusion Reward: {attn_reward:.2f}")
    
    # 3. Generate Attention Heatmap
    if len(attn_history) > 0:
        print("Generating Attention Heatmap...")
        # attn_history is a list of arrays of shape (4,) representing attention to [Base, Workload, Latency, Power]
        attn_matrix = np.array(attn_history) # Shape: (steps, 4)
        
        # Take a 100-step slice for the heatmap to make it readable
        slice_idx = min(200, len(attn_matrix))
        attn_slice = attn_matrix[:slice_idx].T # Shape: (4, steps)
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(attn_slice, cmap="YlGnBu", yticklabels=["Base State", "Workload Oracle", "Latency Oracle", "Power Oracle"])
        plt.title("Self-Attention Weights Over Time (First 200 Steps)")
        plt.xlabel("Simulation Step")
        plt.ylabel("State Component")
        plt.tight_layout()
        
        heatmap_path = os.path.join(ABLATION_DIR, "stage8_attention_heatmap.png")
        plt.savefig(heatmap_path, dpi=300)
        print(f"Saved heatmap to {heatmap_path}")
        
        # 4. Write Markdown Report
        report_path = os.path.join(ABLATION_DIR, "stage8_attention_ablation.md")
        with open(report_path, "w") as f:
            f.write("# Stage 8 Ablation: Attention-Based State Fusion\n\n")
            f.write("We replaced the standard feature concatenation with a PyTorch `MultiheadAttention` layer. This allows the RL agent to dynamically weigh the importance of raw states vs different Oracle predictions.\n\n")
            f.write("## Test Set Metrics (Cumulative Reward, 30k Train Steps)\n")
            f.write(f"- **Flat Concat (Baseline):** `{flat_reward:.2f}`\n")
            f.write(f"- **Attention Fusion:** `{attn_reward:.2f}`\n\n")
            f.write("## Analysis\n")
            f.write("*(See the heatmap to understand which Oracles the agent relies on during workload fluctuations)*\n\n")
            f.write(f"![Attention Heatmap](file:///{heatmap_path.replace(chr(92), '/')})\n")
        print(f"Report written to {report_path}")
    else:
        print("Failed to capture attention weights.")
