import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
from src.rl_agent.environment import K8sScalingEnv
from src.rl_agent.safe_wrapper import SafeActionShield

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_trace.parquet")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
ABLATION_DIR = os.path.join(PROJECT_ROOT, "ablations")

def evaluate_agent(model, env, steps=1000):
    obs, _ = env.reset()
    cumulative_reward = 0
    sla_breaches = 0
    interventions = 0
    
    for _ in range(steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        cumulative_reward += reward
        
        # In the unwrapped environment, we get utilization
        util = info.get('utilization', 0.0)
        if util > 0.8:
            sla_breaches += 1
            
        if done:
            break
            
    # Shield interventions are tracked in the wrapper
    if hasattr(env, 'interventions'):
        interventions = env.interventions
            
    return cumulative_reward, sla_breaches, interventions

def train_agent_and_track_safety(use_shield=False, timesteps=30000):
    print(f"\n--- Training PPO (use_shield={use_shield}) ---")
    # Set use_forecasts=False to isolate the effect of the Safe Shield and speed up training massively.
    base_env = K8sScalingEnv(DATA_PATH, MODEL_DIR, use_forecasts=False, use_attention=False, max_steps=10000, start_offset=0)
    
    if use_shield:
        env = SafeActionShield(base_env)
    else:
        env = base_env
        
    model = PPO("MlpPolicy", env, verbose=0, n_steps=2048, batch_size=64, n_epochs=10, ent_coef=0.01)
    
    # Train
    model.learn(total_timesteps=timesteps)
    return model, env

if __name__ == "__main__":
    # 1. Train Standard Agent
    standard_model, standard_env = train_agent_and_track_safety(use_shield=False, timesteps=30000)
    
    # 2. Train Shielded Agent
    shield_model, shield_env = train_agent_and_track_safety(use_shield=True, timesteps=30000)
    
    print("\n--- Evaluating Both Agents (Unseen Test Data) ---")
    test_base_standard = K8sScalingEnv(DATA_PATH, MODEL_DIR, use_forecasts=False, use_attention=False, max_steps=2000, start_offset=100000)
    test_base_shield = K8sScalingEnv(DATA_PATH, MODEL_DIR, use_forecasts=False, use_attention=False, max_steps=2000, start_offset=100000)
    test_shield_env = SafeActionShield(test_base_shield)
    
    std_reward, std_breaches, std_interventions = evaluate_agent(standard_model, test_base_standard, steps=2000)
    shl_reward, shl_breaches, shl_interventions = evaluate_agent(shield_model, test_shield_env, steps=2000)
    
    print(f"Standard Agent -> Reward: {std_reward:.2f} | SLA Breaches: {std_breaches}")
    print(f"Shielded Agent -> Reward: {shl_reward:.2f} | SLA Breaches: {shl_breaches} | Interventions: {shl_interventions}")
    
    # 3. Generate Visualization
    print("Generating Safety Graph...")
    labels = ['Standard PPO', 'Shielded PPO (Safe RL)']
    breaches = [std_breaches, shl_breaches]
    
    plt.figure(figsize=(8, 6))
    colors = ['#ff9999','#66b3ff']
    bars = plt.bar(labels, breaches, color=colors, edgecolor='black')
    
    # Add data labels
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(breaches)*0.01), int(yval), ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    plt.title("SLA Breaches on Unseen Data (Lower is Better)")
    plt.ylabel("Number of Catastrophic Delay Events")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    graph_path = os.path.join(ABLATION_DIR, "stage9_safe_rl_breaches.png")
    plt.savefig(graph_path, dpi=300)
    print(f"Saved graph to {graph_path}")
    
    # 4. Write Markdown Report
    report_path = os.path.join(ABLATION_DIR, "stage9_safe_rl_ablation.md")
    with open(report_path, "w") as f:
        f.write("# Stage 9 Ablation: Safe Reinforcement Learning (Action Shield)\n\n")
        f.write("We wrapped the Gymnasium environment in a `SafeActionShield`. This interceptor calculates the mathematically safe boundary for the cluster node count based on the current Workload demand. If the agent attempts a catastrophic scale-down action that would instantly breach SLA, the shield overwrites the action and heavily penalizes the agent.\n\n")
        f.write("## Test Set Metrics (2000 Steps, Unseen Data)\n")
        f.write(f"- **Standard Agent:** Reward: `{std_reward:.2f}`, SLA Breaches: `{std_breaches}`\n")
        f.write(f"- **Shielded Agent:** Reward: `{shl_reward:.2f}`, SLA Breaches: `{shl_breaches}` (Interventions: `{shl_interventions}`)\n\n")
        f.write("## Analysis\n")
        f.write("*(See graph below)*\n\n")
        f.write(f"![Safety Graph](file:///{graph_path.replace(chr(92), '/')})\n")
    print(f"Report written to {report_path}")
