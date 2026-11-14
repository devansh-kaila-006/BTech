import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
from src.rl_agent.environment import K8sScalingEnv

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_trace.parquet")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
ABLATION_DIR = os.path.join(PROJECT_ROOT, "ablations")

def evaluate_model(model_name, use_forecasts, steps=5000, offset=100000):
    print(f"\nEvaluating {model_name} on unseen test data (offset={offset})...")
    # We use start_offset to ensure the agent is tested on data it never saw during training
    env = K8sScalingEnv(DATA_PATH, MODEL_DIR, use_forecasts=use_forecasts, max_steps=steps, start_offset=offset)
    model_path = os.path.join(MODEL_DIR, f"{model_name}.zip")
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find model at {model_path}")
        return None
        
    model = PPO.load(model_path)
    
    obs, _ = env.reset()
    
    metrics = {
        'nodes': [],
        'utilization': [],
        'cumulative_reward': [0],
        'reward': [],
        'delay_penalty': [],
        'power_penalty': [],
        'sla_breaches': 0,
        'max_delay_penalty': 0
    }
    
    cumulative_reward = 0
    for _ in range(steps):
        # Deterministic evaluation (no random exploration)
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        
        if done:
            break
            
        cumulative_reward += reward
        metrics['nodes'].append(info['nodes'])
        metrics['utilization'].append(info['utilization'])
        metrics['cumulative_reward'].append(cumulative_reward)
        metrics['reward'].append(reward)
        
        delay_pen = info['reward_delay']
        power_pen = info['reward_power']
        
        metrics['delay_penalty'].append(delay_pen)
        metrics['power_penalty'].append(power_pen)
        
        if delay_pen < 0: # Note: reward_delay is a negative number
            metrics['sla_breaches'] += 1
            metrics['max_delay_penalty'] = min(metrics['max_delay_penalty'], delay_pen)
            
    print(f"Final Cumulative Reward for {model_name}: {cumulative_reward:.2f}")
    return metrics

if __name__ == "__main__":
    # Test the Long models trained for 100k steps
    baseline_metrics = evaluate_model("Baseline_RL_Long_long_ppo", use_forecasts=False, steps=5000, offset=100000)
    predictive_metrics = evaluate_model("Predictive_RL_Long_long_ppo", use_forecasts=True, steps=5000, offset=100000)
    
    if baseline_metrics and predictive_metrics:
        print("Generating Deep Evaluation Graphs...")
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 16))
        
        # Plot 1: Cumulative Reward
        axes[0].plot(baseline_metrics['cumulative_reward'], label="Baseline (100k)", color="red", linestyle="--")
        axes[0].plot(predictive_metrics['cumulative_reward'], label="Predictive (100k)", color="green")
        axes[0].set_title("Cumulative RL Reward on Unseen Data (Higher is Better)")
        axes[0].set_ylabel("Total Reward")
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot 2: Delay Penalties (SLA Breaches)
        axes[1].plot(baseline_metrics['delay_penalty'], label="Baseline SLA Penalty", color="red", alpha=0.6)
        axes[1].plot(predictive_metrics['delay_penalty'], label="Predictive SLA Penalty", color="green", alpha=0.6)
        axes[1].set_title("SLA Delay Penalties Over Time (Closer to 0 is Better)")
        axes[1].set_ylabel("Penalty")
        axes[1].legend()
        axes[1].grid(True)
        
        # Plot 3: Power Penalties
        axes[2].plot(baseline_metrics['power_penalty'], label="Baseline Power Cost", color="red", alpha=0.6)
        axes[2].plot(predictive_metrics['power_penalty'], label="Predictive Power Cost", color="green", alpha=0.6)
        axes[2].set_title("Power Cost Over Time (Closer to 0 is Better)")
        axes[2].set_ylabel("Penalty")
        axes[2].legend()
        axes[2].grid(True)
        
        # Plot 4: Number of Nodes
        axes[3].plot(baseline_metrics['nodes'], label="Baseline Nodes", color="red", alpha=0.8)
        axes[3].plot(predictive_metrics['nodes'], label="Predictive Nodes", color="green", alpha=0.8)
        axes[3].set_title("Active Kubernetes Nodes Over Time")
        axes[3].set_xlabel("Time Steps")
        axes[3].set_ylabel("Node Count")
        axes[3].legend()
        axes[3].grid(True)
        
        plt.tight_layout()
        save_path = os.path.join(ABLATION_DIR, "final_long_results.png")
        plt.savefig(save_path, dpi=300)
        print(f"Graph saved to {save_path}")
        
        # Save metrics to text file for markdown
        with open(os.path.join(ABLATION_DIR, "final_long_metrics.txt"), "w") as f:
            f.write(f"--- BASELINE (100k Steps) ---\n")
            f.write(f"Cumulative Reward: {baseline_metrics['cumulative_reward'][-1]:.2f}\n")
            f.write(f"SLA Breach Count: {baseline_metrics['sla_breaches']}\n")
            f.write(f"Max Delay Penalty: {baseline_metrics['max_delay_penalty']:.2f}\n")
            f.write(f"Average Nodes: {np.mean(baseline_metrics['nodes']):.2f}\n\n")
            
            f.write(f"--- PREDICTIVE (100k Steps) ---\n")
            f.write(f"Cumulative Reward: {predictive_metrics['cumulative_reward'][-1]:.2f}\n")
            f.write(f"SLA Breach Count: {predictive_metrics['sla_breaches']}\n")
            f.write(f"Max Delay Penalty: {predictive_metrics['max_delay_penalty']:.2f}\n")
            f.write(f"Average Nodes: {np.mean(predictive_metrics['nodes']):.2f}\n")
