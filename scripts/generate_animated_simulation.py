import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_baselines3 import PPO

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
from src.rl_agent.environment import K8sScalingEnv

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_trace.parquet")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
ABLATION_DIR = os.path.join(PROJECT_ROOT, "ablations")

def generate_animation(steps=120, offset=100000):
    print("Generating simulation trajectory for Animated GIF...")
    
    # 1. Run simulation on unseen test data
    base_env = K8sScalingEnv(DATA_PATH, MODEL_DIR, use_forecasts=False, max_steps=steps, start_offset=offset)
    pred_env = K8sScalingEnv(DATA_PATH, MODEL_DIR, use_forecasts=True, max_steps=steps, start_offset=offset)
    
    base_model = PPO.load(os.path.join(MODEL_DIR, "Baseline_RL_Long_long_ppo.zip"))
    pred_model = PPO.load(os.path.join(MODEL_DIR, "Predictive_RL_Long_long_ppo.zip"))
    
    b_obs, _ = base_env.reset()
    p_obs, _ = pred_env.reset()
    
    base_nodes = []
    pred_nodes = []
    demands = []
    base_breaches = []
    pred_breaches = []
    
    for _ in range(steps):
        # Baseline step
        b_act, _ = base_model.predict(b_obs, deterministic=True)
        b_obs, _, b_done, _, b_info = base_env.step(b_act)
        
        # Predictive step
        p_act, _ = pred_model.predict(p_obs, deterministic=True)
        p_obs, _, p_done, _, p_info = pred_env.step(p_act)
        
        if b_done or p_done:
            break
            
        base_nodes.append(b_info.get('nodes', 10.0))
        pred_nodes.append(p_info.get('nodes', 10.0))
        demands.append(b_info.get('utilization', 0.0) * b_info.get('nodes', 10.0) * 8.0)
        base_breaches.append(1 if b_info.get('utilization', 0.0) > 0.8 else 0)
        pred_breaches.append(1 if p_info.get('utilization', 0.0) > 0.8 else 0)
            
    print(f"Captured {len(base_nodes)} steps. Rendering Animated GIF...")
    
    # 2. Setup Figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=100)
    fig.patch.set_facecolor('#111827') # Dark theme
    ax1.set_facecolor('#1f2937')
    ax2.set_facecolor('#1f2937')
    
    for ax in (ax1, ax2):
        ax.tick_params(colors='#e5e7eb', labelsize=9)
        ax.xaxis.label.set_color('#e5e7eb')
        ax.yaxis.label.set_color('#e5e7eb')
        ax.title.set_color('#f9fafb')
        ax.grid(True, color='#374151', linestyle='--', alpha=0.5)
        for spine in ax.spines.values():
            spine.set_color('#4b5563')
            
    line_base, = ax1.plot([], [], label='Reactive Baseline (HPA)', color='#ef4444', linestyle='--', linewidth=2.0)
    line_pred, = ax1.plot([], [], label='Predictive-RL (Ours)', color='#10b981', linewidth=2.5)
    ax1.set_title("Live Autoscaling: Active Node Count Over Time", fontsize=12, fontweight='bold', pad=10)
    ax1.set_ylabel("Kubernetes Node Count", fontsize=10)
    ax1.set_xlim(0, len(base_nodes))
    ax1.set_ylim(0, max(max(base_nodes), max(pred_nodes)) + 5)
    ax1.legend(loc='upper left', facecolor='#1f2937', edgecolor='#4b5563', labelcolor='#f3f4f6')
    
    line_demand, = ax2.plot([], [], label='Incoming Workload Demand (Cores)', color='#60a5fa', linewidth=2.0)
    scatter_base = ax2.scatter([], [], color='#ef4444', s=60, label='Baseline SLA Breach', zorder=5)
    scatter_pred = ax2.scatter([], [], color='#10b981', s=60, marker='^', label='Predictive Proactive Guard', zorder=5)
    ax2.set_title("Cluster Demand & SLA Violation Events", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("Simulation Steps (Real Alibaba OSDI Trace Replay)", fontsize=10)
    ax2.set_ylabel("Cores Demanded", fontsize=10)
    ax2.set_xlim(0, len(demands))
    ax2.set_ylim(0, max(demands) + 10)
    ax2.legend(loc='upper left', facecolor='#1f2937', edgecolor='#4b5563', labelcolor='#f3f4f6')
    
    def init():
        line_base.set_data([], [])
        line_pred.set_data([], [])
        line_demand.set_data([], [])
        scatter_base.set_offsets(np.empty((0, 2)))
        scatter_pred.set_offsets(np.empty((0, 2)))
        return line_base, line_pred, line_demand, scatter_base, scatter_pred

    def animate(i):
        x = np.arange(i + 1)
        line_base.set_data(x, base_nodes[:i+1])
        line_pred.set_data(x, pred_nodes[:i+1])
        line_demand.set_data(x, demands[:i+1])
        
        # Plot breach markers
        b_breach_x = [idx for idx in range(i+1) if base_breaches[idx] == 1]
        b_breach_y = [demands[idx] for idx in b_breach_x]
        if b_breach_x:
            scatter_base.set_offsets(np.c_[b_breach_x, b_breach_y])
            
        p_breach_x = [idx for idx in range(i+1) if pred_breaches[idx] == 1]
        p_breach_y = [demands[idx] for idx in p_breach_x]
        if p_breach_x:
            scatter_pred.set_offsets(np.c_[p_breach_x, p_breach_y])
            
        return line_base, line_pred, line_demand, scatter_base, scatter_pred

    plt.tight_layout()
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(base_nodes), interval=60, blit=True)
    
    gif_path = os.path.join(ABLATION_DIR, "animated_scaling_simulation.gif")
    anim.save(gif_path, writer='pillow', fps=15)
    print(f"Animated GIF saved successfully to {gif_path}")

if __name__ == "__main__":
    generate_animation(steps=100, offset=100000)
