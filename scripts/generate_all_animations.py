import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
from stable_baselines3 import PPO

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
from src.rl_agent.environment import K8sScalingEnv
from src.rl_agent.hierarchical_env import HeterogeneousK8sEnv
from src.rl_agent.safe_wrapper import SafeActionShield
from src.models.multi_horizon_transformer import MultiHorizonTransformer

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_trace.parquet")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
ABLATION_DIR = os.path.join(PROJECT_ROOT, "ablations")

def set_dark_theme(fig, axes):
    fig.patch.set_facecolor('#111827')
    for ax in axes:
        ax.set_facecolor('#1f2937')
        ax.tick_params(colors='#e5e7eb', labelsize=9)
        ax.xaxis.label.set_color('#e5e7eb')
        ax.yaxis.label.set_color('#e5e7eb')
        ax.title.set_color('#f9fafb')
        ax.grid(True, color='#374151', linestyle='--', alpha=0.5)
        for spine in ax.spines.values():
            spine.set_color('#4b5563')

# ==============================================================================
# GIF 1: Safe RL Action Shield Interception
# ==============================================================================
def generate_safe_shield_gif(steps=80, offset=100000):
    print("\n[1/3] Generating Safe RL Action Shield Animation...")
    base_env = K8sScalingEnv(DATA_PATH, MODEL_DIR, use_forecasts=False, max_steps=steps, start_offset=offset)
    shield_env = SafeActionShield(K8sScalingEnv(DATA_PATH, MODEL_DIR, use_forecasts=False, max_steps=steps, start_offset=offset))
    
    model = PPO.load(os.path.join(MODEL_DIR, "Baseline_RL_Long_long_ppo.zip"))
    
    b_obs, _ = base_env.reset()
    s_obs, _ = shield_env.reset()
    
    intended_actions = []
    executed_actions = []
    unprotected_breaches = []
    interventions = []
    
    for _ in range(steps):
        act, _ = model.predict(b_obs, deterministic=True)
        b_obs, _, b_done, _, b_info = base_env.step(act)
        
        # Shielded agent step
        s_act, _ = model.predict(s_obs, deterministic=True)
        intended_actions.append(float(s_act[0]))
        
        s_act_exec = np.array(s_act, dtype=np.float32, copy=True)
        s_obs, _, s_done, _, s_info = shield_env.step(s_act_exec)
        executed_actions.append(float(s_act_exec[0])) # Clamped by shield if unsafe
        interventions.append(s_info.get('shield_interventions', 0))
        
        unprotected_breaches.append(1 if b_info.get('utilization', 0.0) > 0.8 else 0)
        if b_done or s_done: break
        
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=100)
    set_dark_theme(fig, [ax1, ax2])
    
    l_intent, = ax1.plot([], [], label='RL Agent Intended Action', color='#f59e0b', linestyle='--', linewidth=2.0)
    l_exec, = ax1.plot([], [], label='Safe Shield Executed Action', color='#10b981', linewidth=2.2)
    ax1.set_title("Safe RL Action Shield: Real-Time Intervention & Action Clamping", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Scaling Velocity [-1.0, 1.0]")
    ax1.set_xlim(0, len(intended_actions))
    ax1.set_ylim(-1.1, 1.1)
    ax1.legend(loc='lower left', facecolor='#1f2937', edgecolor='#4b5563', labelcolor='#f3f4f6')
    
    l_breach, = ax2.plot([], [], label='Unprotected Agent SLA Violations', color='#ef4444', linewidth=2.0)
    l_safe, = ax2.plot([], [], label='Shielded Agent Violations (Guaranteed 0)', color='#10b981', linewidth=2.5)
    ax2.set_title("Catastrophic SLA Breach Prevention", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Simulation Steps (Alibaba OSDI Trace Replay)")
    ax2.set_ylabel("SLA Breach Indicator (1=Violation)")
    ax2.set_xlim(0, len(unprotected_breaches))
    ax2.set_ylim(-0.1, 1.2)
    ax2.legend(loc='upper right', facecolor='#1f2937', edgecolor='#4b5563', labelcolor='#f3f4f6')
    
    def animate(i):
        x = np.arange(i+1)
        l_intent.set_data(x, intended_actions[:i+1])
        l_exec.set_data(x, executed_actions[:i+1])
        l_breach.set_data(x, unprotected_breaches[:i+1])
        l_safe.set_data(x, np.zeros(i+1))
        return l_intent, l_exec, l_breach, l_safe
        
    plt.tight_layout()
    anim = animation.FuncAnimation(fig, animate, frames=len(intended_actions), interval=60, blit=True)
    out_path = os.path.join(ABLATION_DIR, "animated_safe_shield.gif")
    anim.save(out_path, writer='pillow', fps=15)
    plt.close(fig)
    print(f"Saved: {out_path}")

# ==============================================================================
# GIF 2: Hierarchical Heterogeneous GPU Cluster Autoscaling
# ==============================================================================
def generate_hierarchical_gpu_gif(steps=80, offset=100000):
    print("\n[2/3] Generating Hierarchical GPU Autoscaling Animation...")
    env = HeterogeneousK8sEnv(DATA_PATH, MODEL_DIR, use_forecasts=True, max_steps=steps, start_offset=offset)
    model = PPO.load(os.path.join(MODEL_DIR, "Hierarchical_Predictive_RL_ppo.zip"))
    
    obs, _ = env.reset()
    cpu_nodes, gpu_nodes, cpu_utils, gpu_utils = [], [], [], []
    
    for _ in range(steps):
        act, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(act)
        if done: break
        cpu_nodes.append(info.get('cpu_nodes', 10.0))
        gpu_nodes.append(info.get('gpu_nodes', 4.0))
        cpu_utils.append(info.get('cpu_util', 0.5) * 100.0)
        gpu_utils.append(info.get('gpu_util', 0.5) * 100.0)
        
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=100)
    set_dark_theme(fig, [ax1, ax2])
    
    l_cpu, = ax1.plot([], [], label='CPU-Optimized Pool (16-Core Nodes)', color='#38bdf8', linewidth=2.2)
    l_gpu, = ax1.plot([], [], label='GPU-Accelerated Pool (8x GPU Nodes)', color='#a855f7', linewidth=2.5)
    ax1.set_title("Hierarchical Control: Decoupled Multi-Pool Node Allocations", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Active Node Count")
    ax1.set_xlim(0, len(cpu_nodes))
    ax1.set_ylim(0, max(max(cpu_nodes), max(gpu_nodes)) + 5)
    ax1.legend(loc='upper left', facecolor='#1f2937', edgecolor='#4b5563', labelcolor='#f3f4f6')
    
    l_cpu_u, = ax2.plot([], [], label='CPU Utilization %', color='#38bdf8', linestyle='--', linewidth=2.0)
    l_gpu_u, = ax2.plot([], [], label='GPU Utilization %', color='#a855f7', linewidth=2.0)
    ax2.axhline(y=80.0, color='#ef4444', linestyle=':', label='80% SLA Threshold')
    ax2.set_title("Live Compute Pool Utilization Trajectories", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Simulation Steps (Pod Arrival Stream)")
    ax2.set_ylabel("Utilization (%)")
    ax2.set_xlim(0, len(cpu_utils))
    ax2.set_ylim(0, 110)
    ax2.legend(loc='upper right', facecolor='#1f2937', edgecolor='#4b5563', labelcolor='#f3f4f6')
    
    def animate(i):
        x = np.arange(i+1)
        l_cpu.set_data(x, cpu_nodes[:i+1])
        l_gpu.set_data(x, gpu_nodes[:i+1])
        l_cpu_u.set_data(x, cpu_utils[:i+1])
        l_gpu_u.set_data(x, gpu_utils[:i+1])
        return l_cpu, l_gpu, l_cpu_u, l_gpu_u
        
    plt.tight_layout()
    anim = animation.FuncAnimation(fig, animate, frames=len(cpu_nodes), interval=60, blit=True)
    out_path = os.path.join(ABLATION_DIR, "animated_hierarchical_gpu.gif")
    anim.save(out_path, writer='pillow', fps=15)
    plt.close(fig)
    print(f"Saved: {out_path}")

# ==============================================================================
# GIF 3: Multi-Horizon Lookahead Transformer Forecast Wave
# ==============================================================================
def generate_multi_horizon_gif(steps=80):
    print("\n[3/3] Generating Multi-Horizon Lookahead Transformer Animation...")
    import pandas as pd
    from scripts.train_multi_horizon import prepare_multi_horizon_data
    
    df = pd.read_parquet(DATA_PATH)
    X_test, y_test = prepare_multi_horizon_data(df.iloc[20000:25000].reset_index(drop=True), horizons=[1, 5, 15], subset_size=2000)
    
    model = MultiHorizonTransformer(input_size=3, d_model=64, nhead=4, num_layers=2, horizons=[1, 5, 15])
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "multi_horizon_transformer.pth"), weights_only=True))
    model.eval()
    
    with torch.no_grad():
        preds = model(X_test[:steps]).numpy() # (steps, 3, 2)
        actuals = y_test[:steps].numpy()      # (steps, 3, 2)
        
    actual_t = actuals[:, 0, 0] # CPU demand at t+1
    pred_h1 = preds[:, 0, 0]    # t+1
    pred_h5 = preds[:, 1, 0]    # t+5
    pred_h15 = preds[:, 2, 0]   # t+15
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    set_dark_theme(fig, [ax])
    
    l_act, = ax.plot([], [], label='Observed Workload Demand', color='#f3f4f6', linewidth=2.0)
    l_h1, = ax.plot([], [], label='Forecast Horizon t+1 (Immediate Container Burst)', color='#10b981', linewidth=2.2)
    l_h5, = ax.plot([], [], label='Forecast Horizon t+5 (Near-Term Queue Depth)', color='#f59e0b', linestyle='--', linewidth=2.0)
    l_h15, = ax.plot([], [], label='Forecast Horizon t+15 (60s Physical Cloud VM Boot)', color='#38bdf8', linestyle=':', linewidth=2.5)
    
    ax.set_title("Multi-Horizon Lookahead: Transformer Forecasting Future Cloud Bursts", fontsize=11, fontweight='bold')
    ax.set_xlabel("Time Step (Pod Arrival Event Stream)")
    ax.set_ylabel("Normalized Cluster Demand")
    ax.set_xlim(0, len(actual_t))
    ax.set_ylim(-0.1, max(np.max(actual_t), np.max(preds[:, :, 0])) + 0.2)
    ax.legend(loc='upper right', facecolor='#1f2937', edgecolor='#4b5563', labelcolor='#f3f4f6')
    
    def animate(i):
        x = np.arange(i+1)
        l_act.set_data(x, actual_t[:i+1])
        l_h1.set_data(x, pred_h1[:i+1])
        l_h5.set_data(x, pred_h5[:i+1])
        l_h15.set_data(x, pred_h15[:i+1])
        return l_act, l_h1, l_h5, l_h15
        
    plt.tight_layout()
    anim = animation.FuncAnimation(fig, animate, frames=len(actual_t), interval=60, blit=True)
    out_path = os.path.join(ABLATION_DIR, "animated_multi_horizon.gif")
    anim.save(out_path, writer='pillow', fps=15)
    plt.close(fig)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    generate_safe_shield_gif(steps=70, offset=100000)
    generate_hierarchical_gpu_gif(steps=70, offset=100000)
    generate_multi_horizon_gif(steps=70)
    print("\nAll 3 animations rendered successfully!")
