import os
import sys
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
from src.rl_agent.environment import K8sScalingEnv

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_trace.parquet")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

def train_agent(run_name, use_forecasts, total_timesteps=100000):
    print(f"\n{'='*50}")
    print(f"Starting Training: {run_name} (Predictive State: {use_forecasts})")
    print(f"{'='*50}")
    
    # 1. Initialize custom Kubernetes Scaling Environment (Train on first segment of data)
    env = K8sScalingEnv(DATA_PATH, MODEL_DIR, use_forecasts=use_forecasts, max_steps=10000, start_offset=0)
    
    # 2. Setup Logging
    logger = configure(os.path.join(LOG_DIR, run_name), ["stdout", "csv"])
    
    # 3. Initialize PPO Agent
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=3e-4, clip_range=0.2)
    model.set_logger(logger)
    
    # 4. Train
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    end_time = time.time()
    
    print(f"Training completed in {end_time - start_time:.2f} seconds.")
    
    # 5. Save the final RL weights
    save_path = os.path.join(MODEL_DIR, f"{run_name}_long_ppo")
    model.save(save_path)
    print(f"Saved agent to {save_path}.zip")

if __name__ == "__main__":
    # Run 1: Standard Baseline RL (No ML forecasting, only raw cluster state)
    train_agent("Baseline_RL_Long", use_forecasts=False, total_timesteps=100000)
    
    # Run 2: Predictive Hierarchical RL (With full LSTM/XGBoost/DNN Oracle composite state)
    train_agent("Predictive_RL_Long", use_forecasts=True, total_timesteps=100000)
    
    print("\nAll training runs complete!")
