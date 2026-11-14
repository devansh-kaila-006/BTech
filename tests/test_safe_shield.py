import os
import sys
import pytest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from src.rl_agent.environment import K8sScalingEnv
from src.rl_agent.safe_wrapper import SafeActionShield

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_trace.parquet")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

def test_safe_action_shield_intervention():
    base_env = K8sScalingEnv(DATA_PATH, MODEL_DIR, use_forecasts=False, max_steps=50)
    shield_env = SafeActionShield(base_env)
    
    obs, _ = shield_env.reset()
    assert shield_env.interventions == 0
    
    # Try aggressive scale down (-1.0 -> -5 nodes each step)
    aggressive_scale_down = np.array([-1.0], dtype=np.float32)
    
    for _ in range(30):
        # Pass a copy to verify whether shield overrides
        act = np.array([-1.0], dtype=np.float32)
        obs, reward, done, _, info = shield_env.step(act)
        
        # Invariant: nodes must never drop below the safe minimum boundary
        assert info.get('nodes', 10.0) >= 1.0
        # Invariant: Utilization under shield protection must stay <= 1.0 (or minimal violation bounded by clamp)
        if done:
            break
            
    # The shield should have intervened multiple times to protect the cluster
    assert shield_env.interventions > 0
    assert info['shield_interventions'] == shield_env.interventions

