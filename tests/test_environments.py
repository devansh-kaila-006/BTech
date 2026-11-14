import os
import sys
import pytest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from src.rl_agent.environment import K8sScalingEnv
from src.rl_agent.hierarchical_env import HeterogeneousK8sEnv

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_trace.parquet")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

def test_k8s_scaling_env_step_and_reset():
    env = K8sScalingEnv(DATA_PATH, MODEL_DIR, use_forecasts=False, max_steps=100)
    obs, info = env.reset()
    
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (7,)
    assert env.current_nodes == 10.0
    
    # Step with scale up (+0.5 -> +2.5 nodes)
    action = np.array([0.5], dtype=np.float32)
    next_obs, reward, done, truncated, info = env.step(action)
    
    assert next_obs.shape == (7,)
    assert env.current_nodes == 12.5
    assert 'nodes' in info
    assert 'utilization' in info
    assert 'reward_power' in info
    assert 'reward_delay' in info
    assert not done

def test_heterogeneous_k8s_env():
    env = HeterogeneousK8sEnv(DATA_PATH, MODEL_DIR, use_forecasts=True, max_steps=100)
    obs, info = env.reset()
    
    assert obs.shape == (11,) # Composite 11D
    assert env.action_space.shape == (2,) # [delta_cpu, delta_gpu]
    
    # Actuate dual pools
    action = np.array([0.2, -0.4], dtype=np.float32)
    next_obs, reward, done, _, info = env.step(action)
    
    assert next_obs.shape == (11,)
    assert 'cpu_nodes' in info
    assert 'gpu_nodes' in info
    assert 'cpu_util' in info
    assert 'gpu_util' in info
    assert np.isclose(env.cpu_nodes, 10.0 + (0.2 * 5.0))
    assert np.isclose(env.gpu_nodes, 4.0 + (-0.4 * 2.0))
