import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_engine.streaming_engine import StreamingIngester, RollingWindowBuffer
from rl_agent.state_builder import CompositeStateBuilder

class K8sScalingEnv(gym.Env):
    """
    Custom Gymnasium Environment for the Predictive-RL Autoscaler.
    Simulates a live Kubernetes cluster using Alibaba traces.
    """
    def __init__(self, data_path, model_dir, use_forecasts=True, use_attention=False, max_steps=5000, start_offset=0):
        super(K8sScalingEnv, self).__init__()
        
        self.data_path = data_path
        self.max_steps = max_steps
        self.current_step = 0
        self.start_offset = start_offset
        self.use_attention = use_attention
        
        # State and Engine Initialization
        self.ingester = StreamingIngester(data_path, batch_size=1, start_offset=start_offset)
        self.buffer = RollingWindowBuffer(window_size=64, feature_keys=['cpu_norm', 'mem_norm', 'workload_type'])
        self.state_builder = CompositeStateBuilder(model_dir, use_forecasts=use_forecasts, use_attention=use_attention)
        
        # Action Space: [-1.0, 1.0] representing percentage scale down / scale up of node pool
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation Space
        if self.use_attention:
            # 4 tokens * 16 d_model = 64D
            obs_shape = (64,)
        else:
            # Flat Concat: 3 base + 2 workload + 1 latency + 1 power = 7D
            obs_shape = (7,)
            
        self.observation_space = spaces.Box(low=-1000.0, high=1000.0, shape=obs_shape, dtype=np.float32)
        
        # Cluster tracking
        self.current_nodes = 10.0 # Start with 10 generic nodes
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_nodes = 10.0
        
        self.ingester.reset()
        self.buffer.reset()
        
        # Warm up the buffer
        for _ in range(64):
            pod = self.ingester.step()
            if pod is not None:
                self.buffer.add(pod)
                
        # Get first real state
        pod = self.ingester.step()
        self.buffer.add(pod)
        
        obs = self.state_builder.build_state(pod, self.buffer.get_sequence())
        return obs, {}
        
    def step(self, action):
        self.current_step += 1
        
        # 1. Apply Action (Scale cluster)
        scale_factor = action[0] # [-1.0, 1.0]
        # Max scaling step is +/- 5 nodes per tick
        self.current_nodes += (scale_factor * 5.0)
        self.current_nodes = max(1.0, min(self.current_nodes, 100.0)) # Clamp between 1 and 100 nodes
        
        # 2. Simulate Time (Get next arriving pod)
        pod = self.ingester.step()
        done = False
        if pod is None or self.current_step >= self.max_steps:
            done = True
            info = {
                'nodes': self.current_nodes,
                'utilization': 0.0,
                'reward_power': -(self.current_nodes * 0.1),
                'reward_delay': 0.0
            }
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, done, False, info
            
        self.buffer.add(pod)
        
        # 3. Calculate Reward
        # Base penalty for keeping nodes alive (Power Cost)
        power_penalty = self.current_nodes * 0.1 
        
        # Delay penalty if cluster is over-utilized
        cpu_demand = pod.get('cpu_norm', 0.0) * 100.0 # rough estimate of cores requested
        utilization = cpu_demand / max(1.0, (self.current_nodes * 8.0)) # assume 8 cores per node
        
        delay_penalty = 0.0
        if utilization > 0.8:
            # Heavy penalty for starvation (simulating SLA breaches)
            delay_penalty = (utilization - 0.8) * 10.0
            
        reward = -(power_penalty + delay_penalty)
        
        # 4. Generate Next State
        obs = self.state_builder.build_state(pod, self.buffer.get_sequence())
        
        info = {
            'nodes': self.current_nodes,
            'utilization': utilization,
            'reward_power': -power_penalty,
            'reward_delay': -delay_penalty
        }
        
        return obs, reward, done, False, info
