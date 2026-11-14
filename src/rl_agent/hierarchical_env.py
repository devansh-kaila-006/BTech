import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)
from src.data_engine.streaming_engine import StreamingIngester, RollingWindowBuffer
from src.rl_agent.state_builder import CompositeStateBuilder

class HeterogeneousK8sEnv(gym.Env):
    """
    Hierarchical Heterogeneous Kubernetes Cluster Gymnasium Environment.
    
    Models two distinct node pools:
      - Pool 0: CPU-Optimized Node Pool (16 cores, standard compute)
      - Pool 1: GPU-Accelerated Node Pool (8 GPUs + 96 cores, high power TDP)
      
    Hierarchical Action Space:
      action[0]: Continuous scaling velocity for CPU pool in [-1.0, 1.0]
      action[1]: Continuous scaling velocity for GPU pool in [-1.0, 1.0]
      
    11D Composite State Space:
      [0..2]  Raw pod features (cpu_norm, mem_norm, workload_type)
      [3..6]  Cluster status (cpu_nodes, gpu_nodes, cpu_util, gpu_util)
      [7..10] Predictive Oracle Forecasts (pred_cpu, pred_mem, pred_delay, pred_power)
    """
    def __init__(self, data_path, model_dir, use_forecasts=True, max_steps=5000, start_offset=0):
        super(HeterogeneousK8sEnv, self).__init__()
        
        self.data_path = data_path
        self.model_dir = model_dir
        self.use_forecasts = use_forecasts
        self.max_steps = max_steps
        self.start_offset = start_offset
        self.current_step = 0
        
        # Ingestion & State Builder
        self.ingester = StreamingIngester(data_path, batch_size=1, start_offset=start_offset)
        self.buffer = RollingWindowBuffer(window_size=64, feature_keys=['cpu_norm', 'mem_norm', 'workload_type'])
        self.state_builder = CompositeStateBuilder(model_dir, use_forecasts=use_forecasts, use_attention=False)
        
        # Hierarchical Action Space: [cpu_scale, gpu_scale]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )
        
        # Observation Space: 11 dimensions
        # 3 (pod) + 4 (cluster state) + 4 (oracle forecasts) = 11D
        self.observation_space = spaces.Box(
            low=-1000.0,
            high=1000.0,
            shape=(11,),
            dtype=np.float32
        )
        
        # Heterogeneous Cluster Pools
        self.cpu_nodes = 10.0  # Min 1, Max 100
        self.gpu_nodes = 4.0   # Min 1, Max 40
        self.last_cpu_util = 0.5
        self.last_gpu_util = 0.5

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cpu_nodes = 10.0
        self.gpu_nodes = 4.0
        self.last_cpu_util = 0.5
        self.last_gpu_util = 0.5
        
        self.ingester.reset()
        self.buffer.reset()
        
        # Warm up rolling window buffer
        for _ in range(64):
            pod = self.ingester.step()
            if pod is not None:
                self.buffer.add(pod)
                
        pod = self.ingester.step()
        if pod is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32), {}
        self.buffer.add(pod)
        
        obs = self._build_hetero_state(pod)
        return obs, {}

    def _build_hetero_state(self, pod):
        # 1. Base 7D state from CompositeStateBuilder (raw pod + oracle forecasts)
        base_composite = self.state_builder.build_state(pod, self.buffer.get_sequence()) # shape (7,)
        
        # 2. Cluster state features (dim=4)
        cluster_state = np.array([
            self.cpu_nodes / 100.0,
            self.gpu_nodes / 40.0,
            self.last_cpu_util,
            self.last_gpu_util
        ], dtype=np.float32)
        
        # Fuse into 11D
        full_state = np.concatenate([base_composite, cluster_state]).astype(np.float32)
        return full_state

    def step(self, action):
        self.current_step += 1
        
        # 1. Apply Hierarchical Scaling Actions
        cpu_scale = float(action[0])
        gpu_scale = float(action[1])
        
        # Step velocity: CPU up to +/- 5 nodes, GPU up to +/- 2 nodes
        self.cpu_nodes = max(1.0, min(100.0, self.cpu_nodes + (cpu_scale * 5.0)))
        self.gpu_nodes = max(1.0, min(40.0, self.gpu_nodes + (gpu_scale * 2.0)))
        
        # 2. Stream next pod arrival
        pod = self.ingester.step()
        done = False
        if pod is None or self.current_step >= self.max_steps:
            done = True
            info = {
                'cpu_nodes': self.cpu_nodes,
                'gpu_nodes': self.gpu_nodes,
                'cpu_util': self.last_cpu_util,
                'gpu_util': self.last_gpu_util,
                'cpu_sla_breach': 0,
                'gpu_sla_breach': 0,
                'reward_power': -((self.cpu_nodes * 0.1) + (self.gpu_nodes * 0.3)),
                'reward_sla': 0.0
            }
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, done, False, info
            
        self.buffer.add(pod)
        
        # 3. Resource Allocation & Demand Simulation
        is_gpu_workload = (pod.get('workload_type', 0.0) >= 0.5)
        cpu_demand = pod.get('cpu_norm', 0.0) * 100.0
        
        # CPU Capacity: 16 cores per CPU node + 32 cores per GPU node available for compute
        total_cpu_cores = (self.cpu_nodes * 16.0) + (self.gpu_nodes * 32.0)
        self.last_cpu_util = min(2.0, cpu_demand / max(1.0, total_cpu_cores))
        
        # GPU Capacity: 8 GPUs per GPU node
        total_gpus = self.gpu_nodes * 8.0
        gpu_demand = (pod.get('mem_norm', 0.0) * 16.0) if is_gpu_workload else 0.0
        self.last_gpu_util = min(2.0, gpu_demand / max(1.0, total_gpus))
        
        # 4. Multi-Objective Reward
        # Power penalty: GPU nodes consume 3x more wattage than CPU nodes
        power_penalty = (self.cpu_nodes * 0.1) + (self.gpu_nodes * 0.3)
        
        # CPU SLA penalty
        cpu_sla_penalty = 0.0
        if self.last_cpu_util > 0.8:
            cpu_sla_penalty = (self.last_cpu_util - 0.8) * 12.0
            
        # GPU SLA penalty (starvation on expensive GPU accelerator is severely punished)
        gpu_sla_penalty = 0.0
        if is_gpu_workload and self.last_gpu_util > 0.8:
            gpu_sla_penalty = (self.last_gpu_util - 0.8) * 30.0
            
        reward = -(power_penalty + cpu_sla_penalty + gpu_sla_penalty)
        
        # 5. Build Next Observation
        obs = self._build_hetero_state(pod)
        
        info = {
            'cpu_nodes': self.cpu_nodes,
            'gpu_nodes': self.gpu_nodes,
            'cpu_util': self.last_cpu_util,
            'gpu_util': self.last_gpu_util,
            'cpu_sla_breach': int(self.last_cpu_util > 0.8),
            'gpu_sla_breach': int(is_gpu_workload and self.last_gpu_util > 0.8),
            'reward_power': -power_penalty,
            'reward_sla': -(cpu_sla_penalty + gpu_sla_penalty)
        }
        
        return obs, reward, done, False, info

