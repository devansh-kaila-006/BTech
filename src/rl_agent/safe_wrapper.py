import gymnasium as gym
import numpy as np

class SafeActionShield(gym.Wrapper):
    """
    Action Shield Wrapper for Safe Reinforcement Learning.
    Intercepts the RL agent's actions and prevents catastrophic scale-down
    events when the known cluster workload is high.
    """
    def __init__(self, env):
        super(SafeActionShield, self).__init__(env)
        self.interventions = 0
        
    def step(self, action):
        # We need to know the current CPU demand to calculate the safe boundary.
        # We can extract it from the environment's current state buffer.
        try:
            # Get the most recent pod from the buffer
            recent_pod = self.env.unwrapped.buffer.get_sequence()[-1]
            cpu_demand = recent_pod[0] * 100.0 # index 0 is cpu_norm
        except:
            cpu_demand = 0.0
            
        # The environment calculates utilization as:
        # utilization = cpu_demand / (nodes * 8.0)
        # SLA breach occurs if utilization > 0.8
        # Therefore, Safe Nodes minimum = cpu_demand / (8.0 * 0.8) = cpu_demand / 6.4
        safe_min_nodes = cpu_demand / 6.4
        
        # Calculate what the agent wants to do
        scale_factor = action[0]
        intended_nodes = self.env.unwrapped.current_nodes + (scale_factor * 5.0)
        
        intervention_penalty = 0.0
        
        if intended_nodes < safe_min_nodes:
            # SHIELD INTERVENTION!
            self.interventions += 1
            
            # Clamp the action to exactly the safe boundary
            required_delta = safe_min_nodes - self.env.unwrapped.current_nodes
            # Convert delta back to a scale factor [-1.0, 1.0]
            safe_scale_factor = required_delta / 5.0
            
            # Clip it just in case
            safe_scale_factor = max(-1.0, min(safe_scale_factor, 1.0))
            
            # Overwrite the action
            action[0] = safe_scale_factor
            
            # Heavily penalize the agent for relying on the shield, 
            # so it learns to stay within safe bounds naturally.
            intervention_penalty = -50.0 
            
        # Proceed with the potentially modified action
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Add the intervention penalty to the reward
        reward += intervention_penalty
        info['shield_interventions'] = self.interventions
        
        return obs, reward, done, truncated, info
        
    def reset(self, **kwargs):
        self.interventions = 0
        return self.env.reset(**kwargs)
