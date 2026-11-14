import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class LagrangianConstraintCallback(BaseCallback):
    """
    Dual Gradient Ascent Callback for Constrained Reinforcement Learning (PPO-Lagrangian).
    
    Dynamically adjusts the Lagrange multiplier lambda_t to strictly bound
    the expected SLA violation rate below an exact epsilon budget:
      lambda_{k+1} = max(0, lambda_k + alpha * (sla_violation_rate - epsilon))
    """
    def __init__(self, epsilon_sla=0.05, alpha_dual=0.05, initial_lambda=1.0, verbose=0):
        super(LagrangianConstraintCallback, self).__init__(verbose)
        self.epsilon_sla = epsilon_sla      # Maximum allowable SLA breach fraction (5%)
        self.alpha_dual = alpha_dual        # Dual learning rate
        self.lambda_t = initial_lambda      # Initial Lagrange multiplier
        self.lambda_history = [initial_lambda]
        self.step_sla_violations = []
        self.mean_violation_history = []

    def _on_step(self) -> bool:
        # Extract environment info
        infos = self.locals.get('infos', [])
        for info in infos:
            util = info.get('utilization', info.get('cpu_util', 0.0))
            is_breach = 1.0 if util > 0.8 else 0.0
            self.step_sla_violations.append(is_breach)
        return True

    def _on_rollout_end(self) -> None:
        # Perform dual gradient ascent update at the end of each PPO rollout
        if len(self.step_sla_violations) > 0:
            current_sla_rate = np.mean(self.step_sla_violations)
            constraint_error = current_sla_rate - self.epsilon_sla
            
            # Dual gradient update: lambda = max(0, lambda + alpha * (cost - epsilon))
            self.lambda_t = max(0.0, self.lambda_t + self.alpha_dual * constraint_error)
            
            self.lambda_history.append(self.lambda_t)
            self.mean_violation_history.append(current_sla_rate)
            self.step_sla_violations = []
            
            if self.verbose > 0:
                print(f"[Lagrangian Update] SLA Rate: {current_sla_rate:.3f} | Target: {self.epsilon_sla} | New Lambda: {self.lambda_t:.3f}")

class LagrangianEnvWrapper(gym.Wrapper):
    """
    Gym wrapper that dynamically incorporates the current Lagrange multiplier
    into the scalar reward delivered to the policy network:
      R_t = - Power_Cost - lambda_t * SLA_Cost
    """
    def __init__(self, env, lagrangian_callback):
        super(LagrangianEnvWrapper, self).__init__(env)
        self.callback = lagrangian_callback
        
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Calculate unweighted power and delay penalties
        power_pen = abs(info.get('reward_power', -1.0))
        sla_pen = abs(info.get('reward_delay', info.get('reward_sla', 0.0)))
        
        # Apply Lagrangian dual multiplier
        lambda_val = self.callback.lambda_t
        lagrangian_reward = -(power_pen + (lambda_val * sla_pen))
        
        info['lagrange_multiplier'] = lambda_val
        return obs, lagrangian_reward, done, truncated, info
