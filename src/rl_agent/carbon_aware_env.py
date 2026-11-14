import math
import gymnasium as gym
import numpy as np
from src.rl_agent.environment import K8sScalingEnv

class CarbonAwareK8sEnv(gym.Wrapper):
    """
    Carbon-Aware Cloud Autoscaling Wrapper.
    
    Dynamically models hourly electricity grid carbon intensity (gCO2/kWh)
    following real-world grid diurnal patterns (e.g. CAISO / National Grid):
      - Clean solar hours (10:00 - 15:00): Low carbon intensity (180 - 250 gCO2/kWh)
      - Evening fossil peaker hours (18:00 - 22:00): High carbon intensity (500 - 650 gCO2/kWh)
      
    Multi-objective reward incorporates embodied and operational carbon emissions:
      R_t = - (Power_kW * Carbon_Intensity) - SLA_Delay_Penalty
    """
    def __init__(self, env, base_intensity=350.0):
        super(CarbonAwareK8sEnv, self).__init__(env)
        self.base_intensity = base_intensity
        self.total_carbon_emitted_grams = 0.0
        
    def get_current_carbon_intensity(self, step: int) -> float:
        """Simulates grid carbon intensity as a diurnal cycle."""
        # 1 day = 1440 ticks (1 tick = 1 minute)
        minute_of_day = (step % 1440)
        hour = minute_of_day / 60.0
        
        # Diurnal solar dip around 13:00, peak around 20:00
        solar_dip = -150.0 * math.exp(-((hour - 13.0) ** 2) / 8.0)
        evening_peak = 200.0 * math.exp(-((hour - 20.0) ** 2) / 6.0)
        
        intensity = self.base_intensity + solar_dip + evening_peak
        return max(120.0, min(750.0, intensity))
        
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        # 1. Calculate carbon intensity for current time
        intensity = self.get_current_carbon_intensity(self.env.unwrapped.current_step)
        
        # 2. Power in kilowatts (assume 250W per active node)
        active_nodes = info.get('nodes', 10.0)
        power_kw = (active_nodes * 250.0) / 1000.0
        
        # 3. Carbon emissions for this step: grams CO2
        # (power_kw * (1/60 hours) * intensity gCO2/kWh)
        step_emissions = power_kw * (1.0 / 60.0) * intensity
        self.total_carbon_emitted_grams += step_emissions
        
        # 4. Carbon-penalized reward
        # Normalized by base 350 g/kWh
        carbon_factor = intensity / self.base_intensity
        sla_penalty = abs(info.get('reward_delay', 0.0))
        power_penalty = abs(info.get('reward_power', active_nodes * 0.1))
        
        carbon_reward = -((power_penalty * carbon_factor) + sla_penalty)
        
        info['carbon_intensity_g_per_kwh'] = intensity
        info['step_carbon_emissions_g'] = step_emissions
        info['total_carbon_emissions_kg'] = self.total_carbon_emitted_grams / 1000.0
        
        return obs, carbon_reward, done, truncated, info
        
    def reset(self, **kwargs):
        self.total_carbon_emitted_grams = 0.0
        return self.env.reset(**kwargs)

