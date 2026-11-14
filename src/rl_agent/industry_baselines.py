import numpy as np
from collections import deque

class KubernetesHPAController:
    """
    Emulates the official Kubernetes Horizontal Pod Autoscaler (HPA v2) algorithm:
    desiredReplicas = ceil[currentReplicas * (currentMetricValue / desiredMetricValue)]
    
    Includes:
    - Target utilization threshold (default 80%)
    - Tolerance band (default 10%, no action if ratio within [0.9, 1.1])
    - Stabilization window (prevents flapping during scale-down)
    """
    def __init__(self, target_util=0.80, tolerance=0.10, stabilization_window=10):
        self.target_util = target_util
        self.tolerance = tolerance
        self.stabilization_window = stabilization_window
        self.recommendation_history = deque(maxlen=stabilization_window)
        
    def reset(self):
        self.recommendation_history.clear()
        
    def predict(self, current_nodes, current_util):
        if current_nodes <= 0:
            current_nodes = 1.0
            
        ratio = current_util / self.target_util
        
        # Kubernetes tolerance check: if within +/- 10%, do not scale
        if abs(1.0 - ratio) <= self.tolerance:
            raw_desired = current_nodes
        else:
            raw_desired = np.ceil(current_nodes * ratio)
            
        self.recommendation_history.append(raw_desired)
        
        # K8s stabilization window: for scale down, pick the max recommendation in the window
        # to prevent premature scale down during temporary dips
        if raw_desired < current_nodes and len(self.recommendation_history) > 1:
            desired_nodes = max(self.recommendation_history)
        else:
            desired_nodes = raw_desired
            
        desired_nodes = max(1.0, min(desired_nodes, 100.0))
        delta_nodes = desired_nodes - current_nodes
        
        # Convert delta into environment action [-1.0, 1.0] where delta = action * 5.0
        action = np.clip(delta_nodes / 5.0, -1.0, 1.0)
        return np.array([action], dtype=np.float32)


class PIDController:
    """
    Industrial Proportional-Integral-Derivative (PID) Controller for Cloud Autoscaling.
    Commonly used in large-scale datacenter fleet management (e.g. Google Borg, Autopilot).
    
    u(t) = K_p * e(t) + K_i * int(e) + K_d * de/dt
    where e(t) = current_util - target_util
    """
    def __init__(self, target_util=0.80, kp=2.5, ki=0.15, kd=0.8, anti_windup=5.0):
        self.target_util = target_util
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.anti_windup = anti_windup
        
        self.integral = 0.0
        self.prev_error = 0.0
        
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        
    def predict(self, current_nodes, current_util):
        # Error is positive when cluster is over-utilized (need scale up)
        error = current_util - self.target_util
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error
        self.integral = np.clip(self.integral, -self.anti_windup, self.anti_windup)
        i_term = self.ki * self.integral
        
        # Derivative term
        d_term = self.kd * (error - self.prev_error)
        self.prev_error = error
        
        # Control signal in [-1.0, 1.0]
        control_signal = p_term + i_term + d_term
        action = np.clip(control_signal, -1.0, 1.0)
        
        return np.array([action], dtype=np.float32)


class CloudFinancialAndCarbonLedger:
    """
    Converts simulated telemetry into real-world dollars and carbon metrics.
    - AWS EC2 Pricing:
        * 8-Core Standard Node (c6i.2xlarge): $0.340 / hour ($0.0000944 / sec)
        * 16-Core Node (c6i.4xlarge): $0.680 / hour
    - Enterprise SLA Contract Penalties:
        * $50.00 per hour ($0.01389 / sec) during active SLA latency violations
    - Grid Carbon Footprint:
        * Diurnal carbon intensity I(t) in [200, 600] gCO2 / kWh
        * Node power baseline: 180W idle + 120W active per 8-core server
    """
    def __init__(self, step_duration_sec=1.0, cost_per_node_hr=0.340, sla_penalty_hr=50.0):
        self.step_duration_sec = step_duration_sec
        self.cost_per_node_sec = cost_per_node_hr / 3600.0
        self.sla_penalty_sec = sla_penalty_hr / 3600.0
        
    def compute_step_ledger(self, step_idx, nodes, util, is_sla_breach):
        # 1. Cloud Infrastructure Compute Cost ($)
        compute_cost = nodes * self.cost_per_node_sec * self.step_duration_sec
        
        # 2. SLA Financial Penalty ($)
        sla_fine = self.sla_penalty_sec * self.step_duration_sec if is_sla_breach else 0.0
        
        # 3. Electrical Power & Operational Carbon (kg CO2e)
        # 180W base + 120W * util per node
        power_watts = nodes * (180.0 + 120.0 * min(util, 1.0))
        power_kwh = (power_watts / 1000.0) * (self.step_duration_sec / 3600.0)
        
        # Diurnal carbon grid variation (gCO2 / kWh)
        carbon_intensity = 350.0 + 150.0 * np.sin(2 * np.pi * (step_idx % 1440) / 1440.0)
        carbon_grams = power_kwh * carbon_intensity
        carbon_kg = carbon_grams / 1000.0
        
        return {
            'compute_cost': compute_cost,
            'sla_fine': sla_fine,
            'total_financial_cost': compute_cost + sla_fine,
            'power_kwh': power_kwh,
            'carbon_kg': carbon_kg
        }

