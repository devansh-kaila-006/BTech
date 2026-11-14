import os
import sys
import pytest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from src.rl_agent.industry_baselines import KubernetesHPAController, PIDController, CloudFinancialAndCarbonLedger

def test_kubernetes_hpa_controller():
    hpa = KubernetesHPAController(target_util=0.80, tolerance=0.10, stabilization_window=5)
    
    # 1. Under-utilized (util = 0.40 -> target 0.80, ratio 0.5): should scale down
    action = hpa.predict(current_nodes=10.0, current_util=0.40)
    assert action[0] < 0.0 # scale down signal
    
    # 2. Over-utilized (util = 0.95 -> target 0.80, ratio > 1.1): should scale up
    action = hpa.predict(current_nodes=10.0, current_util=0.95)
    assert action[0] > 0.0 # scale up signal
    
    # 3. Within tolerance (util = 0.82 -> ratio = 1.025, within 10%): should stay neutral
    action = hpa.predict(current_nodes=10.0, current_util=0.82)
    assert np.isclose(action[0], 0.0)

def test_pid_controller():
    pid = PIDController(target_util=0.80, kp=2.0, ki=0.1, kd=0.5)
    
    # High utilization error
    action = pid.predict(current_nodes=10.0, current_util=0.95)
    assert action[0] > 0.0
    
    # Low utilization error
    action = pid.predict(current_nodes=10.0, current_util=0.50)
    assert action[0] < 0.0

def test_financial_and_carbon_ledger():
    ledger = CloudFinancialAndCarbonLedger(step_duration_sec=2.0, cost_per_node_hr=0.340, sla_penalty_hr=50.0)
    
    # Non-breach step
    metrics_safe = ledger.compute_step_ledger(step_idx=0, nodes=10.0, util=0.60, is_sla_breach=False)
    assert metrics_safe['compute_cost'] > 0.0
    assert metrics_safe['sla_fine'] == 0.0
    assert metrics_safe['carbon_kg'] > 0.0
    
    # Breach step
    metrics_breach = ledger.compute_step_ledger(step_idx=0, nodes=10.0, util=0.95, is_sla_breach=True)
    assert metrics_breach['sla_fine'] > 0.0
    assert metrics_breach['total_financial_cost'] > metrics_safe['total_financial_cost']

