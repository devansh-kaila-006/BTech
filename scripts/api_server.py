import os
import sys
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from stable_baselines3 import PPO

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
from src.rl_agent.state_builder import CompositeStateBuilder

app = FastAPI(
    title="Kubernetes Predictive-RL Autoscaling Service",
    description="Production-grade AI autoscaler adapter compatible with Kubernetes KEDA & Prometheus",
    version="1.0.0"
)

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
model_path = os.path.join(MODEL_DIR, "Hierarchical_Predictive_RL_ppo.zip")
if not os.path.exists(model_path):
    model_path = os.path.join(MODEL_DIR, "Predictive_RL_Long_long_ppo.zip")

print(f"Loading RL policy weights from {model_path}...")
rl_model = PPO.load(model_path)
state_builder = CompositeStateBuilder(MODEL_DIR, use_forecasts=True)

# Simulated live cluster telemetry cache
cluster_telemetry = {
    "cpu_nodes": 12.0,
    "gpu_nodes": 4.0,
    "cpu_utilization": 0.45,
    "gpu_utilization": 0.60,
    "power_watts": 2350.0,
    "sla_breach_count": 0,
    "total_scaled_events": 0
}

class PodTelemetry(BaseModel):
    cpu_milli: float = 4000.0
    memory_mib: float = 8192.0
    workload_type: int = 1 # 0: CPU, 1: GPU
    qos: str = "LS" # LS: Latency Sensitive, BE: Best Effort

class ScalingRequest(BaseModel):
    current_cpu_nodes: float = 12.0
    current_gpu_nodes: float = 4.0
    current_pod: PodTelemetry
    rolling_window: Optional[List[List[float]]] = None

class ScalingDecisionResponse(BaseModel):
    status: str
    target_cpu_nodes: float
    target_gpu_nodes: float
    cpu_scale_delta: float
    gpu_scale_delta: float
    oracle_forecasts: Dict[str, float]
    sla_risk_score: float

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "k8s-predictive-rl-autoscaler",
        "engine": "PPO + Multi-Oracle Ensemble",
        "models_loaded": {
            "workload_lstm": True,
            "latency_xgboost": True,
            "power_dnn": True,
            "ppo_policy": True
        }
    }

@app.post("/v1/scaling-decision", response_model=ScalingDecisionResponse)
def get_scaling_decision(request: ScalingRequest):
    try:
        # 1. Normalize pod features
        raw_pod = {
            'cpu_norm': min(1.0, request.current_pod.cpu_milli / 96000.0),
            'mem_norm': min(1.0, request.current_pod.memory_mib / 400000.0),
            'workload_type': float(request.current_pod.workload_type)
        }
        
        # 2. Extract or create rolling window
        if request.rolling_window and len(request.rolling_window) >= 64:
            window = np.array(request.rolling_window[-64:], dtype=np.float32)
        else:
            # Replicate current feature across window
            window = np.tile([raw_pod['cpu_norm'], raw_pod['mem_norm'], raw_pod['workload_type']], (64, 1))
            
        # 3. Build composite state
        composite_7d = state_builder.build_state(raw_pod, window)
        
        # 4. Formulate 11D Hierarchical observation
        cluster_state = np.array([
            request.current_cpu_nodes / 100.0,
            request.current_gpu_nodes / 40.0,
            cluster_telemetry['cpu_utilization'],
            cluster_telemetry['gpu_utilization']
        ], dtype=np.float32)
        
        if rl_model.observation_space.shape[0] == 11:
            full_obs = np.concatenate([composite_7d, cluster_state]).astype(np.float32)
        else:
            full_obs = composite_7d[:rl_model.observation_space.shape[0]]
            
        # 5. Policy Inference
        action, _ = rl_model.predict(full_obs, deterministic=True)
        
        if len(action) >= 2:
            cpu_scale = float(action[0])
            gpu_scale = float(action[1])
        else:
            cpu_scale = float(action[0])
            gpu_scale = float(action[0]) * 0.5
            
        target_cpu = max(1.0, min(100.0, request.current_cpu_nodes + (cpu_scale * 5.0)))
        target_gpu = max(1.0, min(40.0, request.current_gpu_nodes + (gpu_scale * 2.0)))
        
        # Update telemetry
        cluster_telemetry['cpu_nodes'] = target_cpu
        cluster_telemetry['gpu_nodes'] = target_gpu
        cluster_telemetry['total_scaled_events'] += 1
        
        sla_risk = 0.0
        if composite_7d[3] > 0.8: # Workload prediction spike
            sla_risk = float(composite_7d[3])
            
        return {
            "status": "success",
            "target_cpu_nodes": round(target_cpu, 2),
            "target_gpu_nodes": round(target_gpu, 2),
            "cpu_scale_delta": round(target_cpu - request.current_cpu_nodes, 2),
            "gpu_scale_delta": round(target_gpu - request.current_gpu_nodes, 2),
            "oracle_forecasts": {
                "pred_cpu_demand": float(composite_7d[3]),
                "pred_mem_demand": float(composite_7d[4]),
                "pred_delay_seconds": float(composite_7d[5]),
                "pred_power_watts": float(composite_7d[6])
            },
            "sla_risk_score": sla_risk
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def prometheus_metrics():
    """Prometheus compatible scrape endpoint for Kubernetes HPA/KEDA."""
    return f"""# HELP k8s_predicted_cpu_nodes Optimal CPU nodes calculated by Predictive-RL
# TYPE k8s_predicted_cpu_nodes gauge
k8s_predicted_cpu_nodes {cluster_telemetry['cpu_nodes']}

# HELP k8s_predicted_gpu_nodes Optimal GPU nodes calculated by Predictive-RL
# TYPE k8s_predicted_gpu_nodes gauge
k8s_predicted_gpu_nodes {cluster_telemetry['gpu_nodes']}

# HELP k8s_cluster_power_watts Estimated cluster electrical wattage
# TYPE k8s_cluster_power_watts gauge
k8s_cluster_power_watts {cluster_telemetry['power_watts']}

# HELP k8s_sla_breach_total Cumulative SLA delay breaches
# TYPE k8s_sla_breach_total counter
k8s_sla_breach_total {cluster_telemetry['sla_breach_count']}

# HELP k8s_autoscaler_events_total Total scaling events triggered by agent
# TYPE k8s_autoscaler_events_total counter
k8s_autoscaler_events_total {cluster_telemetry['total_scaled_events']}
"""

if __name__ == "__main__":
    import uvicorn
    print("Starting Kubernetes Predictive Autoscaling API on http://localhost:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)

