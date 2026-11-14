import os
import sys
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from scripts.api_server import app

client = TestClient(app)

def test_api_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data
    assert data["models_loaded"]["ppo_policy"] is True

def test_api_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    text = response.text
    assert "k8s_predicted_cpu_nodes" in text
    assert "k8s_predicted_gpu_nodes" in text
    assert "k8s_sla_breach_total" in text

def test_api_scaling_decision_endpoint():
    payload = {
        "current_cpu_nodes": 12.0,
        "current_gpu_nodes": 4.0,
        "current_pod": {
            "cpu_milli": 4000.0,
            "memory_mib": 8192.0,
            "workload_type": 1,
            "qos": "LS"
        }
    }
    response = client.post("/v1/scaling-decision", json=payload)
    assert response.status_code == 200
    res_data = response.json()
    assert "target_cpu_nodes" in res_data
    assert "target_gpu_nodes" in res_data
    assert "cpu_scale_delta" in res_data
    assert "gpu_scale_delta" in res_data
    assert "oracle_forecasts" in res_data
    assert 1.0 <= res_data["target_cpu_nodes"] <= 100.0
