import os
import sys
import pytest
import torch
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from src.rl_agent.state_builder import CompositeStateBuilder
from src.models.multi_horizon_transformer import MultiHorizonTransformer

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

def test_composite_state_builder_inference():
    builder = CompositeStateBuilder(MODEL_DIR, use_forecasts=True, use_attention=False)
    
    mock_pod = {'cpu_norm': 0.75, 'mem_norm': 0.40, 'workload_type': 1.0}
    mock_sequence = np.random.uniform(0.1, 0.9, (64, 3)).astype(np.float32)
    
    state = builder.build_state(mock_pod, mock_sequence)
    assert isinstance(state, np.ndarray)
    assert state.shape == (7,)
    assert not np.isnan(state).any()

def test_multi_horizon_transformer():
    model = MultiHorizonTransformer(input_size=3, d_model=64, nhead=4, num_layers=2, horizons=[1, 5, 15])
    model.eval()
    
    mock_x = torch.randn(8, 64, 3) # Batch 8, Seq 64, Features 3
    with torch.no_grad():
        out = model(mock_x)
        
    # Expected output shape: (8, 3, 2) -> (batch, horizons, target_features)
    assert out.shape == (8, 3, 2)
    assert not torch.isnan(out).any()

