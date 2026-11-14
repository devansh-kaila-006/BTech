import os
import sys
import pytest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from src.data_engine.streaming_engine import StreamingIngester, RollingWindowBuffer

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_trace.parquet")

def test_streaming_ingester_initialization_and_step():
    assert os.path.exists(DATA_PATH), f"Trace dataset missing at {DATA_PATH}"
    ingester = StreamingIngester(DATA_PATH, batch_size=1, start_offset=0)
    
    first_pod = ingester.step()
    assert first_pod is not None
    assert 'cpu_norm' in first_pod
    assert 'mem_norm' in first_pod
    assert 'workload_type' in first_pod
    assert 0.0 <= first_pod['cpu_norm'] <= 1.0

def test_streaming_ingester_fast_skip():
    # Test that start offset skipping is fast and accurate
    offset = 50000
    ingester = StreamingIngester(DATA_PATH, batch_size=1, start_offset=offset)
    pod = ingester.step()
    assert pod is not None
    assert ingester.row_index == offset + 1

def test_rolling_window_buffer():
    buffer = RollingWindowBuffer(window_size=64, feature_keys=['cpu_norm', 'mem_norm', 'workload_type'])
    assert len(buffer) == 0
    
    # Push 100 mock pods
    for i in range(100):
        buffer.add({'cpu_norm': float(i)/100.0, 'mem_norm': 0.5, 'workload_type': 1.0})
        
    assert len(buffer) == 64
    seq = buffer.get_sequence()
    assert seq.shape == (64, 3)
    assert np.isclose(seq[-1, 0], 99.0 / 100.0)

