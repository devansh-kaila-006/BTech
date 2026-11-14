import os
import sys
import time
import tracemalloc
import pandas as pd
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
from src.data_engine.streaming_engine import StreamingIngester, RollingWindowBuffer

def benchmark_static_pandas(data_path, num_rows=50000):
    print("\n--- Benchmarking Naive Static Pandas ---")
    tracemalloc.start()
    start_time = time.time()
    
    # Naive approach: Load everything into memory, then iterate
    df = pd.read_parquet(data_path)
    df = df.head(num_rows)
    
    # Simulate a moving window using static slicing (extremely slow)
    window_size = 64
    features = ['cpu_norm', 'mem_norm', 'workload_type']
    
    for i in range(len(df)):
        if i >= window_size:
            window = df.iloc[i-window_size:i][features].values
            
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    duration = end_time - start_time
    print(f"Time Taken: {duration:.2f} seconds")
    print(f"Throughput: {num_rows/duration:.0f} rows/sec")
    print(f"Peak Memory: {peak / 10**6:.2f} MB")
    
def benchmark_streaming_engine(data_path, num_rows=50000):
    print("\n--- Benchmarking PyArrow Streaming Engine ---")
    tracemalloc.start()
    start_time = time.time()
    
    ingester = StreamingIngester(data_path, batch_size=1000)
    buffer = RollingWindowBuffer(window_size=64, feature_keys=['cpu_norm', 'mem_norm', 'workload_type'])
    
    count = 0
    while count < num_rows:
        pod = ingester.step()
        if pod is None: break
        buffer.add(pod)
        if buffer.is_full():
            seq = buffer.get_sequence()
        count += 1
            
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    duration = end_time - start_time
    print(f"Time Taken: {duration:.2f} seconds")
    print(f"Throughput: {num_rows/duration:.0f} rows/sec")
    print(f"Peak Memory: {peak / 10**6:.2f} MB")

if __name__ == "__main__":
    data_path = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_trace.parquet")
    if not os.path.exists(data_path):
        print("Data not found. Did you run data_processor.py?")
        sys.exit(1)
        
    print("Running Ablation Benchmark: Memory & Throughput (Processing 50,000 pods)")
    benchmark_static_pandas(data_path, 50000)
    benchmark_streaming_engine(data_path, 50000)
