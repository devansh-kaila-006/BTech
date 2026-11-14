# Ablation Study: Data Ingestion Architecture (Stage 1)

**Hypothesis:** For Reinforcement Learning environments simulating live clusters on massive datasets (150k+ rows), static memory loading (Pandas) causes significant memory bloat and slow stepping, bottlenecking the RL training loop. A PyArrow-backed streaming pipeline with a constant-time `RollingWindowBuffer` will vastly outperform the static baseline.

## Methodology
We benchmarked two architectures by forcing them to iterate through 50,000 pod arrivals and extract a sequence of the last 64 arrivals (simulating the inputs required by the LSTM Workload Predictor):

1. **Naive Baseline (Pandas)**: Loads the entire `cleaned_trace.parquet` into RAM and uses `iloc` to slice the past 64 rows on every timestep.
2. **Streaming Engine (PyArrow + Deque)**: Uses `pyarrow.parquet.ParquetFile.iter_batches()` to stream data chunk-by-chunk, and a `collections.deque` for $O(1)$ constant-time append/eviction.

## Results
The ablation proved that the Streaming Data Engine is mathematically and practically superior across all metrics:

| Metric | Naive Pandas | PyArrow Streaming Engine | Improvement |
|--------|--------------|--------------------------|-------------|
| **Throughput (rows/sec)** | 880 | 4,064 | **4.6x Faster** |
| **Peak Memory (MB)** | 6.71 | 1.04 | **6.4x More Efficient** |
| **Total Time (50k rows)** | 56.84s | 12.30s | **~45s Saved** |

## Conclusion
The **PyArrow Streaming Engine** is strictly necessary for this project. If an RL agent trains for 1,000 episodes over the full 150k trace, the Streaming Engine will save approximately **38 hours** of raw computational overhead compared to the baseline Pandas approach.
