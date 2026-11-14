import os
import time
import collections
import pyarrow.parquet as pq
import numpy as np
import pandas as pd

class StreamingIngester:
    """
    Simulates a live Kubernetes cluster by streaming pre-processed data 
    tick-by-tick (or chunk-by-chunk) from a Parquet file.
    Reads the massive trace data row-by-row using PyArrow iterators.
    """
    def __init__(self, data_path, batch_size=1, start_offset=0):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Cleaned trace not found at {data_path}. Please run data_processor.py first.")
            
        self.data_path = data_path
        self.batch_size = batch_size
        self.start_offset = start_offset
        self.row_index = start_offset
        self.parquet_file = pq.ParquetFile(self.data_path)
        
        # Generator for streaming
        self._stream = self._create_stream()
        
    def _create_stream(self):
        """Yields chunks of data to avoid loading the entire dataset into memory."""
        rows_skipped = 0
        # Use fast batch skipping for large start offsets
        skip_chunk_size = 5000
        for batch in self.parquet_file.iter_batches(batch_size=skip_chunk_size):
            num_rows = batch.num_rows
            if rows_skipped + num_rows <= self.start_offset:
                rows_skipped += num_rows
                continue
            df_chunk = batch.to_pandas()
            for _, row in df_chunk.iterrows():
                if rows_skipped < self.start_offset:
                    rows_skipped += 1
                    continue
                yield row.to_dict()
                
    def step(self):
        """Simulate one time tick, returning the next arrived pod(s)."""
        try:
            pod = next(self._stream)
            self.row_index += 1
            return pod
        except StopIteration:
            return None # End of trace
            
    def reset(self):
        """Reset the stream to the beginning (useful for RL episodes)."""
        self.row_index = self.start_offset
        self._stream = self._create_stream()


class RollingWindowBuffer:
    """
    Maintains a constant-size temporal memory of the last N arrivals.
    This is critical for the LSTM Workload Predictor, which requires 
    a historical sequence to predict future bursts.
    """
    def __init__(self, window_size, feature_keys):
        """
        :param window_size: Number of past time steps to remember.
        :param feature_keys: List of keys to extract from the incoming pod dictionary.
        """
        self.window_size = window_size
        self.feature_keys = feature_keys
        
        # Use a deque with a maxlen for $O(1)$ fast appends and auto-eviction of old data
        self.buffer = collections.deque(maxlen=window_size)
        
    def __len__(self):
        return len(self.buffer)
        
    def add(self, pod_event):
        """Add a new pod event to the rolling window."""
        # Extract only the necessary features (e.g., cpu_norm, mem_norm, workload_type)
        state_vector = [pod_event.get(k, 0.0) for k in self.feature_keys]
        self.buffer.append(state_vector)
        
    def get_sequence(self):
        """
        Returns the current rolling window as a numpy array of shape (window_size, num_features).
        If the buffer is not full, it pads with zeros at the beginning.
        """
        seq = list(self.buffer)
        num_features = len(self.feature_keys)
        
        if len(seq) < self.window_size:
            # Zero-pad the sequence if we haven't seen enough history yet
            padding = [[0.0] * num_features] * (self.window_size - len(seq))
            seq = padding + seq
            
        return np.array(seq, dtype=np.float32)

    def is_full(self):
        return len(self.buffer) == self.window_size

    def reset(self):
        self.buffer.clear()
