import os
import glob
import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def run_pipeline(self):
        print("Starting Offline Data Preprocessing Pipeline...")
        df = self.load_raw_data()
        df = self.clean_temporal_data(df)
        df = self.normalize_features(df)
        df = self.sort_and_format(df)
        self.save_data(df)
        print("Pipeline Complete!")

    def load_raw_data(self):
        print("Loading raw CSVs...")
        cpu_files = glob.glob(os.path.join(self.data_dir, "*pod_list_cpu*.csv"))
        gpu_files = glob.glob(os.path.join(self.data_dir, "*pod_list_gpu*.csv")) + glob.glob(os.path.join(self.data_dir, "*pod_list_multigpu*.csv"))
        
        cpu_df = pd.concat([pd.read_csv(f) for f in cpu_files], ignore_index=True) if cpu_files else pd.DataFrame()
        gpu_df = pd.concat([pd.read_csv(f) for f in gpu_files], ignore_index=True) if gpu_files else pd.DataFrame()
        
        if not cpu_df.empty: cpu_df['workload_type'] = 0 # 0 for CPU
        if not gpu_df.empty: gpu_df['workload_type'] = 1 # 1 for GPU
        
        df = pd.concat([cpu_df, gpu_df], ignore_index=True)
        print(f"Loaded {len(df)} total pods.")
        return df

    def clean_temporal_data(self, df):
        print("Cleaning missing timestamps...")
        
        # Replace 0 with NaN for temporal columns
        for col in ['creation_time', 'scheduled_time', 'deletion_time']:
            df[col] = df[col].replace(0, np.nan)
            
        # Drop pods that have no creation_time (useless for streaming)
        df = df.dropna(subset=['creation_time'])
        
        # Impute missing scheduled_time with creation_time (assume instant scheduling if missing)
        df['scheduled_time'] = df['scheduled_time'].fillna(df['creation_time'])
        
        # Impute missing deletion_time (running services) with the max time in the trace
        max_time = df['creation_time'].max() + 3600 # Add 1 hour buffer
        df['deletion_time'] = df['deletion_time'].fillna(max_time)
        
        # Calculate derived features
        df['execution_duration'] = df['deletion_time'] - df['creation_time']
        df['scheduling_delay'] = df['scheduled_time'] - df['creation_time']
        
        # Drop invalid negative times (data corruption)
        df = df[df['execution_duration'] >= 0]
        df = df[df['scheduling_delay'] >= 0]
        
        print(f"Remaining pods after temporal cleaning: {len(df)}")
        return df

    def normalize_features(self, df):
        print("Normalizing resource demands...")
        
        # Fill missing numeric values
        df['cpu_milli'] = df['cpu_milli'].fillna(0)
        df['memory_mib'] = df['memory_mib'].fillna(0)
        
        # Normalize to [0, 1] using robust Min-Max scaling
        cpu_max = df['cpu_milli'].max()
        mem_max = df['memory_mib'].max()
        
        df['cpu_norm'] = df['cpu_milli'] / cpu_max if cpu_max > 0 else 0
        df['mem_norm'] = df['memory_mib'] / mem_max if mem_max > 0 else 0
        
        print(f"Normalized CPU max: {cpu_max}, Mem max: {mem_max}")
        return df

    def sort_and_format(self, df):
        print("Sorting by timeline...")
        # Sort chronologically by creation time for the streaming engine
        df = df.sort_values(by='creation_time').reset_index(drop=True)
        
        # Encode categorical features
        if 'qos' in df.columns:
            # Map QOS to integer categories. LS=0, BE=1, others=2
            qos_map = {'LS': 0, 'BE': 1}
            df['qos_encoded'] = df['qos'].map(qos_map).fillna(2).astype(int)
            
        return df

    def save_data(self, df):
        out_path = os.path.join(self.output_dir, "cleaned_trace.parquet")
        print(f"Saving to highly compressed parquet format: {out_path}")
        # Parquet is 10x faster and smaller than CSV
        df.to_parquet(out_path, engine='pyarrow', index=False)
        print("Done!")

if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(PROJECT_ROOT, "data", "raw")
    out_path = os.path.join(PROJECT_ROOT, "data", "processed")
    
    processor = DataProcessor(data_path, out_path)
    processor.run_pipeline()
