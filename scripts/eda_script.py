import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import autocorrelation_plot

class AlibabaTraceEDA:
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set professional aesthetics
        sns.set_theme(style="whitegrid", context="talk")
        plt.rcParams['figure.dpi'] = 300
        
    def run_full_eda(self):
        print("Starting Ultra-Advanced EDA Pipeline...")
        self.load_data()
        self.analyze_nodes()
        self.analyze_pod_demands()
        self.analyze_temporal_patterns()
        self.analyze_qos_and_failures()
        self.analyze_correlations()
        self.analyze_concurrency()
        self.analyze_job_batching()
        self.analyze_seasonality()
        self.analyze_survival_curves()
        print(f"EDA Pipeline Complete. Check {self.output_dir}")

    def load_data(self):
        print("Loading and categorizing files...")
        node_files = glob.glob(os.path.join(self.data_dir, "*node*.csv"))
        cpu_pod_files = glob.glob(os.path.join(self.data_dir, "*pod_list_cpu*.csv"))
        gpu_pod_files = glob.glob(os.path.join(self.data_dir, "*pod_list_gpu*.csv")) + glob.glob(os.path.join(self.data_dir, "*pod_list_multigpu*.csv"))
        
        self.node_df = pd.concat([pd.read_csv(f) for f in node_files], ignore_index=True) if node_files else pd.DataFrame()
        
        cpu_df = pd.concat([pd.read_csv(f) for f in cpu_pod_files], ignore_index=True) if cpu_pod_files else pd.DataFrame()
        gpu_df = pd.concat([pd.read_csv(f) for f in gpu_pod_files], ignore_index=True) if gpu_pod_files else pd.DataFrame()
        
        if not cpu_df.empty: cpu_df['workload_type'] = 'CPU-Bound'
        if not gpu_df.empty: gpu_df['workload_type'] = 'GPU-Bound'
        
        self.pod_df = pd.concat([cpu_df, gpu_df], ignore_index=True)
        
        # Clean Temporal Features globally
        if not self.pod_df.empty:
            for col in ['creation_time', 'scheduled_time', 'deletion_time']:
                self.pod_df[col] = self.pod_df[col].replace(0, np.nan)
                
            self.pod_df['execution_duration'] = self.pod_df['deletion_time'] - self.pod_df['creation_time']
            self.pod_df['scheduling_delay'] = self.pod_df['scheduled_time'] - self.pod_df['creation_time']
            self.pod_df.loc[self.pod_df['execution_duration'] < 0, 'execution_duration'] = np.nan
            self.pod_df.loc[self.pod_df['scheduling_delay'] < 0, 'scheduling_delay'] = np.nan
            self.pod_df['arrival_hour'] = self.pod_df['creation_time'] // 3600
            self.pod_df['arrival_minute'] = self.pod_df['creation_time'] // 60

    def analyze_nodes(self):
        print("Analyzing Cluster Nodes...")
        if self.node_df.empty: return
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        sns.histplot(self.node_df['cpu_milli'], bins=30, color="#3498db", ax=axes[0], kde=True)
        axes[0].set_title("Node CPU Capacity Distribution")
        axes[0].set_xlabel("CPU (milli-cores)")
        
        sns.histplot(self.node_df['memory_mib'], bins=30, color="#2ecc71", ax=axes[1], kde=True)
        axes[1].set_title("Node Memory Capacity Distribution")
        axes[1].set_xlabel("Memory (MiB)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "01_node_capacities.png"))
        plt.close()

    def analyze_pod_demands(self):
        print("Analyzing Pod Workload Demands...")
        if self.pod_df.empty: return
        df = self.pod_df.copy()
        
        # 1. Boxplots for Resource Demands
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        sns.boxplot(data=df, x='workload_type', y='cpu_milli', hue='workload_type', palette="pastel", ax=axes[0], legend=False)
        axes[0].set_yscale('log')
        axes[0].set_title("CPU Request Profile (Log Scale)")
        axes[0].set_ylabel("Milli-cores")
        
        sns.boxplot(data=df, x='workload_type', y='memory_mib', hue='workload_type', palette="pastel", ax=axes[1], legend=False)
        axes[1].set_yscale('log')
        axes[1].set_title("Memory Request Profile (Log Scale)")
        axes[1].set_ylabel("MiB")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "02_pod_resource_demands.png"))
        plt.close()

        # 2. Hexbin Density
        df['log_cpu'] = np.log10(df['cpu_milli'].replace(0, 1))
        df['log_mem'] = np.log10(df['memory_mib'].replace(0, 1))
        g = sns.jointplot(data=df, x='log_cpu', y='log_mem', kind="hex", color="#E6550D", gridsize=40, bins='log')
        g.fig.suptitle("Pod CPU vs Memory Hexbin Density (Log10 Scale)", y=1.02)
        g.set_axis_labels("Log10(CPU Milli-cores)", "Log10(Memory MiB)")
        plt.savefig(os.path.join(self.output_dir, "03_pod_cpu_mem_hexbin.png"))
        plt.close('all')
        
        # 3. GPU Spec Distribution
        if 'gpu_spec' in df.columns:
            spec_df = df.dropna(subset=['gpu_spec'])
            if not spec_df.empty:
                plt.figure(figsize=(14, 7))
                top_specs = spec_df['gpu_spec'].value_counts().nlargest(10).index
                sns.countplot(data=spec_df[spec_df['gpu_spec'].isin(top_specs)], y='gpu_spec', order=top_specs, palette="magma", hue='gpu_spec', legend=False)
                plt.title("Top 10 Requested GPU Architectures")
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, "03b_gpu_specs_requested.png"))
                plt.close()

    def analyze_temporal_patterns(self):
        print("Analyzing Temporal Patterns (Burstiness & Delays)...")
        if self.pod_df.empty: return
        df = self.pod_df.copy()
        
        hourly_counts = df.groupby(['arrival_hour', 'workload_type']).size().reset_index(name='count')
        plt.figure(figsize=(15, 6))
        sns.lineplot(data=hourly_counts, x='arrival_hour', y='count', hue='workload_type', linewidth=2.5)
        plt.title("Workload Arrival Rate (Burstiness)")
        plt.xlabel("Hour (Relative to Trace Start)")
        plt.ylabel("Pods Arriving")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "04_workload_burstiness.png"))
        plt.close()

        plt.figure(figsize=(12, 7))
        sns.ecdfplot(data=df, x='execution_duration', hue='workload_type', linewidth=3)
        plt.xscale('log')
        plt.title("CDF: Pod Execution Durations (Long-Tail)")
        plt.xlabel("Execution Time (Seconds) [Log Scale]")
        plt.ylabel("Cumulative Probability")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "05_duration_cdf.png"))
        plt.close()

        plt.figure(figsize=(12, 7))
        sns.ecdfplot(data=df, x='scheduling_delay', hue='workload_type', linewidth=3)
        plt.xscale('log')
        plt.title("CDF: Pod Scheduling Delays")
        plt.xlabel("Delay Time (Seconds) [Log Scale]")
        plt.ylabel("Cumulative Probability")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "06_scheduling_delay_cdf.png"))
        plt.close()

    def analyze_qos_and_failures(self):
        print("Analyzing QoS classes and Pod Failures...")
        if self.pod_df.empty: return
        df = self.pod_df.copy()

        if 'qos' in df.columns:
            plt.figure(figsize=(12, 6))
            sns.countplot(data=df, x='qos', hue='workload_type', palette="Set2")
            plt.title("Quality of Service (QoS) Distribution")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "07_qos_distribution.png"))
            plt.close()

        if 'pod_phase' in df.columns:
            phase_df = df.dropna(subset=['pod_phase'])
            plt.figure(figsize=(12, 6))
            sns.countplot(data=phase_df, x='pod_phase', hue='workload_type', palette="Set1")
            plt.title("Pod Phase (Success vs Failure) Distribution")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "08_pod_phase_distribution.png"))
            plt.close()

            succ_fail_df = df[df['pod_phase'].isin(['Succeeded', 'Failed'])]
            if not succ_fail_df.empty:
                plt.figure(figsize=(12, 7))
                sns.boxplot(data=succ_fail_df, x='pod_phase', y='cpu_milli', hue='workload_type', palette="Set3")
                plt.yscale('log')
                plt.title("CPU Demand: Succeeded vs Failed Pods")
                plt.ylabel("CPU (Milli-cores) [Log Scale]")
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, "09_failure_vs_cpu_demand.png"))
                plt.close()

    def analyze_correlations(self):
        print("Analyzing Feature Correlations...")
        if self.pod_df.empty: return
        df = self.pod_df.copy()
        numeric_cols = ['cpu_milli', 'memory_mib', 'num_gpu', 'gpu_milli', 'execution_duration', 'scheduling_delay']
        if 'pod_phase' in df.columns:
            df['is_failed'] = (df['pod_phase'] == 'Failed').astype(int)
            numeric_cols.append('is_failed')
            
        corr = df[numeric_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title("Correlation Matrix for Pod Resources and Lifecycle")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "10_correlation_matrix.png"))
        plt.close()

    def analyze_concurrency(self):
        print("Analyzing Global Concurrency (Simultaneous Pods)...")
        if self.pod_df.empty: return
        df = self.pod_df.dropna(subset=['creation_time', 'deletion_time']).copy()
        
        # Create events for arrival (+1) and departure (-1)
        events = pd.concat([
            pd.DataFrame({'time': df['creation_time'], 'event': 1}),
            pd.DataFrame({'time': df['deletion_time'], 'event': -1})
        ]).sort_values('time')
        
        events['concurrent_pods'] = events['event'].cumsum()
        
        # Subsample for plotting so it doesn't crash memory
        events_sampled = events.iloc[::max(1, len(events)//5000)]
        
        plt.figure(figsize=(15, 6))
        sns.lineplot(data=events_sampled, x='time', y='concurrent_pods', color="#9b59b6")
        plt.title("Global Cluster Concurrency (Running Pods over Time)")
        plt.xlabel("Time (Seconds from Trace Start)")
        plt.ylabel("Simultaneous Running Pods")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "11_global_concurrency.png"))
        plt.close()

    def analyze_job_batching(self):
        print("Analyzing Batch Job Grouping (Simultaneous Arrivals)...")
        if self.pod_df.empty: return
        df = self.pod_df.dropna(subset=['creation_time']).copy()
        
        # Group by exact creation time (to the second) to find "Bag of Tasks" batches
        batch_sizes = df.groupby('creation_time').size()
        
        plt.figure(figsize=(12, 6))
        sns.histplot(batch_sizes, bins=50, color="#f1c40f", log_scale=(True, True))
        plt.title("Job Batch Size Distribution (Pods Submitted Simultaneously)")
        plt.xlabel("Number of Pods Submitted in the Same Second (Log Scale)")
        plt.ylabel("Frequency (Log Scale)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "12_job_batch_sizes.png"))
        plt.close()

    def analyze_seasonality(self):
        print("Analyzing Autocorrelation / Seasonality in Arrivals...")
        if self.pod_df.empty: return
        df = self.pod_df.copy()
        
        # Arrival counts per minute
        minute_counts = df.groupby('arrival_minute').size()
        
        plt.figure(figsize=(15, 6))
        if len(minute_counts) > 100:
            autocorrelation_plot(minute_counts)
            plt.title("Autocorrelation of Workload Arrivals (Seasonality Detection)")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "13_arrival_autocorrelation.png"))
        plt.close()

    def analyze_survival_curves(self):
        print("Analyzing Execution Survival Probabilities...")
        if self.pod_df.empty: return
        df = self.pod_df.dropna(subset=['execution_duration']).copy()
        
        plt.figure(figsize=(12, 7))
        for w_type in df['workload_type'].unique():
            subset = df[df['workload_type'] == w_type]['execution_duration'].sort_values()
            y = 1.0 - np.arange(1, len(subset) + 1) / len(subset) # 1 - CDF = Survival
            plt.plot(subset, y, label=w_type, linewidth=2.5)
            
        plt.xscale('log')
        plt.title("Survival Curve: Probability of a Pod Continuing to Run")
        plt.xlabel("Time Running (Seconds) [Log Scale]")
        plt.ylabel("Probability of Survival")
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "14_survival_curve.png"))
        plt.close()

if __name__ == "__main__":
    data_path = r"C:\Users\devan\OneDrive\Desktop\BTech\data\raw"
    out_path = r"C:\Users\devan\OneDrive\Desktop\BTech\eda\plots"
    eda = AlibabaTraceEDA(data_path, out_path)
    eda.run_full_eda()
