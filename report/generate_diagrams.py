import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUT_DIR = os.path.abspath("report/figures")
os.makedirs(OUT_DIR, exist_ok=True)

def create_overall_architecture_flowchart():
    fig, ax = plt.subplots(figsize=(14, 8), dpi=180)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.axis('off')
    
    # Title
    ax.text(7, 7.6, "Predictive-RL Autoscaling Framework: End-to-End System Architecture", 
            ha='center', va='center', fontsize=14, fontweight='bold', color='#0f172a')
    
    def draw_box(x, y, w, h, title, subtitle, bg_color, border_color):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15", 
                                      ec=border_color, fc=bg_color, lw=1.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.62, title, ha='center', va='center', 
                fontsize=9.5, fontweight='bold', color='#0f172a')
        ax.text(x + w/2, y + h*0.28, subtitle, ha='center', va='center', 
                fontsize=7.5, color='#475569', style='italic')

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.8, color='#0284c7', shrinkA=4, shrinkB=4))
        if label:
            mx, my = (x1 + x2)/2, (y1 + y2)/2
            ax.text(mx, my + 0.15, label, ha='center', va='bottom', fontsize=7.5, fontweight='semibold', color='#0369a1')

    # Row 1: Data Ingestion & Cluster Trace
    draw_box(0.5, 5.8, 2.8, 1.2, "Alibaba Cluster Trace", "156k+ Pods, 1.8k Nodes\nCPU & GPU Heterogeneity", "#f1f5f9", "#94a3b8")
    draw_box(4.2, 5.8, 3.2, 1.2, "PyArrow Streaming Engine", "Zero-Copy IterBatches\nO(1) Deque Buffer (W=64)", "#e0f2fe", "#0284c7")
    draw_box(8.5, 5.8, 4.8, 1.2, "Predictive Oracles Ensemble", "Transformer (Workload), XGBoost (Delay)\nPyTorch Deep NN (Server Wattage)", "#fef3c7", "#d97706")

    draw_arrow(3.3, 6.4, 4.2, 6.4, "Raw Trace")
    draw_arrow(7.4, 6.4, 8.5, 6.4, "Sliding Window")

    # Row 2: State Fusion & PPO Agent
    draw_box(8.5, 3.8, 4.8, 1.2, "Composite State Fusion Layer", "Raw Pod [3D] + Cluster Telemetry [4D]\n+ Oracle Forecasts [4D] -> 11D State", "#f3e8ff", "#7e22ce")
    draw_box(4.2, 3.8, 3.2, 1.2, "Hierarchical PPO Policy", "Continuous Actor-Critic\n[ΔCPU_Pool, ΔGPU_Pool] in [-1, 1]²", "#dbeafe", "#2563eb")
    draw_box(0.5, 3.8, 2.8, 1.2, "Safe Action Shield", "Barrier: N_safe = Demand / 6.4\nClamps Unsafe Scale-downs", "#dcfce7", "#16a34a")

    draw_arrow(10.9, 5.8, 10.9, 5.0, "Predictions")
    draw_arrow(8.5, 4.4, 7.4, 4.4, "11D Fused State")
    draw_arrow(4.2, 4.4, 3.3, 4.4, "Tentative Action")

    # Row 3: Cluster Execution & Multi-Objective Feedback
    draw_box(0.5, 1.4, 4.5, 1.5, "Heterogeneous K8s Cluster", "• CPU Pool (16-core nodes, standard compute)\n• GPU Pool (8x GPU accelerators, high TDP)", "#f8fafc", "#64748b")
    draw_box(6.2, 1.4, 4.5, 1.5, "Multi-Objective Reward & Cost", "• Electrical Wattage & Carbon Footprint\n• Contractual SLA Violations (Util > 80%)\n• Shield Intervention Penalty (-50.0)", "#fee2e2", "#dc2626")

    draw_arrow(1.9, 3.8, 1.9, 2.9, "Clamped Safe Action")
    draw_arrow(5.0, 2.15, 6.2, 2.15, "Utilization & Nodes")
    
    # Feedback arrow from Reward to PPO
    ax.annotate('', xy=(5.8, 3.8), xytext=(8.4, 2.9),
                arrowprops=dict(arrowstyle="->", lw=1.8, color='#dc2626', connectionstyle="arc3,rad=-0.2"))
    ax.text(7.6, 3.4, "Reward Feedback: - (Power + SLA)", ha='center', va='center', fontsize=7.5, fontweight='bold', color='#dc2626')

    # Feedback arrow from Cluster to Ingestion / Telemetry
    ax.annotate('', xy=(12.0, 3.8), xytext=(5.0, 1.4),
                arrowprops=dict(arrowstyle="->", lw=1.5, color='#64748b', linestyle="--", connectionstyle="arc3,rad=0.35"))
    ax.text(8.5, 0.65, "Live Cluster Telemetry (Active Nodes, Pool Utilization)", ha='center', va='center', fontsize=8, color='#475569')

    ax.set_xlim(0, 14)
    ax.set_ylim(0.5, 8.0)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "00_system_architecture_flowchart.png")
    plt.savefig(p)
    plt.close(fig)
    print("Saved 00_system_architecture_flowchart.png")

def create_streaming_buffer_diagram():
    fig, ax = plt.subplots(figsize=(13, 6), dpi=180)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.axis('off')
    
    ax.text(6.5, 5.5, "PyArrow Zero-Copy Ingestion & O(1) Rolling Window Deque Architecture", 
            ha='center', va='center', fontsize=13, fontweight='bold', color='#0f172a')
    
    # Blocks
    # 1. Parquet on Disk
    rect1 = patches.FancyBboxPatch((0.5, 2.2), 2.5, 2.4, boxstyle="round,pad=0.15", ec="#64748b", fc="#f8fafc", lw=1.6)
    ax.add_patch(rect1)
    ax.text(1.75, 4.1, "Alibaba Trace File", ha='center', va='center', fontsize=10, fontweight='bold', color='#0f172a')
    ax.text(1.75, 3.5, "cleaned_trace.parquet\n(156,383 rows, compressed)", ha='center', va='center', fontsize=8, color='#475569')
    ax.text(1.75, 2.7, "Columns: cpu_norm,\nmem_norm, workload_type", ha='center', va='center', fontsize=7.5, color='#64748b')

    # 2. PyArrow iter_batches
    rect2 = patches.FancyBboxPatch((3.8, 2.2), 3.0, 2.4, boxstyle="round,pad=0.15", ec="#0284c7", fc="#e0f2fe", lw=1.6)
    ax.add_patch(rect2)
    ax.text(5.3, 4.1, "PyArrow Batch Ingester", ha='center', va='center', fontsize=10, fontweight='bold', color='#0369a1')
    ax.text(5.3, 3.4, "iter_batches(chunk_size=5000)\nZero-Copy Memory Mapping", ha='center', va='center', fontsize=8, color='#0284c7')
    ax.text(5.3, 2.6, "Fast offset skip: 100k in 0.01s\nThroughput: 4,064 rows/s", ha='center', va='center', fontsize=7.5, color='#0369a1')

    # 3. Rolling Window Deque
    rect3 = patches.FancyBboxPatch((7.6, 2.2), 4.8, 2.4, boxstyle="round,pad=0.15", ec="#16a34a", fc="#dcfce7", lw=1.6)
    ax.add_patch(rect3)
    ax.text(10.0, 4.1, "Rolling Window Buffer (collections.deque)", ha='center', va='center', fontsize=10, fontweight='bold', color='#15803d')
    ax.text(10.0, 3.4, "Fixed Window: W = 64 events | O(1) Amortized Append & Eviction", ha='center', va='center', fontsize=8, color='#16a34a')
    
    # Deque visual slots
    for i in range(7):
        slot_x = 8.0 + i*0.58
        color = "#bbf7d0" if i < 6 else "#86efac"
        txt = f"t-{63-i*10}" if i < 6 else "t (new)"
        r = patches.Rectangle((slot_x, 2.5), 0.52, 0.55, ec="#15803d", fc=color, lw=1.0)
        ax.add_patch(r)
        ax.text(slot_x + 0.26, 2.77, txt, ha='center', va='center', fontsize=6.5, color='#14532d')
    
    ax.text(10.0, 2.38, "Auto Zero-Padding if len < 64 -> Output Shape: (64, 3)", ha='center', va='center', fontsize=7.5, fontweight='bold', color='#15803d')

    # Arrows
    ax.annotate('', xy=(3.8, 3.4), xytext=(3.0, 3.4), arrowprops=dict(arrowstyle="->", lw=1.8, color='#0284c7'))
    ax.annotate('', xy=(7.6, 3.4), xytext=(6.8, 3.4), arrowprops=dict(arrowstyle="->", lw=1.8, color='#0284c7'))

    # Output arrow to Oracles
    ax.annotate('', xy=(10.0, 1.2), xytext=(10.0, 2.2), arrowprops=dict(arrowstyle="->", lw=2.0, color='#d97706'))
    ax.text(10.0, 1.0, "Feeds Time-Series Transformer / LSTM Oracles (B, 64, 3)", ha='center', va='top', fontsize=9, fontweight='bold', color='#b45309')

    ax.set_xlim(0, 13)
    ax.set_ylim(0.5, 6.0)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "00_streaming_deque_architecture.png")
    plt.savefig(p)
    plt.close(fig)
    print("Saved 00_streaming_deque_architecture.png")

def create_multi_horizon_transformer_diagram():
    fig, ax = plt.subplots(figsize=(14, 7), dpi=180)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.axis('off')
    
    ax.text(7, 6.6, "Multi-Horizon Lookahead Transformer Architecture", 
            ha='center', va='center', fontsize=13, fontweight='bold', color='#0f172a')
    
    # 1. Input sequence
    r1 = patches.FancyBboxPatch((0.5, 2.8), 2.2, 2.6, boxstyle="round,pad=0.15", ec="#64748b", fc="#f8fafc", lw=1.6)
    ax.add_patch(r1)
    ax.text(1.6, 4.7, "Input Sequence", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#0f172a')
    ax.text(1.6, 4.0, "Shape: (B, 64, 3)\n[cpu_norm, mem_norm,\nworkload_type]", ha='center', va='center', fontsize=8, color='#475569')
    ax.text(1.6, 3.1, "64 Historical Ticks", ha='center', va='center', fontsize=7.5, color='#64748b')

    # 2. Linear Projection + Positional Encoding
    r2 = patches.FancyBboxPatch((3.4, 2.8), 2.5, 2.6, boxstyle="round,pad=0.15", ec="#0284c7", fc="#e0f2fe", lw=1.6)
    ax.add_patch(r2)
    ax.text(4.65, 4.7, "Projection & PE", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#0369a1')
    ax.text(4.65, 4.0, "Linear(3 -> d_model=64)\n+\nSinusoidal Positional\nEncoding", ha='center', va='center', fontsize=8, color='#0284c7')
    ax.text(4.65, 3.1, "PE(pos, 2i) = sin(...)", ha='center', va='center', fontsize=7.5, color='#0369a1')

    # 3. Transformer Encoder
    r3 = patches.FancyBboxPatch((6.6, 2.4), 3.0, 3.4, boxstyle="round,pad=0.15", ec="#7e22ce", fc="#f3e8ff", lw=1.8)
    ax.add_patch(r3)
    ax.text(8.1, 5.3, "Transformer Encoder", ha='center', va='center', fontsize=10, fontweight='bold', color='#6b21a8')
    ax.text(8.1, 4.6, "• 2 Encoder Layers\n• 4 Attention Heads\n• Feedforward: 128\n• LayerNorm & Dropout (0.1)", ha='center', va='center', fontsize=8, color='#7e22ce')
    ax.text(8.1, 3.5, "Self-Attention Mechanism:\nsoftmax(Q K^T / √d) V", ha='center', va='center', fontsize=7.5, style='italic', color='#6b21a8')
    ax.text(8.1, 2.8, "Extract Last Hidden State:\nencoded[:, -1, :] in R^64", ha='center', va='center', fontsize=7.5, fontweight='bold', color='#581c87')

    # 4. Decoupled Horizon Heads
    # Head 1
    r_h1 = patches.FancyBboxPatch((10.3, 4.6), 3.2, 1.2, boxstyle="round,pad=0.1", ec="#16a34a", fc="#dcfce7", lw=1.5)
    ax.add_patch(r_h1)
    ax.text(11.9, 5.35, "Head 1: Horizon t+1", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#15803d')
    ax.text(11.9, 4.9, "Immediate pod arrival burst | MSE: 0.000680\nLinear(64 -> 32 -> 2)", ha='center', va='center', fontsize=7.5, color='#16a34a')

    # Head 2
    r_h2 = patches.FancyBboxPatch((10.3, 3.1), 3.2, 1.2, boxstyle="round,pad=0.1", ec="#2563eb", fc="#dbeafe", lw=1.5)
    ax.add_patch(r_h2)
    ax.text(11.9, 3.85, "Head 2: Horizon t+5", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#1e40af')
    ax.text(11.9, 3.4, "Near-term queue buildup | MSE: 0.001387\nLinear(64 -> 32 -> 2)", ha='center', va='center', fontsize=7.5, color='#2563eb')

    # Head 3
    r_h3 = patches.FancyBboxPatch((10.3, 1.6), 3.2, 1.2, boxstyle="round,pad=0.1", ec="#d97706", fc="#fef3c7", lw=1.5)
    ax.add_patch(r_h3)
    ax.text(11.9, 2.35, "Head 3: Horizon t+15", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#b45309')
    ax.text(11.9, 1.9, "60s VM Cold-Boot Window | MSE: 0.001812\nLinear(64 -> 32 -> 2)", ha='center', va='center', fontsize=7.5, color='#d97706')

    # Arrows
    ax.annotate('', xy=(3.4, 4.1), xytext=(2.7, 4.1), arrowprops=dict(arrowstyle="->", lw=1.8, color='#0284c7'))
    ax.annotate('', xy=(6.6, 4.1), xytext=(5.9, 4.1), arrowprops=dict(arrowstyle="->", lw=1.8, color='#0284c7'))
    
    # Split arrows from encoder to 3 heads
    ax.annotate('', xy=(10.3, 5.2), xytext=(9.6, 4.1), arrowprops=dict(arrowstyle="->", lw=1.6, color='#16a34a'))
    ax.annotate('', xy=(10.3, 3.7), xytext=(9.6, 4.1), arrowprops=dict(arrowstyle="->", lw=1.6, color='#2563eb'))
    ax.annotate('', xy=(10.3, 2.2), xytext=(9.6, 4.1), arrowprops=dict(arrowstyle="->", lw=1.6, color='#d97706'))

    # Final output label
    ax.text(11.9, 0.9, "Stacked Output: (B, 3, 2) [CPU demand, Mem demand]", 
            ha='center', va='center', fontsize=8.5, fontweight='bold', color='#0f172a')

    ax.set_xlim(0, 14)
    ax.set_ylim(0.5, 7.2)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "00_multi_horizon_transformer_arch.png")
    plt.savefig(p)
    plt.close(fig)
    print("Saved 00_multi_horizon_transformer_arch.png")

def create_safe_shield_flowchart():
    fig, ax = plt.subplots(figsize=(11, 8), dpi=180)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.axis('off')
    
    ax.text(5.5, 7.5, "Safe RL Action Shield: Control Barrier Verification Logic", 
            ha='center', va='center', fontsize=13, fontweight='bold', color='#0f172a')

    def draw_f_box(x, y, w, h, text, bg, border):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12", ec=border, fc=bg, lw=1.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=8.5, fontweight='semibold', color='#0f172a')

    # Step 1: Input Action
    draw_f_box(3.5, 6.2, 4.0, 0.8, "PPO Policy Proposed Action\na_t in [-1.0, 1.0]", "#e0f2fe", "#0284c7")

    # Step 2: Compute Safe Boundary
    draw_f_box(2.5, 4.8, 6.0, 0.9, "Calculate Minimum Safe Boundary Capacity:\nN_safe = CPU_Demand_t / (8.0 * 0.8) = CPU_Demand_t / 6.4", "#fef3c7", "#d97706")

    # Step 3: Tentative Node
    draw_f_box(2.8, 3.5, 5.4, 0.8, "Compute Tentative Capacity:\nN_tentative = N_current + (a_t * 5.0)", "#f1f5f9", "#64748b")

    # Diamond: Condition
    diamond = patches.Polygon([[5.5, 2.8], [7.3, 2.0], [5.5, 1.2], [3.7, 2.0]], 
                              closed=True, ec="#7e22ce", fc="#f3e8ff", lw=1.8)
    ax.add_patch(diamond)
    ax.text(5.5, 2.0, "Is N_tentative\n< N_safe ?", ha='center', va='center', fontsize=8, fontweight='bold', color='#6b21a8')

    # Branch 1: YES (Unsafe Scale-down detected)
    draw_f_box(0.2, 0.4, 4.5, 1.1, "SHIELD INTERVENTION:\n• Clamp: a_t* = (N_safe - N_current) / 5.0\n• Inject Penalty: R_t += -50.0\n• Increment: interventions += 1", "#fee2e2", "#dc2626")

    # Branch 2: NO (Safe Action)
    draw_f_box(6.3, 0.4, 4.5, 1.1, "PASS-THROUGH DIRECT ACTION:\n• Execute original action: a_t\n• Zero intervention penalty\n• Normal continuous execution", "#dcfce7", "#16a34a")

    # Connectors
    ax.annotate('', xy=(5.5, 5.7), xytext=(5.5, 6.2), arrowprops=dict(arrowstyle="->", lw=1.6, color='#0284c7'))
    ax.annotate('', xy=(5.5, 4.3), xytext=(5.5, 4.8), arrowprops=dict(arrowstyle="->", lw=1.6, color='#d97706'))
    ax.annotate('', xy=(5.5, 2.8), xytext=(5.5, 3.5), arrowprops=dict(arrowstyle="->", lw=1.6, color='#64748b'))
    
    # Diamond branches
    ax.annotate('', xy=(2.4, 1.5), xytext=(3.7, 2.0), arrowprops=dict(arrowstyle="->", lw=1.8, color='#dc2626'))
    ax.text(2.8, 1.9, "YES (Violation Risk)", ha='center', va='bottom', fontsize=8, fontweight='bold', color='#dc2626')

    ax.annotate('', xy=(8.6, 1.5), xytext=(7.3, 2.0), arrowprops=dict(arrowstyle="->", lw=1.8, color='#16a34a'))
    ax.text(8.2, 1.9, "NO (Compliant)", ha='center', va='bottom', fontsize=8, fontweight='bold', color='#16a34a')

    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8.0)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "00_safe_shield_flowchart.png")
    plt.savefig(p)
    plt.close(fig)
    print("Saved 00_safe_shield_flowchart.png")

def create_hierarchical_control_diagram():
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=180)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.axis('off')
    
    ax.text(6.5, 6.0, "Decoupled Hierarchical Control Architecture for Heterogeneous GPU Clusters", 
            ha='center', va='center', fontsize=13, fontweight='bold', color='#0f172a')
    
    # Left: 11D State
    r_state = patches.FancyBboxPatch((0.5, 1.8), 2.8, 3.4, boxstyle="round,pad=0.15", ec="#7e22ce", fc="#f3e8ff", lw=1.6)
    ax.add_patch(r_state)
    ax.text(1.9, 4.8, "11D Composite State", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#581c87')
    ax.text(1.9, 3.4, "• Raw Pod Demands [3D]\n• Active CPU Nodes / 100\n• Active GPU Nodes / 40\n• Last CPU Utilization\n• Last GPU Utilization\n• Oracle Forecasts [4D]", ha='center', va='center', fontsize=8, color='#7e22ce')

    # Middle: PPO Policy
    r_ppo = patches.FancyBboxPatch((4.0, 2.2), 3.0, 2.6, boxstyle="round,pad=0.15", ec="#2563eb", fc="#dbeafe", lw=1.8)
    ax.add_patch(r_ppo)
    ax.text(5.5, 4.3, "Hierarchical PPO Actor", ha='center', va='center', fontsize=10, fontweight='bold', color='#1e40af')
    ax.text(5.5, 3.4, "MLP Policy Network\nGaussian Action Dist.\na_t = [a_cpu, a_gpu] in [-1, 1]²", ha='center', va='center', fontsize=8, color='#2563eb')
    ax.text(5.5, 2.5, "Decoupled Multi-Channel", ha='center', va='center', fontsize=7.5, style='italic', color='#1e3a8a')

    # Right Top: CPU Pool
    r_cpu = patches.FancyBboxPatch((7.8, 3.6), 4.7, 1.8, boxstyle="round,pad=0.15", ec="#0284c7", fc="#e0f2fe", lw=1.6)
    ax.add_patch(r_cpu)
    ax.text(10.15, 5.0, "Pool 0: CPU-Optimized Server Pool", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#0369a1')
    ax.text(10.15, 4.2, "• 16 cores per node, standard compute tier (c6i.4xlarge)\n• Continuous scaling velocity: ΔN_cpu = a_cpu * 5.0\n• Capacity range: N_cpu in [1, 100] nodes", ha='center', va='center', fontsize=7.5, color='#0284c7')

    # Right Bottom: GPU Pool
    r_gpu = patches.FancyBboxPatch((7.8, 1.2), 4.7, 1.8, boxstyle="round,pad=0.15", ec="#d97706", fc="#fef3c7", lw=1.6)
    ax.add_patch(r_gpu)
    ax.text(10.15, 2.6, "Pool 1: GPU-Accelerated Server Pool", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#b45309')
    ax.text(10.15, 1.8, "• 8 GPUs + 32 CPU cores per node (g4dn.12xlarge)\n• Continuous scaling velocity: ΔN_gpu = a_gpu * 2.0\n• Severe penalty for starvation on expensive accelerator tier", ha='center', va='center', fontsize=7.5, color='#d97706')

    # Arrows
    ax.annotate('', xy=(4.0, 3.5), xytext=(3.3, 3.5), arrowprops=dict(arrowstyle="->", lw=1.8, color='#2563eb'))
    ax.annotate('', xy=(7.8, 4.5), xytext=(7.0, 3.8), arrowprops=dict(arrowstyle="->", lw=1.8, color='#0284c7'))
    ax.annotate('', xy=(7.8, 2.1), xytext=(7.0, 3.2), arrowprops=dict(arrowstyle="->", lw=1.8, color='#d97706'))

    ax.set_xlim(0, 13)
    ax.set_ylim(0.5, 6.8)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "00_hierarchical_control_diagram.png")
    plt.savefig(p)
    plt.close(fig)
    print("Saved 00_hierarchical_control_diagram.png")

def create_cmdp_lagrangian_flowchart():
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=180)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.axis('off')
    
    ax.text(6.0, 6.0, "Constrained MDP via PPO-Lagrangian: Two-Timescale Dual Ascent", 
            ha='center', va='center', fontsize=13, fontweight='bold', color='#0f172a')
    
    def d_box(x, y, w, h, t1, t2, fc, ec):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12", ec=ec, fc=fc, lw=1.6)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.62, t1, ha='center', va='center', fontsize=9, fontweight='bold', color='#0f172a')
        ax.text(x + w/2, y + h*0.28, t2, ha='center', va='center', fontsize=7.5, color='#475569')

    d_box(0.5, 3.4, 3.2, 1.4, "Environment Rollout", "PPO Policy interacts with K8sScalingEnv\nRecords: I(U_t > 0.8) at every step", "#f8fafc", "#64748b")
    d_box(4.4, 3.4, 3.2, 1.4, "Rollout End Evaluation", "Mean SLA violation fraction:\nC_sla = (1/T) Σ I(U_t > 0.8)", "#e0f2fe", "#0284c7")
    d_box(8.3, 3.4, 3.2, 1.4, "Dual Gradient Ascent", "Error = C_sla - ε_sla (ε_sla = 0.05)\nλ_{k+1} = max(0, λ_k + α * Error)", "#fef3c7", "#d97706")
    
    d_box(4.4, 1.0, 5.0, 1.4, "Lagrangian Reward Augmentation", "R_t = - C_power - λ_k * C_sla\nGuides actor away from constraint boundary", "#f3e8ff", "#7e22ce")

    # Arrows
    ax.annotate('', xy=(4.4, 4.1), xytext=(3.7, 4.1), arrowprops=dict(arrowstyle="->", lw=1.6, color='#0284c7'))
    ax.annotate('', xy=(8.3, 4.1), xytext=(7.6, 4.1), arrowprops=dict(arrowstyle="->", lw=1.6, color='#d97706'))
    ax.annotate('', xy=(8.3, 2.4), xytext=(9.9, 3.4), arrowprops=dict(arrowstyle="->", lw=1.6, color='#7e22ce'))
    ax.annotate('', xy=(2.1, 3.4), xytext=(4.4, 1.7), arrowprops=dict(arrowstyle="->", lw=1.6, color='#7e22ce', connectionstyle="arc3,rad=0.2"))

    ax.text(2.6, 2.2, "Updated Reward Signal for Next Rollout", ha='center', va='center', fontsize=7.5, fontweight='semibold', color='#7e22ce')

    ax.set_xlim(0, 12)
    ax.set_ylim(0.5, 6.8)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "00_cmdp_lagrangian_flowchart.png")
    plt.savefig(p)
    plt.close(fig)
    print("Saved 00_cmdp_lagrangian_flowchart.png")

def create_cloud_deployment_topology():
    fig, ax = plt.subplots(figsize=(13, 7), dpi=180)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    ax.axis('off')
    
    ax.text(6.5, 6.6, "Production Kubernetes Cloud Deployment Topology & KEDA Integration", 
            ha='center', va='center', fontsize=13, fontweight='bold', color='#0f172a')
    
    def d_box(x, y, w, h, t1, t2, fc, ec):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12", ec=ec, fc=fc, lw=1.6)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.62, t1, ha='center', va='center', fontsize=9, fontweight='bold', color='#0f172a')
        ax.text(x + w/2, y + h*0.28, t2, ha='center', va='center', fontsize=7.5, color='#475569')

    # Top left: Worker nodes & metrics
    d_box(0.5, 4.4, 3.2, 1.4, "K8s Worker Nodes", "cAdvisor + node-exporter\nExports CPU, Mem, GPU telemetry", "#f8fafc", "#64748b")
    # Top center: Prometheus
    d_box(4.5, 4.4, 3.2, 1.4, "Prometheus Monitoring", "Scrapes pods & node metrics\nStores time-series history", "#fed7aa", "#ea580c")
    # Top right: Grafana
    d_box(8.5, 4.4, 3.2, 1.4, "Grafana Dashboards", "Visualizes SLA breach rates,\ncluster wattage, active node count", "#fee2e2", "#dc2626")

    # Center: FastAPI Autoscaler Service
    d_box(4.5, 2.2, 3.2, 1.5, "FastAPI Autoscaler Pod", "• /v1/scaling-decision (<10ms)\n• /metrics (OpenMetrics format)\n• PPO Policy + Safe Action Shield", "#dbeafe", "#2563eb")

    # Bottom left: KEDA External Scaler
    d_box(0.5, 0.6, 3.2, 1.3, "KEDA External Scaler", "Consumes predictive metric:\nrecommended_cpu_nodes", "#dcfce7", "#16a34a")
    # Bottom center: K8s HPA & Cluster Autoscaler
    d_box(4.5, 0.6, 3.2, 1.3, "Kubernetes HPA & CA", "• Adjusts deployment replicas\n• Triggers AWS/GCP node provisioning", "#f1f5f9", "#475569")
    # Bottom right: Cloud Infrastructure
    d_box(8.5, 0.6, 3.2, 1.3, "Cloud Managed Pools", "AWS EC2 ASG / GCP MIG\nProvisions standard & GPU instances", "#fef3c7", "#d97706")

    # Arrows
    ax.annotate('', xy=(4.5, 5.1), xytext=(3.7, 5.1), arrowprops=dict(arrowstyle="->", lw=1.6, color='#ea580c'))
    ax.annotate('', xy=(8.5, 5.1), xytext=(7.7, 5.1), arrowprops=dict(arrowstyle="->", lw=1.6, color='#ea580c'))
    
    ax.annotate('', xy=(6.1, 4.4), xytext=(6.1, 3.7), arrowprops=dict(arrowstyle="->", lw=1.6, color='#ea580c'))
    ax.text(6.3, 4.05, "Scrape /metrics", ha='left', va='center', fontsize=7.5, color='#ea580c')

    ax.annotate('', xy=(0.5+1.6, 1.9), xytext=(4.5, 2.7), arrowprops=dict(arrowstyle="->", lw=1.6, color='#16a34a'))
    ax.text(2.6, 2.4, "Polling /v1/scaling-decision", ha='center', va='center', fontsize=7.5, color='#16a34a')

    ax.annotate('', xy=(4.5, 1.25), xytext=(3.7, 1.25), arrowprops=dict(arrowstyle="->", lw=1.6, color='#16a34a'))
    ax.annotate('', xy=(8.5, 1.25), xytext=(7.7, 1.25), arrowprops=dict(arrowstyle="->", lw=1.6, color='#475569'))

    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 7.3)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "00_production_k8s_deployment.png")
    plt.savefig(p)
    plt.close(fig)
    print("Saved 00_production_k8s_deployment.png")

if __name__ == "__main__":
    create_overall_architecture_flowchart()
    create_streaming_buffer_diagram()
    create_multi_horizon_transformer_diagram()
    create_safe_shield_flowchart()
    create_hierarchical_control_diagram()
    create_cmdp_lagrangian_flowchart()
    create_cloud_deployment_topology()
