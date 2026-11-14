# K8s Predictive-RL Autoscaler | Cloud Operations Dashboard

A modern, glassmorphic React 19 + Vite web dashboard providing real-time telemetry, trace simulation playback, multi-objective Pareto analysis, and interactive chaos injection for heterogeneous Kubernetes GPU clusters.

Developed for **B.Tech Capstone Project (22AIE498, Team ID: `23UG109`, Panel: 12)**.

---

## 🌟 Key Features

1. **Live Cluster Telemetry Panel:**
   - Real-time gauge metrics for CPU Utilization (%), GPU Utilization (%), Cluster Fleet VM counts, electrical power (Watts with DVFS savings), and carbon intensity.
   - 80% SLA contract breach alert indicators.
2. **Real-Time Scaling Dynamics Chart (Recharts):**
   - Live multi-line and gradient-area chart comparing our **Predictive RL + Safe Shield** trajectory against the **Reactive Baseline RL** curve under live Alibaba trace replay.
   - Reference threshold line at 80% SLA limit.
3. **Trace Playback & Scrubbing Controller:**
   - Play, pause, reset, step forward (+1 tick), and speed selector (1x, 2x, 5x, 10x).
   - Scrubber slider over 250 real-world Alibaba Cloud trace events with live incoming pod metadata.
4. **Architectural Tier Comparator:**
   - Real-time switcher between:
     - 🔴 **Tier 1: Reactive Baseline RL**
     - 🟠 **Tier 2: Oracle-Augmented Predictive RL**
     - 🔵 **Tier 3: Hierarchical GPU DVFS RL**
     - 🟢 **Tier 4: Full Safe Autonomous (Ours - Strictly Pareto Optimal)**
5. **Interactive Chaos Engineering Simulator:**
   - **💥 Crash 50% Worker Nodes:** Simulates hardware panic / spot preemption.
   - **⚡ Trigger 4.5x Flash Crowd Storm:** Injects instant traffic surge.
   - **🛡️ Safe Action Shield Invariant Monitor:** Displays live barrier clamping state and mathematical invariant ($N_{\text{safe}} = \frac{\text{Demand}}{6.4}$).
6. **Financial Ledger & Carbon Savings Tracker:**
   - Real-time counter of compute spend ($), SLA contract fine savings, and net 92.6% cost reduction.
7. **Animation Vault Modal:**
   - Embedded browser pop-up modal featuring all 7 dark-mode simulation GIF recordings with technical takeaways.

---

## 🚀 Quickstart

```bash
# Navigate to dashboard directory
cd dashboard

# Install dependencies (React 19, Recharts, Lucide-react, Tailwind CSS v4)
npm install

# Start local development server
npm run dev

# Or build for production
npm run build
```

Open your browser at: **`http://localhost:5173`**
