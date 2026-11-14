import React from 'react';
import { X, BookOpen, Award, CheckCircle, ShieldCheck, Cpu, Zap, Activity } from 'lucide-react';

export default function VivaWalkthroughModal({ isOpen, onClose }) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md">
      <div className="bg-gray-900 border border-gray-800 rounded-2xl w-full max-w-4xl overflow-hidden shadow-2xl flex flex-col max-h-[92vh]">
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-800 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <BookOpen className="w-5 h-5 text-indigo-400" />
            <h2 className="text-base font-bold text-white tracking-tight">
              Capstone Viva Defense Guide & System Architecture
            </h2>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-gray-400 hover:text-white hover:bg-gray-800 transition-all cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto space-y-6 text-xs text-gray-300">
          {/* Project Header Card */}
          <div className="p-4 rounded-xl bg-gradient-to-r from-blue-950/60 to-purple-950/60 border border-blue-800/40">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="text-sm font-bold text-white">
                Hierarchical-Action, Streaming-Ingestion Predictive-RL Autoscaling
              </span>
              <div className="flex space-x-2">
                <span className="px-2 py-0.5 rounded bg-blue-600/40 text-blue-200 font-mono font-bold">
                  Team 23UG109
                </span>
                <span className="px-2 py-0.5 rounded bg-purple-600/40 text-purple-200 font-mono font-bold">
                  Panel 12
                </span>
              </div>
            </div>
            <p className="mt-2 text-gray-300 leading-relaxed">
              <strong>Course:</strong> B.Tech Capstone Project (22AIE498) • <strong>Project Guide:</strong> Dr. Beena • <strong>Student Researchers:</strong> Devansh Kaila & Akshay Damera
            </p>
          </div>

          {/* End-to-End Pipeline ASCII Architecture */}
          <div>
            <h3 className="text-xs uppercase font-bold tracking-wider text-indigo-400 mb-2 flex items-center space-x-1.5">
              <Activity className="w-4 h-4" />
              <span>End-to-End Control Plane Architecture</span>
            </h3>
            <div className="p-4 rounded-xl bg-gray-950 font-mono text-[11px] leading-relaxed text-gray-300 border border-gray-800 overflow-x-auto">
              {`Alibaba OSDI Trace  ──►  PyArrow Zero-Copy Ingestion (44.6k rows/s)
                              │
                              ▼
                Composite State Fusion Engine (7D State Space, <2.5ms)
                [Pod CPU, Mem, Type]  +  [LSTM Workload, XGBoost Latency, Power DNN]
                              │
                              ▼
                PPO Actor-Critic Policy Network (Proximal Policy Optimization)
                              │  (Continuous action: ΔN_cpu, ΔN_gpu)
                              ▼
                Safe Action Clamping Shield (Runtime Barrier Function)
                if (N_intended < N_safe) ──► Clamps scaling to prevent SLA breaches
                              │
                              ▼
                Heterogeneous Kubernetes Cluster Actuation
                ├── Worker Pool 0: 16-Core Compute Servers (c6i.4xlarge)
                └── Worker Pool 1: 8x GPU Accelerator Instances (g5.12xlarge + DVFS 1.2GHz)`}
            </div>
          </div>

          {/* Core Mathematical Formulations */}
          <div>
            <h3 className="text-xs uppercase font-bold tracking-wider text-emerald-400 mb-2 flex items-center space-x-1.5">
              <Award className="w-4 h-4" />
              <span>Core Mathematical Formulations</span>
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
              <div className="p-3.5 rounded-xl bg-gray-950/80 border border-gray-800">
                <span className="font-bold text-white block mb-1">1. Multi-Objective Reward Function</span>
                <p className="font-mono text-emerald-300 bg-gray-900 p-2 rounded border border-gray-800 my-1.5">
                  R_t = -[ (N_nodes × 0.1) + max(0, U_t - 0.8) × 10.0 ]
                </p>
                <p className="text-gray-400 text-[11px]">
                  Balances power consumption against contractual SLA delay penalties.
                </p>
              </div>

              <div className="p-3.5 rounded-xl bg-gray-950/80 border border-gray-800">
                <span className="font-bold text-white block mb-1">2. Safe Shield Invariant Clamping</span>
                <p className="font-mono text-blue-300 bg-gray-900 p-2 rounded border border-gray-800 my-1.5">
                  N_safe = CPU_demand / (8 cores × 0.8 util) = CPU_demand / 6.4
                </p>
                <p className="text-gray-400 text-[11px]">
                  Guarantees cluster capacity strictly prevents utilization exceeding 80%.
                </p>
              </div>

              <div className="p-3.5 rounded-xl bg-gray-950/80 border border-gray-800">
                <span className="font-bold text-white block mb-1">3. PPO-Lagrangian Dual Update</span>
                <p className="font-mono text-purple-300 bg-gray-900 p-2 rounded border border-gray-800 my-1.5">
                  λ_{k+1} = max(0, λ_k + α_λ [ C_SLA(π_k) - ε_SLA ])
                </p>
                <p className="text-gray-400 text-[11px]">
                  Adaptive dual multiplier ascents to penalize policy when breaches exceed 5%.
                </p>
              </div>

              <div className="p-3.5 rounded-xl bg-gray-950/80 border border-gray-800">
                <span className="font-bold text-white block mb-1">4. DVFS Power Scaling Formula</span>
                <p className="font-mono text-amber-300 bg-gray-900 p-2 rounded border border-gray-800 my-1.5">
                  P = C × V^2 × f + P_static (yielding -22.5% energy)
                </p>
                <p className="text-gray-400 text-[11px]">
                  Dynamically scales GPU clock frequency down when tensor load is moderate.
                </p>
              </div>
            </div>
          </div>

          {/* Top 3 Viva Questions & Answers */}
          <div>
            <h3 className="text-xs uppercase font-bold tracking-wider text-amber-400 mb-2 flex items-center space-x-1.5">
              <ShieldCheck className="w-4 h-4" />
              <span>Top Viva Defense Questions & Answers</span>
            </h3>
            <div className="space-y-2.5">
              <div className="p-3 rounded-xl bg-gray-950/80 border border-gray-800">
                <strong className="text-white block">Q1: How does your model handle physical cloud VM cold-boot latency (2–5 minutes)?</strong>
                <p className="text-gray-300 mt-1 leading-relaxed">
                  <strong>Answer:</strong> Reactive autoscalers wait until after utilization breaches 80%, causing 3-minute outages while VMs boot. Our <strong>Multi-Horizon Transformer</strong> forecasts demand at $t+15$ (60s ahead), dispatching VM spin-up commands in advance so nodes are warm when traffic arrives.
                </p>
              </div>

              <div className="p-3 rounded-xl bg-gray-950/80 border border-gray-800">
                <strong className="text-white block">Q2: Why use a Safe Action Shield instead of pure unconstrained RL?</strong>
                <p className="text-gray-300 mt-1 leading-relaxed">
                  <strong>Answer:</strong> Unconstrained RL exploration occasionally attempts aggressive downscaling during temporary traffic lulls. The <strong>Safe Action Shield</strong> acts as a runtime barrier function, mathematically clamping scale actions to N &gt;= N_safe, guaranteeing <strong>zero SLA violations (100% compliance)</strong>.
                </p>
              </div>

              <div className="p-3 rounded-xl bg-gray-950/80 border border-gray-800">
                <strong className="text-white block">Q3: How do you prove your architecture is better than standard baselines?</strong>
                <p className="text-gray-300 mt-1 leading-relaxed">
                  <strong>Answer:</strong> Through <strong>Strict Pareto Dominance (Stage 13)</strong>. Tier 4 strictly dominates Tiers 1 through 3 by eliminating SLA contract fines (92.6% net cost savings) and cutting operational carbon by 22.5%, verified with p &lt; 0.001 statistical significance.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
