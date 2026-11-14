import React, { useState } from 'react';
import { Compass, ChevronDown, ChevronUp, CheckCircle, ExternalLink, Award, Sparkles } from 'lucide-react';

const stages = [
  {
    num: 1,
    title: 'Streaming Ingestion',
    metric: '44.6k rows/s (22.4 μs/event)',
    summary: 'Zero-copy PyArrow memory mapping streaming 156,000 trace rows with constant O(1) memory footprint.',
    defensePoint: 'Eliminates Python JSON/CSV deserialization bottlenecks, enabling real-time trace ingestion.'
  },
  {
    num: 2,
    title: 'Oracle Bayesian HPO',
    metric: 'Optuna (30 trials)',
    summary: 'Hyperparameter optimization for LSTM Workload (MSE: 0.00295), XGBoost Latency, and Power DNN.',
    defensePoint: 'Ensures proactive oracles provide high-fidelity state features without manual hyperparameter bias.'
  },
  {
    num: 3,
    title: 'Composite State Fusion',
    metric: '7D Unified State (<2.5ms)',
    summary: 'Fuses raw pod telemetry [CPU, Mem, Type] with oracle predictions [Workload, Latency, Power, Nodes].',
    defensePoint: 'Transforms reactive MDP into a predictive Markov Decision Process.'
  },
  {
    num: 4,
    title: 'Custom Gymnasium MDP',
    metric: 'Continuous Action Space',
    summary: 'Gymnasium environment enforcing multi-objective reward: R = -(Power Penalty + SLA Delay Penalty).',
    defensePoint: 'Models realistic continuous cluster scaling with bounded scaling velocity [-5, +5 nodes/tick].'
  },
  {
    num: 5,
    title: '5k-Step Rapid POC',
    metric: 'State-Space Complexity Curve',
    summary: 'Discovered that in early training (<5k steps), baseline converged faster due to lower 3D dimensionality.',
    defensePoint: 'Directly explains the State-Space Complexity Hypothesis to professors.'
  },
  {
    num: 6,
    title: '100k-Step Core Evaluation',
    metric: '-32.8% SLA Breaches',
    summary: 'Predictive RL slashed SLA starvation breaches from 2,692 down to 1,807 on 5,000 unseen steps (+4.7% reward).',
    defensePoint: 'Proves the foundational hypothesis that predictive lookahead prevents SLA starvation.'
  },
  {
    num: 7,
    title: 'Transformer vs LSTM',
    metric: 'MSE: 0.000679 (4.3x better)',
    summary: 'Self-attention Time-Series Transformer captured long-range multi-scale temporal dependencies.',
    defensePoint: 'Validates upgrading from recurrent LSTM to multi-head self-attention.'
  },
  {
    num: 8,
    title: 'Attention Fusion Ablation',
    metric: 'Flat Concat Superiority',
    summary: 'Evaluated MultiheadAttention vs Flat Concatenation; flat fusion yielded faster RL policy convergence.',
    defensePoint: 'Honest empirical ablation: dynamic attention weights added non-stationarity in early RL training.'
  },
  {
    num: 9,
    title: 'Safe RL Action Shield',
    metric: '0 Breaches (100% Compliance)',
    summary: 'Runtime barrier function clamped unsafe downscaling: N_safe = Demand / 6.4 cores (breaches: 1,158 -> 0).',
    defensePoint: 'Provides mathematical safety invariance guaranteeing cluster will never violate 80% SLA.'
  },
  {
    num: 10,
    title: 'Hierarchical GPU Autoscaling',
    metric: '-22.5% Power via DVFS',
    summary: 'Decoupled continuous action for CPU worker nodes and GPU accelerators with DVFS frequency modulation.',
    defensePoint: 'Prevents wasteful scale-up of $3.91/hr GPU instances when only CPU preprocessing traffic spikes.'
  },
  {
    num: 11,
    title: 'Multi-Horizon Lookahead',
    metric: '60s Cloud VM Lookahead (t+15)',
    summary: 'Temporal Transformer forecasting across t+1 (burst), t+5 (buffer), and t+15 (cloud VM boot window).',
    defensePoint: 'Overcomes the 2–5 minute physical VM provisioning delay by ordering nodes before traffic arrives.'
  },
  {
    num: 12,
    title: 'PPO-Lagrangian Constrained RL',
    metric: 'SLA Violations < 5% (λ = 1.079)',
    summary: 'Dual gradient ascent constrained MDP optimizing power subject to strict SLA violation upper bound.',
    defensePoint: 'Automates hyperparameter tuning via Lagrange multiplier dual updates without manual reward hacking.'
  },
  {
    num: 13,
    title: 'Multi-Objective Pareto Frontier',
    metric: 'Tier 4 Strictly Optimal',
    summary: 'Benchmarked 4 tiers; Tier 4 achieved 0 breaches, $3.04 cost (92.6% net savings), and 0.62 kg CO2e.',
    defensePoint: 'Proves mathematical Pareto dominance: each component justifies its complexity on the Pareto curve.'
  },
  {
    num: 14,
    title: 'Chaos Engineering & Resilience',
    metric: '12x Faster Recovery (2 vs 24 ticks)',
    summary: 'Injected 50% node crash at t=80 and 4.5x flash crowd at t=220; Safe Shield slashed downtime by 96.1%.',
    defensePoint: 'Demonstrates enterprise fault tolerance under sudden node preemptions and hardware panics.'
  },
  {
    num: 15,
    title: 'Real-Time Systems Profiling',
    metric: '8.57ms p50 (117 decisions/s)',
    summary: 'End-to-end FastAPI microservice profiling with 637.9 MB RAM footprint and sub-15ms p99 tail latency.',
    defensePoint: 'Proves the system is ready for live production Kubernetes deployment as a lightweight sidecar.'
  }
];

export default function StageNavigator() {
  const [selectedStage, setSelectedStage] = useState(13); // Default to Stage 13 (Pareto)
  const [isExpanded, setIsExpanded] = useState(false);

  const current = stages.find((s) => s.num === selectedStage) || stages[12];

  return (
    <div className="p-5 rounded-2xl border border-gray-800 bg-gray-900/60 backdrop-blur-md">
      <div className="flex items-center justify-between pb-3 border-b border-gray-800">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 rounded-lg bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
            <Compass className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-white">
              15-Stage Experimental Roadmap & Defense Navigator
            </h2>
            <p className="text-xs text-gray-400">
              Interactive progression across all 15 scientifically verified ablation stages
            </p>
          </div>
        </div>

        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="flex items-center space-x-1 text-xs text-gray-400 hover:text-white px-2 py-1 rounded-lg bg-gray-800/60 cursor-pointer"
        >
          <span>{isExpanded ? 'Collapse Strip' : 'Expand All Stages'}</span>
          {isExpanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
        </button>
      </div>

      {/* Horizontal Stage Selector Strip */}
      <div className="flex items-center space-x-2 overflow-x-auto py-3 no-scrollbar">
        {stages.map((stg) => {
          const isSelected = selectedStage === stg.num;
          return (
            <button
              key={stg.num}
              onClick={() => setSelectedStage(stg.num)}
              className={`shrink-0 px-3 py-1.5 rounded-xl border text-xs font-semibold transition-all cursor-pointer flex items-center space-x-1.5 ${
                isSelected
                  ? 'bg-blue-600 text-white border-blue-400 shadow-md shadow-blue-500/20'
                  : 'bg-gray-950/70 text-gray-400 border-gray-800 hover:border-gray-700 hover:text-gray-200'
              }`}
            >
              <span className={`w-4 h-4 rounded-full flex items-center justify-center text-[10px] font-bold ${
                isSelected ? 'bg-white text-blue-700' : 'bg-gray-800 text-gray-400'
              }`}>
                {stg.num}
              </span>
              <span>{stg.title}</span>
            </button>
          );
        })}
      </div>

      {/* Active Stage Deep-Dive Card */}
      <div className="mt-2 p-4 rounded-xl bg-gray-950/80 border border-gray-800/90 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="space-y-1.5 flex-1">
          <div className="flex items-center space-x-2">
            <span className="text-xs font-bold px-2 py-0.5 rounded bg-blue-600/30 text-blue-300 border border-blue-500/40">
              Stage {current.num}
            </span>
            <h3 className="text-sm font-bold text-white">{current.title}</h3>
            <span className="text-xs font-mono font-bold px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
              {current.metric}
            </span>
          </div>
          <p className="text-xs text-gray-300 leading-relaxed">{current.summary}</p>
        </div>

        {/* Defense Talking Point Highlight */}
        <div className="p-3 rounded-lg bg-indigo-950/30 border border-indigo-800/40 max-w-md shrink-0">
          <span className="text-[10px] uppercase font-bold text-indigo-400 tracking-wider flex items-center space-x-1">
            <Sparkles className="w-3 h-3" />
            <span>Viva Defense Answer</span>
          </span>
          <p className="text-[11px] text-indigo-200 mt-1 italic">
            "{current.defensePoint}"
          </p>
        </div>
      </div>
    </div>
  );
}

