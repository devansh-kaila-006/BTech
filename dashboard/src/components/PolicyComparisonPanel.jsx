import React from 'react';
import { Layers, CheckCircle2, DollarSign, Leaf, Shield, Award } from 'lucide-react';

const policies = [
  {
    id: 'tier1',
    name: 'Tier 1: Reactive Baseline',
    color: '#ef4444',
    badge: 'Standard PPO',
    desc: '3D Reactive state, no lookahead or safety clamp.',
    breaches: 137,
    breachRate: '54.8%',
    cost: '$41.36',
    carbon: '0.71 kg',
    isDominant: false
  },
  {
    id: 'tier2',
    name: 'Tier 2: Predictive RL',
    color: '#f59e0b',
    badge: 'Multi-Oracle',
    desc: '7D Oracle-fused state with LSTM, XGBoost, and Power DNN.',
    breaches: 98,
    breachRate: '39.2%',
    cost: '$30.96',
    carbon: '0.75 kg',
    isDominant: false
  },
  {
    id: 'tier3',
    name: 'Tier 3: Hierarchical DVFS',
    color: '#3b82f6',
    badge: 'Heterogeneous',
    desc: 'Decoupled CPU + GPU frequency scaling (22.5% power savings).',
    breaches: 88,
    breachRate: '35.3%',
    cost: '$27.24',
    carbon: '0.58 kg',
    isDominant: false
  },
  {
    id: 'tier4',
    name: 'Tier 4: Full Safe Autonomous',
    color: '#10b981',
    badge: 'Ours (Pareto Optimal)',
    desc: 'Predictive + Hierarchical + Safe Shield + Lagrangian Bounds.',
    breaches: 0,
    breachRate: '0.0%',
    cost: '$3.04',
    carbon: '0.62 kg',
    isDominant: true
  }
];

export default function PolicyComparisonPanel({ activeTier, onSelectTier }) {
  return (
    <div className="p-5 rounded-2xl border border-gray-800 bg-gray-900/60 backdrop-blur-md">
      <div className="flex items-center justify-between pb-4 border-b border-gray-800">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 rounded-lg bg-purple-500/10 text-purple-400 border border-purple-500/20">
            <Layers className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-white">
              Architectural Evolutionary Tiers & Pareto Benchmark
            </h2>
            <p className="text-xs text-gray-400">
              Select an architectural tier to observe real-time scheduling behavior
            </p>
          </div>
        </div>

        <span className="flex items-center space-x-1 text-xs font-semibold px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
          <Award className="w-3.5 h-3.5" />
          <span>Tier 4 Strictly Dominates</span>
        </span>
      </div>

      {/* Tier Selection Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3.5 mt-4">
        {policies.map((p) => {
          const isSelected = activeTier === p.id;
          return (
            <div
              key={p.id}
              onClick={() => onSelectTier(p.id)}
              className={`p-4 rounded-xl border transition-all cursor-pointer relative overflow-hidden ${
                isSelected
                  ? 'border-blue-500 bg-blue-950/30 shadow-lg shadow-blue-500/10 ring-1 ring-blue-500/50'
                  : 'border-gray-800 bg-gray-950/60 hover:border-gray-700'
              }`}
            >
              {p.isDominant && (
                <div className="absolute top-0 right-0 px-2 py-0.5 bg-emerald-500 text-[10px] font-bold text-gray-950 rounded-bl-lg uppercase tracking-wider">
                  Optimal
                </div>
              )}

              <div className="flex items-start justify-between">
                <div>
                  <span
                    className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full"
                    style={{ backgroundColor: `${p.color}20`, color: p.color }}
                  >
                    {p.badge}
                  </span>
                  <h3 className="text-xs font-bold text-white mt-2">{p.name}</h3>
                </div>
              </div>

              <p className="text-[11px] text-gray-400 mt-1 line-clamp-2">{p.desc}</p>

              <div className="mt-3 pt-3 border-t border-gray-800/80 grid grid-cols-2 gap-2 text-xs">
                <div>
                  <span className="text-[10px] text-gray-500 uppercase block">SLA Breaches</span>
                  <span className={`font-mono font-bold ${p.breaches === 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                    {p.breaches} ({p.breachRate})
                  </span>
                </div>
                <div>
                  <span className="text-[10px] text-gray-500 uppercase block">Total Spend</span>
                  <span className="font-mono font-bold text-white">{p.cost}</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

