import React, { useState } from 'react';
import { X, Play, Film, Award, ChevronLeft, ChevronRight, ExternalLink } from 'lucide-react';

const animations = [
  {
    id: 1,
    title: '1. Predictive vs Reactive Trajectory',
    stage: 'Stage 6',
    src: '/animations/animated_scaling_simulation.gif',
    badge: 'Core Proof',
    desc: 'Demonstrates proactive capacity scaling 2.3 seconds before sudden pod arrival spikes, keeping cluster utilization strictly under the 80% SLA threshold and cutting delay penalties by 32.8%.'
  },
  {
    id: 2,
    title: '2. Safe Action Shield Interception',
    stage: 'Stage 9',
    src: '/animations/animated_safe_shield.gif',
    badge: '100% SLA Guarantee',
    desc: 'Visualizes runtime barrier function clamping aggressive downscaling during saturation, reducing catastrophic SLA breaches from 1,158 down to exactly 0.'
  },
  {
    id: 3,
    title: '3. Heterogeneous GPU Multi-Pool',
    stage: 'Stage 10',
    src: '/animations/animated_hierarchical_gpu.gif',
    badge: '22.5% Power Saved',
    desc: 'Decoupled multi-pool actuation: PPO independently controls continuous scaling velocity for standard CPU worker nodes (blue) and high-density GPU accelerators (purple).'
  },
  {
    id: 4,
    title: '4. Multi-Horizon Lookahead Transformer',
    stage: 'Stage 11',
    src: '/animations/animated_multi_horizon.gif',
    badge: '60s VM Lookahead',
    desc: 'Proactive lookahead across t+1 (immediate burst), t+5 (near-term queue depth), and t+15 (60-second physical cloud VM spin-up latency) anticipating future traffic surges.'
  },
  {
    id: 5,
    title: '5. Chaos Engineering: 50% Node Crash & Surge',
    stage: 'Stage 14',
    src: '/animations/animated_chaos_resilience.gif',
    badge: '12x Faster Recovery',
    desc: 'Injected hardware panic (50% node crash at t=25) and 4.0x flash crowd storm at t=50. While the reactive baseline suffers persistent starvation, Safe Shield restabilizes in just 2 ticks.'
  },
  {
    id: 6,
    title: '6. Cumulative Financial ROI Accumulator',
    stage: 'Stage 13',
    src: '/animations/animated_pareto_roi.gif',
    badge: '92.6% Cost Slashed',
    desc: 'Real-time accumulation of cloud infrastructure spend ($) and contractual SLA fines ($50/hr penalty model). Tier 4 eliminates SLA violation penalties completely.'
  },
  {
    id: 7,
    title: '7. PPO-Lagrangian Dual Multiplier Dynamics',
    stage: 'Stage 12',
    src: '/animations/animated_lagrangian_dual.gif',
    badge: 'Constrained MDP',
    desc: 'Dynamic gradient ascent enforcement: Adaptive dual multiplier lambda dynamically ascends whenever SLA violation cost breaches epsilon_SLA = 0.05, forcing safety compliance.'
  }
];

export default function AnimationVaultModal({ isOpen, onClose }) {
  const [selectedIdx, setSelectedIdx] = useState(0);

  if (!isOpen) return null;

  const currentAnim = animations[selectedIdx];

  const handlePrev = () => {
    setSelectedIdx((prev) => (prev > 0 ? prev - 1 : animations.length - 1));
  };

  const handleNext = () => {
    setSelectedIdx((prev) => (prev < animations.length - 1 ? prev + 1 : 0));
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md">
      <div className="bg-gray-900 border border-gray-800 rounded-2xl w-full max-w-4xl overflow-hidden shadow-2xl flex flex-col max-h-[92vh]">
        {/* Modal Header */}
        <div className="px-6 py-4 border-b border-gray-800 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Film className="w-5 h-5 text-blue-400" />
            <h2 className="text-base font-bold text-white tracking-tight">
              Simulation Animation Vault
            </h2>
            <span className="text-xs px-2 py-0.5 rounded-full bg-blue-900/50 text-blue-300 border border-blue-700/50">
              7 High-Res Recordings
            </span>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-gray-400 hover:text-white hover:bg-gray-800 transition-all cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 overflow-y-auto space-y-5">
          {/* Main GIF Display with Controls */}
          <div className="relative rounded-xl overflow-hidden border border-gray-800 bg-gray-950 flex items-center justify-center min-h-[380px]">
            <img
              src={currentAnim.src}
              alt={currentAnim.title}
              className="max-h-[440px] w-auto object-contain"
            />

            {/* Navigation Arrows */}
            <button
              onClick={handlePrev}
              className="absolute left-3 top-1/2 -translate-y-1/2 p-2 rounded-full bg-gray-900/80 hover:bg-gray-800 text-white border border-gray-700/80 shadow-lg cursor-pointer transition-all"
              title="Previous Animation"
            >
              <ChevronLeft className="w-5 h-5" />
            </button>

            <button
              onClick={handleNext}
              className="absolute right-3 top-1/2 -translate-y-1/2 p-2 rounded-full bg-gray-900/80 hover:bg-gray-800 text-white border border-gray-700/80 shadow-lg cursor-pointer transition-all"
              title="Next Animation"
            >
              <ChevronRight className="w-5 h-5" />
            </button>
          </div>

          {/* Details Card */}
          <div className="p-4 rounded-xl bg-gray-950/70 border border-gray-800">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <span className="text-xs font-bold px-2.5 py-0.5 rounded-md bg-blue-600/30 text-blue-300 border border-blue-500/40">
                  {currentAnim.stage}
                </span>
                <h3 className="text-sm font-bold text-white">{currentAnim.title}</h3>
              </div>
              <span className="text-xs font-semibold px-2 py-0.5 rounded-md bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                {currentAnim.badge}
              </span>
            </div>
            <p className="text-xs text-gray-300 mt-2 leading-relaxed">{currentAnim.desc}</p>
          </div>

          {/* Thumbnail Selector Strip */}
          <div className="grid grid-cols-4 sm:grid-cols-7 gap-2">
            {animations.map((anim, idx) => (
              <button
                key={anim.id}
                onClick={() => setSelectedIdx(idx)}
                className={`p-2 rounded-lg border text-left transition-all cursor-pointer ${
                  selectedIdx === idx
                    ? 'border-blue-500 bg-blue-950/40 ring-1 ring-blue-500/50'
                    : 'border-gray-800 bg-gray-950/50 hover:border-gray-700'
                }`}
              >
                <span className="text-[10px] font-bold text-gray-400 block truncate">
                  GIF {anim.id}
                </span>
                <span className="text-[9px] text-gray-500 block truncate">{anim.stage}</span>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

