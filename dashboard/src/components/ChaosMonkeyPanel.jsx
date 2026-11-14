import React from 'react';
import { Skull, Zap, RotateCcw, ShieldCheck, ShieldAlert, AlertTriangle } from 'lucide-react';

export default function ChaosMonkeyPanel({
  onInjectNodeCrash,
  onInjectFlashCrowd,
  onResetChaos,
  activeChaos,
  isShieldIntervening
}) {
  return (
    <div className="p-5 rounded-2xl border border-gray-800 bg-gray-900/60 backdrop-blur-md">
      <div className="flex items-center justify-between pb-3 border-b border-gray-800">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 rounded-lg bg-red-500/10 text-red-400 border border-red-500/20">
            <Skull className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-white">
              Chaos Engineering & Fault Tolerance Simulator
            </h2>
            <p className="text-xs text-gray-400">
              Inject non-deterministic datacenter shocks to verify runtime Safe Shield clamping
            </p>
          </div>
        </div>

        {activeChaos ? (
          <span className="flex items-center space-x-1.5 text-xs font-semibold px-2.5 py-1 rounded-full bg-red-500/15 text-red-400 border border-red-500/30 animate-pulse">
            <AlertTriangle className="w-3.5 h-3.5" />
            <span>Active Shock: {activeChaos}</span>
          </span>
        ) : (
          <span className="flex items-center space-x-1.5 text-xs font-semibold px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
            <ShieldCheck className="w-3.5 h-3.5" />
            <span>Nominal State</span>
          </span>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3.5 mt-4">
        {/* Shock 1: Node Crash */}
        <button
          onClick={onInjectNodeCrash}
          className="flex items-center space-x-3 p-3.5 rounded-xl border border-amber-500/30 bg-amber-950/20 hover:bg-amber-900/30 transition-all text-left group cursor-pointer"
        >
          <div className="p-2 rounded-lg bg-amber-500/20 text-amber-400 group-hover:scale-105 transition-all">
            <Skull className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-xs font-bold text-white group-hover:text-amber-300">
              💥 Crash 50% Worker Nodes
            </h3>
            <p className="text-[11px] text-gray-400 mt-0.5">
              Simulates VM spot preemption & GPU ECC panic
            </p>
          </div>
        </button>

        {/* Shock 2: Flash Crowd */}
        <button
          onClick={onInjectFlashCrowd}
          className="flex items-center space-x-3 p-3.5 rounded-xl border border-purple-500/30 bg-purple-950/20 hover:bg-purple-900/30 transition-all text-left group cursor-pointer"
        >
          <div className="p-2 rounded-lg bg-purple-500/20 text-purple-400 group-hover:scale-105 transition-all">
            <Zap className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-xs font-bold text-white group-hover:text-purple-300">
              ⚡ Trigger 4.5x Flash Crowd
            </h3>
            <p className="text-[11px] text-gray-400 mt-0.5">
              Simulates sudden viral API request burst
            </p>
          </div>
        </button>

        {/* Reset Shock */}
        <button
          onClick={onResetChaos}
          className="flex items-center space-x-3 p-3.5 rounded-xl border border-gray-800 bg-gray-950/60 hover:bg-gray-800/40 transition-all text-left cursor-pointer"
        >
          <div className="p-2 rounded-lg bg-gray-800 text-gray-400">
            <RotateCcw className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-xs font-bold text-white">
              🔄 Clear Injected Shocks
            </h3>
            <p className="text-[11px] text-gray-400 mt-0.5">
              Restores nominal cluster scaling dynamics
            </p>
          </div>
        </button>
      </div>

      {/* Shield Formula & Invariant Card */}
      <div className="mt-3.5 p-3 rounded-xl bg-gray-950/80 border border-gray-800 flex flex-col sm:flex-row items-center justify-between gap-2 text-xs">
        <div className="flex items-center space-x-2 text-gray-400">
          <span className="font-mono text-indigo-400 font-bold">Safety Invariant:</span>
          <span className="font-mono text-gray-300">N_safe = Demand / (8 cores × 0.8 util) = Demand / 6.4</span>
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-gray-400">Barrier Clamping:</span>
          <span className={`font-mono font-bold px-2 py-0.5 rounded text-[11px] ${
            isShieldIntervening
              ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40 animate-pulse'
              : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
          }`}>
            {isShieldIntervening ? 'OVERRIDING SCALE-DOWN' : 'PASS-THROUGH (SAFE)'}
          </span>
        </div>
      </div>
    </div>
  );
}

