import React from 'react';
import { Activity, Server, ShieldCheck, Zap, AlertTriangle, Eye } from 'lucide-react';

export default function Navbar({ isLiveBackend, currentStep, onOpenAnimations }) {
  return (
    <header className="border-b border-gray-800 bg-gray-950/80 backdrop-blur-md sticky top-0 z-40 px-6 py-4">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
        {/* Title and Badges */}
        <div className="flex items-center space-x-3">
          <div className="p-2.5 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-700 shadow-lg shadow-blue-500/20">
            <Server className="w-6 h-6 text-white" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-xl font-bold text-white tracking-tight">
                K8s Predictive-RL Autoscaler
              </h1>
              <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-blue-900/60 text-blue-300 border border-blue-700/50">
                Team 23UG109
              </span>
              <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-purple-900/60 text-purple-300 border border-purple-700/50">
                Panel 12
              </span>
            </div>
            <p className="text-xs text-gray-400 mt-0.5">
              Hierarchical Multi-Pool GPU Cloud Orchestrator • Alibaba OSDI '26 Replay
            </p>
          </div>
        </div>

        {/* Status Indicators & Action Buttons */}
        <div className="flex items-center space-x-4">
          {/* Simulation Step Badge */}
          <div className="flex items-center space-x-2 bg-gray-900/90 border border-gray-800 rounded-lg px-3 py-1.5 text-xs text-gray-300">
            <Activity className="w-4 h-4 text-emerald-400" />
            <span>Trace Tick:</span>
            <span className="font-mono font-bold text-emerald-400">{currentStep}</span>
            <span className="text-gray-500">/ 250</span>
          </div>

          {/* Backend Connection Indicator */}
          <div className="flex items-center space-x-2 bg-gray-900/90 border border-gray-800 rounded-lg px-3 py-1.5 text-xs">
            <span className="relative flex h-2 w-2">
              <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${isLiveBackend ? 'bg-emerald-400' : 'bg-blue-400'}`}></span>
              <span className={`relative inline-flex rounded-full h-2 w-2 ${isLiveBackend ? 'bg-emerald-500' : 'bg-blue-500'}`}></span>
            </span>
            <span className="text-gray-300">
              {isLiveBackend ? 'FastAPI Live (:8000)' : 'High-Speed Simulation'}
            </span>
          </div>

          {/* View Animations Button */}
          <button
            onClick={onOpenAnimations}
            className="flex items-center space-x-2 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white text-xs font-semibold px-3.5 py-1.5 rounded-lg shadow-md transition-all cursor-pointer"
          >
            <Eye className="w-4 h-4" />
            <span>Animation Vault (7 GIFs)</span>
          </button>
        </div>
      </div>
    </header>
  );
}

