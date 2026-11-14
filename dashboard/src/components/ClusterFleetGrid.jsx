import React from 'react';
import { Server, Cpu, Zap, Activity } from 'lucide-react';

export default function ClusterFleetGrid({ cpuNodes = 8.0, gpuNodes = 3.0, cpuUtil = 45.0, gpuUtil = 38.0 }) {
  const activeCpuCount = Math.max(1, Math.min(12, Math.round(cpuNodes)));
  const activeGpuCount = Math.max(1, Math.min(6, Math.round(gpuNodes)));

  return (
    <div className="p-5 rounded-2xl border border-gray-800 bg-gray-900/60 backdrop-blur-md">
      <div className="flex items-center justify-between pb-3 border-b border-gray-800">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 rounded-lg bg-blue-500/10 text-blue-400 border border-blue-500/20">
            <Server className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-white">
              Heterogeneous Kubernetes Node Fleet (Physical Hardware View)
            </h2>
            <p className="text-xs text-gray-400">
              Real-time allocation across decoupled CPU worker nodes and GPU accelerators with DVFS frequency scaling
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-3 text-xs">
          <span className="flex items-center space-x-1.5 text-blue-400 font-medium">
            <span className="w-2 h-2 rounded-full bg-blue-400"></span>
            <span>Pool 0: {cpuNodes.toFixed(1)} CPU Nodes</span>
          </span>
          <span className="flex items-center space-x-1.5 text-purple-400 font-medium">
            <span className="w-2 h-2 rounded-full bg-purple-400"></span>
            <span>Pool 1: {gpuNodes.toFixed(1)} GPU Nodes</span>
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-4">
        {/* Pool 0: CPU-Optimized Instances */}
        <div className="p-3.5 rounded-xl bg-gray-950/70 border border-gray-800/80">
          <div className="flex items-center justify-between mb-2.5">
            <span className="text-xs font-bold text-blue-300 flex items-center space-x-1.5">
              <Cpu className="w-3.5 h-3.5" />
              <span>Pool 0: 16-Core Compute Servers (c6i.4xlarge)</span>
            </span>
            <span className="text-[11px] font-mono text-gray-400">
              Avg Util: <strong className="text-white">{cpuUtil.toFixed(1)}%</strong>
            </span>
          </div>

          <div className="grid grid-cols-3 sm:grid-cols-4 gap-2">
            {Array.from({ length: activeCpuCount }).map((_, i) => {
              const nodeUtil = Math.max(10, Math.min(99, Math.round(cpuUtil + (i % 3 - 1) * 8)));
              const isOverloaded = nodeUtil > 80;
              return (
                <div
                  key={`cpu-node-${i}`}
                  className={`p-2 rounded-lg border text-xs transition-all ${
                    isOverloaded
                      ? 'border-red-600/60 bg-red-950/30'
                      : 'border-blue-900/40 bg-blue-950/20'
                  }`}
                >
                  <div className="flex items-center justify-between text-[10px] text-gray-400">
                    <span className="font-mono">k8s-cpu-{i+1}</span>
                    <span className="text-blue-400">16 cores</span>
                  </div>
                  <div className="mt-1 flex items-baseline justify-between">
                    <span className={`font-mono font-bold text-xs ${isOverloaded ? 'text-red-400' : 'text-gray-200'}`}>
                      {nodeUtil}%
                    </span>
                    <span className="text-[9px] text-gray-500">Active</span>
                  </div>
                  <div className="w-full bg-gray-800 rounded-full h-1 mt-1.5 overflow-hidden">
                    <div
                      className={`h-full ${isOverloaded ? 'bg-red-500' : 'bg-blue-500'}`}
                      style={{ width: `${nodeUtil}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Pool 1: GPU-Accelerated Instances */}
        <div className="p-3.5 rounded-xl bg-gray-950/70 border border-gray-800/80">
          <div className="flex items-center justify-between mb-2.5">
            <span className="text-xs font-bold text-purple-300 flex items-center space-x-1.5">
              <Zap className="w-3.5 h-3.5" />
              <span>Pool 1: 8x GPU Accelerators (g5.12xlarge + DVFS)</span>
            </span>
            <span className="text-[11px] font-mono text-purple-400 font-bold px-2 py-0.5 rounded bg-purple-950/40 border border-purple-800/50">
              1,215 MHz (DVFS Active)
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            {Array.from({ length: activeGpuCount }).map((_, i) => {
              const nodeGpuUtil = Math.max(15, Math.min(95, Math.round(gpuUtil + (i % 2 - 0.5) * 10)));
              return (
                <div
                  key={`gpu-node-${i}`}
                  className="p-2.5 rounded-lg border border-purple-900/40 bg-purple-950/20 text-xs"
                >
                  <div className="flex items-center justify-between text-[10px] text-gray-400">
                    <span className="font-mono text-purple-300">k8s-gpu-{i+1}</span>
                    <span className="text-purple-400 font-mono">8x A10G</span>
                  </div>
                  <div className="mt-1 flex items-baseline justify-between">
                    <span className="font-mono font-bold text-xs text-purple-300">
                      {nodeGpuUtil}% Util
                    </span>
                    <span className="text-[9px] text-emerald-400">58°C</span>
                  </div>
                  <div className="w-full bg-gray-800 rounded-full h-1 mt-1.5 overflow-hidden">
                    <div
                      className="h-full bg-purple-500"
                      style={{ width: `${nodeGpuUtil}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
