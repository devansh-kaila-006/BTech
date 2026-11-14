import React from 'react';
import { Cpu, Server, Zap, Leaf, ShieldAlert, ShieldCheck } from 'lucide-react';

export default function ClusterGauges({ telemetry, isShieldActive }) {
  const {
    cpuUtil = 45.2,
    gpuUtil = 38.0,
    cpuNodes = 8.0,
    gpuNodes = 3.0,
    totalPower = 1840,
    carbonRate = 0.42,
    isBreach = false
  } = telemetry || {};

  const getUtilColor = (val) => {
    if (val >= 80) return 'text-red-400 bg-red-950/40 border-red-800/60';
    if (val >= 70) return 'text-amber-400 bg-amber-950/40 border-amber-800/60';
    return 'text-emerald-400 bg-emerald-950/40 border-emerald-800/60';
  };

  const totalNodes = (cpuNodes + gpuNodes).toFixed(1);

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
      {/* 1. CPU Utilization */}
      <div className={`p-4 rounded-xl border backdrop-blur-md transition-all ${getUtilColor(cpuUtil)}`}>
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-gray-400">CPU Util</span>
          <Cpu className="w-4 h-4" />
        </div>
        <div className="mt-2 flex items-baseline justify-between">
          <span className="text-2xl font-bold font-mono tracking-tight">{cpuUtil.toFixed(1)}%</span>
          <span className="text-[10px] uppercase font-bold tracking-wider opacity-80">
            {cpuUtil > 80 ? 'SLA Breach' : 'Healthy'}
          </span>
        </div>
        <div className="w-full bg-gray-800/60 rounded-full h-1.5 mt-3 overflow-hidden">
          <div
            className={`h-full transition-all duration-300 ${cpuUtil > 80 ? 'bg-red-500' : cpuUtil > 70 ? 'bg-amber-400' : 'bg-emerald-400'}`}
            style={{ width: `${Math.min(100, cpuUtil)}%` }}
          />
        </div>
      </div>

      {/* 2. GPU Utilization */}
      <div className={`p-4 rounded-xl border backdrop-blur-md transition-all ${getUtilColor(gpuUtil)}`}>
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-gray-400">GPU Util</span>
          <Zap className="w-4 h-4 text-purple-400" />
        </div>
        <div className="mt-2 flex items-baseline justify-between">
          <span className="text-2xl font-bold font-mono tracking-tight text-purple-400">{gpuUtil.toFixed(1)}%</span>
          <span className="text-[10px] uppercase font-bold tracking-wider text-purple-300">
            DVFS 1.2 GHz
          </span>
        </div>
        <div className="w-full bg-gray-800/60 rounded-full h-1.5 mt-3 overflow-hidden">
          <div
            className="h-full bg-purple-500 transition-all duration-300"
            style={{ width: `${Math.min(100, gpuUtil)}%` }}
          />
        </div>
      </div>

      {/* 3. Total Active Nodes */}
      <div className="p-4 rounded-xl border border-gray-800 bg-gray-900/60 backdrop-blur-md">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-gray-400">Cluster Fleet</span>
          <Server className="w-4 h-4 text-blue-400" />
        </div>
        <div className="mt-2 flex items-baseline justify-between">
          <span className="text-2xl font-bold font-mono tracking-tight text-white">{totalNodes}</span>
          <span className="text-[10px] text-gray-400">VM Nodes</span>
        </div>
        <p className="text-[11px] text-gray-400 mt-2">
          <span className="text-blue-400 font-semibold">{cpuNodes.toFixed(1)} CPU</span> + <span className="text-purple-400 font-semibold">{gpuNodes.toFixed(1)} GPU</span>
        </p>
      </div>

      {/* 4. Power Consumption */}
      <div className="p-4 rounded-xl border border-gray-800 bg-gray-900/60 backdrop-blur-md">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-gray-400">Cluster Power</span>
          <Zap className="w-4 h-4 text-amber-400" />
        </div>
        <div className="mt-2 flex items-baseline justify-between">
          <span className="text-2xl font-bold font-mono tracking-tight text-amber-400">{Math.round(totalPower)}</span>
          <span className="text-[10px] text-gray-400">Watts</span>
        </div>
        <p className="text-[11px] text-emerald-400 mt-2 font-medium">
          -22.5% Energy (DVFS)
        </p>
      </div>

      {/* 5. Carbon Intensity */}
      <div className="p-4 rounded-xl border border-gray-800 bg-gray-900/60 backdrop-blur-md">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-gray-400">Grid Carbon</span>
          <Leaf className="w-4 h-4 text-emerald-400" />
        </div>
        <div className="mt-2 flex items-baseline justify-between">
          <span className="text-2xl font-bold font-mono tracking-tight text-emerald-400">{carbonRate.toFixed(2)}</span>
          <span className="text-[10px] text-gray-400">kg CO2e</span>
        </div>
        <p className="text-[11px] text-gray-400 mt-2">
          Diurnal Clean Grid
        </p>
      </div>

      {/* 6. Safe Shield Status */}
      <div className={`p-4 rounded-xl border backdrop-blur-md transition-all ${isShieldActive ? 'border-amber-500/70 bg-amber-950/30' : 'border-emerald-600/40 bg-emerald-950/20'}`}>
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-gray-400">Safe Action Shield</span>
          {isShieldActive ? (
            <ShieldAlert className="w-4 h-4 text-amber-400 animate-pulse" />
          ) : (
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
          )}
        </div>
        <div className="mt-2 flex items-baseline justify-between">
          <span className={`text-base font-bold tracking-tight ${isShieldActive ? 'text-amber-400' : 'text-emerald-400'}`}>
            {isShieldActive ? 'INTERVENING' : 'MONITORING'}
          </span>
        </div>
        <p className="text-[11px] text-gray-400 mt-2">
          {isShieldActive ? 'Clamping unsafe scale' : 'Zero SLA Breaches'}
        </p>
      </div>
    </div>
  );
}

