import React from 'react';
import { Terminal, ShieldAlert, ShieldCheck, ArrowUpRight, ArrowDownRight, Zap } from 'lucide-react';

export default function DecisionAuditLog({ currentStep, telemetry, activeChaos }) {
  // Generate the last 4 audit log entries dynamically based on step & chaos
  const logEntries = [
    {
      tick: currentStep,
      type: telemetry.isShieldIntervening ? 'SHIELD_OVERRIDE' : 'PPO_DECISION',
      oracleForecast: `${(telemetry.cpuUtil * 1.15).toFixed(1)}% at t+5 (Transformer)`,
      action: telemetry.isShieldIntervening ? 'Clamped: +5.0 VM Nodes (Barrier Bound)' : `Velocity: +${(telemetry.cpuNodes * 0.12).toFixed(2)} [-1, +1]`,
      status: telemetry.isShieldIntervening ? 'CLAMPS_APPLIED' : 'SAFE_PASS',
      latency: '8.4 ms',
      reward: (-telemetry.cpuNodes * 0.1).toFixed(2)
    },
    {
      tick: Math.max(0, currentStep - 1),
      type: 'PPO_DECISION',
      oracleForecast: `${(telemetry.cpuUtil * 1.05).toFixed(1)}% at t+5 (Transformer)`,
      action: `Velocity: +${(telemetry.cpuNodes * 0.08).toFixed(2)}`,
      status: 'SAFE_PASS',
      latency: '8.6 ms',
      reward: (-telemetry.cpuNodes * 0.09).toFixed(2)
    },
    {
      tick: Math.max(0, currentStep - 2),
      type: 'ORACLE_FUSION',
      oracleForecast: 'LSTM Workload + XGBoost Latency + Power DNN',
      action: 'State 7D Tensor Built (Sub-2.5ms)',
      status: 'SYNCHRONIZED',
      latency: '2.4 ms',
      reward: '-0.81'
    }
  ];

  return (
    <div className="p-5 rounded-2xl border border-gray-800 bg-gray-900/60 backdrop-blur-md">
      <div className="flex items-center justify-between pb-3 border-b border-gray-800">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            <Terminal className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-white">
              Real-Time AI Decision Audit Log & Reasoning Trail
            </h2>
            <p className="text-xs text-gray-400">
              Live inspection of Oracle state features, continuous PPO actions, and runtime barrier clamping
            </p>
          </div>
        </div>

        <span className="text-[11px] font-mono text-gray-400">
          Inference Latency: <strong className="text-emerald-400">8.57 ms (p50)</strong>
        </span>
      </div>

      <div className="space-y-2.5 mt-3.5 font-mono text-xs">
        {logEntries.map((log, idx) => {
          const isIntervention = log.type === 'SHIELD_OVERRIDE';
          return (
            <div
              key={idx}
              className={`p-2.5 rounded-xl border flex flex-col md:flex-row items-start md:items-center justify-between gap-2 transition-all ${
                isIntervention
                  ? 'border-amber-500/50 bg-amber-950/30 text-amber-200'
                  : 'border-gray-800/80 bg-gray-950/80 text-gray-300'
              }`}
            >
              <div className="flex items-center space-x-2.5">
                <span className="text-blue-400 font-bold">[t={log.tick}]</span>
                {isIntervention ? (
                  <span className="px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-400 font-bold text-[10px] border border-amber-500/40 flex items-center space-x-1">
                    <ShieldAlert className="w-3 h-3" />
                    <span>SHIELD CLAMP</span>
                  </span>
                ) : (
                  <span className="px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-300 font-bold text-[10px] border border-blue-500/30 flex items-center space-x-1">
                    <Zap className="w-3 h-3" />
                    <span>PPO FORWARD</span>
                  </span>
                )}
                <span className="text-gray-400 text-[11px]">{log.action}</span>
              </div>

              <div className="flex items-center space-x-3 text-[11px] text-gray-400">
                <span>Forecast: <strong className="text-gray-200">{log.oracleForecast}</strong></span>
                <span className="text-gray-600">•</span>
                <span>Latency: <strong className="text-emerald-400">{log.latency}</strong></span>
                <span className="text-gray-600">•</span>
                <span>Reward: <strong className="text-indigo-300">{log.reward}</strong></span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

