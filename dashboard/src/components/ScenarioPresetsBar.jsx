import React from 'react';
import { Sparkles, Zap, Skull, Award, CheckCircle } from 'lucide-react';

export default function ScenarioPresetsBar({
  onSelectPreset,
  activePreset
}) {
  const presets = [
    {
      id: 'steady',
      name: 'Preset 1: Nominal Steady State',
      icon: Sparkles,
      color: 'from-blue-600/30 to-indigo-600/30 border-blue-500/40 text-blue-300',
      activeColor: 'bg-blue-600 text-white border-blue-400 shadow-blue-500/30 shadow-md',
      desc: 'Smooth diurnal trace with energy-efficient baseline scaling.'
    },
    {
      id: 'burst',
      name: 'Preset 2: Flash Crowd Surge (4.5x)',
      icon: Zap,
      color: 'from-purple-600/30 to-pink-600/30 border-purple-500/40 text-purple-300',
      activeColor: 'bg-purple-600 text-white border-purple-400 shadow-purple-500/30 shadow-md',
      desc: 'Sudden traffic spike; verifies Transformer 60s lookahead pre-warming.'
    },
    {
      id: 'crash',
      name: 'Preset 3: 50% Node Crash Panic',
      icon: Skull,
      color: 'from-amber-600/30 to-red-600/30 border-amber-500/40 text-amber-300',
      activeColor: 'bg-amber-600 text-white border-amber-400 shadow-amber-500/30 shadow-md',
      desc: 'Simulates hardware panic; verifies Safe Shield 2-tick instant clamp.'
    },
    {
      id: 'pareto',
      name: 'Preset 4: Strict Pareto Dominance',
      icon: Award,
      color: 'from-emerald-600/30 to-teal-600/30 border-emerald-500/40 text-emerald-300',
      activeColor: 'bg-emerald-600 text-white border-emerald-400 shadow-emerald-500/30 shadow-md',
      desc: 'Demonstrates 92.6% net cost reduction and zero SLA fines.'
    }
  ];

  return (
    <div className="p-3.5 rounded-2xl border border-gray-800 bg-gray-900/80 backdrop-blur-md">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 mb-2.5 px-1">
        <div className="flex items-center space-x-2">
          <span className="text-xs font-bold text-gray-200 uppercase tracking-wider">
            Viva Presentation Quick Presets
          </span>
          <span className="text-[10px] text-gray-400 font-medium">
            (One-click demonstration scenarios for committee examiners)
          </span>
        </div>
        <span className="text-[11px] text-emerald-400 font-semibold flex items-center space-x-1">
          <CheckCircle className="w-3.5 h-3.5" />
          <span>Interactive Live Controls</span>
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5">
        {presets.map((preset) => {
          const Icon = preset.icon;
          const isActive = activePreset === preset.id;
          return (
            <button
              key={preset.id}
              onClick={() => onSelectPreset(preset.id)}
              className={`flex items-center space-x-3 p-2.5 rounded-xl border text-left transition-all cursor-pointer ${
                isActive
                  ? preset.activeColor
                  : `bg-gradient-to-r ${preset.color} hover:brightness-125`
              }`}
            >
              <div className={`p-2 rounded-lg ${isActive ? 'bg-black/20 text-white' : 'bg-gray-900/60'}`}>
                <Icon className="w-4 h-4 shrink-0" />
              </div>
              <div className="min-w-0">
                <span className="text-xs font-bold block truncate">{preset.name}</span>
                <span className={`text-[10px] block truncate ${isActive ? 'text-white/80' : 'text-gray-400'}`}>
                  {preset.desc}
                </span>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
