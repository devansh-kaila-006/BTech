import React, { useState, useEffect } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  Legend
} from 'recharts';
import { Play, Pause, RotateCcw, AlertTriangle, ShieldCheck, Sparkles, TrendingDown, Eye } from 'lucide-react';

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-gray-900/95 border border-gray-700/80 rounded-xl p-3 shadow-2xl backdrop-blur-md text-xs">
        <p className="font-mono text-gray-400 font-bold mb-1.5 border-b border-gray-800 pb-1">
          Tick Step: {label}
        </p>
        {payload.map((entry, index) => (
          <div key={`item-${index}`} className="flex items-center justify-between space-x-4 py-0.5">
            <span style={{ color: entry.color }} className="font-medium">
              {entry.name}:
            </span>
            <span className="font-mono font-bold text-white">
              {entry.value}
              {entry.unit || ''}
            </span>
          </div>
        ))}
      </div>
    );
  }
  return null;
};

export default function ScenarioCard({ scenario, isCompact = false }) {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false); // Paused by default!
  const [showFullGraph, setShowFullGraph] = useState(true); // Default to full graph for immediate inspection!

  const totalSteps = scenario.data.length;

  // Playback timer
  useEffect(() => {
    let timer = null;
    if (isPlaying) {
      setShowFullGraph(false);
      timer = setInterval(() => {
        setCurrentStep((prev) => {
          if (prev >= totalSteps - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 120);
    }
    return () => clearInterval(timer);
  }, [isPlaying, totalSteps]);

  // Data to display
  const displayData = showFullGraph ? scenario.data : scenario.data.slice(0, currentStep + 1);

  const isFinancial = scenario.id === 'roi';

  const badgeColors = {
    blue: 'bg-blue-600/30 text-blue-300 border-blue-500/40',
    amber: 'bg-amber-600/30 text-amber-300 border-amber-500/40',
    purple: 'bg-purple-600/30 text-purple-300 border-purple-500/40',
    emerald: 'bg-emerald-600/30 text-emerald-300 border-emerald-500/40'
  };

  return (
    <div className="p-6 rounded-2xl border border-gray-800 bg-gray-900/70 backdrop-blur-md space-y-5">
      {/* Header Banner */}
      <div className="flex flex-col lg:flex-row lg:items-start justify-between gap-3 border-b border-gray-800/80 pb-4">
        <div className="space-y-1.5">
          <div className="flex items-center space-x-2">
            <span className={`text-[10px] font-bold uppercase tracking-wider px-2.5 py-0.5 rounded-full border ${badgeColors[scenario.tagColor] || badgeColors.blue}`}>
              {scenario.badge}
            </span>
            <h3 className="text-base font-bold text-white tracking-tight">{scenario.title}</h3>
          </div>

          <div className="text-xs text-gray-300 space-y-1 mt-1 max-w-4xl">
            <p><strong className="text-red-400">The Problem:</strong> {scenario.issue}</p>
            <p><strong className="text-emerald-400">Our Solution:</strong> {scenario.solution}</p>
          </div>
        </div>

        {/* Presentation Script Box */}
        <div className="p-3 rounded-xl bg-indigo-950/40 border border-indigo-800/50 lg:max-w-xs shrink-0">
          <span className="text-[10px] uppercase font-bold text-indigo-300 tracking-wider flex items-center space-x-1">
            <Sparkles className="w-3 h-3 text-indigo-400" />
            <span>Say to Examiner</span>
          </span>
          <p className="text-[11px] text-indigo-100 mt-1 italic leading-snug">
            "{scenario.vivaScript}"
          </p>
        </div>
      </div>

      {/* Dedicated Graph */}
      <div className="h-[300px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={displayData} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
            <defs>
              <linearGradient id={`predGrad-${scenario.id}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10b981" stopOpacity={0.35} />
                <stop offset="95%" stopColor="#10b981" stopOpacity={0.0} />
              </linearGradient>
              <linearGradient id={`baseGrad-${scenario.id}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#ef4444" stopOpacity={0.25} />
                <stop offset="95%" stopColor="#ef4444" stopOpacity={0.0} />
              </linearGradient>
            </defs>

            <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.4} />
            <XAxis
              dataKey="step"
              stroke="#9ca3af"
              fontSize={11}
              tickLine={false}
              label={{ value: 'Simulation Steps (Alibaba Trace Ticks)', position: 'insideBottom', offset: -5, fill: '#6b7280', fontSize: 10 }}
            />
            <YAxis
              stroke="#9ca3af"
              fontSize={11}
              tickLine={false}
              domain={isFinancial ? [0, 45] : [0, (dataMax) => Math.max(120, Math.ceil(dataMax / 20) * 20)]}
              label={{ value: isFinancial ? 'Spend ($ USD)' : 'Utilization (%)', angle: -90, position: 'insideLeft', fill: '#6b7280', fontSize: 10 }}
            />

            <Tooltip content={<CustomTooltip />} />
            <Legend verticalAlign="top" align="right" wrapperStyle={{ paddingBottom: '8px', fontSize: '11px' }} />

            {/* SLA Reference Line for utilization graphs */}
            {!isFinancial && (
              <ReferenceLine
                y={80}
                stroke="#f43f5e"
                strokeWidth={2}
                strokeDasharray="4 4"
                label={{ value: '80% SLA Limit', fill: '#f43f5e', fontSize: 10, position: 'insideTopLeft' }}
              />
            )}

            {/* Crash Event Marker for Scenario 2 */}
            {scenario.id === 'crash' && (
              <ReferenceLine
                x={25}
                stroke="#f59e0b"
                strokeWidth={2}
                strokeDasharray="3 3"
                label={{ value: '💥 50% Node Crash', fill: '#f59e0b', fontSize: 10, position: 'insideTopRight' }}
              />
            )}

            {/* Burst Event Window for Scenario 3 */}
            {scenario.id === 'burst' && (
              <ReferenceLine
                x={30}
                stroke="#a855f7"
                strokeWidth={2}
                strokeDasharray="3 3"
                label={{ value: '⚡ 4.5x Surge Influx', fill: '#a855f7', fontSize: 10, position: 'insideTopRight' }}
              />
            )}

            {/* Curves */}
            {isFinancial ? (
              <>
                <Line
                  type="monotone"
                  dataKey="baselineCost"
                  name="Reactive Baseline Spend ($)"
                  stroke="#ef4444"
                  strokeWidth={2.2}
                  strokeDasharray="4 4"
                  dot={false}
                  unit=" USD"
                  isAnimationActive={false}
                />
                <Line
                  type="monotone"
                  dataKey="predictiveCost"
                  name="Full Safe Autonomous Spend ($)"
                  stroke="#10b981"
                  strokeWidth={2.5}
                  dot={false}
                  unit=" USD"
                  isAnimationActive={false}
                />
              </>
            ) : (
              <>
                <Area
                  type="monotone"
                  dataKey="baselineUtil"
                  name="Reactive Baseline RL"
                  stroke="#ef4444"
                  strokeWidth={2}
                  strokeDasharray="4 4"
                  fill={`url(#baseGrad-${scenario.id})`}
                  unit="%"
                  isAnimationActive={false}
                />
                <Area
                  type="monotone"
                  dataKey="predictiveUtil"
                  name="Predictive RL + Safe Shield (Ours)"
                  stroke="#10b981"
                  strokeWidth={2.5}
                  fill={`url(#predGrad-${scenario.id})`}
                  unit="%"
                  isAnimationActive={false}
                />
                <Line
                  type="stepAfter"
                  dataKey="predictiveNodes"
                  name="Cluster Nodes (Ours)"
                  stroke="#38bdf8"
                  strokeWidth={1.5}
                  dot={false}
                  unit=" nodes"
                  isAnimationActive={false}
                />
              </>
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Clean Control Bar */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-3 p-3 rounded-xl bg-gray-950/80 border border-gray-800">
        <div className="flex items-center space-x-2 w-full sm:w-auto">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className={`flex items-center space-x-1.5 px-4 py-2 rounded-xl text-xs font-bold transition-all shadow-md cursor-pointer ${
              isPlaying
                ? 'bg-amber-500 hover:bg-amber-600 text-gray-950 shadow-amber-500/20'
                : 'bg-emerald-500 hover:bg-emerald-600 text-gray-950 shadow-emerald-500/20'
            }`}
          >
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 fill-current" />}
            <span>{isPlaying ? 'PAUSE DEMO' : 'RUN SCENARIO'}</span>
          </button>

          <button
            onClick={() => {
              setIsPlaying(false);
              setCurrentStep(0);
              setShowFullGraph(false);
            }}
            className="p-2 rounded-xl bg-gray-800 hover:bg-gray-700 text-gray-300 transition-all cursor-pointer"
            title="Reset from Start"
          >
            <RotateCcw className="w-4 h-4" />
          </button>

          <button
            onClick={() => {
              setIsPlaying(false);
              setShowFullGraph(true);
            }}
            className={`px-3 py-2 rounded-xl text-xs font-semibold border transition-all cursor-pointer ${
              showFullGraph
                ? 'bg-blue-600/30 text-blue-300 border-blue-500/40'
                : 'bg-gray-800/60 text-gray-400 border-gray-700 hover:text-white'
            }`}
          >
            Show Full Trajectory
          </button>
        </div>

        {/* Step Counter */}
        <div className="text-xs font-mono text-gray-400">
          Step: <strong className="text-white">{showFullGraph ? totalSteps : currentStep + 1}</strong> / {totalSteps}
        </div>
      </div>

      {/* 3 Key Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-1">
        {scenario.metrics.map((m, i) => (
          <div key={i} className="p-3 rounded-xl bg-gray-950/70 border border-gray-800/80">
            <span className="text-[11px] text-gray-400 block">{m.label}</span>
            <div className="mt-1 flex items-baseline justify-between">
              <span className="text-xs text-red-400 line-through opacity-80">{m.reactive}</span>
              <span className="text-sm font-bold font-mono text-emerald-400">{m.predictive}</span>
            </div>
            <span className="text-[10px] font-bold text-emerald-400 uppercase tracking-wider block mt-1">
              ✓ {m.winner}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
