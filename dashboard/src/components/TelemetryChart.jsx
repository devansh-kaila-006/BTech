import React from 'react';
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
import { TrendingUp, AlertTriangle } from 'lucide-react';

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-gray-900/95 border border-gray-700/80 rounded-xl p-3 shadow-2xl backdrop-blur-md text-xs">
        <p className="font-mono text-gray-400 font-bold mb-1.5 border-b border-gray-800 pb-1">
          Trace Step: {label}
        </p>
        {payload.map((entry, index) => (
          <div key={`item-${index}`} className="flex items-center justify-between space-x-4 py-0.5">
            <span style={{ color: entry.color }} className="font-medium">
              {entry.name}:
            </span>
            <span className="font-mono font-bold text-white">
              {typeof entry.value === 'number' ? entry.value.toFixed(1) : entry.value}
              {entry.unit || '%'}
            </span>
          </div>
        ))}
      </div>
    );
  }
  return null;
};

export default function TelemetryChart({ chartData, activePolicy, chaosEvent }) {
  return (
    <div className="p-5 rounded-2xl border border-gray-800 bg-gray-900/60 backdrop-blur-md">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-gray-800 gap-2">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 rounded-lg bg-blue-500/10 text-blue-400 border border-blue-500/20">
            <TrendingUp className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-white">
              Real-Time Cluster Utilization & Starvation Dynamics
            </h2>
            <p className="text-xs text-gray-400">
              Live comparison of continuous scaling trajectories against the 80% SLA contract threshold
            </p>
          </div>
        </div>

        {chaosEvent && (
          <div className="flex items-center space-x-2 px-3 py-1 bg-amber-500/10 border border-amber-500/30 rounded-lg text-xs text-amber-400 animate-pulse font-medium">
            <AlertTriangle className="w-3.5 h-3.5" />
            <span>Chaos Injected: {chaosEvent}</span>
          </div>
        )}
      </div>

      <div className="h-[360px] w-full mt-4">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
            <defs>
              <linearGradient id="predGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#10b981" stopOpacity={0.0} />
              </linearGradient>
              <linearGradient id="baseGradient" x1="0" y1="0" x2="0" y2="1">
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
              label={{ value: 'Alibaba Cluster Trace Ticks (2s / tick)', position: 'insideBottom', offset: -5, fill: '#6b7280', fontSize: 10 }}
            />
            <YAxis
              stroke="#9ca3af"
              fontSize={11}
              tickLine={false}
              domain={[0, (dataMax) => Math.max(120, Math.ceil(dataMax / 20) * 20)]}
              label={{ value: 'Utilization (%)', angle: -90, position: 'insideLeft', fill: '#6b7280', fontSize: 10 }}
            />

            <Tooltip content={<CustomTooltip />} />
            <Legend
              verticalAlign="top"
              align="right"
              wrapperStyle={{ paddingBottom: '10px', fontSize: '12px' }}
            />

            {/* 80% SLA Threshold */}
            <ReferenceLine
              y={80}
              stroke="#f43f5e"
              strokeWidth={2}
              strokeDasharray="4 4"
              label={{
                value: '80% SLA Violation Limit',
                fill: '#f43f5e',
                fontSize: 11,
                position: 'insideTopLeft',
                fontWeight: 600
              }}
            />

            {/* Reactive Baseline Line & Area */}
            <Area
              type="monotone"
              dataKey="baselineUtil"
              name="Reactive Baseline RL"
              stroke="#ef4444"
              strokeWidth={2}
              strokeDasharray="4 4"
              fill="url(#baseGradient)"
              isAnimationActive={false}
            />

            {/* Predictive RL (Ours) */}
            <Area
              type="monotone"
              dataKey="predictiveUtil"
              name="Predictive RL + Safe Shield (Ours)"
              stroke="#10b981"
              strokeWidth={2.5}
              fill="url(#predGradient)"
              isAnimationActive={false}
            />

            {/* Active Nodes Line (secondary) */}
            <Line
              type="stepAfter"
              dataKey="nodes"
              name="Active Nodes (Count)"
              stroke="#38bdf8"
              strokeWidth={1.5}
              dot={false}
              unit=" nodes"
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

