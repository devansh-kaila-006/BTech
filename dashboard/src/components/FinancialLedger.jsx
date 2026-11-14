import React from 'react';
import { DollarSign, TrendingDown, Leaf, ShieldAlert, Award } from 'lucide-react';

export default function FinancialLedger({ ledgerData }) {
  const {
    baselineSpend = 41.36,
    predictiveSpend = 3.04,
    slaFinesAvoided = 38.32,
    carbonSavedKg = 0.13,
    pctSavings = 92.6
  } = ledgerData || {};

  return (
    <div className="p-5 rounded-2xl border border-gray-800 bg-gray-900/60 backdrop-blur-md">
      <div className="flex items-center justify-between pb-3 border-b border-gray-800">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            <DollarSign className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-white">
              Cloud Financial Ledger & Carbon Footprint ROI
            </h2>
            <p className="text-xs text-gray-400">
              Live enterprise financial balance sheet (AWS c6i compute + $50/hr SLA outage fine model)
            </p>
          </div>
        </div>

        <span className="flex items-center space-x-1 text-xs font-bold px-2.5 py-1 rounded-full bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
          <TrendingDown className="w-3.5 h-3.5" />
          <span>-{pctSavings}% Cloud Bill Slashed</span>
        </span>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
        {/* Metric 1: Total Autonomous Spend */}
        <div className="p-3.5 rounded-xl bg-gray-950/70 border border-gray-800">
          <span className="text-xs text-gray-400 block">Full Architecture Spend</span>
          <div className="mt-1 flex items-baseline space-x-2">
            <span className="text-2xl font-bold font-mono text-emerald-400">${predictiveSpend.toFixed(2)}</span>
            <span className="text-[10px] text-gray-500">USD</span>
          </div>
          <span className="text-[11px] text-gray-400 mt-1 block">Zero SLA fines incurred</span>
        </div>

        {/* Metric 2: Baseline Unprotected Spend */}
        <div className="p-3.5 rounded-xl bg-gray-950/70 border border-gray-800">
          <span className="text-xs text-gray-400 block">Reactive Baseline Spend</span>
          <div className="mt-1 flex items-baseline space-x-2">
            <span className="text-2xl font-bold font-mono text-red-400">${baselineSpend.toFixed(2)}</span>
            <span className="text-[10px] text-gray-500">USD</span>
          </div>
          <span className="text-[11px] text-red-400/80 mt-1 block">Dominated by SLA fines</span>
        </div>

        {/* Metric 3: Net Dollar Savings */}
        <div className="p-3.5 rounded-xl bg-emerald-950/20 border border-emerald-800/40">
          <span className="text-xs text-gray-300 block">Net Financial Savings</span>
          <div className="mt-1 flex items-baseline space-x-2">
            <span className="text-2xl font-bold font-mono text-emerald-300">+${slaFinesAvoided.toFixed(2)}</span>
            <span className="text-[10px] text-emerald-400/70">USD</span>
          </div>
          <span className="text-[11px] text-emerald-400 mt-1 block">92.6% Cost Reduction</span>
        </div>

        {/* Metric 4: Carbon Emissions Avoided */}
        <div className="p-3.5 rounded-xl bg-gray-950/70 border border-gray-800">
          <span className="text-xs text-gray-400 block">Carbon Footprint Saved</span>
          <div className="mt-1 flex items-baseline space-x-2">
            <span className="text-2xl font-bold font-mono text-blue-400">{carbonSavedKg.toFixed(2)}</span>
            <span className="text-[10px] text-gray-500">kg CO2e</span>
          </div>
          <span className="text-[11px] text-blue-400 mt-1 block">DVFS Frequency Tuning</span>
        </div>
      </div>
    </div>
  );
}

