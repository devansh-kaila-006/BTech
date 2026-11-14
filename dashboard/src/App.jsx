import React, { useState } from 'react';
import Navbar from './components/Navbar';
import ScenarioCard from './components/ScenarioCard';
import StageNavigator from './components/StageNavigator';
import ClusterFleetGrid from './components/ClusterFleetGrid';
import DecisionAuditLog from './components/DecisionAuditLog';
import AnimationVaultModal from './components/AnimationVaultModal';
import VivaWalkthroughModal from './components/VivaWalkthroughModal';

import scenariosData from './data/scenarios_data.json';

export default function App() {
  const [activeTab, setActiveTab] = useState('scenario1'); // scenario1, scenario2, scenario3, scenario4, or 'all'
  const [isAnimationModalOpen, setIsAnimationModalOpen] = useState(false);
  const [isVivaModalOpen, setIsVivaModalOpen] = useState(false);

  const scenarioTabs = [
    { id: 'scenario1', name: '1. Nominal Trace Replay', badge: 'Core Proof', icon: '📊' },
    { id: 'scenario2', name: '2. Node Crash Panic (-50%)', badge: 'Chaos Resilience', icon: '💥' },
    { id: 'scenario3', name: '3. Flash Crowd Storm (4.5x)', badge: '60s Lookahead', icon: '⚡' },
    { id: 'scenario4', name: '4. Financial & Carbon ROI', badge: 'Pareto Optimal', icon: '💰' },
    { id: 'all', name: 'View All 4 Scenarios', badge: 'Comparison Grid', icon: '🔲' }
  ];

  return (
    <div className="min-h-screen bg-[#090d16] text-gray-100 flex flex-col selection:bg-blue-600 selection:text-white pb-16 font-sans">
      {/* Top Navbar */}
      <Navbar
        isLiveBackend={false}
        currentStep={60}
        onOpenAnimations={() => setIsAnimationModalOpen(true)}
        onOpenVivaGuide={() => setIsVivaModalOpen(true)}
      />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 w-full mt-6 space-y-6">
        {/* Scenario Selection Tabs */}
        <div className="p-2 rounded-2xl border border-gray-800 bg-gray-900/80 backdrop-blur-md flex flex-wrap items-center justify-between gap-2">
          <div className="flex flex-wrap items-center gap-1.5 w-full sm:w-auto">
            {scenarioTabs.map((tab) => {
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center space-x-2 px-3.5 py-2 rounded-xl text-xs font-bold transition-all cursor-pointer ${
                    isActive
                      ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/25 border border-blue-400'
                      : 'bg-gray-950/60 text-gray-400 hover:text-gray-200 border border-gray-800 hover:border-gray-700'
                  }`}
                >
                  <span>{tab.icon}</span>
                  <span>{tab.name}</span>
                </button>
              );
            })}
          </div>

          <div className="hidden lg:flex items-center space-x-2 px-3 text-xs text-gray-400">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            <span>All scenarios pre-configured & paused • Click Run to demo</span>
          </div>
        </div>

        {/* Scenarios Display Area */}
        {activeTab === 'all' ? (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <ScenarioCard scenario={scenariosData.scenario1} isCompact={true} />
            <ScenarioCard scenario={scenariosData.scenario2} isCompact={true} />
            <ScenarioCard scenario={scenariosData.scenario3} isCompact={true} />
            <ScenarioCard scenario={scenariosData.scenario4} isCompact={true} />
          </div>
        ) : (
          <div className="w-full">
            <ScenarioCard scenario={scenariosData[activeTab]} />
          </div>
        )}

        {/* Physical Heterogeneous Hardware Fleet View */}
        <ClusterFleetGrid
          cpuNodes={7.5}
          gpuNodes={3.0}
          cpuUtil={54.2}
          gpuUtil={41.0}
        />

        {/* AI Decision Audit Trail */}
        <DecisionAuditLog
          currentStep={42}
          telemetry={{
            cpuUtil: 54.2,
            cpuNodes: 7.5,
            isShieldIntervening: activeTab === 'scenario2'
          }}
          activeChaos={activeTab === 'scenario2' ? '50% Node Crash' : activeTab === 'scenario3' ? '4.5x Flash Crowd' : null}
        />

        {/* 15-Stage Experimental Roadmap & Defense Navigator */}
        <StageNavigator />
      </main>

      {/* Animation Vault Modal */}
      <AnimationVaultModal
        isOpen={isAnimationModalOpen}
        onClose={() => setIsAnimationModalOpen(false)}
      />

      {/* Viva Walkthrough Modal */}
      <VivaWalkthroughModal
        isOpen={isVivaModalOpen}
        onClose={() => setIsVivaModalOpen(false)}
      />
    </div>
  );
}
