import React, { useState, useEffect, useRef, useMemo } from 'react';
import Navbar from './components/Navbar';
import ClusterGauges from './components/ClusterGauges';
import TelemetryChart from './components/TelemetryChart';
import SimulationControlBar from './components/SimulationControlBar';
import PolicyComparisonPanel from './components/PolicyComparisonPanel';
import ChaosMonkeyPanel from './components/ChaosMonkeyPanel';
import FinancialLedger from './components/FinancialLedger';
import AnimationVaultModal from './components/AnimationVaultModal';

import traceSample from './data/trace_sample.json';

export default function App() {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);
  const [speed, setSpeed] = useState(2);
  const [activeTier, setActiveTier] = useState('tier4');
  const [activeChaos, setActiveChaos] = useState(null);
  const [isAnimationModalOpen, setIsAnimationModalOpen] = useState(false);
  const [isLiveBackend, setIsLiveBackend] = useState(false);

  const maxSteps = traceSample.length; // 250 steps

  // Check if local FastAPI backend is alive
  useEffect(() => {
    fetch('http://localhost:8000/health')
      .then((res) => {
        if (res.ok) setIsLiveBackend(true);
      })
      .catch(() => setIsLiveBackend(false));
  }, []);

  // Playback timer loop
  useEffect(() => {
    let timer = null;
    if (isPlaying) {
      const intervalMs = Math.max(80, Math.floor(600 / speed));
      timer = setInterval(() => {
        setCurrentStep((prev) => {
          if (prev >= maxSteps - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, intervalMs);
    }
    return () => clearInterval(timer);
  }, [isPlaying, speed, maxSteps]);

  // Current pod metadata
  const currentPod = traceSample[currentStep] || traceSample[0];

  // Live telemetry calculation based on current state & chaos
  const telemetry = useMemo(() => {
    let demand = (currentPod.cores_demanded || 8.5);

    // Chaos multipliers
    let burstFactor = 1.0;
    let crashPenalty = 1.0;

    if (activeChaos === '4.5x Flash Storm') {
      burstFactor = 4.2;
    } else if (activeChaos === '50% Node Crash') {
      crashPenalty = 0.5;
    }

    const effectiveDemand = demand * burstFactor;

    // Node dynamics per tier
    let cpuNodes = 8.0;
    let gpuNodes = 3.0;
    let isShieldIntervening = false;

    if (activeTier === 'tier1') {
      // Reactive baseline: sluggish and collapses under crash
      cpuNodes = Math.max(1.0, (effectiveDemand / 12.0) * crashPenalty);
      gpuNodes = 2.0;
    } else if (activeTier === 'tier2') {
      // Predictive RL: anticipates demand ahead of time
      cpuNodes = Math.max(2.0, (effectiveDemand / 9.5) * crashPenalty);
      gpuNodes = 3.0;
    } else if (activeTier === 'tier3') {
      // Hierarchical DVFS
      cpuNodes = Math.max(2.0, (effectiveDemand / 9.0) * crashPenalty);
      gpuNodes = Math.max(1.0, (effectiveDemand / 18.0));
    } else {
      // Tier 4: Full Safe Autonomous (Safe Shield clamps nodes)
      const safeMinNodes = effectiveDemand / 6.4; // 80% SLA threshold
      let intendedNodes = (effectiveDemand / 8.5) * crashPenalty;
      if (intendedNodes < safeMinNodes) {
        intendedNodes = safeMinNodes; // SHIELD OVERRIDE
        isShieldIntervening = true;
      }
      cpuNodes = Math.max(2.0, intendedNodes);
      gpuNodes = Math.max(1.0, intendedNodes * 0.35);
    }

    const totalCores = cpuNodes * 8.0;
    const cpuUtil = Math.min(250, (effectiveDemand / Math.max(1.0, totalCores)) * 100.0);
    const gpuUtil = Math.min(100, (currentPod.mem_norm * 100.0) * (activeTier === 'tier3' || activeTier === 'tier4' ? 0.8 : 1.0));

    // Power calculation
    const basePower = (cpuNodes * 180) + (gpuNodes * 280);
    const dvfsSavings = (activeTier === 'tier3' || activeTier === 'tier4') ? 0.775 : 1.0;
    const totalPower = basePower * dvfsSavings;
    const carbonRate = (totalPower / 1000.0) * 0.45;

    return {
      cpuUtil,
      gpuUtil,
      cpuNodes,
      gpuNodes,
      totalPower,
      carbonRate,
      isBreach: cpuUtil > 80.0,
      isShieldIntervening
    };
  }, [currentStep, activeTier, activeChaos, currentPod]);

  // Rolling history for Recharts (last 35 steps)
  const chartData = useMemo(() => {
    const windowSize = 35;
    const startIdx = Math.max(0, currentStep - windowSize);
    const endIdx = currentStep + 1;

    return traceSample.slice(startIdx, endIdx).map((item, idx) => {
      const stepNum = startIdx + idx;
      let demand = item.cores_demanded;

      // Simulate reactive baseline util vs predictive util
      let bNodes = 4.5;
      let pNodes = 8.5;

      if (activeChaos === '4.5x Flash Storm' && stepNum >= currentStep - 15) {
        demand *= 4.0;
      } else if (activeChaos === '50% Node Crash' && stepNum >= currentStep - 15) {
        bNodes *= 0.5; // Baseline suffers crash
        pNodes = Math.max(demand / 6.4, pNodes); // Shield clamps
      }

      const baselineUtil = Math.min(180, (demand / (bNodes * 8.0)) * 100.0);
      const predictiveUtil = Math.min(120, (demand / (pNodes * 8.0)) * 100.0 * 0.88);

      return {
        step: stepNum,
        baselineUtil: parseFloat(baselineUtil.toFixed(1)),
        predictiveUtil: parseFloat(predictiveUtil.toFixed(1)),
        nodes: parseFloat(pNodes.toFixed(1))
      };
    });
  }, [currentStep, activeChaos]);

  // Ledger summary
  const ledgerData = useMemo(() => {
    const elapsedHrs = (currentStep * 2.0) / 3600.0;
    const baselineSpend = Math.max(0.5, 41.36 * (currentStep / maxSteps));
    const predictiveSpend = Math.max(0.05, 3.04 * (currentStep / maxSteps));
    const slaFinesAvoided = baselineSpend - predictiveSpend;

    return {
      baselineSpend,
      predictiveSpend,
      slaFinesAvoided,
      carbonSavedKg: 0.13 * (currentStep / maxSteps),
      pctSavings: 92.6
    };
  }, [currentStep, maxSteps]);

  // Chaos handlers
  const handleInjectNodeCrash = () => {
    setActiveChaos('50% Node Crash');
  };

  const handleInjectFlashCrowd = () => {
    setActiveChaos('4.5x Flash Storm');
  };

  const handleResetChaos = () => {
    setActiveChaos(null);
  };

  return (
    <div className="min-h-screen bg-[#090d16] text-gray-100 flex flex-col selection:bg-blue-600 selection:text-white pb-12">
      {/* Top Navigation */}
      <Navbar
        isLiveBackend={isLiveBackend}
        currentStep={currentStep}
        onOpenAnimations={() => setIsAnimationModalOpen(true)}
      />

      {/* Main Container */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 w-full mt-6 space-y-6">
        {/* Row 1: Key Cluster Telemetry Gauges */}
        <ClusterGauges
          telemetry={telemetry}
          isShieldActive={telemetry.isShieldIntervening}
        />

        {/* Row 2: Live Time-Series Chart */}
        <TelemetryChart
          chartData={chartData}
          activePolicy={activeTier}
          chaosEvent={activeChaos}
        />

        {/* Row 3: Simulation Control Bar */}
        <SimulationControlBar
          isPlaying={isPlaying}
          onTogglePlay={() => setIsPlaying(!isPlaying)}
          onReset={() => {
            setCurrentStep(0);
            setIsPlaying(false);
          }}
          onStepForward={() => setCurrentStep((prev) => Math.min(maxSteps - 1, prev + 1))}
          speed={speed}
          onChangeSpeed={setSpeed}
          currentStep={currentStep}
          maxSteps={maxSteps}
          onScrub={setCurrentStep}
          currentPod={currentPod}
        />

        {/* Row 4: Two-Column Interactive Controls (Policy + Chaos) */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-8">
            <PolicyComparisonPanel
              activeTier={activeTier}
              onSelectTier={setActiveTier}
            />
          </div>
          <div className="lg:col-span-4">
            <ChaosMonkeyPanel
              onInjectNodeCrash={handleInjectNodeCrash}
              onInjectFlashCrowd={handleInjectFlashCrowd}
              onResetChaos={handleResetChaos}
              activeChaos={activeChaos}
              isShieldIntervening={telemetry.isShieldIntervening}
            />
          </div>
        </div>

        {/* Row 5: Financial Ledger & Carbon Footprint */}
        <FinancialLedger ledgerData={ledgerData} />
      </main>

      {/* Animation Vault Modal */}
      <AnimationVaultModal
        isOpen={isAnimationModalOpen}
        onClose={() => setIsAnimationModalOpen(false)}
      />
    </div>
  );
}
