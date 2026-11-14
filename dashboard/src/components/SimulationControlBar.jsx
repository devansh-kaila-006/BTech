import React from 'react';
import { Play, Pause, RotateCcw, SkipForward, FastForward, Clock, Box } from 'lucide-react';

export default function SimulationControlBar({
  isPlaying,
  onTogglePlay,
  onReset,
  onStepForward,
  speed,
  onChangeSpeed,
  currentStep,
  maxSteps,
  onScrub,
  currentPod
}) {
  const speeds = [1, 2, 5, 10];

  return (
    <div className="p-4 rounded-2xl border border-gray-800 bg-gray-900/60 backdrop-blur-md">
      <div className="flex flex-col lg:flex-row items-center justify-between gap-4">
        {/* Playback Controls */}
        <div className="flex items-center space-x-2 w-full lg:w-auto justify-center lg:justify-start">
          <button
            onClick={onTogglePlay}
            className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-bold transition-all shadow-md cursor-pointer ${
              isPlaying
                ? 'bg-amber-500 hover:bg-amber-600 text-gray-950 shadow-amber-500/20'
                : 'bg-emerald-500 hover:bg-emerald-600 text-gray-950 shadow-emerald-500/20'
            }`}
          >
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 fill-current" />}
            <span>{isPlaying ? 'PAUSE STREAM' : 'RUN STREAM'}</span>
          </button>

          <button
            onClick={onStepForward}
            disabled={isPlaying}
            className="p-2 rounded-xl bg-gray-800 hover:bg-gray-700 text-gray-300 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
            title="Step Forward (+1 Tick)"
          >
            <SkipForward className="w-4 h-4" />
          </button>

          <button
            onClick={onReset}
            className="p-2 rounded-xl bg-gray-800 hover:bg-gray-700 text-gray-300 transition-all"
            title="Reset Simulation"
          >
            <RotateCcw className="w-4 h-4" />
          </button>

          {/* Speed Selectors */}
          <div className="flex items-center space-x-1 bg-gray-950/80 p-1 rounded-xl border border-gray-800 ml-2">
            {speeds.map((s) => (
              <button
                key={s}
                onClick={() => onChangeSpeed(s)}
                className={`px-2.5 py-1 rounded-lg text-xs font-bold font-mono transition-all ${
                  speed === s
                    ? 'bg-blue-600 text-white shadow-sm'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                {s}x
              </button>
            ))}
          </div>
        </div>

        {/* Timeline Scrubber */}
        <div className="flex items-center space-x-3 w-full lg:flex-1 max-w-md">
          <Clock className="w-4 h-4 text-gray-500 shrink-0" />
          <input
            type="range"
            min="0"
            max={maxSteps - 1}
            value={currentStep}
            onChange={(e) => onScrub(parseInt(e.target.value))}
            className="w-full h-1.5 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
          />
          <span className="font-mono text-xs text-gray-400 shrink-0">
            {currentStep} / {maxSteps}
          </span>
        </div>

        {/* Live Arriving Pod Telemetry */}
        <div className="flex items-center space-x-3 bg-gray-950/70 border border-gray-800/80 rounded-xl px-3.5 py-1.5 w-full lg:w-auto">
          <Box className="w-4 h-4 text-indigo-400 shrink-0" />
          <div className="text-xs">
            <span className="text-gray-400">Incoming Pod: </span>
            <span className="font-mono font-bold text-white">
              {currentPod?.cores_demanded || 8.5} Cores
            </span>
            <span className="text-gray-500 mx-1.5">•</span>
            <span className="font-mono text-indigo-300">
              {currentPod?.memory_gb || 16.0} GB RAM
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

