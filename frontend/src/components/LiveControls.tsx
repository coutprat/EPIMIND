/**
 * LiveControls: Control panel for LIVE streaming simulation.
 * Play/pause, speed control, and reset button.
 */

import React from 'react';

interface LiveControlsProps {
  isLive: boolean;
  onToggleLive: (enabled: boolean) => void;
  isPlaying: boolean;
  onPlayPause: () => void;
  speed: number; // 1, 2, 4
  onSpeedChange: (speed: number) => void;
  onReset: () => void;
  progress: number; // 0 to 1
}

export const LiveControls: React.FC<LiveControlsProps> = ({
  isLive,
  onToggleLive,
  isPlaying,
  onPlayPause,
  speed,
  onSpeedChange,
  onReset,
  progress,
}) => {
  return (
    <div className="bg-white rounded-lg shadow p-6 border-l-4 border-cyan-500">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-bold text-gray-900">
          📡 Live Streaming Simulation
        </h3>
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={isLive}
            onChange={(e) => onToggleLive(e.target.checked)}
            className="w-4 h-4 rounded"
          />
          <span className="text-sm font-semibold text-gray-700">
            {isLive ? 'LIVE ON' : 'LIVE OFF'}
          </span>
        </label>
      </div>

      {isLive && (
        <div className="space-y-4">
          {/* Progress bar */}
          <div>
            <div className="flex justify-between text-xs text-gray-600 mb-1">
              <span>Progress</span>
              <span>{(progress * 100).toFixed(0)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-cyan-500 h-2 rounded-full transition-all"
                style={{ width: `${progress * 100}%` }}
              />
            </div>
          </div>

          {/* Controls */}
          <div className="flex gap-3">
            <button
              onClick={onPlayPause}
              className={`flex-1 px-4 py-2 rounded-md font-semibold text-sm transition ${
                isPlaying
                  ? 'bg-cyan-600 text-white hover:bg-cyan-700'
                  : 'bg-gray-200 text-gray-900 hover:bg-gray-300'
              }`}
            >
              {isPlaying ? '⏸ Pause' : '▶ Play'}
            </button>
            <button
              onClick={onReset}
              className="px-4 py-2 bg-gray-200 text-gray-900 rounded-md font-semibold text-sm hover:bg-gray-300"
            >
              🔄 Reset
            </button>
          </div>

          {/* Speed selector */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">
              Speed
            </label>
            <div className="flex gap-2">
              {[1, 2, 4].map((s) => (
                <button
                  key={s}
                  onClick={() => onSpeedChange(s)}
                  className={`flex-1 px-3 py-2 rounded-md font-semibold text-sm transition ${
                    speed === s
                      ? 'bg-cyan-600 text-white'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  {s}x
                </button>
              ))}
            </div>
          </div>

          {/* Info text */}
          <div className="text-xs text-gray-500 bg-cyan-50 p-3 rounded border border-cyan-200">
            📊 Timeline updates in real-time. Alerts popup as they're detected. Simulates live EEG monitoring.
          </div>
        </div>
      )}

      {!isLive && (
        <div className="text-xs text-gray-500 bg-gray-50 p-3 rounded border border-gray-200">
          Toggle LIVE mode to stream probability data in real-time. Perfect for live demos!
        </div>
      )}
    </div>
  );
};
