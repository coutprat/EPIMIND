import React from 'react';
import { getStabilityLabel, getStabilityColor, getStabilityWarning } from '../lib/volatility';
import type { VolatilityAnalysis } from '../lib/volatility';

interface StabilityCardProps {
  analysis: VolatilityAnalysis;
}

export const StabilityCard: React.FC<StabilityCardProps> = ({ analysis }) => {
  const label = getStabilityLabel(analysis.stability_score);
  const colorClass = getStabilityColor(analysis.stability_score);
  const warning = getStabilityWarning(analysis);

  // Emoji based on stability
  const emoji = analysis.stability_score >= 75 ? '✅' : analysis.stability_score >= 50 ? '⚠️' : '🚨';

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Signal Stability Meter</h3>

      {/* Stability Score */}
      <div className={`rounded-lg border-2 p-4 mb-4 ${colorClass}`}>
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium">Stability Status</p>
            <p className="text-2xl font-bold mt-1">{label}</p>
          </div>
          <div className="text-right">
            <p className="text-4xl">{emoji}</p>
            <p className="text-sm font-semibold mt-1">{analysis.stability_score}/100</p>
          </div>
        </div>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="bg-gray-50 rounded p-3">
          <p className="text-xs text-gray-600 font-medium">Avg Volatility</p>
          <p className="text-lg font-bold text-gray-900 mt-1">
            {(analysis.volatility_mean * 100).toFixed(1)}%
          </p>
        </div>
        <div className="bg-gray-50 rounded p-3">
          <p className="text-xs text-gray-600 font-medium">Peak Volatility</p>
          <p className="text-lg font-bold text-gray-900 mt-1">
            {(analysis.volatility_max * 100).toFixed(1)}%
          </p>
        </div>
      </div>

      {/* Warning message if noisy */}
      {warning && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-800">
          <p className="font-semibold mb-1">⚠️ High Fluctuations Detected</p>
          <p>{warning}</p>
        </div>
      )}

      {/* Info if stable */}
      {!warning && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-3 text-sm text-green-800">
          <p>✅ Signal quality is good. Current settings are reliable.</p>
        </div>
      )}
    </div>
  );
};
