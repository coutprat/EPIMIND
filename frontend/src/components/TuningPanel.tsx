/**
 * TuningPanel: Interactive threshold + consecutive windows tuning.
 * Real-time metrics computation showing safety vs noise tradeoff.
 */

import type { Metrics, SummaryMetrics } from '../lib/types';

interface TuningPanelProps {
  threshold: number;
  onThresholdChange: (threshold: number) => void;
  consecutiveWindows: number;
  onConsecutiveChange: (windows: number) => void;
  metrics: Metrics | SummaryMetrics;
  baselineMetrics?: Metrics | null;
  showComparison?: boolean;
}

export const TuningPanel: React.FC<TuningPanelProps> = ({
  threshold,
  onThresholdChange,
  consecutiveWindows,
  onConsecutiveChange,
  metrics,
  baselineMetrics,
  showComparison = false,
}) => {
  const getDelta = (current: number, baseline?: number) => {
    if (baseline === undefined) return null;
    const delta = current - baseline;
    return {
      value: delta,
      symbol: delta > 0 ? '↑' : delta < 0 ? '↓' : '→',
      color: delta > 0 ? 'text-red-600' : delta < 0 ? 'text-green-600' : 'text-gray-600',
    };
  };

  const alertsDelta = getDelta(metrics.alerts_count, baselineMetrics?.alerts_count);
  const fpDelta = getDelta(metrics.fp_estimate_per_hour, baselineMetrics?.fp_estimate_per_hour);

  return (
    <div className="bg-white rounded-lg shadow p-6 border-l-4 border-purple-500">
      <h3 className="text-lg font-bold text-gray-900 mb-6">
        ⚙️ Safety vs Noise Tuning
      </h3>

      <div className="space-y-6">
        {/* Threshold Slider */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="text-sm font-semibold text-gray-700">
              Detection Threshold
            </label>
            <span className="text-lg font-bold text-purple-600">
              {threshold.toFixed(2)}
            </span>
          </div>
          <input
            type="range"
            min="0.05"
            max="0.95"
            step="0.05"
            value={threshold}
            onChange={(e) => onThresholdChange(parseFloat(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-purple-500"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>More Alerts (0.05)</span>
            <span>Fewer Alerts (0.95)</span>
          </div>
          <p className="text-xs text-gray-600 mt-2">
            💡 Lower = catch more events (sensitive). Higher = fewer false positives.
          </p>
        </div>

        {/* Consecutive Windows Slider */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="text-sm font-semibold text-gray-700">
              Consecutive Windows
            </label>
            <span className="text-lg font-bold text-purple-600">
              {consecutiveWindows}
            </span>
          </div>
          <input
            type="range"
            min="1"
            max="10"
            step="1"
            value={consecutiveWindows}
            onChange={(e) => onConsecutiveChange(parseInt(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-purple-500"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>Instant (1)</span>
            <span>Sustained (10)</span>
          </div>
          <p className="text-xs text-gray-600 mt-2">
            💡 Lower = instant detection. Higher = only sustained events.
          </p>
        </div>

        {/* Metrics Grid */}
        <div className="bg-purple-50 rounded-lg p-4 border border-purple-200">
          <h4 className="text-sm font-semibold text-purple-900 mb-4">
            📊 Live Metrics
          </h4>
          <div className="grid grid-cols-2 gap-4">
            {/* Alerts Count */}
            <div className="bg-white rounded p-3 border border-purple-200">
              <div className="text-2xl font-bold text-red-600">
                {metrics.alerts_count}
              </div>
              <div className="text-xs text-gray-600">Alerts</div>
              {showComparison && alertsDelta && (
                <div className={`text-xs font-semibold mt-1 ${alertsDelta.color}`}>
                  {alertsDelta.symbol} {Math.abs(alertsDelta.value)}
                </div>
              )}
            </div>

            {/* FP/hour */}
            <div className="bg-white rounded p-3 border border-purple-200">
              <div className="text-2xl font-bold text-yellow-600">
                {metrics.fp_estimate_per_hour.toFixed(1)}
              </div>
              <div className="text-xs text-gray-600">FP/hour</div>
              {showComparison && fpDelta && (
                <div className={`text-xs font-semibold mt-1 ${fpDelta.color}`}>
                  {fpDelta.symbol} {Math.abs(fpDelta.value).toFixed(1)}
                </div>
              )}
            </div>

            {/* Peak Probability */}
            <div className="bg-white rounded p-3 border border-purple-200">
              <div className="text-2xl font-bold text-orange-600">
                {(metrics.peak_probability * 100).toFixed(0)}%
              </div>
              <div className="text-xs text-gray-600">Peak Risk</div>
            </div>

            {/* Mean Probability */}
            <div className="bg-white rounded p-3 border border-purple-200">
              <div className="text-2xl font-bold text-blue-600">
                {(metrics.mean_probability * 100).toFixed(1)}%
              </div>
              <div className="text-xs text-gray-600">Mean Risk</div>
            </div>
          </div>
        </div>

        {/* Baseline Comparison (if available) */}
        {showComparison && baselineMetrics && (
          <div className="bg-gray-50 rounded-lg p-4 border border-gray-300">
            <h4 className="text-sm font-semibold text-gray-900 mb-3">
              📈 vs Baseline (Initial Settings)
            </h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-700">Alerts: {baselineMetrics.alerts_count}</span>
                <span className={alertsDelta ? alertsDelta.color : ''}>
                  {alertsDelta?.symbol} {Math.abs(alertsDelta?.value || 0)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-700">FP/hour: {baselineMetrics.fp_estimate_per_hour.toFixed(1)}</span>
                <span className={fpDelta ? fpDelta.color : ''}>
                  {fpDelta?.symbol} {Math.abs(fpDelta?.value || 0).toFixed(1)}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
