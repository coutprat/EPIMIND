import type { AnalysisResponse } from '../lib/types';

interface ThresholdPlaygroundProps {
  result: AnalysisResponse;
  currentThreshold: number;
  onThresholdChange: (threshold: number) => void;
}

export const ThresholdPlayground: React.FC<ThresholdPlaygroundProps> = ({
  result,
  currentThreshold,
  onThresholdChange,
}) => {
  // Compute alerts for a given threshold
  const computeAlertsForThreshold = (threshold: number) => {
    const timeline = result.timeline || [];
    const alerts = [];
    let inAlert = false;
    let alertStart = 0;

    for (let i = 0; i < timeline.length; i++) {
      const point = timeline[i];
      if ((point.prob || 0) > threshold && !inAlert) {
        inAlert = true;
        alertStart = i;
      } else if ((point.prob || 0) <= threshold && inAlert) {
        inAlert = false;
        alerts.push({
          start: alertStart,
          end: i,
          duration: i - alertStart,
          peak: Math.max(...timeline.slice(alertStart, i).map(p => p.prob || 0)),
        });
      }
    }
    if (inAlert) {
      alerts.push({
        start: alertStart,
        end: timeline.length,
        duration: timeline.length - alertStart,
        peak: Math.max(...timeline.slice(alertStart).map(p => p.prob || 0)),
      });
    }
    return alerts;
  };

  const alerts = computeAlertsForThreshold(currentThreshold);
  const summary = result.summary;
  const duration = result.duration_sec || 0;
  const fpPerHour = (alerts.length / Math.max(duration / 3600, 1)) * 2; // Estimate

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-lg font-bold text-gray-900 mb-4">
        🎚️ Alert Generation Threshold
      </h3>

      {/* Slider */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Detection Threshold: <span className="text-blue-600 font-bold">{currentThreshold.toFixed(2)}</span>
        </label>
        <p className="text-xs text-gray-500 mb-3">
          Adjust the risk level at which alerts are generated. This threshold is independent of explanations.
        </p>
        <input
          type="range"
          min="0.1"
          max="0.9"
          step="0.05"
          value={currentThreshold}
          onChange={(e) => onThresholdChange(parseFloat(e.target.value))}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-500"
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>Very Low (0.1)</span>
          <span>Current</span>
          <span>Very High (0.9)</span>
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-2 gap-4">
        {/* Alerts Count */}
        <div className="bg-red-50 rounded p-3 border border-red-200">
          <div className="text-2xl font-bold text-red-600">{alerts.length}</div>
          <div className="text-xs text-gray-600">Alerts at {currentThreshold.toFixed(2)}</div>
        </div>

        {/* False Positive Rate */}
        <div className="bg-yellow-50 rounded p-3 border border-yellow-200">
          <div className="text-2xl font-bold text-yellow-600">{fpPerHour.toFixed(1)}</div>
          <div className="text-xs text-gray-600">Est. FP/hour</div>
        </div>

        {/* Peak Risk */}
        <div className="bg-purple-50 rounded p-3 border border-purple-200">
          <div className="text-2xl font-bold text-purple-600">
            {((summary?.peak_probability || 0) * 100).toFixed(0)}%
          </div>
          <div className="text-xs text-gray-600">Peak Risk (Fixed)</div>
        </div>

        {/* Mean Risk */}
        <div className="bg-blue-50 rounded p-3 border border-blue-200">
          <div className="text-2xl font-bold text-blue-600">
            {((summary?.mean_probability || 0) * 100).toFixed(0)}%
          </div>
          <div className="text-xs text-gray-600">Mean Risk</div>
        </div>
      </div>

      {/* Alert Details */}
      {alerts.length > 0 && (
        <div className="mt-4 p-3 bg-red-50 rounded border border-red-200">
          <h4 className="font-semibold text-red-700 text-sm mb-2">Alert Details:</h4>
          <ul className="text-xs text-gray-700 space-y-1">
            {alerts.map((alert, i) => (
              <li key={i}>
                • Window {alert.start}-{alert.end} ({alert.duration} windows, peak {(alert.peak * 100).toFixed(0)}%)
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Recommendation */}
      <div className="mt-4 p-3 bg-blue-50 rounded border border-blue-200 text-xs text-gray-700">
        <strong>💡 Tip:</strong> Slide right to reduce alerts, left to catch more cases.
      </div>
    </div>
  );
};
