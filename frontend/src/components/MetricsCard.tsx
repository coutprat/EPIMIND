import type { SummaryMetrics } from '../lib/types';

interface MetricsCardProps {
  metrics: SummaryMetrics | null;
}

export const MetricsCard: React.FC<MetricsCardProps> = ({ metrics }) => {
  if (!metrics) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Summary Metrics</h2>
        <p className="text-gray-500">No analysis data available</p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Summary Metrics</h2>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Alerts Count */}
        <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-4 border border-blue-200">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-xs font-semibold text-blue-700 uppercase tracking-wider">
                Seizure Events
              </p>
              <p className="text-3xl font-bold text-blue-900 mt-2">
                {metrics.alerts_count}
              </p>
            </div>
            <div className="text-4xl text-blue-300">⚠️</div>
          </div>
          <p className="text-xs text-blue-600 mt-2">
            {metrics.alerts_count === 0
              ? 'No seizures detected'
              : `${metrics.alerts_count} event${metrics.alerts_count !== 1 ? 's' : ''} detected`}
          </p>
        </div>

        {/* Peak Probability */}
        <div className="bg-gradient-to-br from-red-50 to-red-100 rounded-lg p-4 border border-red-200">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-xs font-semibold text-red-700 uppercase tracking-wider">
                Peak Probability
              </p>
              <p className="text-3xl font-bold text-red-900 mt-2">
                {(metrics.peak_probability * 100).toFixed(1)}%
              </p>
            </div>
            <div className="text-4xl text-red-300">📊</div>
          </div>
          <p className="text-xs text-red-600 mt-2">
            Maximum confidence level
          </p>
        </div>

        {/* Mean Probability */}
        <div className="bg-gradient-to-br from-yellow-50 to-yellow-100 rounded-lg p-4 border border-yellow-200">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-xs font-semibold text-yellow-700 uppercase tracking-wider">
                Mean Probability
              </p>
              <p className="text-3xl font-bold text-yellow-900 mt-2">
                {(metrics.mean_probability * 100).toFixed(1)}%
              </p>
            </div>
            <div className="text-4xl text-yellow-300">📈</div>
          </div>
          <p className="text-xs text-yellow-600 mt-2">
            Average across recording
          </p>
        </div>

        {/* FP per hour */}
        <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg p-4 border border-purple-200">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-xs font-semibold text-purple-700 uppercase tracking-wider">
                Est. FP / Hour
              </p>
              <p className="text-3xl font-bold text-purple-900 mt-2">
                {metrics.fp_estimate_per_hour.toFixed(2)}
              </p>
            </div>
            <div className="text-4xl text-purple-300">🔔</div>
          </div>
          <p className="text-xs text-purple-600 mt-2">
            Estimated false positive rate
          </p>
        </div>
      </div>

      {/* Quality indicators */}
      <div className="mt-6 pt-6 border-t border-gray-200">
        <h3 className="text-sm font-semibold text-gray-700 mb-3">Quality Assessment</h3>
        <div className="space-y-2">
          {/* Alert rate indicator */}
          <div className="flex items-center justify-between">
            <span className="text-xs text-gray-600">Alert Detection Rate</span>
            <div className="flex items-center gap-2">
              <div className="w-32 bg-gray-200 rounded-full h-2">
                <div
                  className={`h-2 rounded-full transition-colors ${
                    metrics.alerts_count > 0
                      ? 'bg-red-500'
                      : 'bg-green-500'
                  }`}
                  style={{
                    width: `${Math.min(
                      (metrics.alerts_count / 5) * 100,
                      100
                    )}%`,
                  }}
                ></div>
              </div>
              <span className="text-xs font-semibold text-gray-700">
                {metrics.alerts_count > 0 ? 'HIGH' : 'LOW'}
              </span>
            </div>
          </div>

          {/* Confidence indicator */}
          <div className="flex items-center justify-between">
            <span className="text-xs text-gray-600">Peak Confidence</span>
            <div className="flex items-center gap-2">
              <div className="w-32 bg-gray-200 rounded-full h-2">
                <div
                  className="bg-blue-500 h-2 rounded-full"
                  style={{
                    width: `${metrics.peak_probability * 100}%`,
                  }}
                ></div>
              </div>
              <span className="text-xs font-semibold text-gray-700">
                {metrics.peak_probability > 0.8
                  ? 'HIGH'
                  : metrics.peak_probability > 0.5
                    ? 'MEDIUM'
                    : 'LOW'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
