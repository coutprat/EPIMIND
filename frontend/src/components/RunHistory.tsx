import type { StoredReport } from '../lib/types';

interface RunHistoryProps {
  reports: StoredReport[];
  onSelectRun: (report: StoredReport) => void;
  selectedId?: string;
}

export const RunHistory: React.FC<RunHistoryProps> = ({
  reports,
  onSelectRun,
  selectedId,
}) => {
  const formatDate = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  const calculateStats = (report: StoredReport) => {
    const result = report.result;
    const alerts = result.alerts?.length || 0;
    const peak = result.summary?.peak_probability || 0;
    const fp = result.summary?.fp_estimate_per_hour || 0;
    return { alerts, peak, fp };
  };

  const compareRuns = (r1: StoredReport, r2: StoredReport) => {
    const s1 = calculateStats(r1);
    const s2 = calculateStats(r2);
    
    const alertDelta = s2.alerts - s1.alerts;
    const fpDelta = s2.fp - s1.fp;
    
    return { alertDelta, fpDelta };
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-lg font-bold text-gray-900 mb-4">
        📋 Run History ({reports.length})
      </h3>

      {reports.length === 0 ? (
        <p className="text-gray-500 text-sm">No analyses yet. Upload an EEG file to start.</p>
      ) : (
        <div className="space-y-3">
          {reports.map((report, idx) => {
            const stats = calculateStats(report);
            const isSelected = selectedId === report.id;
            const prevReport = idx > 0 ? reports[idx - 1] : null;
            const comparison = prevReport ? compareRuns(prevReport, report) : null;

            return (
              <div
                key={report.id}
                onClick={() => onSelectRun(report)}
                className={`p-4 rounded-lg border-2 cursor-pointer transition ${
                  isSelected
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 bg-gray-50 hover:border-gray-300'
                }`}
              >
                <div className="flex justify-between items-start mb-2">
                  <div>
                    <h4 className="font-semibold text-gray-900">
                      Run #{reports.length - idx}
                    </h4>
                    <p className="text-xs text-gray-600">
                      {formatDate(report.timestamp)}
                    </p>
                  </div>
                  <div className="text-right">
                    <div className={`text-sm font-bold ${
                      stats.alerts > 0 ? 'text-red-600' : 'text-green-600'
                    }`}>
                      {stats.alerts} alerts
                    </div>
                  </div>
                </div>

                {/* Quick Stats */}
                <div className="grid grid-cols-3 gap-2 mb-2 text-xs">
                  <div className="bg-white rounded p-2 border border-gray-200">
                    <div className="font-bold text-gray-900">{(stats.peak * 100).toFixed(0)}%</div>
                    <div className="text-gray-600">Peak Risk</div>
                  </div>
                  <div className="bg-white rounded p-2 border border-gray-200">
                    <div className="font-bold text-gray-900">{stats.fp.toFixed(1)}</div>
                    <div className="text-gray-600">FP/hr</div>
                  </div>
                  <div className="bg-white rounded p-2 border border-gray-200">
                    <div className="font-bold text-gray-900">{report.patientId}</div>
                    <div className="text-gray-600">Patient</div>
                  </div>
                </div>

                {/* Thresholds and Explainability */}
                <div className="mt-2 pt-2 border-t border-gray-200 text-xs space-y-1">
                  <div className="text-gray-700">
                    <strong>Detection:</strong> {report.detectionThreshold?.toFixed(2) ?? 'N/A'}
                    {report.explanationThreshold && (
                      <span className="ml-3"><strong>Explanation:</strong> {report.explanationThreshold.toFixed(2)}</span>
                    )}
                  </div>
                  {report.topChannels && report.topChannels.length > 0 && (
                    <div className="text-gray-600">
                      <strong>Top Channels:</strong> {report.topChannels.slice(0, 3).map(c => c.channel).join(', ')}
                    </div>
                  )}
                </div>

                {/* Comparison with previous run */}
                {comparison && (
                  <div className="mt-2 pt-2 border-t border-gray-200 text-xs">
                    <div className="text-gray-700">
                      vs previous run:
                      {comparison.alertDelta !== 0 && (
                        <span className={comparison.alertDelta > 0 ? 'text-red-600' : 'text-green-600'}>
                          {' '}alerts {comparison.alertDelta > 0 ? '+' : ''}{comparison.alertDelta}
                        </span>
                      )}
                      {comparison.fpDelta !== 0 && (
                        <span className={comparison.fpDelta > 0 ? 'text-red-600' : 'text-green-600'}>
                          {' '} FP/hr {comparison.fpDelta > 0 ? '+' : ''}{comparison.fpDelta.toFixed(1)}
                        </span>
                      )}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      <div className="mt-4 p-3 bg-blue-50 rounded text-xs text-gray-700 border border-blue-200">
        <strong>💡 Click any run</strong> to view detailed explanation and adjust threshold.
      </div>
    </div>
  );
};
