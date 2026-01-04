import React from 'react';
import type { AlertEvent, TimelinePoint } from '../lib/types';
import { getAlertConfidence, getConfidenceColor } from '../lib/confidence';

interface AlertsTableProps {
  alerts: AlertEvent[];
  title?: string;
  onAlertClick?: (alert: AlertEvent) => void;
  timeline?: TimelinePoint[];
  strideSec?: number;
}

export const AlertsTable: React.FC<AlertsTableProps> = ({
  alerts,
  title = 'Detected Seizure Events',
  onAlertClick,
  timeline,
  strideSec = 1,
}) => {
  if (!alerts || alerts.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">{title}</h2>
        <div className="text-center py-8">
          <p className="text-gray-500">No seizure events detected</p>
          <p className="text-xs text-gray-400 mt-2">
            Try adjusting the threshold or smoothing parameters
          </p>
        </div>
      </div>
    );
  }

  const formatTime = (seconds: number) => {
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${minutes}:${String(secs).padStart(2, '0')}`;
  };

  return (
    <div className="bg-white rounded-lg shadow overflow-hidden">
      <div className="p-6 border-b border-gray-200">
        <h2 className="text-2xl font-bold text-gray-900">{title}</h2>
        <p className="text-sm text-gray-600 mt-1">
          {alerts.length} event{alerts.length !== 1 ? 's' : ''} detected
        </p>
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider">
                Start Time
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider">
                End Time
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider">
                Duration
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider">
                Peak Probability
              </th>
              {timeline && (
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider">
                  Confidence
                </th>
              )}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {alerts.map((alert, idx) => {
              const confidence = timeline ? getAlertConfidence(alert, timeline, strideSec) : null;
              return (
                <tr
                  key={idx}
                  onClick={() => onAlertClick?.(alert)}
                  className={`hover:bg-blue-50 transition ${
                    onAlertClick ? 'cursor-pointer' : ''
                  }`}
                >
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-semibold text-gray-900">
                    {formatTime(alert.start_sec)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-semibold text-gray-900">
                    {formatTime(alert.end_sec)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">
                    {alert.duration_sec.toFixed(1)}s
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center gap-2">
                      <div className="w-16 bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-red-600 h-2 rounded-full"
                          style={{ width: `${alert.peak_prob * 100}%` }}
                        ></div>
                      </div>
                      <span className="text-sm font-semibold text-gray-900">
                        {(alert.peak_prob * 100).toFixed(1)}%
                      </span>
                    </div>
                  </td>
                  {confidence && (
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span
                        className={`inline-block px-3 py-1 rounded-full text-xs font-semibold border ${getConfidenceColor(
                          confidence.level
                        )}`}
                      >
                        {confidence.level} ({confidence.score})
                      </span>
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Alert summary */}
      {alerts.length > 0 && (
        <div className="bg-red-50 border-t border-red-200 px-6 py-3">
          <p className="text-sm text-red-700">
            <strong>Total seizure time:</strong>{' '}
            {alerts.reduce((sum, a) => sum + a.duration_sec, 0).toFixed(1)}s
          </p>
        </div>
      )}
    </div>
  );
};
