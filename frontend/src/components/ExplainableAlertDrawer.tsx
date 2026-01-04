/**
 * ExplainableAlertDrawer: Right-side panel explaining why an alert was detected.
 * Shows timing, probabilities, threshold logic, confidence, and zoom button.
 */

import React from 'react';
import type { AlertEvent, TimelinePoint } from '../lib/types';
import { getAlertConfidence, getConfidenceExplanation, getConfidenceColor } from '../lib/confidence';

interface ExplainableAlertDrawerProps {
  alert: AlertEvent | null;
  threshold: number;
  consecutiveWindows: number;
  strideSec: number;
  onClose: () => void;
  onZoomToAlert: (startSec: number, endSec: number) => void;
  timeline?: TimelinePoint[];
}

export const ExplainableAlertDrawer: React.FC<ExplainableAlertDrawerProps> = ({
  alert,
  threshold,
  consecutiveWindows,
  strideSec,
  onClose,
  onZoomToAlert,
  timeline,
}) => {
  if (!alert) {
    return null;
  }

  const startMin = Math.floor(alert.start_sec / 60);
  const startSec = Math.floor(alert.start_sec % 60);
  const endMin = Math.floor(alert.end_sec / 60);
  const endSec = Math.floor(alert.end_sec % 60);

  // Compute confidence if timeline available
  const confidence = timeline ? getAlertConfidence(alert, timeline, strideSec) : null;

  return (
    <div className="fixed inset-0 z-50 overflow-hidden">
      {/* Overlay */}
      <div
        className="absolute inset-0 bg-black bg-opacity-50 transition-opacity"
        onClick={onClose}
      />

      {/* Drawer */}
      <div className="absolute right-0 top-0 bottom-0 w-full max-w-md bg-white shadow-2xl overflow-y-auto">
        <div className="p-6 space-y-6">
          {/* Header */}
          <div className="flex items-center justify-between">
            <h2 className="text-2xl font-bold text-gray-900">
              🎯 Alert Explanation
            </h2>
            <button
              onClick={onClose}
              className="text-gray-500 hover:text-gray-700 text-2xl"
            >
              ✕
            </button>
          </div>

          {/* Alert Timing */}
          <div className="bg-red-50 rounded-lg p-4 border border-red-200">
            <h3 className="font-semibold text-red-900 mb-3">⏱️ Timing</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-700">Start:</span>
                <span className="font-mono font-semibold text-gray-900">
                  {startMin}:{String(startSec).padStart(2, '0')}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-700">End:</span>
                <span className="font-mono font-semibold text-gray-900">
                  {endMin}:{String(endSec).padStart(2, '0')}
                </span>
              </div>
              <div className="flex justify-between border-t border-red-300 pt-2 mt-2">
                <span className="text-gray-700 font-semibold">Duration:</span>
                <span className="font-mono font-bold text-red-600">
                  {alert.duration_sec.toFixed(1)}s
                </span>
              </div>
            </div>
          </div>

          {/* Probabilities */}
          <div className="bg-orange-50 rounded-lg p-4 border border-orange-200">
            <h3 className="font-semibold text-orange-900 mb-3">📊 Probability</h3>
            <div className="space-y-3">
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm text-gray-700">Peak</span>
                  <span className="text-sm font-bold text-orange-600">
                    {(alert.peak_prob * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-orange-600 h-2 rounded-full"
                    style={{ width: `${alert.peak_prob * 100}%` }}
                  />
                </div>
              </div>

              {alert.mean_prob !== undefined && (
                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm text-gray-700">Mean</span>
                    <span className="text-sm font-semibold text-gray-900">
                      {(alert.mean_prob * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-yellow-500 h-2 rounded-full"
                      style={{ width: `${alert.mean_prob * 100}%` }}
                    />
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Detection Rule */}
          <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
            <h3 className="font-semibold text-blue-900 mb-3">🔍 Detection Rule</h3>
            <div className="space-y-2 text-sm text-blue-900">
              <p>
                Alert triggered when probability{' '}
                <span className="font-bold">≥ {threshold.toFixed(2)}</span>
              </p>
              <p>
                for <span className="font-bold">{consecutiveWindows}</span> consecutive{' '}
                {consecutiveWindows === 1 ? 'window' : 'windows'} ({' '}
                {(consecutiveWindows * strideSec).toFixed(1)}s
                )
              </p>
            </div>
          </div>

          {/* Confidence Badge */}
          {confidence && (
            <div className={`rounded-lg p-4 border-2 ${getConfidenceColor(confidence.level)}`}>
              <h3 className="font-semibold mb-2">📈 Event Confidence</h3>
              <div
                className="text-sm whitespace-pre-wrap"
                style={{
                  fontFamily: 'system-ui, -apple-system, sans-serif',
                  lineHeight: '1.5',
                }}
              >
                {getConfidenceExplanation(confidence)}
              </div>
            </div>
          )}

          {/* Human Explanation */}
          <div className="bg-green-50 rounded-lg p-4 border border-green-200">
            <h3 className="font-semibold text-green-900 mb-3">💬 Explanation</h3>
            <p className="text-sm text-green-900">
              This alert triggered because risk exceeded threshold{' '}
              <span className="font-bold">{threshold.toFixed(2)}</span> for{' '}
              <span className="font-bold">{consecutiveWindows}</span> consecutive
              windows between <span className="font-mono">{startMin}:{String(startSec).padStart(2, '0')}</span> –{' '}
              <span className="font-mono">{endMin}:{String(endSec).padStart(2, '0')}</span>. Peak probability was{' '}
              <span className="font-bold text-red-600">{(alert.peak_prob * 100).toFixed(1)}%</span>.
            </p>
          </div>

          {/* Zoom Button */}
          <button
            onClick={() => {
              const margin = 30;
              onZoomToAlert(
                Math.max(0, alert.start_sec - margin),
                alert.end_sec + margin
              );
              onClose();
            }}
            className="w-full px-4 py-3 bg-blue-600 text-white rounded-md font-semibold hover:bg-blue-700 transition"
          >
            🔎 Zoom to Alert
          </button>
        </div>
      </div>
    </div>
  );
};
