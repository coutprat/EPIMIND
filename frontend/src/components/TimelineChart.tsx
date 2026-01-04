import React, { useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import type { TimelinePoint, AlertEvent } from '../lib/types';

interface TimelineChartProps {
  data: TimelinePoint[];
  threshold: number;
  title?: string;
  alerts?: AlertEvent[];
  zoomRange?: [number, number]; // [startSec, endSec]
  onReset?: () => void;
  onZoomChange?: (startSec: number, endSec: number) => void;
}

export const TimelineChart: React.FC<TimelineChartProps> = ({
  data,
  threshold,
  title = 'Seizure Probability Timeline',
  alerts = [],
  zoomRange,
  onReset,
  onZoomChange,
}) => {
  const maxData = data?.length > 0 ? data[data.length - 1]?.t_sec || 100 : 100;
  const [zoomStart, setZoomStart] = useState(0);
  const [zoomEnd, setZoomEnd] = useState(maxData);

  if (!data || data.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow p-6 flex items-center justify-center h-80">
        <p className="text-gray-500">No data available. Upload an EDF file to analyze.</p>
      </div>
    );
  }

  const handleZoomApply = () => {
    if (onZoomChange) {
      onZoomChange(zoomStart, zoomEnd);
    }
  };
  const displayData = zoomRange
    ? data.filter((point) => point.t_sec >= zoomRange[0] && point.t_sec <= zoomRange[1])
    : data;

  // Format data for Recharts
  const chartData = displayData.map((point) => ({
    t_sec: point.t_sec,
    prob: Math.round(point.prob * 100) / 100, // Round to 2 decimals
    displayTime: `${Math.floor(point.t_sec / 60)}:${String(Math.floor(point.t_sec % 60)).padStart(2, '0')}`,
  }));

  const minProb = Math.min(...displayData.map((p) => p.prob), threshold) - 0.05;
  const maxProb = Math.max(...displayData.map((p) => p.prob), threshold) + 0.05;

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-bold text-gray-900">{title}</h2>
        {zoomRange && onReset && (
          <button
            onClick={onReset}
            className="px-3 py-1 text-sm bg-blue-100 text-blue-700 rounded hover:bg-blue-200"
          >
            🔄 Reset Zoom
          </button>
        )}
      </div>

      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={chartData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis
            dataKey="displayTime"
            stroke="#6b7280"
            style={{ fontSize: '12px' }}
            tick={{ fill: '#6b7280' }}
          />
          <YAxis
            stroke="#6b7280"
            domain={[Math.max(0, minProb), Math.min(1, maxProb)]}
            style={{ fontSize: '12px' }}
            tick={{ fill: '#6b7280' }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#ffffff',
              border: '1px solid #e5e7eb',
              borderRadius: '0.5rem',
            }}
            formatter={(value: number) => [value.toFixed(3), 'Probability']}
            labelFormatter={(label) => `Time: ${label}`}
          />
          <Legend wrapperStyle={{ fontSize: '12px' }} />

          {/* Alert regions as reference lines */}
          {alerts.map((alert, idx) => (
            <ReferenceLine
              key={`alert-${idx}`}
              x={alert.start_sec}
              stroke="#ef4444"
              strokeOpacity={0.3}
              strokeWidth={3}
              label={{
                value: '🚨',
                position: 'top',
                fill: '#dc2626',
                fontSize: 14,
              }}
            />
          ))}
          <ReferenceLine
            y={threshold}
            stroke="#dc2626"
            strokeDasharray="5 5"
            label={{
              value: `Threshold (${threshold.toFixed(2)})`,
              position: 'insideTopRight',
              offset: -5,
              fill: '#dc2626',
              fontSize: 12,
            }}
          />

          {/* Probability line */}
          <Line
            type="monotone"
            dataKey="prob"
            stroke="#3b82f6"
            strokeWidth={2}
            dot={false}
            name="Seizure Probability"
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>

      {/* Mini-map and Zoom Controls */}
      <div className="mt-6 bg-gray-50 rounded-lg p-4 border border-gray-200">
        <h3 className="text-sm font-semibold text-gray-900 mb-3">🔍 Zoom Range Slider</h3>

        {/* Mini-map (compressed view) */}
        <div className="mb-4">
          <ResponsiveContainer width="100%" height={60}>
            <LineChart data={data.map((p) => ({ t_sec: p.t_sec, prob: p.prob }))}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="t_sec" type="number" stroke="#d1d5db" style={{ fontSize: '10px' }} />
              <YAxis stroke="#d1d5db" hide />

              {/* Mini alert regions */}
              {alerts.map((alert, idx) => (
                <ReferenceLine
                  key={`minimap-alert-${idx}`}
                  x={alert.start_sec}
                  stroke="#ef4444"
                  strokeOpacity={0.5}
                  strokeWidth={2}
                />
              ))}

              <Line
                type="monotone"
                dataKey="prob"
                stroke="#6b7280"
                strokeWidth={1}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
          <p className="text-xs text-gray-500 mt-1">Full timeline overview</p>
        </div>

        {/* Range inputs */}
        <div className="space-y-3">
          <div>
            <label className="text-sm font-medium text-gray-700 block mb-1">
              Start: {zoomStart.toFixed(1)}s
            </label>
            <input
              type="range"
              min="0"
              max={maxData}
              step="1"
              value={zoomStart}
              onChange={(e) => setZoomStart(parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
            />
          </div>
          <div>
            <label className="text-sm font-medium text-gray-700 block mb-1">
              End: {zoomEnd.toFixed(1)}s
            </label>
            <input
              type="range"
              min="0"
              max={maxData}
              step="1"
              value={zoomEnd}
              onChange={(e) => setZoomEnd(parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
            />
          </div>
        </div>

        {/* Apply button */}
        <button
          onClick={handleZoomApply}
          className="mt-3 w-full px-3 py-2 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 text-sm transition"
        >
          Apply Zoom
        </button>
      </div>
      <div className="mt-4 grid grid-cols-3 gap-4">
        <div className="bg-gray-50 p-3 rounded">
          <p className="text-xs text-gray-600">Duration</p>
          <p className="text-lg font-semibold text-gray-900">
            {Math.floor(data[data.length - 1].t_sec / 60)}:{String(Math.floor(data[data.length - 1].t_sec % 60)).padStart(2, '0')} min
          </p>
        </div>
        <div className="bg-gray-50 p-3 rounded">
          <p className="text-xs text-gray-600">Max Probability</p>
          <p className="text-lg font-semibold text-gray-900">
            {(Math.max(...data.map((p) => p.prob)) * 100).toFixed(1)}%
          </p>
        </div>
        <div className="bg-gray-50 p-3 rounded">
          <p className="text-xs text-gray-600">Mean Probability</p>
          <p className="text-lg font-semibold text-gray-900">
            {(
              (data.reduce((sum, p) => sum + p.prob, 0) / data.length) *
              100
            ).toFixed(1)}%
          </p>
        </div>
      </div>
    </div>
  );
};
