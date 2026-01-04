import React, { useState } from 'react';
import {
  generateReport,
  reportAsText,
  reportAsMarkdown,
  type ReportData,
} from '../lib/reportBuilder';
import { copyToClipboard, downloadFile } from '../lib/exporters';

interface AutoReportCardProps {
  patientId: string;
  testId: string;
  timestamp: string;
  duration_hours: number;
  total_alerts: number;
  strongest_alert: { start_sec: number; peak_prob: number } | null;
  threshold: number;
  consecutive_windows: number;
  fp_estimate_per_hour: number | null;
  stability_score?: number;
}

export const AutoReportCard: React.FC<AutoReportCardProps> = ({
  patientId,
  testId,
  timestamp,
  duration_hours,
  total_alerts,
  strongest_alert,
  threshold,
  consecutive_windows,
  fp_estimate_per_hour,
  stability_score,
}) => {
  const [copied, setCopied] = useState(false);

  const reportData: ReportData = {
    patientId,
    testId,
    timestamp,
    duration_hours,
    total_alerts,
    strongest_alert: strongest_alert
      ? {
          start_sec: strongest_alert.start_sec,
          end_sec: strongest_alert.start_sec + 1, // Placeholder
          peak_prob: strongest_alert.peak_prob,
          duration_sec: 1, // Placeholder
          mean_prob: strongest_alert.peak_prob,
        }
      : null,
    threshold,
    consecutive_windows,
    fp_estimate_per_hour,
    stability_score,
  };

  const report = generateReport(reportData);
  const textVersion = reportAsText(report);
  const markdownVersion = reportAsMarkdown(report);

  const handleCopyText = async () => {
    await copyToClipboard(textVersion);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadText = () => {
    const filename = `EpiMind_Report_${patientId}_${new Date().toISOString().split('T')[0]}.txt`;
    downloadFile(textVersion, filename, 'text/plain');
  };

  const handleDownloadMarkdown = () => {
    const filename = `EpiMind_Report_${patientId}_${new Date().toISOString().split('T')[0]}.md`;
    downloadFile(markdownVersion, filename, 'text/markdown');
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">📄 Auto Report</h3>

      {/* Report Preview */}
      <div className="bg-gray-50 rounded-lg p-4 mb-4 border border-gray-200 max-h-60 overflow-y-auto">
        <h4 className="font-bold text-gray-900 mb-2">{report.title}</h4>
        <p className="text-sm text-gray-700 mb-3">{report.summary}</p>

        <h5 className="font-semibold text-gray-800 text-sm mb-2">Metrics:</h5>
        <ul className="space-y-1 mb-3 text-sm text-gray-700">
          {report.metrics_bullets.map((bullet, idx) => (
            <li key={idx}>• {bullet}</li>
          ))}
        </ul>

        <h5 className="font-semibold text-gray-800 text-sm mb-2">Interpretation:</h5>
        <p className="text-sm text-gray-700">{report.interpretation}</p>
      </div>

      {/* Action Buttons */}
      <div className="space-y-2">
        <button
          onClick={handleCopyText}
          className="w-full px-4 py-2 bg-blue-50 text-blue-700 rounded-lg border border-blue-200 hover:bg-blue-100 font-medium text-sm transition"
        >
          {copied ? '✓ Copied!' : '📋 Copy to Clipboard'}
        </button>

        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={handleDownloadText}
            className="px-3 py-2 bg-green-50 text-green-700 rounded-lg border border-green-200 hover:bg-green-100 font-medium text-sm transition"
          >
            📥 Download .txt
          </button>
          <button
            onClick={handleDownloadMarkdown}
            className="px-3 py-2 bg-purple-50 text-purple-700 rounded-lg border border-purple-200 hover:bg-purple-100 font-medium text-sm transition"
          >
            📥 Download .md
          </button>
        </div>
      </div>
    </div>
  );
};
