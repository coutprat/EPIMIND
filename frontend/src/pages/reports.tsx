import { useEffect, useState } from 'react';
import type { StoredReport } from '../lib/types';

export default function Reports() {
  const [reports, setReports] = useState<StoredReport[]>([]);
  const [selectedReport, setSelectedReport] = useState<StoredReport | null>(null);

  useEffect(() => {
    const stored = localStorage.getItem('seizure_reports');
    if (stored) {
      setReports(JSON.parse(stored));
    }
  }, []);

  const handleDelete = (id: string) => {
    const updated = reports.filter((r) => r.id !== id);
    setReports(updated);
    localStorage.setItem('seizure_reports', JSON.stringify(updated));
    if (selectedReport?.id === id) {
      setSelectedReport(null);
    }
  };

  const formatDate = (iso: string) => {
    const date = new Date(iso);
    return date.toLocaleString();
  };

  const formatDuration = (seconds: number) => {
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${minutes}:${String(secs).padStart(2, '0')}`;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      {/* Header */}
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center gap-4">
            <a
              href="/"
              className="text-gray-600 hover:text-gray-900 font-semibold"
            >
              ← Back to Dashboard
            </a>
            <h1 className="text-3xl font-bold text-gray-900">Report History</h1>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {reports.length === 0 ? (
          <div className="bg-white rounded-lg shadow p-12 text-center">
            <p className="text-2xl mb-2">📋</p>
            <p className="text-gray-600 font-medium">No analysis reports saved</p>
            <p className="text-sm text-gray-500 mt-2">
              Your analysis results will appear here after running analyses
            </p>
            <a
              href="/"
              className="mt-4 inline-block px-4 py-2 bg-blue-600 text-white rounded-md
                hover:bg-blue-700 font-semibold"
            >
              Go to Dashboard
            </a>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Reports List */}
            <div className="lg:col-span-1">
              <div className="bg-white rounded-lg shadow overflow-hidden">
                <div className="p-6 border-b border-gray-200">
                  <h2 className="text-lg font-bold text-gray-900">
                    Saved Reports ({reports.length})
                  </h2>
                </div>
                <div className="divide-y divide-gray-200 max-h-96 overflow-y-auto">
                  {reports.map((report) => (
                    <button
                      key={report.id}
                      onClick={() => setSelectedReport(report)}
                      className={`w-full text-left p-4 hover:bg-gray-50 transition-colors ${
                        selectedReport?.id === report.id
                          ? 'bg-blue-50 border-l-4 border-blue-600'
                          : ''
                      }`}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-semibold text-gray-900 truncate">
                            {report.filename || 'Analysis'}
                          </p>
                          <p className="text-xs text-gray-600 mt-1">
                            {report.patientId}
                          </p>
                          <p className="text-xs text-gray-500 mt-1">
                            {formatDate(report.timestamp)}
                          </p>
                        </div>
                        {report.result.alerts.length > 0 && (
                          <span className="text-red-600 font-bold text-sm">
                            ⚠️
                          </span>
                        )}
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Report Detail */}
            <div className="lg:col-span-2">
              {selectedReport ? (
                <div className="space-y-6">
                  {/* Header Card */}
                  <div className="bg-white rounded-lg shadow p-6">
                    <div className="flex items-start justify-between">
                      <div>
                        <h2 className="text-2xl font-bold text-gray-900">
                          {selectedReport.filename || 'Analysis Report'}
                        </h2>
                        <p className="text-gray-600 mt-2">
                          <strong>Patient ID:</strong> {selectedReport.patientId}
                        </p>
                        <p className="text-gray-600">
                          <strong>Analyzed:</strong> {formatDate(selectedReport.timestamp)}
                        </p>
                      </div>
                      <button
                        onClick={() => {
                          handleDelete(selectedReport.id);
                        }}
                        className="px-4 py-2 bg-red-100 text-red-700 rounded-md
                          hover:bg-red-200 font-semibold text-sm"
                      >
                        🗑️ Delete
                      </button>
                    </div>
                  </div>

                  {/* Metrics Summary */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-white rounded-lg shadow p-4">
                      <p className="text-xs font-semibold text-gray-700 uppercase">
                        Seizure Events
                      </p>
                      <p className="text-2xl font-bold text-gray-900 mt-2">
                        {selectedReport.result.alerts.length}
                      </p>
                    </div>
                    <div className="bg-white rounded-lg shadow p-4">
                      <p className="text-xs font-semibold text-gray-700 uppercase">
                        Peak Probability
                      </p>
                      <p className="text-2xl font-bold text-gray-900 mt-2">
                        {(selectedReport.result.summary.peak_probability * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>

                  {/* Alerts */}
                  {selectedReport.result.alerts.length > 0 && (
                    <div className="bg-white rounded-lg shadow overflow-hidden">
                      <div className="p-6 border-b border-gray-200">
                        <h3 className="text-lg font-bold text-gray-900">
                          Detected Events
                        </h3>
                      </div>
                      <div className="divide-y divide-gray-200">
                        {selectedReport.result.alerts.map((alert, idx) => (
                          <div key={idx} className="p-4 hover:bg-gray-50">
                            <div className="flex items-center justify-between">
                              <div>
                                <p className="text-sm font-semibold text-gray-900">
                                  Event {idx + 1}
                                </p>
                                <p className="text-xs text-gray-600 mt-1">
                                  {formatDuration(alert.start_sec)} -{' '}
                                  {formatDuration(alert.end_sec)} ({alert.duration_sec.toFixed(1)}s)
                                </p>
                              </div>
                              <div className="text-right">
                                <p className="text-sm font-semibold text-gray-900">
                                  {(alert.peak_prob * 100).toFixed(1)}%
                                </p>
                                <p className="text-xs text-gray-600">confidence</p>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Analysis Parameters */}
                  <div className="bg-white rounded-lg shadow p-6">
                    <h3 className="text-lg font-bold text-gray-900 mb-4">
                      Analysis Parameters
                    </h3>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Threshold:</span>
                        <span className="font-semibold text-gray-900">
                          {(selectedReport.result.analysis_params?.threshold || 0.5).toFixed(2)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Smoothing Window:</span>
                        <span className="font-semibold text-gray-900">
                          {selectedReport.result.analysis_params?.smooth_window || 5} samples
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Consecutive Windows:</span>
                        <span className="font-semibold text-gray-900">
                          {selectedReport.result.analysis_params?.consecutive_windows || 3}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Export */}
                  <div className="flex gap-3">
                    <button
                      onClick={() => {
                        const json = JSON.stringify(selectedReport.result, null, 2);
                        const blob = new Blob([json], {
                          type: 'application/json',
                        });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `report_${selectedReport.id}.json`;
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        URL.revokeObjectURL(url);
                      }}
                      className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-md
                        hover:bg-blue-700 font-semibold text-sm"
                    >
                      💾 Download
                    </button>
                    <button
                      onClick={() => window.print()}
                      className="flex-1 px-4 py-2 bg-gray-600 text-white rounded-md
                        hover:bg-gray-700 font-semibold text-sm"
                    >
                      🖨️ Print
                    </button>
                  </div>
                </div>
              ) : (
                <div className="bg-white rounded-lg shadow p-12 text-center">
                  <p className="text-2xl mb-2">📄</p>
                  <p className="text-gray-600 font-medium">
                    Select a report to view details
                  </p>
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
