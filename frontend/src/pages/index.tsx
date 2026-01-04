import { useEffect, useState } from 'react';
import { UploadCard } from '../components/UploadCard';
import { TimelineChart } from '../components/TimelineChart';
import { AlertsTable } from '../components/AlertsTable';
import { MetricsCard } from '../components/MetricsCard';
import { DemoToggle } from '../components/DemoToggle';
import { ExplainPanel } from '../components/ExplainPanel';
import { ThresholdPlayground } from '../components/ThresholdPlayground';
import { RunHistory } from '../components/RunHistory';
import { ModelStatus } from '../components/ModelStatus';
import { ExplainabilityPanel } from '../components/ExplainabilityPanel';
import { LiveControls } from '../components/LiveControls';
import { ExplainableAlertDrawer } from '../components/ExplainableAlertDrawer';
import { TuningPanel } from '../components/TuningPanel';
import { StabilityCard } from '../components/StabilityCard';
import { AutoReportCard } from '../components/AutoReportCard';
import { DemoModeRunner } from '../components/DemoModeRunner';
import { apiClient } from '../lib/api';
import { computeAlerts, computeMetrics, extractProbabilities, getStrideSec } from '../lib/alertLogic';
import { computeVolatility } from '../lib/volatility';
import type { AnalysisResponse, StoredReport, AlertEvent, Metrics } from '../lib/types';

export default function Index() {
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [demoMode, setDemoMode] = useState(false);
  const [apiAvailable, setApiAvailable] = useState(false);
  const [detectionThreshold, setDetectionThreshold] = useState(0.5); // Threshold for alert generation
  const [explanationThreshold, setExplanationThreshold] = useState(0.5); // Threshold for explanations
  const [reports, setReports] = useState<StoredReport[]>([]);
  const [selectedReportId, setSelectedReportId] = useState<string | null>(null);

  // NEW: Live mode and tuning state
  const [isLiveMode, setIsLiveMode] = useState(false);
  const [isLivePlaying, setIsLivePlaying] = useState(false);
  const [liveSpeed, setLiveSpeed] = useState<1 | 2 | 4>(1);
  const [liveRevealedIndex, setLiveRevealedIndex] = useState(0);
  const [liveAlerts, setLiveAlerts] = useState<AlertEvent[]>([]);

  // NEW: Tuning state (independent from baseline)
  const [tuneThreshold, setTuneThreshold] = useState(0.5);
  const [tuneConsecutive, setTuneConsecutive] = useState(3);
  const [tunedAlerts, setTunedAlerts] = useState<AlertEvent[]>([]);
  const [tunedMetrics, setTunedMetrics] = useState<Metrics | null>(null);
  const [baselineMetrics, setBaselineMetrics] = useState<Metrics | null>(null);

  // NEW: Alert explanation drawer
  const [selectedAlert, setSelectedAlert] = useState<AlertEvent | null>(null);
  const [zoomRange, setZoomRange] = useState<[number, number] | undefined>();

  // NEW: Stability analysis and volatility
  const [volatilityAnalysis, setVolatilityAnalysis] = useState(computeVolatility([]));

  // NEW: Demo mode runner
  const [demoModeRunning, setDemoModeRunning] = useState(false);

  // Check API health on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        await apiClient.health();
        setApiAvailable(true);
      } catch {
        setApiAvailable(false);
      }
    };

    checkHealth();
    // Check every 30 seconds
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  // Load reports from localStorage on mount
  useEffect(() => {
    const savedReports = JSON.parse(
      localStorage.getItem('seizure_reports') || '[]'
    ) as StoredReport[];
    setReports(savedReports);
  }, []);

  const handleAnalysisComplete = (analysisResult: AnalysisResponse) => {
    setResult(analysisResult);
    setError('');
    setDetectionThreshold(0.5);
    setExplanationThreshold(0.5);
    setTuneThreshold(0.5);
    setTuneConsecutive(3);
    setLiveRevealedIndex(0);
    setLiveAlerts([]);
    setSelectedAlert(null);
    setZoomRange(undefined);

    // Compute baseline metrics
    const probs = analysisResult.timeline.map((p) => p.prob);
    const strideSec = analysisResult.stride_sec || 1;
    const baselineAlerts = computeAlerts(probs, 0.5, 3, strideSec);
    const baselineMetrics = computeMetrics(probs, baselineAlerts, strideSec);
    setBaselineMetrics(baselineMetrics);

    // Compute volatility/stability
    const volatility = computeVolatility(analysisResult.timeline);
    setVolatilityAnalysis(volatility);

    // Save to localStorage
    const existingReports: StoredReport[] = JSON.parse(
      localStorage.getItem('seizure_reports') || '[]'
    );
    const newReport: StoredReport = {
      id: Date.now().toString(),
      timestamp: new Date().toISOString(),
      patientId: analysisResult.patient_id || 'unknown',
      filename: analysisResult.filename || 'analysis',
      result: analysisResult,
      detectionThreshold: 0.5,
      explanationThreshold: 0.5,
      topChannels: [],
    };
    existingReports.unshift(newReport);
    const updatedReports = existingReports.slice(0, 20);
    localStorage.setItem('seizure_reports', JSON.stringify(updatedReports));
    setReports(updatedReports);
    setSelectedReportId(newReport.id);
  };

  const handleError = (errorMsg: string) => {
    setError(errorMsg);
    setResult(null);
  };

  const handleLoadingChange = (isLoading: boolean) => {
    setLoading(isLoading);
  };

  const handleDemoToggle = (enabled: boolean) => {
    setDemoMode(enabled);
    if (!enabled) {
      setResult(null);
      setError('');
    }
  };

  // Effect: Live mode streaming
  useEffect(() => {
    if (!isLiveMode || !isLivePlaying || !result) return;

    const interval = setInterval(() => {
      setLiveRevealedIndex((prev) => {
        const max = result.timeline.length;
        if (prev >= max - 1) {
          setIsLivePlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, 1000 / liveSpeed);

    return () => clearInterval(interval);
  }, [isLiveMode, isLivePlaying, liveSpeed, result]);

  // Effect: Update live alerts when revealed index changes
  useEffect(() => {
    if (!isLiveMode || !result) return;

    const probs = extractProbabilities(result.timeline.slice(0, liveRevealedIndex + 1));
    const strideSec = getStrideSec(result.timeline);
    const alerts = computeAlerts(probs, detectionThreshold, 3, strideSec);
    setLiveAlerts(alerts);
  }, [isLiveMode, liveRevealedIndex, result, detectionThreshold]);

  // Effect: Compute tuned metrics when thresholds change
  useEffect(() => {
    if (!result) return;

    const probs = extractProbabilities(result.timeline);
    const strideSec = getStrideSec(result.timeline);
    const alerts = computeAlerts(probs, tuneThreshold, tuneConsecutive, strideSec);
    const metrics = computeMetrics(probs, alerts, strideSec);
    
    setTunedAlerts(alerts);
    setTunedMetrics(metrics);
  }, [tuneThreshold, tuneConsecutive, result]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      {/* Header */}
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">
                EpiMind Dashboard
              </h1>
              <p className="text-gray-600 mt-1">
                Real-time seizure detection and analysis
              </p>
            </div>
            <div className="text-right space-y-2">
              {/* Model Status Badge */}
              <ModelStatus />
              <p className="text-sm font-semibold text-gray-700">
                {apiAvailable ? '🟢 Online' : '🔴 Offline'}
              </p>
              <p className="text-xs text-gray-500">
                {apiAvailable
                  ? 'Backend API connected'
                  : 'Using demo mode'}
              </p>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Error Alert */}
        {error && (
          <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4">
            <div className="flex items-start gap-3">
              <span className="text-2xl">⚠️</span>
              <div>
                <h3 className="font-semibold text-red-900">Analysis Error</h3>
                <p className="text-sm text-red-700 mt-1">{error}</p>
              </div>
              <button
                onClick={() => setError('')}
                className="ml-auto text-red-500 hover:text-red-700 font-semibold"
              >
                ✕
              </button>
            </div>
          </div>
        )}

        {/* Loading Indicator */}
        {loading && (
          <div className="mb-6 bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="flex items-center gap-3">
              <div className="animate-spin h-5 w-5 text-blue-600">⟳</div>
              <p className="text-sm font-semibold text-blue-900">
                Analyzing... This may take a moment.
              </p>
            </div>
          </div>
        )}

        {/* Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column: Upload and Controls */}
          <div className="lg:col-span-1 space-y-6">
            {/* API Status */}
            <DemoToggle
              demoMode={demoMode}
              onToggle={handleDemoToggle}
              apiAvailable={apiAvailable}
            />

            {/* Upload Card */}
            <UploadCard
              onAnalysisComplete={handleAnalysisComplete}
              onLoading={handleLoadingChange}
              onError={handleError}
              demoMode={demoMode && !apiAvailable}
            />

            {/* Run History */}
            {reports.length > 0 && (
              <RunHistory
                reports={reports}
                onSelectRun={(report) => {
                  setResult(report.result);
                  setSelectedReportId(report.id);
                  setDetectionThreshold(report.detectionThreshold ?? 0.5);
                  setExplanationThreshold(report.explanationThreshold ?? 0.5);
                }}
                selectedId={selectedReportId ?? undefined}
              />
            )}

            {/* Demo Mode Button */}
            {result && (
              <button
                onClick={() => setDemoModeRunning(true)}
                className="w-full px-4 py-3 bg-purple-600 text-white rounded-lg font-semibold hover:bg-purple-700 transition"
              >
                🎬 Start Demo (Panel Presentation)
              </button>
            )}

            {/* Quick Links */}
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-sm font-semibold text-gray-900 mb-4">
                Quick Info
              </h3>
              <ul className="space-y-3 text-sm">
                <li>
                  <a
                    href="/reports"
                    className="text-blue-600 hover:text-blue-700 font-medium"
                  >
                    📋 View Report History
                  </a>
                </li>
                <li>
                  <a
                    href="#"
                    onClick={() =>
                      window.open(
                        'https://github.com',
                        '_blank'
                      )
                    }
                    className="text-blue-600 hover:text-blue-700 font-medium"
                  >
                    📖 Documentation
                  </a>
                </li>
                <li>
                  <a
                    href="#"
                    onClick={() =>
                      alert(
                        'Model Type: ' +
                          (apiAvailable ? 'TorchScript/ONNX' : 'Demo')
                      )
                    }
                    className="text-blue-600 hover:text-blue-700 font-medium"
                  >
                    🤖 Model Info
                  </a>
                </li>
              </ul>
            </div>
          </div>

          {/* Right Column: Analysis Results */}
          <div className="lg:col-span-2 space-y-6">
            {result ? (
              <>
                {/* Live Controls */}
                <LiveControls
                  isLive={isLiveMode}
                  onToggleLive={setIsLiveMode}
                  isPlaying={isLivePlaying}
                  onPlayPause={() => setIsLivePlaying(!isLivePlaying)}
                  speed={liveSpeed}
                  onSpeedChange={(speed) => setLiveSpeed(speed as 1 | 2 | 4)}
                  onReset={() => {
                    setLiveRevealedIndex(0);
                    setIsLivePlaying(false);
                  }}
                  progress={liveRevealedIndex / result.timeline.length}
                />

                {/* Timeline Chart with Zoom Support */}
                <TimelineChart
                  data={isLiveMode ? result.timeline.slice(0, liveRevealedIndex + 1) : result.timeline}
                  threshold={detectionThreshold}
                  title="Seizure Probability Timeline"
                  alerts={isLiveMode ? liveAlerts : tunedAlerts.length > 0 ? tunedAlerts : result.alerts}
                  zoomRange={zoomRange}
                  onReset={() => setZoomRange(undefined)}
                  onZoomChange={(start, end) => setZoomRange([start, end])}
                />

                {/* Metrics */}
                <MetricsCard metrics={result.summary} />

                {/* Tuning Panel */}
                <TuningPanel
                  threshold={tuneThreshold}
                  onThresholdChange={setTuneThreshold}
                  consecutiveWindows={tuneConsecutive}
                  onConsecutiveChange={setTuneConsecutive}
                  metrics={tunedMetrics || result.summary}
                  baselineMetrics={baselineMetrics}
                  showComparison={true}
                />

                {/* Alerts Table (Now Clickable) */}
                <AlertsTable
                  alerts={tunedAlerts.length > 0 ? tunedAlerts : result.alerts}
                  title={`Detected Seizure Events${tunedAlerts.length > 0 ? ' (Tuned)' : ''}`}
                  onAlertClick={setSelectedAlert}
                  timeline={result.timeline}
                  strideSec={result.stride_sec || 1}
                />

                {/* AI Features Section */}
                <div className="space-y-6">
                  {/* Stability Meter */}
                  <StabilityCard analysis={volatilityAnalysis} />

                  {/* Auto Report */}
                  <div data-report-section>
                    <AutoReportCard
                      patientId={result.patient_id || result.patient || 'unknown'}
                      testId={result.filename || 'analysis'}
                      timestamp={new Date().toISOString()}
                      duration_hours={result.duration_sec / 3600}
                      total_alerts={tunedAlerts.length > 0 ? tunedAlerts.length : result.alerts.length}
                      strongest_alert={
                        tunedAlerts.length > 0
                          ? { start_sec: tunedAlerts[0].start_sec, peak_prob: tunedAlerts[0].peak_prob }
                          : result.alerts.length > 0
                          ? { start_sec: result.alerts[0].start_sec, peak_prob: result.alerts[0].peak_prob }
                          : null
                      }
                      threshold={tuneThreshold}
                      consecutive_windows={tuneConsecutive}
                      fp_estimate_per_hour={tunedMetrics?.fp_estimate_per_hour || baselineMetrics?.fp_estimate_per_hour || null}
                      stability_score={volatilityAnalysis.stability_score}
                    />
                  </div>

                  {/* Explain Panel - references explanation threshold */}
                  <ExplainPanel 
                    result={result} 
                    explanationThreshold={explanationThreshold}
                    detectionThreshold={detectionThreshold}
                  />

                  {/* Threshold Playground - for alert generation */}
                  <ThresholdPlayground
                    result={result}
                    currentThreshold={detectionThreshold}
                    onThresholdChange={setDetectionThreshold}
                  />

                  {/* Explainability Panel - local occlusion analysis */}
                  <ExplainabilityPanel result={result} />
                </div>

                {/* Export/Save Options */}
                <div className="bg-white rounded-lg shadow p-6">
                  <h3 className="text-sm font-semibold text-gray-900 mb-4">
                    Actions
                  </h3>
                  <div className="flex gap-3">
                    <button
                      onClick={() => {
                        const json = JSON.stringify(result, null, 2);
                        const blob = new Blob([json], {
                          type: 'application/json',
                        });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `analysis_${Date.now()}.json`;
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        URL.revokeObjectURL(url);
                      }}
                      className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-md
                        hover:bg-blue-700 font-semibold text-sm"
                    >
                      💾 Download Report
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
              </>
            ) : (
              <div className="bg-white rounded-lg shadow p-12 text-center">
                <p className="text-2xl mb-2">📊</p>
                <p className="text-gray-600 font-medium">
                  No analysis results yet
                </p>
                <p className="text-sm text-gray-500 mt-2">
                  Upload an EDF file or select sample data to begin analysis
                </p>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Explainable Alert Drawer */}
      <ExplainableAlertDrawer
        alert={selectedAlert}
        threshold={tuneThreshold}
        consecutiveWindows={tuneConsecutive}
        strideSec={result?.stride_sec || 1}
        onClose={() => setSelectedAlert(null)}
        onZoomToAlert={(startSec, endSec) => setZoomRange([startSec, endSec])}
        timeline={result?.timeline}
      />

      {/* Demo Mode Runner */}
      <DemoModeRunner
        isActive={demoModeRunning}
        onStepChange={(step) => console.log('Demo step:', step)}
        onStartLive={() => {
          setIsLiveMode(true);
          setIsLivePlaying(true);
          setLiveSpeed(2);
        }}
        onSetThreshold={(value) => setTuneThreshold(value)}
        onOpenAlert={() => {
          if (result?.alerts && result.alerts.length > 0) {
            setSelectedAlert(result.alerts[0]);
          }
        }}
        onScrollToReport={() => {
          const reportElement = document.querySelector('[data-report-section]');
          reportElement?.scrollIntoView({ behavior: 'smooth' });
        }}
        onStop={() => setDemoModeRunning(false)}
      />

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-center text-sm text-gray-600">
            EpiMind Demo System © 2024 | For research and demonstration purposes only
          </p>
        </div>
      </footer>
    </div>
  );
}
