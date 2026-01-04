/**
 * Export utilities: JSON, CSV, and combined formats.
 */

import type { AnalysisResponse, AlertEvent } from './types';

export interface ExportConfig {
  threshold: number;
  consecutive_windows: number;
  stride_sec: number;
  mode: string;
  patient_id: string;
  timestamp: string;
}

export interface ExportRun {
  id: string;
  config: ExportConfig;
  timeline: Array<{ t_sec: number; prob: number }>;
  alerts: AlertEvent[];
  summary: {
    total_alerts: number;
    peak_probability: number;
    mean_probability: number;
    duration_sec: number;
  };
}

/**
 * Build an exportable run JSON from analysis response and tuning params.
 */
export function buildExportRun(
  result: AnalysisResponse,
  threshold: number,
  consecutive: number,
  alerts: AlertEvent[]
): ExportRun {
  return {
    id: `run_${Date.now()}`,
    config: {
      threshold,
      consecutive_windows: consecutive,
      stride_sec: result.stride_sec || 1,
      mode: 'tuned',
      patient_id: result.patient_id || result.patient || 'unknown',
      timestamp: new Date().toISOString(),
    },
    timeline: result.timeline,
    alerts,
    summary: {
      total_alerts: alerts.length,
      peak_probability: alerts.length > 0 ? Math.max(...alerts.map(a => a.peak_prob)) : 0,
      mean_probability: alerts.length > 0 ? alerts.reduce((sum, a) => sum + (a.mean_prob || 0), 0) / alerts.length : 0,
      duration_sec: result.duration_sec,
    },
  };
}

/**
 * Export single run as JSON string.
 */
export function exportRunAsJson(run: ExportRun): string {
  return JSON.stringify(run, null, 2);
}

/**
 * Export alerts as CSV string.
 * Columns: start_s,end_s,duration_s,peak_prob,mean_prob,label
 */
export function exportAlertsAsCsv(alerts: AlertEvent[]): string {
  const header = 'start_s,end_s,duration_s,peak_prob,mean_prob,label';
  const rows = alerts.map((alert, idx) => {
    const start = alert.start_sec.toFixed(2);
    const end = alert.end_sec.toFixed(2);
    const duration = alert.duration_sec.toFixed(2);
    const peak = alert.peak_prob.toFixed(4);
    const mean = (alert.mean_prob || 0).toFixed(4);
    const label = `Alert_${idx + 1}`;
    return `${start},${end},${duration},${peak},${mean},${label}`;
  });
  return [header, ...rows].join('\n');
}

/**
 * Export both runs (compare A vs B) as JSON.
 */
export function exportComparativeJson(runA: ExportRun, runB: ExportRun): string {
  const comparative = {
    comparison_type: 'run_a_vs_run_b',
    timestamp: new Date().toISOString(),
    run_a: runA,
    run_b: runB,
    delta: {
      alert_count_diff: runB.summary.total_alerts - runA.summary.total_alerts,
      peak_prob_diff: (runB.summary.peak_probability - runA.summary.peak_probability).toFixed(4),
      mean_prob_diff: (runB.summary.mean_probability - runA.summary.mean_probability).toFixed(4),
    },
  };
  return JSON.stringify(comparative, null, 2);
}

/**
 * Trigger browser download of content with filename.
 */
export function downloadFile(content: string, filename: string, mimeType: string = 'text/plain'): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/**
 * Copy text to clipboard.
 */
export async function copyToClipboard(text: string): Promise<void> {
  try {
    await navigator.clipboard.writeText(text);
  } catch (err) {
    // Fallback for older browsers
    const textarea = document.createElement('textarea');
    textarea.value = text;
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand('copy');
    document.body.removeChild(textarea);
  }
}
