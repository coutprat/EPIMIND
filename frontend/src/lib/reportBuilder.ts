/**
 * Report builder: Generate plain-English medical-style summaries without LLM.
 * Uses rule-based templates and metadata.
 */

import type { AlertEvent } from './types';

export interface ReportData {
  patientId: string;
  testId: string;
  timestamp: string;
  duration_hours: number;
  total_alerts: number;
  strongest_alert: AlertEvent | null;
  threshold: number;
  consecutive_windows: number;
  fp_estimate_per_hour: number | null;
  stability_score?: number;
}

export interface GeneratedReport {
  title: string;
  summary: string;
  metrics_bullets: string[];
  interpretation: string;
}

/**
 * Generate a human-readable medical report from analysis data.
 */
export function generateReport(data: ReportData): GeneratedReport {
  const alertDensity = data.total_alerts / Math.max(data.duration_hours, 0.01);
  const hasAlerts = data.total_alerts > 0;

  // Title
  const title = `EEG Analysis Report: Patient ${data.patientId}`;

  // Summary paragraph
  const summaryLines = [
    `Analysis Date: ${new Date(data.timestamp).toLocaleDateString('en-US')}`,
    `Recording Duration: ${formatHours(data.duration_hours)}`,
    `Total Events Detected: ${data.total_alerts}`,
  ];

  if (hasAlerts && data.strongest_alert) {
    const timeStr = formatSeconds(data.strongest_alert.start_sec);
    summaryLines.push(
      `Strongest Event: Peak probability ${(data.strongest_alert.peak_prob * 100).toFixed(1)}% at ${timeStr}`
    );
  }

  const summary = summaryLines.join('. ') + '.';

  // Metrics bullets
  const metrics_bullets = [
    `Detection Threshold: ${(data.threshold * 100).toFixed(0)}%`,
    `Consecutive Windows Required: ${data.consecutive_windows}`,
    `Event Density: ${alertDensity.toFixed(2)} events/hour`,
  ];

  if (data.fp_estimate_per_hour !== null) {
    metrics_bullets.push(`Estimated False Positive Rate: ${data.fp_estimate_per_hour.toFixed(2)} per hour`);
  }

  if (data.stability_score !== undefined) {
    const stability = data.stability_score > 75 ? 'Stable' : data.stability_score > 50 ? 'Moderate' : 'Noisy';
    metrics_bullets.push(`Signal Stability: ${stability} (score: ${data.stability_score.toFixed(0)}/100)`);
  }

  // Interpretation paragraph (rule-based)
  let interpretation = '';

  if (!hasAlerts) {
    interpretation = `No events detected above the ${(data.threshold * 100).toFixed(0)}% threshold during this recording. ` +
      `If the threshold is too high, consider reducing it to improve sensitivity. ` +
      `The recording appears to be within normal parameters.`;
  } else if (alertDensity < 1) {
    interpretation = `Low event density (${alertDensity.toFixed(2)} events/hour) suggests good control with current settings. ` +
      `Consider these settings as a baseline. If clinical correlation indicates need for improvement, ` +
      `reduce threshold or consecutive windows to increase detection sensitivity.`;
  } else if (alertDensity < 5) {
    interpretation = `Moderate event density (${alertDensity.toFixed(2)} events/hour). ` +
      `Review events for clinical significance. Current threshold of ${(data.threshold * 100).toFixed(0)}% appears reasonable. ` +
      `If false alarms are present, increase threshold or consecutive windows.`;
  } else {
    interpretation = `High event density (${alertDensity.toFixed(2)} events/hour) detected. ` +
      `This may indicate true pathology or threshold sensitivity. ` +
      `Recommend increasing threshold or consecutive windows to reduce false positives. ` +
      `Correlate with clinical presentation and EEG expert review.`;
  }

  // Add signal quality note
  if (data.stability_score !== undefined && data.stability_score < 50) {
    interpretation += ` Note: Signal has high fluctuations, which may increase false detections. `;
    interpretation += `Check electrode contact and reduce artifacts if possible.`;
  }

  return {
    title,
    summary,
    metrics_bullets,
    interpretation,
  };
}

/**
 * Format hours to human-readable string (e.g., "2h 30m" or "45m").
 */
export function formatHours(hours: number): string {
  const h = Math.floor(hours);
  const m = Math.round((hours - h) * 60);
  if (h === 0) return `${m}m`;
  if (m === 0) return `${h}h`;
  return `${h}h ${m}m`;
}

/**
 * Format seconds to MM:SS format.
 */
export function formatSeconds(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

/**
 * Export report as plain text.
 */
export function reportAsText(report: GeneratedReport): string {
  const lines = [
    report.title,
    '='.repeat(report.title.length),
    '',
    report.summary,
    '',
    'Metrics:',
    ...report.metrics_bullets.map(b => `• ${b}`),
    '',
    'Interpretation:',
    report.interpretation,
  ];
  return lines.join('\n');
}

/**
 * Export report as markdown.
 */
export function reportAsMarkdown(report: GeneratedReport): string {
  const lines = [
    `# ${report.title}`,
    '',
    report.summary,
    '',
    '## Metrics',
    ...report.metrics_bullets.map(b => `- ${b}`),
    '',
    '## Interpretation',
    report.interpretation,
  ];
  return lines.join('\n');
}
