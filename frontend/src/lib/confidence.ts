/**
 * Confidence scoring for alerts with explainable heuristics.
 * High/Medium/Low confidence based on multiple factors.
 */

import type { AlertEvent, TimelinePoint } from './types';

export type ConfidenceLevel = 'High' | 'Medium' | 'Low';

export interface ConfidenceInfo {
  level: ConfidenceLevel;
  score: number; // 0-100
  reasons: string[];
}

/**
 * Compute confidence for a single alert.
 * Considers: peak probability, duration, volatility within alert window.
 */
export function getAlertConfidence(
  alert: AlertEvent,
  timeline: TimelinePoint[],
  strideSec: number = 1
): ConfidenceInfo {
  const reasons: string[] = [];
  let score = 50; // Base score

  // 1. Peak probability (0–40 points)
  const peakProb = alert.peak_prob;
  if (peakProb >= 0.9) {
    score += 40;
    reasons.push('Very high peak probability (≥0.90)');
  } else if (peakProb >= 0.8) {
    score += 35;
    reasons.push('High peak probability (≥0.80)');
  } else if (peakProb >= 0.6) {
    score += 20;
    reasons.push('Moderate peak probability (≥0.60)');
  } else if (peakProb >= 0.5) {
    score += 10;
    reasons.push('Lower peak probability (<0.60)');
  }

  // 2. Duration (0–30 points)
  const duration = alert.duration_sec;
  if (duration >= 20) {
    score += 30;
    reasons.push('Long sustained event (≥20s)');
  } else if (duration >= 10) {
    score += 25;
    reasons.push('Moderate duration (10–20s)');
  } else if (duration >= 5) {
    score += 15;
    reasons.push('Short but present event (5–10s)');
  } else {
    score += 5;
    reasons.push('Very brief event (<5s)');
  }

  // 3. Volatility within alert window (0–30 points)
  const volatility = computeAlertVolatility(alert, timeline, strideSec);
  if (volatility < 0.1) {
    score += 30;
    reasons.push('Very stable signal within alert');
  } else if (volatility < 0.2) {
    score += 20;
    reasons.push('Stable signal within alert');
  } else if (volatility < 0.35) {
    score += 10;
    reasons.push('Moderate volatility within alert');
  } else {
    reasons.push('High volatility (may affect reliability)');
  }

  // 4. Mean probability consistency (bonus/penalty)
  if (alert.mean_prob !== undefined && alert.peak_prob - alert.mean_prob > 0.3) {
    score -= 10;
    reasons.push('Peak much higher than mean (spiky, less consistent)');
  } else if (alert.mean_prob !== undefined && alert.peak_prob - alert.mean_prob < 0.1) {
    score += 5;
    reasons.push('Consistent probability throughout event');
  }

  // Clamp score to 0–100
  score = Math.max(0, Math.min(100, score));

  // Determine level
  let level: ConfidenceLevel;
  if (score >= 75) {
    level = 'High';
  } else if (score >= 50) {
    level = 'Medium';
  } else {
    level = 'Low';
  }

  return { level, score, reasons };
}

/**
 * Compute volatility (std dev) within alert window.
 * Returns normalized volatility (0–1).
 */
function computeAlertVolatility(
  alert: AlertEvent,
  timeline: TimelinePoint[],
  strideSec: number
): number {
  const startIdx = Math.floor(alert.start_sec / strideSec);
  const endIdx = Math.ceil(alert.end_sec / strideSec);

  const probs = timeline
    .slice(Math.max(0, startIdx), Math.min(timeline.length, endIdx))
    .map(p => p.prob);

  if (probs.length === 0) return 0;

  const mean = probs.reduce((a, b) => a + b) / probs.length;
  const variance = probs.reduce((sum, p) => sum + (p - mean) ** 2, 0) / probs.length;
  const stdDev = Math.sqrt(variance);

  // Normalize by range (0–1)
  const normalized = Math.min(1, stdDev * 2);
  return normalized;
}

/**
 * Generate human-readable explanation for confidence level.
 */
export function getConfidenceExplanation(info: ConfidenceInfo): string {
  const levelEmoji = info.level === 'High' ? '🟢' : info.level === 'Medium' ? '🟡' : '🔴';
  const levelText = info.level === 'High' ? 'very reliable' : info.level === 'Medium' ? 'moderately reliable' : 'less reliable';

  const intro = `${levelEmoji} **${info.level} Confidence (${info.score}/100)**: This event is ${levelText}.`;
  const factors = info.reasons.map(r => `• ${r}`).join('\n');

  return `${intro}\n\nKey Factors:\n${factors}`;
}

/**
 * Get confidence badge color class for UI.
 */
export function getConfidenceColor(level: ConfidenceLevel): string {
  if (level === 'High') return 'bg-green-100 text-green-800 border-green-300';
  if (level === 'Medium') return 'bg-yellow-100 text-yellow-800 border-yellow-300';
  return 'bg-red-100 text-red-800 border-red-300';
}
