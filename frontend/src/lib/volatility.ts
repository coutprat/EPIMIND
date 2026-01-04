/**
 * Volatility and signal stability analysis.
 * Compute rolling variance, generate stability scores.
 */

import type { TimelinePoint } from './types';

export interface VolatilityAnalysis {
  stability_score: number; // 0–100 (100 = most stable)
  volatility_mean: number; // Average rolling std dev
  volatility_max: number; // Peak rolling std dev
  is_noisy: boolean; // True if volatility > threshold
}

/**
 * Compute rolling volatility (standard deviation) over timeline.
 * Window size in number of points (default 30).
 */
export function computeVolatility(
  timeline: TimelinePoint[],
  windowSize: number = 30
): VolatilityAnalysis {
  if (timeline.length < 2) {
    return {
      stability_score: 100,
      volatility_mean: 0,
      volatility_max: 0,
      is_noisy: false,
    };
  }

  // Compute rolling standard deviation
  const rollingStdDevs: number[] = [];

  for (let i = 0; i <= timeline.length - windowSize; i++) {
    const window = timeline.slice(i, i + windowSize).map(p => p.prob);
    const mean = window.reduce((a, b) => a + b) / window.length;
    const variance = window.reduce((sum, p) => sum + (p - mean) ** 2, 0) / window.length;
    const stdDev = Math.sqrt(variance);
    rollingStdDevs.push(stdDev);
  }

  // If timeline too short for full rolling window, pad with edge values
  if (rollingStdDevs.length === 0) {
    const probs = timeline.map(p => p.prob);
    const mean = probs.reduce((a, b) => a + b) / probs.length;
    const variance = probs.reduce((sum, p) => sum + (p - mean) ** 2, 0) / probs.length;
    const stdDev = Math.sqrt(variance);
    rollingStdDevs.push(stdDev);
  }

  // Statistics
  const volatility_mean = rollingStdDevs.reduce((a, b) => a + b, 0) / rollingStdDevs.length;
  const volatility_max = Math.max(...rollingStdDevs);

  // Normalize volatility to stability score (0–100)
  // High volatility → low score
  // Assume max reasonable volatility is ~0.3 (30% swings)
  const normalized_volatility = Math.min(1, volatility_mean / 0.3);
  const stability_score = Math.round((1 - normalized_volatility) * 100);

  // Mark as noisy if volatility exceeds threshold
  const is_noisy = volatility_mean > 0.2;

  return {
    stability_score,
    volatility_mean,
    volatility_max,
    is_noisy,
  };
}

/**
 * Get human-readable stability label.
 */
export function getStabilityLabel(score: number): string {
  if (score >= 75) return 'Stable';
  if (score >= 50) return 'Moderate';
  return 'Noisy';
}

/**
 * Get stability warning message if noisy.
 */
export function getStabilityWarning(analysis: VolatilityAnalysis): string | null {
  if (!analysis.is_noisy) return null;

  const msg = 'High signal fluctuations detected. ' +
    'This may increase false alarms. Consider: ' +
    '(1) Increasing detection threshold, ' +
    '(2) Requiring more consecutive windows, ' +
    '(3) Checking electrode contact and reducing artifacts.';

  return msg;
}

/**
 * Get color for stability visualization.
 */
export function getStabilityColor(score: number): string {
  if (score >= 75) return 'bg-green-100 text-green-900 border-green-300'; // Stable
  if (score >= 50) return 'bg-yellow-100 text-yellow-900 border-yellow-300'; // Moderate
  return 'bg-red-100 text-red-900 border-red-300'; // Noisy
}
