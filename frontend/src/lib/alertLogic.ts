/**
 * Alert detection and metrics computation utilities.
 * Reusable functions for computing alerts from probability timelines.
 */

import type { AlertEvent, Metrics, TimelinePoint } from './types';

/**
 * Compute alert segments from probability timeline.
 * An alert is triggered when probability >= threshold for consecutive_windows.
 */
export function computeAlerts(
  probabilities: number[],
  threshold: number,
  consecutiveWindows: number,
  strideSec: number = 1
): AlertEvent[] {
  const alerts: AlertEvent[] = [];
  
  if (probabilities.length === 0) return alerts;

  let inAlert = false;
  let alertStart = 0;
  let alertProbs: number[] = [];

  for (let i = 0; i < probabilities.length; i++) {
    const prob = probabilities[i];
    
    if (prob >= threshold) {
      // Extend or start alert
      if (!inAlert) {
        // Check if we have enough consecutive windows
        const consecutiveCount = countConsecutive(probabilities, i, threshold);
        if (consecutiveCount >= consecutiveWindows) {
          inAlert = true;
          alertStart = i;
          alertProbs = [];
        }
      }
      
      if (inAlert) {
        alertProbs.push(prob);
      }
    } else {
      // End alert if active
      if (inAlert && alertProbs.length >= consecutiveWindows) {
        const startSec = alertStart * strideSec;
        const endSec = i * strideSec;
        const peakProb = Math.max(...alertProbs);
        const meanProb = alertProbs.reduce((a, b) => a + b, 0) / alertProbs.length;
        
        alerts.push({
          start_sec: startSec,
          end_sec: endSec,
          peak_prob: peakProb,
          mean_prob: meanProb,
          duration_sec: endSec - startSec,
        });
      }
      
      inAlert = false;
      alertProbs = [];
    }
  }

  // Handle alert that extends to end of data
  if (inAlert && alertProbs.length >= consecutiveWindows) {
    const startSec = alertStart * strideSec;
    const endSec = probabilities.length * strideSec;
    const peakProb = Math.max(...alertProbs);
    const meanProb = alertProbs.reduce((a, b) => a + b, 0) / alertProbs.length;
    
    alerts.push({
      start_sec: startSec,
      end_sec: endSec,
      peak_prob: peakProb,
      mean_prob: meanProb,
      duration_sec: endSec - startSec,
    });
  }

  return alerts;
}

/**
 * Count how many consecutive windows exceed threshold starting from index.
 */
function countConsecutive(
  probabilities: number[],
  startIndex: number,
  threshold: number
): number {
  let count = 0;
  for (let i = startIndex; i < probabilities.length; i++) {
    if (probabilities[i] >= threshold) {
      count++;
    } else {
      break;
    }
  }
  return count;
}

/**
 * Compute metrics from probabilities and alerts.
 */
export function computeMetrics(
  probabilities: number[],
  alerts: AlertEvent[],
  strideSec: number = 1
): Metrics {
  const totalDurationSec = probabilities.length * strideSec;
  const hoursRepresented = totalDurationSec / 3600;
  const peakProb = probabilities.length > 0 ? Math.max(...probabilities) : 0;
  const meanProb = probabilities.length > 0 
    ? probabilities.reduce((a, b) => a + b, 0) / probabilities.length 
    : 0;

  return {
    alerts_count: alerts.length,
    peak_probability: peakProb,
    mean_probability: meanProb,
    fp_estimate_per_hour: hoursRepresented > 0 ? (alerts.length / hoursRepresented) * 2 : 0, // Rough FP estimate
    total_duration_sec: totalDurationSec,
  };
}

/**
 * Extract raw probabilities from timeline data.
 */
export function extractProbabilities(timeline: TimelinePoint[]): number[] {
  return timeline.map((point) => point.prob);
}

/**
 * Get stride seconds from timeline (if available).
 */
export function getStrideSec(timeline: TimelinePoint[]): number {
  if (timeline.length < 2) return 1;
  return timeline[1].t_sec - timeline[0].t_sec;
}
