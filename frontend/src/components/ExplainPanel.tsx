import type { AnalysisResponse } from '../lib/types';

interface ExplainPanelProps {
  result: AnalysisResponse;
  explanationThreshold: number; // Threshold for explanation (may differ from detection threshold)
  detectionThreshold?: number; // Optional: show detection threshold for reference
}

export const ExplainPanel: React.FC<ExplainPanelProps> = ({ 
  result, 
  explanationThreshold, 
  detectionThreshold 
}) => {
  // Generate rule-based explanation based on explanation threshold
  const generateExplanation = () => {
    const alerts = result.alerts || [];
    const summary = result.summary;
    const peakRisk = summary?.peak_probability || 0;
    const meanRisk = summary?.mean_probability || 0;
    const fpPerHour = summary?.fp_estimate_per_hour || 0;
    const alertCount = alerts.length;

    let explanation = '';

    // Part 1: Alert trigger explanation (using explanation threshold)
    if (alertCount === 0) {
      explanation = `✅ No alerts detected. Peak risk was ${peakRisk.toFixed(3)} (below explanation threshold of ${explanationThreshold.toFixed(2)}).`;
    } else {
      const longestAlert = alerts.reduce((max, a) => 
        (a.duration_sec > max.duration_sec) ? a : max
      );
      
      explanation = `⚠️ Alert triggered: Risk stayed above ${explanationThreshold.toFixed(2)} for ${longestAlert.duration_sec.toFixed(1)}s.\n`;
      explanation += `Peak risk was ${peakRisk.toFixed(3)} at ${(longestAlert.start_sec + longestAlert.duration_sec / 2).toFixed(0)}s.\n`;
      explanation += `${alertCount} alert(s) detected in total.`;
    }

    // Part 2: Risk statistics
    explanation += `\n\n📊 Risk Statistics:\n`;
    explanation += `• Average risk: ${meanRisk.toFixed(3)}\n`;
    explanation += `• Peak risk: ${peakRisk.toFixed(3)}\n`;
    explanation += `• Estimated false alarms/hour: ${fpPerHour.toFixed(1)}`;

    // Part 3: Confidence assessment
    explanation += `\n\n🎯 Confidence:\n`;
    if (peakRisk > 0.8) {
      explanation += `• High confidence alert (peak ${peakRisk.toFixed(3)})`;
    } else if (peakRisk > 0.6) {
      explanation += `• Moderate confidence alert (peak ${peakRisk.toFixed(3)})`;
    } else if (alertCount > 0) {
      explanation += `• Low confidence alert (peak ${peakRisk.toFixed(3)})`;
    } else {
      explanation += `• No significant risk detected`;
    }

    // Part 4: Recommendation
    explanation += `\n\n💡 Suggestion:\n`;
    if (fpPerHour > 2 && alertCount > 0) {
      explanation += `Consider raising explanation threshold to reduce false alarms.`;
    } else if (alertCount === 0 && peakRisk > 0.5) {
      explanation += `Consider lowering explanation threshold to catch marginal cases.`;
    } else {
      explanation += `Current explanation threshold appears appropriate for this recording.`;
    }

    return explanation;
  };

  return (
    <div className="bg-white rounded-lg shadow p-6 border-l-4 border-blue-500">
      <h3 className="text-lg font-bold text-gray-900 mb-4">
        🤖 Why This Result?
      </h3>
      <div className="bg-gray-50 rounded p-4 font-mono text-sm whitespace-pre-wrap text-gray-700">
        {generateExplanation()}
      </div>
      <div className="mt-4 text-xs text-gray-500 space-y-1">
        <div>Explanation Threshold: <span className="font-semibold">{explanationThreshold.toFixed(2)}</span></div>
        {detectionThreshold !== undefined && (
          <div>Detection Threshold: <span className="font-semibold">{detectionThreshold.toFixed(2)}</span></div>
        )}
        <div className="mt-2">Generated from model metrics using local rule-based reasoning.</div>
      </div>
    </div>
  );
};
