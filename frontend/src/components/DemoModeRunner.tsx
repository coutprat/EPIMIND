/**
 * DemoModeRunner: Guided walkthrough for college panel presentation.
 * Runs a state machine that triggers UI interactions: live mode, tuning, report display.
 */

import React, { useEffect, useState } from 'react';

type DemoStep = 'idle' | 'starting' | 'live_mode' | 'tuning' | 'drawer' | 'report' | 'finished';

interface DemoModeRunnerProps {
  isActive: boolean;
  onStepChange?: (step: DemoStep) => void;
  onStartLive?: () => void;
  onSetThreshold?: (value: number) => void;
  onOpenAlert?: () => void;
  onScrollToReport?: () => void;
  onStop?: () => void;
}

interface Caption {
  text: string;
  visible: boolean;
}

export const DemoModeRunner: React.FC<DemoModeRunnerProps> = ({
  isActive,
  onStepChange,
  onStartLive,
  onSetThreshold,
  onOpenAlert,
  onScrollToReport,
  onStop,
}) => {
  const [caption, setCaption] = useState<Caption>({ text: '', visible: false });
  const [progress, setProgress] = useState(0); // 0-100

  // State machine: transition through demo steps
  useEffect(() => {
    if (!isActive) {
      setCaption({ text: '', visible: false });
      return;
    }

    // Timeline: 0-30s demo
    const timings = {
      starting: 2000, // 0-2s: Intro message
      live_mode: 5000, // 2-7s: Start LIVE, play at 2x speed
      tuning: 5000, // 7-12s: Highlight tuning panel, adjust threshold
      drawer: 5000, // 12-17s: Click alert, show drawer
      report: 8000, // 17-25s: Scroll to report
      finished: 3000, // 25-28s: Finished message
    };

    const totalDuration = Object.values(timings).reduce((a, b) => a + b, 0);

    // Step 1: Intro (0-2s)
    const step1Timer = setTimeout(() => {
      onStepChange?.('live_mode');
      setCaption({ text: '🚀 Starting LIVE mode with 2x speed...', visible: true });
      onStartLive?.();
      setProgress(Math.round((2000 / totalDuration) * 100));
    }, timings.starting);

    // Step 2: Live mode (2-7s)
    const step2Timer = setTimeout(() => {
      onStepChange?.('tuning');
      setCaption({
        text: '⚙️ Now adjusting the detection threshold for better accuracy...',
        visible: true,
      });
      // Pulse the tuning panel with CSS would be here
      setProgress(Math.round(((timings.starting + timings.live_mode) / totalDuration) * 100));
    }, timings.starting + timings.live_mode);

    // Step 3: Adjust threshold (7-12s)
    const step3Timer = setTimeout(() => {
      onSetThreshold?.(0.60); // Increase from default 0.50 to 0.60
      setProgress(Math.round(((timings.starting + timings.live_mode + timings.tuning) / totalDuration) * 100));
    }, timings.starting + timings.live_mode + 1000);

    // Step 4: Click alert (12-17s)
    const step4Timer = setTimeout(() => {
      onStepChange?.('drawer');
      setCaption({
        text: '🎯 Opening the Explainable Alert Drawer to show event details...',
        visible: true,
      });
      onOpenAlert?.();
      setProgress(
        Math.round(
          ((timings.starting + timings.live_mode + timings.tuning) / totalDuration) * 100
        )
      );
    }, timings.starting + timings.live_mode + timings.tuning);

    // Step 5: Scroll to report (17-25s)
    const step5Timer = setTimeout(() => {
      onStepChange?.('report');
      setCaption({
        text: '📄 Generating and displaying the Auto Report section...',
        visible: true,
      });
      onScrollToReport?.();
      setProgress(
        Math.round(
          ((timings.starting + timings.live_mode + timings.tuning + timings.drawer) /
            totalDuration) *
            100
        )
      );
    }, timings.starting + timings.live_mode + timings.tuning + timings.drawer);

    // Step 6: Finished (25-28s)
    const step6Timer = setTimeout(() => {
      onStepChange?.('finished');
      setCaption({
        text: '✅ Demo complete! Try manual tuning or upload your own EEG data.',
        visible: true,
      });
      setProgress(100);
    }, timings.starting + timings.live_mode + timings.tuning + timings.drawer + timings.report);

    // Auto-stop after full duration
    const stopTimer = setTimeout(() => {
      setCaption({ text: '', visible: false });
      onStop?.();
    }, totalDuration);

    return () => {
      clearTimeout(step1Timer);
      clearTimeout(step2Timer);
      clearTimeout(step3Timer);
      clearTimeout(step4Timer);
      clearTimeout(step5Timer);
      clearTimeout(step6Timer);
      clearTimeout(stopTimer);
    };
  }, [isActive, onStepChange, onStartLive, onSetThreshold, onOpenAlert, onScrollToReport, onStop]);

  if (!isActive || !caption.visible) {
    return null;
  }

  return (
    <div className="fixed top-6 right-6 z-40 max-w-xs">
      {/* Caption Box */}
      <div className="bg-blue-600 text-white rounded-lg shadow-2xl p-4 mb-3 animate-pulse">
        <p className="font-semibold text-sm">{caption.text}</p>
      </div>

      {/* Progress Bar */}
      <div className="bg-white rounded-lg shadow p-3">
        <div className="flex justify-between items-center mb-2">
          <span className="text-xs font-semibold text-gray-700">Demo Progress</span>
          <span className="text-xs font-semibold text-gray-600">{progress}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
          <div
            className="bg-blue-600 h-2 rounded-full transition-all duration-500"
            style={{ width: `${progress}%` }}
          />
        </div>
        <p className="text-xs text-gray-500 mt-2 text-center">
          Follow along with the highlighted features
        </p>
      </div>
    </div>
  );
};
