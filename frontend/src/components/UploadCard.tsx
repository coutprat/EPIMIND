import { useState } from 'react';
import { apiClient } from '../lib/api';
import type { AnalysisResponse } from '../lib/types';

interface UploadCardProps {
  onAnalysisComplete: (result: AnalysisResponse) => void;
  onLoading: (loading: boolean) => void;
  onError: (error: string) => void;
  demoMode: boolean;
}

export const UploadCard: React.FC<UploadCardProps> = ({
  onAnalysisComplete,
  onLoading,
  onError,
  demoMode,
}) => {
  const [mode, setMode] = useState<'edf' | 'sample'>('edf');
  const [threshold, setThreshold] = useState(0.5);
  const [smoothWindow, setSmoothWindow] = useState(5);
  const [consecutiveWindows, setConsecutiveWindows] = useState(3);
  const [selectedSample, setSelectedSample] = useState('chb01');

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    onLoading(true);
    onError('');

    try {
      const result = await apiClient.analyzeEDF(
        file,
        threshold,
        smoothWindow,
        consecutiveWindows
      );
      onAnalysisComplete(result);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Analysis failed';
      onError(message);
    } finally {
      onLoading(false);
      // Reset file input
      event.target.value = '';
    }
  };

  const handleSampleAnalysis = async () => {
    onLoading(true);
    onError('');

    try {
      const result = await apiClient.analyzeNPZ(
        selectedSample,
        threshold,
        smoothWindow,
        consecutiveWindows
      );
      onAnalysisComplete(result);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Analysis failed';
      onError(message);
    } finally {
      onLoading(false);
    }
  };

  const handleDemoMode = async () => {
    onLoading(true);
    onError('');

    try {
      const result = await apiClient.loadDemoResult();
      onAnalysisComplete(result);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load demo';
      onError(message);
    } finally {
      onLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow p-6 space-y-6">
      <h2 className="text-2xl font-bold text-gray-900">EEG Analysis</h2>

      {/* Mode Selection */}
      <div className="space-y-3">
        <label className="text-sm font-semibold text-gray-700">Analysis Mode</label>
        <div className="flex gap-4">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="radio"
              name="mode"
              value="edf"
              checked={mode === 'edf'}
              onChange={(e) => setMode(e.target.value as 'edf' | 'sample')}
              className="w-4 h-4"
            />
            <span className="text-sm text-gray-700">Upload EDF File</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="radio"
              name="mode"
              value="sample"
              checked={mode === 'sample'}
              onChange={(e) => setMode(e.target.value as 'edf' | 'sample')}
              className="w-4 h-4"
            />
            <span className="text-sm text-gray-700">Sample Data</span>
          </label>
        </div>
      </div>

      {/* Mode-specific Input */}
      {mode === 'edf' && (
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">
            EDF File
          </label>
          <input
            type="file"
            accept=".edf"
            onChange={handleFileUpload}
            className="block w-full text-sm text-gray-500
              file:mr-4 file:py-2 file:px-4
              file:rounded-md file:border-0
              file:text-sm file:font-semibold
              file:bg-blue-50 file:text-blue-700
              hover:file:bg-blue-100
              cursor-pointer"
          />
          <p className="mt-2 text-xs text-gray-500">
            Supported formats: EDF, EDF+, BDF (all channels with timestamps)
          </p>
        </div>
      )}

      {mode === 'sample' && (
        <div className="space-y-3">
          <label className="block text-sm font-semibold text-gray-700">
            Sample Patient
          </label>
          <select
            value={selectedSample}
            onChange={(e) => setSelectedSample(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm
              focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="chb01">CHB-MIT 01</option>
            <option value="chb02">CHB-MIT 02</option>
          </select>
          <button
            onClick={handleSampleAnalysis}
            className="w-full px-4 py-2 bg-blue-600 text-white rounded-md
              hover:bg-blue-700 font-semibold text-sm"
          >
            Analyze Sample
          </button>
        </div>
      )}

      {/* Analysis Parameters */}
      <div className="space-y-4 border-t pt-4">
        <h3 className="text-sm font-semibold text-gray-700">Parameters</h3>

        <div>
          <label className="block text-sm text-gray-700 mb-2">
            Threshold: {threshold.toFixed(2)}
          </label>
          <input
            type="range"
            min="0.05"
            max="0.95"
            step="0.05"
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>0.05</span>
            <span>0.95</span>
          </div>
        </div>

        <div>
          <label className="block text-sm text-gray-700 mb-2">
            Smoothing Window: {smoothWindow} samples
          </label>
          <input
            type="range"
            min="1"
            max="20"
            step="1"
            value={smoothWindow}
            onChange={(e) => setSmoothWindow(parseInt(e.target.value))}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>1</span>
            <span>20</span>
          </div>
        </div>

        <div>
          <label className="block text-sm text-gray-700 mb-2">
            Consecutive Windows: {consecutiveWindows}
          </label>
          <input
            type="range"
            min="1"
            max="10"
            step="1"
            value={consecutiveWindows}
            onChange={(e) => setConsecutiveWindows(parseInt(e.target.value))}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>1</span>
            <span>10</span>
          </div>
        </div>
      </div>

      {/* Demo Mode Button */}
      {demoMode && (
        <button
          onClick={handleDemoMode}
          className="w-full px-4 py-2 bg-gray-600 text-white rounded-md
            hover:bg-gray-700 font-semibold text-sm"
        >
          Load Demo Result
        </button>
      )}
    </div>
  );
};
