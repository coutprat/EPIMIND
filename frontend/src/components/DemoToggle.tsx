import React from 'react';

interface DemoToggleProps {
  demoMode: boolean;
  onToggle: (enabled: boolean) => void;
  apiAvailable: boolean;
}

export const DemoToggle: React.FC<DemoToggleProps> = ({
  demoMode,
  onToggle,
  apiAvailable,
}) => {
  return (
    <div className="bg-white rounded-lg shadow p-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-gray-900">Demo Mode</h3>
          <p className="text-xs text-gray-600 mt-1">
            {apiAvailable
              ? 'Backend API is connected'
              : 'Backend API is unavailable - using demo data'}
          </p>
        </div>
        <button
          onClick={() => onToggle(!demoMode)}
          disabled={!apiAvailable}
          className={`relative inline-flex h-8 w-14 items-center rounded-full transition-colors ${
            demoMode
              ? 'bg-blue-600'
              : 'bg-gray-300'
          } ${!apiAvailable ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          <span
            className={`inline-block h-6 w-6 transform rounded-full bg-white transition-transform ${
              demoMode ? 'translate-x-7' : 'translate-x-1'
            }`}
          />
        </button>
      </div>

      {/* Status indicator */}
      <div className="mt-3 flex items-center gap-2">
        <div
          className={`h-2 w-2 rounded-full ${
            apiAvailable ? 'bg-green-500' : 'bg-red-500'
          }`}
        />
        <span
          className={`text-xs font-medium ${
            apiAvailable ? 'text-green-700' : 'text-red-700'
          }`}
        >
          {apiAvailable ? 'API Connected' : 'API Disconnected'}
        </span>
      </div>
    </div>
  );
};
