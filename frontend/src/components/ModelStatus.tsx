import { useEffect, useState } from 'react';
import { apiClient } from '../lib/api';

interface ModelStatusProps {
  className?: string;
}

export const ModelStatus: React.FC<ModelStatusProps> = ({ className = '' }) => {
  const [modelStatus, setModelStatus] = useState<'real' | 'fallback' | 'loading' | 'unknown'>('loading');
  const [modelType, setModelType] = useState<string>('');

  useEffect(() => {
    const checkStatus = async () => {
      try {
        const health = await apiClient.health();
        if (health.model_available) {
          setModelStatus('real');
          setModelType(health.model_type || 'ONNX/TorchScript');
        } else {
          setModelStatus('fallback');
          setModelType('Dummy/Mock');
        }
      } catch {
        setModelStatus('unknown');
        setModelType('Unknown');
      }
    };

    checkStatus();
    // Re-check every 30 seconds
    const interval = setInterval(checkStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const getBadgeColor = () => {
    switch (modelStatus) {
      case 'real':
        return 'bg-green-100 text-green-800 border-green-300';
      case 'fallback':
        return 'bg-yellow-100 text-yellow-800 border-yellow-300';
      case 'unknown':
        return 'bg-gray-100 text-gray-800 border-gray-300';
      default:
        return 'bg-blue-100 text-blue-800 border-blue-300';
    }
  };

  const getStatusIcon = () => {
    switch (modelStatus) {
      case 'real':
        return '✓';
      case 'fallback':
        return '⚠';
      case 'unknown':
        return '?';
      default:
        return '⟳';
    }
  };

  const getStatusLabel = () => {
    switch (modelStatus) {
      case 'real':
        return 'Real Model';
      case 'fallback':
        return 'Fallback Mode';
      case 'unknown':
        return 'Unknown Status';
      default:
        return 'Checking...';
    }
  };

  return (
    <div className={`inline-flex items-center gap-2 px-3 py-2 rounded-full border text-sm font-medium ${getBadgeColor()} ${className}`}>
      <span className={modelStatus === 'loading' ? 'animate-spin' : ''}>{getStatusIcon()}</span>
      <span>{getStatusLabel()}</span>
      {modelType && modelStatus !== 'loading' && (
        <span className="text-xs opacity-70">({modelType})</span>
      )}
    </div>
  );
};
