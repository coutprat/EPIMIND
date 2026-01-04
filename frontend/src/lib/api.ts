import axios from 'axios';
import type { AxiosInstance } from 'axios';
import type { AnalysisResponse, HealthResponse } from './types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

class APIClient {
  private client: AxiosInstance;

  constructor(baseURL: string = API_BASE_URL) {
    this.client = axios.create({
      baseURL,
      timeout: 120000, // 2 minutes for long-running analysis
    });
  }

  /**
   * Check API health
   */
  async health(): Promise<HealthResponse> {
    const response = await this.client.get<HealthResponse>('/health');
    return response.data;
  }

  /**
   * Analyze EDF file
   */
  async analyzeEDF(
    file: File,
    threshold: number = 0.5,
    smoothWindow: number = 5,
    consecutiveWindows: number = 3
  ): Promise<AnalysisResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await this.client.post<AnalysisResponse>(
      '/analyze/edf',
      formData,
      {
        params: {
          threshold,
          smooth_window: smoothWindow,
          consecutive_windows: consecutiveWindows,
        },
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );

    return response.data;
  }

  /**
   * Analyze preprocessed NPZ file
   */
  async analyzeNPZ(
    patient: string,
    threshold: number = 0.5,
    smoothWindow: number = 5,
    consecutiveWindows: number = 3
  ): Promise<AnalysisResponse> {
    const response = await this.client.post<AnalysisResponse>(
      '/analyze/npz',
      {},
      {
        params: {
          patient,
          threshold,
          smooth_window: smoothWindow,
          consecutive_windows: consecutiveWindows,
        },
      }
    );

    return response.data;
  }

  /**
   * Load demo result
   */
  async loadDemoResult(): Promise<AnalysisResponse> {
    const response = await fetch('/demo/demo_result.json');
    if (!response.ok) throw new Error('Failed to load demo result');
    return response.json();
  }
}

export const apiClient = new APIClient();
