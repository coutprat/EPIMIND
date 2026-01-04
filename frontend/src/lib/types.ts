export interface TimelinePoint {
  t_sec: number;
  prob: number;
}

export interface AlertEvent {
  start_sec: number;
  end_sec: number;
  peak_prob: number;
  duration_sec: number;
  mean_prob?: number; // Mean probability during alert
}

export interface Metrics {
  alerts_count: number;
  peak_probability: number;
  mean_probability: number;
  fp_estimate_per_hour: number;
  total_duration_sec: number;
}

export interface SummaryMetrics {
  alerts_count: number;
  peak_probability: number;
  mean_probability: number;
  fp_estimate_per_hour: number;
}

export interface AnalysisResponse {
  patient: string;
  patient_id?: string;
  filename?: string;
  fs: number;
  window_samples: number;
  stride_samples: number;
  stride_sec: number;
  threshold: number;
  num_windows: number;
  duration_sec: number;
  timeline: TimelinePoint[];
  alerts: AlertEvent[];
  summary: SummaryMetrics;
  analysis_params?: {
    threshold: number;
    smooth_window: number;
    consecutive_windows: number;
  };
}

export interface HealthResponse {
  status: string;
  model_available: boolean;
  model_type: string;
}

export interface TopChannel {
  channel: string;
  importance: number; // Probability drop when channel is zeroed
}

export interface StoredReport {
  id: string;
  timestamp: string;
  patientId: string;
  filename: string;
  result: AnalysisResponse;
  detectionThreshold: number; // Threshold used to generate alerts
  explanationThreshold?: number; // Threshold used in explanation (defaults to detection threshold)
  topChannels?: TopChannel[]; // Top 5 influential channels from explainability
}
