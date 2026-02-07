export interface TimelineScore {
  time: number;
  score: number;
}

export interface Annotation {
  time: number;
  text: string;
}

export interface AnalysisSummary {
  finalScore: number;
  avgScore: number;
  stability: number;
  spikeCount: number;
}

export interface AnalysisResult {
  videoUrl: string;

  // Backend-generated plot image (served from /outputs/...)
  plotUrl?: string | null;

  // Some backend variants return snake_case.
  plot_url?: string | null;

  // Raw backend report (time/resp/peaks/inhalations, etc.)
  report?: any;

  // Optional extra outputs from the backend pipeline
  singing_analysis?: any;
  breath_feedback?: any;
  recommendations?: string[];

  timelineScores: TimelineScore[];
  annotations: Annotation[];
  summary: AnalysisSummary;
  shortExplanation: string;
}

export type JobStatus = "queued" | "running" | "done" | "error" | "cancelled";

export interface JobState {
  status: JobStatus;
  created_at: number;
  started_at?: number | null;
  finished_at?: number | null;
  error?: string | null;
  result?: any | null;
  cancel_requested?: boolean;
}

export interface AnalyzeEnqueueResponse {
  job_id: string;
  video_id: string;
  status_url: string;
  cancel_url: string;
  video_url: string;
}

export type JobStatusResponse = JobState;
