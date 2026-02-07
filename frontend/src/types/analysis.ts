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

  // Raw backend report (time/resp/peaks/inhalations, etc.)
  report?: any;
timelineScores: TimelineScore[];
  annotations: Annotation[];
  summary: AnalysisSummary;
  shortExplanation: string;
}
