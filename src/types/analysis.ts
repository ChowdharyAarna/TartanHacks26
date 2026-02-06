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
  timelineScores: TimelineScore[];
  annotations: Annotation[];
  summary: AnalysisSummary;
  shortExplanation: string;
}
