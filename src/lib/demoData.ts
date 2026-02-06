import { AnalysisResult } from "@/types/analysis";

// Demo data for testing the UI without a backend
export const demoResult: AnalysisResult = {
  videoUrl: "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
  timelineScores: Array.from({ length: 60 }, (_, i) => ({
    time: i,
    score: Math.round(
      75 +
        15 * Math.sin(i * 0.15) +
        Math.random() * 5 -
        (i === 12 || i === 31 ? 15 : 0)
    ),
  })),
  annotations: [
    { time: 12.4, text: "Shoulder spike detected - sudden upward movement" },
    { time: 24.0, text: "Good breathing rhythm established" },
    { time: 31.0, text: "Breathing becomes more stable" },
    { time: 45.5, text: "Excellent breath control during sustained note" },
  ],
  summary: {
    finalScore: 84,
    avgScore: 81,
    stability: 0.92,
    spikeCount: 3,
  },
  shortExplanation:
    "Breathing rhythm is stable with minimal shoulder spikes. Good overall control with consistent patterns throughout the performance.",
};
