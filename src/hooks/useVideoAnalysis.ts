import { useState } from "react";
import { AnalysisResult } from "@/types/analysis";
import { demoResult } from "@/lib/demoData";

// Set to true to use demo data instead of real API
const USE_DEMO_MODE = true;

interface UseVideoAnalysisReturn {
  isAnalyzing: boolean;
  error: string | null;
  result: AnalysisResult | null;
  analyzeVideo: (file: File) => Promise<void>;
  reset: () => void;
}

export function useVideoAnalysis(): UseVideoAnalysisReturn {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);

  const analyzeVideo = async (file: File) => {
    setIsAnalyzing(true);
    setError(null);

    try {
      // Demo mode - simulate API delay and return mock data
      if (USE_DEMO_MODE) {
        await new Promise((resolve) => setTimeout(resolve, 2000));
        // Create a local URL for the uploaded video
        const localVideoUrl = URL.createObjectURL(file);
        setResult({
          ...demoResult,
          videoUrl: localVideoUrl,
        });
        return;
      }

      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`);
      }

      const data: AnalysisResult = await response.json();
      setResult(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong while analyzing your video. Please try again."
      );
    } finally {
      setIsAnalyzing(false);
    }
  };

  const reset = () => {
    setIsAnalyzing(false);
    setError(null);
    setResult(null);
  };

  return {
    isAnalyzing,
    error,
    result,
    analyzeVideo,
    reset,
  };
}
