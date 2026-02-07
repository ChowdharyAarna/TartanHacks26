import { useState } from "react";
import { AnalysisResult, AnalyzeEnqueueResponse, JobStatusResponse, JobStatus } from "@/types/analysis";
import { demoResult } from "@/lib/demoData";

// Set to true to use demo data instead of real API
const USE_DEMO_MODE = false;

// Types are defined in src/types/analysis.ts and match the FastAPI backend.

interface UseVideoAnalysisReturn {
  isAnalyzing: boolean;
  error: string | null;
  result: AnalysisResult | null;
  analyzeVideo: (file: File) => Promise<void>;
  reset: () => void;

  // Optional: allow user to cancel a running job.
  cancel: () => Promise<void>;

  // NEW: for nicer UI labels
  jobStatus: JobStatus | "uploading" | null;
}

async function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

export function useVideoAnalysis(): UseVideoAnalysisReturn {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatus | "uploading" | null>(
    null
  );

  const [cancelUrl, setCancelUrl] = useState<string | null>(null);

  const analyzeVideo = async (file: File) => {
    setIsAnalyzing(true);
    setError(null);
    setJobStatus("uploading");

    try {
      // Demo mode - simulate API delay and return mock data
      if (USE_DEMO_MODE) {
        await sleep(2000);
        const localVideoUrl = URL.createObjectURL(file);
        setResult({
          ...demoResult,
          videoUrl: localVideoUrl,
        });
        setJobStatus("done");
        return;
      }

      const formData = new FormData();
      formData.append("file", file);

      // 1) enqueue job
      const enqueueRes = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });

      if (!enqueueRes.ok) {
        throw new Error(`Analysis failed: ${enqueueRes.statusText}`);
      }

      const enqueue: AnalyzeEnqueueResponse = await enqueueRes.json();
      setJobStatus("queued");
      setCancelUrl(enqueue.cancel_url ?? null);

      // 2) poll job status
      const intervalMs = 750;
      const timeoutMs = 5 * 60 * 1000; // 5 minutes
      const start = Date.now();

      while (true) {
        const statusRes = await fetch(enqueue.status_url);
        if (!statusRes.ok) {
          throw new Error(`Job status failed: ${statusRes.statusText}`);
        }

        const job: JobStatusResponse = await statusRes.json();
        setJobStatus(job.status);

        if (job.status === "cancelled") {
          // Job was cancelled (either by the user or server-side).
          setCancelUrl(null);
          break;
        }

        if (job.status === "error") {
          throw new Error(job.error ?? "Analysis job failed");
        }

        if (job.status === "done") {
          const raw = job.result;

          // Backend returns plot_url (snake_case). Map to plotUrl for TS + UI.
          const data: AnalysisResult = {
            ...raw,
            plotUrl: raw?.plotUrl ?? raw?.plot_url ?? null,
          };

          setResult(data);
          try {
            localStorage.setItem("breathmap:lastResult", JSON.stringify(data));
          } catch {
            // ignore storage errors
          }
          break;
        }

        if (Date.now() - start > timeoutMs) {
          throw new Error("Timed out waiting for analysis");
        }

        await sleep(intervalMs);
      }
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

  const cancel = async () => {
    // Best-effort cancel. Backend will mark job as cancelled immediately,
    // but the worker process may still finish in the background.
    if (!cancelUrl) return;
    try {
      await fetch(cancelUrl, { method: "POST" });
      setJobStatus("cancelled");
    } catch {
      // ignore
    }
  };

  const reset = () => {
    setIsAnalyzing(false);
    setError(null);
    setResult(null);
    setJobStatus(null);
    setCancelUrl(null);
  };

  return {
    isAnalyzing,
    error,
    result,
    analyzeVideo,
    cancel,
    reset,
    jobStatus,
  };
}
