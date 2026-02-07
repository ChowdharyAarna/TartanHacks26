import { useRef, useState } from "react";
import { AnalysisResult } from "@/types/analysis";
import { demoResult } from "@/lib/demoData";

// Set to true to use demo data instead of real API
const USE_DEMO_MODE = false;

type JobStatus = "queued" | "running" | "done" | "error" | "cancelled";

type AnalyzeEnqueueResponse = {
  job_id: string;
  video_id: string;
  status_url: string; // e.g. "/api/jobs/<job_id>"
  cancel_url: string; // e.g. "/api/jobs/<job_id>/cancel"
  video_url: string; // e.g. "/media/<vid>.mp4"
};

type JobResponse = {
  status: JobStatus;
  created_at: number;
  started_at?: number | null;
  finished_at?: number | null;
  error?: string | null;
  result?: any | null; // when done, this is your AnalysisResult payload
};

interface UseVideoAnalysisReturn {
  isAnalyzing: boolean;
  error: string | null;
  result: AnalysisResult | null;
  analyzeVideo: (file: File) => Promise<void>;
  reset: () => void;

  // UI labels
  jobStatus: JobStatus | "uploading" | null;

  // NEW: cancel support
  cancel: () => Promise<void>;
}

async function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

// Backend base URL:
// - local dev: http://localhost:8000
// - production (GitHub Pages): set VITE_API_BASE in build step
const API_BASE = (import.meta.env.VITE_API_BASE ?? "http://localhost:8000").replace(
  /\/$/,
  ""
);

function apiUrl(p: string) {
  if (!p) return API_BASE;
  if (p.startsWith("http://") || p.startsWith("https://")) return p;
  return `${API_BASE}${p.startsWith("/") ? p : `/${p}`}`;
}

export function useVideoAnalysis(): UseVideoAnalysisReturn {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatus | "uploading" | null>(
    null
  );

  // store enqueue info so we can cancel
  const enqueueRef = useRef<AnalyzeEnqueueResponse | null>(null);

  const analyzeVideo = async (file: File) => {
    setIsAnalyzing(true);
    setError(null);
    setJobStatus("uploading");
    enqueueRef.current = null;

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
      const enqueueRes = await fetch(apiUrl("/api/analyze"), {
        method: "POST",
        body: formData,
      });

      if (!enqueueRes.ok) {
        throw new Error(`Analysis failed: ${enqueueRes.statusText}`);
      }

      const enqueue: AnalyzeEnqueueResponse = await enqueueRes.json();
      enqueueRef.current = enqueue;

      setJobStatus("queued");

      // 2) poll job status
      const intervalMs = 750;
      const timeoutMs = 5 * 60 * 1000; // 5 minutes
      const start = Date.now();

      while (true) {
        const statusRes = await fetch(apiUrl(enqueue.status_url));
        if (!statusRes.ok) {
          throw new Error(`Job status failed: ${statusRes.statusText}`);
        }

        const job: JobResponse = await statusRes.json();
        setJobStatus(job.status);

        if (job.status === "error") {
          throw new Error(job.error ?? "Analysis job failed");
        }

        if (job.status === "cancelled") {
          throw new Error("Analysis cancelled.");
        }

        if (job.status === "done") {
          const raw = job.result;

          // Backend returns plot_url (snake_case). Map to plotUrl for TS + UI.
          const data: AnalysisResult = {
            ...raw,
            plotUrl: raw?.plot_url ?? raw?.plotUrl ?? null,
            videoUrl: raw?.videoUrl ?? raw?.video_url ?? raw?.video_url ?? raw?.videoUrl,
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
    const enqueue = enqueueRef.current;
    if (!enqueue?.cancel_url) return;

    try {
      await fetch(apiUrl(enqueue.cancel_url), { method: "POST" });
      setJobStatus("cancelled");
      setIsAnalyzing(false);
    } catch {
      // ignore cancel errors
    }
  };

  const reset = () => {
    setIsAnalyzing(false);
    setError(null);
    setResult(null);
    setJobStatus(null);
    enqueueRef.current = null;
  };

  return {
    isAnalyzing,
    error,
    result,
    analyzeVideo,
    reset,
    jobStatus,
    cancel,
  };
}
