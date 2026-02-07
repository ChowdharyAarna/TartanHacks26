import { useRef, useState, useCallback } from "react";
import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { AnalysisResult } from "@/types/analysis";
import { VideoPlayer, VideoPlayerHandle } from "./VideoPlayer";
import { ScoreDisplay } from "./ScoreDisplay";
import { BreathTimeline } from "./BreathTimeline";
import { AnnotationList } from "./AnnotationList";
import { SummaryCards } from "./SummaryCards";
import { BackendDataPanel } from "./BackendDataPanel";
import { WaveformPlot } from "./WaveformPlot";

interface ResultsViewProps {
  result: AnalysisResult;
  onReset: () => void;
}

export function ResultsView({ result, onReset }: ResultsViewProps) {
  const videoRef = useRef<VideoPlayerHandle>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  const handleSeek = useCallback((time: number) => {
    videoRef.current?.seekTo(time);
  }, []);

  // ✅ Pull messages safely (works even if breath_feedback is missing)
  const feedbackMessages =
    (result as any)?.breath_feedback?.messages ??
    (result as any)?.breathFeedback?.messages ??
    (result as any)?.postGraphMessages ??
    [];

  const hasFeedback =
    Array.isArray(feedbackMessages) && feedbackMessages.length > 0;

  const plotUrl =
    (result as any)?.plotUrl ?? (result as any)?.plot_url ?? (result as any)?.plotUrl;
  const singingPngUrl =
    (result as any)?.singing_analysis?.results?.artifacts?.report_png_url;
  const singingCsvUrl =
    (result as any)?.singing_analysis?.results?.artifacts?.timeline_csv_url;

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b bg-card/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Button variant="ghost" size="sm" onClick={onReset}>
              <ArrowLeft className="w-4 h-4 mr-2" />
              New Analysis
            </Button>
            <h1 className="text-lg font-semibold">Breath Control Analysis</h1>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-6">
        <div className="grid lg:grid-cols-5 gap-6">
          {/* Left Column - Video */}
          <div className="lg:col-span-3 space-y-4">
            <VideoPlayer
              ref={videoRef}
              src={result.videoUrl}
              onTimeUpdate={setCurrentTime}
              onDurationChange={setDuration}
            />

            {/* Timeline below video */}
            <div className="bg-card rounded-xl p-4 border">
              <BreathTimeline
                timelineScores={result.timelineScores}
                annotations={result.annotations}
                currentTime={currentTime}
                duration={duration}
                onSeek={handleSeek}
              />
            </div>

            {/* ✅ Feedback card BEFORE graphs */}
            {hasFeedback && (
              <div className="bg-card rounded-xl p-4 border">
                <div className="flex items-center justify-between mb-2">
                  <h2 className="text-base font-semibold">Breath feedback</h2>
                  <span className="text-xs text-muted-foreground">
                    Suggestions based on your clip
                  </span>
                </div>

                <div className="space-y-2">
                  {feedbackMessages.map((msg: string, i: number) => (
                    <div
                      key={i}
                      className="rounded-lg border bg-background/40 px-3 py-2 text-sm"
                    >
                      {msg}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Live waveform synced to playback */}
            {result.report && (
              <div className="bg-card rounded-xl p-4 border">
                <h2 className="text-base font-semibold mb-3">
                  Breathing Waveform (synced)
                </h2>
                <WaveformPlot report={result.report} currentTime={currentTime} />
              </div>
            )}

            {/* Backend-generated plot (static PNG) */}
            {plotUrl && (
              <div className="bg-card rounded-xl p-4 border">
                <div className="flex items-center justify-between mb-3">
                  <h2 className="text-base font-semibold">Breath plot (PNG)</h2>
                  <a
                    href={plotUrl}
                    target="_blank"
                    rel="noreferrer"
                    className="text-sm text-primary hover:underline"
                  >
                    Open
                  </a>
                </div>
                <img
                  src={plotUrl}
                  alt="Breath plot"
                  className="w-full rounded-lg border bg-background"
                  loading="lazy"
                />
              </div>
            )}

            {/* Singing analysis artifacts (if enabled on backend) */}
            {(singingPngUrl || singingCsvUrl) && (
              <div className="bg-card rounded-xl p-4 border">
                <div className="flex items-center justify-between mb-2">
                  <h2 className="text-base font-semibold">Singing analysis</h2>
                  <span className="text-xs text-muted-foreground">Artifacts</span>
                </div>
                <div className="flex flex-wrap gap-2">
                  {singingPngUrl && (
                    <a
                      href={singingPngUrl}
                      target="_blank"
                      rel="noreferrer"
                      className="px-3 py-2 rounded-md border bg-background/40 text-sm hover:bg-muted"
                    >
                      Open report plot
                    </a>
                  )}
                  {singingCsvUrl && (
                    <a
                      href={singingCsvUrl}
                      target="_blank"
                      rel="noreferrer"
                      className="px-3 py-2 rounded-md border bg-background/40 text-sm hover:bg-muted"
                    >
                      Download timeline CSV
                    </a>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Right Column - Analysis */}
          <div className="lg:col-span-2 space-y-4">
            <ScoreDisplay
              score={result.summary.finalScore}
              explanation={result.shortExplanation}
            />

            <SummaryCards summary={result.summary} />

            <AnnotationList
              annotations={result.annotations}
              onSeek={handleSeek}
              currentTime={currentTime}
            />

            <BackendDataPanel result={result} />
          </div>
        </div>
      </main>
    </div>
  );
}
