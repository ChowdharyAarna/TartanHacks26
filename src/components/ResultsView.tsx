import { useRef, useState, useCallback } from "react";
import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { AnalysisResult } from "@/types/analysis";
import { VideoPlayer, VideoPlayerHandle } from "./VideoPlayer";
import { ScoreDisplay } from "./ScoreDisplay";
import { BreathTimeline } from "./BreathTimeline";
import { AnnotationList } from "./AnnotationList";
import { SummaryCards } from "./SummaryCards";

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
          </div>
        </div>
      </main>
    </div>
  );
}
