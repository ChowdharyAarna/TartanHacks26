import { useState } from "react";
import { VideoUpload } from "@/components/VideoUpload";
import { ResultsView } from "@/components/ResultsView";
import { ErrorCard } from "@/components/ErrorCard";
import { useVideoAnalysis } from "@/hooks/useVideoAnalysis";
import { Activity } from "lucide-react";

const Index = () => {
  const { isAnalyzing, error, result, analyzeVideo, reset, jobStatus, cancel } = useVideoAnalysis();
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  const handleAnalyze = async (file: File) => {
    setUploadedFile(file);
    await analyzeVideo(file);
  };

  const handleReset = () => {
    reset();
    setUploadedFile(null);
  };

  const handleRetry = () => {
    if (uploadedFile) {
      analyzeVideo(uploadedFile);
    }
  };

  // Show results view if we have a result
  if (result) {
    return <ResultsView result={result} onReset={handleReset} />;
  }

  // Upload screen
  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Header */}
      <header className="border-b">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center">
              <Activity className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h1 className="text-xl font-bold">Video Breath Control Analysis</h1>
              <p className="text-sm text-muted-foreground">
                Analyze your singing technique
              </p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 flex items-center justify-center p-4">
        <div className="w-full max-w-xl space-y-6">
          {error ? (
            <ErrorCard message={error} onRetry={handleRetry} />
          ) : (
            <>
              <div className="text-center mb-8">
                <h2 className="text-2xl font-bold mb-2">
                  Upload Your Singing Video
                </h2>
                <p className="text-muted-foreground">
                  We'll analyze your breathing patterns and provide detailed
                  feedback on breath control
                </p>
              </div>
              <VideoUpload
                onAnalyze={handleAnalyze}
                onCancel={cancel}
                isAnalyzing={isAnalyzing}
                statusText={jobStatus}
              />
            </>
          )}

          {/* How it works */}
          {!error && !isAnalyzing && (
            <div className="mt-12 text-center">
              <h3 className="text-sm font-medium text-muted-foreground mb-4">
                How it works
              </h3>
              <div className="grid grid-cols-3 gap-4 text-center">
                <div>
                  <div className="w-8 h-8 rounded-full bg-muted flex items-center justify-center mx-auto mb-2">
                    <span className="text-sm font-medium">1</span>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Upload your video
                  </p>
                </div>
                <div>
                  <div className="w-8 h-8 rounded-full bg-muted flex items-center justify-center mx-auto mb-2">
                    <span className="text-sm font-medium">2</span>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    AI analyzes breathing
                  </p>
                </div>
                <div>
                  <div className="w-8 h-8 rounded-full bg-muted flex items-center justify-center mx-auto mb-2">
                    <span className="text-sm font-medium">3</span>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Get detailed feedback
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
};

export default Index;
