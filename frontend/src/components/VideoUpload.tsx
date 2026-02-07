import { useCallback, useState, useRef, useEffect } from "react";
import { Upload, FileVideo, X, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";

interface VideoUploadProps {
  onAnalyze: (file: File) => void;
  onCancel?: () => void | Promise<void>;
  isAnalyzing: boolean;
  statusText?: string | null;
}

export function VideoUpload({ onAnalyze, onCancel, isAnalyzing, statusText }: VideoUploadProps) {
  const [file, setFile] = useState<File | null>(null);
  const [duration, setDuration] = useState<number | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // NEW: fake progress bar value
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (!isAnalyzing) {
      setProgress(0);
      return;
    }

    // Start at a non-zero value so it doesn’t look stuck
    setProgress(8);

    const id = window.setInterval(() => {
      setProgress((p) => {
        const cap = 92; // hover under 100% until done
        if (p >= cap) return cap;
        const bump = 2 + Math.random() * 6; // 2..8
        return Math.min(cap, p + bump);
      });
    }, 500);

    return () => window.clearInterval(id);
  }, [isAnalyzing]);

  const handleFile = useCallback((selectedFile: File) => {
    const okType =
      selectedFile.type === "video/mp4" || selectedFile.type === "video/quicktime";
    const okExt = /\.(mp4|mov)$/i.test(selectedFile.name);
    if (!okType && !okExt) return;
    setFile(selectedFile);

    // Get video duration
    const url = URL.createObjectURL(selectedFile);
    const video = document.createElement("video");
    video.preload = "metadata";
    video.onloadedmetadata = () => {
      setDuration(video.duration);
      URL.revokeObjectURL(url);
    };
    video.src = url;
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile) {
        handleFile(droppedFile);
      }
    },
    [handleFile]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleClick = () => {
    inputRef.current?.click();
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      handleFile(selectedFile);
    }
  };

  const clearFile = () => {
    setFile(null);
    setDuration(null);
    if (inputRef.current) {
      inputRef.current.value = "";
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024 * 1024) {
      return `${(bytes / 1024).toFixed(1)} KB`;
    }
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const analyzingLabel =
    statusText === "queued"
      ? "Queued…"
      : statusText === "running"
      ? "Analyzing…"
      : statusText === "cancelled"
      ? "Cancelling…"
      : "Uploading…";

  return (
    <div className="w-full max-w-xl mx-auto">
      <input
        ref={inputRef}
        type="file"
        accept="video/mp4,video/quicktime,.mp4,.mov"
        onChange={handleInputChange}
        className="hidden"
        disabled={isAnalyzing}
      />

      {!file ? (
        <Card
          className={cn(
            "border-2 border-dashed transition-all duration-200 cursor-pointer",
            isDragOver
              ? "border-primary bg-primary/5"
              : "border-muted-foreground/25 hover:border-primary/50 hover:bg-muted/50"
          )}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={handleClick}
        >
          <CardContent className="flex flex-col items-center justify-center py-16 px-8">
            <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center mb-4">
              <Upload className="w-8 h-8 text-primary" />
            </div>
            <h3 className="text-lg font-semibold mb-2">
              Drop your singing video here
            </h3>
            <p className="text-muted-foreground text-center mb-4">
              or click to browse
            </p>
            <p className="text-sm text-muted-foreground">Accepts MP4 or MOV</p>
          </CardContent>
        </Card>
      ) : (
        <Card className="border-2 border-primary/30 bg-primary/5">
          <CardContent className="py-6 px-6">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                <FileVideo className="w-6 h-6 text-primary" />
              </div>
              <div className="flex-1 min-w-0">
                <h4 className="font-medium truncate">{file.name}</h4>
                <div className="flex gap-3 text-sm text-muted-foreground mt-1">
                  <span>{formatFileSize(file.size)}</span>
                  {duration && <span>• {formatDuration(duration)}</span>}
                </div>
              </div>
              {!isAnalyzing && (
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={(e) => {
                    e.stopPropagation();
                    clearFile();
                  }}
                  className="flex-shrink-0"
                >
                  <X className="w-4 h-4" />
                </Button>
              )}
            </div>

            <Button
              className="w-full mt-6"
              size="lg"
              onClick={() => onAnalyze(file)}
              disabled={isAnalyzing}
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  {analyzingLabel}
                </>
              ) : (
                "Analyze Video"
              )}
            </Button>

            {isAnalyzing && (
              <div className="mt-4 space-y-2">
                <Progress value={progress} className="h-2" />
                <p className="text-sm text-muted-foreground text-center">
                  This may take up to a minute
                </p>

                {onCancel && (
                  <div className="flex justify-center">
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation();
                        onCancel();
                      }}
                    >
                      Cancel
                    </Button>
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
