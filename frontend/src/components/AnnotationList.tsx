import { Annotation } from "@/types/analysis";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { MessageSquare } from "lucide-react";

interface AnnotationListProps {
  annotations: Annotation[];
  onSeek: (time: number) => void;
  currentTime: number;
}

export function AnnotationList({
  annotations,
  onSeek,
  currentTime,
}: AnnotationListProps) {
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const isActive = (time: number) => {
    return Math.abs(currentTime - time) < 2;
  };

  if (annotations.length === 0) return null;

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <MessageSquare className="w-4 h-4" />
          Key Moments
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="space-y-2">
          {annotations.map((annotation, index) => (
            <button
              key={index}
              onClick={() => onSeek(annotation.time)}
              className={`w-full text-left p-3 rounded-lg transition-all hover:bg-muted/80 ${
                isActive(annotation.time)
                  ? "bg-primary/10 border border-primary/30"
                  : "bg-muted/40"
              }`}
            >
              <div className="flex items-start gap-3">
                <span
                  className={`text-sm font-mono px-2 py-0.5 rounded ${
                    isActive(annotation.time)
                      ? "bg-primary text-primary-foreground"
                      : "bg-muted"
                  }`}
                >
                  {formatTime(annotation.time)}
                </span>
                <span className="text-sm">{annotation.text}</span>
              </div>
            </button>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
