import { useMemo, useRef, useState } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  ResponsiveContainer,
  ReferenceLine,
  Tooltip,
} from "recharts";
import { TimelineScore, Annotation } from "@/types/analysis";
import { cn } from "@/lib/utils";

interface BreathTimelineProps {
  timelineScores: TimelineScore[];
  annotations: Annotation[];
  currentTime: number;
  duration: number;
  onSeek: (time: number) => void;
}

export function BreathTimeline({
  timelineScores,
  annotations,
  currentTime,
  duration,
  onSeek,
}: BreathTimelineProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoveredAnnotation, setHoveredAnnotation] = useState<Annotation | null>(
    null
  );

  const currentScore = useMemo(() => {
    if (timelineScores.length === 0) return null;
    const closest = timelineScores.reduce((prev, curr) =>
      Math.abs(curr.time - currentTime) < Math.abs(prev.time - currentTime)
        ? curr
        : prev
    );
    return closest.score;
  }, [timelineScores, currentTime]);

  const handleChartClick = (e: any) => {
    if (e && e.activeLabel !== undefined) {
      onSeek(e.activeLabel);
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-popover border rounded-lg shadow-lg px-3 py-2">
          <p className="text-sm font-medium">
            Score: <span className="text-primary">{payload[0].value}</span>
          </p>
          <p className="text-xs text-muted-foreground">{formatTime(label)}</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full" ref={containerRef}>
      {/* Current Score Label */}
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm text-muted-foreground">
          Breathing Score Over Time
        </span>
        {currentScore !== null && (
          <div className="flex items-center gap-2 bg-primary/10 rounded-full px-3 py-1">
            <span className="text-xs text-muted-foreground">
              {formatTime(currentTime)}
            </span>
            <span className="text-sm font-semibold text-primary">
              Score: {currentScore}
            </span>
          </div>
        )}
      </div>

      {/* Chart */}
      <div className="h-32 w-full cursor-pointer">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={timelineScores}
            onClick={handleChartClick}
            margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
          >
            <defs>
              <linearGradient id="scoreGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.3} />
                <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis
              dataKey="time"
              tickFormatter={formatTime}
              axisLine={false}
              tickLine={false}
              tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
              interval="preserveStartEnd"
            />
            <YAxis
              domain={[0, 100]}
              axisLine={false}
              tickLine={false}
              tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
              width={30}
            />
            <Tooltip content={<CustomTooltip />} />
            <Area
              type="monotone"
              dataKey="score"
              stroke="hsl(var(--primary))"
              strokeWidth={2}
              fill="url(#scoreGradient)"
            />
            {/* Playhead */}
            <ReferenceLine
              x={currentTime}
              stroke="hsl(var(--primary))"
              strokeWidth={2}
              strokeDasharray="none"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Annotation Markers */}
      <div className="relative h-8 mt-2 border-t">
        {annotations.map((annotation, index) => {
          const position = (annotation.time / duration) * 100;
          return (
            <button
              key={index}
              className={cn(
                "absolute -top-2 w-4 h-4 rounded-full bg-accent border-2 border-accent-foreground/30 transition-all hover:scale-125 hover:bg-primary",
                hoveredAnnotation === annotation && "bg-primary scale-125"
              )}
              style={{ left: `${position}%`, transform: "translateX(-50%)" }}
              onClick={() => onSeek(annotation.time)}
              onMouseEnter={() => setHoveredAnnotation(annotation)}
              onMouseLeave={() => setHoveredAnnotation(null)}
              title={annotation.text}
            />
          );
        })}
        {hoveredAnnotation && (
          <div
            className="absolute top-6 bg-popover border rounded-lg shadow-lg px-3 py-2 text-sm z-10 max-w-xs"
            style={{
              left: `${(hoveredAnnotation.time / duration) * 100}%`,
              transform: "translateX(-50%)",
            }}
          >
            <p className="text-xs text-muted-foreground mb-1">
              {formatTime(hoveredAnnotation.time)}
            </p>
            <p>{hoveredAnnotation.text}</p>
          </div>
        )}
      </div>
    </div>
  );
}
