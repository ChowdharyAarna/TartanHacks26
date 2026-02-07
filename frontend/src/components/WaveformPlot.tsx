import { useEffect, useMemo, useRef } from "react";

type BreathEvent = { time_s?: number; type?: string; confidence?: number };
type Inhalation = { t_start: number; t_peak: number; duration_s?: number; confidence?: number };

function downsampleXY(x: number[], y: number[], maxPoints: number) {
  if (x.length <= maxPoints) return { x, y };
  const step = Math.ceil(x.length / maxPoints);
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i < x.length; i += step) {
    xs.push(x[i]);
    ys.push(y[i]);
  }
  return { x: xs, y: ys };
}

export function WaveformPlot({
  report,
  currentTime,
}: {
  report: any;
  currentTime: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const data = useMemo(() => {
    const t: number[] = (report?.time ?? []) as number[];
    const resp: number[] = (report?.resp ?? []) as number[];

    // Downsample for smooth rendering
    const ds = downsampleXY(t, resp, 1400);

    const peaks: BreathEvent[] = (report?.breath_events_video ?? []) as BreathEvent[];
    const inhalations: Inhalation[] = (report?.inhalations ?? []) as Inhalation[];

    // Compute y-range
    let ymin = Infinity;
    let ymax = -Infinity;
    for (const v of ds.y) {
      if (v < ymin) ymin = v;
      if (v > ymax) ymax = v;
    }
    if (!isFinite(ymin) || !isFinite(ymax) || ymin === ymax) {
      ymin = -1;
      ymax = 1;
    }
    // pad
    const pad = (ymax - ymin) * 0.08;
    ymin -= pad;
    ymax += pad;

    return { t: ds.x, resp: ds.y, peaks, inhalations, ymin, ymax };
  }, [report]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Handle HiDPI
    const cssW = canvas.clientWidth;
    const cssH = canvas.clientHeight;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(cssW * dpr));
    canvas.height = Math.max(1, Math.floor(cssH * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Layout
    const W = cssW;
    const H = cssH;
    const margin = { l: 44, r: 16, t: 14, b: 26 };
    const pw = W - margin.l - margin.r;
    const ph = H - margin.t - margin.b;

    ctx.clearRect(0, 0, W, H);

    const tArr = data.t;
    const yArr = data.resp;
    if (!tArr.length || !yArr.length) {
      // no data
      ctx.font = "14px system-ui";
      ctx.fillText("No waveform data available.", 12, 24);
      return;
    }

    const t0 = tArr[0];
    const t1 = tArr[tArr.length - 1];
    const y0 = data.ymin;
    const y1 = data.ymax;

    const xOfT = (t: number) => margin.l + ((t - t0) / (t1 - t0)) * pw;
    const yOf = (v: number) => margin.t + (1 - (v - y0) / (y1 - y0)) * ph;

    // Background
    ctx.fillStyle = "rgba(0,0,0,0)";
    ctx.fillRect(0, 0, W, H);

    // Inhalation spans
    ctx.fillStyle = "rgba(0,0,0,0.08)";
    for (const inh of data.inhalations) {
      if (typeof inh?.t_start !== "number" || typeof inh?.t_peak !== "number") continue;
      const xs = xOfT(inh.t_start);
      const xe = xOfT(inh.t_peak);
      ctx.fillRect(Math.min(xs, xe), margin.t, Math.abs(xe - xs), ph);
    }

    // Grid (light)
    ctx.strokeStyle = "rgba(0,0,0,0.12)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i <= 4; i++) {
      const yy = margin.t + (ph * i) / 4;
      ctx.moveTo(margin.l, yy);
      ctx.lineTo(margin.l + pw, yy);
    }
    ctx.stroke();

    // Waveform
    ctx.strokeStyle = "rgba(0,0,0,0.85)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(xOfT(tArr[0]), yOf(yArr[0]));
    for (let i = 1; i < tArr.length; i++) {
      ctx.lineTo(xOfT(tArr[i]), yOf(yArr[i]));
    }
    ctx.stroke();

    // Peaks (dots)
    ctx.fillStyle = "rgba(0,0,0,0.85)";
    const maxPeakDots = 80;
    const peaks = data.peaks.slice(0, maxPeakDots);
    for (const p of peaks) {
      const tt = Number(p.time_s);
      if (!isFinite(tt)) continue;
      const x = xOfT(tt);
      // find nearest y
      let bestI = 0;
      let bestD = Infinity;
      // linear scan over downsampled points (ok at <=1400)
      for (let i = 0; i < tArr.length; i++) {
        const d = Math.abs(tArr[i] - tt);
        if (d < bestD) {
          bestD = d;
          bestI = i;
        }
      }
      const y = yOf(yArr[bestI]);
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();
    }

    // Playhead (currentTime)
    if (isFinite(currentTime)) {
      const x = xOfT(Math.min(Math.max(currentTime, t0), t1));
      ctx.strokeStyle = "rgba(0,0,0,0.9)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, margin.t);
      ctx.lineTo(x, margin.t + ph);
      ctx.stroke();
    }

    // Axes labels
    ctx.fillStyle = "rgba(0,0,0,0.7)";
    ctx.font = "12px system-ui";
    ctx.fillText("Resp (a.u.)", 8, margin.t + 12);
    ctx.fillText("Time (s)", margin.l + pw / 2 - 18, H - 6);
  }, [data, currentTime]);

  return (
    <canvas
      ref={canvasRef}
      className="w-full h-[220px] rounded-lg border bg-background"
    />
  );
}
