import { AnalysisSummary } from "@/types/analysis";
import { Card, CardContent } from "@/components/ui/card";
import { TrendingUp, Activity, Gauge, AlertTriangle } from "lucide-react";

interface SummaryCardsProps {
  summary: AnalysisSummary;
}

export function SummaryCards({ summary }: SummaryCardsProps) {
  const stats = [
    {
      label: "Average Score",
      value: summary.avgScore,
      icon: TrendingUp,
      suffix: "",
    },
    {
      label: "Consistency",
      value: Math.round(summary.stability * 100),
      icon: Gauge,
      suffix: "%",
    },
    {
      label: "Sudden Movements",
      value: summary.spikeCount,
      icon: AlertTriangle,
      suffix: "",
      warning: summary.spikeCount > 5,
    },
  ];

  return (
    <div className="grid grid-cols-3 gap-3">
      {stats.map((stat) => (
        <Card key={stat.label} className="bg-muted/30">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <stat.icon
                className={`w-4 h-4 ${
                  stat.warning ? "text-destructive" : "text-muted-foreground"
                }`}
              />
              <span className="text-xs text-muted-foreground">
                {stat.label}
              </span>
            </div>
            <p
              className={`text-2xl font-bold ${
                stat.warning ? "text-destructive" : ""
              }`}
            >
              {stat.value}
              {stat.suffix}
            </p>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
