import { Card, CardContent } from "@/components/ui/card";

interface ScoreDisplayProps {
  score: number;
  explanation: string;
}

export function ScoreDisplay({ score, explanation }: ScoreDisplayProps) {
  const getScoreColor = (score: number) => {
    if (score >= 80) return "text-green-500";
    if (score >= 60) return "text-yellow-500";
    return "text-red-500";
  };

  const getScoreLabel = (score: number) => {
    if (score >= 90) return "Excellent";
    if (score >= 80) return "Great";
    if (score >= 70) return "Good";
    if (score >= 60) return "Fair";
    return "Needs Work";
  };

  return (
    <Card className="bg-gradient-to-br from-primary/5 to-primary/10 border-primary/20">
      <CardContent className="p-6 text-center">
        <p className="text-sm text-muted-foreground mb-2">
          Video Breath Control Score
        </p>
        <div className="flex items-baseline justify-center gap-2 mb-2">
          <span className={`text-6xl font-bold ${getScoreColor(score)}`}>
            {score}
          </span>
          <span className="text-2xl text-muted-foreground">/ 100</span>
        </div>
        <p className={`text-lg font-medium ${getScoreColor(score)} mb-3`}>
          {getScoreLabel(score)}
        </p>
        <p className="text-sm text-muted-foreground max-w-sm mx-auto">
          {explanation}
        </p>
      </CardContent>
    </Card>
  );
}
