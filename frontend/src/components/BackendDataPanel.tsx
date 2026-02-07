import { useMemo, useState } from "react";
import { Copy, Download, ChevronDown, ChevronUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { AnalysisResult } from "@/types/analysis";

interface BackendDataPanelProps {
  result: AnalysisResult;
  storageKey?: string;
}

export function BackendDataPanel({
  result,
  storageKey = "breathmap:lastResult",
}: BackendDataPanelProps) {
  const [open, setOpen] = useState(false);
  const jsonText = useMemo(() => JSON.stringify(result, null, 2), [result]);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(jsonText);
    } catch {
      // If clipboard is blocked, do nothing.
    }
  };

  const handleDownload = () => {
    const blob = new Blob([jsonText], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "analysis_result.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleSaveAgain = () => {
    try {
      localStorage.setItem(storageKey, jsonText);
    } catch {
      // ignore storage errors
    }
  };

  return (
    <Card className="border rounded-xl overflow-hidden">
      <div className="p-4 flex items-center justify-between gap-3">
        <div>
          <div className="font-medium">Backend result (JSON)</div>
          <div className="text-sm text-muted-foreground">
            Stored in localStorage as <span className="font-mono">{storageKey}</span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={handleCopy}>
            <Copy className="w-4 h-4 mr-2" />
            Copy
          </Button>
          <Button variant="outline" size="sm" onClick={handleDownload}>
            <Download className="w-4 h-4 mr-2" />
            Download
          </Button>
          <Button variant="ghost" size="sm" onClick={() => setOpen((v) => !v)}>
            {open ? (
              <>
                <ChevronUp className="w-4 h-4 mr-2" />
                Hide
              </>
            ) : (
              <>
                <ChevronDown className="w-4 h-4 mr-2" />
                Show
              </>
            )}
          </Button>
        </div>
      </div>

      {open && (
        <div className="border-t">
          <div className="p-3 flex items-center justify-end">
            <Button variant="ghost" size="sm" onClick={handleSaveAgain}>
              Save to localStorage
            </Button>
          </div>
          <pre className="p-4 text-xs overflow-auto max-h-[420px] bg-muted/30">
{jsonText}
          </pre>
        </div>
      )}
    </Card>
  );
}
