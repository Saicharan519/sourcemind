export default function EvalScoreCard({ scores }) {
  const metrics = [
    { key: "faithfulness", label: "Faithfulness" },
    { key: "answer_relevancy", label: "Answer Relevancy" },
    { key: "context_recall", label: "Context Recall" },
    { key: "context_precision", label: "Context Precision" },
  ];

  return (
    <div className="mt-3 pt-3 border-t border-border space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs uppercase tracking-wider text-muted">
          RAGAS Evaluation
        </span>
        {typeof scores.overall_score === "number" && (
          <span className="text-sm font-bold text-cyan">
            Overall: {(scores.overall_score * 100).toFixed(1)}%
          </span>
        )}
      </div>
      <div className="grid grid-cols-2 gap-2">
        {metrics.map((m) => {
          const v = scores[m.key];
          const pct = typeof v === "number" ? Math.max(0, Math.min(1, v)) * 100 : 0;
          return (
            <div key={m.key}>
              <div className="flex justify-between text-[11px] text-muted">
                <span>{m.label}</span>
                <span className="text-cyan font-mono">
                  {typeof v === "number" ? pct.toFixed(0) + "%" : "—"}
                </span>
              </div>
              <div className="h-1.5 bg-surface-2 rounded-full mt-1 overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-accent to-cyan rounded-full transition-all"
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
