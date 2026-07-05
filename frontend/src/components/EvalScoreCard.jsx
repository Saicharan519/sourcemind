import { motion } from "framer-motion";

export default function EvalScoreCard({ scores }) {
  const metrics = [
    { key: "faithfulness", label: "Faithfulness" },
    { key: "answer_relevancy", label: "Answer Relevancy" },
    { key: "context_recall", label: "Context Recall" },
    { key: "context_precision", label: "Context Precision" },
  ];

  return (
    <div className="mt-4 border-t border-line pt-4">
      <div className="flex items-baseline justify-between">
        <span className="eyebrow text-ink-faint">RAGAS Evaluation</span>
        {typeof scores.overall_score === "number" && (
          <span className="font-display text-xl font-semibold text-emerald">
            {(scores.overall_score * 100).toFixed(1)}%
          </span>
        )}
      </div>
      <div className="mt-3 grid grid-cols-1 gap-x-8 gap-y-3 sm:grid-cols-2">
        {metrics.map((m, i) => {
          const v = scores[m.key];
          const pct =
            typeof v === "number" ? Math.max(0, Math.min(1, v)) * 100 : 0;
          return (
            <div key={m.key}>
              <div className="flex justify-between text-xs">
                <span className="text-ink-soft">{m.label}</span>
                <span className="font-mono text-emerald">
                  {typeof v === "number" ? pct.toFixed(0) + "%" : "—"}
                </span>
              </div>
              <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-paper-sunken">
                <motion.div
                  className="h-full rounded-full bg-gradient-to-r from-emerald to-gold"
                  initial={{ width: 0 }}
                  animate={{ width: `${pct}%` }}
                  transition={{ duration: 0.9, delay: 0.1 + i * 0.08, ease: "easeOut" }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
