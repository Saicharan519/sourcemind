import { useState } from "react";
import { useParams, Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { getSource, triggerEvaluation } from "../api/client";
import EvalScoreCard from "../components/EvalScoreCard.jsx";

export default function SourceDetailPage() {
  const { sourceId } = useParams();
  const qc = useQueryClient();
  const [expandedQ, setExpandedQ] = useState(null);

  const { data: source, isLoading } = useQuery({
    queryKey: ["source", sourceId],
    queryFn: () => getSource(sourceId),
    enabled: !!sourceId,
  });

  const evalMut = useMutation({
    mutationFn: () => triggerEvaluation(sourceId),
    onSuccess: () => {
      // poll for new results after a delay
      setTimeout(() => qc.invalidateQueries({ queryKey: ["source", sourceId] }), 5000);
    },
  });

  if (isLoading || !source) {
    return <p className="text-muted">Loading…</p>;
  }

  const evalResults = source.eval_results;

  return (
    <div className="space-y-4 max-w-4xl">
      <div className="flex justify-between items-center">
        <Link to="/" className="text-sm text-muted hover:text-white">← back</Link>
        <Link
          to={`/chat/${sourceId}`}
          className="text-sm bg-accent hover:bg-accent-glow text-white px-3 py-1.5 rounded-md"
        >
          Open chat →
        </Link>
      </div>

      <div className="bg-surface border border-border rounded-xl p-5">
        <div className="flex items-center gap-3">
          <span className="text-2xl">{source.source_type === "video" ? "🎬" : "📄"}</span>
          <div>
            <h1 className="text-lg font-bold">{source.title}</h1>
            <div className="text-xs text-muted mt-1 flex gap-3">
              <span>Status: {source.status}</span>
              {source.page_count && <span>{source.page_count} pages</span>}
              {source.duration_s && (
                <span>{Math.round(source.duration_s / 60)} min</span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* RAGAS scores */}
      <div className="bg-surface border border-border rounded-xl p-5">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm uppercase tracking-wider text-muted font-semibold">
            RAGAS Evaluation
          </h2>
          <button
            onClick={() => evalMut.mutate()}
            disabled={evalMut.isPending}
            className="text-xs bg-surface-2 hover:bg-border text-muted hover:text-white border border-border px-3 py-1 rounded-md disabled:opacity-50"
          >
            {evalMut.isPending ? "Running…" : "Re-run evaluation"}
          </button>
        </div>

        {evalResults ? (
          <>
            <EvalScoreCard scores={evalResults} />

            {evalResults.eval_questions && evalResults.eval_questions.length > 0 && (
              <div className="mt-6 space-y-2">
                <h3 className="text-xs uppercase tracking-wider text-muted">
                  Test Questions ({evalResults.eval_questions.length})
                </h3>
                <ul className="space-y-1">
                  {evalResults.eval_questions.map((q, i) => (
                    <li
                      key={i}
                      className="bg-surface-2 border border-border rounded-md overflow-hidden"
                    >
                      <button
                        onClick={() => setExpandedQ(expandedQ === i ? null : i)}
                        className="w-full text-left px-3 py-2 text-sm hover:bg-border/30 flex justify-between items-center"
                      >
                        <span>{q.question}</span>
                        <span className="text-muted text-xs">
                          {expandedQ === i ? "▼" : "▶"}
                        </span>
                      </button>
                      {expandedQ === i && (
                        <div className="px-3 pb-3 text-xs space-y-2 border-t border-border">
                          <div>
                            <div className="text-muted uppercase tracking-wider mt-2">
                              Ground truth
                            </div>
                            <div className="font-mono text-cyan/80">{q.ground_truth}</div>
                          </div>
                          <div>
                            <div className="text-muted uppercase tracking-wider">
                              Generated answer
                            </div>
                            <div className="font-mono">{q.generated_answer}</div>
                          </div>
                        </div>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </>
        ) : (
          <p className="text-sm text-muted">
            No evaluation results yet. Click "Re-run evaluation" above.
          </p>
        )}
      </div>
    </div>
  );
}
