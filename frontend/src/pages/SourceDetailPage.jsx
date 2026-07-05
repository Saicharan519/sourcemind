import { useState } from "react";
import { useParams, Link } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
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
      setTimeout(
        () => qc.invalidateQueries({ queryKey: ["source", sourceId] }),
        5000
      );
    },
  });

  if (isLoading || !source) {
    return <p className="font-display text-lg italic text-ink-soft">Loading…</p>;
  }

  const evalResults = source.eval_results;

  return (
    <div className="mx-auto max-w-3xl space-y-5">
      <div className="flex items-center justify-between">
        <Link
          to="/"
          className="text-sm text-ink-soft transition-colors hover:text-emerald"
        >
          ← Library
        </Link>
        <Link
          to={`/chat/${sourceId}`}
          className="rounded-lg bg-emerald px-4 py-1.5 text-sm font-semibold text-paper transition-colors hover:bg-emerald-deep"
        >
          Open chat →
        </Link>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        className="rounded-2xl border border-line bg-paper-raised p-6 shadow-card"
      >
        <div className="flex items-center gap-4">
          <span className="grid h-12 w-12 place-items-center rounded-xl bg-paper-sunken text-xl">
            {source.source_type === "video" ? "▶" : "❧"}
          </span>
          <div>
            <h1 className="font-display text-2xl font-semibold text-ink">
              {source.title}
            </h1>
            <div className="mt-1 flex flex-wrap gap-3 font-mono text-xs text-ink-faint">
              <span>status: {source.status}</span>
              {source.page_count && <span>{source.page_count} pages</span>}
              {source.duration_s && (
                <span>{Math.round(source.duration_s / 60)} min</span>
              )}
            </div>
          </div>
        </div>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.08 }}
        className="rounded-2xl border border-line bg-paper-raised p-6 shadow-card"
      >
        <div className="flex items-center justify-between">
          <h2 className="font-display text-lg font-semibold text-ink">
            Retrieval quality
          </h2>
          <motion.button
            whileTap={{ scale: 0.96 }}
            onClick={() => evalMut.mutate()}
            disabled={evalMut.isPending}
            className="rounded-lg border border-line px-3 py-1.5 text-xs text-ink-soft transition-colors hover:border-ink hover:text-ink disabled:opacity-50"
          >
            {evalMut.isPending ? "Running…" : "Re-run evaluation"}
          </motion.button>
        </div>

        {evalResults ? (
          <>
            <EvalScoreCard scores={evalResults} />

            {evalResults.eval_questions?.length > 0 && (
              <div className="mt-6">
                <p className="eyebrow mb-2 text-ink-faint">
                  Test questions ({evalResults.eval_questions.length})
                </p>
                <ul className="space-y-2">
                  {evalResults.eval_questions.map((q, i) => (
                    <li
                      key={i}
                      className="overflow-hidden rounded-xl border border-line bg-paper"
                    >
                      <button
                        onClick={() => setExpandedQ(expandedQ === i ? null : i)}
                        className="flex w-full items-center justify-between gap-3 px-4 py-3 text-left text-sm text-ink transition-colors hover:bg-paper-sunken/50"
                      >
                        <span>{q.question}</span>
                        <motion.span
                          animate={{ rotate: expandedQ === i ? 90 : 0 }}
                          className="text-ink-faint"
                        >
                          ▸
                        </motion.span>
                      </button>
                      <AnimatePresence>
                        {expandedQ === i && (
                          <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: "auto", opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            className="overflow-hidden border-t border-line"
                          >
                            <div className="space-y-3 px-4 py-3 text-xs">
                              <div>
                                <p className="eyebrow text-ink-faint">
                                  Ground truth
                                </p>
                                <p className="mt-1 font-display italic text-ink-soft">
                                  {q.ground_truth}
                                </p>
                              </div>
                              <div>
                                <p className="eyebrow text-ink-faint">
                                  Generated answer
                                </p>
                                <p className="mt-1 leading-relaxed text-ink">
                                  {q.generated_answer}
                                </p>
                              </div>
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </>
        ) : (
          <p className="mt-3 text-sm text-ink-soft">
            No evaluation yet. Click “Re-run evaluation” to score this source.
          </p>
        )}
      </motion.div>
    </div>
  );
}
