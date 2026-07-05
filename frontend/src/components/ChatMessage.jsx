import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import ReactMarkdown from "react-markdown";

export default function ChatMessage({ msg }) {
  const [showCitations, setShowCitations] = useState(false);
  const [copied, setCopied] = useState(false);

  if (msg.role === "user") {
    return (
      <motion.div
        initial={{ opacity: 0, y: 10, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ type: "spring", stiffness: 300, damping: 24 }}
        className="flex justify-end"
      >
        <div className="max-w-[80%] rounded-2xl rounded-br-md bg-ink px-4 py-2.5 text-paper shadow-card">
          <p className="whitespace-pre-wrap text-[15px] leading-relaxed">
            {msg.content}
          </p>
        </div>
      </motion.div>
    );
  }

  const copy = () => {
    navigator.clipboard?.writeText(msg.content || "");
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
      className="flex justify-start"
    >
      <div className="max-w-[88%] space-y-3 rounded-2xl rounded-bl-md border border-line bg-paper-raised px-5 py-4 shadow-card">
        <div className="flex items-center gap-2">
          {msg.queryType && (
            <span className="eyebrow rounded-full border border-emerald/30 bg-emerald-soft px-2 py-0.5 text-[10px] text-emerald-deep">
              {msg.queryType.replace("_", "-")}
            </span>
          )}
          {msg.content && (
            <button
              onClick={copy}
              className="ml-auto text-xs text-ink-faint transition-colors hover:text-emerald"
            >
              {copied ? "copied ✓" : "copy"}
            </button>
          )}
        </div>

        {msg.content ? (
          <div className="prose-answer text-[15px]">
            <ReactMarkdown>{msg.content}</ReactMarkdown>
          </div>
        ) : (
          <ThinkingDots />
        )}

        {msg.subQueries && msg.subQueries.length > 0 && (
          <details className="text-xs text-ink-soft">
            <summary className="cursor-pointer select-none font-medium hover:text-ink">
              sub-queries
            </summary>
            <ul className="mt-2 space-y-1 border-l-2 border-line pl-3">
              {msg.subQueries.map((q, i) => (
                <li key={i} className="italic">
                  {q}
                </li>
              ))}
            </ul>
          </details>
        )}

        {msg.citations && msg.citations.length > 0 && (
          <div className="border-t border-line pt-2">
            <button
              onClick={() => setShowCitations((v) => !v)}
              className="flex items-center gap-1.5 text-xs font-medium text-ink-soft transition-colors hover:text-emerald"
            >
              <motion.span animate={{ rotate: showCitations ? 90 : 0 }}>
                ▸
              </motion.span>
              {msg.citations.length} citation
              {msg.citations.length !== 1 ? "s" : ""}
            </button>
            <AnimatePresence>
              {showCitations && (
                <motion.ul
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  className="mt-3 space-y-3 overflow-hidden"
                >
                  {msg.citations.map((c, i) => (
                    <li key={i} className="flex gap-3">
                      <span className="mt-0.5 font-mono text-xs text-emerald">
                        [{i + 1}]
                      </span>
                      <div className="min-w-0">
                        <div className="text-xs font-semibold text-ink">
                          {c.type === "document"
                            ? `${c.source_title || "Document"} · page ${c.page_number ?? "?"}`
                            : c.type === "video"
                            ? `${c.source_title || "Video"} · ${c.segment}`
                            : "Source"}
                        </div>
                        <div className="mt-1 border-l-2 border-line pl-3 font-display text-sm italic leading-snug text-ink-soft">
                          {c.excerpt}
                        </div>
                      </div>
                    </li>
                  ))}
                </motion.ul>
              )}
            </AnimatePresence>
          </div>
        )}
      </div>
    </motion.div>
  );
}

function ThinkingDots() {
  return (
    <div className="flex items-center gap-1.5 py-1">
      {[0, 1, 2].map((i) => (
        <motion.span
          key={i}
          className="h-2 w-2 rounded-full bg-emerald"
          animate={{ y: [0, -5, 0], opacity: [0.4, 1, 0.4] }}
          transition={{ duration: 0.9, repeat: Infinity, delay: i * 0.15 }}
        />
      ))}
    </div>
  );
}
