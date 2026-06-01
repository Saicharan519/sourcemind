import { useState } from "react";

export default function ChatMessage({ msg }) {
  const [showCitations, setShowCitations] = useState(false);

  if (msg.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] bg-accent/20 border border-accent/40 rounded-2xl rounded-br-sm px-4 py-2">
          <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-[85%] bg-cyan/10 border border-cyan/30 rounded-2xl rounded-bl-sm px-4 py-2 space-y-2">
        {msg.queryType && (
          <span className="inline-block text-[10px] uppercase tracking-wider bg-cyan/20 text-cyan border border-cyan/40 rounded-full px-2 py-0.5">
            {msg.queryType.replace("_", "-")}
          </span>
        )}
        <p className="text-sm whitespace-pre-wrap font-mono leading-relaxed">
          {msg.content || <span className="text-muted">▌</span>}
        </p>

        {msg.subQueries && msg.subQueries.length > 0 && (
          <details className="text-xs text-muted">
            <summary className="cursor-pointer">sub-queries</summary>
            <ul className="mt-1 space-y-0.5 pl-4 list-disc">
              {msg.subQueries.map((q, i) => (
                <li key={i}>{q}</li>
              ))}
            </ul>
          </details>
        )}

        {msg.citations && msg.citations.length > 0 && (
          <div>
            <button
              onClick={() => setShowCitations((v) => !v)}
              className="text-xs text-muted hover:text-cyan"
            >
              {showCitations ? "▼" : "▶"} {msg.citations.length} citation
              {msg.citations.length !== 1 ? "s" : ""}
            </button>
            {showCitations && (
              <ul className="mt-2 space-y-2 text-xs">
                {msg.citations.map((c, i) => (
                  <li
                    key={i}
                    className="border-l-2 border-cyan/40 pl-3 text-muted"
                  >
                    <div className="text-cyan">
                      {c.type === "document"
                        ? `📄 ${c.source_title || "Document"} — page ${c.page_number ?? "?"}`
                        : c.type === "video"
                        ? `🎬 ${c.source_title || "Video"} — ${c.segment}`
                        : "Source"}
                    </div>
                    <div className="font-mono mt-1">{c.excerpt}</div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
