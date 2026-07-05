import { useState, useRef, useEffect } from "react";
import { useParams, Link } from "react-router-dom";
import { motion } from "framer-motion";
import { useQuery } from "@tanstack/react-query";
import { getSource } from "../api/client";
import { useStreamingChat } from "../hooks/useStreamingChat.js";
import ChatMessage from "../components/ChatMessage.jsx";
import EvalScoreCard from "../components/EvalScoreCard.jsx";

export default function ChatPage() {
  const { sourceId } = useParams();
  const [input, setInput] = useState("");
  const bottomRef = useRef(null);

  const { data: source } = useQuery({
    queryKey: ["source", sourceId],
    queryFn: () => getSource(sourceId),
    enabled: !!sourceId,
  });

  const { messages, isStreaming, send, clear } = useStreamingChat({
    mode: "source",
    sourceId,
  });

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const notReady = source && source.status !== "ready";

  const handleSend = () => {
    if (!input.trim() || isStreaming || notReady) return;
    send(input.trim());
    setInput("");
  };

  return (
    <div className="mx-auto max-w-3xl space-y-5">
      <div className="flex items-center justify-between">
        <Link
          to="/"
          className="text-sm text-ink-soft transition-colors hover:text-emerald"
        >
          ← Library
        </Link>
        {messages.length > 0 && (
          <button
            onClick={clear}
            className="text-xs text-ink-faint transition-colors hover:text-coral"
          >
            clear conversation
          </button>
        )}
      </div>

      {source && (
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          className="rounded-2xl border border-line bg-paper-raised p-5 shadow-card"
        >
          <div className="flex items-center gap-3">
            <span className="grid h-10 w-10 place-items-center rounded-xl bg-paper-sunken text-lg">
              {source.source_type === "video" ? "▶" : "❧"}
            </span>
            <div>
              <p className="eyebrow text-ink-faint">
                {source.source_type === "video" ? "Video" : "Document"}
              </p>
              <h1 className="font-display text-xl font-semibold leading-tight text-ink">
                {source.title}
              </h1>
            </div>
          </div>
          {source.eval_results && <EvalScoreCard scores={source.eval_results} />}
        </motion.div>
      )}

      {notReady && (
        <div
          className={`rounded-xl border p-3 text-sm ${
            source.status === "failed"
              ? "border-coral/40 bg-coral-soft text-coral"
              : "border-gold/40 bg-gold/10 text-gold"
          }`}
        >
          {source.status === "failed"
            ? "Ingestion failed for this source — you can't chat with it."
            : "This source is still processing. Chat will be available once it's ready."}
        </div>
      )}

      <div className="min-h-[42vh] space-y-4">
        {messages.length === 0 && (
          <div className="rounded-2xl border border-dashed border-line-strong bg-paper-raised/40 py-16 text-center">
            <p className="font-display text-lg italic text-ink-soft">
              Ask anything about this source.
            </p>
            <p className="mt-1 text-sm text-ink-faint">
              Answers arrive with page & timestamp citations.
            </p>
          </div>
        )}
        {messages.map((m) => (
          <ChatMessage key={m.id} msg={m} />
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="sticky bottom-4 flex gap-2 rounded-2xl border border-line bg-paper-raised/90 p-2 shadow-card backdrop-blur">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          placeholder={notReady ? "Source not ready…" : "Ask a question…"}
          disabled={isStreaming || notReady}
          className="flex-1 rounded-xl bg-transparent px-4 py-2.5 text-ink placeholder:text-ink-faint focus:outline-none disabled:opacity-50"
        />
        <motion.button
          whileTap={{ scale: 0.95 }}
          onClick={handleSend}
          disabled={isStreaming || notReady || !input.trim()}
          className="rounded-xl bg-emerald px-6 py-2.5 font-semibold text-paper transition-colors hover:bg-emerald-deep disabled:opacity-40"
        >
          {isStreaming ? "…" : "Send"}
        </motion.button>
      </div>
    </div>
  );
}
