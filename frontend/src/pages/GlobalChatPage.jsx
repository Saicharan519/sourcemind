import { useState, useRef, useEffect } from "react";
import { motion } from "framer-motion";
import { useQuery } from "@tanstack/react-query";
import { listSources } from "../api/client";
import { useStreamingChat } from "../hooks/useStreamingChat.js";
import ChatMessage from "../components/ChatMessage.jsx";

export default function GlobalChatPage() {
  const [input, setInput] = useState("");
  const bottomRef = useRef(null);

  const { data: sources = [] } = useQuery({
    queryKey: ["sources"],
    queryFn: listSources,
  });
  const readyCount = sources.filter((s) => s.status === "ready").length;

  const { messages, isStreaming, send, clear } = useStreamingChat({
    mode: "global",
  });

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = () => {
    if (!input.trim() || isStreaming) return;
    send(input.trim());
    setInput("");
  };

  return (
    <div className="mx-auto max-w-3xl space-y-5">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        className="rounded-2xl border border-line bg-paper-raised p-6 shadow-card"
      >
        <p className="eyebrow text-emerald">Cross-source</p>
        <h1 className="font-display mt-1 text-3xl font-semibold text-ink">
          Global chat
        </h1>
        <p className="mt-2 text-ink-soft">
          One question, answered across{" "}
          <span className="font-semibold text-ink">
            {readyCount} source{readyCount !== 1 ? "s" : ""}
          </span>{" "}
          at once — citations name their origin.
        </p>
      </motion.div>

      <div className="min-h-[42vh] space-y-4">
        {messages.length === 0 && (
          <div className="rounded-2xl border border-dashed border-line-strong bg-paper-raised/40 py-16 text-center">
            <p className="font-display text-lg italic text-ink-soft">
              Ask a question that spans your entire library.
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
          placeholder="Ask across all sources…"
          disabled={isStreaming}
          className="flex-1 rounded-xl bg-transparent px-4 py-2.5 text-ink placeholder:text-ink-faint focus:outline-none"
        />
        <motion.button
          whileTap={{ scale: 0.95 }}
          onClick={handleSend}
          disabled={isStreaming || !input.trim()}
          className="rounded-xl bg-emerald px-6 py-2.5 font-semibold text-paper transition-colors hover:bg-emerald-deep disabled:opacity-40"
        >
          {isStreaming ? "…" : "Send"}
        </motion.button>
        {messages.length > 0 && (
          <button
            onClick={clear}
            className="px-2 text-xs text-ink-faint hover:text-coral"
          >
            clear
          </button>
        )}
      </div>
    </div>
  );
}
