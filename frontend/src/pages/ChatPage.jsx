import { useState } from "react";
import { useParams, Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { getSource } from "../api/client";
import { useStreamingChat } from "../hooks/useStreamingChat.js";
import ChatMessage from "../components/ChatMessage.jsx";
import EvalScoreCard from "../components/EvalScoreCard.jsx";

export default function ChatPage() {
  const { sourceId } = useParams();
  const [input, setInput] = useState("");

  const { data: source } = useQuery({
    queryKey: ["source", sourceId],
    queryFn: () => getSource(sourceId),
    enabled: !!sourceId,
  });

  const { messages, isStreaming, send, clear } = useStreamingChat({
    mode: "source",
    sourceId,
  });

  const handleSend = () => {
    if (!input.trim() || isStreaming) return;
    send(input.trim());
    setInput("");
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <Link to="/" className="text-sm text-muted hover:text-white">
          ← back
        </Link>
        <button
          onClick={clear}
          className="text-xs text-muted hover:text-red-400"
        >
          clear chat
        </button>
      </div>

      {source && (
        <div className="bg-surface border border-border rounded-xl p-4">
          <div className="flex items-center gap-2 text-sm">
            <span>{source.source_type === "video" ? "🎬" : "📄"}</span>
            <span className="font-medium">{source.title}</span>
          </div>
          {source.eval_results && <EvalScoreCard scores={source.eval_results} />}
        </div>
      )}

      <div className="bg-surface border border-border rounded-xl p-4 min-h-[400px] max-h-[60vh] overflow-y-auto space-y-3">
        {messages.length === 0 && (
          <p className="text-center text-muted text-sm py-10">
            Ask anything about this source.
          </p>
        )}
        {messages.map((m) => (
          <ChatMessage key={m.id} msg={m} />
        ))}
      </div>

      <div className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          placeholder="Ask a question…"
          disabled={isStreaming}
          className="flex-1 bg-surface-2 border border-border rounded-md px-4 py-2 focus:outline-none focus:border-accent"
        />
        <button
          onClick={handleSend}
          disabled={isStreaming || !input.trim()}
          className="bg-accent hover:bg-accent-glow text-white font-medium px-5 py-2 rounded-md disabled:opacity-40"
        >
          {isStreaming ? "…" : "Send"}
        </button>
      </div>
    </div>
  );
}
