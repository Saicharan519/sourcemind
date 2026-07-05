import { useState, useRef, useEffect } from "react";
import { useStreamingChat } from "../hooks/useStreamingChat.js";
import ChatMessage from "../components/ChatMessage.jsx";

export default function GlobalChatPage() {
  const [input, setInput] = useState("");
  const bottomRef = useRef(null);
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
    <div className="space-y-4">
      <div className="bg-surface border border-border rounded-xl p-4">
        <h2 className="text-sm font-semibold tracking-wider text-muted uppercase">
          🌐 Global Chat
        </h2>
        <p className="text-xs text-muted mt-1">
          Queries are answered across <em>all</em> ingested sources at once.
        </p>
      </div>

      <div className="bg-surface border border-border rounded-xl p-4 min-h-[400px] max-h-[60vh] overflow-y-auto space-y-3">
        {messages.length === 0 && (
          <p className="text-center text-muted text-sm py-10">
            Ask a question that spans your entire library.
          </p>
        )}
        {messages.map((m) => (
          <ChatMessage key={m.id} msg={m} />
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          placeholder="Ask across all sources…"
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
        {messages.length > 0 && (
          <button onClick={clear} className="text-xs text-muted px-2">
            clear
          </button>
        )}
      </div>
    </div>
  );
}
