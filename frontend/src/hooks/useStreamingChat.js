import { useCallback, useRef, useState } from "react";
import { buildChatStreamUrl, buildGlobalChatStreamUrl } from "../api/client";

/**
 * Hook for SSE-based streaming chat.
 *
 * Returns: { messages, isStreaming, send, cancel }
 *
 * `mode`: "source" | "global"
 * `sourceId`: required when mode === "source"
 */
export function useStreamingChat({ mode = "source", sourceId = null } = {}) {
  const [messages, setMessages] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const esRef = useRef(null);

  const cancel = useCallback(() => {
    if (esRef.current) {
      esRef.current.close();
      esRef.current = null;
    }
    setIsStreaming(false);
  }, []);

  const send = useCallback(
    (question) => {
      if (!question.trim() || isStreaming) return;
      if (mode === "source" && !sourceId) return;

      // append user msg
      const userMsg = { role: "user", content: question, id: Date.now() };
      setMessages((prev) => [...prev, userMsg]);

      // create placeholder assistant msg
      const asstId = Date.now() + 1;
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "",
          id: asstId,
          citations: [],
          queryType: null,
          subQueries: null,
        },
      ]);

      const history = messages
        .filter((m) => m.role === "user" || m.role === "assistant")
        .map((m) => ({ role: m.role, content: m.content }));

      const url =
        mode === "source"
          ? buildChatStreamUrl({ sourceId, question, history })
          : buildGlobalChatStreamUrl({ question, history });

      const es = new EventSource(url);
      esRef.current = es;
      setIsStreaming(true);

      es.onmessage = (event) => {
        let data;
        try {
          data = JSON.parse(event.data);
        } catch {
          return;
        }

        if (data.type === "token") {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === asstId ? { ...m, content: m.content + data.value } : m
            )
          );
        } else if (data.type === "citations") {
          setMessages((prev) =>
            prev.map((m) => (m.id === asstId ? { ...m, citations: data.value } : m))
          );
        } else if (data.type === "query_type") {
          setMessages((prev) =>
            prev.map((m) => (m.id === asstId ? { ...m, queryType: data.value } : m))
          );
        } else if (data.type === "sub_queries") {
          setMessages((prev) =>
            prev.map((m) => (m.id === asstId ? { ...m, subQueries: data.value } : m))
          );
        } else if (data.type === "error") {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === asstId
                ? { ...m, content: m.content + `\n\n[error: ${data.value}]` }
                : m
            )
          );
        } else if (data.type === "done") {
          es.close();
          esRef.current = null;
          setIsStreaming(false);
        }
      };

      es.onerror = () => {
        es.close();
        esRef.current = null;
        setIsStreaming(false);
      };
    },
    [mode, sourceId, isStreaming, messages]
  );

  const clear = useCallback(() => {
    cancel();
    setMessages([]);
  }, [cancel]);

  return { messages, isStreaming, send, cancel, clear };
}
