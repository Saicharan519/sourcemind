import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  deleteSource,
  ingestDocument,
  ingestVideo,
  listSources,
} from "../api/client";

export default function Dashboard() {
  const nav = useNavigate();
  const qc = useQueryClient();
  const [url, setUrl] = useState("");

  const { data: sources = [], refetch } = useQuery({
    queryKey: ["sources"],
    queryFn: listSources,
    refetchInterval: (q) =>
      // poll while anything is processing
      q.state.data?.some?.((s) => s.status === "processing") ? 4000 : false,
  });

  const docMut = useMutation({
    mutationFn: ingestDocument,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["sources"] }),
  });

  const vidMut = useMutation({
    mutationFn: ingestVideo,
    onSuccess: () => {
      setUrl("");
      qc.invalidateQueries({ queryKey: ["sources"] });
    },
  });

  const delMut = useMutation({
    mutationFn: deleteSource,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["sources"] }),
  });

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[1fr_2fr] gap-6">
      {/* Upload column */}
      <section className="space-y-4">
        <div className="bg-surface border border-border rounded-xl p-5">
          <h2 className="text-sm font-semibold tracking-wider text-muted uppercase mb-3">
            Upload PDF
          </h2>
          <input
            type="file"
            accept="application/pdf"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) docMut.mutate(f);
            }}
            disabled={docMut.isPending}
            className="block w-full text-sm text-muted file:mr-3 file:rounded-md file:border-0 file:bg-accent file:px-3 file:py-2 file:text-white hover:file:bg-accent-glow"
          />
          {docMut.isPending && <p className="mt-2 text-xs text-cyan">Uploading…</p>}
          {docMut.isError && (
            <p className="mt-2 text-xs text-red-400">
              {docMut.error.response?.data?.detail || "Upload failed"}
            </p>
          )}
        </div>

        <div className="bg-surface border border-border rounded-xl p-5">
          <h2 className="text-sm font-semibold tracking-wider text-muted uppercase mb-3">
            YouTube URL
          </h2>
          <div className="flex gap-2">
            <input
              type="url"
              placeholder="https://youtube.com/watch?v=..."
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              className="flex-1 bg-surface-2 border border-border rounded-md px-3 py-2 text-sm font-mono focus:outline-none focus:border-accent"
            />
            <button
              onClick={() => url.trim() && vidMut.mutate(url.trim())}
              disabled={vidMut.isPending || !url.trim()}
              className="bg-accent hover:bg-accent-glow text-white text-sm font-medium px-4 py-2 rounded-md disabled:opacity-50"
            >
              {vidMut.isPending ? "Adding…" : "Add"}
            </button>
          </div>
        </div>
      </section>

      {/* Sources list */}
      <section className="space-y-3">
        <h2 className="text-sm font-semibold tracking-wider text-muted uppercase">
          Ingested Sources ({sources.length})
        </h2>
        {sources.length === 0 ? (
          <div className="bg-surface border border-border rounded-xl p-10 text-center text-muted">
            No sources yet. Upload a PDF or paste a YouTube URL to get started.
          </div>
        ) : (
          <ul className="space-y-2">
            {sources.map((s) => (
              <li
                key={s.id}
                className="bg-surface border border-border rounded-xl p-4 flex items-center justify-between hover:border-accent transition"
              >
                <div className="flex items-center gap-3">
                  <span className="text-lg">
                    {s.source_type === "video" ? "🎬" : "📄"}
                  </span>
                  <div>
                    <div className="font-medium">{s.title}</div>
                    <div className="text-xs text-muted flex gap-2 mt-1">
                      <StatusPill status={s.status} />
                      {typeof s.eval_score === "number" && (
                        <span className="text-cyan">
                          RAG score: {(s.eval_score * 100).toFixed(0)}%
                        </span>
                      )}
                    </div>
                  </div>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => nav(`/source/${s.id}`)}
                    className="text-sm bg-surface-2 hover:bg-border text-muted hover:text-white border border-border px-3 py-1.5 rounded-md"
                  >
                    Details
                  </button>
                  <button
                    onClick={() => nav(`/chat/${s.id}`)}
                    disabled={s.status !== "ready"}
                    className="text-sm bg-accent hover:bg-accent-glow text-white px-3 py-1.5 rounded-md disabled:opacity-30"
                  >
                    Chat →
                  </button>
                  <button
                    onClick={() => {
                      if (confirm("Delete this source?")) delMut.mutate(s.id);
                    }}
                    className="text-sm text-muted hover:text-red-400 px-2"
                  >
                    🗑
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

function StatusPill({ status }) {
  const styles = {
    processing: "bg-yellow-500/10 text-yellow-400 border-yellow-500/30",
    ready: "bg-green-500/10 text-green-400 border-green-500/30",
    failed: "bg-red-500/10 text-red-400 border-red-500/30",
  };
  return (
    <span
      className={`text-[10px] uppercase tracking-wider border rounded-full px-2 py-0.5 ${
        styles[status] || styles.processing
      }`}
    >
      {status}
    </span>
  );
}
