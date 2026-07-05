import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
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
  const [dragging, setDragging] = useState(false);
  const [notice, setNotice] = useState("");

  const { data: sources = [] } = useQuery({
    queryKey: ["sources"],
    queryFn: listSources,
    refetchInterval: (q) =>
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

  const handleFiles = (fileList) => {
    const f = fileList?.[0];
    if (!f) return;
    if (f.type !== "application/pdf" && !f.name.toLowerCase().endsWith(".pdf")) {
      setNotice("Only PDF files are accepted.");
      return;
    }
    setNotice("");
    docMut.mutate(f);
  };

  return (
    <div className="space-y-14">
      {/* Hero */}
      <section className="pt-4">
        <motion.p
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="eyebrow text-emerald"
        >
          Multi-modal knowledge, cited
        </motion.p>
        <motion.h1
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.05, ease: [0.22, 1, 0.36, 1] }}
          className="font-display mt-3 max-w-3xl text-5xl font-semibold leading-[1.05] tracking-tight text-ink md:text-6xl"
        >
          Read, watch, and{" "}
          <span className="italic text-emerald">interrogate</span> your sources.
        </motion.h1>
        <motion.p
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.15 }}
          className="mt-5 max-w-xl text-lg leading-relaxed text-ink-soft"
        >
          Upload a document or drop a video link. SourceMind transcribes,
          indexes, and answers your questions — with every claim traced back to a
          page or a timestamp.
        </motion.p>
      </section>

      {/* Ingest */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.25 }}
        className="grid grid-cols-1 gap-4 md:grid-cols-2"
      >
        {/* PDF drop zone */}
        <div>
          <p className="eyebrow mb-2 text-ink-faint">Document</p>
          <motion.label
            whileHover={{ y: -2 }}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              handleFiles(e.dataTransfer.files);
            }}
            className={`flex h-[132px] cursor-pointer flex-col items-center justify-center gap-1.5 rounded-2xl border-2 border-dashed px-4 text-center shadow-card transition-colors ${
              dragging
                ? "border-emerald bg-emerald-soft"
                : "border-line-strong bg-paper-raised hover:border-emerald/60"
            } ${docMut.isPending ? "pointer-events-none opacity-60" : ""}`}
          >
            <span className="font-display text-2xl text-ink">＋</span>
            <span className="text-sm font-medium text-ink">
              Drop a PDF, or <span className="text-emerald underline">browse</span>
            </span>
            <span className="text-xs text-ink-faint">up to 100 MB</span>
            <input
              type="file"
              accept="application/pdf"
              onChange={(e) => handleFiles(e.target.files)}
              disabled={docMut.isPending}
              className="hidden"
            />
          </motion.label>
          {docMut.isPending && (
            <p className="mt-2 text-xs text-emerald">Uploading & indexing…</p>
          )}
          {(notice || docMut.isError) && (
            <p className="mt-2 text-xs text-coral">
              {notice || docMut.error?.response?.data?.detail || "Upload failed"}
            </p>
          )}
        </div>

        {/* YouTube */}
        <div>
          <p className="eyebrow mb-2 text-ink-faint">Video</p>
          <div className="flex h-[132px] flex-col justify-center gap-3 rounded-2xl border border-line bg-paper-raised px-5 shadow-card">
            <div className="flex gap-2">
              <input
                type="url"
                placeholder="Paste a YouTube link…"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                onKeyDown={(e) =>
                  e.key === "Enter" && url.trim() && vidMut.mutate(url.trim())
                }
                className="min-w-0 flex-1 rounded-lg border border-line bg-paper px-3 py-2.5 font-mono text-sm text-ink placeholder:text-ink-faint focus:border-emerald focus:outline-none"
              />
              <motion.button
                whileTap={{ scale: 0.96 }}
                onClick={() => url.trim() && vidMut.mutate(url.trim())}
                disabled={vidMut.isPending || !url.trim()}
                className="rounded-lg bg-ink px-5 py-2.5 text-sm font-semibold text-paper transition-colors hover:bg-emerald disabled:opacity-40"
              >
                {vidMut.isPending ? "Adding…" : "Add"}
              </motion.button>
            </div>
            {vidMut.isError ? (
              <p className="text-xs text-coral">
                {vidMut.error?.response?.data?.detail || "Could not add video"}
              </p>
            ) : (
              <p className="text-xs text-ink-faint">
                Transcribed locally, translated to English, then indexed.
              </p>
            )}
          </div>
        </div>
      </motion.section>

      {/* Library */}
      <section>
        <div className="mb-5 flex items-baseline justify-between border-b border-line pb-3">
          <h2 className="font-display text-2xl font-semibold text-ink">
            Your library
          </h2>
          <span className="font-mono text-sm text-ink-faint">
            {String(sources.length).padStart(2, "0")} source
            {sources.length !== 1 ? "s" : ""}
          </span>
        </div>

        {sources.length === 0 ? (
          <div className="rounded-2xl border border-dashed border-line-strong bg-paper-raised/50 py-16 text-center">
            <p className="font-display text-xl italic text-ink-soft">
              The shelves are empty.
            </p>
            <p className="mt-1 text-sm text-ink-faint">
              Add a document or video above to begin.
            </p>
          </div>
        ) : (
          <motion.ul
            initial="hidden"
            animate="show"
            variants={{
              hidden: {},
              show: { transition: { staggerChildren: 0.06 } },
            }}
            className="space-y-3"
          >
            <AnimatePresence>
              {sources.map((s, i) => (
                <motion.li
                  key={s.id}
                  layout
                  variants={{
                    hidden: { opacity: 0, y: 14 },
                    show: { opacity: 1, y: 0 },
                  }}
                  exit={{ opacity: 0, x: -20 }}
                  whileHover={{ y: -3 }}
                  transition={{ type: "spring", stiffness: 300, damping: 26 }}
                  className="group flex flex-wrap items-center gap-4 rounded-2xl border border-line bg-paper-raised p-5 shadow-card transition-shadow hover:shadow-card-hover"
                >
                  <span className="font-mono text-sm text-ink-faint">
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  <span className="grid h-11 w-11 shrink-0 place-items-center rounded-xl bg-paper-sunken text-lg">
                    {s.source_type === "video" ? "▶" : "❧"}
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="font-display text-lg font-medium leading-snug text-ink line-clamp-1">
                      {s.title}
                    </div>
                    <div className="mt-1 flex flex-wrap items-center gap-3 text-xs">
                      <StatusPill status={s.status} />
                      <span className="font-mono uppercase tracking-wider text-ink-faint">
                        {s.source_type}
                      </span>
                      {typeof s.eval_score === "number" && (
                        <span className="flex items-center gap-1 font-medium text-emerald">
                          <span className="h-1.5 w-1.5 rounded-full bg-emerald" />
                          RAG {(s.eval_score * 100).toFixed(0)}%
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => nav(`/source/${s.id}`)}
                      className="rounded-lg border border-line px-3 py-1.5 text-sm text-ink-soft transition-colors hover:border-ink hover:text-ink"
                    >
                      Details
                    </button>
                    <motion.button
                      whileTap={{ scale: 0.95 }}
                      onClick={() => nav(`/chat/${s.id}`)}
                      disabled={s.status !== "ready"}
                      className="rounded-lg bg-emerald px-4 py-1.5 text-sm font-semibold text-paper transition-colors hover:bg-emerald-deep disabled:opacity-30"
                    >
                      Chat →
                    </motion.button>
                    <button
                      onClick={() => {
                        if (confirm("Remove this source?")) delMut.mutate(s.id);
                      }}
                      className="px-1.5 text-ink-faint transition-colors hover:text-coral"
                      title="Delete"
                    >
                      ✕
                    </button>
                  </div>
                </motion.li>
              ))}
            </AnimatePresence>
          </motion.ul>
        )}
      </section>
    </div>
  );
}

function StatusPill({ status }) {
  const map = {
    processing: { c: "text-gold border-gold/40 bg-gold/10", label: "processing" },
    ready: { c: "text-emerald border-emerald/40 bg-emerald-soft", label: "ready" },
    failed: { c: "text-coral border-coral/40 bg-coral-soft", label: "failed" },
  };
  const s = map[status] || map.processing;
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider ${s.c}`}
    >
      {status === "processing" && (
        <motion.span
          animate={{ opacity: [1, 0.3, 1] }}
          transition={{ duration: 1.2, repeat: Infinity }}
          className="h-1.5 w-1.5 rounded-full bg-gold"
        />
      )}
      {s.label}
    </span>
  );
}
