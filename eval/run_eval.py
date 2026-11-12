"""
Run one experiment arm of the SourceMind benchmark.

    python -m eval.run_eval --config eval/configs/v1_as_shipped.yaml
    python -m eval.run_eval --config ... --limit 5          # quick try
    python -m eval.run_eval --config ... --skip-ingest      # reuse the eval DB as-is

What it does:
1. Points the backend at the isolated eval DB + Qdrant collection (eval/bootstrap.py).
2. Rebuilds both from scratch using the *arm's own* schema and ingestion code, and
   ingests the whole corpus (PDFs from eval/corpus/files, videos from the committed
   transcripts, so Whisper never reruns).
3. Asks every question through the real chat path (`route_and_stream`), exactly as
   the API does, and records the answer, citations, query type and latency.
4. Writes eval/results/<arm>/run.json (metadata) and records.jsonl (one line per
   question). Interrupted runs resume where they stopped.

Scoring is a separate step (eval/score.py), so re-scoring never re-runs the RAG.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import re
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from eval.bootstrap import EVAL_COLLECTION, EVAL_DB, git_dirty, git_sha, setup_backend
from eval.paths import CACHE_DIR, REPO_DIR, RESULTS_DIR, WORK_DIR


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(path: str) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg.setdefault("overrides", {})
    cfg.setdefault("throttle_s", 4.0)
    return cfg


# -- database reset --------------------------------------------------------

async def reset_stores(backend: Path) -> None:
    import asyncpg
    from qdrant_client import AsyncQdrantClient
    from config import settings

    dsn = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://", 1)
    admin = await asyncpg.connect(dsn.rsplit("/", 1)[0] + "/postgres")
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS "{EVAL_DB}" WITH (FORCE)')
        await admin.execute(f'CREATE DATABASE "{EVAL_DB}"')
    finally:
        await admin.close()
    schema = (backend.parent / "scripts" / "init_db.sql").read_text(encoding="utf-8")
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(schema)
    finally:
        await conn.close()

    client = AsyncQdrantClient(url=settings.QDRANT_URL)
    try:
        names = {c.name for c in (await client.get_collections()).collections}
        if EVAL_COLLECTION in names:
            await client.delete_collection(EVAL_COLLECTION)
    finally:
        await client.close()


# -- ingestion -------------------------------------------------------------

async def ingest_corpus() -> dict[str, str]:
    """Ingest every available corpus source; return {doc_key: source_id}."""
    from db.postgres import Database, create_source, get_source
    import core.video_pipeline as vp
    from core.document_pipeline import ingest_document
    from eval.corpus.download import load_manifest, pdf_path
    from eval.corpus.text import transcript

    manifest = load_manifest()
    mapping: dict[str, str] = {}

    for doc in manifest["documents"]:
        path = pdf_path(doc["key"])
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - run: python -m eval.corpus.download")
        sid = await create_source("document", doc["title"], filename=f"{doc['key']}.pdf")
        t0 = time.time()
        # ingest_document deletes its input file when done (it's meant for uploads),
        # so hand it a throwaway copy, never the corpus file itself.
        upload = WORK_DIR / "uploads" / f"{sid}_{doc['key']}.pdf"
        upload.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, upload)
        await ingest_document(sid, str(upload), f"{doc['key']}.pdf")
        src = await get_source(sid)
        if src["status"] != "ready":
            raise RuntimeError(f"ingest of {doc['key']} failed: {src.get('error_message')}")
        mapping[doc["key"]] = str(sid)
        print(f"  ingested {doc['key']:22s} {time.time() - t0:5.1f}s", flush=True)

    original = vp.transcribe_audio
    try:
        for v in manifest.get("videos", []):
            data = transcript(v["key"])
            if not data:
                print(f"  skipped {v['key']} (no transcript committed)")
                continue
            segments = data["segments"]
            vp.transcribe_audio = lambda _wav, _s=segments: _s  # replay committed transcript
            sid = await create_source("video", v["title"], youtube_url=v["url"])
            t0 = time.time()
            await vp._transcribe_and_index(sid, "committed-transcript", data.get("duration_s") or 0,
                                           v["title"])
            mapping[v["key"]] = str(sid)
            print(f"  ingested {v['key']:22s} {time.time() - t0:5.1f}s", flush=True)
    finally:
        vp.transcribe_audio = original

    counts = await Database.fetch(
        "SELECT source_id::text AS sid, count(*) AS n FROM bm25_index GROUP BY source_id")
    by_sid = {r["sid"]: r["n"] for r in counts}
    for k, sid in mapping.items():
        print(f"    {k:22s} {by_sid.get(sid, 0):5d} chunks")
    return mapping


# -- asking ----------------------------------------------------------------

def _to_retrieved(citation: dict, sid_to_doc: dict[str, str]) -> dict:
    item = {"doc": sid_to_doc.get(str(citation.get("source_id")), "?"), "type": citation.get("type")}
    if citation.get("type") == "document":
        item["page"] = citation.get("page_number")
    elif citation.get("type") == "video":
        item["start"] = citation.get("segment_start")
        item["end"] = citation.get("segment_end")
    return item


async def ask(question: dict, source_id: str, sid_to_doc: dict, retries: int = 6) -> dict:
    from uuid import UUID
    from core.agent_router import route_and_stream

    delay = 10.0
    for attempt in range(1, retries + 1):
        t0 = time.time()
        tokens, citations, sub_queries = [], [], []
        query_type, error = None, None
        log = io.StringIO()  # the backend reports helper failures via print()
        with contextlib.redirect_stdout(log):
          async for ev in route_and_stream(question["question"], UUID(source_id),
                                           question.get("history") or []):
            t = ev.get("type")
            if t == "token":
                tokens.append(ev.get("value") or "")
            elif t == "citations":
                citations.extend(ev.get("value") or [])
            elif t == "query_type":
                query_type = ev.get("value")
            elif t == "sub_queries":
                sub_queries = ev.get("value") or []
            elif t == "error":
                error = str(ev.get("value"))
        latency = time.time() - t0
        # A helper step (HyDE, multi-query, classifier...) that failed is swallowed by
        # the backend, which then answers with a weaker pipeline. Never score that as
        # a normal answer: retry, and if it persists, mark the record degraded.
        degraded = sorted(set(re.findall(r"^\[(\w+)\] (?:failed|.*failed)", log.getvalue(), re.M)))
        if degraded and not error and "429" in log.getvalue():
            error = f"degraded by rate limit: {', '.join(degraded)}"
        transient = error and any(s in error for s in (
            "429", "rate", "Rate", "timeout", "Timeout", "503", "502", "500", "Connection", "connection"))
        if transient and attempt < retries:
            print(f"    transient error ({error[:60]}), retry in {delay:.0f}s", flush=True)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 120)
            continue
        return {
            "id": question["id"],
            "answer": "".join(tokens),
            "query_type": query_type,
            "sub_queries": sub_queries,
            "citations": citations,
            "retrieved": [_to_retrieved(c, sid_to_doc) for c in citations],
            "latency_s": round(latency, 3),
            "attempts": attempt,
            "degraded": degraded,
            "error": error,
        }
    raise RuntimeError("unreachable")


async def run(cfg: dict, args, backend: Path) -> int:
    from langchain_community.cache import SQLiteCache
    from langchain_core.globals import set_llm_cache

    from config import settings
    from db.postgres import Database
    from db.qdrant import QdrantStore
    from eval.datasets.schema import load, validate_file
    from eval.corpus.text import sources

    dataset = (REPO_DIR / (args.dataset or cfg["dataset"])).resolve()
    problems = validate_file(dataset, set(sources()))
    if problems:
        print("dataset is invalid:\n  " + "\n  ".join(problems[:20]))
        return 1
    questions = load(dataset)
    if args.limit:
        questions = questions[: args.limit]

    out_dir = RESULTS_DIR / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    run_path, rec_path = out_dir / "run.json", out_dir / "records.jsonl"

    # Cache the system's non-streamed helper calls (classify, multi-query, HyDE...)
    # so reruns are cheaper; streamed final answers are always generated live.
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    set_llm_cache(SQLiteCache(database_path=str(CACHE_DIR / "sut_llm.sqlite")))

    meta_prev = json.loads(run_path.read_text()) if run_path.exists() else {}
    if not args.skip_ingest:
        print("resetting eval database + collection ...", flush=True)
        await reset_stores(backend)
    await Database.connect()
    await QdrantStore.connect()
    try:
        if args.skip_ingest and meta_prev.get("source_ids"):
            mapping = meta_prev["source_ids"]
            print("reusing ingested corpus from previous run")
        else:
            print("ingesting corpus ...", flush=True)
            t0 = time.time()
            mapping = await ingest_corpus()
            print(f"ingest done in {time.time() - t0:.0f}s", flush=True)
            if rec_path.exists():
                rec_path.unlink()  # fresh index -> fresh answers
        sid_to_doc = {v: k for k, v in mapping.items()}

        meta = {
            "arm": cfg["name"],
            "description": cfg.get("description", ""),
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "backend_dir": str(backend),
            "backend_git_sha": git_sha(backend),
            "backend_dirty": git_dirty(backend),
            "dataset": str(dataset.relative_to(REPO_DIR)),
            "dataset_sha256": sha256_file(dataset),
            "n_questions": len(questions),
            "models": {"main": settings.GROQ_MODEL_MAIN, "fast": settings.GROQ_MODEL_FAST,
                       "embedding": settings.EMBEDDING_MODEL},
            "settings": {k: getattr(settings, k) for k in (
                "MMR_LAMBDA", "MMR_FETCH_K", "MMR_FINAL_K", "BM25_TOP_K", "DENSE_TOP_K",
                "PARENT_CHUNK_SIZE", "CHILD_CHUNK_SIZE", "VIDEO_CHUNK_WORDS")},
            "overrides": cfg["overrides"],
            "source_ids": mapping,
        }
        run_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        done = set()
        if rec_path.exists():
            # keep successful answers; drop errored ones so they are asked again
            kept = [line for line in rec_path.read_text(encoding="utf-8").splitlines()
                    if line and not json.loads(line).get("error")
                    and not json.loads(line).get("degraded")]
            rec_path.write_text("".join(k + "\n" for k in kept), encoding="utf-8")
            done = {json.loads(line)["id"] for line in kept}
        todo = [q for q in questions if q["id"] not in done]
        print(f"asking {len(todo)} question(s) ({len(done)} already done)", flush=True)

        with open(rec_path, "a", encoding="utf-8") as f:
            for i, q in enumerate(todo, 1):
                if q["doc"] not in mapping:
                    print(f"  [{i}/{len(todo)}] {q['id']}: source {q['doc']} not ingested, skipped")
                    continue
                rec = await ask(q, mapping[q["doc"]], sid_to_doc)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                status = "ERROR " + rec["error"][:50] if rec["error"] else f"{rec['latency_s']:.1f}s"
                print(f"  [{i}/{len(todo)}] {q['id']:<24s} {status}", flush=True)
                await asyncio.sleep(cfg["throttle_s"])

        meta["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        run_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"\nrecords -> {rec_path.relative_to(REPO_DIR)}")
        print(f"next: python -m eval.score --arm {cfg['name']}")
        return 0
    finally:
        await Database.disconnect()
        await QdrantStore.disconnect()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", help="override the config's dataset (repo-relative path)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--skip-ingest", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    backend = setup_backend(cfg.get("backend_dir") and (REPO_DIR / cfg["backend_dir"]),
                            cfg["overrides"])
    return asyncio.run(run(cfg, args, backend))


if __name__ == "__main__":
    sys.exit(main())
