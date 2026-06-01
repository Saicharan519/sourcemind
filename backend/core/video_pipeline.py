"""
Video ingestion pipeline.

YouTube URL → yt-dlp audio → Whisper transcription → segment-aware chunking →
embed + Qdrant upsert + BM25 index → auto-title → status update.

Designed to run as a FastAPI BackgroundTask.
"""
from __future__ import annotations

import os
from functools import lru_cache
from uuid import UUID

import whisper

from config import settings
from core.chunker import segment_chunk
from core.embedder import embed_texts
from core.rag_engine import get_fast_llm
from db.postgres import insert_bm25_rows, update_source_status
from db.qdrant import QdrantStore
from utils.audio import download_youtube_audio


@lru_cache(maxsize=1)
def _whisper_model():
    print(f"[whisper] loading model: {settings.WHISPER_MODEL}")
    m = whisper.load_model(settings.WHISPER_MODEL)
    print("[whisper] loaded.")
    return m


def transcribe_audio(wav_path: str) -> list[dict]:
    """Returns Whisper segments: [{text, start, end}, ...]."""
    model = _whisper_model()
    result = model.transcribe(wav_path, task="transcribe", verbose=False)
    return [
        {"text": s.get("text", ""), "start": s.get("start", 0.0), "end": s.get("end", 0.0)}
        for s in result.get("segments", [])
    ]


async def _generate_title(sample_text: str, fallback_title: str) -> str:
    prompt = (
        "Generate a concise descriptive title (max 12 words) for the following "
        "video transcript excerpt. Return only the title, no quotes or commentary.\n\n"
        f"Excerpt:\n{sample_text[:2000]}"
    )
    try:
        llm = get_fast_llm()
        resp = await llm.ainvoke(prompt)
        title = (resp.content if hasattr(resp, "content") else str(resp)).strip()
        title = title.strip('"').strip("'")
        return title[:200] or fallback_title
    except Exception as e:
        print(f"[ingest_video] title gen failed: {e}")
        return fallback_title


async def ingest_video(source_id: UUID, youtube_url: str) -> None:
    wav_path: str | None = None
    try:
        # 1. Download audio
        wav_path, meta = download_youtube_audio(youtube_url)
        duration = meta.get("duration", 0)

        if duration > settings.MAX_VIDEO_DURATION_MIN * 60:
            raise ValueError(f"Video exceeds max duration of {settings.MAX_VIDEO_DURATION_MIN} minutes.")

        # 2. Transcribe
        segments = transcribe_audio(wav_path)
        if not segments:
            raise ValueError("Transcription produced no segments.")

        # 3. Chunk by segments (~400 words)
        chunks = segment_chunk(segments)
        if not chunks:
            raise ValueError("Chunking produced no chunks.")

        # 4. Title from first few chunks
        sample = " ".join(c.content for c in chunks[:3])
        title = await _generate_title(sample, fallback_title=meta.get("title") or "Untitled video")

        # 5. Embed + upsert
        texts = [c.content for c in chunks]
        embeddings = embed_texts(texts)
        payloads = [
            {
                "source_id": str(source_id),
                "source_type": "video",
                "source_title": title,
                "content": c.content,
                "segment_start": c.segment_start,
                "segment_end": c.segment_end,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]
        qdrant_ids = await QdrantStore.upsert(embeddings, payloads)

        # 6. BM25 corpus
        bm25_rows = [
            {"chunk_id": qid, "content": c.content, "chunk_index": c.chunk_index}
            for qid, c in zip(qdrant_ids, chunks)
        ]
        await insert_bm25_rows(source_id, bm25_rows)

        # 7. Done
        await update_source_status(
            source_id,
            status="ready",
            duration_s=int(duration),
            title=title,
        )

    except Exception as e:
        print(f"[ingest_video] FAILED for source_id={source_id}: {e}")
        await update_source_status(source_id, status="failed", error_message=str(e))
    finally:
        # Clean up audio file
        try:
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            pass
