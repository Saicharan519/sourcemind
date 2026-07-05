"""
Video ingestion pipeline.

Audio source (YouTube/Drive URL via yt-dlp, or an uploaded video file via ffmpeg)
→ Whisper transcription → segment-aware chunking → embed + Qdrant upsert +
BM25 index → auto-title → status update.

Designed to run as a FastAPI BackgroundTask.
"""
from __future__ import annotations

import asyncio
import os
from functools import lru_cache
from pathlib import Path
from uuid import UUID

from faster_whisper import WhisperModel

from config import settings
from core.chunker import segment_chunk
from core.embedder import embed_texts
from core.rag_engine import get_fast_llm
from db.postgres import insert_bm25_rows, update_source_status
from db.qdrant import QdrantStore
from utils.audio import download_audio_from_url, extract_audio_from_file


@lru_cache(maxsize=1)
def _whisper_model() -> WhisperModel:
    device = "cuda" if settings.EMBEDDING_DEVICE == "cuda" else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"[whisper] loading faster-whisper model: {settings.WHISPER_MODEL} ({device}/{compute_type})")
    m = WhisperModel(settings.WHISPER_MODEL, device=device, compute_type=compute_type)
    print("[whisper] loaded.")
    return m


def transcribe_audio(wav_path: str) -> list[dict]:
    """Returns Whisper segments: [{text, start, end}, ...]. Blocking/CPU-bound.

    Uses greedy decoding (beam_size=1) + a VAD filter that skips silence/music —
    together these are what actually deliver faster-whisper's speedup on CPU.
    Iterating the returned generator is what drives the work, so we log progress
    against the known audio duration (faster-whisper prints no progress bar).
    """
    model = _whisper_model()
    # Whisper's "translate" task transcribes AND translates to English in one pass
    # (it only ever targets English). Since chat is English-only, this gives an
    # English transcript for any source language, with segment timings preserved.
    task = "translate" if settings.TRANSLATE_TO_ENGLISH else "transcribe"
    segments, info = model.transcribe(
        wav_path,
        task=task,
        beam_size=1,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    total = float(getattr(info, "duration", 0.0)) or 0.0
    print(
        f"[whisper] {task} ~{int(total)}s of audio "
        f"(detected lang={getattr(info, 'language', '?')})"
    )

    out: list[dict] = []
    next_mark = 0.10
    for s in segments:
        out.append({"text": s.text or "", "start": float(s.start or 0.0), "end": float(s.end or 0.0)})
        if total and s.end and (s.end / total) >= next_mark:
            print(f"[whisper] ~{int(100 * s.end / total)}% ({int(s.end)}/{int(total)}s)")
            next_mark += 0.10
    print(f"[whisper] done — {len(out)} segments")
    return out


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


def _cleanup(path: str | None) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


async def _transcribe_and_index(
    source_id: UUID,
    wav_path: str,
    duration: int,
    fallback_title: str,
) -> None:
    """Shared tail of both ingest paths: transcribe → chunk → embed → upsert →
    BM25 → mark ready. Assumes `wav_path` is a ready-to-transcribe WAV."""
    if duration and duration > settings.MAX_VIDEO_DURATION_MIN * 60:
        raise ValueError(
            f"Video exceeds max duration of {settings.MAX_VIDEO_DURATION_MIN} minutes."
        )

    # Transcribe (CPU-bound — offload so chat/polling stays responsive)
    segments = await asyncio.to_thread(transcribe_audio, wav_path)
    if not segments:
        raise ValueError("Transcription produced no segments.")

    chunks = segment_chunk(segments)
    if not chunks:
        raise ValueError("Chunking produced no chunks.")

    sample = " ".join(c.content for c in chunks[:3])
    title = await _generate_title(sample, fallback_title=fallback_title)

    texts = [c.content for c in chunks]
    embeddings = await asyncio.to_thread(embed_texts, texts)
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

    bm25_rows = [
        {"chunk_id": qid, "content": c.content, "chunk_index": c.chunk_index}
        for qid, c in zip(qdrant_ids, chunks)
    ]
    await insert_bm25_rows(source_id, bm25_rows)

    await update_source_status(
        source_id,
        status="ready",
        duration_s=int(duration),
        title=title,
    )


async def ingest_video(source_id: UUID, video_url: str) -> None:
    """Ingest from a URL (YouTube or public Google Drive) via yt-dlp."""
    wav_path: str | None = None
    try:
        # Network + ffmpeg — offload so the event loop stays free.
        wav_path, meta = await asyncio.to_thread(download_audio_from_url, video_url)
        await _transcribe_and_index(
            source_id,
            wav_path,
            meta.get("duration", 0),
            meta.get("title") or "Untitled video",
        )
    except Exception as e:
        print(f"[ingest_video] FAILED for source_id={source_id}: {e}")
        await update_source_status(source_id, status="failed", error_message=str(e))
    finally:
        _cleanup(wav_path)


async def ingest_video_file(source_id: UUID, video_path: str, filename: str) -> None:
    """Ingest from an uploaded video file: ffmpeg-extract audio, then index."""
    wav_path: str | None = None
    try:
        wav_path, meta = await asyncio.to_thread(extract_audio_from_file, video_path)
        await _transcribe_and_index(
            source_id,
            wav_path,
            meta.get("duration", 0),
            Path(filename).stem or "Untitled video",
        )
    except Exception as e:
        print(f"[ingest_video_file] FAILED for source_id={source_id}: {e}")
        await update_source_status(source_id, status="failed", error_message=str(e))
    finally:
        _cleanup(wav_path)
        _cleanup(video_path)  # remove the uploaded original once processed
