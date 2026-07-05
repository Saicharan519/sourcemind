"""
Document ingestion pipeline.

PDF → PyMuPDF text extraction → hierarchical chunking (parents to Postgres,
children to Qdrant) → BM25 index → auto-title via Groq → status update.

Designed to run as a FastAPI BackgroundTask.
"""
from __future__ import annotations

import asyncio
import os
from uuid import UUID

import fitz  # pymupdf
from langdetect import detect

from config import settings
from core.chunker import hierarchical_chunk
from core.embedder import embed_texts
from core.rag_engine import get_fast_llm
from db.postgres import (
    insert_bm25_rows,
    insert_parent_chunks,
    update_source_status,
)
from db.qdrant import QdrantStore


# -- Translation (chat is English-only; translate non-English PDFs at ingest) ---

def _detect_language(pages: list[dict]) -> str:
    sample = " ".join(p["text"] for p in pages[:3])[:2000]
    try:
        return detect(sample) if sample.strip() else "en"
    except Exception:
        return "en"


async def _translate_page_to_english(text: str) -> str:
    prompt = (
        "Translate the following text to English. Preserve the meaning, structure, "
        "and any lists or headings. Return ONLY the translation — no preamble, "
        "notes, or the original text.\n\n"
        f"{text}"
    )
    try:
        llm = get_fast_llm()
        resp = await llm.ainvoke(prompt)
        out = (resp.content if hasattr(resp, "content") else str(resp)).strip()
        return out or text
    except Exception as e:
        print(f"[ingest_document] page translate failed, keeping original: {e}")
        return text


async def _maybe_translate_pages(pages: list[dict]) -> list[dict]:
    """If the document isn't English, translate each page via Groq 8B, keeping
    page numbers intact. Bounded concurrency to stay within rate limits."""
    if not settings.TRANSLATE_TO_ENGLISH:
        return pages
    lang = _detect_language(pages)
    if lang == "en":
        return pages
    print(f"[ingest_document] detected language '{lang}' — translating {len(pages)} pages to English")

    sem = asyncio.Semaphore(4)

    async def _one(p: dict) -> dict:
        async with sem:
            return {**p, "text": await _translate_page_to_english(p["text"])}

    return list(await asyncio.gather(*(_one(p) for p in pages)))


async def _generate_title(sample_text: str, filename: str) -> str:
    prompt = (
        "Generate a concise descriptive title (max 10 words) for the following "
        f"document excerpt. Return only the title, no quotes or commentary.\n\n"
        f"Filename: {filename}\n\nExcerpt:\n{sample_text[:2000]}"
    )
    try:
        llm = get_fast_llm()
        resp = await llm.ainvoke(prompt)
        title = (resp.content if hasattr(resp, "content") else str(resp)).strip()
        title = title.strip('"').strip("'")
        return title[:200] or filename
    except Exception as e:
        print(f"[ingest_document] title gen failed: {e}")
        return filename


def extract_pdf_pages(pdf_path: str) -> list[dict]:
    """Extract text page-by-page from a PDF. Returns [{page_number, text}]."""
    pages = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            text = text.strip()
            if text:
                pages.append({"page_number": i, "text": text})
    return pages


async def ingest_document(source_id: UUID, pdf_path: str, original_filename: str) -> None:
    """
    Full async pipeline. Updates the source row to 'ready' on success,
    'failed' with an error message on exception.
    """
    try:
        # 1. Extract (PyMuPDF is blocking — offload so the event loop stays free)
        pages = await asyncio.to_thread(extract_pdf_pages, pdf_path)
        if not pages:
            raise ValueError("No extractable text in PDF.")

        # 1b. Translate to English if the document isn't already English.
        pages = await _maybe_translate_pages(pages)

        # 2. Hierarchical chunking
        parents, children = hierarchical_chunk(pages)
        if not children:
            raise ValueError("Chunking produced no children.")

        # 3. Insert parents into Postgres, get UUIDs
        parent_id_map = await insert_parent_chunks(
            source_id,
            [
                {
                    "content": p.content,
                    "page_start": p.page_start,
                    "page_end": p.page_end,
                    "chunk_index": p.chunk_index,
                }
                for p in parents
            ],
        )

        # 4. Generate title from first few parents
        sample = "\n".join(p.content for p in parents[:2])
        title = await _generate_title(sample, original_filename)

        # 5. Embed children + upsert to Qdrant with payload
        child_texts = [c.content for c in children]
        embeddings = await asyncio.to_thread(embed_texts, child_texts)

        payloads = [
            {
                "source_id": str(source_id),
                "source_type": "document",
                "source_title": title,
                "content": c.content,
                "page_number": c.page_number,
                "parent_chunk_id": str(parent_id_map[c.parent_chunk_index]),
                "chunk_index": c.chunk_index,
            }
            for c in children
        ]
        qdrant_ids = await QdrantStore.upsert(embeddings, payloads)

        # 6. Insert BM25 corpus rows (one per child chunk)
        bm25_rows = [
            {
                "chunk_id": qid,
                "content": c.content,
                "chunk_index": c.chunk_index,
            }
            for qid, c in zip(qdrant_ids, children)
        ]
        await insert_bm25_rows(source_id, bm25_rows)

        # 7. Done
        await update_source_status(
            source_id,
            status="ready",
            page_count=len(pages),
            title=title,
        )

        # 8. Trigger eval in the background — caller will schedule this
        # (we let the route layer schedule eval to avoid coupling)

    except Exception as e:
        print(f"[ingest_document] FAILED for source_id={source_id}: {e}")
        await update_source_status(source_id, status="failed", error_message=str(e))
    finally:
        # Clean up uploaded PDF
        try:
            if pdf_path and os.path.exists(pdf_path):
                os.remove(pdf_path)
        except Exception:
            pass
