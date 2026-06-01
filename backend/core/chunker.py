"""
Chunking strategies:

- Documents: hierarchical parent (1500 tokens) / child (300 tokens) chunking.
  Children are embedded into Qdrant for retrieval, parents stored in Postgres
  for full context to pass to the LLM.

- Video transcripts: segment-aware chunking. Whisper produces segments with
  start/end positions; we group those segments into chunks of ~400 words.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings


# -- Documents ---------------------------------------------------------------

@dataclass
class ParentChunk:
    chunk_index: int
    content: str
    page_start: int | None
    page_end: int | None


@dataclass
class ChildChunk:
    chunk_index: int
    content: str
    parent_chunk_index: int
    page_number: int | None


def _split(text: str, size: int, overlap: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return splitter.split_text(text)


def _page_at_offset(page_offsets: list[tuple[int, int, int]], offset: int) -> int | None:
    """Given offsets [(start, end, page_no)], find the page containing `offset`."""
    for start, end, page_no in page_offsets:
        if start <= offset < end:
            return page_no
    return None


def hierarchical_chunk(pages: list[dict[str, Any]]) -> tuple[list[ParentChunk], list[ChildChunk]]:
    """
    Chunk a document into parent + child levels.

    `pages` items: { "page_number": int, "text": str }
    """
    full_text_parts: list[str] = []
    page_offsets: list[tuple[int, int, int]] = []
    cursor = 0
    for p in pages:
        txt = p["text"]
        full_text_parts.append(txt)
        page_offsets.append((cursor, cursor + len(txt), p["page_number"]))
        cursor += len(txt) + 2  # +2 accounts for the "\n\n" joiner below

    full_text = "\n\n".join(full_text_parts)

    # parents
    parent_texts = _split(
        full_text,
        size=settings.PARENT_CHUNK_SIZE,
        overlap=settings.PARENT_CHUNK_OVERLAP,
    )

    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []

    search_start = 0
    for p_idx, p_text in enumerate(parent_texts):
        # locate parent in full_text to derive page span
        pos = full_text.find(p_text, search_start)
        if pos == -1:
            pos = full_text.find(p_text)
        if pos == -1:
            page_start = page_end = None
        else:
            page_start = _page_at_offset(page_offsets, pos)
            page_end = _page_at_offset(page_offsets, max(pos, pos + len(p_text) - 1))
            search_start = max(search_start, pos + 1)

        parents.append(ParentChunk(
            chunk_index=p_idx,
            content=p_text,
            page_start=page_start,
            page_end=page_end,
        ))

        # children inside this parent
        child_texts = _split(
            p_text,
            size=settings.CHILD_CHUNK_SIZE,
            overlap=settings.CHILD_CHUNK_OVERLAP,
        )
        for c_text in child_texts:
            # approximate page lookup: take page of first occurrence within parent
            if pos != -1:
                c_pos = full_text.find(c_text, pos)
                page_no = _page_at_offset(page_offsets, c_pos) if c_pos != -1 else page_start
            else:
                page_no = page_start
            children.append(ChildChunk(
                chunk_index=len(children),
                content=c_text,
                parent_chunk_index=p_idx,
                page_number=page_no,
            ))

    return parents, children


# -- Video ------------------------------------------------------------------

@dataclass
class VideoChunk:
    chunk_index: int
    content: str
    segment_start: float
    segment_end: float


def segment_chunk(segments: list[dict[str, Any]], words_per_chunk: int | None = None) -> list[VideoChunk]:
    """
    Group Whisper segments into chunks of ~N words.

    `segments` items: { "text": str, "start": float, "end": float }
    """
    words_per_chunk = words_per_chunk or settings.VIDEO_CHUNK_WORDS

    chunks: list[VideoChunk] = []
    buf_text: list[str] = []
    buf_words = 0
    buf_start: float | None = None
    buf_end: float | None = None
    idx = 0

    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        words = text.split()

        if buf_start is None:
            buf_start = float(seg["start"])
        buf_end = float(seg["end"])
        buf_text.append(text)
        buf_words += len(words)

        if buf_words >= words_per_chunk:
            chunks.append(VideoChunk(
                chunk_index=idx,
                content=" ".join(buf_text),
                segment_start=buf_start,
                segment_end=buf_end,
            ))
            idx += 1
            buf_text = []
            buf_words = 0
            buf_start = None
            buf_end = None

    if buf_text and buf_start is not None and buf_end is not None:
        chunks.append(VideoChunk(
            chunk_index=idx,
            content=" ".join(buf_text),
            segment_start=buf_start,
            segment_end=buf_end,
        ))

    return chunks
