"""
Read corpus text the same way the app does, without importing the backend.

- PDFs: PyMuPDF `page.get_text("text").strip()` per page (mirrors
  core/document_pipeline.extract_pdf_pages), 1-based page numbers.
- Videos: committed Whisper transcripts, grouped into ~90 s windows.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache

import fitz

from eval.corpus.download import load_manifest, pdf_path
from eval.paths import CORPUS_DIR

_TOC_LINE = re.compile(r"\.{5,}|…{3,}")
_REF_HEADING = re.compile(r"^\s*(references|bibliography)\s*$", re.I | re.M)


@dataclass(frozen=True)
class Passage:
    doc: str
    kind: str              # "pdf" | "video"
    text: str
    page: int | None = None
    start: float | None = None
    end: float | None = None

    @property
    def label(self) -> str:
        if self.kind == "pdf":
            return f"{self.doc} p.{self.page}"
        return f"{self.doc} {int(self.start)}-{int(self.end)}s"

    def evidence(self) -> dict:
        if self.kind == "pdf":
            return {"doc": self.doc, "pages": [self.page]}
        return {"doc": self.doc, "start": round(self.start, 1), "end": round(self.end, 1)}


def sources() -> dict[str, dict]:
    m = load_manifest()
    out = {d["key"]: d for d in m["documents"]}
    out.update({v["key"]: v for v in m.get("videos", [])})
    return out


@lru_cache(maxsize=None)
def pdf_pages(doc: str) -> dict[int, str]:
    pages = {}
    with fitz.open(pdf_path(doc)) as pdf:
        for i, page in enumerate(pdf, start=1):
            text = (page.get_text("text") or "").strip()
            if text:
                pages[i] = text
    return pages


@lru_cache(maxsize=None)
def transcript(doc: str) -> dict | None:
    meta = sources()[doc]
    path = CORPUS_DIR / meta["transcript"]
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def video_windows(doc: str, window_s: float = 90.0) -> list[Passage]:
    data = transcript(doc)
    if not data:
        return []
    out, buf, start = [], [], None
    for seg in data["segments"]:
        if start is None:
            start = seg["start"]
        buf.append(seg["text"].strip())
        if seg["end"] - start >= window_s:
            out.append(Passage(doc, "video", " ".join(buf), start=start, end=seg["end"]))
            buf, start = [], None
    if buf:
        out.append(Passage(doc, "video", " ".join(buf), start=start, end=data["segments"][-1]["end"]))
    return out


def _is_toc(text: str) -> bool:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return bool(lines) and sum(bool(_TOC_LINE.search(ln)) for ln in lines) / len(lines) > 0.2


_CITATION_LINE = re.compile(r"\b(19|20)\d\d[a-z]?\.|arXiv|Proceedings|In Advances in|URL https?://")


def _is_reference_page(text: str) -> bool:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return bool(lines) and sum(bool(_CITATION_LINE.search(ln)) for ln in lines) / len(lines) > 0.25


def usable_passages(doc: str, min_chars: int = 600) -> list[Passage]:
    """Pages/windows worth generating questions from (no TOC, no reference lists)."""
    meta = sources()[doc]
    if meta["type"] == "video":
        return [p for p in video_windows(doc) if len(p.text) >= min_chars]
    pages = pdf_pages(doc)
    cutoff = None
    for n in sorted(pages):  # drop everything from the reference list on (papers)
        if n > len(pages) * 0.5 and _REF_HEADING.search(pages[n][:400]):
            cutoff = n
            break
    out = []
    for n, text in sorted(pages.items()):
        if cutoff and n >= cutoff:
            continue
        if len(text) < min_chars or _is_toc(text) or _is_reference_page(text):
            continue
        out.append(Passage(doc, "pdf", text, page=n))
    return out


def evidence_text(ev: dict) -> str:
    """Text behind a gold evidence item (one or more pages, or a time window)."""
    meta = sources()[ev["doc"]]
    if meta["type"] == "pdf":
        pages = pdf_pages(ev["doc"])
        return "\n\n".join(f"[page {p}]\n{pages.get(p, '')}" for p in ev["pages"])
    data = transcript(ev["doc"]) or {"segments": []}
    segs = [s["text"].strip() for s in data["segments"]
            if s["end"] > ev["start"] - 5 and s["start"] < ev["end"] + 5]
    return " ".join(segs)


def all_units(doc: str) -> list[Passage]:
    """Every page / window of a source (for absence checks)."""
    meta = sources()[doc]
    if meta["type"] == "video":
        return video_windows(doc)
    return [Passage(doc, "pdf", t, page=n) for n, t in sorted(pdf_pages(doc).items())]


def normalize(text: str) -> str:
    text = re.sub(r"-\s*\n\s*", "", text)        # de-hyphenate line breaks
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"[‐-―]", "-", text)
    return re.sub(r"\s+", " ", text).strip().lower()
