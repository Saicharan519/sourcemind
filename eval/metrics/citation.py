"""
Citation metrics.

Two different things are measured:

1. **Inline citations**: the page / segment references the model writes in its
   answer text, e.g. "(page 12)" or "(segment 04:32)". These are what a reader
   checks, so precision/recall here is the user-facing citation quality.
2. **Shown citations**: the citation cards the UI displays (the `citations`
   SSE event). In v1 these are simply every retrieved chunk, so their precision
   mostly measures how much irrelevant context is shown as "sources".
"""
from __future__ import annotations

import re

from eval.metrics.retrieval import matches

_PAGE = re.compile(r"\bp(?:age|ages|p\.?|\.)\s*(\d+)(?:\s*(?:-|–|to|and|,)\s*(\d+))?", re.I)
_SEGMENT = re.compile(r"\b(?:segment|timestamp|at)\s*(\d{1,2}):(\d{2})(?::(\d{2}))?", re.I)
SEGMENT_SPAN_S = 60.0  # a cited timestamp is taken to cover the following minute


def inline_citations(answer: str, doc: str, is_video: bool) -> list[dict]:
    items = []
    if is_video:
        for m in _SEGMENT.finditer(answer):
            h_or_m, m_or_s, s = m.groups()
            secs = (int(h_or_m) * 3600 + int(m_or_s) * 60 + int(s)) if s else int(h_or_m) * 60 + int(m_or_s)
            items.append({"doc": doc, "start": float(secs), "end": float(secs) + SEGMENT_SPAN_S})
    else:
        for m in _PAGE.finditer(answer):
            a = int(m.group(1))
            b = int(m.group(2)) if m.group(2) else a
            if b < a or b - a > 10:
                b = a
            items.extend({"doc": doc, "page": p} for p in range(a, b + 1))
    # de-duplicate, keep order
    seen, out = set(), []
    for it in items:
        key = tuple(sorted(it.items()))
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out


def precision_recall(cited: list[dict], gold: list[dict]) -> tuple[float | None, float]:
    """precision = cited items matching any gold item (None if nothing cited);
    recall = gold items matched by at least one cited item."""
    precision = None
    if cited:
        precision = sum(any(matches(c, ev) for ev in gold) for c in cited) / len(cited)
    recall = (sum(any(matches(c, ev) for c in cited) for ev in gold) / len(gold)) if gold else 0.0
    return precision, recall
