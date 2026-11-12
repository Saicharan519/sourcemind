"""
Retrieval metrics, judged against page / time-window gold evidence.

A retrieved item matches a gold evidence item when it is from the same source and
  - documents: its page is one of the gold pages, or
  - videos: its time span overlaps the gold window (with a small tolerance).

Items are the ranked list of what the system put in front of the LLM.
"""
from __future__ import annotations

VIDEO_TOLERANCE_S = 5.0


def matches(item: dict, ev: dict) -> bool:
    if item.get("doc") != ev.get("doc"):
        return False
    if "pages" in ev:
        return item.get("page") in ev["pages"]
    if item.get("start") is None or item.get("end") is None:
        return False
    return (item["start"] < ev["end"] + VIDEO_TOLERANCE_S
            and item["end"] > ev["start"] - VIDEO_TOLERANCE_S)


def first_hit_rank(retrieved: list[dict], gold: list[dict]) -> int | None:
    """1-based rank of the first retrieved item matching ANY gold evidence."""
    for rank, item in enumerate(retrieved, start=1):
        if any(matches(item, ev) for ev in gold):
            return rank
    return None


def evidence_recall_at_k(retrieved: list[dict], gold: list[dict], k: int) -> float:
    """Fraction of gold evidence items matched by at least one of the top-k items.
    Matters for comparative / multi-hop questions, which need two pieces."""
    if not gold:
        return 0.0
    top = retrieved[:k]
    return sum(any(matches(it, ev) for it in top) for ev in gold) / len(gold)


def score_question(retrieved: list[dict], gold: list[dict]) -> dict:
    rank = first_hit_rank(retrieved, gold)
    return {
        "first_hit_rank": rank,
        "hit@1": rank is not None and rank <= 1,
        "hit@3": rank is not None and rank <= 3,
        "hit@5": rank is not None and rank <= 5,
        "hit@any": rank is not None,
        "rr": 1.0 / rank if rank else 0.0,
        "evidence_recall@5": evidence_recall_at_k(retrieved, gold, 5),
        "n_retrieved": len(retrieved),
    }
