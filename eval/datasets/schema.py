"""
Benchmark question schema + validator. Pure Python (used in CI).

One JSON object per line:

  id            unique string
  category      one of CATEGORIES
  doc           the source the question is asked against (retrieval is scoped to it)
  question      the question text
  answerable    bool
  gold_answer   short reference answer (null when unanswerable)
  gold_evidence list of {"doc","pages":[...]} or {"doc","start","end"} (empty when unanswerable)
  history       optional prior turns [{"role","content"}], used by follow_up
  meta          free-form provenance (generator model, quotes, cross-check, review)
"""
from __future__ import annotations

import json
from pathlib import Path

CATEGORIES = (
    "factoid", "exact_id", "paraphrase", "comparative",
    "multi_hop", "unanswerable", "follow_up",
)
REQUIRED = ("id", "category", "doc", "question", "answerable", "gold_answer", "gold_evidence")


def validate(rec: dict, known_docs: set[str] | None = None) -> list[str]:
    errs = []
    for k in REQUIRED:
        if k not in rec:
            errs.append(f"missing field {k}")
    if errs:
        return errs
    if rec["category"] not in CATEGORIES:
        errs.append(f"bad category {rec['category']!r}")
    if not isinstance(rec["question"], str) or len(rec["question"].strip()) < 8:
        errs.append("question too short")
    if known_docs is not None and rec["doc"] not in known_docs:
        errs.append(f"unknown doc {rec['doc']!r}")
    if rec["answerable"]:
        if not rec["gold_answer"]:
            errs.append("answerable question needs gold_answer")
        if not rec["gold_evidence"]:
            errs.append("answerable question needs gold_evidence")
    else:
        if rec["gold_evidence"]:
            errs.append("unanswerable question must have empty gold_evidence")
        if rec["category"] != "unanswerable":
            errs.append("only the unanswerable category may be unanswerable")
    for ev in rec["gold_evidence"] or []:
        if "doc" not in ev:
            errs.append("evidence item missing doc")
        elif known_docs is not None and ev["doc"] not in known_docs:
            errs.append(f"evidence references unknown doc {ev['doc']!r}")
        if "pages" in ev:
            if not ev["pages"] or not all(isinstance(p, int) and p >= 1 for p in ev["pages"]):
                errs.append("pages must be positive ints")
        elif "start" in ev and "end" in ev:
            if not (0 <= ev["start"] < ev["end"]):
                errs.append("time window must satisfy 0 <= start < end")
        else:
            errs.append("evidence needs pages or start/end")
    if rec["category"] == "follow_up" and not rec.get("history"):
        errs.append("follow_up needs history")
    return errs


def load(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dump(records: list[dict], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def validate_file(path: str | Path, known_docs: set[str] | None = None) -> list[str]:
    problems, seen = [], set()
    for i, rec in enumerate(load(path), start=1):
        if rec.get("id") in seen:
            problems.append(f"line {i}: duplicate id {rec.get('id')}")
        seen.add(rec.get("id"))
        problems += [f"line {i} ({rec.get('id')}): {e}" for e in validate(rec, known_docs)]
    return problems
