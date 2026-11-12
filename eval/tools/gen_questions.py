"""
Generate candidate benchmark questions from RANDOM pages / transcript windows.

    python -m eval.tools.gen_questions --plan pilot --out eval/datasets/candidates_pilot.jsonl
    python -m eval.tools.gen_questions --plan full  --out eval/datasets/candidates_full.jsonl

Uses the fast Groq model (model-routing rule: helper work stays on the cheap model).
Every answerable candidate must quote its evidence verbatim; a quote that can't be
found on the cited page is rejected immediately (cheap, deterministic check). The
surviving candidates then go through crosscheck.py (a different model family).
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from collections import Counter

from eval.corpus.text import Passage, normalize, sources, usable_passages
from eval.datasets.schema import dump, validate
from eval.llm import GROQ_FAST, complete_json

# How many candidates to request per category. Over-generate (~1.4x) because the
# quote check, the cross-check and human review all drop some.
PLANS = {
    "pilot": {"factoid": 10, "exact_id": 6, "paraphrase": 6, "comparative": 5,
              "multi_hop": 5, "unanswerable_in_doc": 5, "unanswerable_cross_doc": 5,
              "follow_up": 4},
    "full": {"factoid": 150, "exact_id": 85, "paraphrase": 85, "comparative": 65,
             "multi_hop": 65, "unanswerable_in_doc": 55, "unanswerable_cross_doc": 55,
             "follow_up": 45},
}

_ID_LIKE = re.compile(
    r"\b(?:[A-Z]{2,}[.\-]?[A-Z]{0,4}-?\d+(?:[.\-]\d+)*|\d+(?:\.\d+)?\s?%|\$\s?\d[\d,.]*|"
    r"(?:Section|Table|Figure|Appendix)\s+[A-Z0-9][\w.]*)\b"
)

COMMON_RULES = """\
Rules:
- The question must make sense to someone who has NOT seen the passage: never say
  "the passage", "the text", "this section", "the author" without naming what you mean.
- The question must have exactly one correct answer, found in the passage.
- "evidence_quote" must be copied VERBATIM from the passage (5 to 30 words).
- Do not ask about page numbers, figure numbers or citation markers.
Return ONLY a JSON object."""

PROMPTS = {
    "factoid": """\
You are writing evaluation questions for a document question-answering system.
Source: {title}

PASSAGE:
\"\"\"{text}\"\"\"

Write ONE factual question whose answer is a short fact stated in the passage
(a name, number, date, definition or short phrase; at most 8 words).
{rules}
JSON keys: "question", "answer", "evidence_quote".""",

    "exact_id": """\
You are writing evaluation questions that test exact keyword lookup.
Source: {title}

PASSAGE:
\"\"\"{text}\"\"\"

Write ONE question whose answer is an exact identifier, code, number, percentage
or technical term that appears verbatim in the passage (e.g. a control ID like
"PR.AA-01", a value like "8 characters" or "4.1%"). Either the question names the
identifier and asks what it says, or it asks for the identifier/value itself.
The answer must be at most 8 words.
{rules}
JSON keys: "question", "answer", "evidence_quote".""",

    "paraphrase": """\
You are writing evaluation questions that test semantic (not keyword) search.
Source: {title}

PASSAGE:
\"\"\"{text}\"\"\"

Write ONE question answerable from the passage, but phrased with DIFFERENT
vocabulary: use synonyms and avoid reusing the passage's distinctive words or
phrases. The answer should be one or two sentences.
{rules}
JSON keys: "question", "answer", "evidence_quote".""",

    "comparative": """\
You are writing evaluation questions that require comparing two passages
from the same source: {title}

PASSAGE A:
\"\"\"{text_a}\"\"\"

PASSAGE B:
\"\"\"{text_b}\"\"\"

Write ONE question that asks to compare or contrast something described in
passage A with something described in passage B (both passages must be needed).
Name the actual things involved (e.g. "How does X differ from Y?"). NEVER write
"passage A", "passage B", "which passage", "the first/second passage" in the
question or the answer: the reader will not see any passages.
If the passages share nothing meaningful to compare, return {{"skip": true}}.
The answer should be one to three sentences.
{rules}
JSON keys: "question", "answer", "quote_a", "quote_b" (each verbatim from its passage).""",

    "multi_hop": """\
You are writing evaluation questions that need two reasoning steps across two
passages of the same source: {title}

PASSAGE A:
\"\"\"{text_a}\"\"\"

PASSAGE B:
\"\"\"{text_b}\"\"\"

Write ONE question where answering requires a fact from passage A AND a fact from
passage B (for example, step 1 finds something in A, step 2 uses it with B).
Name the actual things involved (e.g. "How does X differ from Y?"). NEVER write
"passage A", "passage B", "which passage", "the first/second passage" in the
question or the answer: the reader will not see any passages.
If no natural question links them, return {{"skip": true}}.
The answer should be one to three sentences.
{rules}
JSON keys: "question", "answer", "quote_a", "quote_b" (each verbatim from its passage).""",

    "unanswerable_in_doc": """\
You are writing TRICK evaluation questions to test whether a question-answering
system admits when information is missing.
Source: {title}

Here is one passage from the source, for topic context:
\"\"\"{text}\"\"\"

Write ONE specific, plausible question that someone reading this source might
ask, on its topic, but that this source almost certainly does NOT answer (for
example: a specific number, date, name or detail the document would not contain).
It must not be answerable from the passage above.
Return ONLY a JSON object with key "question".""",

    "follow_up": """\
You are writing a two-turn conversation for evaluating a chat assistant.
Source: {title}

PASSAGE:
\"\"\"{text}\"\"\"

Write a first question (q1) with its answer (a1), then a FOLLOW-UP question (q2)
that only makes sense after q1: it must refer back with a pronoun or ellipsis
("it", "they", "that method", "what about ..."). Both answers must come from the
passage; q2's answer must be different from a1 and short (at most 2 sentences).
{rules}
JSON keys: "q1", "a1", "q2", "answer", "evidence_quote" (supports the answer to q2).""",
}


_NOT_SELF_CONTAINED = re.compile(
    r"passages?|excerpt|the (?:above|following|given) text|the text|"
    r"(?:first|second|third) case|the table above|in this (?:section|excerpt)",
    re.I)


def self_contained(*texts: str) -> bool:
    return not any(_NOT_SELF_CONTAINED.search(t or "") for t in texts)


def quote_found(quote: str, passage_text: str) -> bool:
    q, t = normalize(quote), normalize(passage_text)
    if len(q) < 12:
        return False
    if q in t:
        return True
    # tolerate small extraction differences: require a long contiguous overlap
    from difflib import SequenceMatcher
    m = SequenceMatcher(None, q, t, autojunk=False).find_longest_match(0, len(q), 0, len(t))
    return m.size >= 0.8 * len(q)


def _clip(text: str, limit: int = 3500) -> str:
    return text if len(text) <= limit else text[:limit] + " ..."


class Generator:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.meta = sources()
        self.passages = {k: usable_passages(k) for k in self.meta}
        self.passages = {k: v for k, v in self.passages.items() if v}
        self.counter = Counter()

    def _new_id(self, cat: str) -> str:
        self.counter[cat] += 1
        return f"c-{cat}-{self.counter[cat]:03d}"

    def _ask(self, cat: str, **fields) -> dict | None:
        prompt = PROMPTS[cat].format(rules=COMMON_RULES, **fields)
        try:
            return complete_json(prompt, provider="groq", model=GROQ_FAST, temperature=0.7,
                                 cache_tag=fields.get("_tag", ""))
        except Exception as e:
            print(f"  [{cat}] generation failed: {e}")
            return None

    def _docs_cycle(self, allow_video: bool = True):
        keys = [k for k in self.passages if allow_video or self.meta[k]["type"] == "pdf"]
        while True:
            self.rng.shuffle(keys)
            yield from keys

    def single(self, cat: str, n: int) -> list[dict]:
        out, attempts = [], 0
        docs = self._docs_cycle(allow_video=(cat != "exact_id"))
        while len(out) < n and attempts < n * 4:
            attempts += 1
            doc = next(docs)
            pool = self.passages[doc]
            if cat == "exact_id":
                pool = [p for p in pool if len(_ID_LIKE.findall(p.text)) >= 2] or pool
            p: Passage = self.rng.choice(pool)
            r = self._ask(cat, title=self.meta[doc]["title"], text=_clip(p.text),
                          _tag=f"{cat}:{p.label}:{attempts}")
            if not r or not r.get("question") or not r.get("answer"):
                continue
            if not self_contained(r["question"]):
                print(f"  [{cat}] rejected: not self-contained")
                continue
            quote = r.get("evidence_quote", "")
            if not quote_found(quote, p.text):
                print(f"  [{cat}] rejected: quote not on {p.label}")
                continue
            out.append({
                "id": self._new_id(cat), "category": cat, "doc": doc,
                "question": r["question"].strip(), "answerable": True,
                "gold_answer": str(r["answer"]).strip(), "gold_evidence": [p.evidence()],
                "meta": {"generator": GROQ_FAST, "quotes": [quote], "source_unit": p.label},
            })
            print(f"  [{cat}] {len(out)}/{n} from {p.label}")
        return out

    def pair(self, cat: str, n: int) -> list[dict]:
        out, attempts = [], 0
        docs = self._docs_cycle()
        while len(out) < n and attempts < n * 5:
            attempts += 1
            doc = next(docs)
            pool = self.passages[doc]
            if len(pool) < 3:
                continue
            a, b = self.rng.sample(pool, 2)
            if a.kind == "pdf" and abs(a.page - b.page) < 2:
                continue  # force genuinely separate evidence
            a, b = sorted((a, b), key=lambda p: p.page or p.start)
            r = self._ask(cat, title=self.meta[doc]["title"], text_a=_clip(a.text, 2500),
                          text_b=_clip(b.text, 2500), _tag=f"{cat}:{a.label}:{b.label}")
            if not r or r.get("skip") or not r.get("question") or not r.get("answer"):
                continue
            if not self_contained(r["question"], str(r["answer"])):
                print(f"  [{cat}] rejected: not self-contained")
                continue
            if not (quote_found(r.get("quote_a", ""), a.text) and quote_found(r.get("quote_b", ""), b.text)):
                print(f"  [{cat}] rejected: quotes not found on {a.label} / {b.label}")
                continue
            out.append({
                "id": self._new_id(cat), "category": cat, "doc": doc,
                "question": r["question"].strip(), "answerable": True,
                "gold_answer": str(r["answer"]).strip(),
                "gold_evidence": [a.evidence(), b.evidence()],
                "meta": {"generator": GROQ_FAST, "quotes": [r["quote_a"], r["quote_b"]],
                         "source_unit": f"{a.label} + {b.label}"},
            })
            print(f"  [{cat}] {len(out)}/{n} from {a.label} + {b.label}")
        return out

    def unanswerable_in_doc(self, n: int) -> list[dict]:
        out, attempts = [], 0
        docs = self._docs_cycle()
        while len(out) < n and attempts < n * 3:
            attempts += 1
            doc = next(docs)
            p = self.rng.choice(self.passages[doc])
            r = self._ask("unanswerable_in_doc", title=self.meta[doc]["title"],
                          text=_clip(p.text, 2500), _tag=f"un:{p.label}:{attempts}")
            if not r or not r.get("question"):
                continue
            out.append({
                "id": self._new_id("unanswerable"), "category": "unanswerable", "doc": doc,
                "question": r["question"].strip(), "answerable": False,
                "gold_answer": None, "gold_evidence": [],
                "meta": {"generator": GROQ_FAST, "kind": "in_doc", "source_unit": p.label},
            })
            print(f"  [unanswerable/in-doc] {len(out)}/{n} for {doc}")
        return out

    def unanswerable_cross_doc(self, n: int) -> list[dict]:
        """Generate a factoid from doc B, then ask it against a doc from another group."""
        out, attempts = [], 0
        groups = {k: self.meta[k]["group"] for k in self.passages}
        docs = self._docs_cycle()
        while len(out) < n and attempts < n * 3:
            attempts += 1
            src = next(docs)
            targets = [k for k in self.passages if groups[k] != groups[src]]
            target = self.rng.choice(targets)
            p = self.rng.choice(self.passages[src])
            r = self._ask("factoid", title=self.meta[src]["title"], text=_clip(p.text),
                          _tag=f"xdoc:{p.label}:{attempts}")
            if not r or not r.get("question"):
                continue
            out.append({
                "id": self._new_id("unanswerable"), "category": "unanswerable", "doc": target,
                "question": r["question"].strip(), "answerable": False,
                "gold_answer": None, "gold_evidence": [],
                "meta": {"generator": GROQ_FAST, "kind": "cross_doc", "borrowed_from": p.label},
            })
            print(f"  [unanswerable/cross-doc] {len(out)}/{n}: {p.label} -> {target}")
        return out

    def follow_up(self, n: int) -> list[dict]:
        out, attempts = [], 0
        docs = self._docs_cycle()
        while len(out) < n and attempts < n * 4:
            attempts += 1
            doc = next(docs)
            p = self.rng.choice(self.passages[doc])
            r = self._ask("follow_up", title=self.meta[doc]["title"], text=_clip(p.text),
                          _tag=f"fu:{p.label}:{attempts}")
            if not r or not all(r.get(k) for k in ("q1", "a1", "q2", "answer")):
                continue
            if not self_contained(r["q1"]):
                print("  [follow_up] rejected: not self-contained")
                continue
            if not quote_found(r.get("evidence_quote", ""), p.text):
                print(f"  [follow_up] rejected: quote not on {p.label}")
                continue
            out.append({
                "id": self._new_id("follow_up"), "category": "follow_up", "doc": doc,
                "question": r["q2"].strip(), "answerable": True,
                "gold_answer": str(r["answer"]).strip(), "gold_evidence": [p.evidence()],
                "history": [{"role": "user", "content": r["q1"].strip()},
                            {"role": "assistant", "content": str(r["a1"]).strip()}],
                "meta": {"generator": GROQ_FAST, "quotes": [r["evidence_quote"]], "source_unit": p.label},
            })
            print(f"  [follow_up] {len(out)}/{n} from {p.label}")
        return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", choices=sorted(PLANS), default="pilot")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    g = Generator(args.seed)
    missing = [k for k, m in g.meta.items() if k not in g.passages]
    if missing:
        print(f"note: no usable text for {missing} (transcripts missing?) - skipping them")
    plan = PLANS[args.plan]
    records: list[dict] = []
    for cat in ("factoid", "exact_id", "paraphrase"):
        print(f"== {cat}")
        records += g.single(cat, plan[cat])
    for cat in ("comparative", "multi_hop"):
        print(f"== {cat}")
        records += g.pair(cat, plan[cat])
    print("== unanswerable")
    records += g.unanswerable_in_doc(plan["unanswerable_in_doc"])
    records += g.unanswerable_cross_doc(plan["unanswerable_cross_doc"])
    print("== follow_up")
    records += g.follow_up(plan["follow_up"])

    known = set(g.meta)
    bad = [(r["id"], validate(r, known)) for r in records if validate(r, known)]
    for rid, errs in bad:
        print(f"invalid {rid}: {errs}")
    records = [r for r in records if not validate(r, known)]
    dump(records, args.out)
    print(f"\nwrote {len(records)} candidates -> {args.out}")
    print(dict(Counter(r["category"] for r in records)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
