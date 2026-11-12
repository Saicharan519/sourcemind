"""
Cross-check candidate questions with a model from a DIFFERENT family than the
generator (DeepSeek or Mistral vs. the gpt-oss generator), then mark each one:

  status = "pass"   -> all checks passed
  status = "flag"   -> needs a human look in the review app (reason recorded)
  status = "fail"   -> dropped (clearly broken)

    python -m eval.tools.crosscheck --in eval/datasets/candidates_pilot.jsonl \
                                    --out eval/datasets/checked_pilot.jsonl

Answerable questions: the checker sees the question, the reference answer and the
cited page(s), and judges support, ambiguity and self-containment.
Unanswerable questions: the checker sees the ~8 pages of the target source most
lexically similar to the question (BM25) and judges whether any of them answers it.

If neither DEEPSEEK_API_KEY nor MISTRAL_API_KEY is set the script stops, unless --allow-same-family is given
(then gpt-oss-120b checks gpt-oss-20b's work, and every record says so).
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter

from rank_bm25 import BM25Okapi

from eval.corpus.text import all_units, evidence_text, sources
from eval.datasets.schema import dump, load
from eval.llm import GROQ_MAIN, complete_json, independent_checker

ANSWERABLE_PROMPT = """\
You are auditing a question for a document question-answering benchmark.

SOURCE: {title}
EVIDENCE (the page(s) the question was written from):
\"\"\"{evidence}\"\"\"
{history}
QUESTION: {question}
REFERENCE ANSWER: {answer}

Check strictly:
1. answer_supported: is the reference answer fully supported by the evidence?
2. answer_correct: is it a correct and complete answer to the question?
3. unambiguous: does the question have one clear correct answer (not several
   equally valid ones, not vague)?
4. self_contained: is the question understandable WITHOUT seeing the evidence
   (no "this passage", "the text", "the table above", unnamed "the author")?
   For a follow-up question, judge it together with the earlier conversation.
5. needs_all_evidence: if there are two evidence pages, is information from BOTH
   needed? If there is one page, answer true.

Return ONLY JSON: {{"answer_supported": bool, "answer_correct": bool,
"unambiguous": bool, "self_contained": bool, "needs_all_evidence": bool,
"issue": "short explanation of any problem, else empty"}}"""

UNANSWERABLE_PROMPT = """\
You are auditing a TRICK question for a document question-answering benchmark.
The question is supposed to be UNANSWERABLE from this source.

SOURCE: {title}
The pages of the source most related to the question:
\"\"\"{pages}\"\"\"

QUESTION: {question}

Check strictly:
1. answered_in_source: do these pages contain the answer (even partially or
   implied)? If yes, give the page/segment label.
2. plausible: is it a sensible question a reader of this source might ask (not
   absurd or nonsensical)?

Return ONLY JSON: {{"answered_in_source": bool, "where": "label or empty",
"plausible": bool, "issue": "short explanation, else empty"}}"""

_TOKEN = re.compile(r"[a-z0-9]+")


def _tok(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _related_units(doc: str, question: str, k: int = 8) -> str:
    units = all_units(doc)
    bm25 = BM25Okapi([_tok(u.text) for u in units])
    scores = bm25.get_scores(_tok(question))
    top = sorted(range(len(units)), key=lambda i: scores[i], reverse=True)[:k]
    return "\n\n".join(f"[{units[i].label}]\n{units[i].text[:2500]}" for i in sorted(top))


def expand_gold_pages(rec: dict) -> list[int]:
    """If a short exact answer also appears on other pages of the same PDF, those pages
    are equally valid evidence (e.g. a publication date on both the cover and the
    colophon). Add up to 3 such pages so retrieval isn't scored as a miss for finding
    the 'other' copy. Returns the pages added."""
    from eval.corpus.text import normalize, pdf_pages
    from eval.metrics.answer import is_short_gold

    if (rec["category"] not in ("factoid", "exact_id") or len(rec["gold_evidence"]) != 1
            or "pages" not in rec["gold_evidence"][0] or not is_short_gold(rec["gold_answer"])):
        return []
    gold = normalize(rec["gold_answer"])
    if len(gold) < 6:  # too generic ("8", "yes") to search for
        return []
    ev = rec["gold_evidence"][0]
    hits = [n for n, t in pdf_pages(ev["doc"]).items() if gold in normalize(t) and n not in ev["pages"]]
    if not hits or len(hits) > 3:
        return []
    ev["pages"] = sorted(set(ev["pages"]) | set(hits))
    return hits


def check(rec: dict, provider: str, model: str) -> dict:
    meta_src = sources()[rec["doc"]]
    if rec["answerable"]:
        evidence = "\n\n".join(evidence_text(ev) for ev in rec["gold_evidence"])
        history = ""
        if rec.get("history"):
            history = "EARLIER CONVERSATION:\n" + "\n".join(
                f"{t['role'].upper()}: {t['content']}" for t in rec["history"]) + "\n"
        r = complete_json(ANSWERABLE_PROMPT.format(
            title=meta_src["title"], evidence=evidence[:9000], history=history,
            question=rec["question"], answer=rec["gold_answer"]),
            provider=provider, model=model, temperature=0.0, min_interval=1.5)
        keys = ["answer_supported", "answer_correct", "unambiguous"]
        if rec["category"] != "follow_up":       # follow-ups lean on history by design
            keys.append("self_contained")
        if len(rec["gold_evidence"]) > 1:        # only two-piece questions must need both
            keys.append("needs_all_evidence")
        failed = [k for k in keys if r.get(k) is not True]
        if not failed:
            status = "pass"
        elif "answer_supported" in failed and "answer_correct" in failed:
            status = "fail"
        else:
            status = "flag"
    else:
        r = complete_json(UNANSWERABLE_PROMPT.format(
            title=meta_src["title"], pages=_related_units(rec["doc"], rec["question"])[:16000],
            question=rec["question"]),
            provider=provider, model=model, temperature=0.0, min_interval=1.5)
        failed = []
        if r.get("answered_in_source") is not False:
            failed.append("answered_in_source")
        # cross-document negatives are off-topic on purpose; only in-doc ones must be plausible
        if r.get("plausible") is not True and rec.get("meta", {}).get("kind") != "cross_doc":
            failed.append("plausible")
        status = "pass" if not failed else ("fail" if "answered_in_source" in failed
                                             and r.get("where") else "flag")
    return {"checker": model, "status": status, "failed_checks": failed,
            "issue": r.get("issue", ""), "raw": r}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-same-family", action="store_true",
                    help="use gpt-oss-120b if no Mistral key (weaker: same model family)")
    args = ap.parse_args()

    checker = independent_checker()
    if checker:
        provider, model = checker
    elif args.allow_same_family:
        provider, model = "groq", GROQ_MAIN
        print("WARNING: no DEEPSEEK/MISTRAL key - checking with gpt-oss-120b (same family as generator)")
    else:
        print("No independent checker configured. Add DEEPSEEK_API_KEY (or MISTRAL_API_KEY) to\n"
              "backend/.env, or rerun with --allow-same-family (weaker check, recorded in every record).")
        return 2

    records = load(args.inp)
    for i, rec in enumerate(records, 1):
        added = expand_gold_pages(rec)
        if added:
            rec.setdefault("meta", {})["alt_pages_added"] = added
        try:
            rec.setdefault("meta", {})["crosscheck"] = check(rec, provider, model)
        except Exception as e:
            rec.setdefault("meta", {})["crosscheck"] = {
                "checker": model, "status": "flag", "failed_checks": ["checker_error"],
                "issue": str(e)[:300]}
        cc = rec["meta"]["crosscheck"]
        print(f"[{i}/{len(records)}] {rec['id']}: {cc['status']} {cc['failed_checks'] or ''} {cc['issue'][:80]}")

    dump(records, args.out)
    print(f"\nwrote {args.out}")
    print(dict(Counter(r["meta"]["crosscheck"]["status"] for r in records)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
