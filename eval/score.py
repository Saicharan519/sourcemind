"""
Score a finished run.

    python -m eval.score --arm v1_as_shipped
    python -m eval.score --arm v1_as_shipped --allow-same-family   # no Mistral key

Per question it decides:
- correct: deterministic match for short golds; otherwise (or when the match fails)
  an LLM judge from a different model family (DeepSeek, else Mistral) rates correct / partial /
  incorrect. Judge verdicts are cached on disk.
- refused: the answer says the information is not in the source.
- retrieval: rank of the first gold page/window among what the LLM was shown.
- citations: inline page/segment references vs gold; shown citation cards vs gold.
- failure stage (wrong answers only): retrieval / generation / citation / hallucination.

Writes eval/results/<arm>/scores.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

from eval.corpus.text import sources
from eval.datasets.schema import load
from eval.llm import GROQ_MAIN, complete_json, independent_checker
from eval.metrics import answer as A
from eval.metrics import citation as C
from eval.metrics import retrieval as R
from eval.metrics.stats import bootstrap_ci, mean, percentile
from eval.paths import REPO_DIR, RESULTS_DIR

JUDGE_PROMPT = """\
You are grading an answer from a document question-answering system.

QUESTION: {question}
REFERENCE ANSWER (ground truth): {gold}
SYSTEM ANSWER: {answer}

Grade the SYSTEM ANSWER against the REFERENCE ANSWER:
- "correct": it states the same key facts as the reference (extra correct detail,
  citations or different wording are fine) and does not contradict it.
- "partial": it gets some of the key facts but misses or garbles others.
- "incorrect": it is wrong, contradicts the reference, misses the point, or says
  the information is unavailable.

Return ONLY JSON: {{"verdict": "correct" | "partial" | "incorrect", "reason": "short"}}"""

VERDICT_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


def judge(q: dict, answer: str, provider: str, model: str) -> tuple[float, str]:
    r = complete_json(JUDGE_PROMPT.format(question=q["question"], gold=q["gold_answer"],
                                          answer=answer[:4000]),
                      provider=provider, model=model, temperature=0.0, min_interval=1.5,
                      cache_tag="judge-v1")
    verdict = str(r.get("verdict", "incorrect")).lower()
    return VERDICT_SCORE.get(verdict, 0.0), verdict


def score_one(q: dict, rec: dict, is_video: bool, judge_cfg) -> dict:
    out = {"id": q["id"], "category": q["category"], "doc": q["doc"],
           "answerable": q["answerable"], "latency_s": rec.get("latency_s"),
           "error": rec.get("error"), "query_type": rec.get("query_type"),
           "degraded": rec.get("degraded") or []}
    answer = rec.get("answer") or ""
    out["refused"] = A.is_refusal(answer)

    if not q["answerable"]:
        out["correct"] = 1.0 if out["refused"] else 0.0
        out["correct_method"] = "refusal"
        out["failure"] = None if out["refused"] else "hallucination"
        return out

    gold = q["gold_evidence"]
    out["retrieval"] = R.score_question(rec.get("retrieved") or [], gold)

    if out["refused"]:
        out["correct"], out["correct_method"] = 0.0, "refusal"
    elif A.is_short_gold(q["gold_answer"]) and A.contains_match(q["gold_answer"], answer):
        out["correct"], out["correct_method"] = 1.0, "contains"
    elif judge_cfg:
        out["correct"], verdict = judge(q, answer, *judge_cfg)
        out["correct_method"] = f"judge:{verdict}"
    else:
        out["correct"], out["correct_method"] = None, "unjudged"

    cited = C.inline_citations(answer, q["doc"], is_video)
    out["inline_cite_precision"], out["inline_cite_recall"] = C.precision_recall(cited, gold)
    out["n_inline_citations"] = len(cited)
    shown_p, _ = C.precision_recall(rec.get("retrieved") or [], gold)
    out["shown_cite_precision"] = shown_p

    if out["correct"] is None or out["correct"] >= 1.0:
        out["failure"] = "citation" if out["correct"] == 1.0 and out["inline_cite_recall"] == 0 and cited else None
    elif not out["retrieval"]["hit@any"]:
        out["failure"] = "retrieval"
    else:
        out["failure"] = "generation"
    return out


def aggregate(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]

    def ci(xs):
        c = bootstrap_ci(xs)
        return [round(c[0], 3), round(c[1], 3)] if c else None

    def m(xs):
        v = mean(xs)
        return round(v, 3) if v is not None else None

    agg = {"n": len(rows), "n_answerable": len(ans), "n_unanswerable": len(una),
           "errors": sum(bool(r.get("error")) for r in rows),
           "degraded": sum(bool(r.get("degraded")) for r in rows)}
    if ans:
        ret = [r["retrieval"] for r in ans]
        for k in ("hit@1", "hit@3", "hit@5"):
            xs = [float(x[k]) for x in ret]
            agg[k], agg[f"{k}_ci95"] = m(xs), ci(xs)
        rr = [x["rr"] for x in ret]
        agg["mrr"], agg["mrr_ci95"] = m(rr), ci(rr)
        agg["evidence_recall@5"] = m([x["evidence_recall@5"] for x in ret])
        judged = [r["correct"] for r in ans if r["correct"] is not None]
        strict = [1.0 if c >= 1.0 else 0.0 for c in judged]
        agg["answer_correct"], agg["answer_correct_ci95"] = m(strict), ci(strict)
        agg["answer_score_with_partial"] = m(judged)
        agg["false_refusal_rate"] = m([float(r["refused"]) for r in ans])
        prec = [r["inline_cite_precision"] for r in ans if r["inline_cite_precision"] is not None]
        agg["inline_cite_precision"] = m(prec)
        agg["inline_cite_recall"] = m([r["inline_cite_recall"] for r in ans])
        agg["answers_with_inline_citation"] = m([float(r["n_inline_citations"] > 0) for r in ans])
        shown = [r["shown_cite_precision"] for r in ans if r["shown_cite_precision"] is not None]
        agg["shown_cite_precision"] = m(shown)
    if una:
        cr = [float(r["refused"]) for r in una]
        agg["correct_refusal_rate"], agg["correct_refusal_rate_ci95"] = m(cr), ci(cr)
    lat = [r["latency_s"] for r in rows if r.get("latency_s") is not None]
    agg["latency_p50_s"] = round(percentile(lat, 0.5), 2) if lat else None
    agg["latency_p95_s"] = round(percentile(lat, 0.95), 2) if lat else None
    fails = defaultdict(int)
    for r in rows:
        if r.get("failure"):
            fails[r["failure"]] += 1
    agg["failures"] = dict(fails)
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--allow-same-family", action="store_true",
                    help="judge with gpt-oss-120b if no Mistral key (weaker)")
    args = ap.parse_args()

    run_dir = RESULTS_DIR / args.arm
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    records = {json.loads(l)["id"]: json.loads(l)
               for l in (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines() if l}
    questions = {q["id"]: q for q in load(REPO_DIR / run["dataset"])}

    checker = independent_checker()
    if checker:
        judge_cfg = checker
    elif args.allow_same_family:
        judge_cfg = ("groq", GROQ_MAIN)
        print("WARNING: judging with gpt-oss-120b (same family as the system under test)")
    else:
        judge_cfg = None
        print("note: no DEEPSEEK/MISTRAL key - long answers left unjudged "
              "(add a key to backend/.env, or pass --allow-same-family)")

    meta = sources()
    rows = []
    for qid, rec in records.items():
        q = questions.get(qid)
        if not q:
            continue
        rows.append(score_one(q, rec, meta[q["doc"]]["type"] == "video", judge_cfg))

    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    result = {
        "arm": args.arm,
        "run": {k: run.get(k) for k in ("description", "backend_git_sha", "backend_dirty", "dataset",
                                        "dataset_sha256", "models", "settings", "started_at",
                                        "finished_at")},
        "judge": judge_cfg[1] if judge_cfg else None,
        "overall": aggregate(rows),
        "by_category": {c: aggregate(rs) for c, rs in sorted(by_cat.items())},
        "questions": rows,
    }
    (run_dir / "scores.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    o = result["overall"]
    print(json.dumps({k: v for k, v in o.items() if not k.endswith("_ci95")}, indent=2))
    print(f"\nscores -> {(run_dir / 'scores.json').relative_to(REPO_DIR)}")
    print("next: python -m eval.report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
