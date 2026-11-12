"""
Build eval/REPORT.md from every eval/results/*/scores.json.

    python -m eval.report

Arms evaluated on the same dataset are compared question-by-question against the
baseline arm (`v1_as_shipped` if present) with a paired bootstrap.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

from eval.metrics.stats import paired_bootstrap
from eval.paths import REPORT, RESULTS_DIR

BASELINE = "v1_as_shipped"
HEADLINE = [
    ("hit@5", "Retrieval hit@5", "pct"),
    ("mrr", "MRR", "num"),
    ("evidence_recall@5", "Evidence recall@5", "pct"),
    ("answer_correct", "Answer correct", "pct"),
    ("correct_refusal_rate", "Correct refusals (unanswerable)", "pct"),
    ("false_refusal_rate", "False refusals (answerable)", "pct"),
    ("inline_cite_precision", "Inline citation precision", "pct"),
    ("inline_cite_recall", "Inline citation recall", "pct"),
    ("shown_cite_precision", "Shown citation-card precision", "pct"),
    ("latency_p50_s", "Latency p50 (s)", "num"),
    ("latency_p95_s", "Latency p95 (s)", "num"),
]


def fmt(v, kind, ci=None):
    if v is None:
        return "—"
    s = f"{v * 100:.0f}%" if kind == "pct" else f"{v:.2f}"
    if ci:
        lo, hi = ci
        s += f" ({lo * 100:.0f}–{hi * 100:.0f})" if kind == "pct" else f" ({lo:.2f}–{hi:.2f})"
    return s


def main() -> int:
    arms = {}
    for p in sorted(RESULTS_DIR.glob("*/scores.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        arms[s["arm"]] = s
    if not arms:
        print("no scores found - run eval.run_eval and eval.score first")
        return 1
    order = sorted(arms, key=lambda a: (a != BASELINE, a))

    L = ["# SourceMind benchmark report", "",
         f"_Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by `python -m eval.report`. "
         "Do not edit by hand._", "",
         "Numbers in parentheses are 95% bootstrap confidence intervals. "
         "With small question counts the intervals are wide, so small differences are noise. "
         "Per-category numbers are descriptive only.", ""]

    L += ["## Arms", "", "| Arm | Dataset | Questions | Models (main / fast) | Judge | Code |",
          "|---|---|---|---|---|---|"]
    for a in order:
        r = arms[a]["run"]
        sha = (r.get("backend_git_sha") or "?")[:7] + (" (dirty)" if r.get("backend_dirty") else "")
        L.append(f"| `{a}` | `{r.get('dataset')}` (sha256 `{(r.get('dataset_sha256') or '')[:10]}`) | "
                 f"{arms[a]['overall']['n']} | {r['models']['main']} / {r['models']['fast']} | "
                 f"{arms[a].get('judge') or 'none'} | `{sha}` |")
    L.append("")

    L += ["## Headline metrics", "", "| Metric | " + " | ".join(f"`{a}`" for a in order) + " |",
          "|---|" + "---|" * len(order)]
    for key, label, kind in HEADLINE:
        cells = [fmt(arms[a]["overall"].get(key), kind, arms[a]["overall"].get(f"{key}_ci95"))
                 for a in order]
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L.append("")

    if BASELINE in arms and len(arms) > 1:
        base = {q["id"]: q for q in arms[BASELINE]["questions"]}
        L += [f"## Paired comparison vs `{BASELINE}`", "",
              "| Arm | Metric | Δ (arm − baseline) | 95% CI | P(arm better) |", "|---|---|---|---|---|"]
        for a in order[1:]:
            if arms[a]["run"].get("dataset_sha256") != arms[BASELINE]["run"].get("dataset_sha256"):
                L.append(f"| `{a}` | — | different dataset, not comparable | | |")
                continue
            other = {q["id"]: q for q in arms[a]["questions"]}
            ids = [i for i in base if i in other and base[i]["answerable"]]
            for metric, get in (("hit@5", lambda q: float(q["retrieval"]["hit@5"])),
                                ("answer correct", lambda q: float((q["correct"] or 0) >= 1))):
                pb = paired_bootstrap([get(base[i]) for i in ids], [get(other[i]) for i in ids])
                if pb:
                    lo, hi = pb["ci95"]
                    L.append(f"| `{a}` | {metric} | {pb['mean_diff'] * 100:+.0f} pts | "
                             f"{lo * 100:+.0f} to {hi * 100:+.0f} | {pb['p_better']:.2f} |")
        L.append("")

    for a in order:
        s = arms[a]
        L += [f"## `{a}`: by category", "",
              "| Category | n | hit@5 | Answer correct | Refusal (correct / false) | p50 latency |",
              "|---|---|---|---|---|---|"]
        for cat, agg in s["by_category"].items():
            refusal = (fmt(agg.get("correct_refusal_rate"), "pct") if cat == "unanswerable"
                       else fmt(agg.get("false_refusal_rate"), "pct"))
            L.append(f"| {cat} | {agg['n']} | {fmt(agg.get('hit@5'), 'pct')} | "
                     f"{fmt(agg.get('answer_correct'), 'pct')} | {refusal} | "
                     f"{fmt(agg.get('latency_p50_s'), 'num')} |")
        f = s["overall"].get("failures", {})
        L += ["", "Failure attribution (wrong answers only): " +
              (", ".join(f"**{k}** {v}" for k, v in sorted(f.items(), key=lambda x: -x[1])) or "none"),
              "", "- *retrieval*: no gold page/window reached the LLM",
              "- *generation*: gold evidence was retrieved but the answer was wrong or a false refusal",
              "- *citation*: answer correct but its inline citations miss the gold page",
              "- *hallucination*: answered a question whose answer is not in the source", ""]

    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
