"""
Turn a cross-checked (and reviewed) candidate file into benchmark datasets.

    # pilot: everything that is usable goes into one small file
    python -m eval.tools.split --in eval/datasets/checked_pilot.jsonl --mode pilot

    # full: stratified frozen test set + dev set + CI smoke set
    python -m eval.tools.split --in eval/datasets/checked_full.jsonl --mode full

Which records are usable:
  - cross-check "fail"                         -> dropped
  - human review "reject"                      -> dropped
  - human review "keep"/"edit"                 -> kept
  - flagged by the cross-check, not reviewed   -> dropped (always)
  - passed, but unanswerable / multi_hop and not reviewed
                                               -> dropped in full mode (logged);
                                                  kept in pilot mode only with --pilot-keep-unreviewed
  - cross-check "pass", not reviewed           -> kept

The test set is written once with a sha256 lock file; re-running refuses to
overwrite it (use --force only before any result has been reported).
"""
from __future__ import annotations

import argparse
import hashlib
import random
import sys
from collections import Counter, defaultdict

from eval.datasets.schema import dump, load
from eval.paths import DATASETS_DIR

TEST_TARGET = {"factoid": 35, "exact_id": 20, "paraphrase": 20, "comparative": 15,
               "multi_hop": 15, "unanswerable": 25, "follow_up": 10}
SMOKE_PER_CATEGORY = 3
ALWAYS_REVIEW = {"unanswerable", "multi_hop"}


def usable(r: dict, keep_unreviewed: bool) -> tuple[bool, str]:
    m = r.get("meta", {})
    cc, rv = m.get("crosscheck", {}), m.get("review")
    if cc.get("status") == "fail":
        return False, "crosscheck_fail"
    if rv:
        return (rv["decision"] != "reject"), f"review_{rv['decision']}"
    if cc.get("status") != "pass":
        return False, "flagged_unreviewed"   # flagged questions always need a human
    required = r["category"] in ALWAYS_REVIEW
    if required and not keep_unreviewed:
        return False, "needs_review"
    return True, "auto_pass" if not required else "unreviewed_kept"


def renumber(records: list[dict], prefix: str) -> list[dict]:
    out = []
    for i, r in enumerate(records, 1):
        r = dict(r)
        r.setdefault("meta", {})["candidate_id"] = r["id"]
        r["id"] = f"{prefix}-{i:03d}"
        out.append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--mode", choices=("pilot", "full"), required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pilot-keep-unreviewed", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    records = load(args.inp)
    kept, reasons = [], Counter()
    for r in records:
        ok, why = usable(r, args.mode == "pilot" and args.pilot_keep_unreviewed)
        reasons[why] += 1
        if ok:
            kept.append(r)
    print("triage:", dict(reasons))

    if args.mode == "pilot":
        out = DATASETS_DIR / "pilot.jsonl"
        dump(renumber(kept, "p"), out)
        print(f"pilot: {len(kept)} questions -> {out}")
        print(dict(Counter(r["category"] for r in kept)))
        return 0

    test_path, lock = DATASETS_DIR / "test.jsonl", DATASETS_DIR / "test.sha256"
    if test_path.exists() and not args.force:
        print(f"{test_path} already exists and is frozen. Refusing to overwrite (use --force "
              "only if no result from it has been reported).")
        return 1

    rng = random.Random(args.seed)
    by_cat = defaultdict(list)
    for r in kept:
        by_cat[r["category"]].append(r)
    test, dev = [], []
    for cat, rs in by_cat.items():
        rng.shuffle(rs)
        # at most the target, and never more than 40% of a category (dev needs the rest)
        n_test = min(TEST_TARGET.get(cat, 0), round(len(rs) * 0.4))
        test += rs[:n_test]
        dev += rs[n_test:]
        print(f"  {cat:13s} test {n_test:3d} / target {TEST_TARGET.get(cat, 0):3d}   dev {len(rs) - n_test:3d}")

    test, dev = renumber(test, "t"), renumber(dev, "d")
    dump(test, test_path)
    dump(dev, DATASETS_DIR / "dev.jsonl")
    digest = hashlib.sha256(test_path.read_bytes()).hexdigest()
    lock.write_text(f"{digest}  test.jsonl\n", encoding="utf-8")
    smoke = []
    for cat in TEST_TARGET:
        smoke += [r for r in dev if r["category"] == cat][:SMOKE_PER_CATEGORY]
    dump(smoke, DATASETS_DIR / "smoke.jsonl")
    print(f"test {len(test)}  dev {len(dev)}  smoke {len(smoke)}; test sha256 {digest[:12]}... locked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
