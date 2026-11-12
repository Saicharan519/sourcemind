from eval.metrics import answer as A
from eval.metrics import citation as C
from eval.metrics import retrieval as R
from eval.metrics.stats import bootstrap_ci, paired_bootstrap, percentile

PDF_GOLD = [{"doc": "nist_csf", "pages": [12]}]
TWO_GOLD = [{"doc": "d", "pages": [3]}, {"doc": "d", "pages": [9]}]


# -- retrieval -------------------------------------------------------------

def test_page_match_requires_same_doc_and_page():
    assert R.matches({"doc": "nist_csf", "page": 12}, PDF_GOLD[0])
    assert not R.matches({"doc": "nist_csf", "page": 13}, PDF_GOLD[0])
    assert not R.matches({"doc": "other", "page": 12}, PDF_GOLD[0])


def test_video_window_overlap_with_tolerance():
    ev = {"doc": "v", "start": 100.0, "end": 190.0}
    assert R.matches({"doc": "v", "start": 180.0, "end": 260.0}, ev)
    assert R.matches({"doc": "v", "start": 192.0, "end": 260.0}, ev)      # within 5 s tolerance
    assert not R.matches({"doc": "v", "start": 200.0, "end": 260.0}, ev)
    assert not R.matches({"doc": "v", "start": None, "end": None}, ev)


def test_first_hit_rank_and_rr():
    retrieved = [{"doc": "nist_csf", "page": 2}, {"doc": "nist_csf", "page": 12}]
    s = R.score_question(retrieved, PDF_GOLD)
    assert s["first_hit_rank"] == 2 and s["rr"] == 0.5
    assert not s["hit@1"] and s["hit@3"] and s["hit@5"]


def test_no_hit():
    s = R.score_question([{"doc": "nist_csf", "page": 1}], PDF_GOLD)
    assert s["first_hit_rank"] is None and s["rr"] == 0.0 and not s["hit@any"]


def test_evidence_recall_needs_both_pieces():
    retrieved = [{"doc": "d", "page": 3}, {"doc": "d", "page": 4}]
    assert R.evidence_recall_at_k(retrieved, TWO_GOLD, 5) == 0.5
    retrieved.append({"doc": "d", "page": 9})
    assert R.evidence_recall_at_k(retrieved, TWO_GOLD, 5) == 1.0
    assert R.evidence_recall_at_k(retrieved, TWO_GOLD, 2) == 0.5


# -- answers ---------------------------------------------------------------

def test_contains_match_normalizes_numbers_and_case():
    assert A.contains_match("1,234", "The total was 1234 units.")
    assert A.contains_match("PR.AA-01", "Control pr.aa-01 covers identities.")
    assert A.contains_match("64 bits", "The nonce must be at least 64-bit long.")
    assert A.contains_match("4.1%", "It rose by 4.1% in 2023.")
    assert not A.contains_match("4.1%", "It rose by 41% in 2023.")


def test_contains_match_rejects_missing_tokens():
    assert not A.contains_match("February 26, 2024", "It was published in 2024.")


def test_refusal_detection():
    assert A.is_refusal("I could not find this in the source.")
    assert A.is_refusal("The document does not mention the budget.")
    assert A.is_refusal("")
    assert not A.is_refusal("The minimum length is 8 characters (page 12).")


def test_short_gold():
    assert A.is_short_gold("8 characters")
    assert not A.is_short_gold("one two three four five six seven eight nine")
    assert not A.is_short_gold(None)


# -- citations -------------------------------------------------------------

def test_inline_page_citations_parsed_and_ranges_expanded():
    cites = C.inline_citations("Yes (page 12). Also see pages 3-4 and p. 7.", "d", False)
    assert [c["page"] for c in cites] == [12, 3, 4, 7]


def test_inline_segment_citations():
    cites = C.inline_citations("He says so (segment 04:32).", "v", True)
    assert cites == [{"doc": "v", "start": 272.0, "end": 332.0}]


def test_citation_precision_recall():
    cited = [{"doc": "d", "page": 3}, {"doc": "d", "page": 5}]
    p, r = C.precision_recall(cited, TWO_GOLD)
    assert p == 0.5 and r == 0.5
    p, r = C.precision_recall([], TWO_GOLD)
    assert p is None and r == 0.0


# -- stats -----------------------------------------------------------------

def test_percentile_and_bootstrap():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    lo, hi = bootstrap_ci([0.0, 1.0] * 20)
    assert 0.3 < lo < 0.5 < hi < 0.7
    assert bootstrap_ci([1.0]) is None


def test_paired_bootstrap_detects_improvement():
    a = [0.0] * 30 + [1.0] * 10
    b = [1.0] * 30 + [1.0] * 10
    pb = paired_bootstrap(a, b)
    assert pb["mean_diff"] == 0.75 and pb["ci95"][0] > 0 and pb["p_better"] == 1.0
