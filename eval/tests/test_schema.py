from eval.datasets.schema import validate

DOCS = {"nist_csf", "mit_ai_lec1"}


def good(**kw):
    rec = {"id": "x-1", "category": "factoid", "doc": "nist_csf",
           "question": "When was CSF 2.0 published?", "answerable": True,
           "gold_answer": "February 26, 2024",
           "gold_evidence": [{"doc": "nist_csf", "pages": [29]}]}
    rec.update(kw)
    return rec


def test_valid_record():
    assert validate(good(), DOCS) == []


def test_video_evidence():
    assert validate(good(doc="mit_ai_lec1",
                         gold_evidence=[{"doc": "mit_ai_lec1", "start": 10.0, "end": 95.5}]), DOCS) == []


def test_answerable_needs_answer_and_evidence():
    assert "answerable question needs gold_answer" in validate(good(gold_answer=None), DOCS)
    assert "answerable question needs gold_evidence" in validate(good(gold_evidence=[]), DOCS)


def test_unanswerable_rules():
    ok = good(category="unanswerable", answerable=False, gold_answer=None, gold_evidence=[])
    assert validate(ok, DOCS) == []
    bad = good(category="factoid", answerable=False, gold_answer=None, gold_evidence=[])
    assert "only the unanswerable category may be unanswerable" in validate(bad, DOCS)


def test_rejects_unknown_doc_bad_pages_and_category():
    assert any("unknown doc" in e for e in validate(good(doc="nope"), DOCS))
    assert "pages must be positive ints" in validate(good(gold_evidence=[{"doc": "nist_csf", "pages": [0]}]), DOCS)
    assert any("bad category" in e for e in validate(good(category="trivia"), DOCS))


def test_follow_up_needs_history():
    assert "follow_up needs history" in validate(good(category="follow_up"), DOCS)
    assert validate(good(category="follow_up",
                         history=[{"role": "user", "content": "What is CSF?"}]), DOCS) == []
