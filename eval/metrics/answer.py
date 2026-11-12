"""
Answer metrics: deterministic matching first, LLM judge only where needed.

- `contains_match`: the normalized short gold answer appears in the system answer.
  Used for factoid / exact_id golds of at most SHORT_GOLD_WORDS words.
- `is_refusal`: the answer says the information isn't in the source.
"""
from __future__ import annotations

import re

SHORT_GOLD_WORDS = 8

_REFUSAL = re.compile(
    r"(could not find|couldn't find|can ?not find|cannot be found|not (?:be )?found in|"
    r"no information (?:about|on|regarding)|does not (?:mention|contain|provide|specify|say|state|include)|"
    r"doesn't (?:mention|contain|provide|specify|say|state|include)|"
    r"(?:is|are) not (?:mentioned|provided|specified|stated|included|available|covered|discussed)|"
    r"not (?:mentioned|provided|specified|stated|covered|discussed) in|"
    r"no (?:mention|details?|data) (?:of|about|on)|unable to (?:find|locate|answer))",
    re.I,
)
_NUM_SEP = re.compile(r"(?<=\d)[,\s](?=\d{3}\b)")
_PUNCT = re.compile(r"[^\w%$.\-/ ]+")


def normalize(text: str) -> str:
    text = text.lower().replace("’", "'").replace("–", "-").replace("—", "-")
    text = _NUM_SEP.sub("", text)                      # 1,234 -> 1234
    text = _PUNCT.sub(" ", text)
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text)    # drop periods that aren't decimals
    return re.sub(r"\s+", " ", text).strip()


def is_short_gold(gold: str | None) -> bool:
    return bool(gold) and len(gold.split()) <= SHORT_GOLD_WORDS


def contains_match(gold: str, answer: str) -> bool:
    g, a = normalize(gold), normalize(answer)
    if not g:
        return False
    if g in a:
        return True
    # all content tokens of the gold present (handles reordering, "64-bit" vs "64 bits")
    toks = [t for t in re.split(r"[\s\-/]+", g) if t]
    a_toks = set(re.split(r"[\s\-/]+", a))
    stem = lambda t: t[:-1] if len(t) > 3 and t.endswith("s") else t
    a_stems = {stem(t) for t in a_toks}
    return bool(toks) and all(stem(t) in a_stems for t in toks)


def is_refusal(answer: str) -> bool:
    if not answer.strip():
        return True
    head = answer.strip()[:400]
    return bool(_REFUSAL.search(head))
