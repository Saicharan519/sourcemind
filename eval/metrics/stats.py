"""Small statistics helpers: percentiles and bootstrap confidence intervals."""
from __future__ import annotations

import random


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def percentile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    pos = (len(s) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def bootstrap_ci(xs: list[float], iters: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float] | None:
    """Percentile bootstrap CI for the mean."""
    if len(xs) < 2:
        return None
    rng = random.Random(seed)
    n = len(xs)
    means = sorted(sum(rng.choice(xs) for _ in range(n)) / n for _ in range(iters))
    return means[int(iters * alpha / 2)], means[int(iters * (1 - alpha / 2)) - 1]


def paired_bootstrap(a: list[float], b: list[float], iters: int = 2000,
                     seed: int = 0) -> dict | None:
    """Mean difference (b - a) over paired per-question scores, with a 95% CI and
    the share of resamples where b beats a."""
    if len(a) != len(b) or len(a) < 2:
        return None
    rng = random.Random(seed)
    diffs = [y - x for x, y in zip(a, b)]
    n = len(diffs)
    boots = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(iters))
    return {"mean_diff": sum(diffs) / n,
            "ci95": (boots[int(iters * 0.025)], boots[int(iters * 0.975) - 1]),
            "p_better": sum(d > 0 for d in boots) / iters}
