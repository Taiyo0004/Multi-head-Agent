"""Exact significance tests, with no SciPy dependency.

Accuracy differences in this project are counted over a few hundred
questions, where 32 correct against 19 correct looks like a result and is
not one.  Reporting a p-value next to every delta keeps that visible.
"""
from __future__ import annotations

from math import comb
from typing import Dict

TOL = 1.0000001  # guards against float ties being dropped


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test on the table [[a, b], [c, d]].

    ``a``/``b`` are successes/failures in group 1, ``c``/``d`` in group 2.
    Exact, so it is valid for the small counts this repo produces.
    """
    n = a + b + c + d
    if n == 0:
        return 1.0
    row1, col1 = a + b, a + c

    def prob(x: int) -> float:
        return comb(row1, x) * comb(n - row1, col1 - x) / comb(n, col1)

    observed = prob(a)
    lo = max(0, col1 - (n - row1))
    hi = min(row1, col1)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= observed * TOL))


def compare_accuracy(name_a: str, correct_a: int, n_a: int,
                     name_b: str, correct_b: int, n_b: int) -> Dict[str, object]:
    """Accuracy difference between two configurations, with a p-value."""
    p = fisher_exact_two_sided(correct_a, n_a - correct_a, correct_b, n_b - correct_b)
    acc_a = 100.0 * correct_a / max(1, n_a)
    acc_b = 100.0 * correct_b / max(1, n_b)
    return {
        "comparison": f"{name_b} vs {name_a}",
        "accuracy_points": round(acc_b - acc_a, 2),
        "counts": f"{correct_b}/{n_b} vs {correct_a}/{n_a}",
        "fisher_p": round(p, 6),
        "significant_at_0.05": bool(p < 0.05),
        "verdict": _verdict(p, acc_b - acc_a),
    }


def _verdict(p: float, delta: float) -> str:
    if p >= 0.05:
        return ("difference not statistically significant at this sample size; "
                "do not report it as an effect")
    direction = "lower" if delta < 0 else "higher"
    return f"significantly {direction} (p={p:.2g})"
