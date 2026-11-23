"""Role supervision built from GSM8K.

A multi-role pipeline needs *role-specific* targets, which a plain
instruction dataset does not provide.  GSM8K does, if you split it:

* **planner**  question -> the reasoning steps, with the final answer removed.
* **solver**   question + plan -> the final numeric answer.
* **critic**   question + plan -> a verdict.  Negatives are produced by
  perturbing one number in the plan, so the critic learns to check
  arithmetic consistency rather than to always say "looks fine".

That last point matters: a critic trained only on correct plans learns a
constant function.  Half of its examples are corrupted on purpose, and the
corruption is recorded so the eval can report per-class accuracy.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

ANSWER_RE = re.compile(r"####\s*([\-0-9\.,]+)")
NUMBER_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])")
CALC_RE = re.compile(r"<<[^>]*>>")

VERDICT_OK = "OK"
VERDICT_BAD = "WRONG"


@dataclass
class Example:
    role: str
    prompt: str
    target: str
    meta: Dict[str, object]


def extract_answer(solution: str) -> Optional[str]:
    m = ANSWER_RE.search(solution)
    return m.group(1).replace(",", "").strip() if m else None


def clean_plan(solution: str) -> str:
    """Reasoning steps with calculator annotations and the answer removed."""
    body = ANSWER_RE.split(solution)[0]
    body = CALC_RE.sub("", body)
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in body.strip().splitlines() if ln.strip()]
    return "\n".join(f"{i+1}. {ln}" for i, ln in enumerate(lines))


def corrupt_plan(plan: str, rng: random.Random, question: str | None = None) -> Tuple[str, bool]:
    """Perturb one number in the plan. Returns (plan, was_corrupted).

    When ``question`` is given, a number that the plan *copied from the
    question* is corrupted in preference to any other.  That makes the
    negative detectable by grounding -- the plan now contradicts the
    premises -- rather than only by re-doing the arithmetic.

    This matters empirically: a 135M backbone trained on arithmetic-only
    negatives learns to answer OK unconditionally (measured: 5.7% recall on
    perturbed plans, below the majority-class baseline), because verifying
    multi-step arithmetic is beyond it.  Checking that the numbers trace
    back to the question is not.
    """
    matches = list(NUMBER_RE.finditer(plan))
    if not matches:
        return plan, False
    if question:
        q_numbers = {m.group(1) for m in NUMBER_RE.finditer(question)}
        grounded = [m for m in matches if m.group(1) in q_numbers]
        if grounded:
            matches = grounded
    m = rng.choice(matches)
    original = m.group(1)
    try:
        value = float(original)
    except ValueError:
        return plan, False
    delta = rng.choice([1, 2, 3, -1, -2, 10])
    new = value + delta
    new_s = str(int(new)) if float(new).is_integer() and "." not in original else f"{new:g}"
    if new_s == original:
        return plan, False
    return plan[:m.start(1)] + new_s + plan[m.end(1):], True


def mask_answer(plan: str, answer: str) -> str:
    """Blank the final answer where the plan already states it.

    GSM8K solutions normally announce the result in their last step, so a
    solver handed the gold plan can score well by copying a number out of
    its own prompt.  Replacing that number with a placeholder forces the
    last arithmetic step to actually be computed.
    """
    if not answer:
        return plan
    return re.sub(rf"(?<![\w.]){re.escape(answer)}(?![\w.])", "___", plan)


def build_examples(rows, roles=("planner", "critic", "solver"), seed: int = 0,
                   mask_answer_in_plan: bool = False) -> List[Example]:
    """Turn GSM8K rows (dicts with 'question' and 'answer') into role examples.

    ``mask_answer_in_plan`` hides the answer inside the solver's plan; see
    ``mask_answer``.  Measure the leakage first (``evaluate.py`` reports
    ``gold_plan_contains_answer_pct``) before deciding to turn it on.
    """
    rng = random.Random(seed)
    out: List[Example] = []
    for row in rows:
        question = row["question"].strip()
        solution = row["answer"]
        plan = clean_plan(solution)
        answer = extract_answer(solution)
        if not plan or answer is None:
            continue

        if "planner" in roles:
            out.append(Example("planner",
                               f"Task: {question}\nPlanner: decompose the task into ordered, "
                               f"checkable steps.\nPlan:\n", plan,
                               {"answer": answer}))
        if "critic" in roles:
            shown, corrupted = (corrupt_plan(plan, rng, question)
                                if rng.random() < 0.5 else (plan, False))
            verdict = VERDICT_BAD if corrupted else VERDICT_OK
            out.append(Example("critic",
                               f"Task: {question}\nPlan:\n{shown}\nCritic: is the plan "
                               f"consistent with the task? Answer {VERDICT_OK} or "
                               f"{VERDICT_BAD}.\nCritique:", f" {verdict}",
                               {"corrupted": corrupted}))
        if "solver" in roles:
            solver_plan = mask_answer(plan, answer) if mask_answer_in_plan else plan
            out.append(Example("solver",
                               f"Task: {question}\nPlan:\n{solver_plan}\nSolver: give the final "
                               f"numeric answer.\nAnswer:", f" {answer}",
                               {"answer": answer, "answer_masked": mask_answer_in_plan}))
    return out


def load_gsm8k(split: str = "train", limit: Optional[int] = None):
    """Load GSM8K, with a clear error if `datasets` or the network is absent."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pip install datasets") from exc
    ds = load_dataset("openai/gsm8k", "main", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds
