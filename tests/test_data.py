"""Role supervision is built from GSM8K; these run without torch."""
import random

import pytest

from mha_torch.data import (VERDICT_BAD, VERDICT_OK, build_examples, clean_plan,
                            corrupt_plan, extract_answer)

SOLUTION = ("Janet has 3 eggs.\n"
            "She buys 5 more <<3+5=8>> eggs.\n"
            "So she has 8 eggs.\n"
            "#### 8")
ROW = {"question": "How many eggs does Janet have?", "answer": SOLUTION}


def test_extract_answer():
    assert extract_answer(SOLUTION) == "8"
    assert extract_answer("no marker here") is None
    assert extract_answer("#### 1,234") == "1234"


def test_clean_plan_strips_calculator_and_answer():
    plan = clean_plan(SOLUTION)
    assert "<<" not in plan and "####" not in plan
    assert plan.startswith("1. ") and "3. " in plan
    assert "  " not in plan  # calculator removal must not leave double spaces


def test_corrupt_plan_changes_exactly_one_number():
    plan = clean_plan(SOLUTION)
    corrupted, did = corrupt_plan(plan, random.Random(0))
    assert did and corrupted != plan
    differing = [a != b for a, b in zip(plan.split(), corrupted.split())]
    assert sum(differing) == 1


def test_corrupt_plan_without_numbers_is_a_noop():
    plan, did = corrupt_plan("1. think about it", random.Random(0))
    assert not did and plan == "1. think about it"


def test_build_examples_covers_every_role():
    ex = build_examples([ROW], seed=0)
    assert {e.role for e in ex} == {"planner", "critic", "solver"}


def test_solver_target_is_the_answer():
    solver = [e for e in build_examples([ROW], seed=0) if e.role == "solver"][0]
    assert solver.target.strip() == "8"
    assert "####" not in solver.prompt


def test_critic_sees_both_classes():
    """A critic trained only on good plans learns a constant; it must not."""
    rows = [ROW] * 40
    critics = [e for e in build_examples(rows, seed=0) if e.role == "critic"]
    verdicts = {e.target.strip() for e in critics}
    assert verdicts == {VERDICT_OK, VERDICT_BAD}
    corrupted = sum(e.meta["corrupted"] for e in critics)
    assert 0.2 * len(critics) < corrupted < 0.8 * len(critics)


def test_critic_verdict_matches_its_meta():
    for e in build_examples([ROW] * 20, seed=3):
        if e.role == "critic":
            expected = VERDICT_BAD if e.meta["corrupted"] else VERDICT_OK
            assert e.target.strip() == expected


def test_rows_without_an_answer_are_dropped():
    assert build_examples([{"question": "q", "answer": "no answer marker"}]) == []


def test_mask_answer_hides_the_result_in_the_plan():
    from mha_torch.data import mask_answer
    plan = "1. Janet has 3 eggs.\n2. So she has 8 eggs."
    masked = mask_answer(plan, "8")
    assert "8" not in masked and "3" in masked and "___" in masked


def test_mask_answer_leaves_longer_numbers_alone():
    from mha_torch.data import mask_answer
    assert mask_answer("she paid 18 dollars", "8") == "she paid 18 dollars"


def test_solver_plan_is_masked_only_when_requested():
    plain = [e for e in build_examples([ROW], seed=0) if e.role == "solver"][0]
    masked = [e for e in build_examples([ROW], seed=0, mask_answer_in_plan=True)
              if e.role == "solver"][0]
    assert "8" in plain.prompt.split("Solver:")[0]
    assert "___" in masked.prompt and masked.target.strip() == "8"
