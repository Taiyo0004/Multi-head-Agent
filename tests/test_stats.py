"""Significance helpers (pure Python, no torch, no SciPy)."""
import pytest

from mha_torch.stats import compare_accuracy, fisher_exact_two_sided


def test_identical_groups_are_not_significant():
    assert fisher_exact_two_sided(10, 90, 10, 90) == pytest.approx(1.0, abs=1e-9)


def test_extreme_separation_is_significant():
    assert fisher_exact_two_sided(100, 0, 0, 100) < 1e-9


def test_known_small_table():
    """[[1,9],[11,3]]: two-sided p = 2*(0.0013460 + 0.0000337) = 0.0027595.

    Hand-checked against the hypergeometric terms, C(24,12) = 2704156.
    The one-sided value is 0.0013797, so this also pins down that the
    implementation is two-sided.
    """
    assert fisher_exact_two_sided(1, 9, 11, 3) == pytest.approx(0.0027595, abs=1e-6)


def test_empty_table_is_safe():
    assert fisher_exact_two_sided(0, 0, 0, 0) == 1.0


def test_p_value_never_exceeds_one():
    for table in [(1, 1, 1, 1), (3, 197, 32, 168), (5, 5, 6, 4), (0, 200, 1, 199)]:
        assert 0.0 <= fisher_exact_two_sided(*table) <= 1.0


def test_borderline_difference_is_reported_as_not_significant():
    """32/200 vs 19/200 is the real case from the GSM8K eval: p ~ 0.07."""
    got = compare_accuracy("pipeline_text", 32, 200, "pipeline_shared_kv", 19, 200)
    assert got["fisher_p"] == pytest.approx(0.0712, abs=0.005)
    assert got["significant_at_0.05"] is False
    assert "not statistically significant" in got["verdict"]


def test_clear_difference_is_reported_as_significant():
    got = compare_accuracy("direct", 3, 200, "pipeline_text", 32, 200)
    assert got["significant_at_0.05"] is True
    assert got["accuracy_points"] == pytest.approx(14.5)
    assert "higher" in got["verdict"]
