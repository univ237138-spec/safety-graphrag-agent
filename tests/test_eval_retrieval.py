import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from eval_retrieval import mean, score  # noqa: E402


def test_all_gold_found_with_extra():
    result = score({'A': 5, 'B': 1}, {'A', 'B', 'C'})
    assert result['recall'] == pytest.approx(1.0)
    assert result['miss0'] == pytest.approx(1.0)
    assert result['rare_recall'] == pytest.approx(1.0)
    assert result['precision'] == pytest.approx(2 / 3)


def test_common_factor_only():
    result = score({'A': 5, 'B': 1}, {'A'})
    assert result['recall'] == pytest.approx(0.5)
    assert result['miss0'] == pytest.approx(0.0)
    assert result['rare_recall'] == pytest.approx(0.0)
    assert result['precision'] == pytest.approx(1.0)


def test_no_rare_factors():
    assert score({'A': 3}, {'A'})['rare_recall'] is None


def test_two_cases_is_rare():
    result = score({'A': 2}, set())
    assert result['rare_recall'] == pytest.approx(0.0)
    assert result['precision'] == pytest.approx(0.0)
    assert result['recall'] == pytest.approx(0.0)


def test_mean_ignores_none():
    assert mean([1, None, 0]) == pytest.approx(0.5)
    assert mean([None]) is None
