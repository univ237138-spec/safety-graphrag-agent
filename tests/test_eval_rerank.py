import sys
from pathlib import Path

import math

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from eval_rerank import ndcg_at_5, p_at_5, rrf_order  # noqa: E402


def test_rrf_order():
    assert rrf_order(["a", "b", "c"], ["b", "c", "a"]) == ["b", "a", "c"]
    # 동점은 현재 순서를 유지한다.
    assert rrf_order(["a", "b"], ["b", "a"]) == ["a", "b"]
    assert rrf_order([], []) == []


def test_p_at_5():
    assert p_at_5(["a", "b", "c", "d", "e", "f"], {"a", "c", "f"}) == 2 / 5
    assert p_at_5(["a"], {"a"}) == 1 / 5
    assert p_at_5(["a"], set()) == 0
    assert p_at_5([], {"a"}) == 0


def test_ndcg_at_5():
    assert ndcg_at_5(["a", "b"], {"a", "b"}) == 1
    assert math.isclose(ndcg_at_5(["x", "a", "b"], {"a", "b"}),
                        (1 / math.log2(3) + 0.5) / (1 + 1 / math.log2(3)))
    assert math.isclose(ndcg_at_5(["a"], {"a", "missing"}), 1 / (1 + 1 / math.log2(3)))
    assert ndcg_at_5(["x", "y", "z", "w", "v", "a"], {"a"}) == 0
    assert ndcg_at_5(["a"], set()) == 0
    assert ndcg_at_5([], {"a"}) == 0
