"""요인 재정렬 실험: 질문 관련도로 Agent의 요인 순서를 바꿀지 평가한다.

저장된 조건과 정규화 벡터만 읽는다 (API 호출 0회).
실행: python src/eval_rerank.py → docs/eval/rerank.md
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph import ROOT, load_graph  # noqa: E402
from answer import slot_order  # noqa: E402
from vector_rag import factor_relevance, load_case_vectors  # noqa: E402

QUESTIONS = ROOT / "data" / "eval" / "questions.json"
RESULTS = ROOT / "data" / "eval" / "agent_results_v1.json"  # 9단계 실험은 개체 연결 보강(11단계) 전 실행 기준
CACHE = ROOT / "data" / "cache"
REPORT = ROOT / "docs" / "eval" / "rerank.md"
ORDERS = ["count", "relevance", "rrf", "slot", "ideal"]


def rrf_order(count, relevance):
    """두 순위의 RRF 점수를 합친다 (k=60, 동점은 count 순서)."""
    scores = dict.fromkeys(count, 0.0)
    for order in (count, relevance):
        for r, factor in enumerate(order):
            scores[factor] += 1 / (61 + r)
    return sorted(count, key=lambda f: scores[f], reverse=True)


def p_at_5(order, gold):
    return len(set(order[:5]) & gold) / 5


def ndcg_at_5(order, gold):
    ideal = sum(1 / math.log2(r + 2) for r in range(min(5, len(gold))))
    dcg = sum(1 / math.log2(r + 2) for r, f in enumerate(order[:5]) if f in gold)
    return dcg / ideal if ideal else 0.0


def load_cache(g):
    for name in ("case_docs.json", "case_docs.npy", "queries.json", "queries.npy"):
        if not (CACHE / name).is_file():
            raise SystemExit(f"캐시 파일이 없습니다: {CACHE / name}. API 호출 없이 평가를 중단합니다.")
    try:
        doc_vectors = load_case_vectors(g)
    except ValueError as exc:
        raise SystemExit(f"{exc} API 호출 없이 평가를 중단합니다.") from None
    queries = json.loads((CACHE / "queries.json").read_text(encoding="utf-8"))
    query_vectors = np.load(CACHE / "queries.npy", allow_pickle=False)
    if (query_vectors.ndim != 2 or len(query_vectors) != len(queries)
            or doc_vectors.shape[1] != query_vectors.shape[1]):
        raise SystemExit("캐시 벡터의 개수 또는 차원이 맞지 않습니다. API 호출 없이 평가를 중단합니다.")
    return doc_vectors, dict(zip(queries, query_vectors))


def relevance_order(g, case_ids, count, positions, vectors, query):
    scores = factor_relevance(g, case_ids, positions, vectors, query)
    return sorted(count, key=lambda f: scores[f], reverse=True)


def evaluate():
    from agent import find_cases

    g = load_graph()
    vectors, queries = load_cache(g)
    positions = {case_id: r for r, case_id in enumerate(g.cases)}
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    conditions = {r["id"]: r["conditions"] for r in json.loads(RESULTS.read_text(encoding="utf-8"))}
    rows = []
    for q in questions:
        if q["type"] != "Q4":
            continue
        if q["question"] not in queries:
            raise SystemExit(f"질문 벡터 캐시가 없습니다: {q['id']}. API 호출 없이 평가를 중단합니다.")
        if q["id"] not in conditions:
            raise SystemExit(f"저장된 Agent 조건이 없습니다: {q['id']}.")
        gold = {x["factor"] for x in q["gold_factors"]}
        rare = {x["factor"] for x in q["gold_factors"] if x["cases"] <= 2}
        case_ids = find_cases(g, conditions[q["id"]])
        count = [f for f, _ in g.factors(case_ids)]
        relevance = relevance_order(g, case_ids, count, positions, vectors, queries[q["question"]])
        scores = factor_relevance(g, case_ids, positions, vectors, queries[q["question"]])
        orders = {"count": count, "relevance": relevance, "rrf": rrf_order(count, relevance),
                  "slot": slot_order(count, scores),
                  "ideal": sorted(count, key=lambda f: f not in gold)}
        # 후보 안에 정답과 오답이 섞여야 P@5가 달라질 수 있다.
        changeable = len(count) > 5 and bool(set(count) & gold) and bool(set(count) - gold)
        rows.append({"id": q["id"], "candidates": len(count), "changeable": changeable,
                     "rare_total": len(rare),
                     "rare": {name: len(set(order[:5]) & rare) for name, order in orders.items()},
                     "scores": {name: (p_at_5(order, gold), ndcg_at_5(order, gold))
                                for name, order in orders.items()}})
    if not rows:
        raise SystemExit("평가할 Q4 질문이 없습니다.")
    return rows


def report(rows):
    means = {name: tuple(sum(r["scores"][name][i] for r in rows) / len(rows) for i in (0, 1))
             for name in ORDERS}
    lines = ["# Agent 요인 재정렬 실험", "",
             "질문 관련도로 요인 순서를 바꾸면 상위 5개 요인의 품질이 나아지는지 오프라인으로 평가한다.",
             "저장된 Agent 조건과 벡터 캐시만 사용하며 API 호출은 0회다.",
             "결정 규칙은 사전에 고정했다: hybrid 기준선과 같은 RRF k=60을 사용한다.",
             "count는 근거 사례 수 순서, relevance는 후보 사례와 질문의 최대 코사인 유사도 순서다.",
             "rrf는 두 순위의 점수 1/(61+r)를 합치며, 동점은 count 순서를 유지한다.",
             "slot은 사례 수 상위 3개 + 나머지 중 질문 관련도 상위 2개이며, 동점은 count 순서를 유지한다.",
             "ideal은 후보 내 정답 요인을 count 순서대로 앞에 둔 P@5 기준 상한이다(희귀 정답 수의 최대는 아님).",
             "P@5의 분모는 항상 5이며, 이진 nDCG@5의 이상 DCG는 전체 정답 수를 기준으로 한다.", "",
             "| 순서 | 평균 P@5 | 평균 nDCG@5 | 희귀 정답 top-5 / 전체 |", "|---|---|---|---|"]
    for name in ORDERS:
        p, n = means[name]
        rare = sum(r["rare"][name] for r in rows)
        total = sum(r["rare_total"] for r in rows)
        lines.append(f"| {name} | {p:.1%} | {n:.3f} | {rare} / {total} |")
    lines += ["", f"P@5가 순서에 따라 바뀔 수 있는 질문: {sum(r['changeable'] for r in rows)}/{len(rows)}개.", "",
              "| id | 후보 요인 수 | P@5 count | P@5 relevance | P@5 rrf | P@5 slot | nDCG count | nDCG relevance | nDCG rrf | nDCG slot |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ps = " | ".join(f"{r['scores'][name][0]:.1%}" for name in ORDERS[:4])
        ns = " | ".join(f"{r['scores'][name][1]:.3f}" for name in ORDERS[:4])
        lines.append(f"| {r['id']} | {r['candidates']} | {ps} | {ns} |")
    # 예상값과 다르면 지정된 결론을 쓰지 않는다.
    expected = {"count": ("61.0%", "0.864"), "relevance": ("63.0%", "0.888"), "rrf": ("61.0%", "0.873")}
    actual = {name: (f"{means[name][0]:.1%}", f"{means[name][1]:.3f}") for name in expected}
    lines += [""]
    if actual == expected and len(rows) == 20:
        lines.append("결론: 미리 정한 RRF는 P@5 개선 없음(61.0%→61.0%) → 채택 안 함. Q4는 개발에 쓴 20문항이라 독립 평가 아님")
    else:
        lines.append(f"결론: 예상 결과와 계산 결과가 불일치하여 지정 결론을 보류한다. 예상={expected}, 실제={actual}, 질문 수={len(rows)}.")
    p, n = means["slot"]
    rare = sum(r["rare"]["slot"] for r in rows)
    lines.append(f"채택한 slot(사례 수 3 + 질문 관련도 2): P@5 {p:.1%}, nDCG@5 {n:.3f}, 희귀 정답 top-5 {rare}/{total}. slot 규칙은 Q4 결과를 본 뒤 정했으므로 개발셋 결과")
    return "\n".join(lines) + "\n"


def main():
    text = report(evaluate())
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
