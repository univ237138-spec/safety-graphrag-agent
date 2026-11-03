"""5단계: 검색 단계 비교 — 벡터 / BM25 / 하이브리드 top-k vs 그래프 조회.

가져온 사례들의 재해유발요인을 정답 요인과 비교한다 (LLM 없음, 임베딩만).
그래프는 아직 질문 해석이 없으므로 정답 조건(설비·작업)을 그대로 넣은 값 = 상한선. Q4 해석은 6단계.
실행: python src/eval_retrieval.py → docs/eval/retrieval.md, data/eval/retrieval_results.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph import ROOT, load_graph  # noqa: E402
from vector_rag import Retriever, embed_cached  # noqa: E402

QUESTIONS = ROOT / "data" / "eval" / "questions.json"
REPORT = ROOT / "docs" / "eval" / "retrieval.md"
RESULTS = ROOT / "data" / "eval" / "retrieval_results.json"
METHODS = ["vector", "bm25", "hybrid"]
KS = [5, 10, 20, "match"]  # match: 그래프가 찾은 사례 수만큼 가져옴
RARE = 2  # 근거 사례 2건 이하 = 드문 요인


def score(gold, found):
    """gold: {요인: 사례 수}, found: 가져온 요인 집합."""
    hit = set(gold) & found
    rare = {f for f, n in gold.items() if n <= RARE}
    return {"recall": len(hit) / len(gold), "miss0": float(hit == set(gold)),
            "rare_recall": len(rare & found) / len(rare) if rare else None,
            "precision": len(hit) / len(found) if found else 0.0}


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def evaluate():
    g = load_graph()
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    retriever = Retriever(g)
    query_vectors = embed_cached("queries", [q["question"] for q in questions])
    rows = []
    for q, qv in zip(questions, query_vectors):
        gold = {x["factor"]: x["cases"] for x in q["gold_factors"]}
        factors_of = lambda ids: {g.cases[i]["factor"] for i in ids}  # noqa: E731
        base = {"id": q["id"], "type": q["type"], "domain": q.get("domain", False), "flags": q.get("review_flags", []),
                "gold": len(gold), "gold_cases": len(q["case_ids"])}
        rows.append(base | {"method": "graph", "k": "match"} | score(gold, factors_of(q["case_ids"])))
        for method in METHODS:
            ranked = retriever.rank(method, np.asarray(qv), q["question"])
            for k in KS:
                n = len(q["case_ids"]) if k == "match" else k
                rows.append(base | {"method": method, "k": k} | score(gold, factors_of(ranked[:n])))
    return rows


def table(rows, title, keep):
    groups = defaultdict(list)
    for r in rows:
        if keep(r):
            groups[(r["method"], r["k"])].append(r)
    lines = [f"### {title}", "", "| 방식 | k | 요인 재현율 | 누락 0 비율 | 드문 요인 재현율 | 정밀도 | 질문 수 |", "|---|---|---|---|---|---|---|"]
    order = [("graph", "match")] + [(m, k) for m in METHODS for k in KS]
    for key in order:
        rs = groups.get(key)
        if not rs:
            continue
        pct = lambda v: "-" if v is None else f"{v:.1%}"  # noqa: E731
        name = "그래프 (정답 조건)" if key[0] == "graph" else key[0]
        lines.append(f"| {name} | {'그래프와 같은 수' if key[1] == 'match' else key[1]} | {pct(mean(r['recall'] for r in rs))} | "
                     f"{pct(mean(r['miss0'] for r in rs))} | {pct(mean(r['rare_recall'] for r in rs))} | "
                     f"{pct(mean(r['precision'] for r in rs))} | {len(rs)} |")
    return "\n".join(lines) + "\n"


def main():
    rows = evaluate()
    RESULTS.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    parts = ["# 검색 단계 비교 (5단계)", "",
             "가져온 사례의 재해유발요인 vs 정답 요인. 그래프는 정답 조건을 넣은 상한선 (질문 해석은 6단계).", "",
             table(rows, "전체 (193문항)", lambda r: True)]
    for qtype, name in [("Q1", "Q1 작업 기준"), ("Q2", "Q2 세부 설비 기준"), ("Q3", "Q3 설비+작업"), ("Q4", "Q4 현장 자연어")]:
        parts.append(table(rows, name, lambda r, t=qtype: r["type"] == t))
    parts.append(table(rows, "Q4 회사 분야 (로봇·자동화·물류 동선)", lambda r: r["type"] == "Q4" and r["domain"]))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(parts), encoding="utf-8")
    print(REPORT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
