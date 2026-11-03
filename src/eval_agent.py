"""7단계: Agent 전체 실행 평가 (질문 해석 → 그래프 조회 → 설명·검증). OpenAI 키 사용 — 실행 전 승인.

지표: 질문 해석 정확도(작업 상황·설비), 요인 재현율·누락 0·드문 요인·정밀도(5단계와 같은 계산),
설명 검증 통과율·재시도, 지연·토큰. 5단계 벡터/BM25/하이브리드 결과(Q4)와 나란히 비교한다.
실행: python src/eval_agent.py [--types Q4]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent import SafetyAgent  # noqa: E402
from eval_retrieval import QUESTIONS, RESULTS as RETRIEVAL_RESULTS, mean, score  # noqa: E402
from graph import ROOT  # noqa: E402

OUT = ROOT / "data" / "eval" / "agent_results.json"
REPORT = ROOT / "docs" / "eval" / "agent.md"


def run(types):
    agent = SafetyAgent()
    g = agent.graph
    rows = []
    for q in [q for q in json.loads(QUESTIONS.read_text(encoding="utf-8")) if q["type"] in types]:
        out = agent.ask(q["question"])
        gold = {x["factor"]: x["cases"] for x in q["gold_factors"]}
        found = {f["factor"] for f in (out["evidence"] or {}).get("factors", [])}
        cond = out["conditions"] or {"targets": [], "situations": []}
        gold_type = g.cases[q["case_ids"][0]]["equipment_type"]
        rows.append({
            "id": q["id"], "type": q["type"], "domain": q.get("domain", False), "flags": q.get("review_flags", []),
            "question": q["question"], "gold_equipment": q["equipment"], "gold_situation": q["situation"],
            "conditions": cond, "relax_level": out["relax_level"],
            "situation_hit": q["situation"] in cond["situations"],
            "equipment_hit": any(t["name"] in (q["equipment"], gold_type) for t in cond["targets"]),
            "explained": out["items"] is not None, "attempts": out["attempts"], "llm_error": out["llm_error"],
            **score(gold, found), **out["metrics"]})
        print(f"{q['id']} 재현율 {rows[-1]['recall']:.0%} 해석(작업 {rows[-1]['situation_hit']}, 설비 {rows[-1]['equipment_hit']}) "
              f"설명 {rows[-1]['explained']} {out['metrics']}")
    return rows


def report(rows):
    pct = lambda v: "-" if v is None else f"{v:.1%}"  # noqa: E731
    lines = ["# Agent 전체 실행 평가 (7단계)", "",
             f"질문 {len(rows)}개, 모델은 .env의 OPENAI_MODEL. 그래프 결과는 질문 해석까지 포함한 실제 값.", "",
             "## 검색 비교 (같은 질문)", "", "| 방식 | 요인 재현율 | 누락 0 비율 | 드문 요인 재현율 | 정밀도 |", "|---|---|---|---|---|",
             f"| **그래프 RAG Agent** | {pct(mean(r['recall'] for r in rows))} | {pct(mean(r['miss0'] for r in rows))} | "
             f"{pct(mean(r['rare_recall'] for r in rows))} | {pct(mean(r['precision'] for r in rows))} |"]
    ids = {r["id"] for r in rows}
    if RETRIEVAL_RESULTS.exists():
        base = defaultdict(list)
        for r in json.loads(RETRIEVAL_RESULTS.read_text(encoding="utf-8")):
            if r["id"] in ids and r["method"] != "graph" and r["k"] != "match":
                base[(r["method"], r["k"])].append(r)
        for (method, k), rs in sorted(base.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            lines.append(f"| {method} k={k} | {pct(mean(r['recall'] for r in rs))} | {pct(mean(r['miss0'] for r in rs))} | "
                         f"{pct(mean(r['rare_recall'] for r in rs))} | {pct(mean(r['precision'] for r in rs))} |")
    n = len(rows)
    lines += ["", "## Agent 동작", "",
              f"- 질문 해석: 작업 상황 정답 포함 {sum(r['situation_hit'] for r in rows)}/{n}, 설비 정답(세부 또는 유형) 포함 {sum(r['equipment_hit'] for r in rows)}/{n}",
              f"- 조건 완화 사용: {sum(1 for r in rows if (r['relax_level'] or 0) > 0)}/{n}",
              f"- 설명 검증 통과: {sum(r['explained'] for r in rows)}/{n}, 재생성 필요: {sum(1 for r in rows if (r['attempts'] or 0) > 1)}",
              f"- 평균 지연 {mean(r['latency_s'] for r in rows):.1f}초, LLM 호출 평균 {mean(r['llm_calls'] for r in rows):.1f}회, "
              f"토큰 합계 입력 {sum(r['input_tokens'] for r in rows):,} / 출력 {sum(r['output_tokens'] for r in rows):,}",
              "", "## 문항별", "", "| ID | 회사 분야 | 플래그 | 재현율 | 정밀도 | 작업 해석 | 설비 해석 | 완화 | 설명 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['id']} | {'○' if r['domain'] else ''} | {','.join(r['flags'])} | {pct(r['recall'])} | {pct(r['precision'])} | "
                     f"{'○' if r['situation_hit'] else '×'} | {'○' if r['equipment_hit'] else '×'} | {r['relax_level'] or 0} | {'○' if r['explained'] else '×'} |")
    return "\n".join(lines) + "\n"


def main():
    types = sys.argv[sys.argv.index("--types") + 1].split(",") if "--types" in sys.argv else ["Q4"]
    rows = run(types)
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report(rows), encoding="utf-8")
    print(REPORT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
