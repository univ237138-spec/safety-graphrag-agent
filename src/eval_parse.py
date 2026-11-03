"""7단계 점검: 질문 해석만 Q4 20문항 (설명 생성 없음, LLM은 해석 1회/문항).

해석 → 그래프 용어 연결 → 그래프 조회(완화 없이)까지 계산해 해석 정확도와 요인 재현율을 본다.
실행: python src/eval_parse.py → data/eval/parse_check.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm  # noqa: E402
import query_parser  # noqa: E402
from agent import find_cases  # noqa: E402
from eval_retrieval import QUESTIONS, mean, score  # noqa: E402
from graph import ROOT, load_graph  # noqa: E402

OUT = ROOT / "data" / "eval" / "parse_check.json"


def main():
    g, model = load_graph(), llm.model_from_env()
    usage = []
    llm.USAGE.set(usage)
    rows = []
    for q in [q for q in json.loads(QUESTIONS.read_text(encoding="utf-8")) if q["type"] == "Q4"]:
        try:
            parsed, error = query_parser.parse(q["question"], g, model), None
        except llm.LLMError as exc:
            parsed, error = {}, str(exc)
        cond = query_parser.link(g, parsed, q["question"])
        gold_type = g.cases[q["case_ids"][0]]["equipment_type"]
        ids = find_cases(g, cond)
        gold = {x["factor"]: x["cases"] for x in q["gold_factors"]}
        row = {"id": q["id"], "flags": q.get("review_flags", []), "error": error, "parsed": parsed, "conditions": cond,
               "gold": [q["equipment"], q["situation"]],
               "situation_hit": q["situation"] in cond["situations"],
               "equipment_hit": any(t["name"] in (q["equipment"], gold_type) for t in cond["targets"]),
               "cases": len(ids), **score(gold, {g.cases[i]["factor"] for i in ids})}
        rows.append(row)
        print(f"{q['id']} 작업 {'○' if row['situation_hit'] else '×'} 설비 {'○' if row['equipment_hit'] else '×'} "
              f"사례 {row['cases']} 재현율 {row['recall']:.0%} 정밀도 {row['precision']:.0%} | "
              f"작업={cond['situations']} 설비={[t['name'] for t in cond['targets']]} | 정답={row['gold']}")
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    n = len(rows)
    print(f"\n작업 정답 포함 {sum(r['situation_hit'] for r in rows)}/{n}, 설비 정답 포함 {sum(r['equipment_hit'] for r in rows)}/{n}, "
          f"0건 {sum(1 for r in rows if not r['cases'])}/{n}, 재현율 평균 {mean(r['recall'] for r in rows):.1%}, "
          f"누락 0 {mean(r['miss0'] for r in rows):.1%}, 정밀도 {mean(r['precision'] for r in rows):.1%}")
    print(f"LLM {len(usage)}회, 토큰 입력 {sum(u['input_tokens'] for u in usage):,} / 출력 {sum(u['output_tokens'] for u in usage):,}")


if __name__ == "__main__":
    main()
