"""처음 보는 질문 평가. --score는 저장 결과만 읽고, --run만 실제 Agent를 호출한다."""
import argparse
import json
from pathlib import Path

from graph import ROOT, equipment_type
from query_parser import inner_name

GOLD = ROOT / "data/eval/unseen.json"
OUT = ROOT / "data/eval/unseen_results.json"
REPORT = ROOT / "docs/eval/unseen.md"
METRICS = ("equipment", "not_equipment", "situations", "guidance", "cases", "first_guidance")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def equipment_names(names):
    """세부 설비는 원문·괄호 안 이름·실제 유형으로 대조한다."""
    return {alias for name in names for alias in (name, inner_name(name), equipment_type(name))}


def judge(result, gold):
    conditions = result.get("조건") or {}
    names = equipment_names(conditions.get("설비", []))
    situations = set(conditions.get("작업", []))
    expected = gold.get("equipment", [])
    expected_situations = gold.get("situations", [])
    return {
        "equipment": bool(names & set(expected)) if expected else (not names if "guidance" in gold else None),
        "not_equipment": not bool(names & set(gold["not_equipment"])) if gold.get("not_equipment") else None,
        "situations": bool(situations & set(expected_situations)) if expected_situations else not situations,
        "guidance": result.get("안내") == gold["guidance"] if "guidance" in gold else None,
        "cases": result.get("사례", 0) > 0 if expected or expected_situations else None,
    }


def score(results, gold):
    """저장 결과를 질문 본문으로 대응시켜 문항별 점수와 합계를 반환한다."""
    if isinstance(results, (str, Path)):
        results = read(results)
    if isinstance(gold, (str, Path)):
        gold = read(gold)
    singles = {r["질문"]: r for r in results.get("single", [])}
    multis = {tuple(t["질문"] for t in r["대화"]): r for r in results.get("multi", [])}
    rows = []
    for q in gold["single"]:
        result = singles.get(q["question"])
        checks = judge(result or {}, q)
        rows.append({"id": q["id"], "missing": result is None, **checks})
    for q in gold["multi"]:
        result = multis.get(tuple(q["turns"]))
        turns = result["대화"] if result else []
        checks = judge(turns[-1] if turns else {}, q)
        checks["first_guidance"] = bool(turns) and turns[0].get("안내") == q["first_guidance"]
        rows.append({"id": q["id"], "missing": result is None, **checks})
    for row in rows:
        if row["missing"]:
            for key in METRICS:
                if row.get(key) is not None:
                    row[key] = False
        row["passed"] = not row["missing"] and all(row[k] for k in METRICS if row.get(k) is not None)
    totals = {key: {"passed": sum(r.get(key) is True for r in rows),
                    "total": sum(r.get(key) is not None for r in rows)} for key in METRICS}
    totals["all"] = {"passed": sum(r["passed"] for r in rows), "total": len(rows)}
    return {"items": rows, "totals": totals}


def update_carry(out, turns):
    """앱과 동일하게 안내가 두 번 연속되면 조건을 지운다."""
    turns += 1
    hint = (out.get("guidance") or {}).get("type")
    if not out.get("llm_error") and hint in ("need_situation", "need_equipment") and turns < 2:
        return out["conditions"], turns
    return None, 0


def stored(question, out, carried):
    conditions = out["conditions"]
    evidence = out.get("evidence") or {}
    return {"질문": question, "조건": {"설비": [t["name"] for t in conditions["targets"]],
                                      "작업": conditions["situations"]},
            "이어받음": carried, "parse_ok": out.get("parse_ok"),
            "안내": (out.get("guidance") or {}).get("type"),
            "선택지": [o["name"] for o in (out.get("guidance") or {}).get("options", [])],
            "사례": len(out.get("case_ids") or []), "요인": evidence.get("total_factors", 0),
            "상세": [[d.get("pick"), d["cases"], d["factor"]] for d in evidence.get("detailed", [])],
            "요약": (out.get("reply") or {}).get("summary"), "검증": out.get("items") is not None,
            "시도": out.get("attempts"), "오류": out.get("llm_error"),
            "rank_note": out.get("rank_note"), "지표": out.get("metrics", {})}


def run(gold, agent=None):
    """실제 실행 또는 주입한 가짜 Agent로 같은 저장 형식을 만든다."""
    if agent is None:
        from agent import SafetyAgent
        agent = SafetyAgent()
    results = {"single": [], "multi": []}
    for q in gold["single"]:
        results["single"].append({"id": q["id"], "이름": q["id"], **stored(q["question"], agent.ask(q["question"]), False)})
    for q in gold["multi"]:
        carry, count, turns = None, 0, []
        for question in q["turns"]:
            out = agent.ask(question, carry=carry)
            turns.append(stored(question, out, bool(carry)))
            carry, count = update_carry(out, count)
        results["multi"].append({"id": q["id"], "이름": q["id"], "대화": turns})
    return results


def summary(scored):
    return "\n".join(f"{key}: {counts['passed']}/{counts['total']}" for key, counts in scored["totals"].items())


def report(scored):
    columns = (*METRICS, "passed")
    lines = ["# 처음 보는 질문 평가", "", "```text", summary(scored), "```", "",
             "| ID | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
    for row in scored["items"]:
        marks = ["-" if row.get(k) is None else ("○" if row[k] else "×") for k in columns]
        lines.append(f"| {row['id']} | " + " | ".join(marks) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--score", metavar="PATH")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    gold = read(GOLD)
    if args.score:
        scored = score(read(args.score), gold)
    else:
        results = run(gold)
        scored = score(results, gold)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(report(scored), encoding="utf-8")
    print(summary(scored))


if __name__ == "__main__":
    main()
