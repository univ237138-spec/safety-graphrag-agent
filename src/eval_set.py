"""평가셋 Q1~Q3 생성 → data/eval/questions.json

정답 = 질문 조건(작업·설비)에 해당하는 사례들의 재해유발요인 전체 (근거 사례 수 포함).
Q4(자연어 작업 설명)는 make_q4.py로 만들고 사용자 검수 후 data/eval/q4.json에 둔다 (정답은 여기서 다시 계산).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph import ROOT, load_graph  # noqa: E402

OUT = ROOT / "data" / "eval" / "questions.json"
Q4 = ROOT / "data" / "eval" / "q4.json"
MIN_CASES = {"Q1": 5, "Q2": 5, "Q3": 3}  # 사례가 너무 적은 조건은 질문에서 제외
VAGUE_TYPES = {"기타 설비", "기타 차량"}  # 현장에서 이 이름으로 묻지 않음


def question(g, qid, qtype, text, equipment=None, situation=None, **extra):
    ids = g.case_ids(equipment=equipment, situation=situation)
    return {"id": qid, "type": qtype, "question": text, "equipment": equipment, "situation": situation, **extra,
            "case_ids": ids, "gold_factors": [{"factor": f, "cases": n} for f, n in g.factors(ids)]}


def build(g):
    questions = []
    for s in g.names("situation"):
        if len(g.by["situation"][s]) >= MIN_CASES["Q1"]:
            questions.append(question(g, f"Q1-{len(questions) + 1:03d}", "Q1", f"'{s}'의 위험요인과 감소대책은?", situation=s))
    for e in g.names("equipment"):
        if len(g.by["equipment"][e]) >= MIN_CASES["Q2"] and g.cases[g.by["equipment"][e][0]]["equipment_type"] not in VAGUE_TYPES:
            questions.append(question(g, f"Q2-{len(questions) + 1:03d}", "Q2", f"'{e}'을(를) 사용하는 작업의 위험요인과 감소대책은?", equipment=e))
    combos = defaultdict(int)
    for c in g.cases.values():
        if c["equipment"] and c["equipment_type"] not in VAGUE_TYPES:
            combos[(c["equipment"], c["situation"])] += 1
    for (e, s), n in sorted(combos.items()):
        if n >= MIN_CASES["Q3"]:
            questions.append(question(g, f"Q3-{len(questions) + 1:03d}", "Q3", f"'{e}' 관련 '{s}'의 위험요인과 감소대책은?", equipment=e, situation=s))
    if Q4.exists():
        for q in json.loads(Q4.read_text(encoding="utf-8")):
            questions.append(question(g, q["id"], "Q4", q["question"], q["equipment"], q["situation"],
                                      persona=q["persona"], domain=q["domain"], review_flags=q["review_flags"]))
    return questions


def main():
    questions = build(load_graph())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(questions, ensure_ascii=False, indent=1), encoding="utf-8")
    counts = defaultdict(int)
    for q in questions:
        counts[q["type"]] += 1
    print(f"저장: {OUT} {dict(counts)}")


if __name__ == "__main__":
    main()
