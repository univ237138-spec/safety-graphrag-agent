"""Q4: 현장 실무자 말투의 자연어 질문 생성 (gpt-5-mini) → data/eval/q4_draft.json (검수 후 q4.json)

정답은 LLM이 아니라 씨앗(세부 설비 + 고위험작업)에서 그래프로 계산한다. LLM은 질문 문장만 쓴다.
사용자 검수 후 questions.json에 합친다.
실행: python src/make_q4.py [--dry-run] [--domain]
  --domain: 디밀리언 분야(로봇·자동화 라인·물류 동선) 씨앗 D1~D4만 생성해 기존 초안에 덧붙인다
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm  # noqa: E402
from eval_set import VAGUE_TYPES  # noqa: E402
from graph import ROOT, load_graph  # noqa: E402

OUT = ROOT / "data" / "eval" / "q4_draft.json"
N_SEEDS = 16
# 회사 분야와 맞닿고 실제 사례 3건+인 조합만 (AMR·AGV는 데이터 0건이라 제외)
DOMAIN_SEEDS = [("지게차", "혼재작업(작업자-작업차량)"), ("산업용로봇", "비정형 작업(점검)"),
                ("자동화설비(파렛타이저)", "비정형 작업(점검)"), ("산업용로봇", "자동화설비(산업용로봇 포함) 작업")]
MIN_CASES = 3
# 질문에 들어가면 정답(사고 형태)을 흘리는 단어
LEAK_WORDS = ["끼임", "끼여", "추락", "떨어짐", "떨어져", "깔림", "깔려", "협착", "감전", "사망", "질식", "폭발", "화재", "전도", "낙하", "부딪"]

INSTRUCTIONS = """너는 한국 제조·물류 현장의 실무자다. 주어진 각 '씨앗'(업종, 세부 설비, 작업 상황, 참고 사고 개요)마다
작업 전에 AI 안전 도우미에게 보낼 질문을 1개씩 쓴다.

규칙
1. 실무자 1인칭 말투. 직무를 하나 정한다 (예: 생산팀 반장, 설비보전 담당, 물류센터 상하차 조장, 안전관리자, 협력업체 작업자).
2. 실제 현장처럼 구체적으로: 공정·라인 이름, 작업 내용, 인원·시간대·설비 상태(가동 중, 정지 예정, 야간 등) 중 2~3개를 넣는다.
   TBM, 작업허가서, 비정형 작업, 정비 요청, 금형 교체, 파렛트, 도크, 랙 같은 현장 용어를 자연스럽게 쓴다.
3. 설비는 현장에서 부르는 이름으로 써도 되지만 같은 설비임을 알 수 있어야 한다.
4. 정답 누설 금지: 사고 형태(끼임, 추락, 깔림, 감전, 폭발, 사망 등)와 위험요인을 질문에 쓰지 않는다. 참고 사고 개요 문장을 베끼지 않는다.
5. 마지막은 위험요인과 안전조치를 묻는 말로 끝내되 표현을 매번 다르게 한다.
6. 1~3문장, 60~200자.
"""

SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["questions"],
    "properties": {"questions": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["seed_id", "persona", "question"],
        "properties": {"seed_id": {"type": "string"}, "persona": {"type": "string"}, "question": {"type": "string"}}}}},
}


def pick_seeds(g, n=N_SEEDS):
    """사례 많은 (세부 설비, 작업) 조합부터, 작업 상황·설비유형이 겹치지 않게 고른다. 물류(운수창고) 2개 이상."""
    combos = defaultdict(list)
    for c in g.cases.values():
        if c["equipment"] and c["equipment_type"] not in VAGUE_TYPES:
            combos[(c["equipment"], c["situation"])].append(c["id"])
    ranked = sorted(((k, v) for k, v in combos.items() if len(v) >= MIN_CASES), key=lambda kv: -len(kv[1]))
    is_logistics = lambda ids: any(g.cases[i]["industry"] == "운수창고통신업" for i in ids)  # noqa: E731
    seeds, used_s, used_t = [], set(), set()
    for want_logistics in (True, False):
        for (e, s), ids in ranked:
            t = g.cases[ids[0]]["equipment_type"]
            if len(seeds) >= n or (want_logistics and (not is_logistics(ids) or len(seeds) >= 2)):
                continue
            if s in used_s or t in used_t:
                continue
            seeds.append((e, s, ids))
            used_s.add(s)
            used_t.add(t)
    return seeds


def seed_payload(g, seeds, prefix="S"):
    rows = []
    for k, (e, s, ids) in enumerate(seeds, 1):
        sectors = Counter(g.cases[i]["subsector"] for i in ids).most_common(2)
        rows.append({"seed_id": f"{prefix}{k:02d}", "업종": [x for x, _ in sectors], "세부 설비": e, "작업 상황": s,
                     "참고 사고 개요": [g.cases[i]["overview"] for i in ids[:3]]})
    return rows


def leaks(text):
    return [w for w in LEAK_WORDS if w in text]


def main(dry_run=False, domain=False):
    g = load_graph()
    if domain:
        picked = [(e, s, g.case_ids(equipment=e, situation=s)) for e, s in DOMAIN_SEEDS]
        seeds = seed_payload(g, picked, prefix="D")
    else:
        picked = pick_seeds(g)
        seeds = seed_payload(g, picked)
    for s, (_, _, ids) in zip(seeds, picked):
        print(s["seed_id"], s["세부 설비"], "|", s["작업 상황"], "| 사례", len(ids))
    if dry_run:
        return
    usage = []
    llm.USAGE.set(usage)
    request = llm.build_request(llm.model_from_env(), INSTRUCTIONS,
                                json.dumps(seeds, ensure_ascii=False), SCHEMA, "q4_questions", max_output_tokens=16000)
    result = llm.call(request, timeout=300)
    by_seed = {s["seed_id"]: (s, e, st) for s, (e, st, _) in zip(seeds, picked)}
    drafts = []
    for q in result["questions"]:
        seed, e, s = by_seed[q["seed_id"]]
        ids = g.case_ids(equipment=e, situation=s)
        drafts.append({"id": f"Q4-{q['seed_id']}", "type": "Q4", "domain": domain, "persona": q["persona"],
                       "question": q["question"], "equipment": e, "situation": s, "subsectors": seed["업종"],
                       "leak_words": leaks(q["question"]), "case_ids": ids,
                       "gold_factors": [{"factor": f, "cases": n} for f, n in g.factors(ids)]})
    if domain and OUT.exists():  # 기존 일반 질문은 두고 D 질문만 교체
        drafts = [d for d in json.loads(OUT.read_text(encoding="utf-8")) if not d["id"].startswith("Q4-D")] + drafts
    OUT.write_text(json.dumps(drafts, ensure_ascii=False, indent=1), encoding="utf-8")
    tokens = usage[0] if usage else {}
    print(f"저장: {OUT} ({len(drafts)}개) | 토큰 입력 {tokens.get('input_tokens')} / 출력 {tokens.get('output_tokens')}")

if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv, domain="--domain" in sys.argv)
