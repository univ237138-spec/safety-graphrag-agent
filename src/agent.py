"""제조 현장 위험성평가 Agent (LangGraph).

    parse → link → guide ─┬─ (빈 조건: 안내 또는 해석 오류) → 끝
                         └─ retrieve ─┬─ (사례 있음) → rank → explain → verify → 끝
                                ↑     │                       ↑         │
                                │     │                       └─ 검증 실패 (최대 2번)
                                └ relax (0건: 세부 설비 → 설비유형 → 작업만 → 설비만)
                                      └─ (끝까지 0건: 안내) → 끝

위험요인 판단은 그래프 조회만 한다. LLM은 질문 해석과 현장 설명에 쓴다.
질문 임베딩은 세부 설비 후보 검색과 상세 2칸 선택에 함께 쓴다 (질문당 1회).
설명이 끝까지 검증을 통과하지 못하면 설명 없이 그래프 결과와 llm_error를 돌려준다.

    python src/agent.py                 # 워크플로우 그림(Mermaid)
    python src/agent.py "질문"           # 실행 (OpenAI 키 사용)
"""
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

sys.path.insert(0, str(Path(__file__).resolve().parent))
import answer  # noqa: E402
import llm  # noqa: E402
import query_parser  # noqa: E402
from graph import load_graph  # noqa: E402
from vector_rag import CaseRelevance, question_vector  # noqa: E402

MAX_EXPLAIN_RETRIES = 2  # 설명 검증 실패 시 최대 2번 재생성


class State(TypedDict, total=False):
    question: str | None
    given: dict | None        # 질문 해석을 건너뛸 때 주는 조건 {'targets': [...], 'situations': [...]}
    carry: dict | None
    parse_ok: bool | None
    guidance: dict | None
    explain: bool
    parsed: dict | None
    query: Any
    candidates: list
    conditions: dict
    base: dict                # 완화 전 조건 (단계마다 여기서 다시 푼다)
    relax_level: int
    relax_note: str | None
    rank_note: str | None
    case_ids: list
    evidence: dict
    reply: Any
    items: dict | None
    attempts: int
    llm_error: str | None


def find_cases(g, conditions):
    """설비(여럿이면 합집합, 누락 방지 우선) ∩ 작업 상황(여럿이면 합집합)."""
    situations = conditions["situations"]
    by_situation = set().union(*(g.by["situation"].get(s, []) for s in situations)) if situations else None
    if not conditions["targets"]:
        return sorted(by_situation or [])
    ids = set()
    for t in conditions["targets"]:
        cases = set(g.by[t["kind"]].get(t["name"], []))
        ids |= cases & by_situation if by_situation is not None else cases
    return sorted(ids)


def relaxed(conditions, level):
    """0건이면 단계적으로 조건을 푼다. 해당 단계가 의미 없으면 None."""
    targets, situations = conditions["targets"], conditions["situations"]
    if level == 1 and any(t["kind"] == "equipment" for t in targets):
        up = {t["type"]: {"kind": "equipment_type", "name": t["type"], "mention": t["mention"], "type": t["type"]} for t in targets}
        return {"targets": list(up.values()), "situations": situations}, "세부 설비로 찾은 사례가 없어 설비유형 전체로 넓혔습니다"
    if level == 2 and targets and situations:
        return {"targets": [], "situations": situations}, "설비와 작업을 함께 만족하는 사례가 없어 작업 상황만으로 찾았습니다"
    if level == 3 and targets and situations:
        up = {t["type"]: {"kind": "equipment_type", "name": t["type"], "mention": t["mention"], "type": t["type"]} for t in targets}
        return {"targets": list(up.values()), "situations": []}, "작업 상황 기준 사례가 없어 설비유형만으로 찾았습니다"
    return None, None


def guidance(g, conditions):
    """확인된 조건의 사례 합집합에서 다음 선택지를 고른다."""
    targets, situations = conditions["targets"], conditions["situations"]
    if not targets and not situations:
        return {"type": "need_context", "message": "질문에서 설비나 작업을 특정하지 못했습니다. 어떤 설비로 어떤 작업을 하시는지 알려 주세요. "
                "예: '지게차로 도크에서 파렛트 상하차', '산업용로봇 셀 안에서 점검'", "options": []}
    if targets and situations:
        return None
    kind = "situation" if targets else "equipment_type"
    counts = Counter(g.cases[i][kind] for i in find_cases(g, conditions)
                     if g.cases[i][kind] and g.cases[i][kind] != "분류 불가")
    options = [{"kind": kind, "name": name, "cases": count}
               for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:3]]
    return {"type": "need_situation" if targets else "need_equipment",
            "message": "어떤 작업인지(운반·상하차·점검·청소 등) 알려 주시면 위험요인을 더 좁혀 드릴게요." if targets else
                       "어떤 설비인지 알려 주시면 더 좁혀 드릴게요.", "options": options}


def merge_conditions(carry, conditions, question):
    """이어받기는 앞 조건에 더하고, '~말고'로 부정한 설비만 뺀다. 처음부터는 '새 질문으로 시작'."""
    targets, seen = [], set()
    for target in carry.get("targets", []) + conditions["targets"]:
        names = [target.get(key) for key in ("name", "mention", "type")]
        names.append(query_parser.inner_name(target["name"]))
        if any(query_parser.negated_hit(question, name) for name in names):
            continue
        key = target["kind"], target["name"]
        if key not in seen:
            seen.add(key)
            targets.append(target)
    situations = list(dict.fromkeys(carry.get("situations", []) + conditions["situations"]))
    return {"targets": targets, "situations": situations}


def build(agent):
    g = agent.graph

    def parse(state):
        if state.get("given") is not None:
            return {"parsed": None, "parse_ok": None}
        try:
            parsed = (agent.parser(state["question"]) if agent.parser else
                      query_parser.parse(state["question"], g, agent.model, state["candidates"]))
            return {"parsed": parsed, "parse_ok": True, "llm_error": None}
        except llm.LLMError as exc:
            return {"parsed": None, "parse_ok": False, "llm_error": f"질문 해석 실패: {exc}"}

    def link(state):
        conditions = state.get("given")
        if conditions is None:
            conditions = query_parser.link(g, state.get("parsed") or {}, state.get("question") or "", state["candidates"])
            # 새 질문을 이해하지 못했으면 앞 조건으로 대신 답하지 않는다
            if state.get("carry") is not None and state.get("parse_ok") is not False:
                conditions = merge_conditions(state["carry"], conditions, state.get("question") or "")
        return {"conditions": conditions, "base": conditions, "relax_level": 0, "relax_note": None}

    def guide(state):
        c = state["conditions"]
        if state.get("parse_ok") is False and not (c["targets"] or c["situations"]):
            return {"guidance": None}
        return {"guidance": guidance(g, c)}

    def retrieve(state):
        return {"case_ids": agent.find(state["conditions"])}

    def relax(state):
        level = state["relax_level"]
        while level < 3:
            level += 1
            conditions, note = relaxed(state["base"], level)
            if conditions:
                return {"conditions": conditions, "relax_level": level, "relax_note": note}
        return {"relax_level": 4}

    def rank(state):
        scores, note = None, None
        if (state.get("question") or "").strip():
            try:
                if isinstance(agent.relevance, CaseRelevance):
                    if state.get("query") is None:
                        raise ValueError("질문 임베딩을 사용할 수 없습니다.")
                    scores = agent.relevance.score(state["query"], state["case_ids"])
                else:
                    scores = agent.relevance(state["question"], state["case_ids"])
            except Exception:
                note = "질문 관련도를 계산하지 못해 사례 수 순으로 정렬했습니다"
        return {"evidence": answer.build_evidence(g, state["case_ids"], scores), "rank_note": note}

    def explain(state):
        request = answer.build_request(state.get("question") or answer.condition_text(state["conditions"]),
                                       state["evidence"], agent.model, state.get("llm_error"))
        attempts = state.get("attempts", 0) + 1
        try:
            return {"reply": agent.explainer(request), "attempts": attempts, "llm_error": None}
        except llm.LLMError as exc:
            return {"reply": None, "attempts": attempts, "llm_error": str(exc)}

    def verify(state):
        try:
            return {"items": answer.validate(state["reply"], state["evidence"]), "llm_error": None}
        except llm.LLMError as exc:
            return {"items": None, "llm_error": str(exc)}

    def after_guide(state):
        c = state["conditions"]
        return "retrieve" if c["targets"] or c["situations"] else END

    def after_retrieve(state):
        if state["case_ids"]:
            return "rank"
        return "relax" if state["relax_level"] < 3 else END

    def after_relax(state):
        return "retrieve" if state["relax_level"] <= 3 else END

    def after_rank(state):
        return "explain" if state.get("explain") else END

    def after_explain(state):
        return "verify" if state["reply"] is not None else END  # API 오류는 재시도하지 않는다

    def after_verify(state):
        return "explain" if state["items"] is None and state["attempts"] <= MAX_EXPLAIN_RETRIES else END

    flow = StateGraph(State)
    for node in (parse, link, guide, retrieve, relax, rank, explain, verify):
        flow.add_node(node.__name__, node)
    flow.add_edge(START, "parse")
    flow.add_edge("parse", "link")
    flow.add_edge("link", "guide")
    flow.add_conditional_edges("guide", after_guide, ["retrieve", END])
    flow.add_conditional_edges("retrieve", after_retrieve, ["rank", "relax", END])
    flow.add_conditional_edges("relax", after_relax, ["retrieve", END])
    flow.add_conditional_edges("rank", after_rank, ["explain", END])
    flow.add_conditional_edges("explain", after_explain, ["verify", END])
    flow.add_conditional_edges("verify", after_verify, ["explain", END])
    return flow.compile()


class SafetyAgent:
    """store: 사례 조회기 (find_cases(conditions) → 사례 id). 없으면 메모리 그래프, API는 로컬 Neo4j를 넣는다."""

    def __init__(self, graph=None, model=None, parser=None, explainer=None, store=None, relevance=None,
                 embedder=None, candidates=None):
        self.graph = graph or load_graph()
        self.find = store.find_cases if store else (lambda conditions: find_cases(self.graph, conditions))
        self.model = llm.model_from_env(model)
        self.parser = parser
        self.embedder = embedder
        self.candidates = candidates if candidates is not None else query_parser.NodeCandidates(self.graph)
        self.explainer = explainer or llm.call
        self.relevance = relevance if relevance is not None else CaseRelevance(self.graph)
        self.flow = build(self)

    def ask(self, question=None, conditions=None, explain=True, carry=None):
        usage, started = [], time.perf_counter()
        token = llm.USAGE.set(usage)
        try:
            query, candidates = None, []
            if (question or "").strip():
                if self.parser is None or isinstance(self.relevance, CaseRelevance) or self.embedder is not None:
                    try:
                        query = question_vector(question, self.embedder)
                    except Exception:
                        pass  # 임베딩 실패는 글자 검색과 사례 수 정렬로 보완한다.
                if conditions is None:
                    candidates = self.candidates.search(question, query)
            state = self.flow.invoke({"question": question, "given": conditions, "explain": explain, "carry": carry,
                                      "rank_note": None, "query": query, "candidates": candidates})
        finally:
            llm.USAGE.reset(token)
        result = {key: state.get(key) for key in ("question", "parsed", "conditions", "relax_level", "relax_note", "rank_note",
                                                   "case_ids", "evidence", "reply", "items", "attempts", "llm_error", "guidance", "parse_ok")}
        if not state.get("case_ids"):
            result["evidence"] = None
            c = state["conditions"]
            if c["targets"] or c["situations"]:
                result["guidance"] = {"type": "no_cases", "message": "이 조건에 맞는 사망사고 기록이 없습니다. 설비나 작업을 바꿔 보세요.", "options": []}
        result["metrics"] = {"latency_s": round(time.perf_counter() - started, 2), "llm_calls": len(usage),
                             "input_tokens": sum(u["input_tokens"] for u in usage),
                             "output_tokens": sum(u["output_tokens"] for u in usage)}
        result["text"] = answer.render(result)
        return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        class _Stub:  # 그림만 그릴 때는 데이터가 필요 없다
            graph = model = parser = explainer = find = None
        print(build(_Stub()).get_graph().draw_mermaid())
    else:
        out = SafetyAgent().ask(" ".join(sys.argv[1:]))
        print(out["text"])
        print(out["metrics"])
