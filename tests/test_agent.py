"""Agent 회귀 테스트 (API 키 없이: LLM 자리에 가짜 함수)."""
import json
import sys
from pathlib import Path

import pytest
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import llm  # noqa: E402
import vector_rag  # noqa: E402
from agent import SafetyAgent, find_cases, merge_conditions  # noqa: E402
from answer import CASES_PER_FACTOR, build_evidence, build_request, validate  # noqa: E402
from graph import load_graph  # noqa: E402
from query_parser import link, link_equipment  # noqa: E402

G = load_graph()
QUESTIONS = json.loads((ROOT / "data/eval/questions.json").read_text(encoding="utf-8"))
FORKLIFT = {"targets": [{"kind": "equipment_type", "name": "지게차"}], "situations": []}


def good_reply(request):
    """evidence 규칙을 지키는 가짜 LLM 답변."""
    data = json.loads(request["input"][0]["content"])
    items = [{"factor_id": d["id"], "advice": "작업 전 확인", "evidence_ids": list(d["measures"])[:1] or list(d["cases"])[:1]}
             for d in data["detailed"]]
    return {"summary": f"요인 {data['total_factors']}개 중 먼저 F1을 챙기세요.", "items": items}


def no_llm(_):
    raise AssertionError("LLM을 부르면 안 되는 경로")


def no_relevance(question, case_ids):
    return None


@pytest.mark.parametrize("mention, etype, kind, name", [
    ("천장주행크레인", "크레인", "equipment", "크레인(천장주행크레인)"),
    ("천장크레인", "크레인", "equipment_type", "크레인"),  # 애매하면 유형 전체 (누락 방지)
    ("탱크 내부 가스", "유해가스", "equipment_type", "유해가스"),
    ("팔레타이저 리프트", "자동화설비", "equipment_type", "자동화설비"),  # 두 세부 설비 동점 → 유형 전체
    ("오버헤드 크레인", "크레인", "equipment_type", "크레인"),
    ("프레스 금형", "생산설비", "equipment", "생산설비(프레스)"),
    ("2톤 지게차", "지게차", "equipment_type", "지게차"),  # 일반 표기 → 유형 전체
    ("이동식 크레인", "크레인", "equipment", "크레인(이동식크레인/카고크레인)"),
])
def test_link_equipment(mention, etype, kind, name):
    target = link_equipment(G, mention, etype)
    assert (target["kind"], target["name"]) == (kind, name)


def test_link_drops_values_outside_graph():
    parsed = {"equipment": [{"mention": "드론", "type": "없는유형"}], "situations": ["지게차 사용 작업", "없는 작업"]}
    assert link(G, parsed) == {"targets": [], "situations": ["지게차 사용 작업"]}


def test_link_adds_equipment_named_in_question():
    # 실제 실행: LLM이 '산업용로봇 셀'을 '기계식 주차설비'로, '조립1라인'을 설비로 오분류
    parsed = {"equipment": [{"mention": "조립1라인", "type": "해당없음"}, {"mention": "산업용로봇 셀", "type": "기계식 주차설비"}],
              "situations": ["비정형 작업(점검)"]}
    linked = link(G, parsed, "조립1라인 산업용로봇 셀 내부 점검")
    assert {"산업용로봇"} <= {t["type"] for t in linked["targets"]}
    assert "해당없음" not in {t["type"] for t in linked["targets"]}


def test_generic_type_dropped_when_specific_present():
    parsed = {"equipment": [{"mention": "교반기", "type": "혼합기"}, {"mention": "설비", "type": "생산설비"}], "situations": []}
    targets = link(G, parsed)["targets"]
    assert {(t["kind"], t["name"]) for t in targets} == {
        ("equipment", "혼합기(교반기)"), ("equipment", "생산설비(교반기)")}
    only_generic = {"equipment": [{"mention": "설비", "type": "생산설비"}], "situations": []}
    assert [t["type"] for t in link(G, only_generic)["targets"]] == ["생산설비"]  # 구체적인 게 없으면 유지


def test_generic_names_not_added_from_text():
    assert link(G, {"equipment": [], "situations": []}, "생산설비 구조물 주변 작업")["targets"] == []


@pytest.mark.parametrize("q", [q for q in QUESTIONS if q["type"] == "Q3"][:20], ids=lambda q: q["id"])
def test_graph_retrieval_matches_gold(q):
    agent = SafetyAgent(G, parser=no_llm, explainer=no_llm)
    given = {"targets": [{"kind": "equipment", "name": q["equipment"], "mention": q["equipment"], "type": None}],
             "situations": [q["situation"]]}
    out = agent.ask(conditions=given, explain=False)
    assert {f["factor"]: f["cases"] for f in out["evidence"]["factors"]} == {g["factor"]: g["cases"] for g in q["gold_factors"]}
    assert out["metrics"]["llm_calls"] == 0


def test_multiple_equipment_is_union():
    crane = {"kind": "equipment", "name": "크레인(천장주행크레인)", "mention": "", "type": "크레인"}
    press = {"kind": "equipment", "name": "생산설비(프레스)", "mention": "", "type": "생산설비"}
    both = set(find_cases(G, {"targets": [crane, press], "situations": []}))
    assert both == set(G.by["equipment"]["크레인(천장주행크레인)"]) | set(G.by["equipment"]["생산설비(프레스)"])


def test_relax_to_equipment_type_when_no_cases():
    detail = {"kind": "equipment", "name": "생산설비(프레스)", "mention": "프레스", "type": "생산설비"}
    situation = "사다리 사용 작업·통행"
    assert not find_cases(G, {"targets": [detail], "situations": [situation]})
    out = SafetyAgent(G, explainer=no_llm).ask(conditions={"targets": [detail], "situations": [situation]}, explain=False)
    assert out["relax_level"] >= 1 and out["case_ids"] and out["relax_note"]


def test_parse_failure_ends_without_cases():
    def broken(_):
        raise llm.LLMError("API 연결 실패")
    out = SafetyAgent(G, parser=broken, explainer=no_llm, relevance=no_relevance).ask("작업 위험요인 알려줘")
    assert out["case_ids"] is None and "질문 해석 실패" in out["llm_error"]
    assert out["parse_ok"] is False and out["guidance"] is None
    # 해석이 실패해도 질문에 설비 이름이 그대로 있으면 글자 일치로 계속 답한다
    out = SafetyAgent(G, parser=broken, explainer=no_llm, relevance=no_relevance).ask("지게차 작업 위험요인?", explain=False)
    assert out["case_ids"] and out["conditions"]["targets"][0]["type"] == "지게차"
    assert out["parse_ok"] is False and out["guidance"]["type"] == "need_situation"


def test_empty_parse_guides_without_retrieval():
    class Store:
        def find_cases(self, conditions):
            raise AssertionError("빈 조건으로 조회하면 안 됩니다")
    out = SafetyAgent(G, parser=lambda q: {}, store=Store()).ask("무엇을 할까요", explain=False)
    assert out["parse_ok"] is True and out["case_ids"] is None
    assert out["guidance"]["type"] == "need_context" and out["guidance"]["options"] == []


@pytest.mark.parametrize("conditions, hint, kind", [
    (FORKLIFT, "need_situation", "situation"),
    ({"targets": [], "situations": ["비정형 작업(점검)"]}, "need_equipment", "equipment_type"),
    ({"targets": FORKLIFT["targets"] + [{"kind": "equipment", "name": "크레인(천장주행크레인)"}],
      "situations": []}, "need_situation", "situation"),
])
def test_partial_conditions_answer_and_guide(conditions, hint, kind):
    from collections import Counter
    out = SafetyAgent(G, parser=no_llm, explainer=no_llm).ask(conditions=conditions, explain=False)
    counts = Counter(G.cases[i][kind] for i in find_cases(G, conditions)
                     if G.cases[i][kind] and G.cases[i][kind] != "분류 불가")
    expected = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:3]
    assert out["evidence"] and out["parse_ok"] is None
    assert out["guidance"]["type"] == hint
    assert out["guidance"]["options"] == [{"kind": kind, "name": name, "cases": n} for name, n in expected]


def test_carry_merges_conditions_without_concatenating_question():
    questions = []
    def parse(question):
        questions.append(question)
        return {"situations": ["지게차 사용 작업"]}
    out = SafetyAgent(G, parser=parse, relevance=no_relevance).ask("운반할 때", carry=FORKLIFT, explain=False)
    assert questions == ["운반할 때"]
    assert out["conditions"] == {"targets": FORKLIFT["targets"], "situations": ["지게차 사용 작업"]}
    assert out["guidance"] is None


def test_new_targets_add_to_carry_and_given_ignores_carry():
    agent = SafetyAgent(G, parser=lambda q: {"equipment": [{"mention": "산업용로봇", "type": "산업용로봇"}]},
                        relevance=no_relevance)
    out = agent.ask("다른 설비", carry=FORKLIFT, explain=False)
    assert [t["name"] for t in out["conditions"]["targets"]] == ["지게차", "산업용로봇"]
    out = agent.ask(conditions={"targets": [], "situations": []}, carry=FORKLIFT, explain=False)
    assert out["conditions"] == {"targets": [], "situations": []} and out["parse_ok"] is None


def test_carry_keeps_forklift_when_truck_and_unloading_added():
    situation = "상하차 작업 또는 화물(물체)이 적재된 장소 주변에서의 작업"
    parsed = {"equipment": [{"mention": "트럭", "type": "화물운반트럭"}], "situations": [situation]}
    out = SafetyAgent(G, parser=lambda q: parsed, relevance=no_relevance).ask(
        "도크에서 트럭에 파렛트 상하차하는 작업이에요", carry=FORKLIFT, explain=False)
    assert out["conditions"]["targets"][0] == FORKLIFT["targets"][0]
    assert "화물운반트럭" in {t["type"] for t in out["conditions"]["targets"][1:]}
    assert out["conditions"]["situations"] == [situation] and out["case_ids"]


def test_carry_keeps_confined_space_when_cleaning_added():
    confined = "피트, 맨홀, 오·폐수 처리시설 등 밀폐된 장소에서 작업"
    cleaning = "비정형 작업(이물질 제거·청소)"
    carry = {"targets": [], "situations": [confined]}
    out = SafetyAgent(G, parser=lambda q: {"situations": [cleaning]}, relevance=no_relevance).ask(
        "정화조 청소예요", carry=carry, explain=False)
    assert out["conditions"] == {"targets": [], "situations": [confined, cleaning]}
    assert out["case_ids"] and carry == {"targets": [], "situations": [confined]}


def test_carry_removes_negated_forklift_from_old_and_new_targets():
    parsed = {"equipment": [{"mention": "운반장비", "type": "지게차"},
                            {"mention": "산업용로봇", "type": "산업용로봇"}],
              "situations": ["비정형 작업(점검)"]}
    out = SafetyAgent(G, parser=lambda q: parsed, relevance=no_relevance).ask(
        "아니 지게차 말고 산업용로봇 셀 안에서 점검하는 거예요", carry=FORKLIFT, explain=False)
    assert [t["type"] for t in out["conditions"]["targets"]] == ["산업용로봇"]
    assert out["conditions"]["situations"] == ["비정형 작업(점검)"] and out["case_ids"]


@pytest.mark.parametrize("negation", ["말고", "아니라", "아니고", "대신", "빼고"])
@pytest.mark.parametrize("alias", ["name", "mention", "type", "inner"])
def test_carry_negation_matches_each_target_alias(negation, alias):
    target = {"kind": "equipment", "name": "설비", "mention": "장비", "type": "유형"}
    if alias == "inner":
        target["name"] = "생산설비(프레스)"
    else:
        target[alias] = "프레스"
    carry = {"targets": [target], "situations": ["앞 작업"]}
    conditions = {"targets": [target], "situations": ["새 작업"]}
    assert merge_conditions(carry, conditions, f"프레스 {negation} 로봇") == {
        "targets": [], "situations": ["앞 작업", "새 작업"]}


def test_carry_deduplicates_by_target_and_preserves_first_order():
    first = {"kind": "equipment_type", "name": "지게차", "mention": "앞 표현"}
    duplicate = {**first, "mention": "새 표현"}
    robot = {"kind": "equipment_type", "name": "산업용로봇"}
    carry = {"targets": [first], "situations": ["앞 작업", "겹친 작업"]}
    conditions = {"targets": [duplicate, robot, robot], "situations": ["겹친 작업", "새 작업"]}
    assert merge_conditions(carry, conditions, "추가 작업") == {
        "targets": [first, robot], "situations": ["앞 작업", "겹친 작업", "새 작업"]}
    assert carry["targets"] == [first] and len(conditions["targets"]) == 3


def test_failed_parse_ignores_carry():
    # 해석 실패에 carry를 쓰면 새 질문과 무관한 앞 조건으로 답하게 된다 → 쓰지 않는다
    def broken(_):
        raise llm.LLMError("해석 실패")
    out = SafetyAgent(G, parser=broken, relevance=no_relevance).ask("추가 설명", carry=FORKLIFT, explain=False)
    assert out["parse_ok"] is False and out["evidence"] is None and out["llm_error"]
    assert out["guidance"] is None


def test_no_cases_guides_after_relaxation():
    class Store:
        def find_cases(self, conditions):
            return []
    out = SafetyAgent(G, store=Store()).ask(conditions=FORKLIFT, explain=False)
    assert out["relax_level"] == 4 and out["guidance"]["type"] == "no_cases"


def test_explain_success_covers_all_factors():
    out = SafetyAgent(G, parser=lambda q: {"equipment": [{"mention": "지게차", "type": "지게차"}],
                                           "situations": ["지게차 사용 작업"]},
                      explainer=good_reply, relevance=no_relevance).ask("지게차 작업")
    ev = out["evidence"]
    assert out["llm_error"] is None and set(out["items"]) == {d["id"] for d in ev["detailed"]}
    # 상세 5개 + 체크리스트로 요인 전부가 출력에 나온다 (누락 0)
    assert all(f["factor"] in out["text"] for f in ev["factors"])


def test_explain_retries_then_falls_back():
    calls = []

    def bad(request):
        calls.append(1)
        reply = good_reply(request)
        reply["items"] = reply["items"][1:]  # F1 누락
        return reply
    out = SafetyAgent(G, parser=no_llm, explainer=bad, relevance=no_relevance).ask(
        "지게차", conditions={"targets": [], "situations": ["지게차 사용 작업"]})
    assert len(calls) == 3 and out["items"] is None and "누락" in out["llm_error"]
    assert "근거만 표시" in out["text"] and all(f["factor"] in out["text"] for f in out["evidence"]["factors"])


def test_wrong_evidence_id_is_rejected():
    def wrong(request):
        reply = good_reply(request)
        reply["items"][0]["evidence_ids"] = ["M999"]
        return reply
    out = SafetyAgent(G, parser=no_llm, explainer=wrong, relevance=no_relevance).ask(
        "지게차", conditions={"targets": [], "situations": ["지게차 사용 작업"]})
    assert out["items"] is None and "M999" in out["llm_error"]


def test_build_evidence_uses_latest_year_cases():
    # 원본 뒤쪽 사례가 최신이 아닌 요인을 고른다
    case_ids = next(ids for ids in G.by["factor"].values()
                    if len(ids) > CASES_PER_FACTOR
                    and min(G.cases[i]["year"] or 0 for i in ids[-CASES_PER_FACTOR:])
                    < max(G.cases[i]["year"] or 0 for i in ids[:-CASES_PER_FACTOR]))
    evidence = build_evidence(G, case_ids)
    chosen = evidence["detailed"][0]["case_ids"]
    assert len(chosen) == CASES_PER_FACTOR
    assert all((G.cases[cid]["year"] or 0) >= (G.cases[other]["year"] or 0)
               for cid in chosen for other in case_ids if other not in chosen)


def test_validate_requires_exact_total_in_summary():
    evidence = build_evidence(G, list(G.cases))
    reply = good_reply(build_request("작업 위험요인", evidence, "test"))
    total = evidence["total_factors"]
    # 다른 숫자에 포함된 요인 수는 거부한다
    for summary in (f"{total}0개", f"1{total}개"):
        reply["summary"] = summary
        with pytest.raises(llm.LLMError):
            validate(reply, evidence)
    reply["summary"] = f"요인 {total}개"
    assert set(validate(reply, evidence)) == {d["id"] for d in evidence["detailed"]}


@pytest.mark.parametrize("explain", [False, True])
def test_question_slots_preserve_recall_and_validate(explain):
    case_ids = find_cases(G, FORKLIFT)
    before = build_evidence(G, case_ids)
    count = [f["factor"] for f in before["factors"]]
    low = count[-1]
    calls = []

    def relevance(question, ids):
        calls.append((question, ids))
        return {factor: float(factor == low) for factor in count}

    out = SafetyAgent(G, parser=no_llm, explainer=good_reply, relevance=relevance).ask(
        "지게차 작업", conditions=FORKLIFT, explain=explain)
    ev = out["evidence"]
    order = [f["factor"] for f in ev["factors"]]
    assert len(count) > 5 and calls == [("지게차 작업", case_ids)]
    assert order == count[:3] + [low, count[3]] + count[4:-1]
    assert {f["factor"]: f["cases"] for f in ev["factors"]} == {f["factor"]: f["cases"] for f in before["factors"]}
    assert [f["id"] for f in ev["factors"]] == [f"F{k}" for k in range(1, len(count) + 1)]
    assert [f["pick"] for f in ev["factors"]] == ["cases"] * 3 + ["question"] * 2 + [None] * (len(count) - 5)
    assert [d["pick"] for d in ev["detailed"]] == ["cases"] * 3 + ["question"] * 2
    assert out["rank_note"] is None and out["text"].count("질문과 관련 깊음") == 2
    if explain:
        assert out["llm_error"] is None and len(out["items"]) == 5
        assert validate(out["reply"], ev) == out["items"]


def test_relevance_failure_keeps_count_order():
    def broken(question, case_ids):
        raise RuntimeError("관련도 계산 실패")

    out = SafetyAgent(G, parser=no_llm, explainer=no_llm, relevance=broken).ask(
        "지게차 작업", conditions=FORKLIFT, explain=False)
    assert out["evidence"] == build_evidence(G, out["case_ids"])
    assert out["rank_note"] == "질문 관련도를 계산하지 못해 사례 수 순으로 정렬했습니다"
    assert out["rank_note"] in out["text"]


@pytest.mark.parametrize("question", [None, "", "   "])
def test_conditions_only_skips_relevance(question):
    calls = []
    out = SafetyAgent(G, parser=no_llm, explainer=no_llm,
                      relevance=lambda *args: calls.append(args)).ask(question, conditions=FORKLIFT, explain=False)
    assert calls == [] and out["rank_note"] is None
    assert out["evidence"] == build_evidence(G, out["case_ids"])


@pytest.mark.parametrize("size", range(6))
def test_question_slots_leave_small_factor_sets_unchanged(size):
    factors = list(G.by["factor"])[:size]
    case_ids = [i for i in G.cases if G.cases[i]["factor"] in factors]
    scores = {f: k for k, (f, _) in enumerate(G.factors(case_ids))}
    before = build_evidence(G, case_ids)
    assert build_evidence(G, case_ids, scores) == before
    assert all(d["pick"] == "cases" for d in before["detailed"])


def test_default_relevance_loads_once_and_records_embedding_usage(monkeypatch):
    loads, posts = [], []
    case_ids = find_cases(G, FORKLIFT)
    low = G.factors(case_ids)[-1][0]
    vectors = np.array([[0., 1.] if c["factor"] == low else [1., 0.] for c in G.cases.values()])

    def load(graph):
        loads.append(graph)
        return vectors

    def post(url, payload, timeout=120):
        posts.append((url, payload["input"]))
        response = {"data": [{"index": 0, "embedding": [0., 2.]}], "usage": {"prompt_tokens": 7}}
        llm.record_usage(payload["model"], response)
        return response

    monkeypatch.setattr(vector_rag, "load_case_vectors", load)
    monkeypatch.setattr(vector_rag, "embed_cached", no_llm)
    monkeypatch.setattr(llm, "post", post)
    agent = SafetyAgent(G, parser=no_llm, explainer=no_llm)
    assert loads == []
    for _ in range(2):
        out = agent.ask("지게차 작업", conditions=FORKLIFT, explain=False)
        assert out["rank_note"] is None and out["evidence"]["detailed"][3]["factor"] == low
        assert {k: out["metrics"][k] for k in ("llm_calls", "input_tokens", "output_tokens")} == \
            {"llm_calls": 1, "input_tokens": 7, "output_tokens": 0}
    assert loads == [G] and posts == [(llm.EMBEDDING_ENDPOINT, ["지게차 작업"])] * 2
    scores = agent.relevance("질문", case_ids)
    assert scores[low] == pytest.approx(1.)


@pytest.mark.parametrize("error", [FileNotFoundError, ValueError])
def test_default_relevance_failure_falls_back(monkeypatch, error):
    def broken(graph):
        raise error("캐시 또는 임베딩 실패")

    monkeypatch.setattr(vector_rag, "load_case_vectors", broken)
    monkeypatch.setattr(llm, "embed", no_llm)
    out = SafetyAgent(G, parser=no_llm, explainer=no_llm).ask(
        "지게차 작업", conditions=FORKLIFT, explain=False)
    assert out["rank_note"] and out["evidence"] == build_evidence(G, out["case_ids"])


def test_embedding_failure_falls_back(monkeypatch):
    def broken(texts):
        raise llm.LLMError("임베딩 연결 실패")

    monkeypatch.setattr(vector_rag, "load_case_vectors", lambda g: np.ones((len(g.cases), 2)))
    monkeypatch.setattr(llm, "embed", broken)
    out = SafetyAgent(G, parser=no_llm, explainer=no_llm).ask(
        "지게차 작업", conditions=FORKLIFT, explain=False)
    assert out["rank_note"] and out["evidence"] == build_evidence(G, out["case_ids"])


def test_case_cache_validates_texts_and_vectors(monkeypatch):
    from types import SimpleNamespace

    graph = SimpleNamespace(cases={"a": next(iter(G.cases.values()))})
    monkeypatch.setattr(llm, "embed", no_llm)
    texts = ["이전 문서"]
    monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: json.dumps(texts))
    with pytest.raises(ValueError, match="문서·순서"):
        vector_rag.load_case_vectors(graph)
    texts[0] = vector_rag.document(graph.cases["a"])
    for vectors in (np.ones(2), np.ones((2, 2))):
        monkeypatch.setattr(np, "load", lambda *args, **kwargs: vectors)
        with pytest.raises(ValueError, match="개수 또는 차원"):
            vector_rag.load_case_vectors(graph)
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: np.ones((1, 2)))
    assert np.array_equal(vector_rag.load_case_vectors(graph), np.ones((1, 2)))


def test_relevance_uses_maximum_within_retrieved_cases():
    from types import SimpleNamespace

    graph = SimpleNamespace(cases={"a": {"factor": "A"}, "b": {"factor": "A"}, "c": {"factor": "B"}})
    vectors = np.array([[1., 0.], [0., 1.], [0.6, 0.8]])
    positions = {"a": 0, "b": 1, "c": 2}
    query = np.array([0., 1.])
    assert vector_rag.factor_relevance(graph, ["a", "c"], positions, vectors, query) == {"A": 0., "B": 0.8}
    assert vector_rag.factor_relevance(graph, ["a", "b", "c"], positions, vectors, query) == {"A": 1., "B": 0.8}


def test_parse_failure_does_not_answer_with_carry():
    # 새 질문 해석이 실패하면 앞 조건(carry)으로 대신 답하지 않는다
    def broken(_):
        raise llm.LLMError("해석 API 오류")
    agent = SafetyAgent(G, model="test", parser=broken, relevance=lambda q, ids: {})
    carry = {"targets": [{"kind": "equipment_type", "name": "지게차", "mention": "지게차", "type": "지게차"}],
             "situations": ["지게차 사용 작업"]}
    out = agent.ask("아니 로봇 셀 점검이에요", explain=False, carry=carry)
    assert out["parse_ok"] is False and out["llm_error"]
    assert not out["case_ids"] and out["evidence"] is None
