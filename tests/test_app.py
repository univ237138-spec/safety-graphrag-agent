"""Streamlit 데모 테스트 (API 키·외부 네트워크 없이).

    python -m pytest -q -p no:warnings
"""
import json
from copy import deepcopy
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import app  # noqa: E402
import llm  # noqa: E402
import make_demo_records  # noqa: E402
import settings  # noqa: E402
import vector_rag  # noqa: E402
from agent import SafetyAgent  # noqa: E402
from graph import load_graph  # noqa: E402


@pytest.fixture
def demo(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "env", lambda name, default=None: default)
    monkeypatch.setattr(llm, "env", lambda name, default=None: default)
    monkeypatch.setenv("SAFETY_DEMO_RECORDS", str(tmp_path / "records.json"))
    app.resources.clear()
    yield AppTest.from_file(str(ROOT / "app.py"), default_timeout=20)
    app.resources.clear()


def fake_reply(request):
    data = json.loads(request["input"][0]["content"])
    items = [{"factor_id": d["id"], "advice": "작업 전 확인", "evidence_ids": list(d["measures"])[:1]}
             for d in data["detailed"]]
    return {"summary": f"요인 {data['total_factors']}개", "items": items}


@pytest.fixture
def records(tmp_path):
    agent = SafetyAgent(load_graph(), model="test", parser=lambda q: {
        "equipment": [{"mention": "지게차", "type": "지게차"}], "situations": []},
        explainer=fake_reply, relevance=lambda q, ids: {})
    out = agent.ask("지게차 작업")
    assert out["items"]
    out.pop("text")
    factors = [f["factor"] for f in out["evidence"]["factors"]]
    out.update(id="Q4-D01", group="물류·운반", order=1, note="지게차 도크 작업, 덱 예시", graph_factors=factors,
               vector_top5_factors=factors[:1], gold_factors=factors[:2] + ["정답인데 못 찾은 요인"])
    out["metrics"] = {"latency_s": 1, "llm_calls": 3, "input_tokens": 100, "output_tokens": 20}
    data = {"generated_at": "2026-11-03T12:00:00+09:00", "model": "test", "records": [out]}
    (tmp_path / "records.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def forbid_ask(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("이 동작은 Agent를 호출하면 안 됩니다.")
    monkeypatch.setattr(SafetyAgent, "ask", forbidden)


def enable_live(monkeypatch):
    on = {"OPENAI_API_KEY": "fake", "DEMO_LIVE": "1"}
    monkeypatch.setattr(settings, "env", lambda name, default=None: on.get(name, default))


def button(demo, label):
    return next(b for b in demo.button if b.label == label)


def offline_agent():
    return SafetyAgent(load_graph(), model="test", parser=lambda q: {
        "equipment": [{"mention": "지게차", "type": "지게차"}], "situations": []},
        explainer=fake_reply, relevance=lambda q, ids: {})


def test_load_does_not_ask(demo, monkeypatch):
    forbid_ask(monkeypatch)
    demo.run()
    assert not demo.exception
    assert demo.session_state["chat"] == []
    assert demo.session_state["lookup"] == []
    assert [tab.label for tab in demo.tabs] == ["질문하기", "목록으로 조회 (AI 미사용)"]
    assert len(demo.tabs[0].chat_input) == 1 and not demo.tabs[1].chat_input
    assert not demo.tabs[0].selectbox and len(demo.tabs[1].selectbox) == 1
    assert demo.title[0].value == "제조 현장 위험성평가 Agent 데모"
    assert demo.chat_input[0].placeholder == "현장 작업을 설명하고 위험요인을 물어보세요"
    assert all("제가 할 수 있는 일" not in m.value for m in demo.chat_message[0].markdown)


def test_lookup_without_cache_or_key(demo, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("조건 조회는 캐시·LLM을 사용하면 안 됩니다.")
    monkeypatch.setattr(vector_rag, "load_case_vectors", forbidden)
    monkeypatch.setattr(llm, "post", forbidden)
    graph = load_graph()
    situation = next(graph.cases[cid]["situation"] for cid in graph.by["equipment_type"]["지게차"]
                     if graph.cases[cid]["situation"])
    demo.run()
    demo.selectbox[0].select("지게차")
    demo.multiselect[0].select(situation)
    button(demo, "조회").click().run()
    assert not demo.exception
    assert demo.session_state["chat"] == []
    exchange = demo.session_state["lookup"][0]
    out = exchange["result"]
    assert exchange["kind"] == "lookup"
    assert exchange["question"] == f"[목록 조회] 설비: 지게차 / 작업: {situation}"
    assert "text" not in out and "case_ids" not in out
    assert out["evidence"]["factors"] and out["metrics"]["llm_calls"] == 0
    assert not any("토큰" in c.value or "검증" in c.value for c in demo.caption)  # 지표 줄은 숨김 (사용자 피드백)
    assert any("에서 위험요인" in i.value for i in demo.info)  # AI 요약 대신 건수 한 줄
    assert any(out["evidence"]["detailed"][0]["factor"] in m.value for m in demo.markdown)


@pytest.mark.parametrize("key_only", [False, True])
def test_live_off_notice(demo, monkeypatch, key_only):
    if key_only:
        monkeypatch.setattr(settings, "env", lambda name, default=None: "fake" if name == "OPENAI_API_KEY" else default)
    forbid_ask(monkeypatch)
    demo.run().chat_input[0].set_value("로봇 셀 점검 위험을 알려주세요").run()
    assert not demo.exception
    assert demo.session_state["chat"][0]["kind"] == "notice"
    assert any(app.escape_markdown(app.NOTICE) == info.value for info in demo.info)
    assert not demo.sidebar.text and not demo.sidebar.expander
    assert len(demo.chat_message) == 3  # 인사 말풍선 + 질문·답


def test_lookup_requires_selection(demo, monkeypatch):
    forbid_ask(monkeypatch)
    demo.run()
    button(demo, "조회").click().run()
    assert not demo.exception and not demo.session_state["chat"]
    assert not demo.session_state["lookup"]
    assert any("하나 이상" in info.value for info in demo.info)


def test_missing_records(demo):
    demo.run()
    assert not demo.exception
    assert not demo.sidebar.button


def test_replay_offline(demo, records, monkeypatch, tmp_path):
    forbid_ask(monkeypatch)
    record = records["records"][0]
    factors = record["graph_factors"]
    demo.run()
    demo.session_state["main_tab"] = "목록으로 조회 (AI 미사용)"
    demo.run()
    button(demo.sidebar, record["question"]).click().run()
    assert not demo.exception
    assert demo.session_state["chat"] == [dict(question=record["question"], kind="replay", record_id=record["id"])]
    assert len(demo.chat_message) == 3  # 인사 말풍선 + 질문·답
    assert demo.session_state["main_tab"] == "질문하기"
    assert len(demo.tabs[0].chat_message) == 3 and not demo.tabs[1].chat_message
    # 사용자 피드백: 재생 배너·지표·정답 비교는 숨기고 AI 요약만 파란 말풍선에
    assert not any("기록 재생" in i.value for i in demo.info)
    assert any(app.escape_markdown(record["reply"]["summary"]) == i.value for i in demo.info)
    assert not any("토큰" in c.value or "검증" in c.value for c in demo.chat_message[-1].caption)
    assert not any(e.label.startswith("평가 정답과 비교") for e in demo.expander)
    assert not any(t.value.startswith("조회 조건") for t in demo.text)
    assert any(e.label == f"추가 점검 체크리스트 ({len(factors) - 5}개)" for e in demo.expander)
    # 이력에는 ID만 남아 파일 변경도 반영한다
    record["reply"]["summary"] = "다시 읽은 기록"
    (tmp_path / "records.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    demo.run()
    assert any(i.value == "다시 읽은 기록" for i in demo.info)


def test_live_once(demo, monkeypatch):
    enable_live(monkeypatch)
    original = offline_agent().ask
    calls = []
    def ask(self, question):
        calls.append(question)
        return original(question)
    monkeypatch.setattr(SafetyAgent, "ask", ask)
    demo.run().chat_input[0].set_value("지게차 작업").run()
    assert not demo.exception and calls == ["지게차 작업"]
    out = demo.session_state["chat"][0]["result"]
    assert "text" not in out and "case_ids" not in out
    assert any("작업 전 확인" == t.value for t in demo.text)
    assert demo.status[0].label == "완료"
    demo.run()
    assert not demo.exception and calls == ["지게차 작업"]


@pytest.mark.parametrize("error", [llm.LLMError, Exception])
def test_live_exception_then_recover(demo, monkeypatch, error):
    enable_live(monkeypatch)
    original = offline_agent().ask
    calls = []
    def ask(self, question):
        calls.append(question)
        if len(calls) == 1:
            raise error("질문 처리 오류")
        return original(question)
    monkeypatch.setattr(SafetyAgent, "ask", ask)
    demo.run().chat_input[0].set_value("첫 질문").run()
    assert not demo.exception and len(calls) == 1
    assert any("질문 처리 오류" in e.value for e in demo.error)
    assert demo.status[0].label == "실패"
    demo.run()
    assert len(calls) == 1
    demo.chat_input[0].set_value("지게차 작업").run()
    assert not demo.exception and len(calls) == 2
    assert any("작업 전 확인" == t.value for t in demo.text)


def test_empty_result_and_parse_error(demo, monkeypatch):
    enable_live(monkeypatch)
    def broken(_):
        raise llm.LLMError("질문 해석 오류")
    offline = SafetyAgent(load_graph(), model="test", parser=broken, relevance=lambda q, ids: None)
    monkeypatch.setattr(SafetyAgent, "ask", offline.ask)
    demo.run().chat_input[0].set_value("위험을 알려주세요").run()
    assert not demo.exception
    assert any("질문을 처리하지 못했습니다" in info.value for info in demo.info)
    assert not any("찾지 못했습니다" in info.value for info in demo.info)


def test_history_limit_and_clear(demo, records, monkeypatch):
    forbid_ask(monkeypatch)
    demo.run()
    for _ in range(7):
        button(demo.sidebar, records["records"][0]["question"]).click().run()
    assert not demo.exception
    assert len(demo.session_state["chat"]) == 5
    assert len(demo.chat_message) == 11  # 인사 + 5개 대화
    assert not any(b.label == "대화 지우기" for b in demo.sidebar.button)
    demo.session_state["carry"] = {"targets": [], "situations": []}
    demo.session_state["guidance_turns"] = 1
    button(demo.tabs[0], "대화 지우기").click().run()
    assert not demo.session_state["chat"] and len(demo.chat_message) == 1  # 인사만 남음
    assert demo.session_state["carry"] is None and demo.session_state["guidance_turns"] == 0
    assert not any(b.label == "대화 지우기" for b in demo.button)


def test_escape_markdown():
    assert app.escape_markdown("[x](http://e)") == r"\[x\]\(http://e\)"
    assert app.escape_markdown("**b**") == r"\*\*b\*\*"


def test_greeting_and_guides(demo, monkeypatch):
    # 첫 화면 인사: 짧은 소개·예시·사용 안내와 라이브 상태
    forbid_ask(monkeypatch)
    demo.run()
    first = demo.chat_message[0]
    assert first.name == "assistant"
    text = " ".join(m.value for m in first.markdown)
    assert "제가 할 수 있는 일" not in text and "제가 못 하는 일" not in text and "왼쪽 예시 질문" in text
    assert "목록으로 바로 조회" not in text
    assert "**목록으로 조회** 탭을 이용하세요" in text
    assert not first.button and "**예시 질문**" not in text
    assert "왼쪽 위 › 버튼" in text
    assert {e.label for e in first.expander} >= {"질문 잘 쓰는 법", "답 읽는 법"}
    on = {"OPENAI_API_KEY": "fake", "DEMO_LIVE": "1"}
    monkeypatch.setattr(settings, "env", lambda name, default=None: on.get(name, default))
    demo.run()
    assert any("직접 질문하면 API 비용" in m.value for m in demo.chat_message[0].markdown)


def lookup_forklift(demo):
    demo.run()
    demo.selectbox[0].select("지게차")
    button(demo, "조회").click().run()
    assert not demo.exception
    return demo.session_state["lookup"][-1]["result"]


@pytest.mark.parametrize("live_on", [False, True])
def test_lookup_guidance_and_option_without_llm(demo, monkeypatch, live_on):
    def forbidden(*args, **kwargs):
        raise AssertionError("옵션 조회는 LLM을 부르면 안 됩니다")
    monkeypatch.setattr(llm, "post", forbidden)
    if live_on:
        enable_live(monkeypatch)
    demo.run()
    carry = {"targets": [], "situations": ["비정형 작업(점검)"]}
    demo.session_state["carry"] = carry
    demo.session_state["guidance_turns"] = 1
    out = lookup_forklift(demo)
    hint = out["guidance"]
    assert hint["type"] == "need_situation" and demo.session_state["carry"] == carry
    # AI가 꺼져 있으면 자유 입력 이어받기 표시 대신 버튼만 안내한다
    assert not any("앞 조건" in c.value for c in demo.tabs[1].caption)
    assert any("아래에서 골라 주세요" in c.value for c in demo.caption)
    option = hint["options"][0]
    label = app.escape_markdown(f"{option['name']} (사고 {option['cases']}건)")
    button(demo, label).click().run()
    assert not demo.exception and len(demo.session_state["lookup"]) == 2
    assert demo.session_state["chat"] == []
    exchange = demo.session_state["lookup"][-1]
    assert exchange["kind"] == "option" and exchange["question"] == f"[선택] {option['name']}"
    assert exchange["result"]["conditions"]["situations"] == [option["name"]]
    assert exchange["result"]["conditions"]["targets"] == out["conditions"]["targets"]
    assert exchange["result"]["metrics"]["llm_calls"] == 0 and demo.session_state["carry"] == carry
    assert demo.session_state["guidance_turns"] == 1
    assert not any(b.key and b.key.startswith("lookup_option_") for b in demo.button)
    assert not any(app.escape_markdown(hint["message"]) == info.value for info in demo.tabs[1].info)


def test_start_new_question_clears_carry(demo, monkeypatch):
    enable_live(monkeypatch)  # 이어받기 표시와 '새 질문으로 시작'은 자유 질문이 켜졌을 때만 나온다
    original = offline_agent().ask
    monkeypatch.setattr(SafetyAgent, "ask", lambda self, question: original(question, explain=False))
    demo.run().chat_input[0].set_value("지게차 작업").run()
    assert any("앞 조건" in c.value for c in demo.caption)
    button(demo, "새 질문으로 시작").click().run()
    assert not demo.exception and demo.session_state["carry"] is None
    assert not any("앞 조건" in c.value for c in demo.caption)


def test_lookup_history_limit_and_clear(demo, records, monkeypatch):
    monkeypatch.setattr(llm, "post", lambda *args, **kwargs: pytest.fail("LLM 호출 금지"))
    lookup_forklift(demo)
    button(demo.sidebar, records["records"][0]["question"]).click().run()
    chat = demo.session_state["chat"]
    carry = {"targets": [], "situations": ["비정형 작업(점검)"]}
    demo.session_state["carry"] = carry
    demo.session_state["guidance_turns"] = 1
    for equipment in demo.selectbox[0].options[1:7]:
        demo.selectbox[0].select(equipment)
        button(demo, "조회").click().run()
    assert not demo.exception
    history = demo.session_state["lookup"]
    assert len(history) == 5
    headings = [m.value for m in demo.tabs[1].markdown if m.value.startswith(r"\[목록 조회\]")]
    assert headings == [app.escape_markdown(e["question"]) for e in history]
    button(demo, "조회 기록 지우기").click().run()
    assert not demo.exception and demo.session_state["lookup"] == []
    assert demo.session_state["chat"] == chat
    assert demo.session_state["carry"] == carry and demo.session_state["guidance_turns"] == 1


def test_option_keys_and_chat_destination(demo, monkeypatch):
    enable_live(monkeypatch)
    original = offline_agent().ask
    monkeypatch.setattr(SafetyAgent, "ask", lambda self, *args, **kwargs: original(*args, **kwargs))
    monkeypatch.setattr(llm, "post", lambda *args, **kwargs: pytest.fail("LLM 호출 금지"))
    demo.run().chat_input[0].set_value("지게차 작업").run()
    lookup_forklift(demo)
    assert any(b.key == "chat_option_0" for b in demo.tabs[0].button)
    assert any(b.key == "lookup_option_0" for b in demo.tabs[1].button)
    lookup_history = demo.session_state["lookup"]
    demo.button(key="chat_option_0").click().run()
    assert not demo.exception and len(demo.session_state["chat"]) == 2
    assert demo.session_state["chat"][-1]["kind"] == "option"
    assert demo.session_state["lookup"] == lookup_history


def test_live_need_context_does_not_carry(demo, monkeypatch):
    enable_live(monkeypatch)
    original = SafetyAgent(load_graph(), model="test", parser=lambda q: {}).ask
    monkeypatch.setattr(SafetyAgent, "ask", lambda self, question: original(question, explain=False))
    demo.run().chat_input[0].set_value("무엇을 할까요").run()
    assert not demo.exception and demo.session_state["carry"] is None
    hint = demo.session_state["chat"][-1]["result"]["guidance"]
    assert hint["type"] == "need_context"
    assert any(app.escape_markdown(hint["message"]) == info.value for info in demo.info)


def test_live_passes_carry_without_question_concatenation(demo, monkeypatch):
    enable_live(monkeypatch)
    calls = []
    original = SafetyAgent(load_graph(), model="test", parser=lambda q: {
        "situations": ["지게차 사용 작업"]}, relevance=lambda q, ids: None).ask
    def ask(self, question, carry=None):
        calls.append((question, carry))
        return original(question, carry=carry, explain=False)
    monkeypatch.setattr(SafetyAgent, "ask", ask)
    carry = {"targets": [{"kind": "equipment_type", "name": "지게차"}], "situations": []}
    demo.run()
    demo.session_state["carry"] = carry
    demo.session_state["guidance_turns"] = 1
    demo.chat_input[0].set_value("운반 작업").run()
    assert not demo.exception and calls == [("운반 작업", carry)]
    assert demo.session_state["chat"][-1]["result"]["guidance"] is None
    assert demo.session_state["carry"] is None
    assert any(c.value == "앞 조건과 합쳐서 이해했어요" for c in demo.caption)


def test_two_guidance_turns_clear_carry(demo, monkeypatch):
    enable_live(monkeypatch)
    original = offline_agent().ask
    calls = []
    def ask(self, question, carry=None):
        calls.append(carry)
        return original(question, carry=carry, explain=False)
    monkeypatch.setattr(SafetyAgent, "ask", ask)
    demo.run().chat_input[0].set_value("지게차 작업").run()
    assert demo.session_state["carry"] and demo.session_state["guidance_turns"] == 1
    demo.chat_input[0].set_value("더 알려줘").run()
    assert not demo.exception and calls[1]
    assert demo.session_state["carry"] is None and demo.session_state["guidance_turns"] == 0
    assert len([b for b in demo.button if b.key and b.key.startswith("chat_option_")]) <= 3


@pytest.mark.parametrize("action", ["notice", "clear", "example"])
def test_other_actions_clear_carry(demo, records, action):
    lookup_forklift(demo)
    button(demo.sidebar, records["records"][0]["question"]).click().run()
    demo.session_state["carry"] = demo.session_state["lookup"][-1]["result"]["conditions"]
    demo.session_state["guidance_turns"] = 1
    if action == "notice":
        demo.chat_input[0].set_value("추가 질문").run()
    elif action == "clear":
        button(demo, "대화 지우기").click().run()
    else:
        button(demo.sidebar, records["records"][0]["question"]).click().run()
    assert not demo.exception and demo.session_state["carry"] is None


def test_situation_option_selects_equipment(demo, monkeypatch):
    monkeypatch.setattr(llm, "post", lambda *args, **kwargs: pytest.fail("LLM 호출 금지"))
    demo.run()
    demo.multiselect[0].select("비정형 작업(점검)")
    button(demo, "조회").click().run()
    out = demo.session_state["lookup"][-1]["result"]
    assert out["guidance"]["type"] == "need_equipment"
    option = out["guidance"]["options"][0]
    button(demo, app.escape_markdown(f"{option['name']} (사고 {option['cases']}건)")).click().run()
    assert not demo.exception
    narrowed = demo.session_state["lookup"][-1]["result"]
    assert narrowed["conditions"]["targets"] == [{"kind": "equipment_type", "name": option["name"],
                                                    "mention": option["name"], "type": option["name"]}]
    assert narrowed["conditions"]["situations"] == ["비정형 작업(점검)"]
    assert narrowed["guidance"] is None and demo.session_state["carry"] is None


def test_live_exception_clears_existing_carry(demo, monkeypatch):
    enable_live(monkeypatch)
    lookup_forklift(demo)
    demo.session_state["carry"] = demo.session_state["lookup"][-1]["result"]["conditions"]
    demo.session_state["guidance_turns"] = 1
    def broken(*args, **kwargs):
        raise llm.LLMError("추가 질문 처리 실패")
    monkeypatch.setattr(SafetyAgent, "ask", broken)
    demo.chat_input[0].set_value("추가 설명").run()
    assert not demo.exception and demo.session_state["carry"] is None
    assert any("추가 질문 처리 실패" in e.value for e in demo.error)


def test_sidebar_examples_only(demo, records, tmp_path, monkeypatch):
    forbid_ask(monkeypatch)
    first = records["records"][0]
    second = dict(first, id="U02", group="정비·청소", order=3, question="컨베이어 청소 질문")
    third = dict(first, id="U03", order=2, question="크레인 운반 질문")
    hidden = dict(first, id="hidden", question="숨긴 질문")
    hidden.pop("group")
    records["records"] = [second, hidden, third, first]
    (tmp_path / "records.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    demo.run()
    assert not demo.exception
    assert [h.value for h in demo.sidebar.header] == ["예시 질문"]
    # 카테고리마다 접고 펴는 토글, 이름 옆에 질문 수
    assert [e.label for e in demo.sidebar.expander] == ["물류·운반 (2)", "정비·청소 (1)"]
    assert [b.label for b in demo.sidebar.button] == [first["question"], third["question"], second["question"]]
    assert [c.value for c in demo.sidebar.caption] == [
        "실제 현장에서 나올 법한 질문입니다. 누르면 실제로 실행했던 AI 답변을 보여 드려요 (비용 0)."]
    assert not demo.sidebar.subheader and not demo.sidebar.text and not demo.sidebar.info
    assert not demo.chat_message[0].button
    assert not any(e.label in ("평가 요약", "작동 구조", "한계") for e in demo.expander)


def test_replay_hides_check_sections(demo, records, tmp_path, monkeypatch):
    # 사용자 피드백: '해석 확인'·'평가 정답과 비교' 영역은 채팅에 보이지 않는다
    forbid_ask(monkeypatch)
    record = records["records"][0]
    record["expected"] = {"equipment": ["생산설비(프레스)"], "situations": [], "guidance": None}
    (tmp_path / "records.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    demo.run()
    button(demo.sidebar, record["question"]).click().run()
    assert not demo.exception
    assert not any(e.label.startswith(("해석 확인", "평가 정답과 비교")) for e in demo.expander)


def test_replay_without_evaluation(demo, records, tmp_path, monkeypatch):
    forbid_ask(monkeypatch)
    record = records["records"][0]
    record.pop("gold_factors")
    (tmp_path / "records.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    demo.run()
    button(demo.sidebar, record["question"]).click().run()
    assert not demo.exception
    assert not any(e.label.startswith(("평가 정답과 비교", "해석 확인")) for e in demo.expander)


def test_meta_only_merge(tmp_path, monkeypatch):
    path = tmp_path / "data/demo/records.json"
    path.parent.mkdir(parents=True)
    original = {"generated_at": "old", "model": "old-model", "commit": "old-commit", "records": [
        {"id": "Q4-S14", "question": "삭제할 기록"},
        {"id": "Q4-S11", "question": "리프트 기록", "group": "old", "order": 99, "note": "old"},
        {"id": "Q4-D01", "question": "지게차 기록", "reply": {"summary": "저장된 답변"}},
    ]}
    path.write_text(json.dumps(original, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(make_demo_records, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["make_demo_records.py", "--meta-only"])
    def forbidden(*args, **kwargs):
        pytest.fail("메타데이터 갱신은 API·그래프·벡터 캐시를 사용하면 안 됩니다")
    for name in ("SafetyAgent", "load_graph", "cached_vectors", "commit_id"):
        monkeypatch.setattr(make_demo_records, name, forbidden)
    monkeypatch.setattr(llm, "post", forbidden)
    make_demo_records.main()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert {k: v for k, v in data.items() if k != "records"} == {k: v for k, v in original.items() if k != "records"}
    assert [r["id"] for r in data["records"]] == ["Q4-D01", "Q4-S11"]
    for record in data["records"]:
        qid, group, order = next(e for e in make_demo_records.EXAMPLES if e[0] == record["id"])
        assert (record["group"], record["order"], record["note"]) == (group, order, make_demo_records.NOTES[qid])
        previous = next(r for r in original["records"] if r["id"] == qid)
        assert {k: v for k, v in record.items() if k not in ("group", "order", "note")} == {
            k: v for k, v in previous.items() if k not in ("group", "order", "note")}
    make_demo_records.main()
    assert json.loads(path.read_text(encoding="utf-8")) == data


@pytest.mark.parametrize("only", ["U03,U11", "Q4-D01", None])
def test_record_selection_and_merge(tmp_path, monkeypatch, records, only):
    path = tmp_path / "data/demo/records.json"
    original = deepcopy(records["records"][0])
    make_demo_records.merge_records(path, [original, {"id": "Q4-S14"}], model="test", generated_at="old")
    # 평가 입력은 저장소에서 읽고 출력만 임시 파일로 보낸다
    real_merge = make_demo_records.merge_records
    monkeypatch.setattr(make_demo_records, "merge_records", lambda _, rows, **meta: real_merge(path, rows, **meta))
    monkeypatch.setattr(sys, "argv", ["make_demo_records.py"] + (["--only", only] if only else []))
    called, cached = [], []
    class FakeAgent:
        def ask(self, question):
            called.append(question)
            out = deepcopy(original)
            for key in ("gold_factors", "vector_top5_factors"):
                out.pop(key)
            return out
    monkeypatch.setattr(make_demo_records, "SafetyAgent", FakeAgent)
    def vectors(graph, questions):
        cached.extend(questions)
        return make_demo_records.np.zeros((len(graph.cases), 1)), {
            qid: make_demo_records.np.zeros(1) for qid in questions}
    monkeypatch.setattr(make_demo_records, "cached_vectors", vectors)
    monkeypatch.setattr(llm, "post", lambda *a, **kw: pytest.fail("API 호출 금지"))
    make_demo_records.main()
    data = json.loads(path.read_text(encoding="utf-8"))
    selected = set(only.split(",")) if only else {e[0] for e in make_demo_records.EXAMPLES}
    assert len(called) == len(selected)
    assert set(cached) == {qid for qid in selected if qid.startswith("Q4-")}
    assert {r["id"] for r in data["records"]} == selected | {"Q4-D01"}
    for record in data["records"]:
        if record["id"] not in selected:
            assert record["reply"] == original["reply"]
        elif record["id"].startswith("U"):
            assert set(record["expected"]) == {"equipment", "situations", "guidance"}
            assert "gold_factors" not in record and "vector_top5_factors" not in record


def test_latest_replay_with_guidance_shows_options(demo, monkeypatch, tmp_path):
    # 안내가 있는 실행 기록(설비만)을 재생하면 선택지 버튼이 나오고, 누르면 AI 없이 좁혀진다
    agent = SafetyAgent(load_graph(), model="test", parser=lambda q: {
        "equipment": [{"mention": "지게차", "type": "지게차"}], "situations": []},
        explainer=fake_reply, relevance=lambda q, ids: {})
    out = agent.ask("지게차 위험요인")
    assert out["guidance"]["type"] == "need_situation"
    out.pop("text")
    out.update(id="U13", question="지게차 위험요인", note="안내 확인", group="자동화·리프트", order=1,
               graph_factors=[f["factor"] for f in out["evidence"]["factors"]],
               expected={"equipment": ["지게차"], "situations": [], "guidance": "need_situation"})
    data = {"generated_at": "2026-11-03T12:00:00+09:00", "model": "test", "records": [out]}
    (tmp_path / "records.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("SAFETY_DEMO_RECORDS", str(tmp_path / "records.json"))
    def forbidden(*args, **kwargs):
        raise AssertionError("선택지는 LLM을 부르면 안 됩니다")
    monkeypatch.setattr(llm, "post", forbidden)
    demo.run()
    button(demo, "지게차 위험요인").click().run()
    option = out["guidance"]["options"][0]
    label = app.escape_markdown(f"{option['name']} (사고 {option['cases']}건)")
    button(demo, label).click().run()
    assert not demo.exception
    last = demo.session_state["chat"][-1]
    assert last["question"] == f"[선택] {option['name']}" and last["result"]["metrics"]["llm_calls"] == 0
