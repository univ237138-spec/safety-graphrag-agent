"""제조 현장 위험성평가 데모 (메모리 그래프).

    streamlit run app.py
"""
import itertools
import json
import os
import re
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import answer  # noqa: E402
import settings  # noqa: E402
from agent import SafetyAgent  # noqa: E402
from graph import load_graph  # noqa: E402

EMBLEM = str(ROOT / "assets" / "emblem_256.png")  # 탭 아이콘·로고·Agent 말풍선 아이콘
NOTICE = ("공개 데모에서는 AI 질문 해석이 꺼져 있어 자유 질문에 답하지 않습니다. "
          "왼쪽 예시 질문(실제 실행 기록)을 누르거나 '목록으로 조회' 탭을 이용해 주세요.")


def escape_markdown(text):
    """데이터의 Markdown 문자를 이스케이프한다."""
    return re.sub(r"([\\`*_{}\[\]()<>#+\-.!|~=$])", r"\\\1", str(text))


@st.cache_resource
def resources():
    """Neo4j 없이 메모리 그래프와 Agent를 한 번만 만든다."""
    graph = load_graph()
    return graph, SafetyAgent(graph)


# 상세 위험 카드의 두 유형: 사례 수 상위(주황, 사고 다발) / 질문과 관련 깊음(파랑, 내 상황).
# 안전 화면에서 초록은 '안전'으로 읽히고 빨강·초록은 색각 차이로 구분이 어려워 주황·파랑을 쓰고, 아이콘·글자로도 구분한다.
PICKS = {"cases": ("사례 수 상위", ":material/local_fire_department:", "orange", "#E8710A"),
         "question": ("질문과 관련 깊음", ":material/my_location:", "blue", "#1A73E8")}
# 사용자 데이터가 들어가지 않는 고정 CSS: key가 card_<유형>_ 으로 시작하는 카드의 왼쪽 테두리 색
CARD_KEYS = itertools.count()  # 같은 화면에 같은 요인 ID가 여러 번 나와도 카드 key가 겹치지 않게
CARD_CSS = "<style>" + "".join(
    f'[class*="st-key-card_{pick}_"] {{border-left: 6px solid {color} !important;}}'
    for pick, (_, _, _, color) in PICKS.items()) + (
    # 실제 채팅처럼 내 질문은 오른쪽, Agent 답변은 왼쪽 (사용자 아바타가 있는 말풍선만 뒤집는다)
    '[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])'
    ' {flex-direction: row-reverse; margin-left: auto; width: fit-content; max-width: 80%;}'
) + "</style>"


def show_factor(d, ev, items, key):
    pick = "question" if d.get("pick") == "question" else "cases"
    label, icon, color, _ = PICKS[pick]
    with st.container(border=True, key=f"card_{pick}_{key}"):
        st.markdown("**" + escape_markdown(f"{d['id']}. {d['factor']} (사례 {d['cases']}건)") + "**")
        st.badge(label, icon=icon, color=color)
        advice = items.get(d["id"], {}).get("advice")
        if advice:
            st.text(advice)
        for mid in d["measure_ids"]:
            st.markdown("- " + escape_markdown(ev["measures"][mid]["text"]))
        with st.expander("근거 사고 사례"):
            for cid in d["case_ids"]:
                st.text(f"{cid}: {ev['cases'][cid]['overview']}")


def show_result(out, kind, latest=False, live_on=False, target="chat"):
    """요약 한 줄(파란 말풍선) → 상세 위험 → 체크리스트 → 안내 순서로 보여준다 (사용자 피드백: 조건·지표 줄은 숨김)."""
    if out.get("relax_level", 0) > 0 and out.get("relax_note"):
        st.caption(escape_markdown(out["relax_note"]))  # 조건을 넓혀 찾았다는 사실은 알린다
    ev, items = out.get("evidence") or {}, out.get("items") or {}
    cases, factors = ev.get("total_cases", 0), ev.get("total_factors", 0)
    summary = (out.get("reply") or {}).get("summary") if items else None
    if summary:
        st.info(escape_markdown(summary))
    elif ev.get("factors"):
        st.info(f"사망사고 사례 {cases}건에서 위험요인 {factors}개를 찾았습니다.")
    elif out.get("parse_ok") is False:
        st.info("질문을 처리하지 못했습니다. 잠시 후 다시 시도해 주세요.")  # 해석 실패를 '사례 없음'과 구분
    elif not out.get("guidance"):
        st.info("해당하는 사망사고 사례를 찾지 못했습니다.")
    for d in ev.get("detailed", []):
        show_factor(d, ev, items, next(CARD_KEYS))
    detailed_ids = {d["id"] for d in ev.get("detailed", [])}
    rest = [f for f in ev.get("factors", []) if f["id"] not in detailed_ids]
    if rest:
        with st.expander(f"추가 점검 체크리스트 ({len(rest)}개)"):
            for f in rest:
                st.text(f"□ {f['factor']} (사례 {f['cases']}건)")
    if out.get("parse_ok") is False and ev.get("factors"):
        st.caption("질문 해석에 실패해 질문 속 설비 이름만으로 찾았습니다. 작업 내용이 반영되지 않았을 수 있습니다.")
    elif out.get("llm_error") and ev.get("factors"):
        st.caption("AI 설명을 만들지 못해 근거만 보여 드립니다.")
    hint = out.get("guidance")
    if hint and (target == "chat" or latest):
        st.info(escape_markdown(hint["message"]))
        if latest and hint["options"]:
            # AI가 꺼져 있으면 자유 입력에 답하지 않으므로 버튼만 안내한다
            st.caption("이 조건에서 기록이 많은 예시입니다. " + ("직접 입력해도 됩니다." if live_on else "아래에서 골라 주세요."))
            for i, option in enumerate(hint["options"]):
                st.button(escape_markdown(f"{option['name']} (사고 {option['cases']}건)"), key=f"{target}_option_{i}",
                          on_click=choose_option, args=(out["conditions"], option, live_on, target))


def clear_carry():
    """이어받을 조건과 연속 안내 횟수를 함께 지운다."""
    st.session_state["carry"] = None
    st.session_state["guidance_turns"] = 0


def clear_chat():
    st.session_state["chat"] = []
    clear_carry()


def update_carry(out):
    hint = out.get("guidance") or {}
    turns = st.session_state.get("guidance_turns", 0) + 1
    if not out.get("llm_error") and hint.get("type") in ("need_situation", "need_equipment") and turns < 2:
        st.session_state["carry"] = out["conditions"]
        st.session_state["guidance_turns"] = turns
    else:
        clear_carry()


def choose_option(carry, option, live_on, target="chat"):
    """선택지는 질문 해석 없이 확인된 조건에 넣는다."""
    conditions = dict(carry)
    name = option["name"]
    if option["kind"] == "situation":
        conditions["situations"] = [name]
    else:
        conditions["targets"] = [{"kind": "equipment_type", "name": name, "mention": name, "type": name}]
    _, agent = resources()
    try:
        out = agent.ask(None, conditions, explain=bool(live_on) and target == "chat")
        append_exchange(f"[선택] {name}", "option", target=target, result=stored_result(out))
    except Exception as exc:
        append_exchange(f"[선택] {name}", "option", target=target, notice=f"답변을 만들지 못했습니다: {exc}")


def append_exchange(question, kind, target="chat", **payload):
    """최근 다섯 질문만 세션에 남긴다."""
    exchange = dict(question=question, kind=kind, **payload)
    st.session_state[target] = (st.session_state[target] + [exchange])[-5:]
    if target == "chat":
        if "result" in payload:
            update_carry(payload["result"])
        else:
            clear_carry()
    return exchange


def stored_result(out):
    """중복 출력과 최상위 사례 ID 목록은 저장하지 않는다."""
    return {key: value for key, value in out.items() if key not in ("text", "case_ids")}


def lookup(graph, agent):
    # 사이드바에 두면 작업 상황 선택 직후 첫 '조회' 클릭이 무시돼(실제 브라우저 확인) 본문에 둔다
    with st.form("lookup_form"):
        equipment = st.selectbox("설비 유형", ["(선택 안 함)"] + graph.names("equipment_type"))
        situations = st.multiselect("작업 상황", graph.names("situation"), max_selections=10,
                                    placeholder="작업 상황을 고르세요 (여러 개 가능)")
        submitted = st.form_submit_button("조회")
    if submitted:
        if equipment == "(선택 안 함)" and not situations:
            st.info("설비 유형 또는 작업 상황을 하나 이상 선택해 주세요.")
        else:
            targets = ([] if equipment == "(선택 안 함)" else
                       [{"kind": "equipment_type", "name": equipment, "mention": equipment, "type": equipment}])
            conditions = {"targets": targets, "situations": situations}
            question = f"[목록 조회] 설비: {equipment} / 작업: {', '.join(situations) or '-'}"
            append_exchange(question, "lookup", target="lookup",
                            result=stored_result(agent.ask(None, conditions, explain=False)))


def read_records():
    """테스트 전용 경로도 지원하며 렌더링할 때 기록을 다시 읽는다."""
    path = Path(os.environ.get("SAFETY_DEMO_RECORDS", str(ROOT / "data/demo/records.json")))
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"records": []}


def examples():
    records = sorted((r for r in read_records()["records"] if r.get("group")),
                     key=lambda r: r["order"])
    # 질문이 많아도 잘리지 않게 카테고리별로 접어 두고, 펼친 상태는 다시 그려도 유지한다
    for group in dict.fromkeys(r["group"] for r in records):
        items = [r for r in records if r["group"] == group]
        with st.expander(f"{group} ({len(items)})", expanded=False):
            for record in items:
                if st.button(record["question"], key="example_" + record["id"], use_container_width=True):
                    append_exchange(record["question"], "replay", record_id=record["id"])
                    st.session_state["main_tab"] = "질문하기"


GREETING = ("안녕하세요. 제조·물류 현장의 작업 전 위험성평가를 돕는 Agent입니다.  \n"
            "하려는 작업을 설명해 주시면 사망사고 기록 1,693건에서 관련 위험요인을 찾아 드립니다.")

WRITE_GUIDE = """1. **설비 이름**을 구체적으로: 지게차, 산업용로봇, 컨베이어, 리프트, 크레인 …
2. **하려는 작업**: 점검, 청소, 정비·수리, 운반, 상하차, 설치 …
3. **작업 상황**: 야간·단독 작업, 인원, 설비 가동·전원 차단 여부, 작업자와 차량이 섞이는지

- 좋은 예: "야간에 2명이 지게차로 도크에서 파렛트를 옮겨요. 작업자와 차량이 자주 섞이는데 무엇을 먼저 조심해야 하나요?"
- 아쉬운 예: "공장에서 위험한 거 알려줘" (설비·작업이 없어 관련 사고를 찾기 어렵습니다)"""

READ_GUIDE = """1. 답에 나온 위험이 내 작업의 설비·작업에 대한 것인지 먼저 보세요. 다르면 작업 내용을 더 구체적으로 다시 적거나(자유 질문이 켜져 있을 때), '목록으로 조회' 탭에서 설비·작업을 직접 고르세요.
2. 상세 위험의 표시: **사례 수 상위**는 사망사고가 많았던 순서로 고른 위험, **질문과 관련 깊음**은 나머지 중 질문 내용과 비슷한 사고가 있는 위험입니다 (질문이 있고 위험이 6개 이상일 때 2개까지). 둘 다 실제 작업과 맞는지 확인하세요.
3. **추가 점검 체크리스트**까지 확인하세요. 사례가 1~2건뿐인 위험도 실제로 사망사고가 난 위험입니다.
4. 작업 전 회의(TBM)에서 대책과 사고 사례를 함께 읽어 보세요. 최종 판단은 현장 책임자가 합니다."""


def greeting(live_on):
    """짧은 소개·예시·사용법을 보여준다 (대화 기록에는 넣지 않음)."""
    with st.chat_message("assistant", avatar=EMBLEM):
        st.markdown(GREETING)
        st.markdown("아래 입력창에 작업을 설명하거나 왼쪽 예시 질문을 눌러 보세요. 직접 질문하면 API 비용이 듭니다." if live_on else
                    "**왼쪽 예시 질문**을 눌러 데모를 써 보세요. 실제로 실행했던 AI 답변을 다시 보여 드려요 (비용 0). "
                    "화면이 좁으면 왼쪽 위 › 버튼을 누르면 예시가 보입니다. 설비·작업을 직접 고르려면 **목록으로 조회** 탭을 이용하세요.")
        with st.expander("질문 잘 쓰는 법"):
            st.markdown(WRITE_GUIDE)
        with st.expander("답 읽는 법"):
            st.markdown(READ_GUIDE)


def replay(record_id, latest=False, live_on=False):
    data = read_records()
    record = next((r for r in data["records"] if r["id"] == record_id), None)
    if record is None:
        st.info("실행 기록이 아직 생성되지 않았습니다.")
        return
    show_result(record, "replay", latest, live_on)  # 최신 재생이면 선택지 버튼도 (조건 확정이라 AI 호출 없음)


def show_exchange(exchange, latest=False, live_on=False):
    """대화 이력을 그릴 때는 Agent를 호출하지 않는다."""
    with st.chat_message("user"):
        st.text(exchange["question"])
    with st.chat_message("assistant", avatar=EMBLEM):
        if exchange["kind"] == "replay":
            replay(exchange["record_id"], latest, live_on)
        elif exchange["kind"] == "notice":
            st.info(escape_markdown(exchange["notice"]))
        elif "notice" in exchange:
            st.error(escape_markdown(exchange["notice"]))
        else:
            if exchange.get("carry_used"):
                st.caption("앞 조건과 합쳐서 이해했어요")
            show_result(exchange["result"], exchange["kind"], latest, live_on)


def live(question, agent):
    """이번 입력에만 답하고 실패해도 다음 질문을 받는다."""
    with st.chat_message("user"):
        st.text(question)
    with st.chat_message("assistant", avatar=EMBLEM):
        with st.status("답변을 만드는 중…") as status:
            try:
                carry = st.session_state.get("carry")
                out = agent.ask(question, carry=carry) if carry else agent.ask(question)
            except Exception as exc:
                status.update(label="실패", state="error")
                notice = f"답변을 만들지 못했습니다: {exc}"
                append_exchange(question, "live", notice=notice)
                st.error(escape_markdown(notice))
                return
            status.update(label="완료", state="complete")
        exchange = append_exchange(question, "live", result=stored_result(out), carry_used=bool(carry))
        if carry:
            st.caption("앞 조건과 합쳐서 이해했어요")
        show_result(exchange["result"], "live", latest=True, live_on=True)


def main():
    st.set_page_config(page_title="제조 현장 위험성평가 Agent 데모", page_icon=EMBLEM, initial_sidebar_state="auto")
    st.logo(EMBLEM, size="large")  # 왼쪽 위 엠블럼
    st.title("제조 현장 위험성평가 Agent 데모")
    st.html(CARD_CSS)  # 고정 CSS만 (데이터 없음)
    st.caption(answer.DISCLAIMER)
    st.session_state.setdefault("chat", [])
    st.session_state.setdefault("lookup", [])
    st.session_state.setdefault("carry", None)
    st.session_state.setdefault("guidance_turns", 0)
    graph, agent = resources()
    # 키가 있어도 DEMO_LIVE=1을 따로 켜야 유료 모드가 열린다
    live_on = settings.env("DEMO_LIVE") == "1" and (settings.env("OPENAI_API_KEY") or "").strip()
    with st.sidebar:
        st.header("예시 질문")
        st.caption("실제 현장에서 나올 법한 질문입니다. 누르면 실제로 실행했던 AI 답변을 보여 드려요 (비용 0).")
        examples()
    chat_tab, lookup_tab = st.tabs(["질문하기", "목록으로 조회 (AI 미사용)"], key="main_tab", on_change="rerun")
    with chat_tab:
        greeting(live_on)
        history_container = st.container()
        question = st.chat_input("현장 작업을 설명하고 위험요인을 물어보세요", max_chars=500)
        if question and not live_on:
            append_exchange(question, "notice", notice=NOTICE)
        # 새 라이브 질문도 포함해 화면에는 최근 다섯 질문만 남긴다
        history = st.session_state["chat"][-4:] if question and live_on else st.session_state["chat"]
        with history_container:
            for i, exchange in enumerate(history):
                show_exchange(exchange, latest=i == len(history) - 1 and not (question and live_on), live_on=live_on)
        if question and live_on:
            live(question, agent)
        if st.session_state["carry"] and live_on:  # 이어받기는 자유 질문이 켜졌을 때만 의미가 있다
            st.caption(escape_markdown("앞 조건(" + answer.condition_text(st.session_state["carry"]) + ")을 이어서 이해합니다"))
            st.button("새 질문으로 시작", on_click=clear_carry)
        if st.session_state["chat"]:
            st.button("대화 지우기", on_click=clear_chat)
    with lookup_tab:
        st.caption("설비 유형·작업 상황을 골라 그래프에서 바로 찾습니다 · AI 호출 0회")
        lookup(graph, agent)
        history = st.session_state["lookup"][-5:]
        for i, exchange in enumerate(history):
            st.markdown(escape_markdown(exchange["question"]))
            if "notice" in exchange:
                st.error(escape_markdown(exchange["notice"]))
            else:
                show_result(exchange["result"], exchange["kind"], latest=i == len(history) - 1, target="lookup")
        st.button("조회 기록 지우기", on_click=lambda: st.session_state.update(lookup=[]))


if __name__ == "__main__":
    main()
