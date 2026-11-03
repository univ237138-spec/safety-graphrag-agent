"""FastAPI POST /assess 테스트 (API 키·외부 네트워크 없이. Neo4j 일치 검사는 로컬 Neo4j가 켜져 있을 때만)."""
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import api  # noqa: E402
import llm  # noqa: E402
from agent import find_cases  # noqa: E402

REAL_OPEN_STORE = api.open_store  # client 픽스처가 바꾸기 전 원래 함수
FORKLIFT = {"targets": [{"kind": "equipment_type", "name": "지게차"}], "situations": ["혼재작업(작업자-작업차량)"]}


@pytest.fixture(scope="module")
def client():
    mp = pytest.MonkeyPatch()
    mp.setattr(api, "open_store", lambda: None)  # 메모리 그래프로 (테스트는 Neo4j·키 없이)
    with TestClient(api.app) as c:
        yield c
    mp.undo()


def fake_post(url, payload, timeout=120):
    """설명 요청에 규칙을 지킨 답을 주는 가짜 OpenAI (토큰 100/20)."""
    data = json.loads(payload["input"][0]["content"])
    items = [{"factor_id": d["id"], "advice": "작업 전 확인", "evidence_ids": list(d["measures"])[:1]} for d in data["detailed"]]
    text = json.dumps({"summary": f"요인 {data['total_factors']}개", "items": items}, ensure_ascii=False)
    response = {"status": "completed", "usage": {"input_tokens": 100, "output_tokens": 20},
                "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}
    llm.record_usage(payload["model"], response)
    return response


def test_health(client):
    assert client.get("/health").json() == {"backend": "memory", "cases": 1693}


def test_graph_only_request(client):
    body = client.post("/assess", json={"conditions": FORKLIFT, "explain": False}).json()
    assert body["factors"] and len(body["detailed"]) == min(5, len(body["factors"]))
    assert sum(f["cases"] for f in body["factors"]) == body["total_cases"]
    assert body["metrics"]["llm_calls"] == 0 and body["summary"] is None
    assert body["rank_note"] is None and all(d["pick"] == "cases" for d in body["detailed"])


def test_explain_with_metrics(client, monkeypatch):
    monkeypatch.setattr(llm, "post", fake_post)
    body = client.post("/assess", json={"conditions": FORKLIFT}).json()
    assert body["llm_error"] is None and all(d["advice"] for d in body["detailed"])
    assert {k: body["metrics"][k] for k in ("llm_calls", "input_tokens", "output_tokens")} == \
        {"llm_calls": 1, "input_tokens": 100, "output_tokens": 20}


def test_question_slots_in_response(client, monkeypatch):
    agent = api.app.state.agent
    given = {"targets": FORKLIFT["targets"], "situations": []}
    count = agent.graph.factors(find_cases(agent.graph, given))
    low = count[-1][0]
    monkeypatch.setattr(agent, "relevance", lambda q, ids: {f: float(f == low) for f, _ in count})
    monkeypatch.setattr(llm, "post", fake_post)
    body = client.post("/assess", json={"question": "지게차 작업", "conditions": given}).json()
    assert body["rank_note"] is None and body["llm_error"] is None
    assert body["detailed"][3]["factor"] == low and body["detailed"][3]["pick"] == "question"
    assert all(d["advice"] for d in body["detailed"])


def test_explain_failure_is_200_with_error(client, monkeypatch):
    def broken(url, payload, timeout=120):
        raise llm.LLMError("API 요청 실패 (HTTP 429).")
    monkeypatch.setattr(llm, "post", broken)
    response = client.post("/assess", json={"conditions": FORKLIFT})
    assert response.status_code == 200 and "429" in response.json()["llm_error"] and response.json()["factors"]


@pytest.mark.parametrize("payload", [
    {}, {"question": "   "}, {"question": "가" * 501},
    {"conditions": {"targets": [{"kind": "equipment", "name": "없는설비"}]}},
    {"conditions": {"targets": [{"kind": "설비", "name": "지게차"}]}},
    {"conditions": {"situations": ["없는 작업"]}},
    {"conditions": {"situations": "지게차 사용 작업"}},
])
def test_bad_input_is_422(client, payload):
    assert client.post("/assess", json=payload).status_code == 422


def test_question_without_key_is_502(client, monkeypatch):
    monkeypatch.setattr(llm, "env", lambda name, default=None: default)  # .env에 키가 없는 상황
    response = client.post("/assess", json={"question": "내일 작업 위험요인 알려줘"})
    assert response.status_code == 502 and "OPENAI_API_KEY" in response.json()["detail"]


def test_question_without_key_but_named_equipment_still_answers(client, monkeypatch):
    monkeypatch.setattr(llm, "env", lambda name, default=None: default)
    body = client.post("/assess", json={"question": "지게차 작업 위험요인", "explain": False}).json()
    assert body["factors"] and body["conditions"]["targets"][0]["type"] == "지게차"
    assert body["rank_note"] == "질문 관련도를 계산하지 못해 사례 수 순으로 정렬했습니다"


def test_carry_and_guidance(client, monkeypatch):
    agent = api.app.state.agent
    monkeypatch.setattr(agent, "parser", lambda q: {})
    monkeypatch.setattr(agent, "relevance", lambda q, ids: None)
    carry = {"targets": FORKLIFT["targets"], "situations": []}
    result = client.post("/assess", json={"question": "추가 설명", "carry": carry, "explain": False})
    assert result.status_code == 200
    body = result.json()
    assert body["guidance"]["type"] == "need_situation" and body["factors"]
    assert body["conditions"]["targets"][0]["name"] == "지게차"


@pytest.mark.parametrize("carry", [
    {"targets": [{"kind": "equipment_type", "name": "없는 설비"}]},
    {"situations": ["없는 작업"]},
    {"targets": [{"kind": "잘못된 종류", "name": "지게차"}]},
])
def test_invalid_carry_is_422(client, carry):
    assert client.post("/assess", json={"question": "추가 설명", "carry": carry}).status_code == 422



def test_neo4j_matches_memory():
    """로컬 Neo4j가 켜져 있으면 평가 질문 조건으로 Cypher 조회 = 메모리 조회인지 확인."""
    store = REAL_OPEN_STORE()
    if store is None:
        pytest.skip("로컬 Neo4j 꺼짐")
    from graph import load_graph
    g = load_graph()
    questions = json.loads((ROOT / "data/eval/questions.json").read_text(encoding="utf-8"))
    try:
        for q in questions:
            cond = {"targets": [{"kind": "equipment", "name": q["equipment"]}] if q["equipment"] else [],
                    "situations": [q["situation"]] if q["situation"] else []}
            assert store.find_cases(cond) == find_cases(g, cond), q["id"]
        types = {"targets": [{"kind": "equipment_type", "name": "크레인"}, {"kind": "equipment", "name": "생산설비(프레스)"}],
                 "situations": ["비정형 작업(점검)", "크레인(천장주행, 갠트리, 지브 등) 사용 작업"]}
        assert store.find_cases(types) == find_cases(g, types)
    finally:
        store.close()
