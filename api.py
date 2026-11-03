"""제조 현장 위험성평가 GraphRAG API (FastAPI). 질문 하나를 LangGraph Agent(src/agent.py)로 처리한다.

    uvicorn api:app --reload        → http://localhost:8000/docs

    POST /assess {"question": "야간에 로봇 셀 들어가서 점검해야 하는데 위험요인 알려줘"}
    POST /assess {"conditions": {"targets": [{"kind": "equipment_type", "name": "지게차"}],
                                 "situations": ["혼재작업(작업자-작업차량)"]}, "explain": false}
                                                    # 질문 해석·설명 없이 그래프만 (API 키 불필요)
    GET  /health                                    # 그래프 저장소(neo4j/memory), 사례 수

설정은 .env에서만 읽는다. 로컬 Neo4j(docker compose up -d)에 연결되면 Neo4j, 아니면 메모리 그래프로 조회한다
(두 결과는 평가 질문 193개에서 일치 확인). 응답 metrics: 지연, LLM 호출 수, 입력·출력 토큰.
상태 코드: 입력 오류 422, 질문 해석 실패로 조건을 하나도 못 찾으면 502, 설명만 실패하면 200 + llm_error.
"""
import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from agent import SafetyAgent  # noqa: E402
from graph import load_graph  # noqa: E402

log = logging.getLogger("safety-agent")


class Target(BaseModel):
    kind: Literal["equipment", "equipment_type"]
    name: str = Field(max_length=100)


class Conditions(BaseModel):
    targets: list[Target] = Field(default=[], max_length=10)
    situations: list[str] = Field(default=[], max_length=10)


class AssessRequest(BaseModel):
    question: str | None = Field(None, max_length=500)  # LLM 비용 상한
    conditions: Conditions | None = None  # 주면 질문 해석(LLM)을 건너뛴다
    carry: Conditions | None = None
    explain: bool = True


def open_store():
    """로컬 Neo4j에 연결되면 그것, 아니면 None(메모리 그래프)."""
    try:
        from load_neo4j import Neo4jStore
        return Neo4jStore()
    except (Exception, SystemExit) as exc:  # 컨테이너 꺼짐·비밀번호 없음·localhost 아님
        log.warning("Neo4j 연결 실패, 메모리 그래프 사용: %s", type(exc).__name__)
        return None


@asynccontextmanager
async def lifespan(app):
    graph = load_graph()
    store = open_store()
    app.state.agent = SafetyAgent(graph, store=store)
    app.state.backend = store.name if store else "memory"
    yield
    if store:
        store.close()


app = FastAPI(title="제조 현장 위험성평가 GraphRAG", lifespan=lifespan)


def validate(agent, conditions):
    """그래프에 없는 설비·작업 이름은 422 (조회 결과 0건과 구별)."""
    g = agent.graph
    for t in conditions["targets"]:
        if t["name"] not in g.by[t["kind"]]:
            raise HTTPException(422, f"그래프에 없는 {t['kind']}: {t['name']}")
        t.setdefault("mention", t["name"])
        t.setdefault("type", t["name"] if t["kind"] == "equipment_type" else g.cases[g.by["equipment"][t["name"]][0]]["equipment_type"])
    unknown = [s for s in conditions["situations"] if s not in g.by["situation"]]
    if unknown:
        raise HTTPException(422, f"그래프에 없는 작업 상황: {unknown}")


def response(out, backend):
    ev, items, reply = out["evidence"], out["items"] or {}, out["reply"] or {}
    return {
        "question": out["question"], "backend": backend,
        "conditions": out["conditions"], "relax_level": out["relax_level"], "relax_note": out["relax_note"],
        "rank_note": out["rank_note"],
        "guidance": out["guidance"], "parse_ok": out["parse_ok"],
        "total_cases": ev["total_cases"] if ev else 0,
        "factors": [{"id": f["id"], "factor": f["factor"], "cases": f["cases"]} for f in ev["factors"]] if ev else [],
        "detailed": [{"id": d["id"], "factor": d["factor"], "cases": d["cases"], "pick": d["pick"],
                      "advice": items.get(d["id"], {}).get("advice"),
                      "measures": [ev["measures"][m]["text"] for m in d["measure_ids"]],
                      "evidence_cases": [{"id": c, "overview": ev["cases"][c]["overview"]} for c in d["case_ids"]]}
                     for d in ev["detailed"]] if ev else [],
        "summary": reply.get("summary") if items else None,
        "llm_error": out["llm_error"], "metrics": out["metrics"], "text": out["text"],
    }


def assess(agent, backend, question, given, explain, carry=None):
    if given is not None:
        validate(agent, given)
    if carry is not None:
        validate(agent, carry)
    out = agent.ask(question, given, explain, carry=carry)
    c = out["conditions"] or {"targets": [], "situations": []}
    if given is None and not (c["targets"] or c["situations"]) and out["llm_error"]:
        raise HTTPException(502, out["llm_error"])  # 질문 해석 실패 (키 없음·API 오류)
    return response(out, backend)


@app.post("/assess")
async def assess_endpoint(body: AssessRequest):
    if not (body.question and body.question.strip()) and body.conditions is None:
        raise HTTPException(422, "question 또는 conditions가 필요합니다.")
    given = body.conditions.model_dump() if body.conditions else None
    carry = body.carry.model_dump() if body.carry else None
    # 그래프 조회·LLM 호출이 동기 코드라 스레드에서 돌려 이벤트 루프를 막지 않는다
    return await asyncio.to_thread(assess, app.state.agent, app.state.backend, body.question, given, body.explain, carry)


@app.get("/health")
async def health():
    return {"backend": app.state.backend, "cases": len(app.state.agent.graph.cases)}
