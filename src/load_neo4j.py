"""메모리 그래프 → 로컬 Neo4j 적재 + 메모리 그래프와 결과 대조.

안전장치: 접속 주소가 localhost/127.0.0.1이 아니면 실행하지 않는다 (적재 전에 DB를 비우므로
클라우드 인스턴스를 실수로 지우는 일을 막는다).
실행: docker compose --profile neo4j up -d 후 python src/load_neo4j.py
"""
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph import load_graph  # noqa: E402
from settings import env  # noqa: E402

LOCAL_HOSTS = {"localhost", "127.0.0.1"}


def local_uri():
    uri = env("NEO4J_URI", "bolt://localhost:7687")
    if urlparse(uri).hostname not in LOCAL_HOSTS:
        raise SystemExit(f"로컬 Neo4j만 허용합니다: {urlparse(uri).hostname}")
    return uri


def driver():
    from neo4j import GraphDatabase
    return GraphDatabase.driver(local_uri(), auth=(env("NEO4J_USERNAME", "neo4j"), env("NEO4J_PASSWORD")))


SCHEMA = [
    "CREATE CONSTRAINT case_id IF NOT EXISTS FOR (n:Case) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT equipment IF NOT EXISTS FOR (n:Equipment) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT equipment_type IF NOT EXISTS FOR (n:EquipmentType) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT situation IF NOT EXISTS FOR (n:Situation) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT factor IF NOT EXISTS FOR (n:Factor) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT measure IF NOT EXISTS FOR (n:Measure) REQUIRE n.text IS UNIQUE",
    "CREATE CONSTRAINT sector IF NOT EXISTS FOR (n:Sector) REQUIRE n.name IS UNIQUE",
]

LOAD = """
UNWIND $rows AS r
CREATE (c:Case {id: r.id, overview: r.overview, year: r.year, industry: r.industry, subsector: r.subsector})
MERGE (s:Sector {name: r.sector})          MERGE (c)-[:IN_SECTOR]->(s)
MERGE (st:Situation {name: r.situation})   MERGE (c)-[:DURING]->(st)
MERGE (f:Factor {name: r.factor})          MERGE (c)-[:CAUSED_BY]->(f)
FOREACH (_ IN CASE WHEN r.equipment IS NULL THEN [] ELSE [1] END |
  MERGE (e:Equipment {name: r.equipment})
  MERGE (t:EquipmentType {name: r.equipment_type})
  MERGE (e)-[:IS_A]->(t)
  MERGE (c)-[:INVOLVES]->(e))
FOREACH (m IN r.measures | MERGE (x:Measure {text: m}) MERGE (c)-[:SUGGESTS]->(x))
"""

# 설비·작업 조건 → 요인과 근거 사례 수 (메모리 그래프 factors()와 같은 결과여야 함)
FACTORS = """
MATCH (c:Case)-[:CAUSED_BY]->(f:Factor)
WHERE ($equipment IS NULL OR EXISTS { (c)-[:INVOLVES]->(:Equipment {name: $equipment}) })
  AND ($situation IS NULL OR EXISTS { (c)-[:DURING]->(:Situation {name: $situation}) })
RETURN f.name AS factor, count(c) AS cases
"""


# Agent 조회: 설비(세부·유형, 여럿이면 합집합) ∩ 작업 상황(여럿이면 합집합). 메모리 agent.find_cases와 같은 뜻
FIND_CASES = """
MATCH (c:Case)
WHERE (size($situations) = 0 OR EXISTS { (c)-[:DURING]->(s:Situation) WHERE s.name IN $situations })
  AND (size($equipment) + size($types) = 0
       OR EXISTS { (c)-[:INVOLVES]->(e:Equipment) WHERE e.name IN $equipment }
       OR EXISTS { (c)-[:INVOLVES]->(:Equipment)-[:IS_A]->(t:EquipmentType) WHERE t.name IN $types })
RETURN c.id AS id ORDER BY id
"""


class Neo4jStore:
    """API용 사례 조회기. 요청이 동시에 오므로 쿼리마다 세션을 연다 (세션은 스레드 간 공유 불가)."""

    name = "neo4j"

    def __init__(self):
        self.driver = driver()
        self.driver.verify_connectivity()

    def find_cases(self, conditions):
        targets = conditions["targets"]
        if not targets and not conditions["situations"]:
            return []
        with self.driver.session() as session:
            return [r["id"] for r in session.run(
                FIND_CASES, situations=conditions["situations"],
                equipment=[t["name"] for t in targets if t["kind"] == "equipment"],
                types=[t["name"] for t in targets if t["kind"] == "equipment_type"])]

    def close(self):
        self.driver.close()


def load(session, graph):
    session.run("MATCH (n) DETACH DELETE n").consume()
    for statement in SCHEMA:
        session.run(statement).consume()
    rows = list(graph.cases.values())
    for start in range(0, len(rows), 500):
        session.run(LOAD, rows=rows[start:start + 500]).consume()


def main():
    import json
    graph = load_graph()
    with driver() as db, db.session() as session:
        load(session, graph)
        counts = {r["label"]: r["n"] for r in session.run(
            "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n")}
        print("노드:", counts)
        questions = json.loads((Path(__file__).resolve().parent.parent / "data/eval/questions.json").read_text(encoding="utf-8"))
        mismatch = 0
        for q in questions:
            cypher = {r["factor"]: r["cases"] for r in session.run(FACTORS, equipment=q["equipment"], situation=q["situation"])}
            memory = {g["factor"]: g["cases"] for g in q["gold_factors"]}
            mismatch += cypher != memory
        print(f"평가 질문 {len(questions)}개 Cypher vs 메모리 그래프 불일치: {mismatch}")


if __name__ == "__main__":
    main()
