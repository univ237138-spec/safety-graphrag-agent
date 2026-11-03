import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from graph import equipment_type, load_graph, measure_lines  # noqa: E402

G = load_graph()


def test_normalize():
    assert equipment_type("크레인(천장주행크레인)") == "크레인"
    assert equipment_type("(기타)") == "(기타)"
    assert equipment_type("자동화설비[자동적재기(Stacking M/C)]") == "자동화설비"
    assert measure_lines("▶ 작업계획서 작성\n\n▶  보호구 착용 ") == ["작업계획서 작성", "보호구 착용"]


def test_scope_and_counts():
    industries = {c["industry"] for c in G.cases.values()}
    assert industries == {"제조업", "운수창고통신업"}
    assert len(G.cases) == 1693


def test_query_is_exhaustive():
    ids = G.case_ids(situation="지게차 사용 작업")
    factors = dict(G.factors(ids))
    # 그래프 조회는 사례 1건짜리 요인까지 전부 돌려준다
    assert sum(factors.values()) == len(ids)
    assert min(factors.values()) == 1
    both = G.case_ids(equipment="지게차", situation="지게차 사용 작업")
    assert set(both) <= set(ids)


def test_equipment_hierarchy():
    press = G.case_ids(equipment="생산설비(프레스)")
    production = G.case_ids(equipment_type="생산설비")
    assert len(press) == 14 and set(press) < set(production)
    # 설비유형의 사례 = 그 아래 세부 설비 사례의 합
    children = {G.cases[i]["equipment"] for i in production}
    assert sum(len(G.by["equipment"][e]) for e in children) == len(production)
