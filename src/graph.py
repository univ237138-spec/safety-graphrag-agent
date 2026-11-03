"""SIF 사고사례 → 위험성평가 그래프 (메모리).

노드: 사고사례, 업종, 설비유형 ← 세부 설비(기인물), 고위험작업(상황), 재해유발요인, 감소대책
관계는 사고사례를 거쳐 이어지고, 고위험작업→요인→대책의 가중치는 근거 사례 수다.
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "sif.json"
SCOPE = {"제조업", "운수창고통신업"}  # 제조 + 물류 현장
UNKNOWN_EQUIPMENT = "분류 불가"


def equipment_type(raw):
    """설비유형: '크레인(천장주행크레인)'·'자동화설비[자동적재기(…)]' → 첫 괄호 앞. 비면 원문."""
    return re.split(r"[(\[]", raw)[0].strip() or raw.strip()


def measure_lines(text):
    """감소대책 원문 → 줄 목록 ('▶' 머리표·앞뒤 공백 제거, 빈 줄 제외)."""
    return [line.strip().lstrip("▶").strip() for line in (text or "").splitlines() if line.strip().lstrip("▶").strip()]


def load_cases(path=RAW):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = []
    for row in rows:
        if row["sifLclsfNm"] not in SCOPE:
            continue
        year = re.match(r"\s*(\d{4})년", row["disasterOverview"] or "")
        cases.append({
            "id": f"C{len(cases) + 1:04d}",
            "overview": row["disasterOverview"],
            "year": int(year.group(1)) if year else None,
            "industry": row["sifLclsfNm"],
            "sector": row["sifMclsfNm"],
            "subsector": row["sifSclsfNm"],
            # 세부 설비(원 표기) ─속함→ 설비유형(괄호 앞). '분류 불가'는 설비 연결 없음
            "equipment": None if row["orgtNm"] == UNKNOWN_EQUIPMENT else row["orgtNm"].strip(),
            "equipment_type": None if row["orgtNm"] == UNKNOWN_EQUIPMENT else equipment_type(row["orgtNm"]),
            "situation": row["situation"],
            "factor": row["disasterFactor"],
            "measures": measure_lines(row["dcrsCntrplnCn"]),
        })
    return cases


class SafetyGraph:
    def __init__(self, cases):
        self.cases = {c["id"]: c for c in cases}
        self.by = {key: defaultdict(list) for key in ("equipment", "equipment_type", "situation", "factor", "sector")}
        for c in cases:
            for key in self.by:
                if c[key]:
                    self.by[key][c[key]].append(c["id"])

    def names(self, kind):
        return sorted(self.by[kind])

    def case_ids(self, equipment=None, situation=None, equipment_type=None):
        """세부 설비·설비유형·고위험작업 조건에 맞는 사례 (여럿 주면 교집합)."""
        conditions = (("equipment", equipment), ("equipment_type", equipment_type), ("situation", situation))
        sets = [set(self.by[k].get(v, [])) for k, v in conditions if v]
        return sorted(set.intersection(*sets)) if sets else []

    def factors(self, case_ids):
        """요인 → 근거 사례 수. 사례 수 많은 순."""
        return Counter(self.cases[i]["factor"] for i in case_ids).most_common()

    def measures(self, case_ids, factor=None):
        """감소대책 → 근거 사례 수 (factor를 주면 그 요인의 사례만)."""
        counts = Counter()
        for i in case_ids:
            if factor is None or self.cases[i]["factor"] == factor:
                counts.update(set(self.cases[i]["measures"]))
        return counts.most_common()


def load_graph(path=RAW):
    return SafetyGraph(load_cases(path))
