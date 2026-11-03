"""실제 키로 데모 실행 기록을 만든다 (API 비용 발생).

    python src/make_demo_records.py
"""
import argparse
import json
import subprocess
from datetime import datetime, timedelta, timezone

import numpy as np

import llm
import vector_rag
from agent import SafetyAgent
from graph import ROOT, load_graph

# 사이드바 그룹 순서와 그룹 안 질문 순서를 함께 정한다
EXAMPLES = [
    ("Q4-D01", "물류·운반", 1),
    ("U03", "물류·운반", 2),
    ("U02", "정비·청소", 3),
    ("U01", "정비·청소", 4),
    ("U04", "정비·청소", 5),
    ("Q4-S11", "자동화·리프트", 6),
    ("U13", "자동화·리프트", 7),
    ("U05", "고소·밀폐·전기·화기", 8),
    # U06(화학 탱크 맨홀 진입) 제외: 산소결핍 환경에 방독마스크를 안내 — ID 검증은 통과했지만 안전상 부적절
    ("U09", "고소·밀폐·전기·화기", 10),
    ("U07", "고소·밀폐·전기·화기", 11),
    ("U14", "고소·밀폐·전기·화기", 12),
    ("U11", "질문이 막연할 때", 13),
]

# 결과에 따라 틀리지 않게 '무엇을 확인하는 질문인지'만 적는다
NOTES = {
    "Q4-S11": "야간 리프트 점검: 사례 적은 위험이 상세 설명에 오르는지",
    "Q4-D01": "지게차 도크 작업 (덱 예시)",
    "U03": "천장크레인 단독 운반: 설비와 작업 해석 확인",
    "U02": "가동 중 컨베이어 청소: 설비와 작업 해석 확인",
    "U01": "프레스 금형 교체: 설비와 작업 해석 확인",
    "U04": "사출성형기 노즐 막힘: 설비와 작업 해석 확인",
    "U13": "산업용 로봇: 작업을 묻는 안내 확인",
    "U05": "지붕 채광창 교체: 설비와 작업 해석 확인",
    "U06": "저장탱크 내부 점검: 밀폐공간 작업 해석 확인",
    "U09": "활선 배전반 점검: 설비와 작업 해석 확인",
    "U07": "인화성 물질 주변 용접: 설비와 작업 해석 확인",
    "U14": "밀폐공간 작업: 설비를 묻는 안내 확인",
    "U11": "막연한 질문: 구체적인 작업을 묻는 안내 확인",
}


def cached_vectors(graph, questions):
    """비교용 벡터는 캐시에서만 읽고 누락 시 중단한다."""
    try:
        cases = vector_rag.load_case_vectors(graph)
        texts = json.loads((vector_rag.CACHE / "queries.json").read_text(encoding="utf-8"))
        queries = np.load(vector_rag.CACHE / "queries.npy", allow_pickle=False)
        if queries.ndim != 2 or len(queries) != len(texts) or queries.shape[1] != cases.shape[1]:
            raise ValueError("질문 벡터의 개수 또는 차원이 맞지 않습니다.")
        selected = {qid: queries[texts.index(question)] for qid, question in questions.items()}
        if not np.isfinite(cases).all() or not all(np.isfinite(v).all() for v in selected.values()):
            raise ValueError("벡터에 유효하지 않은 값이 있습니다.")
        return cases, selected
    except (OSError, ValueError, TypeError) as exc:
        raise SystemExit("비교용 사례·질문 벡터 캐시가 없거나 맞지 않습니다. 캐시를 먼저 준비해 주세요.") from exc


def commit_id():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def merge_records(path, records, **metadata):
    """ID로 병합하고 모든 기존 기록의 메타데이터를 갱신한다 (API 호출 없음)."""
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"records": []}
    merged = {r["id"]: r for r in data["records"]}
    merged.update({r["id"]: r for r in records})
    data.update(metadata)
    data["records"] = [dict(merged[qid], group=group, order=order, note=NOTES[qid])
                       for qid, group, order in EXAMPLES if qid in merged]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", metavar="ID,ID")
    parser.add_argument("--meta-only", action="store_true", help="API 없이 기존 기록의 메타데이터만 갱신")
    args = parser.parse_args()
    selected = {qid for qid, _, _ in EXAMPLES} if args.only is None else set(args.only.split(","))
    unknown = selected - {qid for qid, _, _ in EXAMPLES}
    if unknown:
        parser.error("알 수 없는 예시 ID: " + ", ".join(sorted(unknown)))
    path = ROOT / "data/demo/records.json"
    if args.meta_only:
        merge_records(path, [])
        return
    rows = json.loads((ROOT / "data/eval/questions.json").read_text(encoding="utf-8"))
    unseen = json.loads((ROOT / "data/eval/unseen.json").read_text(encoding="utf-8"))["single"]
    sources = {q["id"]: q for q in rows + unseen}
    questions = {qid: sources[qid]["question"] for qid, _, _ in EXAMPLES if qid in selected}
    graph = load_graph()
    compared = {qid: question for qid, question in questions.items() if qid.startswith("Q4-")}
    if compared:
        vectors, queries = cached_vectors(graph, compared)
    ids, records = list(graph.cases), []
    seoul = timezone(timedelta(hours=9), "Asia/Seoul")
    stamp = dict(generated_at=datetime.now(seoul).isoformat(), model=llm.model_from_env(), commit=commit_id())
    for qid, question in questions.items():
        out = SafetyAgent().ask(question)
        out.update(stamp)  # 기록마다 측정 시각·모델·커밋을 남긴다 (나눠 기록해도 섞이지 않게)
        out.pop("text", None)
        out.update(id=qid, question=question,
                   graph_factors=[f["factor"] for f in (out.get("evidence") or {}).get("factors", [])])
        if qid in compared:
            top5 = np.argsort(-(vectors @ queries[qid]), kind="stable")[:5]
            out.update(vector_top5_factors=list(dict.fromkeys(graph.cases[ids[i]]["factor"] for i in top5)),
                       gold_factors=[x["factor"] for x in sources[qid]["gold_factors"]])
        else:
            out["expected"] = {key: sources[qid].get(key, [] if key != "guidance" else None)
                               for key in ("equipment", "situations", "guidance")}
        records.append(out)
        m = out["metrics"]
        print(f"{qid} latency={m['latency_s']}s llm_calls={m['llm_calls']} "
              f"tokens={m['input_tokens']}/{m['output_tokens']}")
    merge_records(path, records)


if __name__ == "__main__":
    main()
