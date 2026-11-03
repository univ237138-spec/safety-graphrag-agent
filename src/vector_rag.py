"""비교 기준: 벡터 RAG (사례 1건 = 문서 1개) + BM25 + 하이브리드(RRF).

문서에는 그래프와 같은 정보(개요·설비·작업·요인·대책)를 넣는다 (공정한 비교).
임베딩은 OpenAI text-embedding-3-small로 한 번만 만들고 data/cache/에 저장해 다시 쓴다 (git 제외).
"""
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm  # noqa: E402
from graph import ROOT  # noqa: E402

CACHE = ROOT / "data" / "cache"
BATCH = 100


def document(case):
    return "\n".join([
        f"사고개요: {case['overview']}",
        f"설비: {case['equipment'] or '분류 불가'}",
        f"작업: {case['situation']}",
        f"재해유발요인: {case['factor']}",
        "감소대책: " + " / ".join(case["measures"]),
    ])


def load_case_vectors(g):
    """현재 그래프와 일치하는 사례 캐시만 읽는다 (API 호출 없음)."""
    docs = json.loads((CACHE / "case_docs.json").read_text(encoding="utf-8"))
    if docs != [document(g.cases[i]) for i in g.cases]:
        raise ValueError("사례 문서 캐시가 현재 그래프의 문서·순서와 다릅니다.")
    vectors = np.load(CACHE / "case_docs.npy", allow_pickle=False)
    if vectors.ndim != 2 or len(vectors) != len(docs):
        raise ValueError("사례 캐시 벡터의 개수 또는 차원이 맞지 않습니다.")
    return vectors


def factor_relevance(g, case_ids, positions, vectors, query):
    scores = {}
    for case_id in case_ids:
        factor = g.cases[case_id]["factor"]
        scores[factor] = max(scores.get(factor, -math.inf), float(vectors[positions[case_id]] @ query))
    return scores


class CaseRelevance:
    def __init__(self, graph):
        self.graph = graph
        self.positions = {case_id: r for r, case_id in enumerate(graph.cases)}
        self.vectors = None

    def __call__(self, question, case_ids):
        return self.score(question_vector(question), case_ids)

    def score(self, query, case_ids):
        if self.vectors is None:
            self.vectors = load_case_vectors(self.graph)
        return factor_relevance(self.graph, case_ids, self.positions, self.vectors, query)


def question_vector(question, embed=None):
    """질문당 한 번 만든 벡터를 후보 검색과 관련도에 함께 쓴다."""
    query = np.array((embed or llm.embed)([question])[0], dtype=np.float32)
    norm = np.linalg.norm(query)
    if query.ndim != 1 or not np.isfinite(norm) or norm == 0:
        raise ValueError("질문 벡터를 정규화할 수 없습니다.")
    return query / norm


def embed_cached(name, texts):
    """texts → 정규화된 임베딩 행렬. 같은 글 목록이면 캐시를 쓴다 (API 호출 없음)."""
    vec_path, txt_path = CACHE / f"{name}.npy", CACHE / f"{name}.json"
    if vec_path.exists() and txt_path.exists() and json.loads(txt_path.read_text(encoding="utf-8")) == texts:
        return np.load(vec_path)
    vectors = []
    for start in range(0, len(texts), BATCH):
        vectors += llm.embed(texts[start:start + BATCH])
    matrix = np.array(vectors, dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(vec_path, matrix)
    txt_path.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    return matrix


def bigrams(text):
    """한국어 형태소 분석기 없이 쓰는 글자 2-gram (공백·기호 제거)."""
    text = re.sub(r"[^0-9A-Za-z가-힣]", "", text)
    return [text[i:i + 2] for i in range(len(text) - 1)]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.tf = [Counter(bigrams(d)) for d in docs]
        self.len = np.array([sum(tf.values()) for tf in self.tf], dtype=np.float32)
        self.avg, self.k1, self.b = self.len.mean(), k1, b
        df = Counter(t for tf in self.tf for t in tf)
        self.idf = {t: math.log(1 + (len(docs) - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def scores(self, query):
        result = np.zeros(len(self.tf), dtype=np.float32)
        for term in set(bigrams(query)):
            if term not in self.idf:
                continue
            tf = np.array([d.get(term, 0) for d in self.tf], dtype=np.float32)
            result += self.idf[term] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.len / self.avg))
        return result


class Retriever:
    def __init__(self, graph):
        self.ids = list(graph.cases)
        self.position = {case_id: k for k, case_id in enumerate(self.ids)}
        docs = [document(graph.cases[i]) for i in self.ids]
        self.doc_vectors = embed_cached("case_docs", docs)
        self.bm25 = BM25(docs)

    def rank(self, method, query_vector, query_text):
        """사례 id를 관련도 순으로 전부 반환."""
        if method == "vector":
            order = np.argsort(-(self.doc_vectors @ query_vector))
        elif method == "bm25":
            order = np.argsort(-self.bm25.scores(query_text))
        elif method == "hybrid":  # Reciprocal Rank Fusion (k=60)
            fused = np.zeros(len(self.ids))
            for ranks in (self.rank("vector", query_vector, query_text), self.rank("bm25", query_vector, query_text)):
                for r, case_id in enumerate(ranks):
                    fused[self.position[case_id]] += 1 / (60 + r + 1)
            order = np.argsort(-fused)
        else:
            raise ValueError(method)
        return [self.ids[i] for i in order]
