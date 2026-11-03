"""세부 설비 이름 색인을 만든다. 직접 실행할 때만 임베딩 API를 사용한다."""
import json

import numpy as np

import llm
from graph import load_graph
from query_parser import node_names
from vector_rag import BATCH, CACHE


def build(g=None, embed=None, cache=CACHE):
    """괄호 안 이름을 묶음으로 임베딩하고 정규화하여 저장한다."""
    nodes = node_names(g or load_graph())
    embed = embed or llm.embed
    vectors = []
    for start in range(0, len(nodes), BATCH):
        vectors.extend(embed([n["inner"] for n in nodes[start:start + BATCH]]))
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or len(matrix) != len(nodes):
        raise ValueError("노드 벡터 개수 또는 차원이 맞지 않습니다.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(matrix).all() or (norms == 0).any():
        raise ValueError("노드 벡터를 정규화할 수 없습니다.")
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / "node_names.npy", matrix / norms)
    (cache / "node_names.json").write_text(json.dumps(nodes, ensure_ascii=False, indent=1), encoding="utf-8")
    return nodes


if __name__ == "__main__":
    print(f"노드 {len(build())}개 색인 저장")
