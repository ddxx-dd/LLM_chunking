"""청크 파일 -> 임베딩 인덱스 (docs/restructure_plan.md 6.1절). 벡터 저장·검색은 FAISS(`faiss-cpu`, .venv311에만 설치).

  - 인덱스 종류: `faiss.IndexFlatIP` = **정확 검색**(근사 아님). 벡터를 정규화해서 넣으므로 내적 = 코사인 유사도다.
    청커를 비교하는 실험이라 검색 근사 오차가 결과에 섞이면 안 되기 때문에 근사 인덱스(HNSW, IVF 등)는 쓰지 않는다.
  - 저장: data/index/<dataset>/<태그>/ 에 index.faiss(벡터), chunk_ids.json(행 번호 -> 청크 id), info.json(사람이 보는 용도).
    청크의 글과 메타데이터는 청크 파일(jsonl)이 갖고 있으므로 인덱스에는 벡터와 순서만 둔다.
  - 폴더가 있고 청크 id 순서가 같으면 불러오고, 아니면(또는 force) 다시 만든다.
  - 동점(점수가 정확히 같은 청크)은 id(= 청크 순서)가 앞선 것이 먼저 오도록 정렬한다. 상위 k의 경계에서 동점이면 어느 쪽이
    들어가는지는 FAISS가 정하지만 같은 인덱스에서는 항상 같다(실제 임베딩에서는 동점이 사실상 생기지 않는다).

- 인덱스는 청크 텍스트만 임베딩한다(제목 접두 같은 변형 없음 - 청커 간 비교가 흐려진다).
- 질문은 접두 없이 그대로 임베딩한다(bge-m3는 검색 지시문이 필요 없다).
- 모델의 max_seq_length(bge-m3 = 8192토큰)를 넘는 청크는 임베딩에서 뒷부분이 잘린다 -> info.json에 개수를 기록한다.
"""
import datetime
import json

import faiss
import numpy as np

from common.config import CHUNKS_DIR, EMBED_MODEL, INDEX_DIR


def load_chunks(dataset, tag, doc_ids=None):
    path = CHUNKS_DIR / dataset / f"{tag}.jsonl"
    # splitlines()는 U+0085(NEL) 등 텍스트 안의 문자에서도 줄을 나눠 JSON 줄이 깨진다 -> \n 으로만 나눈다
    chunks = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    if doc_ids is not None:
        keep = set(doc_ids)
        chunks = [c for c in chunks if c["doc_id"] in keep]
    return chunks


def make_encoder(embed_model, batch_size=16):
    """SentenceTransformer -> encode(texts) = 정규화된 float32 행렬. 배치를 작게 잡아 긴 청크의 메모리 폭주(OOM)를 막는다."""
    def encode(texts):
        return np.asarray(embed_model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                                             show_progress_bar=False), dtype=np.float32)
    return encode


class ChunkIndex:
    """청크 목록 + FAISS 정확 검색 인덱스. matrix는 (청크 수, 차원) float32, 행 i = 청크 i의 정규화된 임베딩."""

    def __init__(self, chunks, matrix, encode):
        matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        self.chunks, self.encode = chunks, encode
        self.faiss = faiss.IndexFlatIP(matrix.shape[1])
        if len(matrix):
            self.faiss.add(matrix)

    @classmethod
    def from_faiss(cls, chunks, faiss_index, encode):
        """저장된 FAISS 인덱스를 그대로 감싼다(벡터를 다시 넣지 않는다)."""
        idx = cls.__new__(cls)
        idx.chunks, idx.encode, idx.faiss = chunks, encode, faiss_index
        return idx

    @property
    def matrix(self):
        n = self.faiss.ntotal
        return self.faiss.reconstruct_n(0, n) if n else np.zeros((0, self.faiss.d), np.float32)

    def search(self, queries, k=5):
        """queries: 문자열 목록 -> 질문마다 [(청크 dict, 점수)] 상위 k개 (점수 내림차순, 동점은 청크 순서)."""
        queries = list(queries)
        n = self.faiss.ntotal
        if n == 0 or not queries:
            return [[] for _ in queries]
        q = np.ascontiguousarray(self.encode(queries), dtype=np.float32)
        scores, ids = self.faiss.search(q, min(k, n))
        out = []
        for srow, irow in zip(scores, ids):
            hits = sorted(((float(sc), int(i)) for sc, i in zip(srow, irow) if i >= 0), key=lambda t: (-t[0], t[1]))
            out.append([(self.chunks[i], sc) for sc, i in hits])
        return out


def index_dir(dataset, tag):
    return INDEX_DIR / dataset / tag


def build_index(chunks, encode, max_seq_length=None):
    matrix = encode([c["text"] for c in chunks]) if chunks else np.zeros((0, 1), np.float32)
    idx = ChunkIndex(chunks, matrix, encode)
    idx.truncated = sum(1 for c in chunks if max_seq_length and c["n_tokens"] > max_seq_length)
    return idx


def save_index(idx, dataset, tag):
    d = index_dir(dataset, tag)
    d.mkdir(parents=True, exist_ok=True)
    faiss.write_index(idx.faiss, str(d / "index.faiss"))
    (d / "chunk_ids.json").write_text(json.dumps([c["chunk_id"] for c in idx.chunks]), encoding="utf-8")
    (d / "info.json").write_text(json.dumps({
        "backend": "faiss.IndexFlatIP", "faiss_version": faiss.__version__, "dim": idx.faiss.d,
        "embed_model": EMBED_MODEL, "n_chunks": len(idx.chunks), "chunks_over_max_seq_length": getattr(idx, "truncated", None),
        "created_at": datetime.datetime.now().isoformat(timespec="seconds")}, indent=1), encoding="utf-8")


def load_index(chunks, encode, dataset, tag):
    """폴더가 있고 청크 id 순서가 같으면 불러오고, 아니면 None."""
    d = index_dir(dataset, tag)
    if not (d / "index.faiss").exists():
        return None
    ids = json.loads((d / "chunk_ids.json").read_text(encoding="utf-8"))
    if ids != [c["chunk_id"] for c in chunks]:
        return None
    fx = faiss.read_index(str(d / "index.faiss"))
    if fx.ntotal != len(chunks):
        return None
    return ChunkIndex.from_faiss(chunks, fx, encode)


def get_index(dataset, tag, encode, doc_ids=None, force=False, max_seq_length=None, use_cache=True):
    """doc_ids로 청크를 걸러 쓰는 표본 실행에서도 캐시가 섞이지 않는다(청크 id 목록이 다르면 다시 만든다)."""
    chunks = load_chunks(dataset, tag, doc_ids)
    if use_cache and not force:
        idx = load_index(chunks, encode, dataset, tag)
        if idx is not None:
            return idx
    idx = build_index(chunks, encode, max_seq_length)
    if use_cache:
        save_index(idx, dataset, tag)
    return idx
