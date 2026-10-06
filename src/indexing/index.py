"""청크 파일 -> 임베딩 인덱스 (docs/restructure_plan.md 6.1절).

계획서는 LangChain InMemoryVectorStore(dump/load)를 적었지만, 그 dump는 벡터를 들여쓴 JSON으로 저장해 파일이
매우 커지고 load는 내부 store 딕셔너리에 직접 접근한다. 그래서 같은 검색(정규화된 벡터의 코사인 = 내적 top-k)을
numpy로 직접 한다: 임베딩은 embeddings.npy(float32), 청크 순서는 chunk_ids.json. 검색 결과는 같다(동점 순서만 다를 수 있음).

- 인덱스는 청크 텍스트만 임베딩한다(제목 접두 같은 변형 없음 - 청커 간 비교가 흐려진다).
- 질문은 접두 없이 그대로 임베딩한다(bge-m3는 검색 지시문이 필요 없다).
- 폴더가 있고 청크 수가 같으면 불러오고, 아니면(또는 force) 다시 만든다. info.json은 사람이 확인하는 용도다.
- 모델의 max_seq_length(bge-m3 = 8192토큰)를 넘는 청크는 임베딩에서 뒷부분이 잘린다 -> info.json에 개수를 기록한다.
"""
import datetime
import json

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
    def __init__(self, chunks, matrix, encode):
        self.chunks, self.matrix, self.encode = chunks, matrix, encode

    def search(self, queries, k=5):
        """queries: 문자열 목록 -> 질문마다 [(청크 dict, 점수)] 상위 k개 (점수 내림차순, 동점은 청크 순서)."""
        q = self.encode(queries)
        sims = q @ self.matrix.T
        out = []
        for row in sims:
            order = np.argsort(-row, kind="stable")[:k]
            out.append([(self.chunks[i], float(row[i])) for i in order])
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
    np.save(d / "embeddings.npy", idx.matrix)
    (d / "chunk_ids.json").write_text(json.dumps([c["chunk_id"] for c in idx.chunks]), encoding="utf-8")
    (d / "info.json").write_text(json.dumps({
        "embed_model": EMBED_MODEL, "n_chunks": len(idx.chunks), "chunks_over_max_seq_length": getattr(idx, "truncated", None),
        "created_at": datetime.datetime.now().isoformat(timespec="seconds")}, indent=1), encoding="utf-8")


def load_index(chunks, encode, dataset, tag):
    """폴더가 있고 청크 id 순서가 같으면 불러오고, 아니면 None."""
    d = index_dir(dataset, tag)
    if not (d / "embeddings.npy").exists():
        return None
    ids = json.loads((d / "chunk_ids.json").read_text(encoding="utf-8"))
    if ids != [c["chunk_id"] for c in chunks]:
        return None
    return ChunkIndex(chunks, np.load(d / "embeddings.npy"), encode)


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
