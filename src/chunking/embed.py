"""임베딩과 토큰 세기: bge-m3 fp16(fp32 대비 top-3 99% 일치, 약 4배 빠름), 단위 임베딩 디스크 캐시.

청커는 모델을 직접 만지지 않고 Runtime(encode, count_many, cache_dir)만 본다 - 테스트는 가짜 encode를 넣는다.
  encode(texts)      -> float32 (n, d), 행마다 길이 1 (코사인 = 내적)
  count_many(texts)  -> [토큰 수] (bge-m3 토크나이저, 특수 토큰 제외)
캐시: <cache_dir>/<kind>/<dataset>__<doc_id>.npz 에 단위 구간·임베딩(fp16)·토큰 수를 저장한다. 구간이 다르면 쓰지 않는다.
"""
from pathlib import Path

import numpy as np

BATCH = 32


def load_model():
    """bge-m3 SentenceTransformer(fp16). 실행 전에 `source env.sh`로 HF 캐시(오프라인)가 잡혀 있어야 한다."""
    from common.models import get_embeddings

    model = get_embeddings()._client
    model.half()
    return model


def make_encode(model, batch_size=BATCH):
    def encode(texts):
        texts = list(texts)
        if not texts:
            return np.zeros((0, 1024), np.float32)
        return np.asarray(model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
    return encode


def make_count_many(kind="bge-m3"):
    from common.models import get_tokenizer

    tok = get_tokenizer(kind)
    tok.model_max_length = 10 ** 9        # 긴 평문의 토큰 수만 센다(길이 경고 끔)

    def count_many(texts):
        texts = list(texts)
        out = []
        for i in range(0, len(texts), 4096):
            out.extend(len(x) for x in tok(texts[i:i + 4096], add_special_tokens=False)["input_ids"])
        return out
    return count_many


def _safe(name):
    return str(name).replace("/", "_")


class Runtime:
    def __init__(self, encode, count_many, cache_dir=None):
        self.encode, self.count_many = encode, count_many
        self.cache_dir = Path(cache_dir) if cache_dir else None

    @classmethod
    def load(cls, cache_dir=None, need_embeddings=True, tokenizer="bge-m3"):
        """실제 모델·토크나이저를 올린 Runtime. need_embeddings=False(fixed 등만 돌릴 때)면 GPU 모델을 올리지 않는다."""
        encode = make_encode(load_model()) if need_embeddings else None
        return cls(encode, make_count_many(tokenizer), cache_dir)

    def _path(self, dataset, doc_id, kind):
        return self.cache_dir / kind / f"{_safe(dataset)}__{_safe(doc_id)}.npz"

    def cache_get(self, dataset, doc_id, kind, spans):
        if not self.cache_dir:
            return None
        p = self._path(dataset, doc_id, kind)
        if not p.exists():
            return None
        z = np.load(p)
        if z["spans"].shape != (len(spans), 2) or not (z["spans"] == np.array(spans, dtype=np.int64).reshape(-1, 2)).all():
            return None
        return z["emb"].astype(np.float32), z["ntok"]

    def cache_put(self, dataset, doc_id, kind, spans, emb, ntok):
        if not self.cache_dir:
            return
        p = self._path(dataset, doc_id, kind)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, spans=np.array(spans, dtype=np.int64).reshape(-1, 2), emb=np.asarray(emb).astype(np.float16), ntok=np.asarray(ntok, dtype=np.int32))
