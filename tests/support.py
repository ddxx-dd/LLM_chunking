"""테스트 공용 도구 - 모든 테스트 파일이 가장 먼저 `from support import ...`로 가져온다.

  - 이 모듈을 임포트하면 src/가 sys.path에 들어가서, 테스트마다 경로 설정을 반복하지 않는다.
  - 설명용 예시 문서(tests/fixtures: A=논문, B=자막, C=긴 문단)의 이름·파라미터와, 모델 없이 쓰는 가짜 임베딩(키워드 축)과
    공백 단위 토큰 카운터를 둔다(실제 모델·GPU 불필요).
  - 테스트 파일끼리는 서로를 임포트하지 않는다 - 여럿이 쓰는 것은 여기로 모은다.
"""
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent            # tests/
ROOT = HERE.parent
SRC = ROOT / "src"
FIX = HERE / "fixtures"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

TOPIC_AXES = ["alpha", "beta", "gamma"]          # 합성 문서용 주제어 축


AXES = {
    "A": ["free rid|honest|rational|agent|effort|participation", "reward|mechanism|truthful|contribution|align|monotone|strategy",
          "cifar|femnist|evaluate|simulated", "federated|train|model|data"],
    "B": ["cab|wheel|excuse|dopinder|nice|smells|lonesome|help", "wife|gita|cousin|bandhu|dishonorable|attractive|reason|called",
          "love|world|daffodil|yoga|mama|taste|bad|tight", "suit|christmas|pool"],
    "C": ["accuracy|benchmark|seeds|baseline", "cost|gpu|clients|round|hours", "privacy|encrypted|raw data|device|gradient"],
}
# 설명용 작은 파라미터: fixed 글자 수, strip, min/max(공백 단위 단어 수), mode(srt는 legacy smart 설정이 쓴다)
PARAMS = {"A": dict(fixed=160, strip=True, mn=25, mx=60, mode="pdf"),
          "B": dict(fixed=200, strip=False, mn=15, mx=40, mode="srt"),
          "C": dict(fixed=160, strip=True, mn=20, mx=45, mode="pdf")}
FILES = {"A": "example_a", "B": "example_b", "C": "example_c"}

def make_stub(axes):
    def vec(t):
        tl = t.lower()
        v = np.array([len(re.findall(a, tl)) for a in axes], float) + 0.05
        return v / np.linalg.norm(v)

    class Emb:
        def embed_documents(self, ts):
            return [vec(t).tolist() for t in ts]

        def embed_query(self, t):
            return vec(t).tolist()

        @property
        def _client(self):
            class Client:
                def encode(s, ts, normalize_embeddings=True):
                    return np.array([vec(t) for t in ts])
            return Client()
    return Emb()


def count_words(s):
    return len(s.split())


def models_for(key):
    emb = make_stub(AXES[key])
    return {"embeddings": emb, "embed_model": emb._client, "count": count_words}


def cfg_for(key, chunker):
    p = PARAMS[key]
    if chunker == "fixed":
        return {"chunk_size_chars": p["fixed"], "strip_whitespace": p["strip"], "overlap_ratio": 0.0}
    if chunker == "smart" and p["mode"] == "srt":                      # srt smart는 아직 legacy
        return {"mode": "srt", "algorithm": "legacy", "min_tokens": p["mn"], "max_tokens": p["mx"], "window": 2}
    return {"percentile": 90, "min_tokens": p["mn"], "max_tokens": p["mx"]}     # semantic(규칙 off) / smart(규칙 on)


def words(topic, n):
    """주제어 n개로 이뤄진 문장 하나(합성 문서용): 'alpha alpha ... alpha.'"""
    return " ".join([topic] * n) + "."
