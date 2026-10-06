"""모델·토크나이저 로딩 (docs/restructure_plan.md 11.2절).
  get_token_counter  데이터셋별 토큰 카운터(pdf·docx = bge-m3, srt = Gemma 토크나이저)
  get_embeddings     bge-m3 임베딩(LangChain 래퍼). 청킹·평가는 안의 SentenceTransformer(._client)를 직접 쓴다
  get_models         청킹에 필요한 모델 묶음
LLM 로딩(번역·요약용 get_llm)은 아직 없다 - 해당 단계에서 legacy/src/llm.py를 옮겨 온다.

실행 전에 `source env.sh`(HF_HOME, HF_HUB_OFFLINE=1)가 되어 있어야 캐시된 모델을 쓴다.
"""
from common.config import EMBED_MODEL, TOKENIZER

_TOKENIZER_NAMES = {"bge-m3": EMBED_MODEL, "gemma": TOKENIZER}


def get_tokenizer(kind):
    from transformers import AutoTokenizer

    if kind not in _TOKENIZER_NAMES:
        raise KeyError(f"unknown tokenizer kind: {kind} (choices: {sorted(_TOKENIZER_NAMES)})")
    return AutoTokenizer.from_pretrained(_TOKENIZER_NAMES[kind])


def get_token_counter(kind):
    """문자열 -> 토큰 수 (특수 토큰 제외). kind: 'bge-m3' | 'gemma' (DATASETS[...]['counter'])."""
    tok = get_tokenizer(kind)
    return lambda s: len(tok.encode(s, add_special_tokens=False))


def get_embeddings():
    """LangChain Embeddings(bge-m3, 정규화). 청킹 엔진(semantic_core)과 평가(scripts/run.py)는 이 객체 안의
    SentenceTransformer(`._client`)를 꺼내 encode(글 목록, normalize_embeddings=True)로 직접 임베딩한다.
    LangChain의 embed_documents 경로는 청킹에서 쓰지 않는다(SemanticChunker를 호출하지 않으므로)."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True})


def get_models(dataset, need_embeddings=True):
    """청킹에 필요한 모델 묶음: {"embeddings", "embed_model", "count"}.
    embed_model = 청킹 엔진이 쓰는 SentenceTransformer(embeddings._client). "embeddings"(LangChain 래퍼)는 지금 청커가
    쓰지 않고 LangChain 방식 호출용으로만 남겨 두었다.
    count는 데이터셋 토큰 카운터(DATASETS[...]["counter"]: pdf·docx=bge-m3, srt=Gemma 토크나이저).
    need_embeddings=False(fixed만 돌릴 때)면 GPU 임베딩 모델을 올리지 않는다."""
    from common.config import dataset_cfg

    emb = get_embeddings() if need_embeddings else None
    return {"embeddings": emb, "embed_model": emb._client if emb is not None else None,
            "count": get_token_counter(dataset_cfg(dataset)["counter"])}
