"""자막 실험 스크립트들이 공유하는 fixed/semantic 청커 기본 설정.
실측으로 검증된 값(chunk_size=500, min128/max1024) - 여러 스크립트가 각자
따로 들고 있으면 한쪽만 갱신되는 드리프트가 생긴다(실제로 한 번 발생했던 버그)."""
from chunking.fixed_chunker import fixed_chunking
from chunking.semantic_chunker import semantic_chunking


def default_subtitle_chunkers(embed_model):
    return {
        "fixed": lambda text: fixed_chunking(text, chunk_size=500),
        "semantic": lambda text: semantic_chunking(text, embed_model, method="percentile", amount=15,
                                                     min_chunk_tokens=128, max_chunk_tokens=1024),
    }
