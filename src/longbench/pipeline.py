"""longbench 트랙 - 자리만(아직 미구현, 범위 밖). CHUNKERS 골격만 남겨둔다.

TODO: LongBench/SCROLLS 데이터셋 로더 작성, QA 로드, retrieval/summary 실행 붙이기."""
import chunkers

DATASET = "longbench"


def build_chunkers(embeddings, count_tokens):
    return {
        "fixed":    chunkers.make_fixed(chunk_size=500),
        "semantic": chunkers.make_semantic(embeddings, count_tokens, percentile=90,
                                            min_tokens=100, max_tokens=400),
        "smart":    chunkers.make_smart(embeddings._client, count_tokens, mode="longbench",
                                         min_tokens=100, max_tokens=400),
    }


def run(task, chunker, limit=None, fake_llm=False):
    raise NotImplementedError("longbench 트랙은 아직 구현되지 않았습니다(TODO).")
