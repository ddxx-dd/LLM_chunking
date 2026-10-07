"""srt(자막) 청킹. 청킹 경로는 chunking.v1(옛 디스패처) + chunking.sizectl(min/max 엔진) + srt.smart (src/srt/README.md 참고).

  source env.sh
  .venv311/bin/python scripts/srt/chunk.py --chunker fixed --chunk-size-chars 500
  .venv311/bin/python scripts/srt/chunk.py --chunker semantic --percentile 90 --min-tokens 128 --max-tokens 512
  .venv311/bin/python scripts/srt/chunk.py --chunker smart --min-tokens 128 --max-tokens 512 --window 2 [--docs id1,id2] [--limit N] [--force]

- 결과: data/chunks/srt/<태그>.jsonl. 크기 파라미터를 안 주면 src/srt/config.py의 시작값을 쓴다 - 실제 문서에는 사용자가 정한 값을 명시할 것.
- pdf·docx는 scripts/chunk.py (새 청커 7종).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from chunking.v1.runner import CHUNKERS, chunk_dataset  # noqa: E402
from common.models import get_models  # noqa: E402

DATASET = "srt"


def overrides_for(chunker, ns):
    if chunker == "fixed":
        return {"target_tokens": ns.target_tokens, "chunk_size_chars": ns.chunk_size_chars, "overlap_ratio": ns.overlap}
    if chunker == "semantic":
        return {"percentile": ns.percentile, "min_tokens": ns.min_tokens, "max_tokens": ns.max_tokens, "buffer_size": ns.buffer_size}
    return {"min_tokens": ns.min_tokens, "max_tokens": ns.max_tokens, "window": ns.window}        # smart(srt legacy)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--chunker", required=True, choices=list(CHUNKERS) + ["all"])
    ap.add_argument("--docs", help="쉼표로 구분한 doc_id만 처리")
    ap.add_argument("--limit", type=int, help="매니페스트 앞에서 N개 문서만")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--target-tokens", type=int, help="fixed: 목표 토큰 수(Gemma 토크나이저 기준 글자/토큰 비율로 환산)")
    ap.add_argument("--chunk-size-chars", type=int, help="fixed: 글자 수로 직접 지정(주면 --target-tokens보다 우선)")
    ap.add_argument("--overlap", type=float, help="fixed: 겹침 비율. srt에서는 0으로 둘 것")
    ap.add_argument("--percentile", type=float, help="semantic: 의미 거리 percentile")
    ap.add_argument("--min-tokens", type=int, help="semantic·smart")
    ap.add_argument("--max-tokens", type=int, help="semantic·smart")
    ap.add_argument("--buffer-size", type=int, help="semantic: 이웃 결합 단위 수(기본 0, 1 = LangChain 기본)")
    ap.add_argument("--window", type=int, help="smart(srt legacy): 경계 점수 창")
    return ap


def main(argv=None):
    ns = build_parser().parse_args(argv)
    chunkers = list(CHUNKERS) if ns.chunker == "all" else [ns.chunker]
    models = get_models(DATASET, need_embeddings=any(c != "fixed" for c in chunkers))
    ids = [d.strip() for d in ns.docs.split(",")] if ns.docs else None
    for ch in chunkers:
        chunk_dataset(DATASET, ch, models, overrides=overrides_for(ch, ns), doc_ids=ids, limit=ns.limit, force=ns.force)


if __name__ == "__main__":
    main()
