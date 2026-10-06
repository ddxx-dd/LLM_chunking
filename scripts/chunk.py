"""공통 JSON -> 청크 파일 (docs/restructure_plan.md 5절).

  source env.sh
  .venv311/bin/python scripts/chunk.py --dataset vectara --chunker smart [--docs id1,id2] [--limit N] [--force]
      [--target-tokens N --overlap R] [--percentile P] [--min-tokens N --max-tokens N] [--window W]

- 결과: data/chunks/<dataset>/<청커>__<설정>.jsonl (+ .meta.json). 파일이 있으면 건너뛴다(--force로 다시).
- 크기 파라미터(min/max 토큰, 목표 토큰, percentile, window)를 안 주면 common/config.py의 시작값을 쓴다.
  **실제 문서에 돌리기 전에는 사용자가 정한 값을 명시해서 실행할 것** - 쓰는 설정은 실행 첫 줄에 그대로 출력된다.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chunking.runner import CHUNKERS, chunk_dataset  # noqa: E402
from common.config import DATASETS  # noqa: E402
from common.models import get_models  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--chunker", required=True, choices=list(CHUNKERS) + ["all"])
    ap.add_argument("--docs", help="쉼표로 구분한 doc_id만 처리")
    ap.add_argument("--limit", type=int, help="매니페스트 앞에서 N개 문서만")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--target-tokens", type=int, help="fixed: 목표 토큰 수 (글자 수는 측정한 비율로 환산)")
    ap.add_argument("--chunk-size-chars", type=int, help="fixed: 글자 수로 직접 지정 (주면 --target-tokens보다 우선)")
    ap.add_argument("--overlap", type=float, help="fixed: 겹침 비율 (예: 0.1). srt에서는 0으로 둘 것")
    ap.add_argument("--percentile", type=float, help="semantic·smart(constrained): 의미 거리 percentile")
    ap.add_argument("--min-tokens", type=int, help="semantic·smart")
    ap.add_argument("--max-tokens", type=int, help="semantic·smart")
    ap.add_argument("--window", type=int, help="smart")
    args = ap.parse_args()

    chunkers = list(CHUNKERS) if args.chunker == "all" else [args.chunker]
    models = get_models(args.dataset, need_embeddings=any(c != "fixed" for c in chunkers))
    per_chunker = {
        "fixed": {"target_tokens": args.target_tokens, "chunk_size_chars": args.chunk_size_chars, "overlap_ratio": args.overlap},
        "semantic": {"percentile": args.percentile, "min_tokens": args.min_tokens, "max_tokens": args.max_tokens},
        "smart": {"min_tokens": args.min_tokens, "max_tokens": args.max_tokens, "window": args.window, "percentile": args.percentile},
    }
    ids = [d.strip() for d in args.docs.split(",")] if args.docs else None
    for ch in chunkers:
        chunk_dataset(args.dataset, ch, models, overrides=per_chunker[ch], doc_ids=ids, limit=args.limit, force=args.force)


if __name__ == "__main__":
    main()
