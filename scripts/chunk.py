"""공통 JSON -> 청크 파일 (pdf·docx 청커 7종). srt는 scripts/srt/chunk.py.

  source env.sh
  .venv311/bin/python scripts/chunk.py --dataset vectara --chunker A --p 90
  .venv311/bin/python scripts/chunk.py --dataset docx_corpus --chunker fixed --T 260
  .venv311/bin/python scripts/chunk.py --dataset vectara --chunker all --p 90 --T 260 [--docs id1,id2] [--limit N] [--force]

- 청커: fixed, recursive, header, semantic, A, B, E (src/chunking/README.md). 크기 조절 값은 p(백분위) 또는 T(목표 토큰) 또는 --size-chars(글자).
  **값에는 기본값이 없다**: 실제 문서에 돌리기 전에 사용자가 정한 값을 명시해서 실행할 것.
- 결과: data/chunks/<dataset>/<태그>.jsonl (+ .meta.json). 태그 예: A__p90, fixed__T260, B__p90-beta0. 파일이 있으면 건너뛴다(--force로 다시).
- 의미 분할 계열의 단위 임베딩은 data/cache/units/ 에 캐시한다(같은 문서를 다른 p로 다시 돌릴 때 임베딩을 다시 하지 않음, --no-cache로 끔).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chunking import CHUNKERS, REGISTRY  # noqa: E402
from common.config import DATASETS  # noqa: E402

# 청커마다 받는 값(chunker.chunk의 인자 이름)
ALLOWED = {"fixed": ("T", "size_chars"), "recursive": ("T", "size_chars"), "header": ("T", "size_chars", "max_level"),
           "semantic": ("p",), "A": ("p",), "B": ("p", "beta", "decay", "w"), "E": ("p", "delta", "delta_sigma", "decay")}


def params_for(chunker, ns):
    """argparse 결과 -> 그 청커에 넘길 값 dict. 필수 값(크기 조절 값)이 없으면 오류."""
    p = {k: getattr(ns, k) for k in ALLOWED[chunker] if getattr(ns, k, None) is not None}
    knob = REGISTRY[chunker].knob
    if chunker == "header":
        return p                                           # 값이 없으면 재분할 없는 header
    if knob == "p" and "p" not in p:
        raise SystemExit(f"{chunker}에는 --p(백분위)가 필요하다")
    if knob == "T" and not ({"T", "size_chars"} & set(p)):
        raise SystemExit(f"{chunker}에는 --T(목표 토큰) 또는 --size-chars(글자)가 필요하다")
    return p


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, choices=sorted(k for k, v in DATASETS.items() if v["fmt"] != "srt"))
    ap.add_argument("--chunker", required=True, choices=list(CHUNKERS) + ["all"])
    ap.add_argument("--p", type=float, help="semantic·A·B·E: 백분위(클수록 청크가 커진다)")
    ap.add_argument("--T", type=int, help="fixed·recursive·header: 목표 토큰(글자 = T x 글자/토큰 비율, scripts/measure_ratio.py 먼저)")
    ap.add_argument("--size-chars", dest="size_chars", type=int, help="fixed·recursive·header: 글자 수로 직접(--T보다 우선)")
    ap.add_argument("--max-level", dest="max_level", type=int, help="header: 나눌 제목 수준(기본 3)")
    ap.add_argument("--beta", type=float, help="B: 제목 앞 보너스 배수(기본 1)")
    ap.add_argument("--decay", type=float, help="B·E: 제목 수준 감쇠(기본 0.5)")
    ap.add_argument("--w", type=int, help="B: 경계 개선량을 재는 좌우 문장 수(기본 5)")
    ap.add_argument("--delta", type=float, help="E: 제목 경계 감점 절대값(주면 --delta-sigma보다 우선)")
    ap.add_argument("--delta-sigma", dest="delta_sigma", type=float, help="E: 감점 = 초기 이웃 코사인 표준편차의 몇 배(기본 1)")
    ap.add_argument("--docs", help="쉼표로 구분한 doc_id만 처리")
    ap.add_argument("--limit", type=int, help="매니페스트 앞에서 N개 문서만")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-cache", action="store_true", help="단위 임베딩 캐시(data/cache/units)를 쓰지 않는다")
    return ap


def main(argv=None):
    ns = build_parser().parse_args(argv)
    names = list(CHUNKERS) if ns.chunker == "all" else [ns.chunker]
    plan = [(c, params_for(c, ns)) for c in names]            # 값이 빠졌으면 모델을 올리기 전에 여기서 끝낸다
    from chunking import Runtime
    from chunking.io import chunk_dataset

    need_emb = any(REGISTRY[c].units_kind for c in names)
    runtime = Runtime.load(cache_dir=None if ns.no_cache else ROOT / "data" / "cache" / "units", need_embeddings=need_emb)
    ids = [d.strip() for d in ns.docs.split(",")] if ns.docs else None
    for name, params in plan:
        chunk_dataset(ns.dataset, name, params, runtime, doc_ids=ids, limit=ns.limit, force=ns.force)


if __name__ == "__main__":
    main()
