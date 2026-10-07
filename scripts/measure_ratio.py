"""글자/토큰 비율 측정 (글자수 청커의 T(목표 토큰) -> 글자수 환산용).

  source env.sh
  .venv311/bin/python scripts/measure_ratio.py --dataset vectara [--T 260]      # 또는 all
결과: data/manifests/<dataset>.ratio.json  (전처리가 끝난 뒤에 실행. 청킹에 쓴 문서와 같은 문서 집합으로 재측정할 것)
chars_per_token = 글자 합 / 토큰 합 (문서별 비율의 평균이 아니라 합계 비율). pdf·docx는 bge-m3 토크나이저, srt는 Gemma 토크나이저(옛 v1 경로).
--T를 주면 "T토큰이 몇 글자인지"를 언어별로 보여 준다.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common.config import DATASETS  # noqa: E402
from preprocess.datasets import load_entries  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS) + ["all"])
    ap.add_argument("--T", type=int, help="예시로 보여 줄 목표 토큰 수")
    args = ap.parse_args(argv)
    for ds in (sorted(DATASETS) if args.dataset == "all" else [args.dataset]):
        if DATASETS[ds]["fmt"] == "srt":
            from chunking.v1.ratio import measure_ratio, ratio_path, save_ratio
            r = measure_ratio(ds)
        else:
            from chunking.embed import make_count_many
            from chunking.io import measure_ratio, ratio_path, save_ratio
            r = measure_ratio(ds, make_count_many("bge-m3"))
        measured, expected = sum(v["n_docs"] for v in r["by_lang"].values()), len(load_entries(ds))
        if measured != expected:       # 전처리 결과가 빠졌거나 문서를 잘못 걸러낸 경우
            sys.exit(f"[{ds}] 측정한 문서 {measured}개 != 매니페스트 {expected}개 - 전처리 결과를 확인할 것")
        save_ratio(r)
        print(f"[{ds}] -> {ratio_path(ds)}")
        for lang, v in r["by_lang"].items():
            ex = f" → {args.T}토큰 ≈ {round(args.T * v['chars_per_token'])}자" if args.T else ""
            print(f"  {lang}: {v['chars_per_token']} 글자/토큰 (문서 {v['n_docs']}개){ex}")


if __name__ == "__main__":
    main()
