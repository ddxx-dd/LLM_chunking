"""fixed 글자 수 환산용 글자/토큰 비율 측정 (docs/restructure_plan.md 5.4.1절).

  source env.sh
  .venv311/bin/python scripts/measure_ratio.py --dataset vectara      # 또는 all
결과: data/manifests/<dataset>.ratio.json  (전처리가 끝난 뒤에 실행)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chunking.ratio import fixed_chunk_size, measure_ratio, ratio_path, save_ratio  # noqa: E402
from common.config import DATASETS  # noqa: E402
from preprocess.datasets import load_entries  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS) + ["all"])
    args = ap.parse_args()
    for ds in (sorted(DATASETS) if args.dataset == "all" else [args.dataset]):
        r = measure_ratio(ds)
        measured, expected = sum(v["n_docs"] for v in r["by_lang"].values()), len(load_entries(ds))
        if measured != expected:       # 전처리 결과가 빠졌거나 문서를 잘못 걸러낸 경우
            sys.exit(f"[{ds}] 측정한 문서 {measured}개 != 매니페스트 {expected}개 - 전처리 결과를 확인할 것")
        save_ratio(r)
        print(f"[{ds}] 토크나이저={r['counter']} → {ratio_path(ds)}")
        target = DATASETS[ds]["chunkers"]["fixed"]["target_tokens"]
        for lang, v in r["by_lang"].items():
            size, _ = fixed_chunk_size(ds, lang, target)
            print(f"  {lang}: {v['chars_per_token']} 글자/토큰 (문서 {v['n_docs']}개, 문서별 {v['per_doc_min']}~{v['per_doc_max']}, "
                  f"중앙 {v['per_doc_p50']}) → 목표 {target}토큰 ≈ {size}자")


if __name__ == "__main__":
    main()
