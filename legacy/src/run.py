"""통합 실행 진입점 - 트랙별 pipeline.py를 얇게 호출만 한다(모델 로딩·청커 정의·
데이터셋별 로직은 전부 각 트랙 pipeline.py에 있음).
    python run.py --dataset {docx,pdf,srt} --task {retrieval,summary,translate}
                   --chunker {fixed,semantic,smart,all} [--limit N] [--fake-llm]
결과는 results/<dataset>_<task>_<chunker>.json(retrieval/summary) 또는
results/translate_pairs/<dataset>_<chunker>_<문서명>.json(translate)에 저장 - 이미
있으면 그 조합/문서는 건너뛴다(이어하기, docx/pdf). srt는 트랙 자체 리포트
(results/자막_번역_비교결과.txt)에 저장한다."""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from config import RESULTS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["docx", "pdf", "srt"])
    ap.add_argument("--task", required=True, choices=["retrieval", "summary", "translate"])
    ap.add_argument("--chunker", required=True, choices=["fixed", "semantic", "smart", "all"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fake-llm", action="store_true")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    if args.dataset == "docx":
        import docx_track.pipeline as pipeline
    elif args.dataset == "pdf":
        import pdf_track.pipeline as pipeline
    else:
        import srt.pipeline as pipeline

    print(f"\n=== {args.dataset} / {args.task} / {args.chunker} ===", flush=True)
    pipeline.run(args.task, args.chunker, limit=args.limit, fake_llm=args.fake_llm)


if __name__ == "__main__":
    main()
