"""CometKiwi(참조 없는 번역 품질 채점)로 (원문, 번역) 쌍을 채점한다 - .venv-eval 전용
(unbabel-comet은 .venv311의 transformers 버전과 충돌해서 별도 프로세스로 완전히
분리해야 함). translate.py가 results/에 저장한 pairs json([{"start","end","src","mt"}])
을 읽어서 블록 단위로 채점한다.

사용법: .venv-eval/bin/python comet_score.py <pairs.json> [--out out.json]
"""
import argparse
import json
from pathlib import Path

MODEL_NAME = "Unbabel/wmt22-cometkiwi-da"


def score_pairs(pairs, batch_size=16, gpus=1):
    from comet import download_model, load_from_checkpoint
    model = load_from_checkpoint(download_model(MODEL_NAME))
    data = [{"src": p["src"], "mt": p["mt"]} for p in pairs if p["mt"].strip()]
    if not data:
        return [], None
    output = model.predict(data, batch_size=batch_size, gpus=gpus)
    return list(output.scores), output.system_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs_path")
    ap.add_argument("--out")
    ap.add_argument("--gpus", type=int, default=1)
    args = ap.parse_args()

    pairs = json.loads(Path(args.pairs_path).read_text(encoding="utf-8"))
    print(f"채점 대상 {len(pairs)}개 블록", flush=True)
    scores, overall = score_pairs(pairs, gpus=args.gpus)
    scored = [p for p in pairs if p["mt"].strip()]
    for p, s in zip(scored, scores):
        p["cometkiwi"] = s

    print(f"\n평균 CometKiwi: {overall:.4f} (n={len(scores)})")
    if args.out:
        Path(args.out).write_text(json.dumps({"overall": overall, "n": len(scores), "pairs": pairs},
                                              ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
