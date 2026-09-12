"""자막 번역 실험: EN/KO 자막 세트(같은 릴리즈, 타임스탬프 정합성 검증됨) 양방향
번역, fixed vs semantic 비교. 큐 단위(Jaccard 시간겹침 정렬) chrF/BLEU 사용."""
import sys
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import SRT_KOR_DIR, SRT_ENG_DIR, RESULTS_DIR, EMBED_MODEL, TOKENIZER, SUBTITLE_DATASETS
from preprocessing.loader import load_srt
from chunking.defaults import default_subtitle_chunkers
from llm.client import load_llm
from eval.subtitle_translate import run_translation

RESULTS_DIR.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"실행 디바이스: {device.upper()}")
embed_model = SentenceTransformer(EMBED_MODEL, device=device)
tokenizer, llm_model, device = load_llm(TOKENIZER, device)
print("모델 준비 완료")

CHUNKERS = default_subtitle_chunkers(embed_model)

report_lines = ["자막 번역 비교 결과 (큐 단위 chrF/BLEU)", "=" * 60]
for name, (en_name, ko_name) in SUBTITLE_DATASETS.items():
    en_full = load_srt(str(SRT_ENG_DIR / en_name))
    ko_full = load_srt(str(SRT_KOR_DIR / ko_name))

    for direction, src_doc, ref_doc in [("ko2en", ko_full, en_full), ("en2ko", en_full, ko_full)]:
        results = run_translation(direction, src_doc, ref_doc, CHUNKERS, tokenizer, llm_model, device, RESULTS_DIR)
        for method_label, r in results.items():
            ts_str = "OK" if r["ts_ok"] else f"문제 {len(r['ts_bad'])}건"
            score_str = f"chrF={r['chrf']:.2f} BLEU={r['bleu']:.2f}" if r["chrf"] is not None else "N/A"
            report_lines.append(
                f"[{name}/{direction}] {method_label:9s} {score_str}  타임스탬프보존={ts_str}"
            )

report_path = RESULTS_DIR / "자막_번역_비교결과.txt"
report_path.write_text("\n".join(report_lines), encoding="utf-8")
print(f"\n비교 리포트 저장: {report_path}")
