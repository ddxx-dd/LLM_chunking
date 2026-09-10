"""자막 번역 실험: EN/KO 자막 세트(같은 릴리즈, 타임스탬프 정합성 검증됨) 양방향
번역, fixed vs semantic 비교. 큐 단위(Jaccard 시간겹침 정렬) F1 사용."""
import sys
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import SRT_KOR_DIR, SRT_ENG_DIR, RESULTS_DIR, EMBED_MODEL, TOKENIZER
from preprocessing.loader import load_srt
from chunking.fixed_chunker import fixed_chunking
from chunking.semantic_chunker import semantic_chunking
from llm.client import load_llm
from eval.subtitle_translate import run_translation

RESULTS_DIR.mkdir(exist_ok=True)

# 타임스탬프 정합성 실측 검증된 EN/KO 세트 (오정렬률 5~8% 수준, data/README 격
# 역할 - 새 영화 추가 시 여기만 늘리면 됨)
DATASETS = {
    "TrumanShow": ("The_Truman_Show_Eng.srt", "트루먼쇼_PerfectionHD.srt"),
    "AboutTime": ("About_Time_Eng.srt", "어바웃타임.srt"),
    "Interstellar": ("Interstellar_Eng.srt", "인터스텔라.srt"),
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"실행 디바이스: {device.upper()}")
embed_model = SentenceTransformer(EMBED_MODEL, device=device)
tokenizer, llm_model, device = load_llm(TOKENIZER, device)
print("모델 준비 완료")

CHUNKERS = {
    "fixed": lambda text: fixed_chunking(text, chunk_size=120),
    "semantic": lambda text: semantic_chunking(text, embed_model, method="percentile", amount=15),
}

report_lines = ["자막 번역 비교 결과 (큐 단위 F1)", "=" * 60]
for name, (en_name, ko_name) in DATASETS.items():
    en_full = load_srt(str(SRT_ENG_DIR / en_name))
    ko_full = load_srt(str(SRT_KOR_DIR / ko_name))

    for direction, src_doc, ref_doc in [("ko2en", ko_full, en_full), ("en2ko", en_full, ko_full)]:
        results = run_translation(direction, src_doc, ref_doc, CHUNKERS, tokenizer, llm_model, device, RESULTS_DIR)
        for method_label, r in results.items():
            ts_str = "OK" if r["ts_ok"] else f"문제 {len(r['ts_bad'])}건"
            f1_str = f"{r['f1']:.4f}" if r["f1"] is not None else "N/A"
            report_lines.append(
                f"[{name}/{direction}] {method_label:9s} F1={f1_str}  타임스탬프보존={ts_str}"
            )

report_path = RESULTS_DIR / "자막_번역_비교결과.txt"
report_path.write_text("\n".join(report_lines), encoding="utf-8")
print(f"\n비교 리포트 저장: {report_path}")
