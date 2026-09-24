"""자막 번역 실험: EN/KO 자막 세트(같은 릴리즈, 타임스탬프 정합성 검증됨) 양방향
번역, fixed vs semantic 비교. 큐 단위(Jaccard 시간겹침 정렬) chrF/BLEU/BERTScore 사용.
전체 영화를 돈 뒤 fixed vs semantic 방향별 종합 평균까지 집계한다."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import SRT_KOR_DIR, SRT_ENG_DIR, RESULTS_DIR, SUBTITLE_DATASETS
from srt.loader import load_srt
from srt.splitters import make_fixed_splitter, SemanticTextSplitter
from srt.subtitle_translate import run_translation
from common import setup_models


def default_subtitle_chunkers(embed_model):
    """자막 실험이 쓰는 fixed/semantic 기본 설정(실측 검증된 값)."""
    return {
        "fixed": make_fixed_splitter(chunk_size=500),
        "semantic": SemanticTextSplitter(embed_model, method="percentile", amount=15,
                                          min_chunk_tokens=128, max_chunk_tokens=1024),
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    device, embed_model, tokenizer, llm_model = setup_models()
    chunkers = default_subtitle_chunkers(embed_model)

    report_lines = ["자막 번역 비교 결과 (큐 단위 chrF/BLEU/BERTScore)", "=" * 60]
    all_results = {}  # (movie, method, direction) -> run_translation()의 결과 dict
    for name, (en_name, ko_name) in SUBTITLE_DATASETS.items():
        en_full = load_srt(str(SRT_ENG_DIR / en_name))
        ko_full = load_srt(str(SRT_KOR_DIR / ko_name))

        for direction, src_doc, ref_doc in [("ko2en", ko_full, en_full), ("en2ko", en_full, ko_full)]:
            results = run_translation(direction, src_doc, ref_doc, chunkers, tokenizer, llm_model, device, RESULTS_DIR)
            for method_label, r in results.items():
                all_results[(name, method_label, direction)] = r
                ts_str = "OK" if r["ts_ok"] else f"문제 {len(r['ts_bad'])}건"
                score_str = (f"chrF={r['chrf']:.2f} BLEU={r['bleu']:.2f} BERTScore={r['bert_f1']:.2f}"
                             if r["chrf"] is not None else "N/A")
                report_lines.append(f"[{name}/{direction}] {method_label:9s} {score_str}  타임스탬프보존={ts_str}")

    # 영화 전체에 대한 fixed vs semantic 종합 평균 (구 experiment_full_5movies.py의
    # 유일한 고유 기능 - 나머지 로직은 run_translation()과 중복이라 흡수하며 정리함).
    report_lines += ["", "fixed vs semantic 종합 평균 (전체 영화)", "=" * 60]
    for method in chunkers:
        for direction in ["en2ko", "ko2en"]:
            vals = [r for (_, me, d), r in all_results.items() if me == method and d == direction and r["chrf"] is not None]
            if not vals:
                continue
            avg_chrf = sum(v["chrf"] for v in vals) / len(vals)
            avg_bleu = sum(v["bleu"] for v in vals) / len(vals)
            avg_bert = sum(v["bert_f1"] for v in vals) / len(vals)
            line = (f"{method}/{direction}: 평균 chrF={avg_chrf:.2f}, 평균 BLEU={avg_bleu:.2f}, "
                    f"평균 BERTScore={avg_bert:.2f} ({len(vals)}개 영화 평균)")
            print(line)
            report_lines.append(line)

    report_path = RESULTS_DIR / "자막_번역_비교결과.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n비교 리포트 저장: {report_path}")


if __name__ == "__main__":
    main()
