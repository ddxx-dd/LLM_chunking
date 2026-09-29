"""자막 트랙 - EN/KO 자막 세트(같은 릴리즈, 타임스탬프 정합성 검증됨) 양방향 번역,
fixed vs semantic vs smart 비교. 큐 단위(Jaccard 시간겹침 정렬) chrF/BLEU/BERTScore.
전체 영화를 돈 뒤 방향별 종합 평균까지 집계한다(subtitle_pipeline.py에서 이름만 바뀜).

CHUNKERS 값은 기존 그대로 유지: fixed 500자(strip_whitespace=False), semantic/smart는
min 128 / max 1024·512 토큰(문서 트랙의 min100/max400보다 큼 - 자막 큐가 짧아서
min을 낮게 잡으면 청크가 너무 잘게 쪼개지는 걸 실측으로 이미 확인한 값)."""
import chunkers
from config import RESULTS_DIR, SRT_ENG_DIR, SRT_KOR_DIR, SUBTITLE_DATASETS
from srt.loader import load_srt
from srt.mapper import build_prompt, merge_to_units, parse_marked, write_srt
from srt.subtitle_translate import check_timestamp_integrity, run_translation


def build_chunkers(embeddings, count_tokens, lang):
    """lang: 이번 방향의 원문 언어("en"/"ko") - smart의 cue_continues()가 언어별로
    다른 휴리스틱을 쓰므로 방향마다 다시 만들어야 한다(호출부인 run()의 방향 루프 참고)."""
    return {
        "fixed":    chunkers.make_fixed(chunk_size=500, strip_whitespace=False),
        "semantic": chunkers.make_semantic(embeddings, count_tokens, percentile=90,
                                            min_tokens=128, max_tokens=1024),
        "smart":    chunkers.make_smart(embeddings._client, count_tokens, mode="srt", lang=lang,
                                         min_tokens=128, max_tokens=512),
    }


def _setup(fake_llm):
    """srt는 count_tokens에 Gemma 토크나이저를 쓴다(기존 값 그대로 유지) - docx/pdf
    트랙의 count_tokens(bge-m3 토크나이저)와 다르다. fake_llm이면 모델 가중치는 안
    불러오고 토크나이저만 불러서(가벼움) 청킹(=토큰 수 계산)은 그대로 동작하게 한다."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from transformers import AutoTokenizer

    from config import TOKENIZER

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", encode_kwargs={"normalize_embeddings": True})
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    count_tokens = lambda s: len(tokenizer.encode(s, add_special_tokens=False))

    model = device = None
    if not fake_llm:
        import torch
        from llm import load_llm
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _, model, device = load_llm(TOKENIZER, device)
    return embeddings, tokenizer, count_tokens, model, device


def _fake_translate(doc, chunks):
    """--fake-llm용 - Gemma 대신 청크와 겹친 큐 조각을 그대로 돌려준다(끊긴 호출
    0건, docx/pdf 트랙의 translate.fake_translate_batch와 같은 목적). build_prompt가
    이미 "[n] 원문" 형태로 만들어주므로 parse_marked에 그대로 다시 먹이면 원문이
    그대로 복원된다."""
    pieces = []
    for c in chunks:
        frs, body = build_prompt(doc, c, instruction="{body}")
        pieces.extend(parse_marked(body, frs))
    return merge_to_units(doc, pieces)


def run(task, chunker, limit=None, fake_llm=False):
    """run.py가 부르는 진입점. srt 트랙은 검색/요약 태스크가 없다(항상 번역+채점
    하나뿐) - task는 다른 트랙과 인터페이스를 맞추기 위해 받되, translate가 아니면
    안내만 하고 끝낸다. chunker="all"(기본)이면 fixed/semantic/smart 전부, limit은
    SUBTITLE_DATASETS(영화 5편) 중 앞에서 몇 개만 도는지를 뜻한다."""
    if task != "translate":
        print("srt 트랙은 translate만 지원합니다(검색/요약 태스크 없음).")
        return None

    RESULTS_DIR.mkdir(exist_ok=True)
    embeddings, tokenizer, count_tokens, model, device = _setup(fake_llm)

    movies = list(SUBTITLE_DATASETS.items())[:limit] if limit else list(SUBTITLE_DATASETS.items())
    methods = ["fixed", "semantic", "smart"] if chunker in (None, "all") else [chunker]

    report_lines = ["자막 번역 비교 결과 (큐 단위 chrF/BLEU/BERTScore)", "=" * 60]
    all_results = {}
    for name, (en_name, ko_name) in movies:
        en_full = load_srt(str(SRT_ENG_DIR / en_name))
        ko_full = load_srt(str(SRT_KOR_DIR / ko_name))

        for direction, src_doc, ref_doc in [("ko2en", ko_full, en_full), ("en2ko", en_full, ko_full)]:
            all_chunkers = build_chunkers(embeddings, count_tokens, lang=direction[:2])
            active_chunkers = {m: all_chunkers[m] for m in methods}

            if fake_llm:
                for method_label, splitter in active_chunkers.items():
                    chunks = splitter.split_documents([src_doc])
                    unit_texts = _fake_translate(src_doc, chunks)
                    src_name = src_doc.metadata["name"]
                    out_path = RESULTS_DIR / f"{src_name.rsplit('.', 1)[0]}_{direction}_{method_label}.srt"
                    write_srt(src_doc, unit_texts, str(out_path))
                    ts_ok, ts_bad = check_timestamp_integrity(out_path, src_doc)
                    all_results[(name, method_label, direction)] = dict(
                        out_path=out_path, chrf=None, bleu=None, bert_f1=None, n=None, ts_ok=ts_ok, ts_bad=ts_bad)
                    print(f"[fake-llm] [{name}/{direction}] {method_label}: 청크 {len(chunks)}개, "
                          f"타임스탬프보존={'OK' if ts_ok else f'문제 {len(ts_bad)}건'}")
                continue

            results = run_translation(direction, src_doc, ref_doc, active_chunkers, tokenizer, model, device, RESULTS_DIR)
            for method_label, r in results.items():
                all_results[(name, method_label, direction)] = r
                ts_str = "OK" if r["ts_ok"] else f"문제 {len(r['ts_bad'])}건"
                score_str = (f"chrF={r['chrf']:.2f} BLEU={r['bleu']:.2f} BERTScore={r['bert_f1']:.2f}"
                             if r["chrf"] is not None else "N/A")
                report_lines.append(f"[{name}/{direction}] {method_label:9s} {score_str}  타임스탬프보존={ts_str}")

    if fake_llm:
        return all_results

    # 영화 전체에 대한 방법별 종합 평균 (구 experiment_full_5movies.py의 유일한 고유
    # 기능 - 나머지 로직은 run_translation()과 중복이라 흡수하며 정리함).
    report_lines += ["", "fixed vs semantic vs smart 종합 평균 (전체 영화)", "=" * 60]
    for method in methods:
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
    return all_results


if __name__ == "__main__":
    run("translate", "all")
