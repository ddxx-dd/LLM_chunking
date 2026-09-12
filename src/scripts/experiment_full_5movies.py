"""5개 신규 영화 전체 - fixed(500)/semantic(min128,max1024) x en2ko/ko2en 전체 번역
+ chrF/BLEU 채점. About Time(구 데이터셋) 결과와 비교 가능하도록 동일 방식."""
import sys
import time
import pickle
import sacrebleu

sys.path.append("/root/jupyter/LLM_chunking/src")
from config import SRT_ENG_DIR, SRT_KOR_DIR, EMBED_MODEL, TOKENIZER
from preprocessing.loader import load_srt
from chunking.fixed_chunker import fixed_chunking
from chunking.semantic_chunker import semantic_chunking
from llm.client import load_llm
from eval.timestamp_align import align_by_overlap
from eval.subtitle_translate import translate_chunks_batch
from pipeline.mapper import merge_to_units
from sentence_transformers import SentenceTransformer

DATASETS = {
    "Noah": ("Noah_Eng.srt", "노아.srt"),
    "Deadpool": ("Deadpool_Eng.srt", "데드풀.srt"),
    "InsidiousChapter2": ("Insidious_Chapter2_Eng.srt", "인시디어스2.srt"),
    "DoctorStrange": ("Doctor_Strange_Eng.srt", "닥터스트레인지.srt"),
    "CaptainAmericaCivilWar": ("Captain_America_Civil_War_Eng.srt", "캡틴아메리카시빌워.srt"),
}

print("모델 로드 중...", flush=True)
embed_model = SentenceTransformer(EMBED_MODEL, device="cuda")
tokenizer, model, device = load_llm(TOKENIZER)
print("모델 준비 완료\n", flush=True)

CHUNKERS = {
    "fixed": lambda text: fixed_chunking(text, chunk_size=500),
    "semantic": lambda text: semantic_chunking(text, embed_model, method="percentile", amount=15,
                                                min_chunk_tokens=128, max_chunk_tokens=1024),
}


def score(unit_texts, aligned_ref):
    chrfs, bleus = [], []
    for p, r in zip(unit_texts, aligned_ref):
        if p.strip() and r.strip():
            chrfs.append(sacrebleu.sentence_chrf(p, [r]).score)
            bleus.append(sacrebleu.sentence_bleu(p, [r]).score)
    n = len(chrfs)
    return (sum(chrfs) / n if n else 0.0), (sum(bleus) / n if n else 0.0), n


all_results = {}
for movie, (en_name, ko_name) in DATASETS.items():
    en_full = load_srt(str(SRT_ENG_DIR / en_name))
    ko_full = load_srt(str(SRT_KOR_DIR / ko_name))
    aligned_ref = {"en2ko": align_by_overlap(en_full, ko_full), "ko2en": align_by_overlap(ko_full, en_full)}

    for method_label, chunker_fn in CHUNKERS.items():
        for direction, src_doc in [("en2ko", en_full), ("ko2en", ko_full)]:
            chunks = chunker_fn(src_doc.text)
            t0 = time.time()
            print(f"[{movie}/{method_label}/{direction}] {len(chunks)}청크 번역 시작...", flush=True)
            pieces = translate_chunks_batch(src_doc, chunks, direction, tokenizer, model, device,
                                             batch_size=8, label=f"{movie}/{method_label}/{direction}")
            unit_texts = merge_to_units(src_doc, pieces)
            chrf, bleu, n = score(unit_texts, aligned_ref[direction])
            elapsed = time.time() - t0
            print(f"[{movie}/{method_label}/{direction}] chrF={chrf:.2f} BLEU={bleu:.2f} (n={n}, {elapsed:.0f}초)", flush=True)
            all_results[(movie, method_label, direction)] = dict(chrf=chrf, bleu=bleu, n=n, n_chunks=len(chunks))

with open("/tmp/claude-0/-root-jupyter-LLM-chunking/9a24a7b2-903a-4e97-81f3-4032fdcdabeb/scratchpad/full_5movies_results.pkl", "wb") as f:
    pickle.dump(all_results, f)

print("\n" + "=" * 90)
print("전체 결과 요약")
print("=" * 90)
for (movie, method, direction), r in all_results.items():
    print(f"[{movie:24s}/{method:8s}/{direction}] chrF={r['chrf']:6.2f} BLEU={r['bleu']:6.2f} (청크{r['n_chunks']})")

print("\n" + "=" * 90)
print("fixed vs semantic 종합 평균")
print("=" * 90)
for method in ["fixed", "semantic"]:
    for direction in ["en2ko", "ko2en"]:
        vals = [r for (m, me, d), r in all_results.items() if me == method and d == direction]
        avg_chrf = sum(v["chrf"] for v in vals) / len(vals)
        avg_bleu = sum(v["bleu"] for v in vals) / len(vals)
        print(f"{method}/{direction}: 평균 chrF={avg_chrf:.2f}, 평균 BLEU={avg_bleu:.2f} (5개 영화 평균)")

print("\nDONE")
