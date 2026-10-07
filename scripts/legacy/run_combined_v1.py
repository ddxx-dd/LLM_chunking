"""세 데이터셋(vectara, allganize, docx_corpus)을 한 인덱스에 합쳐서 평가한다 (전체 코퍼스 검색 설정, 청킹·임베딩은 새로 하지 않는다).

  source env.sh && .venv311/bin/python scripts/legacy/run_combined_v1.py [--k 3] [--tags fixed__c500,semantic__...,smart__...]

- 데이터셋별로 저장된 인덱스(data/index/<ds>/<tag>)의 벡터를 이어 붙여 합친 인덱스를 만들고, 질문은 데이터셋별로 평가한다(지표 정의는 tasks/retrieval).
- 결과: results/combined/<ds>/<태그>.json, results/combined/summary.json
- 언어 진단: 질문마다 "정답 문서가 아닌 청크 중 가장 비슷한 것"의 유사도(같은 언어 코퍼스 vs 다른 언어 코퍼스)와
  top-k가 다른 데이터셋·다른 언어에서 오는 비율을 센다.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common.config import RESULTS_DIR  # noqa: E402
from common.models import get_embeddings  # noqa: E402
from indexing.index import combine_indexes, get_index, load_chunks, make_encoder  # noqa: E402
from qa_build.build_qa import load_qa  # noqa: E402
from tasks.retrieval import PRIMARY, evaluate  # noqa: E402

DATASETS = ["vectara", "allganize", "docx_corpus"]
LANG = {"vectara": "en", "allganize": "ko", "docx_corpus": "en"}
TAGS = ["fixed__c500", "semantic__p85-min128-max512", "smart__p85-min128-max512"]
MAX_SEQ = 8192
OUT = RESULTS_DIR / "combined"


def best_non_gold(index, queries, gold_docs, k=60):
    """질문마다 정답 문서가 아닌 청크 중 가장 높은 유사도(없으면 NaN)."""
    out = []
    for hits, g in zip(index.search(queries, k), gold_docs):
        sc = [s for c, s in hits if c["doc_id"] != g]
        out.append(sc[0] if sc else float("nan"))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--tags", default=",".join(TAGS))
    args = ap.parse_args()
    encode = make_encoder(get_embeddings()._client)
    qas = {ds: load_qa(ds) for ds in DATASETS}
    summary = {}
    for tag in args.tags.split(","):
        per = {}
        for ds in DATASETS:
            docs = sorted({c["doc_id"] for c in load_chunks(ds, tag)})
            per[ds] = get_index(ds, tag, encode, doc_ids=docs, max_seq_length=MAX_SEQ)
        comb = combine_indexes([per[d] for d in DATASETS], encode)
        owner = {c["chunk_id"]: ds for ds in DATASETS for c in per[ds].chunks}
        en_idx = combine_indexes([per[d] for d in DATASETS if LANG[d] == "en"], encode)
        ko_idx = combine_indexes([per[d] for d in DATASETS if LANG[d] == "ko"], encode)
        print(f"\n[{tag}] 합친 인덱스: 청크 {len(comb.chunks)}개, 문서 {len({c['doc_id'] for c in comb.chunks})}개")
        summary[tag] = {}
        for ds in DATASETS:
            res_c = evaluate(comb, qas[ds], k=args.k)
            res_s = evaluate(per[ds], qas[ds], k=args.k)                      # 비교용: 데이터셋 하나만 넣은 인덱스(실험 1과 같은 설정)
            queries = [q["question"] for q in qas[ds]]
            hits = comb.search(queries, args.k)
            cross_ds = np.mean([sum(owner[c["chunk_id"]] != ds for c, _ in h) / max(len(h), 1) for h in hits])
            cross_lang = np.mean([sum(LANG[owner[c["chunk_id"]]] != LANG[ds] for c, _ in h) / max(len(h), 1) for h in hits])
            gold = [q["doc_id"] for q in qas[ds]]
            same = en_idx if LANG[ds] == "en" else ko_idx
            other = ko_idx if LANG[ds] == "en" else en_idx
            floor_same = float(np.nanmean(best_non_gold(same, queries, gold)))
            floor_other = float(np.nanmean(best_non_gold(other, queries, gold)))
            top1_gold = float(np.mean([h[0][1] for h in hits if h]))
            d = (OUT / ds); d.mkdir(parents=True, exist_ok=True)
            res_c.update(dataset=ds, tag=tag, mode="combined", n_questions=len(qas[ds]))
            (d / f"{tag}.json").write_text(json.dumps(res_c, ensure_ascii=False, indent=1), encoding="utf-8")
            prim = PRIMARY.get(ds)
            pk = (prim + "@k") if prim else None
            row = {"alone": {k: v for k, v in res_s["overall"].items()}, "combined": {k: v for k, v in res_c["overall"].items()},
                   "cross_dataset_share": float(cross_ds), "cross_language_share": float(cross_lang),
                   "top1_score_mean": top1_gold, "best_nongold_same_language": floor_same, "best_nongold_other_language": floor_other}
            summary[tag][ds] = row
            a, c = res_s["overall"], res_c["overall"]
            print(f"  [{ds}] hit@{args.k} {a['hit@k']:.3f} → {c['hit@k']:.3f} | MRR {a['mrr']:.3f} → {c['mrr']:.3f}"
                  + (f" | {prim}@{args.k} {a[pk]:.3f} → {c[pk]:.3f}" if pk else "")
                  + f" | 다른 데이터셋 청크 비율 {cross_ds:.3f} (다른 언어 {cross_lang:.3f})"
                  + f" | 정답 아닌 최고 유사도: 같은 언어 {floor_same:.3f} / 다른 언어 {floor_other:.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[저장] {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
