"""크기 제어 비교: 그리디 vs DP(사전식) - 구조 지표만 쓰므로 QA(질문)를 보지 않는다 -> 전체 문서를 써도 평가 누수가 없다.
(DP는 min 우선 사전식이라 튜닝 값이 없다. 이 스크립트는 "그리디보다 나은가"만 본다.)

  source env.sh
  .venv311/bin/python scripts/legacy/compare_size_control.py --datasets vectara allganize docx_corpus --min-tokens 128 --max-tokens 512 --percentile 95

문서마다 임베딩을 한 번만 계산해 캐시하고(텍스트 -> 벡터) 같은 단위·같은 거리로 청커(semantic, smart) x 크기 제어(greedy, dp)를 돌린다.
보고 지표: under_min / over_max (실제 토큰 수), 부족분 합, strong_kept(거리 > 임계값인 경계 중 자른 비율), mean_cut_d(자른 곳의 평균 거리),
sentence_cut_rate, size_cuts(임계값 아래인데 크기 때문에 자른 곳), smart의 섹션 통계(orig_sections, section_merges, cross_section_chunks).
결과: results/compare_size_control/<태그>.json, .md
"""
import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from chunking.v1 import splitters  # noqa: E402
from chunking.sizectl.semantic_core import consecutive_distances  # noqa: E402
from chunking.v1.stats import compute_stats  # noqa: E402
from chunking.v1.units import build_units, caption_glue, rules_for  # noqa: E402
from common.config import PROCESSED_DIR, dataset_cfg  # noqa: E402
from common.docjson import load_doc, render  # noqa: E402
from common.models import get_models  # noqa: E402
from preprocess.datasets import load_entries  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "compare_size_control"


class CachedEmbed:
    """encode(글 목록, normalize_embeddings=True)를 글 단위로 캐시한다(같은 글은 한 번만 임베딩)."""
    def __init__(self, model):
        self.model, self.cache, self.calls = model, {}, 0

    def encode(self, texts, normalize_embeddings=True, **kw):
        miss = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if miss:
            self.calls += len(miss)
            for t, v in zip(miss, self.model.encode(miss, normalize_embeddings=normalize_embeddings, batch_size=64, **kw)):
                self.cache[t] = np.asarray(v, dtype=np.float32)
        return np.stack([self.cache[t] for t in texts])


def cut_quality(doc, text, pieces, chunker, cfg, embed, spans):
    """같은 단위·거리로 계산한 (자른 경계 번호 집합, 거리, 임계값)에서 의미 경계 보존 지표를 낸다."""
    rules = rules_for(chunker, cfg)
    units, forced = build_units(text, pieces, caption_glue(doc, pieces), rules)
    if len(units) < 2:
        return None
    d = np.array(consecutive_distances(embed, [text[s:e] for s, e in units], cfg.get("buffer_size", 0)))
    tau = float(np.percentile(d, cfg["percentile"]))
    start_of = {u[0]: i for i, u in enumerate(units)}
    cuts = [start_of[s] - 1 for s, _ in spans[1:] if s in start_of]
    strong = {i for i in range(len(d)) if d[i] > tau}
    free_cuts = [i for i in cuts if i not in set(forced)]
    return {"cuts": len(cuts), "strong": len(strong - set(forced)), "strong_cut": len(strong & set(cuts)) - len(strong & set(forced) & set(cuts)),
            "cut_d_sum": float(sum(d[i] for i in free_cuts)), "free_cuts": len(free_cuts)}


def run_config(dataset, docs, chunker, cfg, models, embed):
    chunks_all, q = [], {"cuts": 0, "strong": 0, "strong_cut": 0, "cut_d_sum": 0.0, "free_cuts": 0}
    info_tot = {}
    for doc in docs:
        text, pieces = render(doc)
        chunks, info = splitters.chunk(doc, chunker, cfg, {**models, "embed_model": embed})
        chunks_all += chunks
        for k, v in info.items():
            if isinstance(v, int):
                info_tot[k] = info_tot.get(k, 0) + v
        cq = cut_quality(doc, text, pieces, chunker, cfg, embed, [(c["start"], c["end"]) for c in chunks])
        if cq:
            for k in q:
                q[k] += cq[k]
    stats = compute_stats(chunks_all, {d["doc_id"]: d for d in docs}, cfg["min_tokens"], cfg["max_tokens"])
    pre = chunker + "_"
    row = {"n_chunks": stats["n_chunks"], "tokens_mean": stats["tokens_mean"], "tokens_p50": stats["tokens_p50"],
           "under_min_rate": stats.get("under_min_rate"), "over_max_rate": stats.get("over_max_rate"),
           "sentence_cut_rate": stats.get("sentence_cut_rate"), "orphan_heading_rate": stats.get("orphan_heading_rate"),
           "table_split_rate": stats.get("table_split_rate"),
           "strong_kept": round(q["strong_cut"] / q["strong"], 4) if q["strong"] else None,
           "mean_cut_d": round(q["cut_d_sum"] / q["free_cuts"], 4) if q["free_cuts"] else None,
           "size_cuts": info_tot.get(pre + "size_cuts"), "min_shortfall": info_tot.get(pre + "min_shortfall"),
           "over_max_single_unit": info_tot.get(pre + "over_max_single_unit"), "orig_sections": info_tot.get(pre + "orig_sections"),
           "section_merges": info_tot.get(pre + "section_merges"), "cross_section_chunks": info_tot.get(pre + "cross_section_chunks")}
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["vectara", "allganize", "docx_corpus"])
    ap.add_argument("--min-tokens", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--percentile", type=float, required=True)
    ap.add_argument("--limit", type=int, help="데이터셋마다 앞에서 N개 문서만(시험용)")
    ap.add_argument("--tag")
    args = ap.parse_args()

    tag = args.tag or f"p{args.percentile:g}-min{args.min_tokens}-max{args.max_tokens}"
    print(f"[설정] min {args.min_tokens} / max {args.max_tokens} / percentile {args.percentile:g} / 데이터셋 {args.datasets}", flush=True)
    results, t0 = {}, time.time()
    for ds in args.datasets:
        ids = [e["doc_id"] for e in load_entries(ds)]
        if args.limit:
            ids = ids[:args.limit]
        docs = [load_doc(PROCESSED_DIR / ds / f"{i}.json") for i in ids]
        models = get_models(ds)
        embed = CachedEmbed(models["embed_model"])
        results[ds] = {}
        for chunker in ("semantic", "smart"):
            for sc in ("greedy", "dp"):
                cfg = dict(dataset_cfg(ds)["chunkers"][chunker])
                cfg.update(percentile=args.percentile, min_tokens=args.min_tokens, max_tokens=args.max_tokens, size_control=sc)
                if sc == "greedy":
                    cfg["rules"] = {**cfg["rules"], "section_split": False}          # 섹션 분할은 DP 전용
                name = f"{chunker}/{sc}"
                row = run_config(ds, docs, chunker, cfg, models, embed)
                results[ds][name] = row
                print(f"[{ds}] {name:16} 청크 {row['n_chunks']:>6} under_min {row['under_min_rate']} over_max {row['over_max_rate']} "
                      f"strong_kept {row['strong_kept']} mean_cut_d {row['mean_cut_d']} sentence_cut {row['sentence_cut_rate']} "
                      f"cross_section {row['cross_section_chunks']} ({time.time() - t0:.0f}s, 임베딩 {embed.calls})", flush=True)
        del models, embed

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"args": vars(args), "created_at": datetime.datetime.now().isoformat(timespec="seconds"), "seconds": round(time.time() - t0, 1), "results": results}
    (OUT_DIR / f"{tag}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [f"# 크기 제어 비교 ({tag})", "", f"min {args.min_tokens} / max {args.max_tokens} / percentile {args.percentile:g}, 구조 지표만(QA 미사용)", ""]
    cols = ["n_chunks", "tokens_p50", "under_min_rate", "over_max_rate", "min_shortfall", "strong_kept", "mean_cut_d", "sentence_cut_rate", "size_cuts",
            "orig_sections", "section_merges", "cross_section_chunks"]
    for ds in results:
        lines += [f"## {ds}", "", "| 설정 | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for name, r in results[ds].items():
            lines.append(f"| {name} | " + " | ".join(str(r.get(c)) for c in cols) + " |")
        lines.append("")
    (OUT_DIR / f"{tag}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[저장] {OUT_DIR / (tag + '.json')}", flush=True)


if __name__ == "__main__":
    main()
